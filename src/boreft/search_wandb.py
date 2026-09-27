"""Log Semantle/MolOpt search seeds to Weights & Biases.

Each ``seed_*`` directory becomes one W&B run. History is indexed by
**verifications** (not observation index) so multi-sample BOReFT is comparable
to one-sample baselines. Group by ``method`` and ``split`` on the dashboard.

A project is required (``--wandb-project`` or ``WANDB_PROJECT``). Missing
project means no logging. ``wandb_meta.json`` in the seed directory records the
run id so a later pass skips completed uploads unless the trajectory grew.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Trajectory-defining search fields must stay frozen on --resume; logging knobs
# are allowed to change so older checkpoints can gain W&B after the fact.
WANDB_CONFIG_KEYS = (
    "wandb_project",
    "wandb_entity",
    "wandb_run_name",
    "wandb_group",
    "wandb_dir",
    "no_wandb",
)

DEFAULT_PROJECT = "boreft"
DEFAULT_ENTITY = None
DEFAULT_SEARCH_GROUP = "semantle-search"
DEFAULT_SWEEP_GROUP = "semantle-sweep"
DEFAULT_BUDGET = 500
META_NAME = "wandb_meta.json"
JOB_TYPE = "search"
ANALYSIS_JOB_TYPE = "search-analysis"
OFFLINE_ANALYSIS_JOB_TYPE = "analysis"

_ENCODER_PLOT_KINDS = (
    "purity",
    "metrics",
    "pca",
    "qwen_pca",
    "qwen_llm_pca",
    "llama_pca",
    "mist_pca",
)
_ENCODER_SCORE_KEYS = (
    "knn_purity_at_5",
    "knn_purity_at_10",
    "kmeans_purity",
    "silhouette_cosine",
    "kmeans_nmi",
    "kmeans_ari",
    "n",
    "dim",
    "n_labels",
    "n_truncated",
)
_ENCODER_SKIP_CONFIG = {
    "scores",
    "targets",
    "labels",
    "smiles",
    "example",
    "label_counts",
}
_RANK_SKIP_CONFIG = {"vocab_words", "results", "embeddings"}
_LOCAL_ERANK_SKIP_CONFIG = {"conditions", "per_target", "words"}
_RANK_HISTORY_KEYS = (
    "effective_rank",
    "stable_rank",
    "participation_ratio",
    "sigma_max",
)
_NAMED_ANALYSIS_PREFIXES = (
    "embedding_models",
    "text_variants",
    "rdkit_definition_embeds",
    "code_geometry",
    "oracle_screen",
    "catalog_decode",
    "recon_audit",
    "code_inversion",
)
_NAMED_SKIP_CONFIG = {
    "results",
    "scores",
    "variants",
    "variant_questions",
    "label_counts",
    "unavailable",
    "models",
    "conditions",
    "pairs",
    "texts",
    "targets",
    "labels",
    "smiles",
    "iupac",
    "examples",
    "example",
    "text_templates",
    "drowning",
}


def wandb_enabled(*, project: str | None, no_wandb: bool) -> bool:
    if no_wandb:
        return False
    return bool(project or os.environ.get("WANDB_PROJECT"))


def add_wandb_cli(parser: argparse.ArgumentParser) -> None:
    """Default-on W&B flags shared by search and offline analysis CLIs."""
    parser.add_argument("--wandb-project", default=DEFAULT_PROJECT)
    parser.add_argument("--wandb-entity", default=DEFAULT_ENTITY)
    parser.add_argument("--wandb-dir", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Skip Weights & Biases logging.",
    )


def resolve_project(project: str | None) -> str | None:
    value = project or os.environ.get("WANDB_PROJECT")
    return str(value).strip() or None


def resolve_entity(entity: str | None) -> str | None:
    value = entity or os.environ.get("WANDB_ENTITY")
    return str(value).strip() or None


def _norm(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def observation_texts(obs: Mapping[str, Any]) -> list[str]:
    texts: list[str] = []
    for key in ("decoded_samples", "solution_samples"):
        for item in obs.get(key) or []:
            if item:
                texts.append(str(item))
    for key in ("decoded", "solution"):
        item = obs.get(key)
        if item:
            texts.append(str(item))
    return texts


def sample_scores(obs: Mapping[str, Any]) -> list[float]:
    scores = obs.get("sample_scores")
    if scores:
        return [float(s) for s in scores]
    count = max(int(obs.get("sample_count") or 1), 1)
    return [float(obs.get("score") or 0.0)] * count


def is_warmstart(obs: Mapping[str, Any]) -> bool:
    kind = obs.get("source") or obs.get("phase") or ""
    return kind == "warmstart"


def observation_matches(obs: Mapping[str, Any], target_key: str) -> int | None:
    samples = list(obs.get("decoded_samples") or obs.get("solution_samples") or [])
    if samples:
        for i, text in enumerate(samples):
            if _norm(text) == target_key:
                return i + 1
    for key in ("decoded", "solution"):
        if _norm(obs.get(key) or "") == target_key:
            return 1
    exact = (obs.get("components") or {}).get("exact_match")
    if exact is True:
        return 1
    if isinstance(exact, (int, float)) and exact > 0:
        return 1
    return None


def verification_history(
    observations: Sequence[Mapping[str, Any]],
    target: str,
    *,
    budget: int | None = None,
) -> list[dict[str, Any]]:
    """One row per verifier call, in search order."""
    target_key = _norm(target)
    rows: list[dict[str, Any]] = []
    best = float("-inf")
    found = False
    seen: set[str] = set()
    cursor = 0
    cap = budget if budget is not None and budget > 0 else None

    for obs in observations:
        scores = sample_scores(obs)
        texts = observation_texts(obs)
        match_at = observation_matches(obs, target_key)
        warm = is_warmstart(obs)
        for i, score in enumerate(scores):
            cursor += 1
            if cap is not None and cursor > cap:
                return rows
            best = max(best, score)
            text = texts[i] if i < len(texts) else (texts[0] if texts else "")
            key = _norm(text)
            if key:
                seen.add(key)
            if match_at is not None and i + 1 >= match_at:
                found = True
            rows.append(
                {
                    "verifications": cursor,
                    "score": float(score),
                    "best_so_far": float(best) if best != float("-inf") else 0.0,
                    "found": int(found),
                    "n_unique": len(seen),
                    "is_warmstart": int(warm),
                }
            )
    return rows


def _seed_index(seed_name: str) -> int:
    if not seed_name.startswith("seed_"):
        return 0
    suffix = seed_name.split("_", 1)[1]
    return int(suffix) if suffix.isdigit() else 0


def identity_from_seed_dir(seed_dir: str | Path) -> dict[str, Any]:
    """Infer method / split / target / protocol from the on-disk layout.

    Expected:
      ``.../search/<method>/<split>-<target>/seed_<n>``
      ``.../sweep/<slug>/<split>-<target>/seed_<n>``

    ``target`` from the folder name is a fallback only. Experiment dirs sanitize
    punctuation (``self-esteem`` → ``train-self_esteem``), so callers should
    prefer ``config.json`` / ``--target`` when matching gold strings.
    """
    path = Path(seed_dir).expanduser().resolve()
    seed_name = path.name
    seed = _seed_index(seed_name)
    run_dir = path.parent
    run_name = run_dir.name
    split, target = (run_name.split("-", 1) + [""])[:2] if "-" in run_name else ("", run_name)
    parent = run_dir.parent
    method_or_slug = parent.name
    protocol_dir = parent.parent.name if parent.parent else ""
    protocol = protocol_dir if protocol_dir in ("search", "sweep") else ""
    if protocol == "sweep":
        method = "boreft"
        sweep_slug = method_or_slug
    else:
        method = method_or_slug
        sweep_slug = ""
    group = DEFAULT_SWEEP_GROUP if protocol == "sweep" else DEFAULT_SEARCH_GROUP
    display = sweep_slug or method
    name = f"{display}_{split}-{target}_seed{seed}" if split else f"{display}_seed{seed}"
    return {
        "seed_dir": str(path),
        "run_dir": str(run_dir),
        "method": method,
        "sweep_slug": sweep_slug,
        "protocol": protocol or "search",
        "split": split,
        "target": target,
        "seed": seed,
        "group": group,
        "name": name,
    }


def _meta_path(seed_dir: Path) -> Path:
    return seed_dir / META_NAME


def read_meta(seed_dir: Path) -> dict[str, Any]:
    return _load_json(_meta_path(seed_dir))


def write_meta(seed_dir: Path, payload: Mapping[str, Any]) -> None:
    seed_dir.mkdir(parents=True, exist_ok=True)
    _meta_path(seed_dir).write_text(
        json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8"
    )


def _history_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "search/verifications": int(row["verifications"]),
        "search/score": float(row["score"]),
        "search/best_so_far": float(row["best_so_far"]),
        "search/found": int(row["found"]),
        "search/n_unique": int(row["n_unique"]),
        "search/is_warmstart": int(row["is_warmstart"]),
    }


def _log_without_global_step(run: Any, payload: Mapping[str, Any]) -> None:
    """Log onto a shared run without wandb's monotonic ``step``.

    Search and expansion use ``define_metric(..., step_metric=...)``. Passing
    ``step=`` binds both to the hidden ``_step`` counter, so post-expansion
    search points (verification 110 after hundreds of trainer logs) are dropped.
    """
    log_fn = getattr(run, "log", None)
    if not callable(log_fn):
        raise TypeError("wandb run has no log method")
    log_fn(dict(payload))


def _summary_payload(
    history: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    last = history[-1] if history else {}
    found_at = next(
        (int(row["verifications"]) for row in history if int(row.get("found") or 0)),
        None,
    )
    out = {
        "search/found_target": bool(last.get("found")) if last else False,
        "search/found_at": found_at,
        "search/best_score": float(last["best_so_far"]) if last else None,
        "search/n_verifications": int(last["verifications"]) if last else 0,
        "search/n_unique": int(last["n_unique"]) if last else 0,
    }
    if summary:
        for key in (
            "found_target",
            "found_at_verifications",
            "found_at_index",
            "n_repeat_samples",
            "n_repeat_proposals",
            "elapsed_seconds",
        ):
            if key in summary and summary[key] is not None:
                out[f"search/{key}"] = summary[key]
    return out


def _run_config(
    identity: Mapping[str, Any],
    extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    cfg = {
        "method": identity.get("method"),
        "sweep_slug": identity.get("sweep_slug") or None,
        "protocol": identity.get("protocol"),
        "split": identity.get("split"),
        "target": identity.get("target"),
        "seed": identity.get("seed"),
        "search_dir": identity.get("run_dir"),
        "seed_dir": identity.get("seed_dir"),
    }
    if extra:
        for key, value in extra.items():
            if key in WANDB_CONFIG_KEYS:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                cfg[key] = value
            elif isinstance(value, dict):
                for inner_key, inner_value in value.items():
                    if isinstance(inner_value, (str, int, float, bool)) or inner_value is None:
                        cfg[f"{key}.{inner_key}"] = inner_value
    return cfg


_EXPAND_EVAL_SCALAR_KEYS = (
    ("avg_embed_sim", "avg_embed_sim"),
    ("min_embed_sim", "min_embed_sim"),
    ("embed_sim_gte_tau", "embed_sim_gte_tau"),
    ("embed_sim_tau", "embed_sim_tau"),
    ("avg_selection_sim", "avg_selection_sim"),
    ("min_selection_sim", "min_selection_sim"),
    ("n_recovered", "n_recovered"),
    ("n_targets", "n_targets"),
)
_EXPAND_TRAIN_KEYS = (
    "loss",
    "ce_loss",
    "margin_loss",
    "aux_loss",
    "sdpo_loss",
    "sdpo_n_seq",
    "sdpo_n_tokens",
    "kl_beta_effective",
    "lambda_ce_effective",
    "lambda_sdpo_effective",
    "mu_norm",
    "bias_var",
    "learning_rate",
    "grad_norm",
)


def define_search_wandb_metrics(run: Any) -> None:
    define_metric = getattr(run, "define_metric", None)
    if not callable(define_metric):
        return
    define_metric("search/verifications")
    define_metric("search/*", step_metric="search/verifications")


def define_expand_wandb_metrics(run: Any, round_id: int, phase: str = "joint") -> None:
    define_metric = getattr(run, "define_metric", None)
    if not callable(define_metric):
        return
    prefix = expand_metric_prefix(round_id, phase)
    step_name = f"{prefix}/train/global_step"
    define_metric(step_name)
    define_metric(f"{prefix}/*", step_metric=step_name)


def expand_metric_prefix(round_id: int, phase: str) -> str:
    """Joint phase keeps ``expand/r{{n}}``; new-only nests under ``/new_only``."""
    if str(phase) == "new_only":
        return f"expand/r{round_id}/new_only"
    return f"expand/r{round_id}"


def expand_eval_scalars(metrics: Mapping[str, Any], *, prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for source, leaf in _EXPAND_EVAL_SCALAR_KEYS:
        value = metrics.get(source)
        if isinstance(value, (int, float)) and value == value:
            out[f"{prefix}/{leaf}"] = float(value)
    n_targets = int(metrics.get("n_targets") or 0)
    recovered = out.get(f"{prefix}/n_recovered")
    if n_targets > 0 and isinstance(recovered, (int, float)):
        out[f"{prefix}/recover_rate"] = float(recovered) / n_targets
    return out


def expansion_history_logs(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten one expansion round into wandb.log payloads (no Tables)."""
    round_id = int(record.get("round") or 0)
    rows: list[dict[str, Any]] = []
    for phase in record.get("phases") or []:
        if not isinstance(phase, Mapping):
            continue
        phase_name = str(phase.get("phase") or "joint")
        prefix = expand_metric_prefix(round_id, phase_name)
        for entry in phase.get("eval_history") or []:
            if not isinstance(entry, Mapping):
                continue
            payload = {
                f"{prefix}/train/global_step": float(entry.get("step") or 0),
                f"{prefix}/eval/epoch": float(entry.get("epoch") or 0),
                f"{prefix}/phase": phase_name,
            }
            payload.update(expand_eval_scalars(entry, prefix=f"{prefix}/eval"))
            groups = entry.get("groups")
            if isinstance(groups, Mapping):
                for group_name, group_metrics in groups.items():
                    if isinstance(group_metrics, Mapping):
                        payload.update(
                            expand_eval_scalars(
                                group_metrics,
                                prefix=f"{prefix}/eval/{group_name}",
                            )
                        )
            rows.append(payload)
    return rows


def make_expand_progress_callback(run: Any, round_id: int, phase: str = "joint"):
    """Stream one expansion round's train/eval panels onto the search W&B run."""
    define_expand_wandb_metrics(run, round_id, phase=phase)
    prefix = expand_metric_prefix(round_id, phase)

    def _callback(payload: Mapping[str, Any]) -> None:
        event = payload.get("event")
        if event == "eval_result":
            metrics = {
                f"{prefix}/eval/epoch": float(payload.get("epoch") or 0.0),
                f"{prefix}/train/global_step": float(payload.get("step") or 0),
            }
            metrics.update(expand_eval_scalars(payload, prefix=f"{prefix}/eval"))
            groups = payload.get("groups")
            if isinstance(groups, Mapping):
                for group_name, group_metrics in groups.items():
                    if isinstance(group_metrics, Mapping):
                        metrics.update(
                            expand_eval_scalars(
                                group_metrics,
                                prefix=f"{prefix}/eval/{group_name}",
                            )
                        )
            _log_without_global_step(run, metrics)
            return
        if event != "log":
            return
        train_metrics: dict[str, float] = {}
        for key in _EXPAND_TRAIN_KEYS:
            value = payload.get(key)
            if isinstance(value, (int, float)) and value == value:
                train_metrics[f"{prefix}/train/{key}"] = float(value)
        if train_metrics:
            train_metrics[f"{prefix}/train/global_step"] = float(
                payload.get("step") or 0
            )
            epoch = payload.get("epoch")
            if isinstance(epoch, (int, float)) and epoch == epoch:
                train_metrics[f"{prefix}/train/epoch"] = float(epoch)
            _log_without_global_step(run, train_metrics)

    return _callback


class SearchWandbSession:
    """Live W&B run for one search seed (trajectory + expansion panels)."""

    def __init__(self, run: Any, *, target: str, budget: int, seed_dir: Path):
        self.run = run
        self.target = target
        self.budget = budget
        self.seed_dir = Path(seed_dir)
        self.logged_verifications = 0
        self._expand_rounds: set[int] = set()

    def log_observations(self, observation: Any = None, state: Any = None) -> None:
        if state is None:
            state = observation
        observations = [
            item.to_dict() if hasattr(item, "to_dict") else item
            for item in getattr(state, "observations", ())
        ]
        history = verification_history(
            observations, self.target, budget=self.budget
        )
        for row in history:
            step = int(row["verifications"])
            if step <= self.logged_verifications:
                continue
            _log_without_global_step(self.run, _history_payload(row))
            self.logged_verifications = step

    def expansion_callback(self, round_id: int, phase: str = "joint"):
        self._expand_rounds.add(int(round_id))
        return make_expand_progress_callback(self.run, round_id, phase=phase)

    def finish(self, summary: Mapping[str, Any] | None = None) -> None:
        from boreft.search_expand import load_expansion_records

        history = verification_history(
            load_jsonl(self.seed_dir / "observations.jsonl"),
            self.target,
            budget=self.budget,
        )
        for row in history:
            step = int(row["verifications"])
            if step <= self.logged_verifications:
                continue
            _log_without_global_step(self.run, _history_payload(row))
            self.logged_verifications = step
        for key, value in _summary_payload(history, summary).items():
            if value is not None:
                self.run.summary[key] = value
        n_expand = len(load_expansion_records(self.seed_dir))
        if n_expand:
            self.run.summary["expand/n_rounds"] = n_expand
        try:
            import wandb

            wandb.finish()
        except Exception:
            _safe_wandb_finish()
        meta = read_meta(self.seed_dir)
        write_meta(
            self.seed_dir,
            {
                **meta,
                "n_verifications": int(
                    history[-1]["verifications"] if history else 0
                ),
                "n_expansion_rounds": n_expand,
                "live": True,
            },
        )


def start_search_wandb_session(
    seed_dir: str | Path,
    *,
    project: str | None,
    entity: str | None = None,
    group: str | None = None,
    name: str | None = None,
    wandb_dir: str | None = None,
    extra_config: Mapping[str, Any] | None = None,
    no_wandb: bool = False,
    budget: int | None = None,
    target: str | None = None,
) -> SearchWandbSession | None:
    """Open a live search run. Failures return ``None`` and never abort search."""
    if not wandb_enabled(project=project, no_wandb=no_wandb):
        return None
    path = Path(seed_dir).expanduser().resolve()
    identity = identity_from_seed_dir(path)
    run_cfg = _load_json(Path(identity["run_dir"]) / "config.json")
    resolved_target = target or _resolved_target(identity, run_cfg, extra_config)
    if resolved_target:
        identity = {**identity, "target": resolved_target}
    try:
        hist_budget = budget or int(run_cfg.get("budget") or DEFAULT_BUDGET)
    except (TypeError, ValueError):
        hist_budget = DEFAULT_BUDGET
    resolved_project = resolve_project(project)
    if not resolved_project:
        return None
    init_kwargs: dict[str, Any] = {
        "project": resolved_project,
        "name": name or identity["name"],
        "group": group or identity["group"],
        "job_type": JOB_TYPE,
        "config": _run_config(identity, {**run_cfg, **(extra_config or {})}),
        "tags": [
            str(identity.get("method") or "search"),
            str(identity.get("split") or "run"),
            str(identity.get("protocol") or "search"),
        ],
        "reinit": True,
    }
    resolved_entity = resolve_entity(entity)
    if resolved_entity:
        init_kwargs["entity"] = resolved_entity
    if wandb_dir:
        os.makedirs(wandb_dir, exist_ok=True)
        init_kwargs["dir"] = wandb_dir
    meta = read_meta(path)
    if meta.get("run_id"):
        init_kwargs["id"] = meta["run_id"]
        init_kwargs["resume"] = "allow"
    try:
        import wandb

        run = wandb.init(**init_kwargs)
        if run is None:
            return None
        define_search_wandb_metrics(run)
        write_meta(
            path,
            {
                **meta,
                "run_id": run.id,
                "project": resolved_project,
                "entity": resolved_entity,
                "group": init_kwargs["group"],
                "live": True,
            },
        )
        print(f"[wandb] live run {init_kwargs['name']} → {run.id}", flush=True)
        logged = int(meta.get("n_verifications") or 0)
        session = SearchWandbSession(
            run,
            target=str(resolved_target or ""),
            budget=hist_budget,
            seed_dir=path,
        )
        session.logged_verifications = logged
        return session
    except Exception as exc:
        _safe_wandb_finish()
        print(f"[wandb] failed to start live run for {seed_dir}: {exc}", flush=True)
        return None


def iter_seed_dirs(root: str | Path) -> list[Path]:
    """Yield ``seed_*`` directories that contain ``observations.jsonl``."""
    base = Path(root)
    if not base.is_dir():
        return []
    found: list[Path] = []
    for obs in sorted(base.glob("**/seed_*/observations.jsonl")):
        found.append(obs.parent)
    return found


def _safe_wandb_finish() -> None:
    try:
        import wandb
    except Exception:
        return
    try:
        if getattr(wandb, "run", None) is not None:
            wandb.finish()
    except Exception:
        pass


def _resolved_target(
    identity: Mapping[str, Any],
    run_cfg: Mapping[str, Any],
    extra_config: Mapping[str, Any] | None,
) -> str:
    extra_target = (extra_config or {}).get("target")
    return str(extra_target or run_cfg.get("target") or identity.get("target") or "")


def log_seed_directory(
    seed_dir: str | Path,
    *,
    project: str | None = None,
    entity: str | None = None,
    group: str | None = None,
    name: str | None = None,
    wandb_dir: str | None = None,
    extra_config: Mapping[str, Any] | None = None,
    budget: int | None = None,
    history_stride: int = 1,
    log_images: bool = False,
    force: bool = False,
    overwrite_history: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Upload one seed directory. Returns a status dict (``skipped`` / ``logged``).

    ``overwrite_history`` resumes the recorded ``run_id`` and re-logs the
    full verification curve from step 0 (W&B history is still append-only).
    """
    path = Path(seed_dir).expanduser().resolve()
    identity = identity_from_seed_dir(path)
    try:
        observations = load_jsonl(path / "observations.jsonl")
        run_cfg = _load_json(Path(identity["run_dir"]) / "config.json")
        summary = _load_json(path / "summary.json")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {"status": "error", "error": str(exc), **identity}
    if not observations:
        return {"status": "empty", **identity}

    target = _resolved_target(identity, run_cfg, extra_config)
    if target:
        identity = {**identity, "target": target}
        display = identity.get("sweep_slug") or identity.get("method")
        split = identity.get("split")
        seed = identity.get("seed")
        identity["name"] = (
            f"{display}_{split}-{target}_seed{seed}"
            if split
            else f"{display}_seed{seed}"
        )
    try:
        hist_budget = budget or int(run_cfg.get("budget") or DEFAULT_BUDGET)
    except (TypeError, ValueError) as exc:
        return {"status": "error", "error": str(exc), **identity}
    history = verification_history(observations, target, budget=hist_budget)
    n_verifications = int(history[-1]["verifications"]) if history else 0
    from boreft.search_expand import load_expansion_records

    expand_records = load_expansion_records(path)
    n_expand = len(expand_records)
    meta = read_meta(path)
    if (
        not force
        and not overwrite_history
        and meta.get("run_id")
        and int(meta.get("n_verifications") or 0) >= n_verifications
        and int(meta.get("n_expansion_rounds") or 0) >= n_expand
    ):
        return {"status": "skipped", "run_id": meta.get("run_id"), **identity}

    merged_config = _run_config(identity, {**run_cfg, **(extra_config or {})})
    start_from = 0
    if not force and not overwrite_history and meta.get("run_id"):
        start_from = int(meta.get("n_verifications") or 0)
    stride = max(int(history_stride), 1)
    rows = [
        row
        for row in history
        if int(row["verifications"]) > start_from
        and (
            int(row["verifications"]) == n_verifications
            or int(row["verifications"]) % stride == 0
            or int(row["verifications"]) == 1
        )
    ]
    resolved_project = resolve_project(project)
    if not resolved_project:
        return {"status": "no_project", **identity}
    if dry_run:
        return {
            "status": "dry_run",
            "n_history": len(rows),
            "n_verifications": n_verifications,
            **identity,
        }

    init_kwargs: dict[str, Any] = {
        "project": resolved_project,
        "name": name or identity["name"],
        "group": group or identity["group"],
        "job_type": JOB_TYPE,
        "config": merged_config,
        "tags": [
            str(identity.get("method") or "search"),
            str(identity.get("split") or "run"),
            str(identity.get("protocol") or "search"),
        ],
        "reinit": True,
    }
    resolved_entity = resolve_entity(entity)
    if resolved_entity:
        init_kwargs["entity"] = resolved_entity
    if wandb_dir:
        os.makedirs(wandb_dir, exist_ok=True)
        init_kwargs["dir"] = wandb_dir
    if meta.get("run_id") and (overwrite_history or not force):
        init_kwargs["id"] = meta["run_id"]
        init_kwargs["resume"] = "allow"

    try:
        import wandb

        run = wandb.init(**init_kwargs)
        if run is None:
            return {"status": "error", "error": "wandb.init returned None", **identity}
        run_id = run.id
        define_search_wandb_metrics(run)
        for row in rows:
            _log_without_global_step(run, _history_payload(row))
        logged_expand = (
            0
            if force or overwrite_history
            else int(meta.get("n_expansion_rounds") or 0)
        )
        for record in expand_records:
            round_id = int(record.get("round") or 0)
            if round_id <= logged_expand and not force and not overwrite_history:
                continue
            phases = {
                str(phase.get("phase") or "joint")
                for phase in (record.get("phases") or [])
                if isinstance(phase, Mapping)
            } or {"joint"}
            for phase_name in sorted(phases):
                define_expand_wandb_metrics(run, round_id, phase=phase_name)
            for payload in expansion_history_logs(record):
                _log_without_global_step(run, payload)
        if n_expand:
            run.summary["expand/n_rounds"] = n_expand
        for key, value in _summary_payload(history, summary).items():
            if value is not None:
                run.summary[key] = value
        plot_path = path / "best_so_far.png"
        if log_images and plot_path.is_file():
            wandb.log({"search/best_so_far_plot": wandb.Image(str(plot_path))})
        wandb.finish()
    except Exception as exc:
        _safe_wandb_finish()
        return {"status": "error", "error": str(exc), **identity}

    write_meta(
        path,
        {
            "run_id": run_id,
            "project": resolved_project,
            "entity": resolved_entity,
            "group": init_kwargs["group"],
            "n_verifications": n_verifications,
            "n_observations": len(observations),
            "n_expansion_rounds": n_expand,
        },
    )
    return {
        "status": "logged",
        "run_id": run_id,
        "n_history": len(rows),
        "n_verifications": n_verifications,
        **identity,
    }


def patch_seed_wandb_summary(
    seed_dir: str | Path,
    *,
    project: str | None = None,
    entity: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Overwrite summary fields on the existing W&B run; do not append history."""
    path = Path(seed_dir).expanduser().resolve()
    identity = identity_from_seed_dir(path)
    meta = read_meta(path)
    run_id = meta.get("run_id")
    if not run_id:
        return {"status": "no_run", **identity}
    try:
        observations = load_jsonl(path / "observations.jsonl")
        run_cfg = _load_json(Path(identity["run_dir"]) / "config.json")
        summary = _load_json(path / "summary.json")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {"status": "error", "error": str(exc), **identity}
    if not observations:
        return {"status": "empty", "run_id": run_id, **identity}
    target = _resolved_target(identity, run_cfg, None)
    try:
        hist_budget = int(run_cfg.get("budget") or DEFAULT_BUDGET)
    except (TypeError, ValueError) as exc:
        return {"status": "error", "error": str(exc), "run_id": run_id, **identity}
    history = verification_history(observations, target, budget=hist_budget)
    payload = {
        key: value
        for key, value in _summary_payload(history, summary).items()
        if value is not None
    }
    resolved_project = resolve_project(project) or meta.get("project")
    resolved_entity = resolve_entity(entity) or meta.get("entity")
    if dry_run:
        return {
            "status": "dry_run",
            "run_id": run_id,
            "n_summary": len(payload),
            **identity,
        }
    if not resolved_project:
        return {"status": "no_project", "run_id": run_id, **identity}
    try:
        import wandb

        path_parts = [part for part in (resolved_entity, resolved_project, run_id) if part]
        api_run = wandb.Api().run("/".join(str(part) for part in path_parts))
        for key, value in payload.items():
            api_run.summary[key] = value
        api_run.update()
    except Exception as exc:
        return {"status": "error", "error": str(exc), "run_id": run_id, **identity}
    return {"status": "patched", "run_id": run_id, **identity}


def log_search_tree(
    roots: Iterable[str | Path],
    *,
    project: str | None = None,
    entity: str | None = None,
    wandb_dir: str | None = None,
    history_stride: int = 1,
    log_images: bool = False,
    force: bool = False,
    overwrite_history: bool = False,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for root in roots:
        for seed_dir in iter_seed_dirs(root):
            try:
                results.append(
                    log_seed_directory(
                        seed_dir,
                        project=project,
                        entity=entity,
                        wandb_dir=wandb_dir,
                        history_stride=history_stride,
                        log_images=log_images,
                        force=force,
                        overwrite_history=overwrite_history,
                        dry_run=dry_run,
                    )
                )
            except Exception as exc:
                results.append(
                    {
                        "status": "error",
                        "error": str(exc),
                        "seed_dir": str(seed_dir),
                    }
                )
    return results


def maybe_log_finished_seed(
    seed_dir: str | Path,
    *,
    project: str | None,
    entity: str | None = None,
    group: str | None = None,
    name: str | None = None,
    wandb_dir: str | None = None,
    extra_config: Mapping[str, Any] | None = None,
    no_wandb: bool = False,
) -> dict[str, Any] | None:
    """Called by search runners after writing ``summary.json``.

    W&B failures must not fail a finished search job.
    """
    if not wandb_enabled(project=project, no_wandb=no_wandb):
        return None
    try:
        result = log_seed_directory(
            seed_dir,
            project=project,
            entity=entity,
            group=group,
            name=name,
            wandb_dir=wandb_dir,
            extra_config=extra_config,
        )
    except Exception as exc:
        print(f"[wandb] failed to log {seed_dir}: {exc}", flush=True)
        return {"status": "error", "error": str(exc)}
    status = result.get("status")
    if status == "logged":
        print(
            f"[wandb] logged {result.get('name')} → {result.get('run_id')}",
            flush=True,
        )
    elif status == "skipped":
        print(
            f"[wandb] already logged {result.get('name')} ({result.get('run_id')})",
            flush=True,
        )
    elif status == "no_project":
        print("[wandb] skip: no --wandb-project / WANDB_PROJECT", flush=True)
    elif status == "error":
        print(
            f"[wandb] failed to log {result.get('name') or seed_dir}: "
            f"{result.get('error')}",
            flush=True,
        )
    return result


def _normalize_tag(tag: Any) -> str:
    text = str(tag or "")
    if text and not text.startswith("_"):
        text = "_" + text
    return text


def _analysis_task(directory: Path, payload: Mapping[str, Any] | None = None) -> str:
    task = str((payload or {}).get("task") or "").strip().lower()
    if task in ("semantle", "molopt", "hypogen"):
        return task
    parts = {part.lower() for part in directory.parts}
    for name in ("semantle", "molopt", "hypogen"):
        if name in parts:
            return name
    return "semantle"


def _json_safe_config(
    payload: Mapping[str, Any], *, skip: set[str]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if key in skip:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, (list, tuple)) and len(value) <= 64:
            if all(
                isinstance(item, (str, int, float, bool)) or item is None
                for item in value
            ):
                out[key] = list(value)
    return out


def _source_mtime(paths: Sequence[Path]) -> float:
    mtimes = [path.stat().st_mtime for path in paths if path.is_file()]
    return max(mtimes) if mtimes else 0.0


def read_analysis_meta(directory: Path) -> dict[str, Any]:
    payload = _load_json(directory / META_NAME)
    return payload if isinstance(payload, dict) else {}


def write_analysis_meta(directory: Path, key: str, payload: Mapping[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    meta = read_analysis_meta(directory)
    meta[key] = dict(payload)
    (directory / META_NAME).write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )


def _print_analysis_status(result: Mapping[str, Any]) -> None:
    status = result.get("status")
    name = result.get("name")
    if status == "logged":
        print(
            f"[wandb] logged {name} → {result.get('run_id')}",
            flush=True,
        )
    elif status == "skipped":
        print(
            f"[wandb] already logged {name} ({result.get('run_id')})",
            flush=True,
        )
    elif status == "no_project":
        print("[wandb] skip: no --wandb-project / WANDB_PROJECT", flush=True)
    elif status == "empty":
        print(f"[wandb] skip empty analysis {name or ''}", flush=True)
    elif status == "error":
        print(
            f"[wandb] failed to log {name or 'analysis'}: {result.get('error')}",
            flush=True,
        )


def _pdf_siblings(paths: Sequence[Path]) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        pdf = Path(path).with_suffix(".pdf")
        if not pdf.is_file():
            continue
        resolved = pdf.resolve()
        if resolved not in seen:
            found.append(pdf)
            seen.add(resolved)
    return found


def _raster_images(paths: Sequence[Path]) -> list[Path]:
    raster = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    return [path for path in paths if path.suffix.lower() in raster]


def log_offline_analysis(
    *,
    name: str,
    group: str,
    job_type: str,
    tags: Sequence[str],
    images: Sequence[Path],
    json_paths: Sequence[Path],
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    history: Sequence[tuple[int, Mapping[str, Any]]] = (),
    project: str | None = None,
    entity: str | None = None,
    wandb_dir: str | None = None,
    meta_dir: str | Path | None = None,
    meta_key: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Upload plots + JSON from an offline analysis directory as one W&B run."""
    image_paths = _raster_images(
        [Path(path) for path in images if Path(path).is_file()]
    )
    json_files = [Path(path) for path in json_paths if Path(path).is_file()]
    file_paths = list(json_files)
    seen = {path.resolve() for path in file_paths}
    for extra in _pdf_siblings(image_paths + json_files):
        if extra.resolve() not in seen:
            file_paths.append(extra)
            seen.add(extra.resolve())
    if not image_paths and not file_paths:
        return {"status": "empty", "name": name}
    resolved_project = resolve_project(project)
    if not resolved_project:
        return {"status": "no_project", "name": name}

    meta_path_dir = Path(meta_dir).expanduser().resolve() if meta_dir else None
    existing: dict[str, Any] = {}
    if meta_path_dir is not None and meta_key:
        existing = read_analysis_meta(meta_path_dir).get(meta_key) or {}
        stamp = _source_mtime(image_paths + file_paths)
        if (
            not force
            and existing.get("run_id")
            and float(existing.get("source_mtime") or 0) >= stamp
            and int(existing.get("n_images") or 0) >= len(image_paths)
        ):
            return {
                "status": "skipped",
                "run_id": existing.get("run_id"),
                "name": name,
                "n_images": len(image_paths),
            }

    if dry_run:
        return {
            "status": "dry_run",
            "name": name,
            "n_images": len(image_paths),
            "n_json": len(json_files),
            "n_files": len(file_paths),
            "n_history": len(history),
        }

    init_kwargs: dict[str, Any] = {
        "project": resolved_project,
        "name": name,
        "group": group,
        "job_type": job_type,
        "config": dict(config),
        "tags": [str(tag) for tag in tags if tag],
        "reinit": True,
    }
    resolved_entity = resolve_entity(entity)
    if resolved_entity:
        init_kwargs["entity"] = resolved_entity
    if wandb_dir:
        os.makedirs(wandb_dir, exist_ok=True)
        init_kwargs["dir"] = wandb_dir
    if existing.get("run_id"):
        init_kwargs["id"] = existing["run_id"]
        init_kwargs["resume"] = "allow"

    try:
        import wandb

        run = wandb.init(**init_kwargs)
        if run is None:
            return {"status": "error", "error": "wandb.init returned None", "name": name}
        run_id = run.id
        for step, payload in history:
            wandb.log(dict(payload), step=int(step))
        logged = {
            f"plots/{path.stem}": wandb.Image(str(path)) for path in image_paths
        }
        if logged:
            wandb.log(logged)
        for key, value in summary.items():
            if value is not None:
                run.summary[key] = value
        for extra in file_paths:
            wandb.save(
                str(extra),
                base_path=str(extra.parent),
                policy="now",
            )
        wandb.finish()
    except Exception as exc:
        _safe_wandb_finish()
        return {"status": "error", "error": str(exc), "name": name}

    if meta_path_dir is not None and meta_key:
        write_analysis_meta(
            meta_path_dir,
            meta_key,
            {
                "run_id": run_id,
                "project": resolved_project,
                "entity": resolved_entity,
                "group": group,
                "n_images": len(image_paths),
                "source_mtime": _source_mtime(image_paths + file_paths),
            },
        )
    return {
        "status": "logged",
        "run_id": run_id,
        "name": name,
        "n_images": len(image_paths),
    }


def encoder_cluster_images(directory: Path, tag: str) -> list[Path]:
    suffix = _normalize_tag(tag)
    found: list[Path] = []
    for kind in _ENCODER_PLOT_KINDS:
        path = directory / f"encoder_cluster_{kind}{suffix}.png"
        if path.is_file():
            found.append(path)
    return found


def is_encoder_cluster_report(path: Path) -> bool:
    if path.suffix != ".json" or not path.name.startswith("encoder_cluster"):
        return False
    return "_cache" not in path.name


def encoder_cluster_tag_from_path(path: Path) -> str:
    stem = path.stem
    if stem == "encoder_cluster":
        return ""
    if stem.startswith("encoder_cluster"):
        return stem[len("encoder_cluster") :]
    return ""


def _encoder_cluster_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for encoder, scores in (payload.get("scores") or {}).items():
        if not isinstance(scores, Mapping):
            continue
        for key in _ENCODER_SCORE_KEYS:
            value = scores.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[f"{encoder}/{key}"] = value
    return out


def log_encoder_cluster_report(
    out_dir: str | Path,
    report: Mapping[str, Any],
    *,
    project: str | None = None,
    entity: str | None = None,
    group: str | None = None,
    name: str | None = None,
    wandb_dir: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    directory = Path(out_dir).expanduser().resolve()
    tag = _normalize_tag(report.get("artifact_tag"))
    json_path = directory / f"encoder_cluster{tag}.json"
    payload = dict(report)
    if json_path.is_file():
        payload = {**_load_json(json_path), **payload}
    tag = _normalize_tag(payload.get("artifact_tag") or tag)
    json_path = directory / f"encoder_cluster{tag}.json"
    task = _analysis_task(directory, payload)
    config = _json_safe_config(payload, skip=_ENCODER_SKIP_CONFIG)
    config["analysis_dir"] = str(directory)
    images = encoder_cluster_images(directory, tag)
    json_paths = [json_path] if json_path.is_file() else []
    return log_offline_analysis(
        name=name or f"{task}-encoder-cluster{tag}",
        group=group or f"{task}-analysis",
        job_type=OFFLINE_ANALYSIS_JOB_TYPE,
        tags=[task, "encoder-cluster", "analysis"],
        images=images,
        json_paths=json_paths,
        config=config,
        summary=_encoder_cluster_summary(payload),
        project=project,
        entity=entity,
        wandb_dir=wandb_dir,
        meta_dir=directory,
        meta_key=f"encoder_cluster{tag}",
        force=force,
        dry_run=dry_run,
    )


def maybe_log_encoder_cluster(
    out_dir: str | Path,
    report: Mapping[str, Any],
    *,
    project: str | None,
    entity: str | None = None,
    group: str | None = None,
    name: str | None = None,
    wandb_dir: str | None = None,
    no_wandb: bool = False,
) -> dict[str, Any] | None:
    if not wandb_enabled(project=project, no_wandb=no_wandb):
        return None
    try:
        result = log_encoder_cluster_report(
            out_dir,
            report,
            project=project,
            entity=entity,
            group=group,
            name=name,
            wandb_dir=wandb_dir,
        )
    except Exception as exc:
        print(f"[wandb] failed to log encoder cluster: {exc}", flush=True)
        return {"status": "error", "error": str(exc)}
    _print_analysis_status(result)
    return result


def train_embed_rank_jsons(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.glob("*train_embed_effective_rank*.json")
        if "spectrum" not in path.name
    )


def _rank_history(
    payloads: Sequence[Mapping[str, Any]],
) -> list[tuple[int, dict[str, Any]]]:
    by_n: dict[int, dict[str, Any]] = {}
    for payload in payloads:
        mode = str(payload.get("mode") or "prompt")
        for row in payload.get("results") or []:
            if not isinstance(row, Mapping):
                continue
            n = int(row.get("train_n_samples") or row.get("n_drawn") or 0)
            slot = by_n.setdefault(n, {})
            for key in _RANK_HISTORY_KEYS:
                value = row.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    slot[f"{mode}/{key}"] = float(value)
            dims = row.get("dims_for_variance") or {}
            if isinstance(dims, Mapping) and "0.9" in dims:
                slot[f"{mode}/dims_for_var_0.9"] = int(dims["0.9"])
    return [(n, by_n[n]) for n in sorted(by_n)]


def _rank_summary(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for payload in payloads:
        mode = str(payload.get("mode") or "prompt")
        rows = [row for row in (payload.get("results") or []) if isinstance(row, Mapping)]
        if not rows:
            continue
        last = rows[0]
        out[f"{mode}/effective_rank"] = last.get("effective_rank")
        out[f"{mode}/n"] = last.get("n_drawn") or last.get("train_n_samples")
    return out


def log_train_embed_rank_directory(
    out_dir: str | Path,
    *,
    json_paths: Sequence[Path] | None = None,
    project: str | None = None,
    entity: str | None = None,
    group: str | None = None,
    name: str | None = None,
    wandb_dir: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    directory = Path(out_dir).expanduser().resolve()
    reports = [Path(path) for path in json_paths] if json_paths is not None else train_embed_rank_jsons(directory)
    reports = [path for path in reports if path.is_file()]
    if not reports:
        return {"status": "empty", "name": name or "train-embed-rank"}
    payloads = [_load_json(path) for path in reports]
    task = _analysis_task(directory, payloads[0] if payloads else None)
    images = [
        path.with_suffix(".png")
        for path in reports
        if path.with_suffix(".png").is_file()
    ]
    seen = {path.resolve() for path in images}
    for png in sorted(directory.glob("*train_embed_effective_rank*.png")):
        if png.resolve() not in seen:
            images.append(png)
            seen.add(png.resolve())
    config: dict[str, Any] = {"analysis_dir": str(directory), "task": task}
    for payload in payloads:
        mode = str(payload.get("mode") or "prompt")
        nested = _json_safe_config(payload, skip=_RANK_SKIP_CONFIG)
        for key, value in nested.items():
            config[f"{mode}/{key}" if key not in ("task", "mode") else key] = value
        embed_meta = payload.get("embeddings") or {}
        if isinstance(embed_meta, Mapping) and "embed_dim" in embed_meta:
            config[f"{mode}/embed_dim"] = embed_meta["embed_dim"]
    return log_offline_analysis(
        name=name or f"{task}-train-embed-rank",
        group=group or f"{task}-analysis",
        job_type=OFFLINE_ANALYSIS_JOB_TYPE,
        tags=[task, "train-embed-rank", "analysis"],
        images=images,
        json_paths=reports,
        config=config,
        summary=_rank_summary(payloads),
        history=_rank_history(payloads),
        project=project,
        entity=entity,
        wandb_dir=wandb_dir,
        meta_dir=directory,
        meta_key="train_embed_rank",
        force=force,
        dry_run=dry_run,
    )


def maybe_log_train_embed_rank(
    out_dir: str | Path,
    json_paths: Sequence[str | Path],
    *,
    project: str | None,
    entity: str | None = None,
    group: str | None = None,
    name: str | None = None,
    wandb_dir: str | None = None,
    no_wandb: bool = False,
) -> dict[str, Any] | None:
    if not wandb_enabled(project=project, no_wandb=no_wandb):
        return None
    try:
        result = log_train_embed_rank_directory(
            out_dir,
            json_paths=[Path(path) for path in json_paths],
            project=project,
            entity=entity,
            group=group,
            name=name,
            wandb_dir=wandb_dir,
        )
    except Exception as exc:
        print(f"[wandb] failed to log train embed rank: {exc}", flush=True)
        return {"status": "error", "error": str(exc)}
    _print_analysis_status(result)
    return result


def local_erank_jsons(directory: Path) -> list[Path]:
    return sorted(
        {
            path
            for pattern in (
                "local_semantic_erank*.json",
                "domain_semantic_erank*.json",
            )
            for path in directory.glob(pattern)
            if path.suffix == ".json"
        }
    )


def _as_finite_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _summary_mean(stats: Mapping[str, Any], metric: str) -> float | None:
    nested = stats.get(metric)
    if isinstance(nested, Mapping):
        return _as_finite_float(nested.get("mean"))
    if metric == "erank":
        return _as_finite_float(stats.get("mean"))
    if metric == "teacher_align":
        return _as_finite_float(stats.get("mean_teacher_align"))
    return None


def _local_erank_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    domain_payload = "n_sobol_points" in payload
    for condition in payload.get("conditions") or []:
        if not isinstance(condition, Mapping):
            continue
        name = str(condition.get("name") or "condition")
        if domain_payload:
            erank = _as_finite_float(condition.get("erank"))
            if erank is not None:
                out[f"{name}/domain/erank"] = erank
            std = _as_finite_float(condition.get("erank_std"))
            if std is not None:
                out[f"{name}/domain/erank_std"] = std
            n_unique = condition.get("n_unique")
            if isinstance(n_unique, (int, float)) and not isinstance(n_unique, bool):
                out[f"{name}/domain/n_unique"] = int(n_unique)
            continue
        summary = condition.get("summary") or {}
        if not isinstance(summary, Mapping):
            continue
        for set_name, stats in summary.items():
            if not isinstance(stats, Mapping):
                continue
            mean = _summary_mean(stats, "erank")
            if mean is not None:
                out[f"{name}/{set_name}/mean_erank"] = mean
            align = _summary_mean(stats, "teacher_align")
            if align is not None:
                out[f"{name}/{set_name}/mean_teacher_align"] = align
    return out


def log_local_erank_directory(
    out_dir: str | Path,
    *,
    json_paths: Sequence[Path] | None = None,
    project: str | None = None,
    entity: str | None = None,
    group: str | None = None,
    name: str | None = None,
    wandb_dir: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    directory = Path(out_dir).expanduser().resolve()
    reports = (
        [Path(path) for path in json_paths]
        if json_paths is not None
        else local_erank_jsons(directory)
    )
    reports = [path for path in reports if path.is_file()]
    if not reports:
        return {"status": "empty", "name": name or "local-semantic-erank"}
    payloads = [_load_json(path) for path in reports]
    task = _analysis_task(directory, payloads[0] if payloads else None)
    images = [
        path.with_suffix(".png")
        for path in reports
        if path.with_suffix(".png").is_file()
    ]
    extra_files = list(reports)
    seen = {path.resolve() for path in images}
    for extra in sorted(directory.glob("local_breadth.png")):
        if extra.resolve() not in seen:
            images.append(extra)
            seen.add(extra.resolve())
    config: dict[str, Any] = {"analysis_dir": str(directory), "task": task}
    summary: dict[str, Any] = {}
    for payload in payloads:
        config.update(_json_safe_config(payload, skip=_LOCAL_ERANK_SKIP_CONFIG))
        summary.update(_local_erank_summary(payload))
    return log_offline_analysis(
        name=name or f"{task}-local-semantic-erank",
        group=group or f"{task}-analysis",
        job_type=OFFLINE_ANALYSIS_JOB_TYPE,
        tags=[task, "local-erank", "domain-erank", "analysis"],
        images=images,
        json_paths=extra_files,
        config=config,
        summary=summary,
        project=project,
        entity=entity,
        wandb_dir=wandb_dir,
        meta_dir=directory,
        meta_key="local_semantic_erank",
        force=force,
        dry_run=dry_run,
    )


def maybe_log_local_erank(
    out_dir: str | Path,
    json_path: str | Path,
    *,
    json_paths: Sequence[str | Path] | None = None,
    project: str | None,
    entity: str | None = None,
    group: str | None = None,
    name: str | None = None,
    wandb_dir: str | None = None,
    no_wandb: bool = False,
    force: bool = False,
) -> dict[str, Any] | None:
    if not wandb_enabled(project=project, no_wandb=no_wandb):
        return None
    reports = (
        [Path(path) for path in json_paths]
        if json_paths is not None
        else [Path(json_path)]
    )
    try:
        result = log_local_erank_directory(
            out_dir,
            json_paths=reports,
            project=project,
            entity=entity,
            group=group,
            name=name,
            wandb_dir=wandb_dir,
            force=force,
        )
    except Exception as exc:
        print(f"[wandb] failed to log local erank: {exc}", flush=True)
        return {"status": "error", "error": str(exc)}
    _print_analysis_status(result)
    return result


def named_analysis_reports(directory: Path) -> list[Path]:
    found: list[Path] = []
    for prefix in _NAMED_ANALYSIS_PREFIXES:
        for path in sorted(directory.glob(f"{prefix}*.json")):
            if path.suffix == ".json":
                found.append(path)
    return found


def _pngs_for_named_report(
    directory: Path, json_path: Path, report_stems: Sequence[str]
) -> list[Path]:
    stem = json_path.stem
    longer = [other for other in report_stems if other != stem and other.startswith(stem)]
    matched: list[Path] = []
    for png in sorted(directory.glob("*.png")):
        if png.stem != stem and not png.stem.startswith(stem + "_"):
            continue
        if any(
            png.stem == other or png.stem.startswith(other + "_") for other in longer
        ):
            continue
        matched.append(png)
    return matched


def _flatten_summary_tree(
    block: Mapping[str, Any], out: dict[str, Any], *, prefix: str = ""
) -> None:
    """Copy scalar trees from a report ``summary`` onto the W&B summary."""
    for key, value in block.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            out[name] = float(value)
        elif isinstance(value, Mapping):
            _flatten_summary_tree(value, out, prefix=name)


def _named_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = _encoder_cluster_summary(payload)
    summary = payload.get("summary")
    if isinstance(summary, Mapping):
        _flatten_summary_tree(summary, out)
    drowning = payload.get("drowning")
    if isinstance(drowning, Mapping):
        for key, value in drowning.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[f"drowning/{key}"] = value
    results = payload.get("results")
    if isinstance(results, Mapping):
        for model, block in results.items():
            variants = block.get("variants") if isinstance(block, Mapping) else None
            if not isinstance(variants, Mapping):
                continue
            for variant, stats in variants.items():
                if not isinstance(stats, Mapping):
                    continue
                gap = stats.get("gap_matched_minus_random")
                if isinstance(gap, (int, float)) and not isinstance(gap, bool):
                    out[f"{model}/{variant}/gap"] = gap
    for condition in payload.get("conditions") or []:
        if not isinstance(condition, Mapping):
            continue
        name = str(condition.get("name") or "condition")
        for key in (
            "utilization",
            "spearman_rho",
            "erank",
            "n",
            "rank",
            "mu_norm_mean",
        ):
            value = condition.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[f"{name}/{key}"] = float(value)
    splits = payload.get("splits")
    if isinstance(splits, Mapping):
        for split_name, block in splits.items():
            if not isinstance(block, Mapping):
                continue
            for key in ("n", "n_valid", "validity"):
                value = block.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    out[f"{split_name}/{key}"] = float(value)
            oracles = block.get("oracles")
            if not isinstance(oracles, Mapping):
                continue
            for oracle_name, stats in oracles.items():
                if not isinstance(stats, Mapping):
                    continue
                for key, value in stats.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        out[f"{split_name}/{oracle_name}/{key}"] = float(value)
    return out


def log_named_analysis_report(
    out_dir: str | Path,
    json_path: str | Path,
    *,
    project: str | None = None,
    entity: str | None = None,
    group: str | None = None,
    name: str | None = None,
    wandb_dir: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    directory = Path(out_dir).expanduser().resolve()
    report_path = Path(json_path).expanduser().resolve()
    payload = _load_json(report_path)
    task = _analysis_task(directory, payload)
    stems = [path.stem for path in named_analysis_reports(directory)]
    images = _pngs_for_named_report(directory, report_path, stems)
    slug = report_path.stem.replace("_", "-")
    config = _json_safe_config(payload, skip=_NAMED_SKIP_CONFIG)
    config["analysis_dir"] = str(directory)
    config["report"] = report_path.name
    return log_offline_analysis(
        name=name or f"{task}-{slug}",
        group=group or f"{task}-analysis",
        job_type=OFFLINE_ANALYSIS_JOB_TYPE,
        tags=[task, slug, "analysis"],
        images=images,
        json_paths=[report_path] if report_path.is_file() else [],
        config=config,
        summary=_named_summary(payload),
        project=project,
        entity=entity,
        wandb_dir=wandb_dir,
        meta_dir=directory,
        meta_key=report_path.stem,
        force=force,
        dry_run=dry_run,
    )


def maybe_log_named_analysis(
    out_dir: str | Path,
    json_path: str | Path,
    *,
    project: str | None,
    entity: str | None = None,
    group: str | None = None,
    name: str | None = None,
    wandb_dir: str | None = None,
    no_wandb: bool = False,
) -> dict[str, Any] | None:
    if not wandb_enabled(project=project, no_wandb=no_wandb):
        return None
    try:
        result = log_named_analysis_report(
            out_dir,
            json_path,
            project=project,
            entity=entity,
            group=group,
            name=name,
            wandb_dir=wandb_dir,
        )
    except Exception as exc:
        print(f"[wandb] failed to log {json_path}: {exc}", flush=True)
        return {"status": "error", "error": str(exc)}
    _print_analysis_status(result)
    return result


def log_analysis_dir(
    directory: str | Path,
    *,
    project: str | None = None,
    entity: str | None = None,
    wandb_dir: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Backfill encoder-cluster, train-embed-rank, local-erank, and named JSON+PNG reports."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        return []
    results: list[dict[str, Any]] = []
    reports = [
        path for path in sorted(root.glob("encoder_cluster*.json")) if is_encoder_cluster_report(path)
    ]
    for report_path in reports:
        payload = _load_json(report_path)
        if not payload.get("artifact_tag"):
            payload = {**payload, "artifact_tag": encoder_cluster_tag_from_path(report_path)}
        try:
            results.append(
                log_encoder_cluster_report(
                    root,
                    payload,
                    project=project,
                    entity=entity,
                    wandb_dir=wandb_dir,
                    force=force,
                    dry_run=dry_run,
                )
            )
        except Exception as exc:
            results.append(
                {
                    "status": "error",
                    "error": str(exc),
                    "name": report_path.name,
                }
            )
    rank_jsons = train_embed_rank_jsons(root)
    if rank_jsons:
        try:
            results.append(
                log_train_embed_rank_directory(
                    root,
                    json_paths=rank_jsons,
                    project=project,
                    entity=entity,
                    wandb_dir=wandb_dir,
                    force=force,
                    dry_run=dry_run,
                )
            )
        except Exception as exc:
            results.append(
                {
                    "status": "error",
                    "error": str(exc),
                    "name": "train_embed_rank",
                }
            )
    local_jsons = local_erank_jsons(root)
    if local_jsons:
        try:
            results.append(
                log_local_erank_directory(
                    root,
                    json_paths=local_jsons,
                    project=project,
                    entity=entity,
                    wandb_dir=wandb_dir,
                    force=force,
                    dry_run=dry_run,
                )
            )
        except Exception as exc:
            results.append(
                {
                    "status": "error",
                    "error": str(exc),
                    "name": "local_semantic_erank",
                }
            )
    for report_path in named_analysis_reports(root):
        try:
            results.append(
                log_named_analysis_report(
                    root,
                    report_path,
                    project=project,
                    entity=entity,
                    wandb_dir=wandb_dir,
                    force=force,
                    dry_run=dry_run,
                )
            )
        except Exception as exc:
            results.append(
                {
                    "status": "error",
                    "error": str(exc),
                    "name": report_path.name,
                }
            )
    return results


def log_analysis_directory(
    out_dir: str | Path,
    *,
    project: str | None = None,
    entity: str | None = None,
    group: str | None = DEFAULT_SEARCH_GROUP,
    name: str = "semantle-search-comparison",
    wandb_dir: str | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Upload ``analyze_results.py`` figures and ``metrics.json`` as one run."""
    directory = Path(out_dir).expanduser().resolve()
    images = sorted(directory.glob("*.png"))
    metrics_path = directory / "metrics.json"
    json_paths = [metrics_path] if metrics_path.is_file() else []
    payload = _load_json(metrics_path) if metrics_path.is_file() else {}
    summary: dict[str, Any] = {}
    for row in payload.get("table") or []:
        method = row.get("method")
        if not method:
            continue
        for split in ("train", "test"):
            block = row.get(split) or {}
            if "success_rate" in block:
                summary[f"{method}/{split}/success_rate"] = block["success_rate"]
            if "mean_best" in block:
                summary[f"{method}/{split}/mean_best"] = block["mean_best"]
    return log_offline_analysis(
        name=name,
        group=group or DEFAULT_SEARCH_GROUP,
        job_type=ANALYSIS_JOB_TYPE,
        tags=["search-comparison", "semantle"],
        images=images,
        json_paths=json_paths,
        config={"analysis_dir": str(directory)},
        summary=summary,
        project=project,
        entity=entity,
        wandb_dir=wandb_dir,
        meta_dir=directory,
        meta_key="search-comparison",
        force=force,
        dry_run=dry_run,
    )
