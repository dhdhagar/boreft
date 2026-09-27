"""Rewrite stored molopt search scores after restoring TDC RF probabilities.

Search decisions stay as they were (the queried SMILES do not change). This
replaces exploded GSK3β / JNK3 leaf counts in ``observations.jsonl``,
``summary.json``, and best-so-far plots with class-1 probabilities in
``[0, 1]``. Invalid molecules still score 0. Stored strings are scored as-is
(no second SmiSelf pass); SMILES tags are unwrapped before the oracle call.
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping, Sequence

from boreft.baselines.base import BaselineObservation, BaselineRunState
from boreft.bo.plotting import plot_best_so_far
from boreft.bo.state import Observation, RunState
from boreft.chem import unwrap_smiles_tags
from boreft.oracles import (
    MOLOPT_SEARCH_ORACLE_NAMES,
    MoleculeScore,
    OracleFn,
    load_property_oracle,
    normalize_property_oracle_name,
    score_molecules,
)
from boreft.search_wandb import (
    DEFAULT_ENTITY,
    DEFAULT_PROJECT,
    iter_seed_dirs,
    log_seed_directory,
    patch_seed_wandb_summary,
    read_meta,
)

SCORE_KIND = "class1_proba"
BACKUP_SUFFIX = ".pre_proba"
DEFAULT_SEARCH_ROOT = Path("experiments") / "outputs" / "molopt" / "search"
_BOREFT_SUMMARY_KEEP = (
    "seed",
    "elapsed_seconds",
    "n_repeat_samples",
    "n_repeat_proposals",
    "found_target",
    "found_at_index",
    "found_at_verifications",
    "n_expansion_rounds",
)
_BASELINE_SUMMARY_KEEP = (
    "seed",
    "found_target",
    "found_at_index",
    "found_at_verifications",
)

OracleFactory = Callable[[str], OracleFn]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _is_boreft_row(row: Mapping[str, Any]) -> bool:
    return "point" in row and "decoded" in row


def oracle_name_for_seed(seed_dir: Path) -> str | None:
    """Oracle from ``config.json``, else the run-directory name."""
    config = _load_json(seed_dir.parent / "config.json")
    raw = config.get("oracle") or ""
    if not raw:
        raw = seed_dir.parent.name
    try:
        return normalize_property_oracle_name(str(raw))
    except ValueError:
        return None


def _observation_texts(item: Observation | BaselineObservation) -> list[str]:
    if isinstance(item, Observation):
        samples = list(item.decoded_samples or [])
        fallback = item.decoded
    else:
        samples = list(item.solution_samples or [])
        fallback = item.solution
    if samples:
        return [str(text) for text in samples]
    return [str(fallback or "")]


def _scoring_text(text: str) -> str:
    return unwrap_smiles_tags(text)


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def _peak(item: Observation | BaselineObservation | Mapping[str, Any]) -> float:
    if isinstance(item, (Observation, BaselineObservation)):
        return float(item.peak_score())
    scores = item.get("sample_scores") or []
    if scores:
        return max(float(value) for value in scores)
    return float(item.get("score") or 0.0)


def _backup(path: Path) -> Path | None:
    dest = path.with_name(path.name + BACKUP_SUFFIX)
    if dest.exists() or not path.exists():
        return dest if dest.exists() else None
    shutil.copy2(path, dest)
    return dest


def _mark_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {**summary, "oracle_score_kind": SCORE_KIND}


def _already_rescored(seed_dir: Path) -> bool:
    summary = _load_json(seed_dir / "summary.json")
    return summary.get("oracle_score_kind") == SCORE_KIND


def _keep(previous: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: previous[key] for key in keys if key in previous}


def _apply_row_components(
    components: dict[str, Any],
    rows: Sequence[MoleculeScore],
    scores: Sequence[float],
) -> dict[str, Any]:
    updated = dict(components)
    if len(rows) == 1:
        row = rows[0]
        updated["oracle_score"] = float(row.score)
        updated["valid"] = bool(row.valid)
        updated["canonical"] = row.canonical or ""
        if row.qed is not None:
            updated["qed"] = float(row.qed)
        return updated
    updated["oracle_score"] = float(sum(scores) / len(scores)) if scores else 0.0
    representative = rows[max(range(len(scores)), key=scores.__getitem__)]
    updated["valid"] = bool(representative.valid)
    updated["canonical"] = representative.canonical or ""
    if representative.qed is not None:
        updated["qed"] = float(representative.qed)
    return updated


def rescore_seed_directory(
    seed_dir: str | Path,
    *,
    oracle: OracleFn,
    oracle_name: str,
    dry_run: bool = False,
    force: bool = False,
    write_plots: bool = True,
) -> dict[str, Any]:
    """Rescore one ``seed_*`` directory. Returns a status dict."""
    path = Path(seed_dir)
    obs_path = path / "observations.jsonl"
    identity = {
        "seed_dir": str(path),
        "run_dir": str(path.parent),
        "oracle": oracle_name,
    }
    if not obs_path.is_file():
        return {"status": "missing", **identity}
    if _already_rescored(path) and not force:
        return {"status": "skipped", **identity}

    first = next(
        (
            json.loads(line)
            for line in obs_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ),
        None,
    )
    if first is None:
        return {"status": "empty", **identity}

    boreft = _is_boreft_row(first)
    if boreft:
        loaded: RunState | BaselineRunState = RunState.load(obs_path)
    else:
        loaded = BaselineRunState.load(obs_path)
    observations = loaded.observations
    if not observations:
        return {"status": "empty", **identity}

    texts = [_scoring_text(text) for item in observations for text in _observation_texts(item)]
    spans: list[tuple[int, int]] = []
    cursor = 0
    for item in observations:
        n = len(_observation_texts(item))
        spans.append((cursor, cursor + n))
        cursor += n
    scored = score_molecules(texts, oracle, include_qed=True)
    old_best = max(_peak(item) for item in observations)
    new_best = max((row.score for row in scored), default=0.0)

    payload = {
        "status": "dry_run" if dry_run else "rescored",
        "n_observations": len(observations),
        "n_scores": len(scored),
        "old_best": old_best,
        "new_best": new_best,
        **identity,
    }
    if dry_run:
        return payload

    _backup(obs_path)
    summary_path = path / "summary.json"
    if summary_path.is_file():
        _backup(summary_path)
    previous = _load_json(summary_path)

    if boreft:
        assert isinstance(loaded, RunState)
        rewritten = _rewrite_boreft(observations, scored, spans)
        loaded.rewrite(rewritten)
        summary = _mark_summary({**loaded.summary(), **_keep(previous, _BOREFT_SUMMARY_KEEP)})
        plot_title = f"Seed {previous.get('seed', path.name)}"
    else:
        assert isinstance(loaded, BaselineRunState)
        rewritten_b = _rewrite_baseline(observations, scored, spans)
        loaded.rewrite(rewritten_b)
        summary = _mark_summary(
            {
                **loaded.summary(elapsed_seconds=previous.get("elapsed_seconds")),
                **_keep(previous, _BASELINE_SUMMARY_KEEP),
            }
        )
        plot_title = f"{path.parent.name} {path.name}"
    if write_plots:
        plot_best_so_far(
            [loaded.observations],
            path / "best_so_far.png",
            title=plot_title,
        )
    _write_json(summary_path, summary)
    payload["summary"] = summary
    return payload


def _rewrite_boreft(
    observations: Sequence[Observation],
    scored: Sequence[MoleculeScore],
    spans: Sequence[tuple[int, int]],
) -> list[Observation]:
    rewritten: list[Observation] = []
    best: float | None = None
    for item, (start, end) in zip(observations, spans, strict=True):
        rows = scored[start:end]
        scores = [float(row.score) for row in rows]
        mean, std = _mean_std(scores)
        n = max(len(scores), 1)
        if len(rows) == 1 or not item.decoded_samples:
            decoded = item.decoded
        else:
            decoded = item.decoded_samples[max(range(len(scores)), key=scores.__getitem__)]
        peak = max(scores) if scores else mean
        best = peak if best is None else max(best, peak)
        rewritten.append(
            replace(
                item,
                decoded=decoded,
                score=float(mean),
                score_std=float(std),
                score_sem=float(std / math.sqrt(n)),
                sample_count=n,
                sample_scores=scores,
                components=_apply_row_components(dict(item.components), rows, scores),
                best_so_far=best,
            )
        )
    return rewritten


def _rewrite_baseline(
    observations: Sequence[BaselineObservation],
    scored: Sequence[MoleculeScore],
    spans: Sequence[tuple[int, int]],
) -> list[BaselineObservation]:
    rewritten: list[BaselineObservation] = []
    best: float | None = None
    for item, (start, end) in zip(observations, spans, strict=True):
        rows = scored[start:end]
        scores = [float(row.score) for row in rows]
        mean, std = _mean_std(scores)
        n = max(len(scores), 1)
        if len(rows) == 1 or not item.solution_samples:
            solution = item.solution
        else:
            solution = item.solution_samples[
                max(range(len(scores)), key=scores.__getitem__)
            ]
        best = mean if best is None else max(best, mean)
        rewritten.append(
            replace(
                item,
                solution=solution,
                score=float(mean),
                score_std=float(std),
                score_sem=float(std / math.sqrt(n)),
                sample_count=n,
                sample_scores=scores,
                components=_apply_row_components(dict(item.components), rows, scores),
                best_so_far=best,
            )
        )
    return rewritten


def _seed_dirs(run_dir: Path) -> list[Path]:
    return sorted(
        path.parent
        for path in Path(run_dir).glob("seed_*/observations.jsonl")
        if path.parent.is_dir()
    )


def _run_dir_ready(run_dir: Path) -> bool:
    seed_dirs = _seed_dirs(run_dir)
    return bool(seed_dirs) and all(_already_rescored(path) for path in seed_dirs)


def _sum_field(summaries: Mapping[str, Mapping[str, Any]], key: str) -> float:
    return float(sum(float(summary.get(key) or 0.0) for summary in summaries.values()))


def _baseline_parent_summary(
    summaries: dict[str, dict], seeds: tuple[int, ...]
) -> dict[str, Any]:
    found_seeds = [
        int(summaries[str(seed)].get("seed", seed))
        for seed in seeds
        if summaries[str(seed)].get("found_target")
    ]
    best_run = max(
        summaries.values(),
        key=lambda item: (
            item.get("best_score") is not None,
            item.get("best_score") or 0.0,
        ),
    )
    scores = [
        float(summaries[str(seed)]["best_score"])
        for seed in seeds
        if summaries[str(seed)].get("best_score") is not None
    ]
    fastest = [
        (
            int(summaries[str(seed)]["found_at_verifications"]),
            int(summaries[str(seed)].get("seed", seed)),
        )
        for seed in seeds
        if summaries[str(seed)].get("found_target")
        and summaries[str(seed)].get("found_at_verifications") is not None
    ]
    fastest_hit = min(fastest) if fastest else None
    return {
        "runs": summaries,
        "seeds": list(seeds),
        "elapsed_seconds": _sum_field(summaries, "elapsed_seconds"),
        "bbox_eval_seconds": _sum_field(summaries, "bbox_eval_seconds"),
        "n_repeat_samples": int(_sum_field(summaries, "n_repeat_samples")),
        "n_repeat_proposals": int(_sum_field(summaries, "n_repeat_proposals")),
        "found_target_seeds": found_seeds,
        "n_found_target": len(found_seeds),
        "best_score": best_run.get("best_score"),
        "best_solution": best_run.get("best_solution"),
        "best_seed": best_run.get("seed"),
        "mean_best_score": (sum(scores) / len(scores)) if scores else None,
        "fastest_found_at_verifications": (
            None if fastest_hit is None else fastest_hit[0]
        ),
        "fastest_found_seed": None if fastest_hit is None else fastest_hit[1],
    }


def rebuild_run_summary(run_dir: Path, *, write_plots: bool = True) -> dict[str, Any]:
    """Refresh the parent ``summary.json`` and across-seed plot from seed files."""
    parent = Path(run_dir)
    seed_dirs = _seed_dirs(parent)
    if not seed_dirs:
        return {}
    summaries: dict[str, dict] = {}
    runs = []
    seeds: list[int] = []
    boreft = False
    for seed_dir in seed_dirs:
        summary = _load_json(seed_dir / "summary.json")
        if not summary:
            continue
        seed = int(summary.get("seed", seed_dir.name.rsplit("_", 1)[-1]))
        summaries[str(seed)] = summary
        seeds.append(seed)
        obs_path = seed_dir / "observations.jsonl"
        first = next(
            (
                json.loads(line)
                for line in obs_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ),
            None,
        )
        if first is None:
            continue
        if _is_boreft_row(first):
            boreft = True
            runs.append(RunState.load(obs_path).observations)
        else:
            runs.append(BaselineRunState.load(obs_path).observations)
    if not summaries:
        return {}
    previous = _load_json(parent / "summary.json")
    if boreft:
        aggregate = {
            **previous,
            "runs": summaries,
            "seeds": previous.get("seeds", seeds),
        }
    else:
        ordered = tuple(int(seed) for seed in (previous.get("seeds") or seeds))
        present = tuple(seed for seed in ordered if str(seed) in summaries)
        aggregate = _baseline_parent_summary(summaries, present or tuple(seeds))
        for key, value in previous.items():
            if key not in aggregate:
                aggregate[key] = value
    aggregate["oracle_score_kind"] = SCORE_KIND
    _write_json(parent / "summary.json", aggregate)
    if write_plots and runs:
        plot_best_so_far(runs, parent / "best_so_far.png", title=parent.name)
    return aggregate


def _oracles_filter(names: Sequence[str] | None) -> set[str]:
    if not names:
        return set(MOLOPT_SEARCH_ORACLE_NAMES)
    return {normalize_property_oracle_name(name) for name in names}


def rescore_search_tree(
    roots: Sequence[str | Path],
    *,
    oracles: Sequence[str] | None = None,
    oracle_factory: OracleFactory | None = None,
    dry_run: bool = False,
    force: bool = False,
    write_plots: bool = True,
    wandb_summary: bool = False,
    wandb_relog: bool = False,
    wandb_project: str | None = DEFAULT_PROJECT,
    wandb_entity: str | None = DEFAULT_ENTITY,
) -> list[dict[str, Any]]:
    """Walk search trees, rescore property-oracle seeds, rebuild parent summaries."""
    allowed = _oracles_filter(oracles)
    factory = oracle_factory or load_property_oracle
    loaded: dict[str, OracleFn] = {}
    results: list[dict[str, Any]] = []
    run_dirs: set[Path] = set()

    def oracle_for(name: str) -> OracleFn:
        if name not in loaded:
            loaded[name] = factory(name)
        return loaded[name]

    for root in roots:
        for seed_dir in iter_seed_dirs(root):
            name = oracle_name_for_seed(seed_dir)
            if name is None or name not in allowed:
                continue
            try:
                row = rescore_seed_directory(
                    seed_dir,
                    oracle=oracle_for(name),
                    oracle_name=name,
                    dry_run=dry_run,
                    force=force,
                    write_plots=write_plots and not dry_run,
                )
            except Exception as exc:
                row = {
                    "status": "error",
                    "error": str(exc),
                    "seed_dir": str(seed_dir),
                    "oracle": name,
                }
            disk_ok = row.get("status") in {"rescored", "skipped"}
            if wandb_summary and disk_ok:
                row["wandb_summary"] = patch_seed_wandb_summary(
                    seed_dir,
                    project=wandb_project,
                    entity=wandb_entity,
                    dry_run=dry_run,
                )
            if wandb_relog and disk_ok:
                meta = read_meta(seed_dir)
                row["wandb_relog"] = log_seed_directory(
                    seed_dir,
                    project=wandb_project or meta.get("project"),
                    entity=wandb_entity or meta.get("entity"),
                    group=meta.get("group"),
                    overwrite_history=True,
                    dry_run=dry_run,
                )
            if row.get("status") == "rescored":
                run_dirs.add(seed_dir.parent)
            results.append(row)
    if not dry_run:
        for run_dir in sorted(run_dirs):
            if not _run_dir_ready(run_dir):
                results.append(
                    {
                        "status": "error",
                        "error": "parent summary skipped: not every seed is class1_proba",
                        "run_dir": str(run_dir),
                    }
                )
                continue
            try:
                rebuild_run_summary(run_dir, write_plots=write_plots)
            except Exception as exc:
                results.append(
                    {
                        "status": "error",
                        "error": f"parent summary: {exc}",
                        "run_dir": str(run_dir),
                    }
                )
    return results
