"""Consolidated post-training evaluation for semantle BOReFT models.

Runs all evaluation steps in sequence and logs everything to the same
WandB run that was used for training.

Steps:
  1. RECON / RECON_TEST / GENZ / DIST — in-process via eval.semantle.run_semantle_generation_eval
  2. Oracle screen (molopt) — train / high-tail test / Sobol DRD2·GSK3B·JNK3 histograms
  3. Cluster PCA plot    — bias-vector PCA coloured by cluster or definition category
  4. LIPZ interpolation  — lerp/slerp paths for sampled/specified word pairs

Metrics are logged under the recon/ recon_test/ genz/ oracle_screen/ lipz/ dist/ namespaces.
Numeric metrics go to wandb.run.summary (no one-point charts, no sample tables).
Images go to wandb.log (shows in WandB Media panel).

Usage (automatic via ``python -m boreft.train --run-full-eval``, or standalone):

    python -m boreft.eval.run_full_eval \\
        --output_dir  outputs/out_vae-... \\
        --cache_dir   ~/.cache/huggingface

    Model shape (layer, rank, model_name, …) and eval hyperparameters default from
    training_config.json in ``output_dir``. Override any field explicitly when needed.

Add --no_wandb to skip WandB and only write result files locally.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import sys
import traceback
from dataclasses import dataclass, fields, replace
from itertools import combinations
from typing import List, Optional

import numpy as np
import wandb


class _Tee(io.TextIOBase):
    """Write to both the original stream and a log file simultaneously."""

    def __init__(self, original, log_path: str):
        super().__init__()
        self._orig = original
        self._file = open(log_path, "w", buffering=1, encoding="utf-8")
        self._file_closed = False

    def write(self, s: str) -> int:
        self._orig.write(s)
        self._orig.flush()
        if not self._file_closed:
            try:
                self._file.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass
        if not self._file_closed:
            try:
                self._file.flush()
            except Exception:
                pass

    def stop(self):
        """Restore sys.stdout and close the log file. Safe to call multiple times."""
        if sys.stdout is self:
            sys.stdout = self._orig
        if not self._file_closed:
            self._file_closed = True
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass


from boreft.data_utils import add_load_latest_argument, load_merged_run_config, TORCH_DTYPE_NAMES
from boreft.eval.eval_suite import lipz_aggregate, target_normalizer
from boreft.eval.semantle import (
    DEFAULT_BBOX_PCA_VAR,
    DEFAULT_EMBED_SIM_TAU,
    DEFAULT_TEST_N_SAMPLES,
    EvalCheckpoint,
    SemantleGenerationEvalArgs,
    DEFAULT_MAX_NEW_TOKENS,
    load_eval_checkpoint,
    release_eval_checkpoint,
    run_semantle_generation_eval,
    subset_eval_checkpoint,
)
from boreft.eval.interpolate import (
    normalize_interp_method,
    run_interpolation,
    safe_filename,
)
from boreft.eval.plot_cluster_pca import (
    plot_cluster_pca,
    uses_definition_embed_categories,
)


def _geometry_eval_enabled(output_dir: str, semantle_dir: Optional[str]) -> bool:
    """True when the cluster-PCA plot can resolve cluster or category labels."""
    return bool(semantle_dir) or uses_definition_embed_categories(output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# WandB
# ─────────────────────────────────────────────────────────────────────────────


def _init_wandb(
    output_dir: str,
    no_wandb: bool,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    wandb_group: str | None = None,
    wandb_dir: str | None = None,
):
    """Resume the training WandB run so eval metrics land on the same run page.

    Mirrors train.py: ``--wandb_project`` / ``--wandb_entity`` / ``--wandb_run_name`` /
    ``--wandb_group`` / ``--wandb_dir``; sets ``WANDB_PROJECT`` / ``WANDB_ENTITY`` /
    ``WANDB_GROUP`` in the environment when those args are set.

    train.py writes wandb_meta.json (preferred) or wandb_run_id.txt (legacy) to
    output_dir. When post-training eval runs in the same process as training,
    train.py leaves the WandB run open and we continue it here. Otherwise we
    re-open the closed run with resume='allow' (not 'must', which would fail on
    a finished run).

    If there is no saved run id, behaviour matches train: WandB is only started when a
    project name is available (``--wandb_project``, ``WANDB_PROJECT``, or ``project`` in meta).
    """
    if no_wandb:
        return None

    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_entity:
        os.environ["WANDB_ENTITY"] = wandb_entity
    if wandb_group:
        os.environ["WANDB_GROUP"] = wandb_group
    if wandb_dir:
        os.environ["WANDB_DIR"] = wandb_dir
        os.makedirs(wandb_dir, exist_ok=True)

    meta_path = os.path.join(output_dir, "wandb_meta.json")
    run_id_path = os.path.join(output_dir, "wandb_run_id.txt")

    run_id: str | None = None
    project: str | None = None
    entity: str | None = None
    orig_program: str | None = None  # original training script path, saved by train.py
    orig_args: list = []  # original training CLI args, saved by train.py

    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        run_id = meta.get("run_id")
        project = meta.get("project")
        entity = meta.get("entity")
        orig_program = meta.get("program")  # may be None for older checkpoints
        orig_args = meta.get("args", [])
    elif os.path.exists(run_id_path):
        with open(run_id_path, encoding="utf-8") as f:
            run_id = f.read().strip() or None

    if not project:
        project = wandb_project or os.environ.get("WANDB_PROJECT")

    if not entity:
        entity = wandb_entity or os.environ.get("WANDB_ENTITY")

    # In-process handoff from train.py: reuse the active run instead of finish/resume.
    if wandb.run is not None and (run_id is None or wandb.run.id == run_id):
        os.environ["WANDB_CONSOLE"] = "off"
        print(f"[full_eval] Continuing active WandB run {wandb.run.id}")
        return wandb.run

    # --- init ---
    if run_id:
        if not project:
            print(
                "[full_eval] WARNING: saved WandB run id but no project "
                "(add project to wandb_meta.json or pass --wandb_project / set WANDB_PROJECT). "
                "Skipping WandB.",
                file=sys.stderr,
            )
            return None
        print(
            f"[full_eval] Resuming WandB run {run_id} (entity={entity!r}, project={project!r})"
        )
        # This prevents output.log from being overwritten when eval resumes the training run — training stdout stays intact in WandB Files.
        os.environ["WANDB_CONSOLE"] = "off"
        init_kwargs: dict = {"id": run_id, "project": project, "resume": "allow"}
        if entity:
            init_kwargs["entity"] = entity
        if wandb_dir:
            init_kwargs["dir"] = wandb_dir
        # Preserve the original training command so the WandB UI "Command" field
        # keeps showing train.py (+ its args) instead of run_full_eval.py.
        # WandB captures sys.argv at init time, so we temporarily swap it.
        if orig_program:
            _orig_argv = sys.argv[:]
            try:
                sys.argv = [orig_program] + list(orig_args)
                return wandb.init(**init_kwargs)
            finally:
                sys.argv = _orig_argv
        return wandb.init(**init_kwargs)

    if not project:
        print(
            "[full_eval] No WandB logging: pass --wandb_project (same as train.py) or set WANDB_PROJECT.",
            file=sys.stderr,
        )
        return None

    print(
        "[full_eval] No WandB run ID found — starting a fresh run (project=%r)."
        % (project,)
    )
    config = load_merged_run_config(output_dir)
    name = wandb_run_name or f"eval-{os.path.basename(output_dir)}"
    fresh: dict = {"project": project, "name": name, "dir": wandb_dir, "config": config}
    if entity:
        fresh["entity"] = entity
    if wandb_group:
        fresh["group"] = wandb_group
    return wandb.init(**fresh)


def _wandb_log(metrics: dict):
    """Write metrics to WandB.

    - All metrics (scalars + media) → wandb.log() so charts and media panel work.
    - Scalars also mirrored to wandb.run.summary for the Run Summary table.

    Note: output.log is protected by console="off" in _init_wandb so training
    stdout is not overwritten when eval resumes the run.
    """
    if wandb.run is None:
        return
    wandb.log(metrics)
    for k, v in metrics.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float, np.integer, np.floating)):
            wandb.run.summary[k] = float(v)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Generation quality  (eval/semantle.py)
# ─────────────────────────────────────────────────────────────────────────────


def run_generation_eval(
    cfg: "FullEvalConfig",
    *,
    checkpoint: EvalCheckpoint,
) -> dict | None:
    """Run the RECON / GENZ / DIST eval in-process on the shared checkpoint."""
    gen_args = SemantleGenerationEvalArgs(
        output_dir=cfg.output_dir,
        model_name=cfg.model_name,
        layer=cfg.layer,
        low_rank_dim=cfg.low_rank_dim,
        cache_dir=cfg.cache_dir,
        n_samples=cfg.n_samples,
        eval_batch_size=cfg.eval_batch_size,
        top_p=cfg.top_p if cfg.top_p is not None else 1.0,
        use_word_bias=None,
        variance=None,
        seed=cfg.seed,
        # Subset is applied once on the shared checkpoint in run_full_eval.
        full_eval_n_samples=None
        if checkpoint.full_items is not None
        else cfg.full_eval_n_samples,
        n_uniform=cfg.n_uniform,
        position=cfg.position or "l1",
        torch_dtype=cfg.torch_dtype,
        max_new_tokens=cfg.max_new_tokens,
        embed_sim_tau=cfg.embed_sim_tau
        if cfg.embed_sim_tau is not None
        else DEFAULT_EMBED_SIM_TAU,
        test_n_samples=cfg.test_n_samples
        if cfg.test_n_samples is not None
        else DEFAULT_TEST_N_SAMPLES,
        bbox_pca_var=cfg.bbox_pca_var
        if cfg.bbox_pca_var is not None
        else DEFAULT_BBOX_PCA_VAR,
        load_latest=bool(cfg.load_latest),
    )
    try:
        return run_semantle_generation_eval(
            gen_args,
            checkpoint=checkpoint,
            release_model=False,
        )
    except Exception as e:
        print(f"[full_eval] WARNING: generation eval failed: {e}")
        traceback.print_exc()
        return None


def _suite_metrics_for_wandb(results: dict) -> dict:
    """Flatten recon/recon_test/genz/dist blocks from results.json into namespaced scalars.

    LIPZ is logged separately by the interpolation step. No per-sample tables are
    emitted (only numeric metrics).
    """
    flat: dict = {}
    for group in ("recon", "recon_test", "genz", "dist"):
        block = results.get(group) or {}
        for k, v in block.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                flat[f"{group}/{k}"] = float(v)
    return flat


def _screen_test_smiles(
    gen_results: dict | None,
    output_dir: str,
    saved_cfg: dict,
    *,
    n: int,
    seed: int,
) -> list[str]:
    from boreft.chem import unwrap_smiles_tags
    from boreft.molopt_split import load_oracle_split

    rows = (gen_results or {}).get("recon_test_per_target") or []
    if rows:
        return [unwrap_smiles_tags(str(row.get("target") or "")) for row in rows if row.get("target")]
    split = load_oracle_split(output_dir, saved_cfg)
    pool = list((split or {}).get("test_smiles") or [])
    if not pool:
        return []
    n_draw = min(max(int(n), 0), len(pool))
    if n_draw <= 0:
        return []
    return random.Random(seed).sample(sorted(pool), n_draw)


def run_oracle_screen_panel(
    cfg: "FullEvalConfig",
    eval_dir: str,
    *,
    checkpoint: EvalCheckpoint,
    gen_results: dict | None,
) -> dict | None:
    """Train / high-tail test / Sobol oracle histograms for molopt checkpoints."""
    saved = load_merged_run_config(cfg.output_dir)
    if str(saved.get("task") or "") != "molopt":
        return None
    if cfg.oracle_screen is False:
        print("[full_eval] oracle screen disabled.", flush=True)
        return None
    from boreft.chem import unwrap_smiles_tags
    from boreft.molopt_split import (
        default_oracle_scores_path,
        load_oracle_score_cache,
    )
    from boreft.oracle_screen import (
        decode_sobol_smiles,
        plot_oracle_histograms,
        run_three_way_screen,
        screen_metrics_for_wandb,
        smiles_from_sobol_results,
        write_decoded_smiles,
        write_screen_report,
    )
    from boreft.oracles import TDC_ORACLE_NAMES, load_oracles

    n_sobol = int(cfg.oracle_screen_n_sobol or 0)
    train = [unwrap_smiles_tags(str(w)) for w in checkpoint.words]
    test = _screen_test_smiles(
        gen_results,
        cfg.output_dir,
        saved,
        n=int(cfg.test_n_samples or DEFAULT_TEST_N_SAMPLES),
        seed=int(cfg.seed or 42),
    )
    if not test:
        print(
            "[ORACLE_SCREEN] WARNING: no high-tail test SMILES; "
            "train/Sobol will still be scored.",
            flush=True,
        )
    sobol_path = os.path.join(eval_dir, "sobol_results.json")
    sobol: list[str] = []
    if n_sobol > 0:
        print(
            f"[ORACLE_SCREEN] Greedy Sobol decode n={n_sobol}...",
            flush=True,
        )
        sobol = decode_sobol_smiles(
            reft_model=checkpoint.reft_model,
            tokenizer=checkpoint.tokenizer,
            words=checkpoint.words,
            prompt=checkpoint.prompt,
            n_sobol=n_sobol,
            seed=int(cfg.seed or 42),
            batch_size=int(cfg.eval_batch_size or 32),
            max_new_tokens=int(cfg.max_new_tokens or DEFAULT_MAX_NEW_TOKENS),
            position=str(cfg.position or saved.get("position") or "l1"),
            assistant_suffix=checkpoint.assistant_suffix,
            from_chat_template=checkpoint.from_chat_template,
            intervention_token_id=checkpoint.intervention_token_id,
            content_span=checkpoint.content_span,
        )
        write_decoded_smiles(os.path.join(eval_dir, "oracle_screen_sobol.json"), sobol)
    elif os.path.isfile(sobol_path):
        sobol = smiles_from_sobol_results(sobol_path, section="greedy")
        print(
            f"[ORACLE_SCREEN] Reusing GENZ greedy Sobol ({len(sobol)} SMILES)",
            flush=True,
        )
    split_smiles = {"train": train, "test": test, "sobol": sobol}
    csv_path = saved.get("molopt_csv")
    cache_path = saved.get("molopt_oracle_scores_path") or (
        default_oracle_scores_path(str(csv_path)) if csv_path else ""
    )
    cache = load_oracle_score_cache(cache_path) if cache_path else {}
    oracle_fns = None
    try:
        oracle_fns = load_oracles(TDC_ORACLE_NAMES)
    except Exception as exc:
        if not cache:
            print(f"[full_eval] WARNING: oracle screen skipped (TDC): {exc}", flush=True)
            return None
        print(
            f"[full_eval] WARNING: TDC unavailable ({exc}); "
            "scoring catalog splits from cache only.",
            flush=True,
        )
        split_smiles = {key: value for key, value in split_smiles.items() if key != "sobol"}
    summaries, scores = run_three_way_screen(
        split_smiles,
        oracle_fns,
        oracle_names=TDC_ORACLE_NAMES,
        cache=cache,
    )
    hist_path = os.path.join(eval_dir, "oracle_screen_hist.png")
    plot_oracle_histograms(
        scores,
        oracle_names=TDC_ORACLE_NAMES,
        out_path=hist_path,
    )
    report_path = os.path.join(eval_dir, "oracle_screen.json")
    write_screen_report(
        report_path,
        meta={
            "n_sobol": n_sobol or len(sobol),
            "n_train": len(train),
            "n_test": len(test),
            "oracles": list(TDC_ORACLE_NAMES),
        },
        splits=summaries,
        scores=scores,
    )
    from boreft.oracle_screen import format_screen_table

    print(format_screen_table(summaries, TDC_ORACLE_NAMES), flush=True)
    print(f"[ORACLE_SCREEN] wrote {report_path}", flush=True)
    return {
        "splits": summaries,
        "wandb": screen_metrics_for_wandb(summaries),
        "hist_path": hist_path if os.path.isfile(hist_path) else None,
        "report_path": report_path,
    }


def run_recon_similarity_plot(gen_results: dict, eval_dir: str) -> str | None:
    """Plot RECON structural vs. semantic similarity (molecule runs only)."""
    per_target = ((gen_results or {}).get("embedding_sim") or {}).get("per_target")
    if not per_target:
        return None
    try:
        from boreft.eval.plot_recon_similarity import plot_recon_similarity

        return plot_recon_similarity(
            per_target,
            os.path.join(eval_dir, "recon_similarity.png"),
            embedding_model=str(gen_results.get("embedding_model", "")),
        )
    except Exception as e:
        print(f"[full_eval] WARNING: RECON similarity plot failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Cluster PCA plot  (plot_cluster_pca.py)
# ─────────────────────────────────────────────────────────────────────────────


def run_pca_plot(
    cfg: "FullEvalConfig",
    eval_dir: str,
    *,
    checkpoint: EvalCheckpoint,
) -> str | None:
    """Produce a PCA scatter of per-word bias vectors coloured by cluster or category."""
    if not _geometry_eval_enabled(cfg.output_dir, cfg.semantle_dir):
        print(
            "[full_eval] --semantle_dir not set and use_definition_embeds is off, "
            "skipping PCA plot."
        )
        return None
    save_path = os.path.join(eval_dir, "cluster_pca.png")
    try:
        plot_cluster_pca(
            output_dir=cfg.output_dir,
            model_name=cfg.model_name,
            cache_dir=cfg.cache_dir,
            layer=cfg.layer,
            low_rank_dim=cfg.low_rank_dim,
            semantle_dir=cfg.semantle_dir,
            top_k=cfg.top_k,
            save_path=save_path,
            reft_model=checkpoint.reft_model,
            words=checkpoint.words,
            items=checkpoint.items,
        )
        return save_path
    except Exception as e:
        print(f"[full_eval] WARNING: PCA plot failed: {e}")
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — LIPZ bias interpolation  (interpolate.py)
# ─────────────────────────────────────────────────────────────────────────────


def interp_pair_key(target1: str, target2: str) -> str:
    """Filesystem- and W&B-safe key for one interpolation pair.

    Targets may contain path/namespace separators (SMILES have ``/``, ``\\``,
    ``#``), so the key is sanitized. Sanitizing is lossy, so a hash of the raw
    pair is appended to keep distinct pairs in distinct directories and W&B keys.
    The raw targets are preserved in the results JSON.
    """
    raw = f"{target1}__{target2}"
    safe = safe_filename(raw)
    if safe == raw:
        return safe
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{safe}-{digest}"


def resolve_interp_pairs(
    raw: Optional[List[str]],
    words: List[str],
    seed: int,
    *,
    task: str = "semantle",
) -> tuple[list[tuple[str, str]], dict]:
    """Parse ``--full_eval_interp`` into ``(word1, word2)`` pairs.

    - A single positive integer *N*: sample *N* random distinct pairs from
      ``words`` (reproducible via ``seed``).
    - Otherwise: treat values as an alternating list of explicit target pairs,
      each of which must be in the trained vocabulary (interpolation needs both
      endpoints' learned bias vectors). Endpoints are matched by the task's
      normalized form and rewritten to the trained spelling, so a re-spelled
      SMILES resolves to the molecule it denotes.
    """
    if not raw:
        return [], {}

    if len(raw) == 1:
        try:
            n_pairs = int(raw[0])
        except ValueError:
            n_pairs = None
        else:
            if n_pairs <= 0:
                print(f"[full_eval] --full_eval_interp {n_pairs}: no pairs to run.")
                return [], {"mode": "random_sample", "n_pairs_requested": n_pairs}
            if len(words) < 2:
                print(
                    "[full_eval] WARNING: need at least 2 eval words to sample interpolation pairs."
                )
                return [], {
                    "mode": "random_sample",
                    "n_pairs_requested": n_pairs,
                    "error": "too_few_words",
                }
            candidates = list(combinations(words, 2))
            n_draw = min(n_pairs, len(candidates))
            if n_draw < n_pairs:
                print(
                    f"[full_eval] WARNING: requested {n_pairs} interpolation pairs but only "
                    f"{len(candidates)} distinct pairs exist among {len(words)} eval words; "
                    f"using {n_draw}.",
                    flush=True,
                )
            rng = random.Random(seed)
            drawn = [tuple(p) for p in rng.sample(candidates, n_draw)]
            meta = {
                "mode": "random_sample",
                "n_pairs_requested": n_pairs,
                "n_pairs_sampled": n_draw,
                "eval_sample_seed": seed,
                "pairs": [{"word1": a, "word2": b} for a, b in drawn],
            }
            print(
                f"[full_eval] Sampled {n_draw} interpolation pair(s) from {len(words)} eval words.",
                flush=True,
            )
            return drawn, meta

    if len(raw) % 2 != 0:
        print(
            f"[full_eval] WARNING: --full_eval_interp has odd number of values "
            f"({len(raw)}); ignoring the last entry."
        )
        raw = raw[:-1]

    norm = target_normalizer(task)
    trained_by_key = {norm(w): w for w in words}
    requested = [(raw[i], raw[i + 1]) for i in range(0, len(raw), 2)]
    unknown = sorted(
        {t for pair in requested for t in pair if norm(t) not in trained_by_key}
    )
    if unknown:
        raise ValueError(
            "--full_eval_interp explicit pairs must name trained targets; "
            f"not in the checkpoint vocabulary: {unknown}. "
            "Pass an integer instead to sample N random pairs."
        )
    pairs = [
        (trained_by_key[norm(a)], trained_by_key[norm(b)]) for a, b in requested
    ]
    return pairs, {
        "mode": "explicit",
        "pairs": [{"word1": a, "word2": b} for a, b in pairs],
    }


def run_interpolation_eval(
    cfg: "FullEvalConfig",
    eval_dir: str,
    *,
    checkpoint: EvalCheckpoint,
) -> tuple[dict, dict, list[str], dict]:
    """Run one interpolation per pair resolved from ``--full_eval_interp``."""
    if not cfg.full_eval_interp:
        print("[full_eval] --full_eval_interp not set, skipping interpolation.")
        return {}, {}, [], {}

    pairs, pair_meta = resolve_interp_pairs(
        cfg.full_eval_interp,
        checkpoint.words,
        cfg.seed,
        task=str((checkpoint.saved_cfg or {}).get("task", "semantle")),
    )
    if not pairs:
        return {}, {}, [], pair_meta

    all_results: dict = {}
    all_plots: dict = {}
    failed_pairs: list[str] = []

    for word1, word2 in pairs:
        pair_key = interp_pair_key(word1, word2)
        print(f"\n[full_eval] Interpolation: {word1!r} → {word2!r}")
        save_dir = os.path.join(eval_dir, "interpolation", pair_key)
        try:
            results, plots = run_interpolation(
                output_dir=cfg.output_dir,
                word1=word1,
                word2=word2,
                model_name=cfg.model_name,
                layer=cfg.layer,
                low_rank_dim=cfg.low_rank_dim,
                n_samples=cfg.interp_n_samples,
                top_p=cfg.top_p if cfg.top_p is not None else 1.0,
                t_steps=cfg.interp_t_steps,
                interp_method=cfg.interp_method,
                cache_dir=cfg.cache_dir,
                save_dir=save_dir,
                position=cfg.position,
                max_new_tokens=cfg.max_new_tokens,
                train_vocab=set(checkpoint.full_words or checkpoint.words),
                reft_model=checkpoint.reft_model,
                tokenizer=checkpoint.tokenizer,
                prompt=checkpoint.prompt,
                items=checkpoint.items,
                assistant_suffix=checkpoint.assistant_suffix,
                from_chat_template=checkpoint.from_chat_template,
            )
            all_results[pair_key] = {
                "word1": results["word1"]["target"],
                "word2": results["word2"]["target"],
                "interp_method": results.get("interp_method", cfg.interp_method),
                "endpoints": results["endpoints"],
                # Per-pair LIPZ summary:
                # {variant: {mean_step,max_step,emp_l,emp_l_p95,peakiness,detour}}
                "trajectory_analysis": results.get("trajectory_analysis", {}),
                "plots_saved": list(plots.keys()),
            }
            all_plots[pair_key] = plots
        except Exception as e:
            print(f"[full_eval] WARNING: interpolation {word1!r}→{word2!r} failed: {e}")
            traceback.print_exc()
            failed_pairs.append(pair_key)

    return all_results, all_plots, failed_pairs, pair_meta


def _save_eval_artifacts(eval_dir: str, all_results: dict) -> None:
    """Persist combined results and a small status file (partial saves OK)."""
    out_path = os.path.join(eval_dir, "full_eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[full_eval] Results saved → {out_path}")

    meta = all_results.get("meta", {})
    status_path = os.path.join(eval_dir, "eval_status.json")
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[full_eval] Status saved → {status_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def _section(title: str, step: int, total: int = 4):
    print(f"\n{'=' * 60}")
    print(f"[full_eval] Step {step}/{total} — {title}")
    print(f"{'=' * 60}")


# FullEvalConfig field -> key in load_merged_run_config()
_FULL_EVAL_SAVED_KEYS: dict[str, str] = {
    "model_name": "model_name",
    "cache_dir": "cache_dir",
    "layer": "layer",
    "low_rank_dim": "low_rank_dim",
    "semantle_dir": "semantle_dir",
    "top_k": "train_top_k",
    "n_samples": "full_eval_gen_samples",
    "full_eval_n_samples": "full_eval_n_samples",
    "eval_batch_size": "full_eval_batch_size",
    "n_uniform": "full_eval_n_uniform",
    "top_p": "full_eval_top_p",
    "seed": "seed",
    "max_new_tokens": "full_eval_max_new_tokens",
    "embed_sim_tau": "eval_embed_sim_tau",
    "test_n_samples": "test_n_samples",
    "bbox_pca_var": "eval_bbox_pca_var",
    "oracle_screen": "full_eval_oracle_screen",
    "oracle_screen_n_sobol": "full_eval_oracle_screen_n_sobol",
    "full_eval_interp": "full_eval_interp",
    "interp_n_samples": "full_eval_interp_n_samples",
    "interp_t_steps": "full_eval_interp_t_steps",
    "interp_method": "interp_method",
    "wandb_project": "wandb_project",
    "wandb_entity": "wandb_entity",
    "wandb_run_name": "wandb_run_name",
    "wandb_group": "wandb_group",
    "wandb_dir": "wandb_dir",
}

# Used when a field is unset on the CLI and missing from the checkpoint config.
_FULL_EVAL_FALLBACKS: dict[str, object] = {
    "model_name": "meta-llama/Llama-3.2-1B",
    "layer": 13,
    "low_rank_dim": 64,
    "top_k": 10,
    "n_samples": 25,
    "eval_batch_size": 32,
    "n_uniform": 200,
    "top_p": 1.0,
    "seed": 42,
    "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
    "interp_n_samples": 10,
    "interp_t_steps": 101,
    "interp_method": "lerp",
    "position": "l1",
    "embed_sim_tau": DEFAULT_EMBED_SIM_TAU,
    "test_n_samples": DEFAULT_TEST_N_SAMPLES,
    "bbox_pca_var": DEFAULT_BBOX_PCA_VAR,
    "oracle_screen": True,
    "oracle_screen_n_sobol": 2048,
}


def _resolve_full_eval_config(cfg: FullEvalConfig) -> FullEvalConfig:
    """Fill unset CLI fields from checkpoint config (training_config.json)."""
    from boreft.intervention_marker import validate_intervention_position

    saved = load_merged_run_config(cfg.output_dir)
    updates: dict[str, object] = {}
    autofilled: list[str] = []

    for field, saved_key in _FULL_EVAL_SAVED_KEYS.items():
        if getattr(cfg, field) is not None:
            continue
        if saved_key in saved and saved[saved_key] is not None:
            value = saved[saved_key]
            updates[field] = value
            autofilled.append(f"{field}={value!r}")
        elif field in _FULL_EVAL_FALLBACKS:
            updates[field] = _FULL_EVAL_FALLBACKS[field]

    position = cfg.position
    if position is None:
        position = saved.get("position", _FULL_EVAL_FALLBACKS["position"])
        autofilled.append(f"position={position!r}")
    updates["position"] = validate_intervention_position(position)

    if cfg.torch_dtype is None:
        override = saved.get("torch_dtype_override")
        if override in TORCH_DTYPE_NAMES:
            updates["torch_dtype"] = override
            autofilled.append(f"torch_dtype={override!r}")
        elif saved.get("torch_dtype") in TORCH_DTYPE_NAMES:
            dtype_name = saved["torch_dtype"]
            updates["torch_dtype"] = dtype_name
            autofilled.append(f"torch_dtype={dtype_name!r}")

    resolved = replace(cfg, **updates)
    if resolved.interp_method is not None:
        resolved = replace(
            resolved,
            interp_method=normalize_interp_method(resolved.interp_method),
        )
    print(
        f"[full_eval] model={resolved.model_name} layer={resolved.layer} "
        f"rank={resolved.low_rank_dim} position={resolved.position}",
        flush=True,
    )
    if autofilled:
        print(
            f"[full_eval] Auto-filled from checkpoint: {', '.join(autofilled)}",
            flush=True,
        )
    return resolved


@dataclass
class FullEvalConfig:
    """Typed config for the post-training eval pipeline.

    Field names mirror the CLI flags of ``main()`` so the same object backs both
    ``python -m boreft.eval.run_full_eval`` and the in-process call from ``boreft.train``.
    """

    output_dir: str
    model_name: Optional[str] = None
    cache_dir: Optional[str] = None
    layer: Optional[int] = None
    low_rank_dim: Optional[int] = None

    # cluster plots (Steps 3 & 4)
    semantle_dir: Optional[str] = None
    top_k: Optional[int] = None

    # generation eval (Step 1 / semantle.py)
    n_samples: Optional[int] = None
    full_eval_n_samples: Optional[int] = None
    eval_batch_size: Optional[int] = None
    n_uniform: Optional[int] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None
    position: Optional[str] = None
    torch_dtype: Optional[str] = None
    max_new_tokens: Optional[int] = None

    # RECON / GENZ knobs
    embed_sim_tau: Optional[float] = None
    test_n_samples: Optional[int] = None
    bbox_pca_var: Optional[float] = None
    oracle_screen: Optional[bool] = None
    oracle_screen_n_sobol: Optional[int] = None

    # interpolation / LIPZ (Step 3)
    full_eval_interp: Optional[List[str]] = None
    interp_n_samples: Optional[int] = None
    interp_t_steps: Optional[int] = None
    interp_method: Optional[str] = None

    # WandB
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_dir: Optional[str] = None
    no_wandb: bool = False
    load_latest: bool = False


def run_full_eval(cfg: FullEvalConfig) -> dict:
    """Run the full post-training eval pipeline in-process and return the results dict.

    Loads the checkpoint once and shares it across all steps. When
    ``full_eval_n_samples`` is set, the same word subset is used for every step.
    Always writes ``eval/full_eval_results.json`` and ``eval/eval_status.json``
    (even on failure).
    """
    eval_dir = os.path.join(cfg.output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    cfg = _resolve_full_eval_config(cfg)

    eval_log_path = os.path.join(eval_dir, "eval_output.log")
    _tee = _Tee(sys.stdout, eval_log_path)
    sys.stdout = _tee

    all_results: dict = {"output_dir": cfg.output_dir}
    failed_steps: list[str] = []
    pipeline_error: str | None = None
    checkpoint: EvalCheckpoint | None = None

    try:
        _init_wandb(
            cfg.output_dir,
            cfg.no_wandb,
            cfg.wandb_project,
            cfg.wandb_entity,
            cfg.wandb_run_name,
            cfg.wandb_group,
            cfg.wandb_dir,
        )
        wb_on = wandb.run is not None

        print(
            f"[full_eval] Loading checkpoint once for all steps: {cfg.output_dir}",
            flush=True,
        )
        checkpoint = load_eval_checkpoint(
            cfg.output_dir,
            cfg.model_name,
            cfg.layer,
            cfg.low_rank_dim,
            cfg.cache_dir,
            torch_dtype=cfg.torch_dtype,
            load_latest=bool(cfg.load_latest),
        )
        checkpoint = subset_eval_checkpoint(
            checkpoint, cfg.full_eval_n_samples, cfg.seed
        )
        all_results["eval_word_subset"] = {
            "full_eval_n_samples": cfg.full_eval_n_samples,
            "eval_sample_seed": cfg.seed,
            "n_eval_words": len(checkpoint.words),
            "n_training_words": len(checkpoint.full_words or checkpoint.words),
            "eval_words": list(checkpoint.words),
        }

        # ── Step 1: RECON / GENZ / DIST ───────────────────────────────────────
        _section("RECON / GENZ / DIST (semantle.py)", 1)
        gen_results = run_generation_eval(cfg, checkpoint=checkpoint)
        if gen_results is not None:
            all_results["generation"] = gen_results
            if wb_on:
                _wandb_log(_suite_metrics_for_wandb(gen_results))
                sobol_json = os.path.join(eval_dir, "sobol_results.json")
                if os.path.exists(sobol_json):
                    wandb.save(sobol_json, base_path=eval_dir, policy="now")
            recon_sim_path = run_recon_similarity_plot(gen_results, eval_dir)
            if recon_sim_path:
                print(f"  saved: {recon_sim_path}")
                if wb_on:
                    _wandb_log({"plots/recon_similarity": wandb.Image(recon_sim_path)})
        else:
            failed_steps.append("generation")

        # ── Step 2: molopt oracle screen ──────────────────────────────────────
        _section("Oracle screen (train / high-tail / Sobol)", 2)
        try:
            screen = run_oracle_screen_panel(
                cfg, eval_dir, checkpoint=checkpoint, gen_results=gen_results
            )
        except Exception as e:
            print(f"[full_eval] WARNING: oracle screen failed: {e}", flush=True)
            traceback.print_exc()
            screen = None
            failed_steps.append("oracle_screen")
        if screen is not None:
            all_results["oracle_screen"] = screen.get("splits")
            if wb_on:
                payload = dict(screen.get("wandb") or {})
                if screen.get("hist_path"):
                    payload["plots/oracle_screen"] = wandb.Image(screen["hist_path"])
                if payload:
                    _wandb_log(payload)
                report_path = screen.get("report_path")
                if report_path and os.path.exists(report_path):
                    wandb.save(report_path, base_path=eval_dir, policy="now")

        # ── Step 3: cluster PCA plot ──────────────────────────────────────────
        _section("Cluster PCA plot", 3)
        if _geometry_eval_enabled(cfg.output_dir, cfg.semantle_dir):
            pca_path = run_pca_plot(cfg, eval_dir, checkpoint=checkpoint)
            if pca_path:
                print(f"  saved: {pca_path}")
                if wb_on:
                    _wandb_log({"plots/pca": wandb.Image(pca_path)})
            else:
                failed_steps.append("pca")
        else:
            print(
                "[full_eval] --semantle_dir not set and use_definition_embeds is off, "
                "skipping PCA plot."
            )

        # ── Step 3: LIPZ bias interpolation ───────────────────────────────────
        if cfg.full_eval_interp:
            _section("LIPZ bias interpolation (interpolate.py)", 4)
            interp_results, interp_plots, failed_pairs, interp_pair_meta = (
                run_interpolation_eval(
                    cfg,
                    eval_dir,
                    checkpoint=checkpoint,
                )
            )
            if interp_pair_meta:
                all_results["interpolation_pairs"] = interp_pair_meta
            if interp_results:
                all_results["interpolation"] = interp_results
                pair_summaries = [
                    s["trajectory_analysis"]
                    for s in interp_results.values()
                    if s.get("trajectory_analysis")
                ]
                lipz = lipz_aggregate(pair_summaries)
                all_results["lipz"] = lipz
                if wb_on:
                    wb_interp: dict = {f"lipz/{k}": v for k, v in lipz.items()}
                    for pair_key, plots in interp_plots.items():
                        for plot_name, path in plots.items():
                            wb_interp[f"plots/interp/{pair_key}/{plot_name}"] = (
                                wandb.Image(path)
                            )
                    if wb_interp:
                        _wandb_log(wb_interp)
            if failed_pairs:
                failed_steps.append("interpolation")
                all_results["interpolation_failed_pairs"] = failed_pairs
        else:
            print("[full_eval] --full_eval_interp not set, skipping LIPZ interpolation.")

    except Exception as e:
        pipeline_error = str(e)
        print(f"[full_eval] ERROR: pipeline failed: {e}", flush=True)
        traceback.print_exc()
    finally:
        success = len(failed_steps) == 0 and pipeline_error is None
        all_results["meta"] = {
            "success": success,
            "failed_steps": failed_steps,
            "pipeline_error": pipeline_error,
        }
        try:
            _save_eval_artifacts(eval_dir, all_results)
        except Exception as save_err:
            print(
                f"[full_eval] WARNING: could not save results: {save_err}",
                file=sys.stderr,
            )

        release_eval_checkpoint(checkpoint)
        _tee.stop()
        print(f"[full_eval] Eval log saved → {eval_log_path}")
        try:
            if wandb.run is not None:
                wandb.save(eval_log_path, base_path=eval_dir, policy="now")
                wandb.finish()
        except Exception as e:
            print(f"[full_eval] WARNING: WandB finish failed: {e}", file=sys.stderr)

    if success:
        print("[full_eval] Done.")
    else:
        print(f"[full_eval] Done with failures: {failed_steps}", file=sys.stderr)
        if pipeline_error:
            print(f"[full_eval] Pipeline error: {pipeline_error}", file=sys.stderr)

    return all_results


def main():
    p = argparse.ArgumentParser(
        description=(
            "Consolidated post-training eval for BOReFT checkpoints "
            "(task read from the checkpoint config)"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- model / checkpoint ---
    p.add_argument(
        "--output_dir",
        required=True,
        help="Trained model directory (intervenable_model/, items.json, …)",
    )
    p.add_argument(
        "--model-name",
        dest="model_name",
        default=None,
        help="Base HF model id (default: training_config.json model_name).",
    )
    p.add_argument("--cache_dir", default=None)
    p.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Intervention layer (default: training_config.json layer).",
    )
    p.add_argument(
        "--low_rank_dim",
        type=int,
        default=None,
        help="LoReFT rank (default: training_config.json low_rank_dim).",
    )

    # --- cluster plots (Steps 3 & 4) ---
    p.add_argument(
        "--semantle_dir",
        default=None,
        help="Directory of semantle CSV files; required for PCA and silhouette.",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k per CSV matching training (default: training_config.json train_top_k).",
    )

    # --- generation eval (Step 1 / semantle.py) ---
    p.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Stochastic generations per word (default: training_config.json full_eval_gen_samples).",
    )
    p.add_argument(
        "--full_eval_n_samples",
        type=int,
        default=None,
        help="Evaluate a random subset of N training words on every step "
        "(default: all).",
    )
    p.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="GPU batch size for generation (default: training_config.json full_eval_batch_size).",
    )
    p.add_argument(
        "--top-p",
        dest="top_p",
        type=float,
        default=None,
        help="Nucleus top-p for all temperature-sampling passes "
        "(default: training_config.json full_eval_top_p, else 1.0).",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Max tokens to generate per decode (default: training_config.json full_eval_max_new_tokens).",
    )
    p.add_argument(
        "--n_uniform",
        type=int,
        default=None,
        help="Number of Sobol samples for Case 5 (default: training_config.json full_eval_n_uniform).",
    )
    p.add_argument(
        "--oracle_screen_n_sobol",
        type=int,
        default=None,
        help="Greedy Sobol codes for the molopt oracle screen "
        "(default: training_config.json full_eval_oracle_screen_n_sobol, else 2048).",
    )
    p.add_argument(
        "--no_oracle_screen",
        action="store_true",
        help="Skip the molopt DRD2/GSK3B/JNK3 Sobol screen.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (default: training_config.json seed).",
    )
    p.add_argument(
        "--position",
        default=None,
        help="Intervention position (e.g. 'l1', 'f1'). Defaults to value in "
        "intervention_config.json, or 'l1' if not found.",
    )
    p.add_argument(
        "--torch-dtype",
        dest="torch_dtype",
        choices=TORCH_DTYPE_NAMES,
        default=None,
        help="Override model/intervention dtype (default: checkpoint config, else auto).",
    )

    # --- RECON / GENZ knobs ---
    p.add_argument(
        "--embed-sim-tau",
        dest="embed_sim_tau",
        type=float,
        default=None,
        help="RECON threshold for recon/embed_sim_gte_tau "
        "(default: training_config.json eval_embed_sim_tau, else 0.8).",
    )
    p.add_argument(
        "--test-n-samples",
        dest="test_n_samples",
        type=int,
        default=None,
        help="GENZ: number of non-train test words to sample "
        "(default: training_config.json test_n_samples, else 512).",
    )
    p.add_argument(
        "--bbox-pca-var",
        dest="bbox_pca_var",
        type=float,
        default=None,
        help="GENZ: PCA variance ratio for the train-embedding bounding box "
        "(default: training_config.json eval_bbox_pca_var, else 0.9).",
    )

    # --- interpolation / LIPZ (Step 3) ---
    p.add_argument(
        "--full_eval_interp",
        nargs="+",
        default=None,
        metavar="WORD",
        help="Optional interpolation eval: either a single integer N to sample N "
        "random word pairs from the eval vocabulary, or explicit alternating "
        "pairs (e.g. --full_eval_interp polyethylene birthstone computer meatloaf).",
    )
    p.add_argument(
        "--interp_n_samples",
        type=int,
        default=None,
        help="Generations per interpolation t step (default: training_config.json).",
    )
    p.add_argument(
        "--interp_t_steps",
        type=int,
        default=None,
        help="Number of interpolation steps (default: training_config.json).",
    )
    p.add_argument(
        "--interp_method",
        default=None,
        choices=["lerp", "slerp"],
        help="Bias interpolation method (default: training_config.json or lerp).",
    )

    # --- WandB (same flags as train.py) ---
    p.add_argument(
        "--wandb_project",
        default=None,
        help="WandB project (required for logging, like train.py).",
    )
    p.add_argument(
        "--wandb_entity",
        default=None,
        help="WandB team/user (wandb.ai/<entity>/<project>). Same as train.py --wandb_entity.",
    )
    p.add_argument(
        "--wandb_run_name",
        default=None,
        help="Run display name for a new WandB run (optional).",
    )
    p.add_argument(
        "--wandb_group",
        default=None,
        help="WandB run group for related runs (optional). Same as train.py --wandb_group.",
    )
    p.add_argument(
        "--wandb_dir", default=None, help="Directory for WandB files (optional)."
    )
    p.add_argument(
        "--no_wandb",
        action="store_true",
        help="Skip WandB; only write result files locally.",
    )
    add_load_latest_argument(p)
    args = p.parse_args()
    if getattr(args, "no_oracle_screen", False):
        args.oracle_screen = False
    if args.position is not None:
        from boreft.intervention_marker import validate_intervention_position

        validate_intervention_position(args.position)

    # Filter to known FullEvalConfig fields so adding a CLI-only arg can't break construction.
    valid = {f.name for f in fields(FullEvalConfig)}
    results = run_full_eval(
        FullEvalConfig(**{k: v for k, v in vars(args).items() if k in valid})
    )
    if not results.get("meta", {}).get("success", False):
        sys.exit(1)


if __name__ == "__main__":
    main()
