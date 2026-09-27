"""Extend a high-train-recall LoReFT subspace by absorbing test recall@n misses.

Experiment
----------
Start from a trained ``--add-bias-network`` checkpoint that typically has **high
train recall** and **middling test recall**. Then:

  1. Detect **test-set** recall@n misses at ``--miss-temp`` (train-miss counts are
     reported; include them in the absorption set only with
     ``--also-absorb-train-misses``). Cap with ``--n-unlearnt-targets``. Or pass
     ``--new-targets`` JSONL ``{word, definition}`` to provide the absorption
     set directly (source populations are still decoded for baseline metrics).
  2. Continue-train on those targets under a recipe controlled by
     ``--learn-w`` / ``--learn-r`` / ``--learn-bias-network`` /
     ``--include-prev-targets`` (biases warm-started from the network prediction).
     With ``--include-prev-targets``, ``--previous-n`` / ``--previous-prop``
     optionally sample that many (or that fraction of) prior train targets into
     the loss (default: all), via ``--previous-strategy``
     ``random`` / ``herding`` / ``k-center-greedy`` (geometry on source bias means).
     ``--previous-include`` names targets that must occupy part of that sample
     (for example the search warm starts). They count toward ``--previous-n``
     and are never dropped to meet it.
  3. Answer three questions and write a small schema-v2 summary:

       * **original_train** — how many previously trained points are no longer recalled?
       * **original_test_misses** — how many absorbed (former miss) targets are now recalled?
       * **original_test_hits** — of the original test points that were hits (test
         minus absorbed misses), how many are still misses after extend?

Bulky lists and curves go to sidecars (``new_targets.jsonl``, ``eval_history.json``,
``learn_config.json``). Training eval curves and before/after charts log to a
fresh WandB run.

Example::

    python -m boreft.extend_subspace \\
        --result-dir outputs/my_run --n-unlearnt-targets 32 --epochs 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from boreft.bias_tables import (
    bias_predict_kwargs,
    detach_materialized_bias_tables,
    load_bias_tables,
    predict_bias_vectors_for_words,
    resolve_bias_tables_path,
    save_bias_tables,
    stack_bias_vectors,
)
from boreft.coreset import (
    SAMPLE_STRATEGIES,
    resolve_sample_count,
    select_subset_indices,
)
from boreft.data_utils import (
    add_load_latest_argument,
    INTERVENTION_CONFIG_NAME,
    TRAINING_CONFIG_NAME,
    load_intervention_config,
    load_merged_run_config,
    load_training_config,
)
from boreft.eval.decode_utils import (
    decode_targets,
    greedy_hits,
    load_test_set,
    predict_test_bias_vectors,
    recall_hits_at,
)
from boreft.eval.eval_suite import (
    DEFAULT_BBOX_PCA_VAR,
    DEFAULT_EMBED_SIM_TAU,
    EVAL_TEMPERATURES,
    recon_metrics,
    recon_test_metrics,
    split_interp_extrap,
    target_normalizer,
    temp_key,
)
from boreft.eval.semantle import (
    DEFAULT_MAX_NEW_TOKENS,
    _get_intervention,
    load_eval_checkpoint,
    release_eval_checkpoint,
)
from boreft.learn_bias import (
    BatchLearnConfig,
    learn_biases_batched,
    predict_bias_network_rows,
)
from boreft.task_config import definition_embedding_text, task_supports_sdpo
from boreft.text_similarity import (
    default_definitions_path,
    definition_lookup_for_cfg,
    definition_text_for_cfg,
    embedding_sim_per_text,
    encode_reference_embeddings,
    encode_texts_normalized,
    load_raw_definitions,
    rdkit_definition_lookup_for_cfg,
    rdkit_map_path_for_cfg,
)


# ─────────────────────────────────────────────────────────────────────────────
# Repo layout helpers
# ─────────────────────────────────────────────────────────────────────────────


def _repo_root() -> str:
    """Best-effort repository root (parent of ``src/``)."""
    here = os.path.abspath(__file__)
    # .../src/boreft/extend_subspace.py -> repo root is three levels up.
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))


OUTPUT_SUBDIR = os.path.join(_repo_root(), "experiments", "outputs")


# ─────────────────────────────────────────────────────────────────────────────
# Pure logic (unit-testable without a GPU or a real checkpoint)
# ─────────────────────────────────────────────────────────────────────────────


def select_unlearnt_targets(
    train_misses: Sequence[str],
    test_misses: Sequence[str],
    n_unlearnt: Optional[int],
    *,
    task: str = "semantle",
) -> List[str]:
    """Concatenate train + test misses, de-duplicate, and cap to ``n_unlearnt``.

    Order is train-misses first (decoded via ``word_ids``) then test-misses
    (decoded via predicted bias), de-duplicated by ``task``-normalized form while
    preserving first occurrence. ``n_unlearnt=None`` keeps them all.
    """
    norm = target_normalizer(task)
    seen: set[str] = set()
    ordered: List[str] = []
    for word in list(train_misses) + list(test_misses):
        key = norm(word)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(word)
    if n_unlearnt is not None:
        if n_unlearnt < 0:
            raise ValueError("n_unlearnt_targets must be non-negative")
        ordered = ordered[:n_unlearnt]
    return ordered


def build_combined_items(
    old_items: Sequence[Dict[str, Any]],
    new_words: Sequence[str],
    *,
    prompt: str,
    new_targets_text: Optional[Dict[str, str]] = None,
    new_original_splits: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    """Build the combined, reindexed ``items.json`` list with durable provenance.

    ``original_split`` survives repeated extensions, while ``origin`` describes
    only the current extension step. Every materialized item is ``seen`` because
    its target has participated in subspace training at least once.
    """
    combined: List[Dict[str, Any]] = []
    old_indices: List[int] = []
    for new_id, item in enumerate(old_items):
        word = item.get("word", item.get("target", ""))
        original_split = item.get("original_split")
        if original_split not in {"train", "test", "additional"}:
            original_split = (
                "additional" if item.get("origin") == "new" else "train"
            )
        combined.append({
            **item,
            "id": new_id,
            "prompt": item.get("prompt", prompt),
            "target": item.get("target", word),
            "word": word,
            "origin": "old",
            "original_split": original_split,
            "seen": True,
        })
        old_indices.append(new_id)

    new_indices: List[int] = []
    offset = len(old_items)
    for k, word in enumerate(new_words):
        new_id = offset + k
        target_text = (new_targets_text or {}).get(word, word)
        original_split = (new_original_splits or {}).get(word, "additional")
        if original_split not in {"train", "test", "additional"}:
            raise ValueError(
                f"invalid original split {original_split!r} for {word!r}"
            )
        combined.append(
            {
                "id": new_id,
                "prompt": prompt,
                "target": target_text,
                "word": word,
                "origin": "new",
                "original_split": original_split,
                "seen": True,
            }
        )
        new_indices.append(new_id)
    return combined, old_indices, new_indices


# ─────────────────────────────────────────────────────────────────────────────
# Summary metric helpers (unit-testable without a GPU)
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_VERSION = 2

EXTEND_QUESTION_GOAL = (
    "Absorb original test recall@n misses into a high-train-recall subspace and "
    "measure original_train forgetting vs original_test_misses absorption vs "
    "original_test_hits retention."
)


def metric_cell(
    *,
    recall_at_n: Optional[float] = None,
    recall_greedy: Optional[float] = None,
    embed_sim: Optional[float] = None,
    embed_sim_gte_tau: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """Fixed-key metric cell used in the schema-v2 summary."""
    return {
        "recall_at_n": (
            float(recall_at_n) if recall_at_n is not None else None
        ),
        "recall_greedy": (
            float(recall_greedy) if recall_greedy is not None else None
        ),
        "embed_sim": float(embed_sim) if embed_sim is not None else None,
        "embed_sim_gte_tau": (
            float(embed_sim_gte_tau) if embed_sim_gte_tau is not None else None
        ),
    }


def hit_rate(hits: Dict[str, bool], words: Optional[Sequence[str]] = None) -> float:
    if words is not None:
        values = [bool(hits[w]) for w in words if w in hits]
    else:
        values = [bool(v) for v in hits.values()]
    return float(np.mean(values)) if values else 0.0


def cell_from_panel(panel: Dict[str, Any], miss_temp: float) -> Dict[str, Optional[float]]:
    """Map a flat ``recon_metrics`` / subset panel onto a metric cell."""
    key = f"recall_at_n_{temp_key(miss_temp)}"
    return metric_cell(
        recall_at_n=panel.get(key),
        recall_greedy=panel.get("recall_greedy"),
        embed_sim=panel.get("embed_sim"),
        embed_sim_gte_tau=panel.get("embed_sim_gte_tau"),
    )


def cell_from_hits(
    *,
    recall_hits: Optional[Dict[str, bool]] = None,
    greedy_hit_map: Optional[Dict[str, bool]] = None,
    words: Optional[Sequence[str]] = None,
    embed_sim: Optional[float] = None,
    embed_sim_gte_tau: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    return metric_cell(
        recall_at_n=(
            hit_rate(recall_hits, words) if recall_hits is not None else None
        ),
        recall_greedy=(
            hit_rate(greedy_hit_map, words) if greedy_hit_map is not None else None
        ),
        embed_sim=embed_sim,
        embed_sim_gte_tau=embed_sim_gte_tau,
    )


def hit_maps_from_slice(
    words: Sequence[str],
    greedy: Sequence[str],
    temp_results: Dict[float, List[Dict[str, Any]]],
    indices: Sequence[int],
    miss_temp: float,
) -> Tuple[Dict[str, bool], Dict[str, bool]]:
    """Build per-word recall/greedy maps for one origin slice.

    Slicing before constructing dictionaries keeps duplicate words in different
    origin populations from overwriting one another.
    """
    slice_words = [words[i] for i in indices]
    slice_greedy = [greedy[i] for i in indices]
    slice_temp = {
        temp: [temp_results[temp][i] for i in indices] for temp in temp_results
    }
    return (
        recall_hits_at(slice_words, slice_temp, miss_temp),
        greedy_hits(slice_words, slice_greedy),
    )


def absorption_before_hit_maps(
    words: Sequence[str],
    *,
    train_recall: Dict[str, bool],
    train_greedy: Dict[str, bool],
    test_recall: Dict[str, bool],
    test_greedy: Dict[str, bool],
    prefer_train_words: Optional[Sequence[str]] = None,
    task: str = "semantle",
) -> Tuple[Dict[str, bool], Dict[str, bool]]:
    """Reuse detection-time hit maps for absorbed targets (no re-decode).

    Words listed in ``prefer_train_words`` (e.g. absorbed train misses) take the
    train-side maps; everything else prefers the test-side maps, then train.
    """
    norm = target_normalizer(task)
    prefer_train = {norm(w) for w in (prefer_train_words or [])}
    recall: Dict[str, bool] = {}
    greedy: Dict[str, bool] = {}
    for word in words:
        key = norm(word)
        use_train = key in prefer_train
        if use_train and word in train_recall:
            recall[word] = bool(train_recall[word])
            greedy[word] = bool(train_greedy.get(word, False))
        elif word in test_recall:
            recall[word] = bool(test_recall[word])
            greedy[word] = bool(test_greedy.get(word, False))
        elif word in train_recall:
            recall[word] = bool(train_recall[word])
            greedy[word] = bool(train_greedy.get(word, False))
    return recall, greedy


def count_forgotten(
    before_hits: Dict[str, bool], after_hits: Dict[str, bool]
) -> int:
    """Words that were recall@n hits before and misses after."""
    return sum(
        1
        for word, was_hit in before_hits.items()
        if was_hit and not after_hits.get(word, True)
    )


def count_hits(hits: Dict[str, bool], words: Optional[Sequence[str]] = None) -> int:
    if words is not None:
        return sum(1 for word in words if hits.get(word, False))
    return sum(1 for hit in hits.values() if hit)


def count_misses(hits: Dict[str, bool], words: Optional[Sequence[str]] = None) -> int:
    if words is not None:
        return sum(1 for word in words if not hits.get(word, False))
    return sum(1 for hit in hits.values() if not hit)


def population_block(
    *,
    n: int,
    before: Dict[str, Optional[float]],
    after: Dict[str, Optional[float]],
    n_forgotten: Optional[int] = None,
    n_recovered: Optional[int] = None,
    n_still_misses: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One of original_train / original_test_misses / original_test_hits result blocks."""
    block: Dict[str, Any] = {
        "n": int(n),
        "before": before,
        "after": after,
    }
    if n_forgotten is not None:
        block["n_forgotten"] = int(n_forgotten)
        block["forget_rate"] = float(n_forgotten / n) if n else 0.0
    if n_recovered is not None:
        block["n_recovered"] = int(n_recovered)
        block["recover_rate"] = float(n_recovered / n) if n else 0.0
    if n_still_misses is not None:
        block["n_still_misses"] = int(n_still_misses)
        block["still_miss_rate"] = float(n_still_misses / n) if n else 0.0
    before_r = before.get("recall_at_n")
    after_r = after.get("recall_at_n")
    if before_r is not None and after_r is not None:
        block["delta_recall_at_n"] = float(after_r - before_r)
    if extra:
        block.update(extra)
    return block


def flatten_result_scalars(results_block: Dict[str, Any], prefix: str) -> Dict[str, float]:
    """Flatten nested result cells for WandB scalar logging."""
    out: Dict[str, float] = {}

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                child = f"{path}/{key}" if path else str(key)
                _walk(value, child)
            return
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)) and np.isfinite(node):
            out[path] = float(node)

    _walk(results_block, prefix)
    return out


def training_curve_logs(eval_history: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-eval payloads aligned to ``train/global_step`` when available."""
    rows: List[Dict[str, Any]] = []
    for entry in eval_history:
        epoch = int(entry.get("epoch") or 0)
        step_raw = entry.get("step")
        if isinstance(step_raw, (int, float)) and np.isfinite(step_raw):
            global_step = int(step_raw)
        else:
            global_step = epoch
        payload: Dict[str, Any] = {
            "_step": global_step,
            "train/global_step": float(global_step),
            "eval/epoch": float(epoch),
        }
        payload.update(
            _wandb_eval_scalars(entry, prefix="eval", include_decodes=False)
        )
        groups = entry.get("groups")
        if isinstance(groups, dict):
            for group_name, group_metrics in groups.items():
                if not isinstance(group_metrics, dict):
                    continue
                payload.update(
                    _wandb_eval_scalars(
                        group_metrics,
                        prefix=f"eval/{group_name}",
                        include_decodes=False,
                    )
                )
        rows.append(payload)
    return rows


def before_after_bar_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rows for a before/after recall@n bar chart over the three populations."""
    rows: List[Dict[str, Any]] = []
    for population in (
        "original_train",
        "original_test_misses",
        "original_test_hits",
    ):
        block = results.get(population) or {}
        before = (block.get("before") or {}).get("recall_at_n")
        after = (block.get("after") or {}).get("recall_at_n")
        if before is not None:
            rows.append(
                {
                    "population": population,
                    "phase": "before",
                    "recall_at_n": float(before),
                }
            )
        if after is not None:
            rows.append(
                {
                    "population": population,
                    "phase": "after",
                    "recall_at_n": float(after),
                }
            )
    return rows


def estimate_count_from_rate_delta(
    n: int, before_rate: Optional[float], after_rate: Optional[float]
) -> int:
    """Fallback when per-word before/after hit maps are unavailable.

    Approximates ``n_forgotten`` as ``round((before - after) * n)`` (clipped ≥ 0).
    """
    if n <= 0 or before_rate is None or after_rate is None:
        return 0
    return max(0, int(round((float(before_rate) - float(after_rate)) * n)))


def cell_from_prefixed_panel(
    panel: Dict[str, Any], prefix: str, miss_temp: float
) -> Dict[str, Optional[float]]:
    """Map ``interp_*`` / ``extrap_*`` keys from ``recon_test_metrics`` onto a cell."""
    key = f"recall_at_n_{temp_key(miss_temp)}"
    return metric_cell(
        recall_at_n=panel.get(f"{prefix}_{key}"),
        recall_greedy=panel.get(f"{prefix}_recall_greedy"),
        embed_sim=panel.get(f"{prefix}_embed_sim"),
        embed_sim_gte_tau=panel.get(f"{prefix}_embed_sim_gte_tau"),
    )


def original_train_result(
    *,
    words: Sequence[str],
    before_recall: Optional[Dict[str, bool]],
    before_greedy: Optional[Dict[str, bool]],
    after_recall: Dict[str, bool],
    before_recall_rate: Optional[float] = None,
    before_greedy_rate: Optional[float] = None,
    after_greedy: Optional[Dict[str, bool]] = None,
    after_embed_sim: Optional[float] = None,
    after_embed_sim_gte_tau: Optional[float] = None,
) -> Dict[str, Any]:
    n = len(words)
    before = metric_cell(
        recall_at_n=(
            hit_rate(before_recall, words)
            if before_recall is not None
            else before_recall_rate
        ),
        recall_greedy=(
            hit_rate(before_greedy, words)
            if before_greedy is not None
            else before_greedy_rate
        ),
    )
    after = cell_from_hits(
        recall_hits=after_recall,
        greedy_hit_map=after_greedy,
        words=words,
        embed_sim=after_embed_sim,
        embed_sim_gte_tau=after_embed_sim_gte_tau,
    )
    if before_recall is not None:
        n_forgotten = count_forgotten(before_recall, after_recall)
    else:
        n_forgotten = estimate_count_from_rate_delta(
            n, before.get("recall_at_n"), after.get("recall_at_n")
        )
    return population_block(
        n=n, before=before, after=after, n_forgotten=n_forgotten
    )


def original_test_misses_result(
    *,
    words: Sequence[str],
    before_recall: Optional[Dict[str, bool]],
    before_greedy: Optional[Dict[str, bool]],
    after_recall: Dict[str, bool],
    after_greedy: Optional[Dict[str, bool]] = None,
    after_embed_sim: Optional[float] = None,
    after_embed_sim_gte_tau: Optional[float] = None,
) -> Dict[str, Any]:
    n = len(words)
    before = cell_from_hits(
        recall_hits=before_recall, greedy_hit_map=before_greedy, words=words
    )
    after = cell_from_hits(
        recall_hits=after_recall,
        greedy_hit_map=after_greedy,
        words=words,
        embed_sim=after_embed_sim,
        embed_sim_gte_tau=after_embed_sim_gte_tau,
    )
    n_recovered = count_hits(after_recall, words)
    return population_block(
        n=n, before=before, after=after, n_recovered=n_recovered
    )


def original_test_hits_result(
    *,
    words: Sequence[str],
    before_recall: Optional[Dict[str, bool]],
    before_greedy: Optional[Dict[str, bool]],
    after_recall: Dict[str, bool],
    after_greedy: Optional[Dict[str, bool]] = None,
    after_embed_sim: Optional[float] = None,
    after_embed_sim_gte_tau: Optional[float] = None,
    interp_after: Optional[Dict[str, Optional[float]]] = None,
    extrap_after: Optional[Dict[str, Optional[float]]] = None,
) -> Dict[str, Any]:
    n = len(words)
    before = cell_from_hits(
        recall_hits=before_recall, greedy_hit_map=before_greedy, words=words
    )
    after = cell_from_hits(
        recall_hits=after_recall,
        greedy_hit_map=after_greedy,
        words=words,
        embed_sim=after_embed_sim,
        embed_sim_gte_tau=after_embed_sim_gte_tau,
    )
    n_still_misses = count_misses(after_recall, words)
    extra: Dict[str, Any] = {}
    if interp_after is not None:
        extra["interp"] = {"after": interp_after}
    if extrap_after is not None:
        extra["extrap"] = {"after": extrap_after}
    return population_block(
        n=n,
        before=before,
        after=after,
        n_still_misses=n_still_misses,
        extra=extra or None,
    )


def build_extend_summary(
    *,
    paths: Dict[str, str],
    question: Dict[str, Any],
    setup: Dict[str, Any],
    training: Dict[str, Any],
    results: Dict[str, Any],
    artifacts: Dict[str, str],
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the schema-v2 primary summary (no bulky lists)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at
        or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "paths": paths,
        "question": {
            "goal": question.get("goal", EXTEND_QUESTION_GOAL),
            **{k: v for k, v in question.items() if k != "goal"},
        },
        "setup": setup,
        "training": training,
        "results": results,
        "artifacts": artifacts,
    }


def headline_console_line(results: Dict[str, Any]) -> str:
    """One-line answer to the three experiment questions."""
    old = results.get("original_train") or {}
    new = results.get("original_test_misses") or {}
    rem = results.get("original_test_hits") or {}
    return (
        f"forgotten={old.get('n_forgotten', 0)}/{old.get('n', 0)} "
        f"recovered={new.get('n_recovered', 0)}/{new.get('n', 0)} "
        f"still_test_misses={rem.get('n_still_misses', 0)}/{rem.get('n', 0)}"
    )


ARTIFACT_NEW_TARGETS = "new_targets.jsonl"
ARTIFACT_EVAL_HISTORY = "eval_history.json"
ARTIFACT_LEARN_CONFIG = "learn_config.json"
ARTIFACT_POST_EVAL = os.path.join("eval", "results.json")


def write_extend_sidecars(
    output_dir: str,
    *,
    new_words: Sequence[str],
    new_targets_defs: Dict[str, str],
    eval_history: Sequence[Dict[str, Any]],
    learn_config: Dict[str, Any],
) -> Dict[str, str]:
    """Write bulky sidecars next to the extended checkpoint; return relative paths."""
    os.makedirs(output_dir, exist_ok=True)
    targets_path = os.path.join(output_dir, ARTIFACT_NEW_TARGETS)
    with open(targets_path, "w", encoding="utf-8") as f:
        for word in new_words:
            row = {"target": word}
            if new_targets_defs.get(word):
                row["definition"] = new_targets_defs[word]
            f.write(json.dumps(row) + "\n")

    history_path = os.path.join(output_dir, ARTIFACT_EVAL_HISTORY)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(list(eval_history), f, indent=2)

    learn_path = os.path.join(output_dir, ARTIFACT_LEARN_CONFIG)
    with open(learn_path, "w", encoding="utf-8") as f:
        json.dump(learn_config, f, indent=2)

    return {
        "new_targets": ARTIFACT_NEW_TARGETS,
        "eval_history": ARTIFACT_EVAL_HISTORY,
        "learn_config": ARTIFACT_LEARN_CONFIG,
        "post_eval": ARTIFACT_POST_EVAL,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Definitions
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_run_definitions_path(saved_cfg: dict) -> Optional[str]:
    for key in ("definitions_path", "sdpo_definitions_path", "bias_encoder_definitions_path"):
        path = saved_cfg.get(key)
        if path and os.path.isfile(str(path)):
            return str(path)
    fallback = default_definitions_path(str(saved_cfg.get("task", "semantle")))
    return fallback if os.path.isfile(fallback) else None


def _load_run_raw_definitions(saved_cfg: dict) -> dict[str, str]:
    path = _resolve_run_definitions_path(saved_cfg)
    if not path:
        return {}
    try:
        return load_raw_definitions(path)
    except (OSError, ValueError, KeyError):
        return {}


def _read_new_targets_file(
    path: str, *, task: str = "semantle"
) -> Tuple[list[str], dict[str, str]]:
    """Parse a JSONL of ``{"target", "definition"}`` pairs (legacy ``word`` ok)."""
    from boreft.text_similarity import definitions_row_target

    norm = target_normalizer(task)
    words: list[str] = []
    defs: dict[str, str] = {}
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            word = definitions_row_target(row)
            if not word:
                continue
            key = norm(word)
            if key in seen:
                continue
            seen.add(key)
            words.append(word)
            definition = row.get("definition")
            if definition is not None:
                defs[word] = str(definition).strip()
    return words, defs


# ─────────────────────────────────────────────────────────────────────────────
# Bias-network prediction with merged (new) definitions
# ─────────────────────────────────────────────────────────────────────────────


def _predict_bias_for_words(
    ckpt,
    words: Sequence[str],
    *,
    raw_defs: dict[str, str],
    batch_size: int,
) -> np.ndarray:
    """Predict bias vectors for arbitrary words, merging in any extra defs."""
    saved_cfg = ckpt.saved_cfg or {}
    task = saved_cfg.get("task", "semantle")

    def_embed_lookup = None
    if saved_cfg.get("use_definition_embeds"):
        base = definition_lookup_for_cfg(saved_cfg) or {}
        def_embed_lookup = dict(base)
        rdkit_lookup = rdkit_definition_lookup_for_cfg(saved_cfg)
        for w in words:
            d = raw_defs.get(w)
            if d:
                definition = definition_text_for_cfg(
                    saved_cfg, w, d, rdkit_lookup=rdkit_lookup
                )
                def_embed_lookup[w] = definition_embedding_text(
                    task, w, definition
                )

    merged_raw = None
    if saved_cfg.get("bias_input_source") == "llm_encoder":
        merged_raw = {**_load_run_raw_definitions(saved_cfg), **raw_defs}
    enc_kwargs = bias_predict_kwargs(
        saved_cfg, tokenizer=ckpt.tokenizer, raw_definition_lookup=merged_raw
    )

    vecs = predict_bias_vectors_for_words(
        ckpt.reft_model,
        list(words),
        definition_lookup=def_embed_lookup,
        batch_size=batch_size,
        task=task,
        **enc_kwargs,
    )
    return np.asarray(vecs, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint writing
# ─────────────────────────────────────────────────────────────────────────────


PREVIOUS_STRATEGIES = tuple(
    strategy for strategy in SAMPLE_STRATEGIES if strategy != "all"
)


def resolve_previous_sample_count(
    n_available: int,
    *,
    previous_n: Optional[int] = None,
    previous_prop: Optional[float] = None,
) -> Optional[int]:
    """Resolve ``--previous-n`` / ``--previous-prop`` to a sample count.

    Returns ``None`` when neither flag is set (keep all). Otherwise returns a
    non-negative integer (caller may still cap at ``n_available``).
    """
    return resolve_sample_count(
        n_available,
        n=previous_n,
        prop=previous_prop,
        n_name="--previous-n",
        prop_name="--previous-prop",
    )


def select_previous_train_indices(
    n_words: int,
    *,
    previous_n: Optional[int] = None,
    previous_prop: Optional[float] = None,
    strategy: str = "random",
    features: Optional[np.ndarray] = None,
    seed: int,
    required: Optional[Sequence[int]] = None,
) -> List[int]:
    """Return row indices into the previous-target list for the coreset.

    Indices are sorted ascending so callers can slice aligned feature/bias
    rows without reordering. ``None`` count means keep all rows. ``required``
    indices are always returned. They count toward ``previous_n``; when they
    already meet or exceed that count, no further rows are added and none of
    the required rows are dropped.
    """
    required_idxs = sorted({int(index) for index in (required or [])})
    if required_idxs and (required_idxs[0] < 0 or required_idxs[-1] >= n_words):
        raise ValueError("required previous-target index is out of range")
    if not required_idxs:
        return select_subset_indices(
            n_words,
            n=previous_n,
            prop=previous_prop,
            strategy=strategy,
            features=features,
            seed=seed,
            allowed=PREVIOUS_STRATEGIES,
            n_name="--previous-n",
            prop_name="--previous-prop",
            strategy_name="--previous-strategy",
        )

    count = resolve_sample_count(
        n_words,
        n=previous_n,
        prop=previous_prop,
        n_name="--previous-n",
        prop_name="--previous-prop",
    )
    if count is None or count >= n_words:
        return list(range(n_words))
    if len(required_idxs) >= count:
        return required_idxs

    required_set = set(required_idxs)
    pool = [index for index in range(n_words) if index not in required_set]
    pool_features = None
    if features is not None:
        pool_features = np.asarray(features)[np.asarray(pool, dtype=int)]
    local = select_subset_indices(
        len(pool),
        n=count - len(required_idxs),
        strategy=strategy,
        features=pool_features,
        seed=seed,
        allowed=PREVIOUS_STRATEGIES,
        n_name="--previous-n",
        prop_name="--previous-prop",
        strategy_name="--previous-strategy",
    )
    return sorted(required_idxs + [pool[index] for index in local])


def load_previous_include_targets(path: str) -> list[str]:
    """Targets that must be kept in the previous-train loss set.

    Accepts a JSON list of strings, or a molopt warmstart file whose ``seeds``
    object maps each seed to a list of SMILES.
    """
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        raw_rows: list = data
    elif isinstance(data, dict) and isinstance(data.get("seeds"), dict):
        raw_rows = []
        for seed_rows in data["seeds"].values():
            if not isinstance(seed_rows, list):
                raise ValueError(f"{path}: each seeds entry must be a list of targets")
            raw_rows.extend(seed_rows)
    else:
        raise ValueError(
            f"{path}: expected a JSON list of targets or a warmstart object with 'seeds'"
        )
    targets: list[str] = []
    seen: set[str] = set()
    for item in raw_rows:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        targets.append(text)
    if not targets:
        raise ValueError(f"{path}: no targets to include")
    return targets


def select_previous_train_targets(
    old_words: Sequence[str],
    *,
    previous_n: Optional[int] = None,
    previous_prop: Optional[float] = None,
    strategy: str = "random",
    features: Optional[np.ndarray] = None,
    seed: int,
    required: Optional[Sequence[int]] = None,
) -> List[str]:
    """Choose prior train targets for ``--include-prev-targets`` loss rows.

    With neither ``previous_n`` nor ``previous_prop``, keeps every previous
    target. Otherwise selects the resolved count (capped at ``len(old_words)``)
    via ``strategy``, preserving the original relative order of ``old_words``.
    ``required`` row indices are always kept and count toward that budget.

    Geometry strategies (``herding``, ``k-center-greedy``) require ``features``
    with shape ``[len(old_words), D]`` (typically source bias means).
    """
    words = list(old_words)
    idxs = select_previous_train_indices(
        len(words),
        previous_n=previous_n,
        previous_prop=previous_prop,
        strategy=strategy,
        features=features,
        seed=seed,
        required=required,
    )
    return [words[i] for i in idxs]


def _materialize_combined_bias_tables(
    ckpt,
    *,
    old_words: Sequence[str],
    new_words: Sequence[str],
    old_mu_source: np.ndarray,
    old_logvar_source: Optional[np.ndarray],
    learned,
    include_prev: bool,
    learn_bias_network: bool,
    raw_defs: dict[str, str],
    batch_size: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Assemble the final ``[N, rank]`` bias table for ALL combined ids.

    Old-target biases are refreshed from the final trained model whenever they
    can have changed (network retrained, or old biases retrained via
    ``include_prev``); otherwise they are carried over unchanged. New-target rows
    always come from the final learned/predicted values.

    With ``include_prev``, ``learned`` may cover only a sampled previous subset
    plus the new targets; untrained previous rows fall back to the live network
    (when ``learn_bias_network``) or the source tables.
    """
    intervention = _get_intervention(ckpt.reft_model)
    has_learnable_logvar = (
        getattr(intervention, "fixed_logvar", None) is None
        and getattr(getattr(intervention, "bias_network", None), "learnable_logvar", False)
    )
    norm = target_normalizer(
        str((getattr(ckpt, "saved_cfg", None) or {}).get("task", "semantle"))
    )

    learned_mu = np.asarray(learned.mu, dtype=np.float32)
    learned_logvar = (
        np.asarray(learned.logvar, dtype=np.float32)
        if (has_learnable_logvar and getattr(learned, "logvar", None) is not None)
        else None
    )
    learned_targets = [str(w) for w in getattr(learned, "targets", [])]
    learned_by_word = {norm(word): i for i, word in enumerate(learned_targets)}

    def _row_from_learned(word: str) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        idx = learned_by_word.get(norm(word))
        if idx is None:
            return None
        mu_row = learned_mu[idx]
        lv_row = learned_logvar[idx] if learned_logvar is not None else None
        return mu_row, lv_row

    if include_prev:
        # Fast path: training covered every combined id in order.
        if learned_mu.shape[0] == len(old_words) + len(new_words) and (
            not learned_targets
            or list(learned_targets) == list(old_words) + list(new_words)
        ):
            return learned_mu, learned_logvar

        missing_old = [w for w in old_words if norm(w) not in learned_by_word]
        if missing_old and learn_bias_network:
            miss_defs = {w: raw_defs[w] for w in missing_old if w in raw_defs}
            pred_mu, pred_logvar = predict_bias_network_rows(
                ckpt, list(missing_old), definitions=miss_defs or None
            )
            pred_mu = np.asarray(pred_mu, dtype=np.float32)
            pred_logvar = (
                np.asarray(pred_logvar, dtype=np.float32)
                if (has_learnable_logvar and pred_logvar is not None)
                else None
            )
            pred_by_word = {norm(w): i for i, w in enumerate(missing_old)}
        else:
            pred_mu = None
            pred_logvar = None
            pred_by_word = {}

        old_source = np.asarray(old_mu_source, dtype=np.float32)
        old_source_lv = (
            np.asarray(old_logvar_source, dtype=np.float32)
            if (has_learnable_logvar and old_logvar_source is not None)
            else None
        )
        old_mu_rows: List[np.ndarray] = []
        old_lv_rows: List[np.ndarray] = []
        for i, word in enumerate(old_words):
            learned_row = _row_from_learned(word)
            if learned_row is not None:
                mu_row, lv_row = learned_row
            elif norm(word) in pred_by_word:
                pidx = pred_by_word[norm(word)]
                mu_row = pred_mu[pidx]
                lv_row = pred_logvar[pidx] if pred_logvar is not None else None
            else:
                mu_row = old_source[i]
                lv_row = old_source_lv[i] if old_source_lv is not None else None
            old_mu_rows.append(np.asarray(mu_row, dtype=np.float32))
            if has_learnable_logvar and lv_row is not None:
                old_lv_rows.append(np.asarray(lv_row, dtype=np.float32))

        new_mu_rows: List[np.ndarray] = []
        new_lv_rows: List[np.ndarray] = []
        for word in new_words:
            learned_row = _row_from_learned(word)
            if learned_row is None:
                raise ValueError(
                    f"include_prev learned result missing new target {word!r}"
                )
            mu_row, lv_row = learned_row
            new_mu_rows.append(np.asarray(mu_row, dtype=np.float32))
            if has_learnable_logvar and lv_row is not None:
                new_lv_rows.append(np.asarray(lv_row, dtype=np.float32))

        mu_all = np.stack(old_mu_rows + new_mu_rows, axis=0)
        logvar_all: Optional[np.ndarray] = None
        if has_learnable_logvar and len(old_lv_rows) + len(new_lv_rows) == len(
            old_words
        ) + len(new_words):
            logvar_all = np.stack(old_lv_rows + new_lv_rows, axis=0)
        return mu_all, logvar_all

    # Not include_prev: learned.mu covers only the new targets.
    new_mu = learned_mu
    new_logvar = learned_logvar

    if learn_bias_network and old_words:
        # The network changed for everyone: re-predict old biases from it.
        old_defs = {w: raw_defs[w] for w in old_words if w in raw_defs}
        old_mu, old_logvar = predict_bias_network_rows(
            ckpt, list(old_words), definitions=old_defs or None
        )
        old_mu = np.asarray(old_mu, dtype=np.float32)
        old_logvar = (
            np.asarray(old_logvar, dtype=np.float32)
            if (has_learnable_logvar and old_logvar is not None)
            else None
        )
    else:
        old_mu = np.asarray(old_mu_source, dtype=np.float32)
        old_logvar = (
            np.asarray(old_logvar_source, dtype=np.float32)
            if (has_learnable_logvar and old_logvar_source is not None)
            else None
        )

    mu_all = (
        np.concatenate([old_mu, new_mu], axis=0) if len(old_mu) else new_mu
    )
    logvar_all: Optional[np.ndarray] = None
    if has_learnable_logvar and old_logvar is not None and new_logvar is not None:
        logvar_all = (
            np.concatenate([old_logvar, new_logvar], axis=0)
            if len(old_logvar)
            else new_logvar
        )
    return mu_all, logvar_all


def _write_new_checkpoint(
    ckpt,
    output_dir: str,
    *,
    combined_items: List[Dict[str, Any]],
    combined_words: List[str],
    mu_all: np.ndarray,
    logvar_all: Optional[np.ndarray],
    def_embed_lookup: Optional[dict[str, str]],
    extend_meta: Dict[str, Any],
    resolved_train_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Write the eval-ready checkpoint tree for the extended subspace."""
    os.makedirs(output_dir, exist_ok=True)
    intervention = _get_intervention(ckpt.reft_model)
    device = intervention.rotate_layer.weight.device
    task = (ckpt.saved_cfg or {}).get("task", "semantle")

    # Refresh embed_cache to the combined vocabulary so the saved intervention
    # buffer matches the eval-time rebuild (embed_cache provenance only; the
    # llm_encoder path keeps embed_cache=None and is unaffected).
    if getattr(intervention, "embed_cache", None) is not None:
        feats = encode_reference_embeddings(
            combined_words, definition_lookup=def_embed_lookup, task=task
        )
        intervention.register_buffer(
            "embed_cache",
            torch.as_tensor(feats, dtype=torch.float32, device=device),
        )

    # Drop the in-memory materialized lookup buffers (attached when the source
    # checkpoint was loaded) so they are not persisted into the saved intervention
    # — eval attaches the fresh bias_tables.pt after load, and stale [M, rank]
    # buffers would otherwise be unexpected keys (or a size mismatch) at load time.
    detach_materialized_bias_tables(intervention)

    ckpt.reft_model.save_intervention(
        save_directory=os.path.join(output_dir, "intervenable_model"),
        include_model=False,
    )

    with open(os.path.join(output_dir, "items.json"), "w", encoding="utf-8") as f:
        json.dump(combined_items, f, indent=2)

    # Bias tables materialized for ALL combined ids from the final trained model.
    tables_path = os.path.join(output_dir, "bias_tables.pt")
    meta: Dict[str, Any] = {}
    fixed_logvar = getattr(intervention, "fixed_logvar", None)
    if fixed_logvar is not None:
        meta["fixed_logvar"] = float(fixed_logvar)
    if getattr(intervention, "embed_cache", None) is not None:
        meta["target_embed_dim"] = int(intervention.embed_cache.shape[1])
    elif getattr(getattr(intervention, "bias_network", None), "fc1", None) is not None:
        meta["target_embed_dim"] = int(intervention.bias_network.fc1.in_features)
    save_bias_tables(
        tables_path,
        torch.as_tensor(mu_all, dtype=torch.float32),
        torch.as_tensor(logvar_all, dtype=torch.float32) if logvar_all is not None else None,
        metadata=meta,
    )

    ckpt.tokenizer.save_pretrained(output_dir)

    # intervention_config.json: inherit the source config, update to the extended
    # vocabulary, point at the fresh bias tables, and drop stale embed-cache paths
    # so eval rebuilds from the combined words (values are overridden by the saved
    # intervention buffer regardless).
    src_result_dir = extend_meta["result_dir"]
    intervention_cfg = load_intervention_config(src_result_dir)
    intervention_cfg["num_words"] = len(combined_items)
    intervention_cfg["bias_materialized"] = True
    intervention_cfg["bias_tables_path"] = os.path.abspath(tables_path)
    intervention_cfg["embed_cache_path"] = None
    intervention_cfg["eval_embed_cache_path"] = None
    if "target_embed_dim" in meta:
        intervention_cfg["bias_network_embed_dim"] = meta["target_embed_dim"]
    # Persist annealing / loss coeffs / stop criteria used for this extend so
    # later recovery inherits the override, not only the parent training values.
    if resolved_train_config:
        for key in (
            "linear_annealing_map",
            "lambda_ce",
            "lambda_sdpo",
            "kl_beta",
            "stop_threshold",
            "stop_threshold_min",
            "stop_threshold_frac",
            "eval_selection_metric",
            "eval_epochs",
        ):
            if key in resolved_train_config:
                intervention_cfg[key] = resolved_train_config[key]
    with open(
        os.path.join(output_dir, INTERVENTION_CONFIG_NAME), "w", encoding="utf-8"
    ) as f:
        json.dump(intervention_cfg, f, indent=2)

    # training_config.json: inherit the source training config (so eval's
    # load_merged_run_config resolves model shape, semantle_csv, eval knobs), then
    # record the extend settings.
    training_cfg = load_training_config(src_result_dir)
    training_cfg["num_anchors"] = len(combined_items)
    training_cfg["embed_cache_path"] = None
    training_cfg["eval_embed_cache_path"] = None
    training_cfg["extend"] = extend_meta
    if resolved_train_config:
        for key in (
            "linear_annealing_map",
            "lambda_ce",
            "lambda_sdpo",
            "kl_beta",
            "stop_threshold",
            "stop_threshold_min",
            "stop_threshold_frac",
            "eval_selection_metric",
            "eval_epochs",
        ):
            if key in resolved_train_config:
                training_cfg[key] = resolved_train_config[key]
    with open(
        os.path.join(output_dir, TRAINING_CONFIG_NAME), "w", encoding="utf-8"
    ) as f:
        json.dump(training_cfg, f, indent=2)

    return output_dir


# ─────────────────────────────────────────────────────────────────────────────
# Post-training eval panels
# ─────────────────────────────────────────────────────────────────────────────


def _panel_from_slice(
    words: Sequence[str],
    greedy: Sequence[str],
    greedy_sims: np.ndarray,
    temp_results: dict[float, list[dict]],
    idx: Sequence[int],
    tau: float,
    *,
    task: str = "semantle",
    rdkit_map_path: Optional[str] = None,
) -> dict:
    sub_words = [words[i] for i in idx]
    sub_greedy = [greedy[i] for i in idx]
    sub_sims = np.asarray(greedy_sims, dtype=np.float64)[list(idx)] if len(idx) else np.zeros(0)
    sub_temp = {t: [temp_results[t][i] for i in idx] for t in temp_results}
    return recon_metrics(
        targets=sub_words,
        greedy_decodes=sub_greedy,
        greedy_embed_sims=sub_sims,
        temp_results=sub_temp,
        tau=tau,
        task=task,
        rdkit_map_path=rdkit_map_path,
    )


def _run_post_eval(
    output_dir: str,
    *,
    model_name: str,
    layer: int,
    low_rank_dim: int,
    cache_dir: Optional[str],
    torch_dtype: Optional[str],
    reduced_interp: List[str],
    reduced_extrap: List[str],
    n_samples: int,
    top_p: float,
    eval_batch_size: int,
    max_new_tokens: int,
    tau: float,
    seed: int,
    miss_temp: float,
) -> dict:
    """Load the new checkpoint and compute recon panels + per-word hit maps."""
    ckpt = load_eval_checkpoint(
        output_dir, model_name, layer, low_rank_dim, cache_dir, torch_dtype=torch_dtype
    )
    try:
        saved_cfg = ckpt.saved_cfg or {}
        task = str(saved_cfg.get("task", "semantle"))
        rdkit_map_path = rdkit_map_path_for_cfg(saved_cfg)
        temps = sorted(set(EVAL_TEMPERATURES) | {miss_temp})
        items = ckpt.items
        train_words = [it.get("word", it["target"]) for it in items]
        train_ids = [int(it["id"]) for it in items]
        old_idx = [i for i, it in enumerate(items) if it.get("origin") != "new"]
        new_idx = [i for i, it in enumerate(items) if it.get("origin") == "new"]

        greedy, temp_results = decode_targets(
            ckpt,
            train_words,
            word_ids=train_ids,
            temps=temps,
            n_samples=n_samples,
            top_p=top_p,
            batch_size=eval_batch_size,
            max_new_tokens=max_new_tokens,
            decode_seed=seed + 20_000,
        )
        greedy_sims = embedding_sim_per_text(
            train_words, greedy, task=saved_cfg.get("task", "semantle")
        )
        recon = _panel_from_slice(
            train_words,
            greedy,
            greedy_sims,
            temp_results,
            range(len(train_words)),
            tau,
            task=task,
            rdkit_map_path=rdkit_map_path,
        )
        recon_old = _panel_from_slice(
            train_words,
            greedy,
            greedy_sims,
            temp_results,
            old_idx,
            tau,
            task=task,
            rdkit_map_path=rdkit_map_path,
        )
        recon_new = _panel_from_slice(
            train_words,
            greedy,
            greedy_sims,
            temp_results,
            new_idx,
            tau,
            task=task,
            rdkit_map_path=rdkit_map_path,
        )

        old_recall, old_greedy = hit_maps_from_slice(
            train_words, greedy, temp_results, old_idx, miss_temp
        )
        new_recall, new_greedy = hit_maps_from_slice(
            train_words, greedy, temp_results, new_idx, miss_temp
        )
        hits = {
            "original_train": {
                "recall": old_recall,
                "greedy": old_greedy,
            },
            "original_test_misses": {
                "recall": new_recall,
                "greedy": new_greedy,
            },
        }

        recon_test: dict = {}
        test_meta: dict = {}
        reduced_test = list(reduced_interp) + list(reduced_extrap)
        if reduced_test:
            bias_vecs = predict_test_bias_vectors(ckpt, reduced_test, eval_batch_size)
            rt_greedy, rt_temp = decode_targets(
                ckpt,
                reduced_test,
                bias_vectors=bias_vecs,
                temps=temps,
                n_samples=n_samples,
                top_p=top_p,
                batch_size=eval_batch_size,
                max_new_tokens=max_new_tokens,
                decode_seed=seed + 30_000,
            )
            rt_sims = embedding_sim_per_text(
                reduced_test, rt_greedy, task=saved_cfg.get("task", "semantle")
            )
            recon_test = recon_test_metrics(
                full_targets=reduced_test,
                interp_targets=reduced_interp,
                extrap_targets=reduced_extrap,
                greedy_decodes=rt_greedy,
                greedy_embed_sims=rt_sims,
                temp_results=rt_temp,
                tau=tau,
                task=task,
                rdkit_map_path=rdkit_map_path,
            )
            test_meta = {
                "n_test": len(reduced_test),
                "n_interp": len(reduced_interp),
                "n_extrap": len(reduced_extrap),
                "interp_words": list(reduced_interp),
                "extrap_words": list(reduced_extrap),
            }
            hits["original_test_hits"] = {
                "recall": recall_hits_at(reduced_test, rt_temp, miss_temp),
                "greedy": greedy_hits(reduced_test, rt_greedy),
            }
        else:
            hits["original_test_hits"] = {"recall": {}, "greedy": {}}

        results = {
            "n_words": len(train_words),
            "n_old": len(old_idx),
            "n_new": len(new_idx),
            "recon": recon,
            "recon_old": recon_old,
            "recon_new": recon_new,
            "hits": hits,
        }
        if recon_test:
            results["recon_test"] = recon_test
            results["recon_test_meta"] = test_meta

        eval_dir = os.path.join(output_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        # Hit maps stay in-memory for the summary; raw panel dump omits them.
        dump = {k: v for k, v in results.items() if k != "hits"}
        with open(os.path.join(eval_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(dump, f, indent=2)
        return results
    finally:
        release_eval_checkpoint(ckpt)


def _resolve_wandb_project(args, config: dict) -> Optional[str]:
    """Project from CLI, env, or the source run's wandb_meta.json."""
    if args.wandb_project:
        return args.wandb_project
    if os.environ.get("WANDB_PROJECT"):
        return os.environ["WANDB_PROJECT"]
    meta_path = os.path.join(config.get("result_dir", ""), "wandb_meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                return json.load(f).get("project")
        except (OSError, ValueError):
            return None
    return None


def _wandb_init_kwargs(
    *,
    args,
    output_dir: str,
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build fresh-run WandB init kwargs, or report why logging is unavailable."""
    project = _resolve_wandb_project(args, config)
    if not project:
        print(
            "[extend] no WandB project (pass --wandb-project or set WANDB_PROJECT) "
            "— skipping wandb logging.",
            file=sys.stderr,
        )
        return None

    entity = args.wandb_entity or os.environ.get("WANDB_ENTITY")
    name = args.wandb_run_name or f"extend-{os.path.basename(os.path.normpath(output_dir))}"
    init_kwargs: Dict[str, Any] = {
        "project": project,
        "name": name,
        "config": config,
    }
    if entity:
        init_kwargs["entity"] = entity
    if args.wandb_group:
        init_kwargs["group"] = args.wandb_group
    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir
    return init_kwargs


def _start_live_wandb_run(
    *,
    args,
    output_dir: str,
    config: Dict[str, Any],
) -> Optional[Any]:
    """Start the fresh extend run before training so eval curves stream live."""
    try:
        import wandb
    except ImportError:
        print("[extend] wandb not installed — skipping wandb logging.", file=sys.stderr)
        return None

    init_kwargs = _wandb_init_kwargs(
        args=args, output_dir=output_dir, config=config
    )
    if init_kwargs is None:
        return None
    run = wandb.init(**init_kwargs)
    _define_extend_wandb_metrics(run)
    run_url = getattr(run, "url", None)
    print(
        f"[extend] WandB live run started"
        f"{f' -> {run_url}' if run_url else ''}",
        flush=True,
    )
    return run


def _define_extend_wandb_metrics(run: Any) -> None:
    """Plot train and eval panels on the shared ``train/global_step`` x-axis."""
    define_metric = getattr(run, "define_metric", None)
    if not callable(define_metric):
        return
    define_metric("train/global_step")
    metric_names = [
        "train/loss",
        "train/ce_loss",
        "train/margin_loss",
        "train/aux_loss",
        "train/sdpo_loss",
        "train/sdpo_n_seq",
        "train/sdpo_n_tokens",
        "train/kl_beta_effective",
        "train/lambda_ce_effective",
        "train/lambda_sdpo_effective",
        "train/mu_norm",
        "train/bias_var",
        "train/learning_rate",
        "train/grad_norm",
        "train/epoch",
        "eval/epoch",
        "eval/avg_embed_sim",
        "eval/min_embed_sim",
        "eval/embed_sim_gte_tau",
        "eval/embed_sim_tau",
        "eval/avg_selection_sim",
        "eval/min_selection_sim",
        "eval/n_recovered",
        "eval/n_targets",
        "eval/recover_rate",
    ]
    for group in ("previous", "new", "combined"):
        metric_names.extend(
            [
                f"eval/{group}/avg_embed_sim",
                f"eval/{group}/min_embed_sim",
                f"eval/{group}/embed_sim_gte_tau",
                f"eval/{group}/embed_sim_tau",
                f"eval/{group}/avg_selection_sim",
                f"eval/{group}/min_selection_sim",
                f"eval/{group}/n_recovered",
                f"eval/{group}/n_targets",
                f"eval/{group}/recover_rate",
            ]
        )
    for name in metric_names:
        define_metric(name, step_metric="train/global_step")


_EVAL_SCALAR_KEYS = (
    ("avg_embed_sim", "avg_embed_sim"),
    ("min_embed_sim", "min_embed_sim"),
    ("embed_sim_gte_tau", "embed_sim_gte_tau"),
    ("embed_sim_tau", "embed_sim_tau"),
    # Whichever metric drove early stopping; equal to the embed_sim pair unless
    # the run selected a molecular metric.
    ("avg_selection_sim", "avg_selection_sim"),
    ("min_selection_sim", "min_selection_sim"),
    ("n_recovered", "n_recovered"),
    ("n_targets", "n_targets"),
)


def _wandb_eval_scalars(
    metrics: Dict[str, Any],
    *,
    prefix: str,
    n_eval_targets: Optional[int] = None,
    include_decodes: bool = True,
) -> Dict[str, Any]:
    """Map one eval-slice dict onto WandB ``prefix/*`` keys."""
    out: Dict[str, Any] = {}
    for source, leaf in _EVAL_SCALAR_KEYS:
        value = metrics.get(source)
        if isinstance(value, (int, float)) and np.isfinite(value):
            out[f"{prefix}/{leaf}"] = float(value)
    n_targets = int(
        metrics.get("n_targets")
        if metrics.get("n_targets") is not None
        else (n_eval_targets or 0)
    )
    recovered = out.get(f"{prefix}/n_recovered")
    if n_targets > 0 and isinstance(recovered, (int, float)):
        out[f"{prefix}/recover_rate"] = float(recovered) / n_targets
    if include_decodes:
        decode_rows = metrics.get("greedy_decodes")
        if isinstance(decode_rows, list):
            out[f"{prefix}/greedy_decodes"] = _wandb_eval_decode_table(decode_rows)
    return out


def _wandb_eval_decode_table(rows: Sequence[Dict[str, Any]]) -> Any:
    """Create the sorted per-target decode table logged at each eval epoch."""
    import wandb

    return wandb.Table(
        columns=[
            "rank",
            "target",
            "greedy_decode",
            "embed_similarity",
            "exact_match",
        ],
        data=[
            [
                row.get("rank"),
                row.get("target"),
                row.get("greedy_decode"),
                row.get("embed_similarity"),
                row.get("exact_match"),
            ]
            for row in rows
        ],
    )


def _make_live_wandb_progress_callback(
    wandb_run: Any,
    *,
    n_eval_targets: Optional[int] = None,
):
    """Stream trainer loss panels and learned-set eval panels to WandB."""

    def _callback(payload: Dict[str, Any]) -> None:
        event = payload.get("event")
        if event == "eval_result":
            eval_metrics: Dict[str, Any] = {
                "eval/epoch": float(payload.get("epoch") or 0.0),
                "train/global_step": float(payload.get("step") or 0),
            }
            eval_metrics.update(
                _wandb_eval_scalars(
                    payload,
                    prefix="eval",
                    n_eval_targets=n_eval_targets,
                )
            )
            groups = payload.get("groups")
            if isinstance(groups, dict):
                for group_name, group_metrics in groups.items():
                    if not isinstance(group_metrics, dict):
                        continue
                    eval_metrics.update(
                        _wandb_eval_scalars(
                            group_metrics,
                            prefix=f"eval/{group_name}",
                        )
                    )
            wandb_run.log(eval_metrics)
            return

        if event != "log":
            return

        train_metrics: Dict[str, float] = {}
        for key in (
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
        ):
            value = payload.get(key)
            if isinstance(value, (int, float)) and np.isfinite(value):
                train_metrics[f"train/{key}"] = float(value)
        if train_metrics:
            train_metrics["train/global_step"] = float(payload.get("step") or 0)
            epoch = payload.get("epoch")
            if isinstance(epoch, (int, float)) and np.isfinite(epoch):
                train_metrics["train/epoch"] = float(epoch)
            wandb_run.log(train_metrics)

    return _callback


def wandb_log_payloads(
    summary: Dict[str, Any],
    eval_history: Sequence[Dict[str, Any]],
    *,
    raw_panels: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build WandB payloads (unit-testable without initializing wandb).

    Returns ``{"curves": [...], "final": {...}, "bar_rows": [...]}``.
    """
    curves = training_curve_logs(eval_history)
    results = summary.get("results") or {}
    final: Dict[str, Any] = {}
    final.update(flatten_result_scalars(results, "results"))
    training = summary.get("training") or {}
    for key in (
        "epochs",
        "steps",
        "mean_train_loss",
        "final_eval_avg_embed_sim",
        "final_eval_min_embed_sim",
    ):
        if key in training and isinstance(training[key], (int, float)):
            if np.isfinite(training[key]):
                final[f"training/{key}"] = float(training[key])
    if raw_panels:
        for group in ("recon", "recon_old", "recon_new", "recon_test"):
            block = raw_panels.get(group) or {}
            for k, v in block.items():
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)) and np.isfinite(v):
                    final[f"raw/{group}/{k}"] = float(v)
    return {
        "curves": curves,
        "final": final,
        "bar_rows": before_after_bar_rows(results),
    }


def _log_extend_to_wandb(
    summary: Dict[str, Any],
    eval_history: Sequence[Dict[str, Any]],
    *,
    args,
    output_dir: str,
    wandb_config: Dict[str, Any],
    raw_panels: Optional[Dict[str, Any]] = None,
    run: Optional[Any] = None,
    log_curves: bool = True,
) -> None:
    """Finish live logging, or create a fresh run for deferred logging."""
    try:
        import wandb
    except ImportError:
        print("[extend] wandb not installed — skipping wandb logging.", file=sys.stderr)
        return

    live_run = run is not None
    final_config = {
        **(summary.get("setup") or {}),
        "paths": summary.get("paths"),
        "question": summary.get("question"),
        **{k: v for k, v in wandb_config.items() if k not in ("result_dir",)},
        "result_dir": wandb_config.get("result_dir"),
    }
    if run is None:
        init_kwargs = _wandb_init_kwargs(
            args=args,
            output_dir=output_dir,
            config=final_config,
        )
        if init_kwargs is None:
            return
        run = wandb.init(**init_kwargs)
        _define_extend_wandb_metrics(run)
    elif getattr(run, "config", None) is not None:
        run.config.update(final_config, allow_val_change=True)

    payloads = wandb_log_payloads(
        summary, eval_history, raw_panels=raw_panels
    )
    try:
        if log_curves:
            for history_entry, row in zip(eval_history, payloads["curves"]):
                step = int(row.pop("_step", 0))
                decode_rows = history_entry.get("greedy_decodes")
                if isinstance(decode_rows, list):
                    row["eval/greedy_decodes"] = _wandb_eval_decode_table(
                        decode_rows
                    )
                wandb.log(row, step=step)
        final_payload = dict(payloads["final"])
        bar_rows = payloads["bar_rows"]
        if bar_rows:
            table = wandb.Table(
                columns=["population", "phase", "recall_at_n"],
                data=[
                    [r["population"], r["phase"], r["recall_at_n"]] for r in bar_rows
                ],
            )
            # Combine population+phase so bar labels are unique.
            label_table = wandb.Table(
                columns=["label", "recall_at_n"],
                data=[
                    [f"{r['population']}/{r['phase']}", r["recall_at_n"]]
                    for r in bar_rows
                ],
            )
            final_payload["charts/before_after_recall_at_n"] = wandb.plot.bar(
                label_table,
                "label",
                "recall_at_n",
                title=(
                    "Recall@n before vs after "
                    "(original_train / original_test_misses / original_test_hits)"
                ),
            )
            final_payload["charts/before_after_table"] = table
        if live_run:
            # Live train/eval logs have already advanced WandB's internal step.
            # Let WandB append final panels instead of supplying an older epoch.
            wandb.log(final_payload)
        else:
            final_step = (
                max(
                    (int(r.get("epoch") or 0) for r in eval_history),
                    default=-1,
                )
                + 1
            )
            wandb.log(final_payload, step=final_step)
    finally:
        if run is not None:
            wandb.finish()


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────


def _write_summary(summary: dict, output_dir: str, result_dir: str) -> str:
    with open(os.path.join(output_dir, "extend_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    os.makedirs(OUTPUT_SUBDIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    run_tag = os.path.basename(os.path.normpath(result_dir))
    path = os.path.join(OUTPUT_SUBDIR, f"extend_subspace_{run_tag}_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[extend] summary -> {path}", flush=True)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--result-dir", required=True, help="Previous run checkpoint dir.")
    add_load_latest_argument(p)
    p.add_argument(
        "--new-targets",
        default=None,
        help=(
            "JSONL of {word, definition} pairs; use these as the absorption set "
            "instead of selecting recall@n misses."
        ),
    )
    p.add_argument(
        "--n-unlearnt-targets",
        type=int,
        default=None,
        help="Cap on the number of unlearnt targets to absorb (default: all).",
    )
    p.add_argument(
        "--miss-temp",
        type=float,
        default=1.0,
        help="Temperature whose recall@n defines a miss (default 1.0).",
    )
    p.add_argument(
        "--also-absorb-train-misses",
        action="store_true",
        help=(
            "Also absorb train-set recall@n misses into new_words "
            "(default: test misses only)."
        ),
    )
    # Detection / eval decode knobs (default from saved full-eval settings).
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    # Continue-training knobs (None => inherit saved training value).
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--kl-beta", type=float, default=None)
    p.add_argument("--lambda-ce", type=float, default=None)
    p.add_argument("--lambda-sdpo", type=float, default=None)
    p.add_argument(
        "--linear-annealing-map",
        default=None,
        help="Override checkpoint linear annealing map "
        "(KEY=START:END:EPOCHS or KEY=START:EPOCHS; empty string disables). "
        "Allowed keys: kl_beta, lambda_ce, lambda_sdpo. "
        "Omit to inherit from the checkpoint.",
    )
    p.add_argument("--lr-scheduler-type", default=None)
    p.add_argument("--warmup-ratio", type=float, default=None)
    p.add_argument("--max-grad-norm", type=float, default=None)
    p.add_argument("--weight-decay-mode", default=None)
    p.add_argument("--wd-W", type=float, default=None)
    p.add_argument("--wd-b", type=float, default=None)
    p.add_argument("--eval-epochs", type=int, default=None)
    p.add_argument(
        "--stop-threshold",
        type=float,
        default=None,
        help=(
            "Stop when mean during-train eval embed_sim exceeds this value "
            "(default: inherit from the source run; requires eval)."
        ),
    )
    p.add_argument(
        "--stop-threshold-min",
        type=float,
        default=None,
        help=(
            "Per-target embed_sim bar (tau) for early stop. Combined with "
            "--stop-threshold-frac (default inherit, else 1.0 = all targets). "
            "Requires eval. Combined with --stop-threshold when both are set."
        ),
    )
    p.add_argument(
        "--stop-threshold-frac",
        type=float,
        default=None,
        help=(
            "Minimum fraction of eval targets with embed_sim >= "
            "--stop-threshold-min. Default: inherit checkpoint, else 1.0. "
            "Values < 1 require --stop-threshold-min."
        ),
    )
    p.add_argument(
        "--eval-selection-metric",
        choices=("embed_sim", "rdkit_sim", "tfs"),
        default=None,
        help=(
            "Metric the stop thresholds compare against: embed_sim, rdkit_sim "
            "or tfs (the last two require a molopt checkpoint). Default: "
            "inherit from the source run, else embed_sim."
        ),
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--torch-dtype", default=None)
    # Extend mode flags.
    p.add_argument("--learn-w", action="store_true")
    p.add_argument("--learn-r", action="store_true")
    p.add_argument("--learn-bias-network", action="store_true")
    p.add_argument("--include-prev-targets", action="store_true")
    p.add_argument(
        "--previous-n",
        type=int,
        default=None,
        help=(
            "With --include-prev-targets, select this many previous train "
            "targets into the loss (default: all). Selection method is "
            "--previous-strategy. Mutually exclusive with --previous-prop."
        ),
    )
    p.add_argument(
        "--previous-prop",
        type=float,
        default=None,
        help=(
            "With --include-prev-targets, select this fraction [0, 1] of "
            "previous train targets into the loss (default: all). Selection "
            "method is --previous-strategy. Mutually exclusive with "
            "--previous-n."
        ),
    )
    p.add_argument(
        "--previous-include",
        default=None,
        help=(
            "JSON file of targets that must be in the --include-prev-targets "
            "loss set. A list of strings, or a molopt warmstart file with a "
            "seeds map. These count toward --previous-n. The remaining slots "
            "are filled by --previous-strategy. Targets beyond --previous-n "
            "are still kept."
        ),
    )
    p.add_argument(
        "--previous-strategy",
        choices=list(PREVIOUS_STRATEGIES),
        default="random",
        help=(
            "How to choose the --previous-n / --previous-prop subset of prior "
            "train targets: random (default), herding, or k-center-greedy "
            "(geometry on source bias means)."
        ),
    )
    # Model load overrides.
    p.add_argument("--model-name", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--low-rank-dim", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dry-run", action="store_true", help="Only report unlearnt targets.")
    # WandB pass-throughs.
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--wandb-dir", default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    saved_cfg = load_merged_run_config(args.result_dir)
    if not saved_cfg.get("add_bias_network"):
        raise SystemExit(
            "[extend] checkpoint has no bias network (--add-bias-network required)"
        )

    model_name = args.model_name or saved_cfg.get("model_name", "meta-llama/Llama-3.2-1B")
    layer = args.layer if args.layer is not None else saved_cfg.get("layer", 13)
    low_rank_dim = (
        args.low_rank_dim if args.low_rank_dim is not None else saved_cfg.get("low_rank_dim", 64)
    )
    cache_dir = args.cache_dir or saved_cfg.get("cache_dir")
    task = saved_cfg.get("task", "semantle")
    # Every target-identity comparison below (absorbed vs. old vs. held-out) keys
    # through this, so molopt compares canonical SMILES instead of lower-casing.
    norm = target_normalizer(task)
    seed = int(args.seed if args.seed is not None else saved_cfg.get("seed", 42))
    n_samples = int(
        args.n_samples if args.n_samples is not None else (saved_cfg.get("full_eval_gen_samples") or 25)
    )
    top_p = float(
        args.top_p
        if args.top_p is not None
        else (saved_cfg.get("full_eval_top_p") if saved_cfg.get("full_eval_top_p") is not None else 1.0)
    )
    eval_batch_size = int(
        args.eval_batch_size
        if args.eval_batch_size is not None
        else (saved_cfg.get("full_eval_batch_size") or 32)
    )
    max_new_tokens = int(
        args.max_new_tokens
        if args.max_new_tokens is not None
        else (saved_cfg.get("full_eval_max_new_tokens") or DEFAULT_MAX_NEW_TOKENS)
    )
    tau = float(saved_cfg.get("eval_embed_sim_tau", DEFAULT_EMBED_SIM_TAU))
    miss_temp = float(args.miss_temp)

    sdpo_active = float(saved_cfg.get("lambda_sdpo", 0.0)) > 0.0 and task_supports_sdpo(task)
    encoder_mode = saved_cfg.get("bias_input_source") == "llm_encoder"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = args.output_dir or os.path.join(
        args.result_dir, "extend", time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    )

    print(f"[extend] loading checkpoint {args.result_dir!r}", flush=True)
    ckpt = load_eval_checkpoint(
        args.result_dir,
        model_name,
        layer,
        low_rank_dim,
        cache_dir,
        torch_dtype=args.torch_dtype,
        load_latest=bool(args.load_latest),
    )

    old_items = list(ckpt.items)
    old_words = [it.get("word", it["target"]) for it in old_items]
    old_ids = [int(it["id"]) for it in old_items]
    old_norm = {norm(w) for w in old_words}
    run_raw_defs = _load_run_raw_definitions(saved_cfg)

    # ── Determine unlearnt targets + their definitions ─────────────────────
    new_targets_defs: dict[str, str] = {}
    # Baseline per-word hit maps captured at detection / pre-train decode time.
    before_old_recall: Optional[Dict[str, bool]] = None
    before_old_greedy: Optional[Dict[str, bool]] = None
    before_new_recall: Optional[Dict[str, bool]] = None
    before_new_greedy: Optional[Dict[str, bool]] = None
    before_remaining_recall: Optional[Dict[str, bool]] = None
    before_remaining_greedy: Optional[Dict[str, bool]] = None
    n_train_misses_detected: Optional[int] = None
    n_test_misses_detected: Optional[int] = None
    baseline_train_recall_at_n: Optional[float] = None
    baseline_test_recall_at_n: Optional[float] = None
    absorb_train: List[str] = []
    # The original test set (from the source run) is used for recall@n detection
    # AND for the post-training recon_test panel. Load it before the source
    # checkpoint is released; interp/extrap are re-split against the extended
    # train vocabulary at eval time.
    interp, extrap, test_source = load_test_set(args.result_dir, ckpt, saved_cfg)
    orig_test_words = list(interp) + list(extrap)
    temps = sorted(set(EVAL_TEMPERATURES) | {miss_temp})
    if args.new_targets:
        new_words, new_targets_defs = _read_new_targets_file(
            args.new_targets, task=task
        )
        new_words = [w for w in new_words if norm(w) not in old_norm]
        new_words = select_unlearnt_targets(
            new_words, [], args.n_unlearnt_targets, task=task
        )
        print(f"[extend] {len(new_words)} new targets from {args.new_targets!r}", flush=True)
        source = "new_targets_file"
        if not new_words:
            print("[extend] no new targets to absorb — nothing to do.", flush=True)
            release_eval_checkpoint(ckpt)
            return
        if args.dry_run:
            print("[extend] dry-run: new targets:", flush=True)
            for word in new_words:
                print(f"  - {word}", flush=True)
            release_eval_checkpoint(ckpt)
            return
    else:
        source = "recall_at_n_misses"
        print(
            f"[extend] detecting recall@{n_samples} misses (T={miss_temp}); "
            f"train={len(old_words)} test={len(orig_test_words)} (src={test_source})",
            flush=True,
        )

    # Decode the source populations for before/after comparisons. For a
    # --new-targets run this is measurement only; it does not select targets.
    train_greedy_dec, train_temp = decode_targets(
        ckpt,
        old_words,
        word_ids=old_ids,
        temps=temps,
        n_samples=n_samples,
        top_p=top_p,
        batch_size=eval_batch_size,
        max_new_tokens=max_new_tokens,
        decode_seed=seed + 10_000,
    )
    train_recall = recall_hits_at(old_words, train_temp, miss_temp)
    train_greedy_map = greedy_hits(old_words, train_greedy_dec)
    before_old_recall = train_recall
    before_old_greedy = train_greedy_map
    train_misses = [w for w in old_words if not train_recall[w]]
    baseline_train_recall_at_n = hit_rate(train_recall)

    test_recall: Dict[str, bool] = {}
    test_greedy_map: Dict[str, bool] = {}
    test_misses: List[str] = []
    if orig_test_words:
        pred_vecs = predict_test_bias_vectors(
            ckpt, orig_test_words, eval_batch_size
        )
        test_greedy_dec, test_temp = decode_targets(
            ckpt,
            orig_test_words,
            bias_vectors=pred_vecs,
            temps=temps,
            n_samples=n_samples,
            top_p=top_p,
            batch_size=eval_batch_size,
            max_new_tokens=max_new_tokens,
            decode_seed=seed + 11_000,
        )
        test_recall = recall_hits_at(orig_test_words, test_temp, miss_temp)
        test_greedy_map = greedy_hits(orig_test_words, test_greedy_dec)
        test_misses = [w for w in orig_test_words if not test_recall[w]]
    baseline_test_recall_at_n = hit_rate(test_recall)

    if source == "recall_at_n_misses":
        n_train_misses_detected = len(train_misses)
        n_test_misses_detected = len(test_misses)
        absorb_train = train_misses if args.also_absorb_train_misses else []
        new_words = select_unlearnt_targets(
            absorb_train, test_misses, args.n_unlearnt_targets, task=task
        )
        for word in new_words:
            if word in run_raw_defs:
                new_targets_defs[word] = run_raw_defs[word]
        print(
            f"[extend] misses: train={n_train_misses_detected} "
            f"test={n_test_misses_detected} "
            f"(also_absorb_train={bool(args.also_absorb_train_misses)}) "
            f"-> selected {len(new_words)} absorption targets",
            flush=True,
        )

    absorbed_norm = {norm(w) for w in new_words}
    remaining_words = [w for w in orig_test_words if norm(w) not in absorbed_norm]
    before_remaining_recall = {
        w: test_recall[w] for w in remaining_words if w in test_recall
    }
    before_remaining_greedy = {
        w: test_greedy_map[w] for w in remaining_words if w in test_greedy_map
    }
    before_new_recall, before_new_greedy = absorption_before_hit_maps(
        new_words,
        train_recall=train_recall,
        train_greedy=train_greedy_map,
        test_recall=test_recall,
        test_greedy=test_greedy_map,
        prefer_train_words=absorb_train,
        task=task,
    )

    if not new_words:
        print("[extend] no unlearnt targets to absorb — nothing to do.", flush=True)
        release_eval_checkpoint(ckpt)
        return

    new_norm = {norm(w) for w in new_words}

    if (sdpo_active or encoder_mode):
        missing = [w for w in new_words if not new_targets_defs.get(w)]
        if missing:
            raise SystemExit(
                f"[extend] {len(missing)} unlearnt targets lack definitions required for "
                f"{'SDPO' if sdpo_active else 'llm_encoder'} (e.g. {missing[:5]}). "
                "Provide them via --new-targets or the run's definitions file."
            )

    if args.dry_run:
        print("[extend] dry-run: unlearnt targets:", flush=True)
        for w in new_words:
            print(f"  - {w}", flush=True)
        release_eval_checkpoint(ckpt)
        return

    # ── Warm-start biases + training target set ────────────────────────────
    new_pred = _predict_bias_for_words(
        ckpt, new_words, raw_defs=new_targets_defs, batch_size=eval_batch_size
    )
    old_mu_source = (
        np.asarray(stack_bias_vectors(ckpt.reft_model, old_ids), dtype=np.float32)
        if old_ids
        else np.zeros((0, new_pred.shape[1]), dtype=np.float32)
    )
    # Source logvar (for direct-mode untrained old rows when variance is learnable).
    old_logvar_source: Optional[np.ndarray] = None
    src_tables_path = resolve_bias_tables_path(args.result_dir, saved_cfg=saved_cfg)
    if src_tables_path is not None:
        try:
            _, src_logvar, _ = load_bias_tables(src_tables_path, num_words=len(old_items))
            if src_logvar is not None:
                old_logvar_source = np.asarray(src_logvar, dtype=np.float32)
        except (OSError, ValueError):
            old_logvar_source = None

    include_prev = bool(args.include_prev_targets)
    previous_n = args.previous_n
    previous_prop = args.previous_prop
    previous_strategy = str(args.previous_strategy or "random").strip().lower()
    if (
        previous_n is not None or previous_prop is not None or args.previous_include
    ) and not include_prev:
        raise SystemExit(
            "[extend] --previous-n / --previous-prop / --previous-include "
            "require --include-prev-targets"
        )
    required_idxs: List[int] = []
    if args.previous_include:
        try:
            include_targets = load_previous_include_targets(args.previous_include)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"[extend] {exc}") from exc
        by_norm: Dict[str, int] = {}
        for index, word in enumerate(old_words):
            by_norm.setdefault(norm(word), index)
        missing_include = []
        seen_required: set[int] = set()
        for target in include_targets:
            index = by_norm.get(norm(target))
            if index is None:
                missing_include.append(target)
                continue
            if index not in seen_required:
                seen_required.add(index)
                required_idxs.append(index)
        if missing_include:
            preview = ", ".join(missing_include[:5])
            raise SystemExit(
                f"[extend] {len(missing_include)} --previous-include targets are not "
                f"in the checkpoint train set (e.g. {preview})"
            )
    try:
        resolved_previous_n = resolve_previous_sample_count(
            len(old_words),
            previous_n=previous_n,
            previous_prop=previous_prop,
        )
    except ValueError as exc:
        raise SystemExit(f"[extend] {exc}") from exc
    prev_train_words: List[str] = []
    eval_indices: Optional[List[int]]
    eval_groups: Optional[Dict[str, List[int]]] = None
    stop_eval_indices: Optional[List[int]] = None
    if include_prev:
        try:
            prev_idxs = select_previous_train_indices(
                len(old_words),
                previous_n=previous_n,
                previous_prop=previous_prop,
                strategy=previous_strategy,
                features=old_mu_source if old_words else None,
                seed=seed,
                required=required_idxs,
            )
        except ValueError as exc:
            raise SystemExit(f"[extend] {exc}") from exc
        prev_train_words = [old_words[i] for i in prev_idxs]
        if resolved_previous_n is not None and len(prev_train_words) < len(old_words):
            how = (
                f"previous-prop={previous_prop}"
                if previous_prop is not None
                else f"previous-n={resolved_previous_n}"
            )
            required_note = (
                f", {len(required_idxs)} required" if required_idxs else ""
            )
            print(
                f"[extend] selected {how} via {previous_strategy} -> "
                f"{len(prev_train_words)}/{len(old_words)} prior train targets "
                f"for loss{required_note}",
                flush=True,
            )
        # Warm-start rows must align to train_targets order (prev_idxs order).
        if prev_idxs:
            warm_prev = np.asarray(old_mu_source[prev_idxs], dtype=np.float32)
        else:
            warm_prev = np.zeros((0, new_pred.shape[1]), dtype=np.float32)
        train_targets = list(prev_train_words) + list(new_words)
        warm = (
            np.concatenate([warm_prev, new_pred], axis=0)
            if len(warm_prev)
            else new_pred
        )
        n_prev = len(prev_train_words)
        n_new = len(new_words)
        previous_idxs = list(range(n_prev))
        new_idxs = list(range(n_prev, n_prev + n_new))
        combined_idxs = list(range(n_prev + n_new))
        eval_indices = combined_idxs
        eval_groups = {
            "previous": previous_idxs,
            "new": new_idxs,
            "combined": combined_idxs,
        }
        # Early-stop / top-level eval stay on the absorption (new) set.
        stop_eval_indices = new_idxs
    else:
        train_targets = list(new_words)
        warm = new_pred
        eval_indices = None
        eval_groups = None
        stop_eval_indices = None

    # Raw definitions for every training target (used by SDPO / llm_encoder).
    train_defs: dict[str, str] = {}
    for w in train_targets:
        d = new_targets_defs.get(w) or run_raw_defs.get(w)
        if d:
            train_defs[w] = d

    live_wandb_run = None
    live_progress_callback = None
    if not args.no_wandb:
        live_wandb_run = _start_live_wandb_run(
            args=args,
            output_dir=output_dir,
            config={
                "result_dir": os.path.abspath(args.result_dir),
                "output_dir": os.path.abspath(output_dir),
                "task": task,
                "source": source,
                "miss_temp": miss_temp,
                "n_samples": n_samples,
                "n_original_train": len(old_words),
                "n_original_test_misses": len(new_words),
                "n_original_test_hits": len(remaining_words),
                "include_prev_targets": include_prev,
                "previous_n": previous_n,
                "previous_prop": previous_prop,
                "previous_strategy": previous_strategy,
                "n_previous_train_targets": len(prev_train_words),
                "learn_W": bool(args.learn_w),
                "learn_R": bool(args.learn_r),
                "learn_bias_network": bool(args.learn_bias_network),
            },
        )
        if live_wandb_run is not None:
            live_progress_callback = _make_live_wandb_progress_callback(
                live_wandb_run,
                n_eval_targets=len(new_words),
            )

    # ── Continue training ──────────────────────────────────────────────────
    print(
        f"[extend] training {len(train_targets)} targets "
        f"(new={len(new_words)}, include_prev={include_prev}"
        f"{f', previous_n={len(prev_train_words)}' if include_prev else ''}, "
        f"learn_W={args.learn_w}, learn_R={args.learn_r}, "
        f"learn_bias_network={args.learn_bias_network})...",
        flush=True,
    )
    try:
        learned = learn_biases_batched(
            ckpt,
            targets=train_targets,
            warm_start_mu=warm,
            definitions=train_defs or None,
            config=BatchLearnConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                kl_beta=args.kl_beta,
                lambda_ce=args.lambda_ce,
                lambda_sdpo=args.lambda_sdpo,
                linear_annealing_map=args.linear_annealing_map,
                seed=args.seed,
                max_grad_norm=args.max_grad_norm,
                lr_scheduler_type=args.lr_scheduler_type,
                warmup_ratio=args.warmup_ratio,
                weight_decay_mode=args.weight_decay_mode,
                wd_W=args.wd_W,
                wd_b=args.wd_b,
                torch_dtype=args.torch_dtype,
                eval_epochs=args.eval_epochs,
                eval_batch_size=eval_batch_size,
                eval_max_new_tokens=max_new_tokens,
                stop_threshold=args.stop_threshold,
                stop_threshold_min=args.stop_threshold_min,
                stop_threshold_frac=args.stop_threshold_frac,
                eval_selection_metric=args.eval_selection_metric,
                inherit_stop_threshold=True,
                learn_W=args.learn_w,
                learn_R=args.learn_r,
                learn_bias_network=args.learn_bias_network,
            ),
            eval_indices=eval_indices,
            eval_groups=eval_groups,
            stop_eval_indices=stop_eval_indices,
            show_progress=True,
            progress_callback=live_progress_callback,
            eval_result_callback=live_progress_callback,
        )
    except Exception:
        if live_wandb_run is not None:
            live_wandb_run.finish(exit_code=1)
        raise
    print(
        f"[extend] training done: {learned.steps} steps  "
        f"mean_train_loss={learned.mean_train_loss:.4f}",
        flush=True,
    )

    # ── Materialize combined bias tables + write checkpoint ────────────────
    mu_all, logvar_all = _materialize_combined_bias_tables(
        ckpt,
        old_words=old_words,
        new_words=new_words,
        old_mu_source=old_mu_source,
        old_logvar_source=old_logvar_source,
        learned=learned,
        include_prev=include_prev,
        learn_bias_network=args.learn_bias_network,
        raw_defs={**run_raw_defs, **new_targets_defs},
        batch_size=eval_batch_size,
    )

    def_embed_lookup = None
    if saved_cfg.get("use_definition_embeds"):
        base = definition_lookup_for_cfg(saved_cfg) or {}
        def_embed_lookup = dict(base)
        rdkit_lookup = rdkit_definition_lookup_for_cfg(saved_cfg)
        for w in new_words:
            d = new_targets_defs.get(w) or run_raw_defs.get(w)
            if d:
                definition = definition_text_for_cfg(
                    saved_cfg, w, d, rdkit_lookup=rdkit_lookup
                )
                def_embed_lookup[w] = definition_embedding_text(
                    task, w, definition
                )

    known_train_norm = {
        norm(str(item.get("word", item.get("target", ""))))
        for item in old_items
        if (
            item.get("original_split") == "train"
            or (
                item.get("original_split") is None
                and item.get("origin") != "new"
            )
        )
    }
    original_test_norm = {norm(word) for word in orig_test_words}
    combined_old_items: List[Dict[str, Any]] = []
    for item in old_items:
        original_split = item.get("original_split")
        if original_split not in {"train", "test", "additional"}:
            word_norm = norm(str(item.get("word", item.get("target", ""))))
            original_split = (
                "train"
                if word_norm in known_train_norm
                else "test"
                if word_norm in original_test_norm
                else "additional"
            )
        combined_old_items.append(
            {**item, "original_split": original_split}
        )
    original_train_norm = {
        norm(str(item.get("word", item.get("target", ""))))
        for item in combined_old_items
        if item["original_split"] == "train"
    }
    new_original_splits = {
        word: (
            "train"
            if norm(word) in original_train_norm
            else "test"
            if norm(word) in original_test_norm
            else "additional"
        )
        for word in new_words
    }
    combined_items, old_idx, new_idx = build_combined_items(
        combined_old_items,
        new_words,
        prompt=ckpt.prompt,
        new_original_splits=new_original_splits,
    )
    combined_words = [it["word"] for it in combined_items]

    extend_meta = {
        "result_dir": os.path.abspath(args.result_dir),
        "n_old_targets": len(old_words),
        "n_new_targets": len(new_words),
        "unlearnt_source": source,
        "new_targets": list(new_words),
        "miss_temp": miss_temp,
        "include_prev_targets": include_prev,
        "previous_n": previous_n,
        "previous_prop": previous_prop,
        "previous_strategy": previous_strategy,
        "n_previous_train_targets": len(prev_train_words),
        "previous_train_targets": list(prev_train_words),
        "learn_W": bool(args.learn_w),
        "learn_R": bool(args.learn_r),
        "learn_bias_network": bool(args.learn_bias_network),
        "semantle_csv": saved_cfg.get("semantle_csv"),
        "resolved_train_config": learned.resolved_train_config,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    _write_new_checkpoint(
        ckpt,
        output_dir,
        combined_items=combined_items,
        combined_words=combined_words,
        mu_all=mu_all,
        logvar_all=logvar_all,
        def_embed_lookup=def_embed_lookup,
        extend_meta=extend_meta,
        resolved_train_config=learned.resolved_train_config,
    )
    print(f"[extend] wrote new checkpoint -> {output_dir}", flush=True)

    # Release the source checkpoint before loading the new one for eval.
    release_eval_checkpoint(ckpt)

    # ── Post-training eval panels ──────────────────────────────────────────
    # The test set is the original one minus any new targets that were absorbed;
    # interp/extrap are recomputed against the EXTENDED train vocabulary (old +
    # new) so the split reflects where the test set now lies relative to the
    # subspace the new targets were added to.
    reduced_test = [w for w in orig_test_words if norm(w) not in new_norm]
    if reduced_test:
        pca_var = float(saved_cfg.get("eval_bbox_pca_var", DEFAULT_BBOX_PCA_VAR))
        new_train_emb = encode_texts_normalized(combined_words, task=task)
        reduced_test_emb = encode_texts_normalized(reduced_test, task=task)
        reduced_interp, reduced_extrap = split_interp_extrap(
            new_train_emb, reduced_test_emb, reduced_test, pca_var=pca_var
        )
    else:
        reduced_interp, reduced_extrap = [], []

    print("[extend] running post-training eval panels...", flush=True)
    results = _run_post_eval(
        output_dir,
        model_name=model_name,
        layer=layer,
        low_rank_dim=low_rank_dim,
        cache_dir=cache_dir,
        torch_dtype=args.torch_dtype,
        reduced_interp=reduced_interp,
        reduced_extrap=reduced_extrap,
        n_samples=n_samples,
        top_p=top_p,
        eval_batch_size=eval_batch_size,
        max_new_tokens=max_new_tokens,
        tau=tau,
        seed=seed,
        miss_temp=miss_temp,
    )

    hits = results.get("hits") or {}
    after_old = hits.get("original_train") or {}
    after_new = hits.get("original_test_misses") or {}
    after_rem = hits.get("original_test_hits") or {}
    rem_words = list((after_rem.get("recall") or {}).keys()) or reduced_test

    recon_old_panel = results.get("recon_old") or {}
    recon_new_panel = results.get("recon_new") or {}
    recon_test_panel = results.get("recon_test") or {}

    headline = {
        "original_train": original_train_result(
            words=old_words,
            before_recall=before_old_recall,
            before_greedy=before_old_greedy,
            after_recall=after_old.get("recall") or {},
            after_greedy=after_old.get("greedy"),
            after_embed_sim=recon_old_panel.get("embed_sim"),
            after_embed_sim_gte_tau=recon_old_panel.get("embed_sim_gte_tau"),
        ),
        "original_test_misses": original_test_misses_result(
            words=new_words,
            before_recall=before_new_recall,
            before_greedy=before_new_greedy,
            after_recall=after_new.get("recall") or {},
            after_greedy=after_new.get("greedy"),
            after_embed_sim=recon_new_panel.get("embed_sim"),
            after_embed_sim_gte_tau=recon_new_panel.get("embed_sim_gte_tau"),
        ),
        "original_test_hits": original_test_hits_result(
            words=rem_words,
            before_recall=before_remaining_recall,
            before_greedy=before_remaining_greedy,
            after_recall=after_rem.get("recall") or {},
            after_greedy=after_rem.get("greedy"),
            after_embed_sim=recon_test_panel.get("embed_sim"),
            after_embed_sim_gte_tau=recon_test_panel.get("embed_sim_gte_tau"),
            interp_after=cell_from_prefixed_panel(
                recon_test_panel, "interp", miss_temp
            )
            if recon_test_panel
            else None,
            extrap_after=cell_from_prefixed_panel(
                recon_test_panel, "extrap", miss_temp
            )
            if recon_test_panel
            else None,
        ),
    }

    eval_history = list(learned.eval_history or [])
    final_eval = eval_history[-1] if eval_history else {}
    learn_config_sidecar = {
        **learned.resolved_train_config,
        "loss_config": learned.loss_config,
        "steps": learned.steps,
        "mean_train_loss": (
            float(learned.mean_train_loss)
            if isinstance(learned.mean_train_loss, (int, float))
            and np.isfinite(learned.mean_train_loss)
            else None
        ),
        "stopped_early": bool(learned.stopped_early),
    }
    artifacts = write_extend_sidecars(
        output_dir,
        new_words=new_words,
        new_targets_defs=new_targets_defs,
        eval_history=eval_history,
        learn_config=learn_config_sidecar,
    )

    bias_mode = (
        "bias_network"
        if args.learn_bias_network
        else (saved_cfg.get("bias_learning_mode") or "direct_bias")
    )
    summary = build_extend_summary(
        paths={
            "source_checkpoint": os.path.abspath(args.result_dir),
            "extended_checkpoint": os.path.abspath(output_dir),
        },
        question={
            "goal": EXTEND_QUESTION_GOAL,
            "miss_temp": miss_temp,
            "n_samples": n_samples,
        },
        setup={
            "task": task,
            "source": source,
            "n_original_train": len(old_words),
            "n_original_test_misses": len(new_words),
            "n_original_test_hits": len(rem_words),
            "baseline_train_recall_at_n": baseline_train_recall_at_n,
            "baseline_test_recall_at_n": baseline_test_recall_at_n,
            "n_train_misses_detected": n_train_misses_detected,
            "n_test_misses_detected": n_test_misses_detected,
            "also_absorb_train_misses": bool(args.also_absorb_train_misses),
            "include_prev_targets": include_prev,
            "previous_n": previous_n,
            "previous_prop": previous_prop,
            "previous_strategy": previous_strategy,
            "n_previous_train_targets": len(prev_train_words),
            "learn_W": bool(args.learn_w),
            "learn_R": bool(args.learn_r),
            "learn_bias_network": bool(args.learn_bias_network),
            "bias_learning_mode": bias_mode,
        },
        training={
            "epochs": int(
                learned.resolved_train_config.get("epochs")
                or args.epochs
                or 0
            ),
            "steps": int(learned.steps),
            "mean_train_loss": (
                float(learned.mean_train_loss)
                if isinstance(learned.mean_train_loss, (int, float))
                and np.isfinite(learned.mean_train_loss)
                else None
            ),
            "stopped_early": bool(learned.stopped_early),
            "final_eval_avg_embed_sim": (
                float(final_eval["avg_embed_sim"])
                if "avg_embed_sim" in final_eval
                else None
            ),
            "final_eval_min_embed_sim": (
                float(final_eval["min_embed_sim"])
                if "min_embed_sim" in final_eval
                else None
            ),
        },
        results=headline,
        artifacts=artifacts,
    )

    if live_wandb_run is not None:
        _log_extend_to_wandb(
            summary,
            eval_history,
            args=args,
            output_dir=output_dir,
            wandb_config={
                "result_dir": os.path.abspath(args.result_dir),
                **{k: extend_meta[k] for k in (
                    "include_prev_targets",
                    "previous_n",
                    "previous_prop",
                    "previous_strategy",
                    "n_previous_train_targets",
                    "learn_W",
                    "learn_R",
                    "learn_bias_network",
                    "miss_temp",
                    "unlearnt_source",
                ) if k in extend_meta},
            },
            raw_panels={
                "recon": results.get("recon"),
                "recon_old": results.get("recon_old"),
                "recon_new": results.get("recon_new"),
                "recon_test": results.get("recon_test"),
            },
            run=live_wandb_run,
            log_curves=False,
        )

    _write_summary(summary, output_dir, args.result_dir)
    print(f"[extend] done. {headline_console_line(headline)}", flush=True)


if __name__ == "__main__":
    main()
