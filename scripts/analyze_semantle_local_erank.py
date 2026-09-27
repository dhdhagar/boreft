#!/usr/bin/env python3
"""Local and domain semantic eRank on Semantle checkpoints.

Local (per-target, own-mean centered) bags in the DIST/RECON Qwen space:

  teacher        frozen LM + definition (SDPO teacher), T=1
  mu             T=1 decode at the learned mean bias
  posterior      greedy (T=0) decode at codes drawn from N(μ, σ²)
                 (no-VAE: N(μ, I) neighborhood around the point estimate)
  posterior_t1   T=1 decode at those same codes

Each local bag also records **teacher alignment**: mean cosine of the bag to
that target's teacher centroid. Defaults: 1000 training targets × 8 samples.
The local paper figure is the 2×2 of distillation × VAE. Reconstruction-off
(ce0) is also computed by default and appears in the JSON / domain plot.

Domain (total learned semantic breadth): Sobol draws over that checkpoint's
μ bounding box, 8 T=1 decodes per point, then Roy–Vetterli eRank of each
replicate's pooled embeddings (common-mean). The domain figure is mean ± sd
across those replicates.

    python scripts/analyze_semantle_local_erank.py
    python scripts/analyze_semantle_local_erank.py --n-targets 8 --n-samples 4 --n-sobol 16
    python scripts/analyze_semantle_local_erank.py --from-json data/semantle/analysis/local_semantic_erank.json
    sbatch scripts/analyze_semantle_local_erank.sh
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Optional, Sequence

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
SCRIPTS_ROOT = os.path.dirname(os.path.abspath(__file__))
for _path in (SRC_ROOT, SCRIPTS_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import analyze_semantle_train_embed_rank as ter  # noqa: E402
from boreft.bo.plotting import apply_bo_axes_style, bo_rc_params, save_plot  # noqa: E402
from boreft.search_wandb import add_wandb_cli, maybe_log_local_erank  # noqa: E402
from boreft.task_config import task_embedding_model  # noqa: E402

TASK = "semantle"
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "data", "semantle", "analysis", "breadth_n1000_s8")
DEFAULT_N_TARGETS = 1000
DEFAULT_N_SAMPLES = 8
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 32
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_CACHE_DIR = None
DEFAULT_N_SOBOL = 1000
DEFAULT_N_SOBOL_SAMPLES = 8
EPS = 1e-12

DEFAULT_CONDITIONS: tuple[dict[str, str], ...] = (
    {
        "name": "canonical",
        "wandb_id": "jn0knp44",
        "output_dir": os.path.join(REPO_ROOT, "outputs", "1784053292"),
    },
    {
        "name": "sdpo0",
        "wandb_id": "yt8pnmyh",
        "output_dir": os.path.join(REPO_ROOT, "outputs", "1788622157"),
    },
    {
        "name": "novae",
        "wandb_id": "vz6h0vba",
        "output_dir": os.path.join(REPO_ROOT, "outputs", "1789101817"),
    },
    {
        "name": "sdpo0_novae",
        "wandb_id": "gsuy3cnv",
        "output_dir": os.path.join(REPO_ROOT, "outputs", "1789021668"),
    },
    {
        "name": "ce0",
        "wandb_id": "bh4ceow2",
        "output_dir": os.path.join(REPO_ROOT, "outputs", "1789136583"),
    },
)

SET_ORDER = ("teacher", "mu", "posterior", "posterior_t1")
SET_LABELS = {
    "teacher": "Teacher",
    "mu": "Decode at μ (T=1)",
    "posterior": "N(μ, σ²) T=0",
    "posterior_t1": "N(μ, σ²) T=1",
}
CONDITION_LABELS = {
    "canonical": "BOReFT",
    "sdpo0": "w/o self-distillation",
    "novae": "w/o variational training",
    "sdpo0_novae": "w/o self-distillation and variational training",
    "ce0": "w/o reconstruction",
}
CONDITION_COLORS = {
    "canonical": "#0072B2",
    "sdpo0": "#D55E00",
    "novae": "#009E73",
    "sdpo0_novae": "#CC79A7",
    "ce0": "#E69F00",
}
LOCAL_FIGURE_CONDITIONS = ("canonical", "sdpo0", "novae")  # , "sdpo0_novae")


def condition_label(name: str) -> str:
    key = str(name or "").strip().lower()
    if key in CONDITION_LABELS:
        return CONDITION_LABELS[key]
    if key == "canonical":
        return "BOReFT"
    return str(name)


def remap_repo_path(path: str) -> str:
    """Prefer an existing file; otherwise map ``.../data/...`` onto this repo."""
    if os.path.isfile(path):
        return path
    parts = path.replace("\\", "/").split("/")
    if "data" not in parts:
        return path
    candidate = os.path.join(REPO_ROOT, *parts[parts.index("data") :])
    return candidate if os.path.isfile(candidate) else path


def local_effective_rank(emb: np.ndarray) -> float:
    """Roy–Vetterli eRank of a bag, centered at that bag's mean.

    Identical samples (greedy repeats) would collapse to the zero matrix after
    centering; those bags are reported as rank 1 rather than 0. For Sobol-in-box
    pooled embeddings this is the common-mean (total learned breadth) rank.
    """
    x = np.asarray(emb, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        return 0.0
    if x.shape[0] == 1:
        return 1.0
    centered = x - x.mean(axis=0, keepdims=True)
    if float(np.linalg.norm(centered)) < EPS:
        return 1.0
    sigma = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    return ter.roy_vetterli_effective_rank(sigma)


def normalized_erank(erank: float, n: int, dim: int) -> float:
    denom = min(max(n - 1, 1), max(dim, 1))
    return float(erank) / float(denom)


def summarize_eranks(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
        }
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1) if arr.size > 1 else 0.0),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def unit_centroid(emb: np.ndarray) -> np.ndarray:
    """Mean embedding, renormalized to unit length."""
    x = np.asarray(emb, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    centroid = x.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm < EPS:
        return centroid
    return centroid / norm


def mean_cosine_to_centroid(emb: np.ndarray, centroid: np.ndarray) -> float:
    """Mean cosine of rows to a centroid (centroid is re-normalized)."""
    x = np.asarray(emb, dtype=np.float64)
    center = np.asarray(centroid, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] == 0 or center.size == 0 or x.shape[1] != center.size:
        return float("nan")
    center_norm = float(np.linalg.norm(center))
    if center_norm < EPS:
        return float("nan")
    center = center / center_norm
    row_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(row_norm, EPS)
    return float(np.mean(x @ center))


def teacher_alignment(student_emb: np.ndarray, teacher_emb: np.ndarray) -> float:
    """Mean cosine of a student bag to the teacher centroid."""
    return mean_cosine_to_centroid(student_emb, unit_centroid(teacher_emb))


def sample_items(items: Sequence[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    if n <= 0:
        raise ValueError(f"n_targets must be positive, got {n}")
    pool = list(items)
    if n >= len(pool):
        return pool
    return random.Random(seed).sample(pool, n)


def item_word(item: dict[str, Any]) -> str:
    return str(item.get("word") or item["target"]).strip()


def item_word_id(item: dict[str, Any]) -> int:
    if "id" not in item or item["id"] is None:
        raise ValueError(f"items.json row is missing id: {item!r}")
    return int(item["id"])


def _saved_float(saved: dict[str, Any], key: str, default: float) -> float:
    value = saved.get(key)
    if value is None:
        return float(default)
    return float(value)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _token_ids(raw: Any) -> list[int]:
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        raw = raw[0]
    return [int(t) for t in raw]


def _norm_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def n_unique(texts: Sequence[str]) -> int:
    return len({_norm_text(t) for t in texts if _norm_text(t)})


def domain_replicate_eranks(
    embeddings: np.ndarray,
    n_points: int,
    n_samples: int,
) -> np.ndarray:
    """Pooled eRank of each T=1 replicate across the same Sobol grid.

    ``embeddings`` is laid out as ``np.repeat(points, n_samples, axis=0)``:
    ``[p0s0, p0s1, ..., p1s0, ...]``. Replicate ``k`` is sample ``k`` at every
    Sobol point, so the 8 bars' error bars are decode-temperature variability
    at a fixed domain cover.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    expected = int(n_points) * int(n_samples)
    if x.ndim != 2 or x.shape[0] != expected:
        raise ValueError(
            f"expected embeddings shape [n_points * n_samples, dim] = "
            f"[{expected}, d], got {x.shape}"
        )
    if n_samples == 1:
        return np.asarray([local_effective_rank(x)], dtype=np.float64)
    stacked = x.reshape(int(n_points), int(n_samples), x.shape[1])
    return np.asarray(
        [local_effective_rank(stacked[:, k, :]) for k in range(int(n_samples))],
        dtype=np.float64,
    )


def domain_erank_stats(row: dict[str, Any]) -> tuple[float, float]:
    """Return ``(mean_erank, std_erank)`` for the domain bar chart."""
    replicates = row.get("erank_replicates")
    values: list[float] = []
    if isinstance(replicates, (list, tuple)):
        values = [
            float(v)
            for v in replicates
            if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)
        ]
    if values:
        arr = np.asarray(values, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        return mean, std
    mean = float(row["erank"]) if isinstance(row.get("erank"), (int, float)) else float("nan")
    raw_std = row.get("erank_std")
    std = (
        float(raw_std)
        if isinstance(raw_std, (int, float)) and np.isfinite(raw_std)
        else 0.0
    )
    return mean, std


def posterior_codes(
    mu: np.ndarray,
    logvar: Optional[np.ndarray],
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``n_samples`` codes from N(μ, diag(σ²)).

    ``logvar=None`` (point-estimate / no-VAE models) draws from N(μ, I) so
    the bag is a unit-variance neighborhood around the learned target.
    """
    mean = np.asarray(mu, dtype=np.float64)
    if mean.ndim != 1:
        raise ValueError(f"mu must be 1D, got {mean.shape}")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if logvar is None:
        std = np.ones(mean.shape[0], dtype=np.float64)
    else:
        std = np.exp(0.5 * np.asarray(logvar, dtype=np.float64))
    noise = rng.standard_normal(size=(n_samples, mean.shape[0]))
    return mean[None, :] + std[None, :] * noise


def conditions_for_local_figure(
    conditions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the named local-figure conditions. Falls back to all rows if none match."""
    keep = [
        row
        for row in conditions
        if str(row.get("name") or "").strip().lower() in LOCAL_FIGURE_CONDITIONS
    ]
    return keep if keep else list(conditions)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            **bo_rc_params(),
            "axes.labelsize": 14,
            "axes.titlesize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 120,
            "savefig.dpi": 160,
        }
    )
    return plt


def _metric_values(
    per_target: Sequence[dict[str, Any]],
    set_name: str,
    key: str,
) -> list[float]:
    values: list[float] = []
    for row in per_target:
        block = row.get(set_name)
        if not isinstance(block, dict):
            continue
        value = block.get(key, float("nan"))
        if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
            values.append(float(value))
    return values


def _paint_boxplot(bp, color: str) -> None:
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor("0.2")
    for key in ("whiskers", "caps"):
        for line in bp[key]:
            line.set_color("0.2")
    for line in bp["medians"]:
        line.set_color("0.1")
        line.set_linewidth(1.4)


def _draw_grouped_boxes(
    axis,
    conditions: Sequence[dict[str, Any]],
    *,
    set_order: Sequence[str],
    value_key: str,
) -> None:
    present = [row["name"] for row in conditions]
    n_cond = len(present)
    width = min(0.22, 0.7 / max(n_cond, 1))
    for i, name in enumerate(present):
        block = next(row for row in conditions if row["name"] == name)
        per_target = block.get("per_target") or []
        offset = (i - (n_cond - 1) / 2.0) * (width + 0.02)
        color = CONDITION_COLORS.get(name, "#333333")
        for j, set_name in enumerate(set_order):
            values = _metric_values(per_target, set_name, value_key)
            if not values:
                continue
            bp = axis.boxplot(
                [values],
                positions=[j + 1 + offset],
                widths=width,
                patch_artist=True,
                manage_ticks=False,
                showfliers=False,
            )
            _paint_boxplot(bp, color)
        axis.plot([], [], color=color, lw=8, alpha=0.55, label=condition_label(name))
    axis.set_xlim(0.4, len(set_order) + 0.6)
    axis.set_xticks(list(range(1, len(set_order) + 1)))
    axis.set_xticklabels([SET_LABELS.get(name, name) for name in set_order])


def _has_metric(
    conditions: Sequence[dict[str, Any]],
    key: str,
    set_order: Sequence[str] = SET_ORDER,
) -> bool:
    for block in conditions:
        for set_name in set_order:
            if _metric_values(block.get("per_target") or [], set_name, key):
                return True
    return False


def plot_local_erank(
    conditions: Sequence[dict[str, Any]],
    path: str,
    *,
    n_targets: int,
    n_samples: int,
) -> None:
    """Two-panel local figure: eRank and teacher-centroid alignment.

    Alignment panel is omitted when the JSON has no ``teacher_align`` values.
    """
    plt = _plt()
    shown = conditions_for_local_figure(conditions)
    show_align = _has_metric(shown, "teacher_align")
    n_axes = 2 if show_align else 1
    fig, axes = plt.subplots(1, n_axes, figsize=(7.6 * n_axes, 4.8), squeeze=False)
    erank_ax = axes[0][0]
    _draw_grouped_boxes(erank_ax, shown, set_order=SET_ORDER, value_key="erank")
    apply_bo_axes_style(erank_ax, ylabel="Local semantic eRank")
    erank_ax.set_title("Volume (own-mean eRank)")
    erank_ax.legend(loc="upper left", frameon=True)
    if show_align:
        align_ax = axes[0][1]
        _draw_grouped_boxes(
            align_ax, shown, set_order=SET_ORDER, value_key="teacher_align"
        )
        apply_bo_axes_style(align_ax, ylabel="Mean cosine to teacher centroid")
        align_ax.set_title("Teacher alignment")
        align_ax.set_ylim(0.0, 1.02)
        align_ax.legend(loc="lower left", frameon=True)
    fig.suptitle(
        f"Per-target local breadth ({n_targets} targets × {n_samples} samples)",
        y=1.02,
    )
    fig.tight_layout()
    save_plot(fig, path)
    plt.close(fig)


TRAINING_BOX_COLOR = "#7F7F7F"
LOCAL_BREADTH_GROUPS = (
    ("mu", "Learned Target"),
    ("posterior_t1", "Learned Neighborhood"),
)


def teacher_eranks_for_training_box(
    conditions: Sequence[dict[str, Any]],
) -> list[float]:
    """Single Training box: canonical teacher eRanks (fallback: first condition)."""
    by_name = {str(row.get("name") or "").strip().lower(): row for row in conditions}
    block = by_name.get("canonical") or (conditions[0] if conditions else None)
    if block is None:
        return []
    return _metric_values(block.get("per_target") or [], "teacher", "erank")


def plot_local_breadth(
    conditions: Sequence[dict[str, Any]],
    path: str,
) -> None:
    """Paper local-breadth PDF: one Training box, then μ / posterior T=1 by space.

    No title. Y-axis is per-target eRank. Always writes PDF and PNG siblings.
    """
    plt = _plt()
    shown = conditions_for_local_figure(conditions)
    present = [row["name"] for row in shown]
    n_cond = len(present)
    width = min(0.22, 0.7 / max(n_cond, 1))
    fig, axis = plt.subplots(figsize=(6.6, 4.6))
    teacher_vals = teacher_eranks_for_training_box(shown)
    if teacher_vals:
        bp = axis.boxplot(
            [teacher_vals],
            positions=[1.0],
            widths=0.34,
            patch_artist=True,
            manage_ticks=False,
            showfliers=False,
        )
        _paint_boxplot(bp, TRAINING_BOX_COLOR)
    for i, name in enumerate(present):
        block = next(row for row in shown if row["name"] == name)
        per_target = block.get("per_target") or []
        offset = (i - (n_cond - 1) / 2.0) * (width + 0.02)
        color = CONDITION_COLORS.get(name, "#333333")
        for j, (set_name, _label) in enumerate(LOCAL_BREADTH_GROUPS):
            values = _metric_values(per_target, set_name, "erank")
            if not values:
                continue
            bp = axis.boxplot(
                [values],
                positions=[j + 2 + offset],
                widths=width,
                patch_artist=True,
                manage_ticks=False,
                showfliers=False,
            )
            _paint_boxplot(bp, color)
        axis.plot([], [], color=color, lw=8, alpha=0.55, label=condition_label(name))
    axis.set_xlim(0.4, 3.6)
    axis.set_xticks([1, 2, 3])
    axis.set_xticklabels(
        ["Training"] + [label for _, label in LOCAL_BREADTH_GROUPS]
    )
    apply_bo_axes_style(axis, ylabel="Per-Target eRank")
    if present:
        handles, labels = axis.get_legend_handles_labels()
        n_methods = len(present)
        if n_methods <= 5:
            ncol = max(n_methods, 1)
        elif n_methods <= 8:
            ncol = 4
        else:
            ncol = 5
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=ncol,
            bbox_to_anchor=(0.5, 1.0),
            frameon=True,
            handlelength=2.6,
            columnspacing=1.4,
            borderaxespad=0.2,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_plot(fig, path)
    plt.close(fig)


def plot_domain_erank(
    conditions: Sequence[dict[str, Any]],
    path: str,
    *,
    n_sobol_points: int,
    n_samples: int,
) -> None:
    """Bar chart of pooled Sobol-in-box eRank (mean ± sd across T=1 replicates)."""
    plt = _plt()
    names = [str(row.get("name") or "") for row in conditions]
    means: list[float] = []
    stds: list[float] = []
    for row in conditions:
        mean, std = domain_erank_stats(row)
        means.append(mean)
        stds.append(std)
    mean_arr = np.asarray(means, dtype=np.float64)
    std_arr = np.asarray(stds, dtype=np.float64)
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(names), dtype=np.float64)
    colors = [CONDITION_COLORS.get(name, "#333333") for name in names]
    finite = np.isfinite(mean_arr)
    yerr = np.where(finite & (std_arr > 0), std_arr, 0.0)
    show_err = bool(np.any(yerr > 0))
    bar_kwargs: dict[str, Any] = dict(
        color=colors,
        edgecolor="0.2",
        width=0.62,
        alpha=0.85,
    )
    if show_err:
        bar_kwargs["yerr"] = yerr
        bar_kwargs["capsize"] = 5
        bar_kwargs["error_kw"] = {
            "ecolor": "0.2",
            "elinewidth": 1.2,
            "capthick": 1.2,
        }
    bars = axis.bar(
        x,
        np.where(finite, mean_arr, 0.0),
        **bar_kwargs,
    )
    tops = np.where(finite, mean_arr + yerr, 0.0)
    span = float(np.nanmax(tops)) if np.any(np.isfinite(tops)) else 1.0
    span = max(span, 1.0)
    # Two-line labels sit above the error-bar caps; leave a gap under the spine.
    headroom = max(0.32 * span, 2.5)
    axis.set_ylim(0.0, span + headroom)
    for bar, row, mean, std, top in zip(bars, conditions, means, stds, tops):
        if not np.isfinite(mean):
            continue
        n_u = row.get("n_unique")
        if std > 0:
            label = f"{mean:.1f}±{std:.1f}"
        else:
            label = f"{mean:.1f}"
        if isinstance(n_u, (int, float)):
            label = f"{label}\n({int(n_u)} unique)"
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(top) + 0.04 * span,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            clip_on=False,
            linespacing=1.15,
        )
    axis.set_xticks(x)
    axis.set_xticklabels([condition_label(name) for name in names])
    apply_bo_axes_style(axis, ylabel="Pooled semantic eRank")
    axis.set_title(
        f"Total learned semantic breadth "
        f"(Sobol {n_sobol_points} × {n_samples} T=1)",
        pad=12,
    )
    fig.tight_layout()
    save_plot(fig, path)
    plt.close(fig)


def _definitions_path(saved: dict[str, Any]) -> str:
    from boreft.text_similarity import default_definitions_path

    raw = (
        saved.get("sdpo_definitions_path")
        or saved.get("definitions_path")
        or default_definitions_path(TASK)
    )
    return remap_repo_path(os.path.abspath(os.path.expanduser(str(raw))))


def _teacher_prompts(
    words: Sequence[str],
    saved: dict[str, Any],
    tokenizer,
) -> list[list[int]]:
    from boreft.data_utils import chat_prompt, tokenize_model_text
    from boreft.task_config import sdpo_teacher_instruction
    from boreft.text_similarity import (
        definition_text_for_cfg,
        load_raw_definitions,
    )

    def_path = _definitions_path(saved)
    raw_defs = load_raw_definitions(def_path)
    use_chat = bool(saved.get("use_chat_template"))
    prompts: list[list[int]] = []
    missing: list[str] = []
    for word in words:
        if word not in raw_defs:
            missing.append(word)
            prompts.append([])
            continue
        text = sdpo_teacher_instruction(
            TASK,
            definition_text_for_cfg(saved, word, raw_defs[word], require_lookup=False),
            use_chat_template=use_chat,
        )
        if use_chat:
            text = chat_prompt(tokenizer, text)
        ids = _token_ids(
            tokenize_model_text(
                tokenizer, text, from_chat_template=use_chat
            )["input_ids"]
        )
        prompts.append(ids)
    if missing:
        preview = ", ".join(missing[:5])
        extra = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise ValueError(
            f"{len(missing)} sampled words missing SDPO definitions in "
            f"{def_path} (e.g. {preview}{extra})"
        )
    return prompts


def _decode_ids(tokenizer, token_ids: Sequence[int]) -> str:
    ids = [int(t) for t in token_ids]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None and eos_id in ids:
        ids = ids[: ids.index(eos_id)]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def _sample_teacher(
    base_model,
    tokenizer,
    prompt_ids: Sequence[int],
    n_samples: int,
    *,
    device,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    from boreft.pyreft.sdpo import (
        SDPOConfig,
        _left_pad,
        _teacher_generate,
        _trim_generated,
    )

    cfg = SDPOConfig(
        sample_temperature=temperature,
        sample_top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    texts: list[str] = []
    prompt = list(prompt_ids)
    for start in range(0, n_samples, batch_size):
        chunk = min(batch_size, n_samples - start)
        prompts = [prompt] * chunk
        input_ids, attn = _left_pad(prompts, tokenizer.pad_token_id, device)
        prefix = int(input_ids.shape[1])
        output_ids = _teacher_generate(base_model, input_ids, attn, cfg, tokenizer)
        for row in range(output_ids.shape[0]):
            y = _trim_generated(
                output_ids[row, prefix:].tolist(),
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
            )
            texts.append(_decode_ids(tokenizer, y) if y else "")
    return texts


def _mu_logvar_for_ids(intervention, word_ids: Sequence[int], device) -> tuple[np.ndarray, Optional[np.ndarray]]:
    import torch

    ids = torch.tensor([int(w) for w in word_ids], dtype=torch.long, device=device)
    if hasattr(intervention, "get_bias_mu_logvar"):
        mu, logvar = intervention.get_bias_mu_logvar(ids)
        return (
            mu.detach().float().cpu().numpy(),
            logvar.detach().float().cpu().numpy(),
        )
    mu = intervention.get_bias_mu(ids)
    return mu.detach().float().cpu().numpy(), None


def _erank_row(texts: Sequence[str], embeddings: np.ndarray) -> dict[str, Any]:
    n = len(texts)
    dim = int(embeddings.shape[1]) if embeddings.size else 0
    erank = local_effective_rank(embeddings) if n else 0.0
    return {
        "erank": float(erank),
        "erank_normalized": normalized_erank(erank, n, dim),
        "n": n,
        "n_unique": n_unique(texts),
        "embed_dim": dim,
    }


def _slice_embeddings(emb: np.ndarray, start: int, stop: int) -> np.ndarray:
    if not emb.size:
        return np.zeros((0, 0), dtype=np.float64)
    return emb[start:stop]


def run_domain_erank(
    ckpt,
    saved: dict[str, Any],
    *,
    name: str,
    wandb_id: str,
    output_dir: str,
    n_sobol_points: int,
    n_sobol_samples: int,
    seed: int,
    batch_size: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    """Sobol-in-box T=1 decodes → pooled (common-mean) semantic eRank."""
    import torch

    from boreft.eval.decode_utils import _decode_common
    from boreft.eval.semantle import (
        DEFAULT_MAX_NEW_TOKENS,
        generate_texts_multi_batch,
        sample_sobol_bias_vectors,
    )
    from boreft.text_similarity import encode_texts_normalized

    if n_sobol_points < 1:
        raise ValueError(f"n_sobol_points must be positive, got {n_sobol_points}")
    if n_sobol_samples < 1:
        raise ValueError(f"n_sobol_samples must be positive, got {n_sobol_samples}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    words = [item_word(it) for it in ckpt.items]
    word_ids = [item_word_id(it) for it in ckpt.items]
    sampled_bs, _, _, _ = sample_sobol_bias_vectors(
        ckpt.reft_model,
        words,
        n_sobol_points,
        word_ids=word_ids,
        seed=seed,
    )
    decode_kw = dict(
        max_new_tokens=int(saved.get("full_eval_max_new_tokens") or DEFAULT_MAX_NEW_TOKENS),
        **_decode_common(ckpt),
    )
    codes = (
        np.repeat(np.asarray(sampled_bs, dtype=np.float32), n_sobol_samples, axis=0)
        if n_sobol_samples > 1
        else np.asarray(sampled_bs, dtype=np.float32)
    )
    print(
        f"[domain_erank] [{name}] Sobol {len(sampled_bs)} × {n_sobol_samples} T={temperature}",
        flush=True,
    )
    texts: list[str] = []
    for start in range(0, len(codes), batch_size):
        chunk = codes[start : start + batch_size]
        texts.extend(
            generate_texts_multi_batch(
                ckpt.reft_model,
                ckpt.tokenizer,
                ckpt.prompt,
                [row.astype(np.float32) for row in chunk],
                use_sample=True,
                temperature=temperature,
                top_p=top_p,
                use_stochastic_intervention=False,
                **decode_kw,
            )
        )
        done = min(start + batch_size, len(codes))
        if done % max(batch_size * 4, 1) == 0 or done == len(codes):
            print(f"[domain_erank] [{name}] {done}/{len(codes)} decodes", flush=True)
    embeddings = (
        encode_texts_normalized(texts, task=TASK) if texts else np.zeros((0, 0))
    )
    pooled = _erank_row(texts, embeddings)
    n_points = int(len(sampled_bs))
    replicates = (
        domain_replicate_eranks(embeddings, n_points, n_sobol_samples)
        if embeddings.size
        else np.zeros(0, dtype=np.float64)
    )
    mean_erank = float(np.mean(replicates)) if replicates.size else float(pooled["erank"])
    std_erank = (
        float(np.std(replicates, ddof=1)) if replicates.size > 1 else 0.0
    )
    print(
        f"[domain_erank] [{name}] erank={mean_erank:.3f}±{std_erank:.3f}  "
        f"pooled={pooled['erank']:.3f}  n_unique={pooled['n_unique']}/{pooled['n']}",
        flush=True,
    )
    return {
        "name": name,
        "wandb_id": wandb_id,
        "output_dir": output_dir,
        "bias_type": saved.get("bias_type"),
        "lambda_sdpo": saved.get("lambda_sdpo"),
        "add_bias_network": saved.get("add_bias_network"),
        "n_train": len(ckpt.items),
        "n_sobol_points": n_points,
        "n_samples": int(n_sobol_samples),
        "temperature": float(temperature),
        "erank": mean_erank,
        "erank_std": std_erank,
        "erank_replicates": [float(v) for v in replicates],
        "erank_pooled": float(pooled["erank"]),
        "erank_normalized": normalized_erank(
            mean_erank, n_points, int(pooled["embed_dim"])
        ),
        "n": int(pooled["n"]),
        "n_unique": int(pooled["n_unique"]),
        "embed_dim": int(pooled["embed_dim"]),
    }


def run_local_erank(
    ckpt,
    saved: dict[str, Any],
    *,
    name: str,
    wandb_id: str,
    output_dir: str,
    n_targets: int,
    n_samples: int,
    seed: int,
    batch_size: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    import torch

    from boreft.eval.decode_utils import _decode_common
    from boreft.eval.semantle import (
        DEFAULT_MAX_NEW_TOKENS,
        _get_intervention,
        generate_texts_batch,
        generate_texts_multi_batch,
    )
    from boreft.text_similarity import encode_texts_normalized

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    items = sample_items(ckpt.items, n_targets, seed)
    words = [item_word(it) for it in items]
    word_ids = [item_word_id(it) for it in items]
    print(
        f"[local_erank] [{name}] {output_dir}  "
        f"n_targets={len(words)}/{len(ckpt.items)}  n_samples={n_samples}  "
        f"bias_type={saved.get('bias_type', 'vae')}  "
        f"lambda_sdpo={saved.get('lambda_sdpo', 0.0)}",
        flush=True,
    )

    max_new_tokens = int(saved.get("sdpo_max_new_tokens") or DEFAULT_MAX_NEW_TOKENS)
    decode_kw = dict(
        max_new_tokens=int(saved.get("full_eval_max_new_tokens") or DEFAULT_MAX_NEW_TOKENS),
        **_decode_common(ckpt),
    )
    teacher_prompts = _teacher_prompts(words, saved, ckpt.tokenizer)
    iv = _get_intervention(ckpt.reft_model)
    device = ckpt.reft_model.get_device()
    mu_all, logvar_all = _mu_logvar_for_ids(iv, word_ids, device)
    rng = np.random.default_rng(seed)

    def _decode_codes(codes: np.ndarray, *, use_sample: bool) -> list[str]:
        texts: list[str] = []
        for start in range(0, n_samples, batch_size):
            chunk = codes[start : start + batch_size]
            texts.extend(
                generate_texts_multi_batch(
                    ckpt.reft_model,
                    ckpt.tokenizer,
                    ckpt.prompt,
                    [row.astype(np.float32) for row in chunk],
                    use_sample=use_sample,
                    temperature=temperature,
                    top_p=top_p,
                    **decode_kw,
                )
            )
        return texts

    teacher_texts: list[list[str]] = []
    mu_texts: list[list[str]] = []
    posterior_texts: list[list[str]] = []
    posterior_t1_texts: list[list[str]] = []
    base_model = ckpt.reft_model.model

    for i, word in enumerate(words):
        print(f"[local_erank] [{name}] {i + 1}/{len(words)} {word!r}", flush=True)
        teacher_texts.append(
            _sample_teacher(
                base_model,
                ckpt.tokenizer,
                teacher_prompts[i],
                n_samples,
                device=device,
                temperature=_saved_float(saved, "sdpo_sample_temperature", temperature),
                top_p=_saved_float(saved, "sdpo_sample_top_p", top_p),
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
            )
        )
        mu_runs: list[str] = []
        for start in range(0, n_samples, batch_size):
            chunk = min(batch_size, n_samples - start)
            mu_runs.extend(
                generate_texts_batch(
                    ckpt.reft_model,
                    ckpt.tokenizer,
                    ckpt.prompt,
                    word_ids[i],
                    n_samples=chunk,
                    use_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    use_stochastic_intervention=False,
                    **decode_kw,
                )
            )
        mu_texts.append(mu_runs)
        logvar_row = None if logvar_all is None else logvar_all[i]
        codes = posterior_codes(mu_all[i], logvar_row, n_samples, rng)
        posterior_texts.append(_decode_codes(codes, use_sample=False))
        posterior_t1_texts.append(_decode_codes(codes, use_sample=True))

    bags = {
        "teacher": teacher_texts,
        "mu": mu_texts,
        "posterior": posterior_texts,
        "posterior_t1": posterior_t1_texts,
    }
    embeddings = {
        set_name: (
            encode_texts_normalized([t for bag in texts for t in bag], task=TASK)
            if any(texts)
            else np.zeros((0, 0))
        )
        for set_name, texts in bags.items()
    }

    per_target: list[dict[str, Any]] = []
    for i, word in enumerate(words):
        sl = slice(i * n_samples, (i + 1) * n_samples)
        teacher_emb = _slice_embeddings(embeddings["teacher"], sl.start, sl.stop)
        row: dict[str, Any] = {"word": word, "word_id": word_ids[i]}
        for set_name, texts in bags.items():
            emb = _slice_embeddings(embeddings[set_name], sl.start, sl.stop)
            block = _erank_row(texts[i], emb)
            block["teacher_align"] = float(teacher_alignment(emb, teacher_emb))
            row[set_name] = block
        per_target.append(row)
        print(
            f"[local_erank] [{name}] {word!r}  "
            + "  ".join(
                f"{set_name}={row[set_name]['erank']:.3f}"
                f"/{row[set_name]['teacher_align']:.3f}"
                for set_name in SET_ORDER
            ),
            flush=True,
        )

    summary = {
        set_name: {
            "erank": summarize_eranks([row[set_name]["erank"] for row in per_target]),
            "teacher_align": summarize_eranks(
                [row[set_name]["teacher_align"] for row in per_target]
            ),
        }
        for set_name in SET_ORDER
    }
    return {
        "name": name,
        "wandb_id": wandb_id,
        "output_dir": output_dir,
        "bias_type": saved.get("bias_type"),
        "lambda_sdpo": saved.get("lambda_sdpo"),
        "variance": saved.get("variance"),
        "add_bias_network": saved.get("add_bias_network"),
        "n_train": len(ckpt.items),
        "n_targets": len(words),
        "n_samples": n_samples,
        "has_posterior_variance": logvar_all is not None,
        "summary": summary,
        "per_target": per_target,
        "words": words,
    }


def run_condition(
    *,
    name: str,
    output_dir: str,
    wandb_id: str,
    n_targets: int,
    n_samples: int,
    seed: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    cache_dir: Optional[str] = None,
    include_local: bool = True,
    include_domain: bool = True,
    n_sobol_points: int = DEFAULT_N_SOBOL,
    n_sobol_samples: int = DEFAULT_N_SOBOL_SAMPLES,
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    from boreft.data_utils import load_merged_run_config
    from boreft.eval.semantle import load_eval_checkpoint, release_eval_checkpoint

    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    saved = load_merged_run_config(output_dir)
    model_name = saved.get("model_name")
    if not model_name:
        raise ValueError(f"{output_dir}: training_config.json is missing model_name")
    layer = int(saved.get("layer", 13))
    rank = int(saved.get("low_rank_dim", 64))
    ckpt = load_eval_checkpoint(
        output_dir,
        model_name,
        layer,
        rank,
        cache_dir or saved.get("cache_dir") or DEFAULT_CACHE_DIR,
    )
    local_row: Optional[dict[str, Any]] = None
    domain_row: Optional[dict[str, Any]] = None
    try:
        if include_local:
            local_row = run_local_erank(
                ckpt,
                saved,
                name=name,
                wandb_id=wandb_id,
                output_dir=output_dir,
                n_targets=n_targets,
                n_samples=n_samples,
                seed=seed,
                batch_size=batch_size,
                temperature=temperature,
                top_p=top_p,
            )
        if include_domain:
            domain_row = run_domain_erank(
                ckpt,
                saved,
                name=name,
                wandb_id=wandb_id,
                output_dir=output_dir,
                n_sobol_points=n_sobol_points,
                n_sobol_samples=n_sobol_samples,
                seed=seed,
                batch_size=batch_size,
                temperature=temperature,
                top_p=top_p,
            )
    finally:
        release_eval_checkpoint(ckpt)
    return local_row, domain_row


def parse_condition(raw: str) -> dict[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"condition must be name=path, got {raw!r}"
        )
    name, path = raw.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError(f"condition must be name=path, got {raw!r}")
    known = {row["name"]: row for row in DEFAULT_CONDITIONS}
    wandb_id = known[name]["wandb_id"] if name in known else ""
    return {"name": name, "wandb_id": wandb_id, "output_dir": path}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--condition",
        action="append",
        type=parse_condition,
        default=None,
        help="name=output_dir (repeatable). Default: canonical, sdpo0, novae, ce0.",
    )
    p.add_argument("--n-targets", type=int, default=DEFAULT_N_TARGETS)
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p.add_argument(
        "--n-sobol",
        type=int,
        default=DEFAULT_N_SOBOL,
        help="Sobol points over the μ bounding box (default: 1000).",
    )
    p.add_argument(
        "--n-sobol-samples",
        type=int,
        default=DEFAULT_N_SOBOL_SAMPLES,
        help="T=1 decodes per Sobol point (default: 8).",
    )
    p.add_argument(
        "--skip-local",
        action="store_true",
        help="Skip per-target teacher / μ / posterior bags.",
    )
    p.add_argument(
        "--skip-sobol",
        action="store_true",
        help="Skip Sobol-in-box pooled eRank.",
    )
    p.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="HuggingFace cache for the Llama checkpoint "
        f"(default: {DEFAULT_CACHE_DIR}).",
    )
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--from-json",
        action="append",
        default=None,
        help="Replot an existing local_semantic_erank.json and/or "
        "domain_semantic_erank.json (repeatable; no GPU / decode).",
    )
    add_wandb_cli(p)
    return p.parse_args(argv)


def _summary_block(stats: Any, metric: str) -> Optional[dict[str, Any]]:
    if not isinstance(stats, dict):
        return None
    nested = stats.get(metric)
    if isinstance(nested, dict) and "mean" in nested:
        return nested
    if metric == "erank" and "mean" in stats:
        return stats
    return None


def _print_local_summary(conditions: Sequence[dict[str, Any]]) -> None:
    for row in conditions:
        print(f"[local_erank] {row['name']}", flush=True)
        summary = row.get("summary") or {}
        for set_name in SET_ORDER:
            stats = summary.get(set_name)
            erank = _summary_block(stats, "erank")
            align = _summary_block(stats, "teacher_align")
            if erank is None:
                continue
            line = (
                f"  {set_name:13s}  mean_erank={erank['mean']:.3f}  "
                f"median={erank['median']:.3f}  sd={erank['std']:.3f}"
            )
            if align is not None and np.isfinite(align.get("mean", float("nan"))):
                line += f"  mean_align={align['mean']:.3f}"
            print(line, flush=True)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
        handle.write("\n")


def _dump_and_log(
    *,
    out_dir: str,
    args: argparse.Namespace,
    local_payload: Optional[dict[str, Any]],
    domain_payload: Optional[dict[str, Any]],
    force_wandb: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    json_paths: list[str] = []
    if local_payload is not None:
        json_path = os.path.join(out_dir, "local_semantic_erank.json")
        plot_path = os.path.join(out_dir, "local_semantic_erank.png")
        _write_json(json_path, local_payload)
        plot_local_erank(
            local_payload["conditions"],
            plot_path,
            n_targets=int(local_payload.get("n_targets") or 0),
            n_samples=int(local_payload.get("n_samples") or 0),
        )
        breadth_pdf = os.path.join(out_dir, "local_breadth.pdf")
        plot_local_breadth(local_payload["conditions"], breadth_pdf)
        json_paths.append(json_path)
        print(f"[local_erank] wrote {json_path}", flush=True)
        print(f"[local_erank] wrote {plot_path}", flush=True)
        print(f"[local_erank] wrote {breadth_pdf}", flush=True)
        _print_local_summary(local_payload["conditions"])
    if domain_payload is not None:
        json_path = os.path.join(out_dir, "domain_semantic_erank.json")
        plot_path = os.path.join(out_dir, "domain_semantic_erank.png")
        _write_json(json_path, domain_payload)
        plot_domain_erank(
            domain_payload["conditions"],
            plot_path,
            n_sobol_points=int(domain_payload.get("n_sobol_points") or 0),
            n_samples=int(domain_payload.get("n_samples") or 0),
        )
        json_paths.append(json_path)
        print(f"[domain_erank] wrote {json_path}", flush=True)
        print(f"[domain_erank] wrote {plot_path}", flush=True)
        for row in domain_payload["conditions"]:
            mean, std = domain_erank_stats(row)
            extra = f"±{std:.3f}" if std > 0 else ""
            print(
                f"[domain_erank] {row['name']:10s}  erank={mean:.3f}{extra}  "
                f"n_unique={row['n_unique']}/{row['n']}",
                flush=True,
            )
    if not json_paths:
        return
    maybe_log_local_erank(
        out_dir,
        json_paths[0],
        json_paths=json_paths,
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        group=args.wandb_group,
        name=args.wandb_run_name,
        wandb_dir=args.wandb_dir,
        no_wandb=args.no_wandb,
        force=force_wandb,
    )


def _load_from_json_args(paths: Sequence[str]) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    local_payload: Optional[dict[str, Any]] = None
    domain_payload: Optional[dict[str, Any]] = None
    for raw in paths:
        path = os.path.abspath(os.path.expanduser(raw))
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not payload.get("conditions"):
            raise ValueError(f"{path}: missing conditions")
        if "n_sobol_points" in payload or os.path.basename(path).startswith("domain_"):
            domain_payload = payload
        else:
            local_payload = payload
    return local_payload, domain_payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.from_json:
        local_payload, domain_payload = _load_from_json_args(args.from_json)
        _dump_and_log(
            out_dir=args.out_dir,
            args=args,
            local_payload=local_payload,
            domain_payload=domain_payload,
            force_wandb=True,
        )
        return 0

    if args.skip_local and args.skip_sobol:
        raise SystemExit("nothing to do: both --skip-local and --skip-sobol")

    specs = args.condition or [dict(row) for row in DEFAULT_CONDITIONS]
    local_conditions: list[dict[str, Any]] = []
    domain_conditions: list[dict[str, Any]] = []
    for spec in specs:
        local_row, domain_row = run_condition(
            name=spec["name"],
            output_dir=spec["output_dir"],
            wandb_id=spec.get("wandb_id") or "",
            n_targets=args.n_targets,
            n_samples=args.n_samples,
            seed=args.seed,
            batch_size=args.batch_size,
            temperature=args.temperature,
            top_p=args.top_p,
            cache_dir=args.cache_dir or None,
            include_local=not args.skip_local,
            include_domain=not args.skip_sobol,
            n_sobol_points=args.n_sobol,
            n_sobol_samples=args.n_sobol_samples,
        )
        if local_row is not None:
            local_conditions.append(local_row)
        if domain_row is not None:
            domain_conditions.append(domain_row)

    local_payload = None
    if local_conditions:
        local_payload = {
            "task": TASK,
            "model": task_embedding_model(TASK),
            "n_targets": args.n_targets,
            "n_samples": args.n_samples,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "sets": list(SET_ORDER),
            "effective_rank_definition": (
                "Roy–Vetterli (EUSIPCO 2007, Def. 1) of the column-centered "
                "embedding matrix of one target's samples"
            ),
            "teacher_align_definition": (
                "Mean cosine of the bag to that target's teacher centroid "
                "(centroid = mean teacher embedding, renormalized)"
            ),
            "conditions": local_conditions,
        }
    domain_payload = None
    if domain_conditions:
        domain_payload = {
            "task": TASK,
            "model": task_embedding_model(TASK),
            "n_sobol_points": args.n_sobol,
            "n_samples": args.n_sobol_samples,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "effective_rank_definition": (
                "Roy–Vetterli (EUSIPCO 2007, Def. 1) of T=1 decode embeddings "
                "at one sample per Sobol point, column-centered at the common "
                "mean. Reported mean ± sd is over n_samples independent "
                "replicates of that cover; erank_pooled uses all samples together."
            ),
            "conditions": domain_conditions,
        }
    _dump_and_log(
        out_dir=args.out_dir,
        args=args,
        local_payload=local_payload,
        domain_payload=domain_payload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
