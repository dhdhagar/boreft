#!/usr/bin/env python3
"""
PCA scatter of per-word bias vectors for a trained BOReFT checkpoint.

Cluster colours come from definition categories when ``use_definition_embeds`` is
set in training_config.json, else from the training CSV(s) recorded there
(falls back to all *.csv under semantle_dir). Single-CSV training shows all
points with spread-prioritized, overlap-adjusted labels (adjustText).

CLI usage needs only ``--output_dir``; model shape and paths default from
``training_config.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple, TypedDict

import numpy as np
import torch

from boreft.data_utils import add_load_latest_argument, load_merged_run_config
from boreft.eval.semantle import _load_model_and_items, _get_intervention
from boreft.bias_tables import get_bias_vector
from boreft.eval.interpolate import _bias_type_label
from boreft.text_display import PLOT_LABEL_MAX, plot_label
from boreft.text_similarity import (
    default_definitions_path,
    load_category_normalized_lookup,
)

_META_CLUSTERS = frozenset({"multi-cluster", "unknown"})
_LABEL_ALL_BELOW = 40
_MAX_LABELS = 120
_LABEL_MAX_CHARS = PLOT_LABEL_MAX
_PCA_CHECKPOINT_FALLBACKS = {
    "model_name": "meta-llama/Llama-3.2-1B",
    "layer": 13,
    "low_rank_dim": 64,
    "top_k": 100,
}
_MARKERS = (
    "o",
    "s",
    "^",
    "D",
    "v",
    "P",
    "*",
    "X",
    "h",
    "<",
    ">",
    "p",
    "H",
    "d",
)
_EDGE_COLORS = ("black", "0.25", "0.55", "white")
_META_CLUSTER_STYLES: Dict[str, "ClusterStyle"] = {
    "multi-cluster": {"color": "#808080", "marker": "o", "edgecolor": "black"},
    "unknown": {"color": "#d9d9d9", "marker": "o", "edgecolor": "0.45"},
    "all": {"color": "#2196F3", "marker": "o", "edgecolor": "black"},
}


class ClusterStyle(TypedDict):
    color: str
    marker: str
    edgecolor: str


def _distinct_colors(n: int) -> List[str]:
    """Return *n* visually distinct hex colors from matplotlib qualitative maps."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    colors: List[str] = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(cmap_name)
        for i in range(cmap.N):
            colors.append(mcolors.to_hex(cmap(i)))
        if len(colors) >= n:
            break
    if len(colors) < n:
        raise ValueError(f"need {n} distinct colors but only built {len(colors)}")
    return colors[:n]


def build_cluster_style_map(clusters: Sequence[str]) -> Dict[str, ClusterStyle]:
    """Assign a unique color/marker/edge combination to each cluster label."""
    real_clusters = [c for c in clusters if c not in _META_CLUSTERS]
    colors = _distinct_colors(len(real_clusters))
    style_map: Dict[str, ClusterStyle] = {}
    for i, cluster in enumerate(real_clusters):
        style_map[cluster] = {
            "color": colors[i],
            "marker": _MARKERS[i % len(_MARKERS)],
            "edgecolor": _EDGE_COLORS[(i // len(_MARKERS)) % len(_EDGE_COLORS)],
        }
    for cluster in clusters:
        if cluster in _META_CLUSTER_STYLES:
            style_map[cluster] = dict(_META_CLUSTER_STYLES[cluster])
    return style_map


def _semantle_csv_paths(semantle_dir: str) -> List[str]:
    return sorted(
        os.path.join(semantle_dir, fn)
        for fn in os.listdir(semantle_dir)
        if fn.endswith(".csv")
    )


def resolve_pca_checkpoint_args(
    output_dir: str,
    *,
    model_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
    layer: Optional[int] = None,
    low_rank_dim: Optional[int] = None,
    semantle_dir: Optional[str] = None,
    top_k: Optional[int] = None,
    save_path: Optional[str] = None,
) -> dict[str, object]:
    """Fill unset PCA args from ``training_config.json`` in *output_dir*."""
    saved = load_merged_run_config(output_dir)
    resolved_model_name = (
        model_name
        or saved.get("model_name")
        or _PCA_CHECKPOINT_FALLBACKS["model_name"]
    )
    resolved_layer = (
        layer
        if layer is not None
        else saved.get("layer", _PCA_CHECKPOINT_FALLBACKS["layer"])
    )
    resolved_rank = (
        low_rank_dim
        if low_rank_dim is not None
        else saved.get("low_rank_dim", _PCA_CHECKPOINT_FALLBACKS["low_rank_dim"])
    )
    resolved_top_k = (
        top_k
        if top_k is not None
        else saved.get("train_top_k", _PCA_CHECKPOINT_FALLBACKS["top_k"])
    )
    resolved_save_path = save_path or os.path.join(
        output_dir, "eval", "cluster_pca.png"
    )
    return {
        "model_name": str(resolved_model_name),
        "cache_dir": cache_dir if cache_dir is not None else saved.get("cache_dir"),
        "layer": int(resolved_layer),
        "low_rank_dim": int(resolved_rank),
        "semantle_dir": (
            semantle_dir if semantle_dir is not None else saved.get("semantle_dir")
        ),
        "top_k": int(resolved_top_k),
        "save_path": str(resolved_save_path),
    }


def definitions_path_from_config(output_dir: str) -> str:
    cfg = load_merged_run_config(output_dir)
    path = cfg.get("definitions_path") or default_definitions_path(
        str(cfg.get("task", "semantle"))
    )
    return os.path.abspath(str(path))


def uses_definition_embed_categories(output_dir: str) -> bool:
    return bool(load_merged_run_config(output_dir).get("use_definition_embeds"))


def assign_category_labels(
    words: List[str], definitions_path: str
) -> Dict[str, str]:
    """Map each word to ``category_normalized`` from definitions.jsonl."""
    lookup = load_category_normalized_lookup(definitions_path)
    return {w: lookup.get(w, "unknown") for w in words}


def resolve_word_cluster_labels(
    words: List[str],
    output_dir: str,
    semantle_dir: Optional[str] = None,
    top_k: int = 100,
) -> Tuple[Dict[str, str], str]:
    """Return ``(word -> label, label_source)`` where source is ``category``, ``csv``, or ``none``."""
    if uses_definition_embed_categories(output_dir):
        path = definitions_path_from_config(output_dir)
        return assign_category_labels(words, path), "category"

    csv_paths = resolve_cluster_csv_paths(output_dir, semantle_dir)
    if csv_paths:
        return assign_clusters(words, csv_paths=csv_paths, top_k=top_k), "csv"
    return {w: "all" for w in words}, "none"


def resolve_cluster_csv_paths(
    output_dir: str,
    semantle_dir: Optional[str] = None,
) -> List[str]:
    """CSV paths for cluster assignment: checkpoint semantle_csv, else semantle_dir."""
    saved = load_merged_run_config(output_dir)
    raw = saved.get("semantle_csv")
    if raw:
        paths = [os.path.abspath(p) for p in raw if p]
        existing = [p for p in paths if os.path.isfile(p)]
        if existing:
            return sorted(existing)
        print(
            "[cluster_pca] training_config semantle_csv paths missing on disk "
            f"({len(paths)} listed) — falling back to semantle_dir",
            flush=True,
        )
    if semantle_dir:
        return _semantle_csv_paths(semantle_dir)
    return []


def assign_clusters(
    words: List[str],
    *,
    csv_paths: List[str],
    top_k: int = 100,
) -> Dict[str, str]:
    """Map each word to a CSV stem, ``multi-cluster``, or ``unknown``."""
    if not csv_paths:
        raise ValueError("assign_clusters requires at least one CSV path")

    hits: Dict[str, List[str]] = {w: [] for w in words}
    for path in csv_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append((row["Word"].strip(), float(row["Similarity"])))
        rows.sort(key=lambda x: -x[1])
        top_words = {w for w, _ in rows[:top_k]}
        for w in words:
            if w in top_words:
                hits[w].append(stem)

    labels: Dict[str, str] = {}
    for w in words:
        if not hits[w]:
            labels[w] = "unknown"
        elif len(hits[w]) == 1:
            labels[w] = hits[w][0]
        else:
            labels[w] = "multi-cluster"
    return labels


def detect_single_csv_mode(
    words: List[str], csv_paths: List[str], top_k: int
) -> Tuple[bool, Optional[str]]:
    """True when cluster assignment uses one puzzle CSV."""
    if len(csv_paths) == 1:
        stem = os.path.splitext(os.path.basename(csv_paths[0]))[0]
        return True, stem
    real = {
        c
        for c in assign_clusters(words, csv_paths=csv_paths, top_k=top_k).values()
        if c not in _META_CLUSTERS
    }
    if len(real) == 1:
        return True, next(iter(real))
    return False, None


def _spread_scores(xy: np.ndarray) -> np.ndarray:
    """Larger score = farther from nearest neighbor (better label candidate)."""
    n = len(xy)
    if n <= 1:
        return np.ones(n, dtype=np.float64)
    from scipy.spatial.distance import cdist

    d = cdist(xy, xy)
    np.fill_diagonal(d, np.inf)
    return np.min(d, axis=1)


def select_label_indices(
    xy: np.ndarray,
    *,
    label_all_below: int = _LABEL_ALL_BELOW,
    max_labels: int = _MAX_LABELS,
) -> List[int]:
    """Pick word indices to annotate; cap count for dense plots."""
    n = len(xy)
    if n == 0:
        return []
    if n <= label_all_below:
        return list(range(n))
    order = np.argsort(-_spread_scores(xy))
    return sorted(order[: min(n, max_labels)].tolist())


def _label_text(word: str, max_chars: int = _LABEL_MAX_CHARS) -> str:
    """Shorten a point label so long targets (molopt SMILES) stay readable."""
    return plot_label(word, max_chars)


def _annotate_words(
    ax,
    xy: np.ndarray,
    words: Sequence[str],
    indices: Sequence[int],
    *,
    fontsize: float = 8,
    max_chars: int = _LABEL_MAX_CHARS,
) -> int:
    """Place word labels and repel overlaps with adjustText."""
    if not indices:
        return 0
    texts = [
        ax.text(
            xy[i, 0],
            xy[i, 1],
            _label_text(words[i], max_chars),
            fontsize=fontsize,
            alpha=0.9,
            color="black",
        )
        for i in indices
    ]
    try:
        from adjustText import adjust_text

        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="0.45", lw=0.6),
            expand=(1.05, 1.15),
            force_text=(0.35, 0.55),
        )
    except ImportError:
        for i, text in zip(indices, texts):
            text.set_position((xy[i, 0], xy[i, 1]))
            text.set_ha("left")
            text.set_va("bottom")
    return len(indices)


def _scatter_by_cluster(
    ax,
    xy: np.ndarray,
    words: Sequence[str],
    cluster_labels: Sequence[str],
    unique_clusters: Sequence[str],
    style_map: Dict[str, ClusterStyle],
    point_sizes: np.ndarray,
    *,
    skip_unknown: bool = False,
) -> None:
    for cluster in unique_clusters:
        if skip_unknown and cluster == "unknown":
            continue
        mask = [i for i, c in enumerate(cluster_labels) if c == cluster]
        if not mask:
            continue
        style = style_map[cluster]
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            c=style["color"],
            marker=style["marker"],
            s=[point_sizes[i] for i in mask],
            alpha=0.85,
            label=cluster,
            edgecolors=style["edgecolor"],
            linewidths=0.7,
        )


def _legend_handles(
    unique_clusters: Sequence[str], style_map: Dict[str, ClusterStyle]
) -> list:
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0],
            [0],
            marker=style_map[cluster]["marker"],
            color="w",
            markerfacecolor=style_map[cluster]["color"],
            markeredgecolor=style_map[cluster]["edgecolor"],
            markeredgewidth=0.8,
            markersize=8,
            label=cluster,
        )
        for cluster in unique_clusters
        if cluster in style_map
    ]


def _pc_axis_label(pca, component: int) -> str:
    return f"PC{component + 1} ({100 * pca.explained_variance_ratio_[component]:.1f}% var)"


def plot_cluster_pca(
    output_dir: str,
    model_name: str = "meta-llama/Llama-3.2-1B",
    cache_dir: Optional[str] = None,
    layer: int = 13,
    low_rank_dim: int = 64,
    semantle_dir: Optional[str] = None,
    top_k: int = 100,
    save_path: Optional[str] = "cluster_pca.png",
    annotate: bool = True,
    figsize: tuple = (12, 9),
    *,
    reft_model=None,
    words: Optional[list] = None,
    items: Optional[list] = None,
    load_latest: bool = False,
):
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from boreft.bo.plotting import save_plot
    except ImportError as exc:
        raise ImportError(
            "Install matplotlib and scikit-learn: pip install matplotlib scikit-learn"
        ) from exc

    if reft_model is None or words is None or items is None:
        print(f"[cluster_pca] Loading checkpoint: {output_dir}")
        reft_model, _tokenizer, words, _prompt, items = _load_model_and_items(
            output_dir,
            model_name,
            layer,
            low_rank_dim,
            cache_dir=cache_dir,
            use_word_bias=None,
            variance=None,
            load_latest=load_latest,
        )
    else:
        print(f"[cluster_pca] Using pre-loaded checkpoint: {output_dir}")

    word_ids = [it["id"] for it in items]
    bias_label = _bias_type_label(reft_model)
    print(f"[cluster_pca] Intervention type: {bias_label}  |  {len(words)} words")

    bias_mat = np.stack(
        [get_bias_vector(reft_model, wid) for wid in word_ids], axis=0
    )

    word_to_cluster, label_source = resolve_word_cluster_labels(
        words, output_dir, semantle_dir, top_k
    )
    csv_paths = resolve_cluster_csv_paths(output_dir, semantle_dir)
    single_csv = False
    csv_stem: Optional[str] = None
    if label_source == "category":
        defs_path = definitions_path_from_config(output_dir)
        print(
            f"[cluster_pca] Category labels from {defs_path} "
            f"({len(set(word_to_cluster.values()))} categories)",
            flush=True,
        )
    elif label_source == "csv":
        stems = [os.path.splitext(os.path.basename(p))[0] for p in csv_paths]
        print(
            f"[cluster_pca] Cluster CSVs ({len(csv_paths)}): {', '.join(stems)} "
            f"(top_k={top_k})",
            flush=True,
        )
        single_csv, csv_stem = detect_single_csv_mode(words, csv_paths, top_k)
        if single_csv:
            print(f"[cluster_pca] Single-CSV mode ({csv_stem!r})")
        else:
            print(f"[cluster_pca] Multi-CSV mode ({len(csv_paths)} training puzzles)")
    else:
        single_csv = True
        print("[cluster_pca] No cluster CSVs — treating all words as one group")

    cluster_labels = [word_to_cluster[w] for w in words]
    unique_clusters = sorted(set(cluster_labels))
    print(f"[cluster_pca] Clusters: {unique_clusters}")

    n_components = min(2, bias_mat.shape[0], bias_mat.shape[1])
    pca = PCA(n_components=n_components)
    bias_2d = pca.fit_transform(bias_mat)

    iv = _get_intervention(reft_model)
    word_variance = None
    if hasattr(iv, "get_bias_mu_logvar") and (
        getattr(iv, "word_mu", None) is not None
        or getattr(iv, "bias_network", None) is not None
        or getattr(iv, "materialized_mu", None) is not None
    ):
        device = (
            iv.embed_cache.device
            if getattr(iv, "embed_cache", None) is not None
            else next(iv.parameters()).device
        )
        word_ids_t = torch.tensor(word_ids, dtype=torch.long, device=device)
        _, logvar_t = iv.get_bias_mu_logvar(word_ids_t)
        word_variance = (
            np.exp(0.5 * logvar_t.detach().float().cpu().numpy()).mean(axis=1)
        )

    if word_variance is not None:
        vr = np.ptp(word_variance)
        vr = vr if vr > 1e-8 else 1e-8
        point_sizes = 30 + 270 * (word_variance - word_variance.min()) / vr
    else:
        point_sizes = np.full(len(words), 60.0)

    style_map = build_cluster_style_map(unique_clusters)

    size_note = "  |  size = avg σ" if word_variance is not None else ""
    if label_source == "category":
        title_main = (
            f"Per-word {bias_label}-bias PCA  —  coloured by definition category"
            f"{size_note}\n"
            f"({len(words)} words, layer {layer}, rank {low_rank_dim})"
        )
        legend_title = "Category (normalized)"
    elif single_csv and csv_stem:
        title_main = (
            f"Per-word {bias_label}-bias PCA  —  {csv_stem}{size_note}\n"
            f"({len(words)} words, top_k={top_k}, layer {layer}, rank {low_rank_dim})"
        )
        legend_title = "Cluster (CSV stem)"
    elif len(csv_paths) > 1:
        title_main = (
            f"Per-word {bias_label}-bias PCA  —  {len(csv_paths)} semantle puzzles"
            f"{size_note}\n"
            f"({len(words)} words, top_k={top_k}, layer {layer}, rank {low_rank_dim})"
        )
        legend_title = "Cluster (CSV stem)"
    else:
        title_main = (
            f"Per-word {bias_label}-bias PCA  —  coloured by semantle cluster{size_note}\n"
            f"({len(words)} words, layer {layer}, rank {low_rank_dim})"
        )
        legend_title = "Cluster (CSV stem)"

    fig, ax = plt.subplots(figsize=figsize)
    _scatter_by_cluster(
        ax,
        bias_2d,
        words,
        cluster_labels,
        unique_clusters,
        style_map,
        point_sizes,
        skip_unknown=False,
    )

    if annotate and len(words) > 0:
        label_indices = select_label_indices(bias_2d)
        n_done = _annotate_words(ax, bias_2d, words, label_indices)
        print(f"[cluster_pca] Annotated {n_done}/{len(words)} words")

    ax.set_xlabel(_pc_axis_label(pca, 0), fontsize=11)
    ax.set_ylabel(_pc_axis_label(pca, 1), fontsize=11)
    ax.set_title(title_main, fontsize=12)
    legend_ncol = 2 if len(unique_clusters) > 14 else 1
    ax.legend(
        handles=_legend_handles(unique_clusters, style_map),
        title=legend_title,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=8 if legend_ncol > 1 else 9,
        title_fontsize=10,
        framealpha=0.9,
        ncol=legend_ncol,
    )
    fig.tight_layout()

    if save_path is not None:
        save_plot(fig, save_path, dpi=150)
        print(f"[cluster_pca] Saved: {save_path}")

    return fig


def main() -> None:
    p = argparse.ArgumentParser(
        description="Cluster-coloured PCA of per-word bias vectors."
    )
    p.add_argument("--output_dir", required=True, help="BOReFT checkpoint directory.")
    p.add_argument(
        "--model-name",
        dest="model_name",
        default=None,
        help="Base HF model id (default: training_config.json model_name).",
    )
    p.add_argument(
        "--cache_dir",
        default=None,
        help="HF cache directory (default: training_config.json cache_dir).",
    )
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
    p.add_argument(
        "--semantle_dir",
        default=None,
        help="Semantle CSV directory (default: training_config.json semantle_dir).",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k per CSV for cluster labels (default: training_config.json train_top_k).",
    )
    p.add_argument(
        "--save_path",
        default=None,
        help="Output PNG path (default: <output_dir>/eval/cluster_pca.png).",
    )
    p.add_argument("--no_annotate", action="store_true")
    add_load_latest_argument(p)
    args = p.parse_args()

    resolved = resolve_pca_checkpoint_args(
        args.output_dir,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        layer=args.layer,
        low_rank_dim=args.low_rank_dim,
        semantle_dir=args.semantle_dir,
        top_k=args.top_k,
        save_path=args.save_path,
    )
    print(
        "[cluster_pca] "
        f"model={resolved['model_name']} "
        f"layer={resolved['layer']} "
        f"rank={resolved['low_rank_dim']} "
        f"top_k={resolved['top_k']}",
        flush=True,
    )

    save = resolved["save_path"]
    if isinstance(save, str) and save.lower() == "none":
        save = None
    elif isinstance(save, str):
        save_dir = os.path.dirname(os.path.abspath(save))
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    plot_cluster_pca(
        output_dir=args.output_dir,
        model_name=str(resolved["model_name"]),
        cache_dir=resolved["cache_dir"],  # type: ignore[arg-type]
        layer=int(resolved["layer"]),
        low_rank_dim=int(resolved["low_rank_dim"]),
        semantle_dir=resolved["semantle_dir"],  # type: ignore[arg-type]
        top_k=int(resolved["top_k"]),
        save_path=save,
        annotate=not args.no_annotate,
        load_latest=bool(args.load_latest),
    )


if __name__ == "__main__":
    main()
