"""Silhouette-score evaluation for per-word bias clusters.

Callable API used by eval/run_full_eval.py and notebooks.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

from boreft.bo.plotting import save_plot


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def compute_silhouette(
    bias_vecs: np.ndarray,
    labels: np.ndarray,
    words: List[str],
) -> Tuple[float, Dict[str, float]]:
    """Compute overall and per-cluster silhouette scores.

    Args:
        bias_vecs: [W, rank] array of bias (μ) vectors.
        labels:    [W] array of cluster label strings.
        words:     list of W word strings (used for per-word reporting).

    Returns:
        overall:      float in [-1, 1]
        per_cluster:  {cluster_name: mean_score}
    """
    real_clusters = _real_clusters(labels)
    valid_mask = np.isin(labels, real_clusters)
    X = bias_vecs[valid_mask]
    y = labels[valid_mask]

    if len(set(y)) < 2 or len(X) < 2:
        return float("nan"), {}

    sample_scores = silhouette_samples(X, y, metric="euclidean")
    overall = float(silhouette_score(X, y, metric="euclidean"))
    per_cluster = {c: float(sample_scores[y == c].mean()) for c in sorted(set(y))}
    return overall, per_cluster


def plot_silhouette(
    bias_vecs: np.ndarray,
    labels: np.ndarray,
    words: List[str],
    save_path: str,
    typical_reach: Optional[float] = None,
    save_path_pca: Optional[str] = None,
    save_path_cosine: Optional[str] = None,
    ref_vecs: Optional[np.ndarray] = None,
) -> Dict:
    """Compute silhouette scores and save diagnostic plots.

    save_path        — full-dimensional silhouette plot (Euclidean in rank-dim space).
    save_path_pca    — (optional) same plots but computed on the 2-D PCA projection.
    save_path_cosine — (optional) cosine silhouette on L2-normalised bias vectors,
                       with a side-by-side comparison against ref_vecs silhouette.
    ref_vecs         — (optional) [W, m] reference embeddings (e.g. sentence-transformers)
                       used in the cosine plot for comparison.

    Args:
        bias_vecs:     [W, rank] μ vectors.
        labels:        [W] cluster label strings.
        words:         list of W words (for annotation).
        save_path:     path to write the PNG.
        typical_reach: optional σ√d value to annotate on the distance histogram.
        save_path_pca: optional path to write the 2-D PCA silhouette PNG.

    Returns:
        dict with keys: "overall", "per_cluster", "worst_words", "best_words"
    """
    real_clusters = _real_clusters(labels)
    valid_mask = np.isin(labels, real_clusters)
    X = bias_vecs[valid_mask]
    y = labels[valid_mask]
    valid_words = [w for w, v in zip(words, valid_mask) if v]

    if len(set(y)) < 2 or len(X) < 2:
        return {
            "overall": float("nan"),
            "per_cluster": {},
            "worst_words": [],
            "best_words": [],
        }

    sample_scores = silhouette_samples(X, y, metric="euclidean")
    overall = float(silhouette_score(X, y, metric="euclidean"))
    unique_y = sorted(set(y))
    n_clusters = len(unique_y)
    colors = cm.tab10(np.linspace(0, 1, n_clusters))

    # ── Within / between distance distributions ──────────────────────────────
    within_dists, between_dists = _distance_distributions(
        bias_vecs, labels, real_clusters
    )

    # ── Figure layout: silhouette bars | cluster means | distance histogram ──
    fig = plt.figure(figsize=(18, max(6, len(X) // 4)))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35)
    ax_sil = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_dist = fig.add_subplot(gs[1, :])

    # Plot 1: per-cluster silhouette bars
    y_lower = 0
    gap = 2
    for cluster, color in zip(unique_y, colors):
        cluster_scores = np.sort(sample_scores[y == cluster])
        size = len(cluster_scores)
        y_upper = y_lower + size
        ax_sil.barh(
            np.arange(y_lower, y_upper),
            cluster_scores,
            height=1.0,
            color=color,
            alpha=0.8,
        )
        ax_sil.text(
            -0.05,
            (y_lower + y_upper) / 2,
            cluster,
            ha="right",
            va="center",
            fontsize=7,
            color=color,
        )
        y_lower = y_upper + gap
    ax_sil.axvline(
        overall, color="black", ls="--", lw=1.5, label=f"overall = {overall:.3f}"
    )
    ax_sil.axvline(0, color="grey", ls=":", lw=1)
    ax_sil.set_xlabel("Silhouette score")
    ax_sil.set_title("Per-word silhouette (sorted within cluster)")
    ax_sil.set_yticks([])
    ax_sil.legend(loc="lower right", fontsize=7)

    # Plot 2: per-cluster mean bar chart
    cluster_means = {c: float(sample_scores[y == c].mean()) for c in unique_y}
    sorted_cl = sorted(cluster_means, key=cluster_means.get, reverse=True)
    bar_colors = [colors[unique_y.index(c)] for c in sorted_cl]
    bars = ax_bar.bar(
        sorted_cl, [cluster_means[c] for c in sorted_cl], color=bar_colors, alpha=0.85
    )
    ax_bar.axhline(
        overall, color="black", ls="--", lw=1.5, label=f"overall = {overall:.3f}"
    )
    ax_bar.axhline(0, color="grey", ls=":", lw=1)
    ax_bar.set_ylabel("Mean silhouette score")
    ax_bar.set_title("Per-cluster mean silhouette")
    ax_bar.set_ylim(-1, 1)
    ax_bar.legend(fontsize=7)
    for bar, c in zip(bars, sorted_cl):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{cluster_means[c]:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    plt.setp(ax_bar.get_xticklabels(), rotation=30, ha="right", fontsize=7)

    # Plot 3: within vs between distance histogram
    ax_dist.hist(
        within_dists, bins=40, alpha=0.6, label="within cluster", color="steelblue"
    )
    ax_dist.hist(
        between_dists, bins=40, alpha=0.6, label="between clusters", color="tomato"
    )
    if typical_reach is not None:
        ax_dist.axvline(
            typical_reach,
            color="black",
            ls="--",
            label=f"typical σ√d = {typical_reach:.2f}",
        )
    ax_dist.set_xlabel("Euclidean distance (full bias space)")
    ax_dist.set_ylabel("Count")
    ax_dist.set_title("Within- vs between-cluster Euclidean distances")
    ax_dist.legend()

    save_plot(fig, save_path, dpi=120)
    plt.close(fig)

    # Worst / best words by silhouette score
    order = np.argsort(sample_scores)
    worst = [
        {"target": valid_words[i], "cluster": y[i], "score": float(sample_scores[i])}
        for i in order[:10]
    ]
    best = [
        {"target": valid_words[i], "cluster": y[i], "score": float(sample_scores[i])}
        for i in order[-10:][::-1]
    ]

    # ── Optional: silhouette on 2-D PCA projection ───────────────────────────
    pca2d_metrics: dict = {}
    if save_path_pca is not None:
        pca2d_metrics = _plot_silhouette_pca(
            X, y, overall, colors, unique_y, save_path_pca
        )

    # ── Optional: cosine silhouette on normalised z vs reference embeddings ──
    cosine_metrics: dict = {}
    if save_path_cosine is not None:
        ref_X = ref_vecs[valid_mask] if ref_vecs is not None else None
        cosine_metrics = _plot_silhouette_cosine(
            X, y, overall, colors, unique_y, save_path_cosine, ref_vecs=ref_X
        )

    return {
        "overall": overall,
        "per_cluster": cluster_means,
        "worst_words": worst,
        "best_words": best,
        "pca2d": pca2d_metrics,  # {"overall": float, "per_cluster": {cluster: float}}
        "cosine": cosine_metrics,  # {"overall": float, "per_cluster": {cluster: float}}
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _real_clusters(labels: np.ndarray) -> List[str]:
    return [c for c in np.unique(labels) if c not in ("multi-cluster", "unknown")]


def _plot_silhouette_pca(
    X: np.ndarray,
    y: np.ndarray,
    overall_fullD: float,
    colors: np.ndarray,
    unique_y: List[str],
    save_path: str,
) -> Dict:
    """Compute and save silhouette plots using 2-D PCA projection of X.

    Produces a 3-panel figure (same layout as the full-dim plot):
      - Left:  per-cluster silhouette bars in 2-D PCA space
      - Right: per-cluster mean bars in 2-D PCA space
      - Below: within- vs between-cluster distance histogram in 2-D PCA space

    The overall score from the full-dim space is shown as a dashed reference.
    """
    X2 = PCA(n_components=2).fit_transform(X)

    pca_scores = silhouette_samples(X2, y, metric="euclidean")
    overall_pca = float(silhouette_score(X2, y, metric="euclidean"))

    # within / between in 2-D
    within_dists_pca: List[float] = []
    between_dists_pca: List[float] = []
    for c in unique_y:
        idx = np.where(y == c)[0]
        if len(idx) < 2:
            continue
        d = cdist(X2[idx], X2[idx], metric="euclidean")
        within_dists_pca.extend(d[np.triu_indices(len(idx), k=1)].tolist())
    for c1, c2 in combinations(unique_y, 2):
        i1, i2 = np.where(y == c1)[0], np.where(y == c2)[0]
        between_dists_pca.extend(cdist(X2[i1], X2[i2]).ravel().tolist())

    cluster_means_pca = {c: float(pca_scores[y == c].mean()) for c in unique_y}
    sorted_cl = sorted(cluster_means_pca, key=cluster_means_pca.get, reverse=True)
    bar_colors = [colors[unique_y.index(c)] for c in sorted_cl]

    fig = plt.figure(figsize=(18, max(6, len(X) // 4)))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35)
    ax_sil = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_dist = fig.add_subplot(gs[1, :])

    # silhouette bars
    y_lower = 0
    for cluster, color in zip(unique_y, colors):
        cluster_scores = np.sort(pca_scores[y == cluster])
        y_upper = y_lower + len(cluster_scores)
        ax_sil.barh(
            np.arange(y_lower, y_upper),
            cluster_scores,
            height=1.0,
            color=color,
            alpha=0.8,
        )
        ax_sil.text(
            -0.05,
            (y_lower + y_upper) / 2,
            cluster,
            ha="right",
            va="center",
            fontsize=7,
            color=color,
        )
        y_lower = y_upper + 2
    ax_sil.axvline(
        overall_pca,
        color="black",
        ls="--",
        lw=1.5,
        label=f"PCA overall = {overall_pca:.3f}",
    )
    ax_sil.axvline(
        overall_fullD,
        color="grey",
        ls=":",
        lw=1.5,
        label=f"full-dim overall = {overall_fullD:.3f}",
    )
    ax_sil.axvline(0, color="grey", ls=":", lw=0.8)
    ax_sil.set_xlabel("Silhouette score (2-D PCA)")
    ax_sil.set_title("Per-word silhouette — 2-D PCA space")
    ax_sil.set_yticks([])
    ax_sil.legend(loc="lower right", fontsize=7)

    # per-cluster mean bars
    bars = ax_bar.bar(
        sorted_cl,
        [cluster_means_pca[c] for c in sorted_cl],
        color=bar_colors,
        alpha=0.85,
    )
    ax_bar.axhline(
        overall_pca,
        color="black",
        ls="--",
        lw=1.5,
        label=f"PCA overall = {overall_pca:.3f}",
    )
    ax_bar.axhline(
        overall_fullD,
        color="grey",
        ls=":",
        lw=1.5,
        label=f"full-dim overall = {overall_fullD:.3f}",
    )
    ax_bar.axhline(0, color="grey", ls=":", lw=0.8)
    ax_bar.set_ylabel("Mean silhouette score")
    ax_bar.set_title("Per-cluster mean silhouette — 2-D PCA space")
    ax_bar.set_ylim(-1, 1)
    ax_bar.legend(fontsize=7)
    for bar, c in zip(bars, sorted_cl):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{cluster_means_pca[c]:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    plt.setp(ax_bar.get_xticklabels(), rotation=30, ha="right", fontsize=7)

    # distance histogram
    ax_dist.hist(
        within_dists_pca, bins=40, alpha=0.6, color="steelblue", label="within cluster"
    )
    ax_dist.hist(
        between_dists_pca, bins=40, alpha=0.6, color="tomato", label="between clusters"
    )
    ax_dist.set_xlabel("Euclidean distance (2-D PCA space)")
    ax_dist.set_ylabel("Count")
    ax_dist.set_title("Within- vs between-cluster Euclidean distances — 2-D PCA space")
    ax_dist.legend()

    save_plot(fig, save_path, dpi=120)
    plt.close(fig)

    return {
        "overall": overall_pca,
        "per_cluster": cluster_means_pca,
    }


def _plot_silhouette_cosine(
    X: np.ndarray,
    y: np.ndarray,
    overall_euclidean: float,
    colors: np.ndarray,
    unique_y: List[str],
    save_path: str,
    ref_vecs: Optional[np.ndarray] = None,
) -> Dict:
    """Cosine silhouette on **L2-normalised** bias vectors, with optional reference comparison.

    Steps:
      1. L2-normalise X  →  X_norm  (cosine distance on X_norm == angular distance)
      2. Compute silhouette_samples / silhouette_score with metric="cosine"
      3. If ref_vecs provided: L2-normalise ref_vecs and compute the same metrics
         so we can compare learned-space separation vs reference-space separation.

    Layout (3 rows):
      Row 0: per-word silhouette bars (learned) | per-cluster mean bars (learned vs ref)
      Row 1: within- vs between-cluster cosine-distance histogram (learned)
      Row 2: (only when ref_vecs given) same histogram for reference embeddings
    """
    eps = 1e-12
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / np.where(norms < eps, eps, norms)

    cosine_scores = silhouette_samples(X_norm, y, metric="cosine")
    overall_cosine = float(silhouette_score(X_norm, y, metric="cosine"))
    cluster_means_cos = {c: float(cosine_scores[y == c].mean()) for c in unique_y}

    # ── Reference embedding silhouette (optional) ─────────────────────────────
    ref_scores: Optional[np.ndarray] = None
    overall_ref: Optional[float] = None
    cluster_means_ref: Dict[str, float] = {}
    if ref_vecs is not None and len(ref_vecs) == len(X):
        ref_norms = np.linalg.norm(ref_vecs, axis=1, keepdims=True)
        R_norm = ref_vecs / np.where(ref_norms < eps, eps, ref_norms)
        ref_scores = silhouette_samples(R_norm, y, metric="cosine")
        overall_ref = float(silhouette_score(R_norm, y, metric="cosine"))
        cluster_means_ref = {c: float(ref_scores[y == c].mean()) for c in unique_y}

    # ── Within / between cosine-distance histograms ───────────────────────────
    def _cos_dists(Z: np.ndarray) -> Tuple[List[float], List[float]]:
        w_dists: List[float] = []
        b_dists: List[float] = []
        for c in unique_y:
            idx = np.where(y == c)[0]
            if len(idx) < 2:
                continue
            d = cdist(Z[idx], Z[idx], metric="cosine")
            w_dists.extend(d[np.triu_indices(len(idx), k=1)].tolist())
        for c1, c2 in combinations(unique_y, 2):
            i1, i2 = np.where(y == c1)[0], np.where(y == c2)[0]
            b_dists.extend(cdist(Z[i1], Z[i2], metric="cosine").ravel().tolist())
        return w_dists, b_dists

    within_learned, between_learned = _cos_dists(X_norm)
    within_ref, between_ref = _cos_dists(R_norm) if ref_vecs is not None else ([], [])

    # ── Figure layout ─────────────────────────────────────────────────────────
    n_rows = 3 if ref_vecs is not None else 2
    sorted_cl = sorted(cluster_means_cos, key=cluster_means_cos.get, reverse=True)
    bar_colors = [colors[unique_y.index(c)] for c in sorted_cl]

    fig = plt.figure(figsize=(18, max(8, len(X) // 4)))
    gs = fig.add_gridspec(n_rows, 2, hspace=0.45, wspace=0.35)
    ax_sil = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_dist = fig.add_subplot(gs[1, :])
    if ref_vecs is not None:
        ax_ref = fig.add_subplot(gs[2, :])

    # ── Panel 1: per-word silhouette bars (learned) ───────────────────────────
    y_lower = 0
    for cluster, color in zip(unique_y, colors):
        cs = np.sort(cosine_scores[y == cluster])
        y_upper = y_lower + len(cs)
        ax_sil.barh(np.arange(y_lower, y_upper), cs, height=1.0, color=color, alpha=0.8)
        ax_sil.text(
            -0.05,
            (y_lower + y_upper) / 2,
            cluster,
            ha="right",
            va="center",
            fontsize=7,
            color=color,
        )
        y_lower = y_upper + 2
    ax_sil.axvline(
        overall_cosine,
        color="black",
        ls="--",
        lw=1.5,
        label=f"learned = {overall_cosine:.3f}",
    )
    if overall_ref is not None:
        ax_sil.axvline(
            overall_ref,
            color="darkorange",
            ls="--",
            lw=1.5,
            label=f"ref emb = {overall_ref:.3f}",
        )
    ax_sil.axvline(0, color="grey", ls=":", lw=0.8)
    ax_sil.axvline(
        overall_euclidean,
        color="grey",
        ls=":",
        lw=1.2,
        label=f"euclidean = {overall_euclidean:.3f}",
    )
    ax_sil.set_xlabel("Silhouette score")
    ax_sil.set_title("Per-word silhouette  —  cosine on normalised z")
    ax_sil.set_yticks([])
    ax_sil.legend(loc="lower right", fontsize=7)

    # ── Panel 2: per-cluster mean bars (learned + reference side-by-side) ─────
    x_pos = np.arange(len(sorted_cl))
    width = 0.35 if overall_ref is not None else 0.6
    bars_l = ax_bar.bar(
        x_pos - (width / 2 if overall_ref is not None else 0),
        [cluster_means_cos[c] for c in sorted_cl],
        width,
        color=bar_colors,
        alpha=0.85,
        label="learned z (cosine)",
    )
    if overall_ref is not None:
        bars_r = ax_bar.bar(
            x_pos + width / 2,
            [cluster_means_ref.get(c, float("nan")) for c in sorted_cl],
            width,
            color=bar_colors,
            alpha=0.4,
            hatch="//",
            edgecolor="grey",
            label="ref embeddings (cosine)",
        )
        for bar, c in zip(bars_r, sorted_cl):
            v = cluster_means_ref.get(c, float("nan"))
            if not np.isnan(v):
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 0.02,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="grey",
                )
    ax_bar.axhline(
        overall_cosine,
        color="black",
        ls="--",
        lw=1.5,
        label=f"learned overall = {overall_cosine:.3f}",
    )
    if overall_ref is not None:
        ax_bar.axhline(
            overall_ref,
            color="darkorange",
            ls="--",
            lw=1.5,
            label=f"ref overall = {overall_ref:.3f}",
        )
    ax_bar.axhline(0, color="grey", ls=":", lw=0.8)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(sorted_cl, rotation=30, ha="right", fontsize=7)
    ax_bar.set_ylabel("Mean silhouette score")
    ax_bar.set_title("Per-cluster mean silhouette  —  cosine  (learned vs reference)")
    ax_bar.set_ylim(-1, 1)
    ax_bar.legend(fontsize=7)
    for bar, c in zip(bars_l, sorted_cl):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            cluster_means_cos[c] + 0.02,
            f"{cluster_means_cos[c]:.2f}",
            ha="center",
            va="bottom",
            fontsize=6,
        )

    # ── Panel 3: cosine-distance histogram for learned z ─────────────────────
    ax_dist.hist(
        within_learned, bins=40, alpha=0.6, color="steelblue", label="within cluster"
    )
    ax_dist.hist(
        between_learned, bins=40, alpha=0.6, color="tomato", label="between clusters"
    )
    ax_dist.set_xlabel("Cosine distance  (learned normalised z)")
    ax_dist.set_ylabel("Count")
    ax_dist.set_title("Within- vs between-cluster cosine distances  —  learned z")
    ax_dist.legend()

    # ── Panel 4 (optional): cosine-distance histogram for reference ───────────
    if ref_vecs is not None:
        ax_ref.hist(
            within_ref, bins=40, alpha=0.6, color="steelblue", label="within cluster"
        )
        ax_ref.hist(
            between_ref, bins=40, alpha=0.6, color="tomato", label="between clusters"
        )
        ax_ref.set_xlabel(
            "Cosine distance  (reference sentence-transformer embeddings)"
        )
        ax_ref.set_ylabel("Count")
        ax_ref.set_title(
            "Within- vs between-cluster cosine distances  —  reference embeddings"
        )
        ax_ref.legend()

    save_plot(fig, save_path, dpi=120)
    plt.close(fig)

    result: Dict = {"overall": overall_cosine, "per_cluster": cluster_means_cos}
    if overall_ref is not None:
        result["ref_overall"] = overall_ref
        result["ref_per_cluster"] = cluster_means_ref
    return result


def _distance_distributions(
    bias_vecs: np.ndarray,
    labels: np.ndarray,
    real_clusters: List[str],
) -> Tuple[List[float], List[float]]:
    within_dists: List[float] = []
    for c in real_clusters:
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        d = cdist(bias_vecs[idx], bias_vecs[idx], metric="euclidean")
        within_dists.extend(d[np.triu_indices(len(idx), k=1)].tolist())

    between_dists: List[float] = []
    for c1, c2 in combinations(real_clusters, 2):
        i1 = np.where(labels == c1)[0]
        i2 = np.where(labels == c2)[0]
        d = cdist(bias_vecs[i1], bias_vecs[i2], metric="euclidean")
        between_dists.extend(d.ravel().tolist())

    return within_dists, between_dists
