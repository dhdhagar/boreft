"""RECON similarity plot: Tanimoto distribution + embed_sim/TFS agreement.

Molecule runs score every reconstruction two ways — the embedding cosine from the
task's encoder and Morgan/Tanimoto over ECFP4 fingerprints — and the two can
disagree sharply (every encoder measured compresses unrelated drug-like molecules
into a narrow high-cosine band where Tanimoto separates them cleanly). A single
mean per metric hides that, so this plot shows the TFS distribution and scatters
the two metrics against each other.

Reads the per-target rows that ``eval.semantle.run_semantle_generation_eval``
writes to ``results["embedding_sim"]["per_target"]``; produces nothing for tasks
without fingerprints, whose rows carry no ``tfs`` key.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

# Shared with the eval suite so the plotted cut-off is the one the metrics use.
from boreft.eval.eval_suite import (
    DEFAULT_EMBED_SIM_TAU,
    DEFAULT_RDKIT_SIM_TAU,
    DEFAULT_TFS_TAU,
)

_EMBED_COLOR = "#2166AC"
_TFS_COLOR = "#B2182B"
_RDKIT_COLOR = "#1B7837"


def _style_axes(ax) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def tfs_pairs_from_per_target(
    per_target: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    """``(embed_sims, tfs)`` for rows carrying both metrics; empty when TFS is absent."""
    rows = [r for r in per_target if r.get("tfs") is not None]
    if not rows:
        return np.zeros(0), np.zeros(0)
    return (
        np.asarray([float(r["sim"]) for r in rows], dtype=np.float64),
        np.asarray([float(r["tfs"]) for r in rows], dtype=np.float64),
    )


def rdkit_pairs_from_per_target(
    per_target: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    rows = [r for r in per_target if r.get("rdkit_sim") is not None]
    if not rows:
        return np.zeros(0), np.zeros(0)
    return (
        np.asarray([float(r["sim"]) for r in rows], dtype=np.float64),
        np.asarray([float(r["rdkit_sim"]) for r in rows], dtype=np.float64),
    )


def plot_recon_similarity(
    per_target: Sequence[Mapping[str, object]],
    save_path: str,
    *,
    embed_sim_tau: float = DEFAULT_EMBED_SIM_TAU,
    tfs_tau: float = DEFAULT_TFS_TAU,
    rdkit_sim_tau: float = DEFAULT_RDKIT_SIM_TAU,
    embedding_model: str = "",
) -> Optional[str]:
    """Save the RECON similarity figure, or return ``None`` when there is no TFS.

    ``None`` is also returned when matplotlib is unavailable, matching the rest of
    the eval suite: plots are diagnostics and never fail the pipeline.
    """
    embed_sims, tfs = tfs_pairs_from_per_target(per_target)
    if embed_sims.size == 0:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from boreft.bo.plotting import save_plot
    except ImportError:
        return None

    rdkit_embed_sims, rdkit_sims = rdkit_pairs_from_per_target(per_target)
    if rdkit_sims.size:
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8))
        ax_hist, ax_scatter = axes[0]
        ax_rdkit_hist, ax_rdkit_scatter = axes[1]
    else:
        fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # ── Left: TFS distribution over targets ──────────────────────────────────
    ax_hist.hist(
        tfs, bins=np.linspace(0.0, 1.0, 41), color=_TFS_COLOR, alpha=0.75, log=False
    )
    for value, color, label in (
        (float(tfs.mean()), "0.15", f"mean = {tfs.mean():.3f}"),
        (tfs_tau, _EMBED_COLOR, rf"$\tau$ = {tfs_tau:g}"),
    ):
        ax_hist.axvline(value, color=color, lw=1.4, ls="--", label=label)
    frac_ge_tau = float(np.mean(tfs >= tfs_tau))
    ax_hist.set_xlabel("Tanimoto similarity (ECFP4) to target", fontsize=10)
    ax_hist.set_ylabel("Targets", fontsize=10)
    ax_hist.set_title(
        f"Structural reconstruction  (≥ $\\tau$: {frac_ge_tau:.1%} of "
        f"{tfs.size} targets)",
        fontsize=11,
    )
    ax_hist.set_xlim(0.0, 1.0)
    ax_hist.legend(fontsize=8, loc="upper right")
    _style_axes(ax_hist)

    # ── Right: do the two metrics agree? ─────────────────────────────────────
    ax_scatter.scatter(
        tfs, embed_sims, s=14, alpha=0.5, color=_EMBED_COLOR, edgecolors="none"
    )
    ax_scatter.axhline(
        embed_sim_tau, color="0.35", lw=1.1, ls="--", label=rf"embed $\tau$ = {embed_sim_tau:g}"
    )
    ax_scatter.axvline(
        tfs_tau, color="0.35", lw=1.1, ls=":", label=rf"TFS $\tau$ = {tfs_tau:g}"
    )
    ax_scatter.set_xlabel("Tanimoto similarity (ECFP4)", fontsize=10)
    ax_scatter.set_ylabel("Embedding similarity", fontsize=10)
    ax_scatter.set_xlim(-0.02, 1.02)
    ax_scatter.set_ylim(-0.02, 1.02)

    title = "Metric agreement"
    corr = _correlations(tfs, embed_sims)
    if corr:
        title += f"  ({corr})"
    ax_scatter.set_title(title, fontsize=11)
    ax_scatter.legend(fontsize=8, loc="lower right")
    _style_axes(ax_scatter)

    if rdkit_sims.size:
        ax_rdkit_hist.hist(
            rdkit_sims,
            bins=np.linspace(0.0, 1.0, 41),
            color=_RDKIT_COLOR,
            alpha=0.75,
        )
        for value, color, label in (
            (
                float(rdkit_sims.mean()),
                "0.15",
                f"mean = {rdkit_sims.mean():.3f}",
            ),
            (
                rdkit_sim_tau,
                _EMBED_COLOR,
                rf"$\tau$ = {rdkit_sim_tau:g}",
            ),
        ):
            ax_rdkit_hist.axvline(
                value, color=color, lw=1.4, ls="--", label=label
            )
        frac = float(np.mean(rdkit_sims >= rdkit_sim_tau))
        ax_rdkit_hist.set_xlabel("RDKit descriptor similarity to target", fontsize=10)
        ax_rdkit_hist.set_ylabel("Targets", fontsize=10)
        ax_rdkit_hist.set_title(
            f"Property reconstruction  (≥ $\\tau$: {frac:.1%} of "
            f"{rdkit_sims.size} targets)",
            fontsize=11,
        )
        ax_rdkit_hist.set_xlim(0.0, 1.0)
        ax_rdkit_hist.legend(fontsize=8, loc="upper right")
        _style_axes(ax_rdkit_hist)

        ax_rdkit_scatter.scatter(
            rdkit_sims,
            rdkit_embed_sims,
            s=14,
            alpha=0.5,
            color=_RDKIT_COLOR,
            edgecolors="none",
        )
        ax_rdkit_scatter.axhline(
            embed_sim_tau,
            color="0.35",
            lw=1.1,
            ls="--",
            label=rf"embed $\tau$ = {embed_sim_tau:g}",
        )
        ax_rdkit_scatter.axvline(
            rdkit_sim_tau,
            color="0.35",
            lw=1.1,
            ls=":",
            label=rf"RDKit $\tau$ = {rdkit_sim_tau:g}",
        )
        ax_rdkit_scatter.set_xlabel("RDKit descriptor similarity", fontsize=10)
        ax_rdkit_scatter.set_ylabel("Embedding similarity", fontsize=10)
        ax_rdkit_scatter.set_xlim(-0.02, 1.02)
        ax_rdkit_scatter.set_ylim(-0.02, 1.02)
        rdkit_corr = _correlations(rdkit_sims, rdkit_embed_sims)
        ax_rdkit_scatter.set_title(
            f"Property/semantic agreement"
            + (f"  ({rdkit_corr})" if rdkit_corr else ""),
            fontsize=11,
        )
        ax_rdkit_scatter.legend(fontsize=8, loc="lower right")
        _style_axes(ax_rdkit_scatter)

    suptitle = "RECON: structural vs. semantic reconstruction of training targets"
    if embedding_model:
        suptitle += f"\nembedding model: {embedding_model}"
    fig.suptitle(suptitle, fontsize=12, y=1.04)
    fig.tight_layout()
    save_plot(fig, save_path, dpi=200)
    plt.close(fig)
    return save_path


def _correlations(tfs: np.ndarray, embed_sims: np.ndarray) -> str:
    """``"Pearson r=..., Spearman ρ=..."`` label, or ``""`` when undefined.

    Both are degenerate when either metric is constant across targets (e.g. a
    collapsed model reproducing one molecule), which is exactly when a printed
    correlation would be most misleading.
    """
    if tfs.size < 3 or tfs.std() == 0.0 or embed_sims.std() == 0.0:
        return ""
    pearson = float(np.corrcoef(tfs, embed_sims)[0, 1])
    try:
        from scipy.stats import spearmanr

        spearman = float(spearmanr(tfs, embed_sims).statistic)
    except ImportError:
        return f"Pearson r={pearson:.2f}"
    return f"Pearson r={pearson:.2f}, Spearman ρ={spearman:.2f}"
