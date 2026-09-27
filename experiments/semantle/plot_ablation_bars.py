#!/usr/bin/env python3
"""Two-panel Semantle figure: training-set size and ablation bars.

The ladder series are the values drawn in ``search_n_ladder.pdf``.
Similarity bars match the train / held-out table: the unweighted mean of the
two split means, with the exact-match label equal to their sum.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "notes" / "theory" / "figures" / "search_scaling_panels.pdf"

SIM_COLOR = "#0072B2"
EM_COLOR = "#D55E00"
COVERAGE_COLOR = "0.45"
DELTA_COLOR = "0.78"
EDGE_COLOR = "black"

N_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072)
SIMILARITY_N = (
    0.6858, 0.6960, 0.7211, 0.7595, 0.7748, 0.8039, 0.8139,
    0.8235, 0.8217, 0.8269, 0.8069, 0.8470, 0.8776,
)
EXACT_N = (0, 0, 0, 0, 2, 3, 5, 5, 7, 7, 4, 8, 14)
# Coverage was not drawn at N=512 in the source figure.
COVERAGE_N = {
    1: 0.8879, 2: 0.8550, 4: 0.8402, 8: 0.8110, 16: 0.7992, 32: 0.7840,
    64: 0.7515, 128: 0.7292, 256: 0.6992, 1024: 0.6416, 2048: 0.6081, 3072: 0.5878,
}

# (label, italic, train sim, held-out sim, train EM, held-out EM, delta, solid edge)
ROWS = (
    ("BOReFT", False, 0.863, 0.892, 6, 8, 0.338, True),
    ("w/o distill.", True, 0.876, 0.829, 6, 3, 0.355, False),
    ("w/o recon.", True, 0.821, 0.852, 2, 5, 0.396, False),
    ("w/o encoder", True, 0.792, 0.833, 1, 4, 0.382, False),
    ("w/o var.", True, 0.759, 0.799, 0, 3, 0.356, False),
)
ITALIC = FontProperties(family="DejaVu Serif", style="italic", size=6)
BOLD = FontProperties(family="DejaVu Serif", weight="bold", size=6)


def _style_axis(axis) -> None:
    axis.tick_params(
        axis="both", length=4, width=1.1, labelsize=7, direction="out", top=True, right=True,
    )
    for spine in axis.spines.values():
        spine.set_linewidth(1.3)
    axis.grid(alpha=0.2, zorder=0)
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


def _legend_above(axis, handles, ncol: int) -> None:
    axis.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncol,
        frameon=False,
        fontsize=8,
        borderaxespad=0.0,
        handlelength=2.4,
        columnspacing=1.0,
    )


def _draw_ladder(axis) -> None:
    xs = list(N_SIZES)
    axis.plot(
        xs, SIMILARITY_N,
        color=SIM_COLOR, linestyle="-", marker="o", markersize=5.5, linewidth=2.0,
        label="Similarity", zorder=3,
    )
    axis.plot(
        xs, [k / 30 for k in EXACT_N],
        color=EM_COLOR, linestyle="--", marker="s", markersize=5.0, linewidth=2.0,
        label="EM", zorder=3,
    )
    cov_xs = [n for n in xs if n in COVERAGE_N]
    axis.plot(
        cov_xs, [COVERAGE_N[n] for n in cov_xs],
        color=COVERAGE_COLOR, alpha=0.8, linestyle=":", marker="D", markersize=3.4, linewidth=2.0,
        label=r"$\bar{\varepsilon}_{\mathrm{cov}}$", zorder=2,
    )
    axis.set_xscale("log", base=2)
    labeled = {1, 4, 16, 64, 256, 1024, 3072}
    axis.set_xticks(xs)
    axis.set_xticklabels(
        [str(x) if x in labeled else "" for x in xs],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    axis.set_xlim(min(xs) / 1.15, max(xs) * 1.15)
    axis.set_xlabel("Training set size", fontsize=8, labelpad=2)
    _style_axis(axis)
    _legend_above(axis, axis.get_lines(), ncol=3)


def _draw_bars(axis) -> None:
    x = np.arange(len(ROWS))
    width = 0.38
    dash = (0, (3.2, 1.7))
    for i, row in enumerate(ROWS):
        _label, _italic, train_sim, held_sim, train_em, held_em, dist, is_solid = row
        linestyle = "solid" if is_solid else dash
        sim = (train_sim + held_sim) / 2
        sim_bar = axis.bar(
            x[i] - width / 2, sim, width,
            color=SIM_COLOR, edgecolor=EDGE_COLOR, linewidth=1.25, linestyle=linestyle, zorder=3,
        )
        delta_bar = axis.bar(
            x[i] + width / 2, dist, width,
            color=DELTA_COLOR, edgecolor=EDGE_COLOR, linewidth=1.25, linestyle=linestyle, zorder=3,
        )
        sim_bar[0].set_linestyle(linestyle)
        delta_bar[0].set_linestyle(linestyle)
        axis.text(
            x[i] - width / 2,
            sim + 0.012,
            f"{train_em + held_em}/30",
            color=EM_COLOR,
            ha="center",
            va="bottom",
            fontsize=6,
            zorder=4,
            clip_on=False,
        )
    axis.set_xticks(x)
    axis.set_xticklabels([row[0] for row in ROWS], rotation=0, ha="center", va="top", fontsize=6)
    axis.tick_params(axis="x", pad=1)
    axis.set_xlim(-0.7, len(ROWS) - 0.3)
    axis.set_xlabel("Training ablations", fontsize=8, labelpad=3)
    _style_axis(axis)
    for tick, row in zip(axis.get_xticklabels(), ROWS):
        tick.set_fontsize(6)
        if row[0] == "BOReFT":
            tick.set_fontproperties(BOLD)
        elif row[1]:
            tick.set_fontproperties(ITALIC)
    _legend_above(
        axis,
        [
            Patch(facecolor=SIM_COLOR, edgecolor=EDGE_COLOR, linewidth=0.9, label="Similarity"),
            Patch(facecolor=DELTA_COLOR, edgecolor=EDGE_COLOR, linewidth=0.9, label=r"$\Delta_{\mathrm{interp}}$"),
        ],
        ncol=2,
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Palatino", "Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavuserif",
        }
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(5.55, 2.45),
        gridspec_kw={"width_ratios": [0.98, 1.48]},
    )
    _draw_ladder(axes[0])
    _draw_bars(axes[1])
    fig.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.21, wspace=0.26)
    bottom = min(ax.get_position().y0 for ax in axes)
    height = min(ax.get_position().height for ax in axes)
    for ax in axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, bottom, pos.width, height])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    left_top = axes[0].xaxis.label.get_window_extent(renderer).y1
    right_x = axes[1].get_window_extent(renderer).x0
    right_x += 0.5 * axes[1].get_window_extent(renderer).width
    _, xlabel_y = axes[1].transAxes.inverted().transform((right_x, left_top))
    axes[1].xaxis.set_label_coords(0.5, xlabel_y)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=160)
    print(OUT)


if __name__ == "__main__":
    main()
