"""Best-observed-value plots for individual and repeated BO runs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .state import Observation

_AXIS_LABELPAD = 4
_SPINE_WIDTH = 1.6
_WARMSTART_COLOR = "0.45"
_WARMSTART_LABEL_SIZE = 8
_WARMSTART_LABEL_GAP = 4
_BO_RC = {
    "font.family": "serif",
    "font.serif": [
        "Palatino",
        "Palatino Linotype",
        "Book Antiqua",
        "DejaVu Serif",
    ],
    # Default 6 sits on the outward top ticks; 12 clears them with a small gap.
    "axes.titlepad": 12,
}


def bo_rc_params() -> dict:
    """Serif rcParams shared by BO run plots and Semantle comparison figures."""
    return dict(_BO_RC)


def save_plot(fig, path: str | Path, **kwargs) -> tuple[Path, Path]:
    """Write ``path`` as a vector PDF and a PNG raster sibling.

    W&B Models media panels render PNG/JPEG through ``wandb.Image``, not PDF.
    PDF is the on-disk / paper format. Either suffix on ``path`` is accepted.
    """
    dest = Path(path)
    pdf_path = dest.with_suffix(".pdf")
    png_path = dest.with_suffix(".png")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    save_kw = {"bbox_inches": "tight"}
    save_kw.update(kwargs)
    # Raster DPI / explicit format belong on the PNG; PDFs are vectors.
    pdf_kw = {
        key: value for key, value in save_kw.items() if key not in {"dpi", "format"}
    }
    fig.savefig(pdf_path, **pdf_kw)
    fig.savefig(png_path, **save_kw)
    return pdf_path, png_path


def apply_bo_axes_style(
    axis,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
    spine_width: float | None = None,
) -> None:
    """Box frame, outward ticks, and Palatino-sized labels as in ``plot_best_so_far``."""
    axis.tick_params(
        axis="both",
        length=6,
        width=_SPINE_WIDTH,
        labelsize=11,
        direction="out",
        top=True,
        right=True,
    )
    frame_width = _SPINE_WIDTH if spine_width is None else spine_width
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(frame_width)
    if grid:
        axis.grid(alpha=0.2)
    if xlabel is not None:
        axis.set_xlabel(xlabel, fontsize=14, labelpad=_AXIS_LABELPAD)
    if ylabel is not None:
        axis.set_ylabel(ylabel, fontsize=14, labelpad=_AXIS_LABELPAD)


def annotate_warmstart_xaxis(
    fig, axis, warmstart_count: int, *, label_size: float | None = None
) -> None:
    """Bracket + ``warmstart`` label under ticks 1..N on the x-axis."""
    if warmstart_count < 1:
        return
    from matplotlib.transforms import ScaledTranslation

    trans = axis.get_xaxis_transform()
    tick = axis.xaxis.get_major_ticks()[0]
    tick_pad = float(tick.get_pad())
    tick_length = float(axis.xaxis.get_tick_params().get("length", 0.0))
    tick_labelsize = float(axis.xaxis.get_ticklabels()[0].get_fontsize())
    below_labels = tick_length + tick_pad + tick_labelsize + _WARMSTART_LABEL_GAP
    label_shift = ScaledTranslation(0, -below_labels / 72.0, fig.dpi_scale_trans)
    for label, loc in zip(axis.get_xticklabels(), axis.get_xticks()):
        if 1.0 - 1e-9 <= float(loc) <= warmstart_count + 1e-9:
            label.set_color(_WARMSTART_COLOR)
    for major in axis.xaxis.get_major_ticks():
        loc = float(major.get_loc())
        if 1.0 - 1e-9 <= loc <= warmstart_count + 1e-9:
            major.tick1line.set_color(_WARMSTART_COLOR)
            major.tick2line.set_color(_WARMSTART_COLOR)
            major.tick1line.set_markeredgecolor(_WARMSTART_COLOR)
            major.tick2line.set_markeredgecolor(_WARMSTART_COLOR)
    axis.plot(
        [1.0, float(warmstart_count)],
        [0.0, 0.0],
        color=_WARMSTART_COLOR,
        linewidth=_SPINE_WIDTH,
        transform=trans,
        clip_on=False,
        solid_capstyle="butt",
        zorder=3,
    )
    axis.annotate(
        "",
        xy=(1, 0.0),
        xytext=(warmstart_count, 0.0),
        xycoords=trans,
        textcoords=trans,
        arrowprops={
            "arrowstyle": "|-|",
            "color": _WARMSTART_COLOR,
            "lw": 0.8,
            "mutation_scale": 6,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        annotation_clip=False,
    )
    axis.annotate(
        "warmstart",
        xy=((1 + warmstart_count) / 2.0, 0.0),
        xycoords=trans + label_shift,
        ha="center",
        va="top",
        fontsize=_WARMSTART_LABEL_SIZE if label_size is None else label_size,
        color=_WARMSTART_COLOR,
        annotation_clip=False,
    )


def best_so_far(observations: Sequence[Observation]) -> np.ndarray:
    if not observations:
        return np.zeros(0, dtype=np.float64)
    return np.maximum.accumulate(
        [observation.peak_score() for observation in observations]
    )


def _is_warmstart(observation) -> bool:
    """BO observations carry ``source``; baseline observations carry ``phase``."""
    kind = getattr(observation, "source", None) or getattr(observation, "phase", "")
    return kind == "warmstart"


def _warmstart_count(runs: Sequence[Sequence[Observation]]) -> int:
    counts = [
        sum(_is_warmstart(observation) for observation in run) for run in runs if run
    ]
    return int(counts[0]) if counts else 0


def bo_trajectory(
    observations: Sequence[Observation],
) -> tuple[np.ndarray, np.ndarray]:
    """Best-so-far after each evaluation; x is the 1-based iteration index."""
    values = best_so_far(observations)
    if not len(values):
        return np.zeros(0, dtype=int), values
    return np.arange(1, len(values) + 1), values


def aggregate_trajectories(
    runs: Sequence[Sequence[Observation]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return iteration indices, mean, and SD over aligned runs."""
    if not runs:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    trajectories = [bo_trajectory(run)[1] for run in runs]
    length = max((len(values) for values in trajectories), default=0)
    if length == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    matrix = np.full((len(runs), length), np.nan, dtype=np.float64)
    for row, values in enumerate(trajectories):
        if len(values):
            matrix[row, : len(values)] = values
            matrix[row, len(values) :] = values[-1]
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0, ddof=0)
    return np.arange(1, length + 1), mean, std


def iteration_ticks(
    upper: float,
    locator_ticks: Sequence[float],
    *,
    warmstart_count: int = 0,
) -> list[float]:
    """Keep locator ticks in ``[1, upper]`` and force warmstart-end and budget."""
    lo = 1.0
    hi = float(upper)
    ticks = [
        float(tick)
        for tick in locator_ticks
        if lo - 1e-9 <= float(tick) <= hi + 1e-9
    ]
    required = [hi]
    if warmstart_count >= 1:
        required.append(float(warmstart_count))
        required.append(1.0)
    span = max(hi - lo, 1.0)
    min_gap = 0.04 * span
    for value in required:
        if lo - 1e-9 <= value <= hi + 1e-9 and not any(
            np.isclose(tick, value) for tick in ticks
        ):
            ticks.append(value)
    kept: list[float] = []
    for tick in sorted(ticks):
        is_required = any(np.isclose(tick, value) for value in required)
        if not is_required and any(
            abs(tick - value) < min_gap - 1e-12 for value in required
        ):
            continue
        kept.append(tick)
    return kept


def plot_best_so_far(
    runs: Sequence[Sequence[Observation]],
    path: str | Path,
    *,
    title: str = "Bayesian optimization",
) -> None:
    """Write a compact PNG with per-seed traces and mean ± standard deviation."""
    import matplotlib.pyplot as plt

    x, mean, std = aggregate_trajectories(runs)
    if not len(x):
        return
    warmstart_count = _warmstart_count(runs)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(bo_rc_params()):
        fig, axis = plt.subplots(figsize=(7, 4.5))
        for run in runs:
            run_x, values = bo_trajectory(run)
            axis.plot(
                run_x,
                values,
                color="0.7",
                linewidth=1,
                alpha=0.65,
            )
        label = "Best-so-far" if len(runs) == 1 else "Mean best-so-far"
        (mean_line,) = axis.plot(x, mean, linewidth=2, label=label)
        if len(runs) > 1:
            axis.fill_between(
                x,
                mean - std,
                mean + std,
                color=mean_line.get_color(),
                alpha=0.2,
                label="±1 std.",
            )
        upper = float(x[-1])
        axis.set_xlim(1, upper)
        ticks = iteration_ticks(
            upper,
            axis.get_xticks(),
            warmstart_count=warmstart_count,
        )
        axis.set_xticks(ticks)
        apply_bo_axes_style(
            axis,
            xlabel="Iteration",
            ylabel=r"Objective ($\uparrow$)",
        )
        if warmstart_count >= 1:
            annotate_warmstart_xaxis(fig, axis, warmstart_count)
        axis.set_title(title)
        axis.legend(loc="lower right")
        fig.tight_layout()
        save_plot(fig, target, dpi=160)
        plt.close(fig)
