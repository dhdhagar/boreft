#!/usr/bin/env python3
"""Aggregate protocol search on the training-size and rank ladders.

Loads ``search/boreft_N*`` and ``search/boreft_rank*`` plus the published
``s1_t0_ard_d64`` cell (train rerun when present) as N=3072 / rank 64.

    python experiments/semantle/compare_ladders.py
    python experiments/semantle/compare_ladders.py --no-wandb
    python experiments/semantle/compare_ladders.py --from-metrics --copy-paper --no-wandb
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = str(REPO_ROOT / "src")
EXPERIMENT_DIR = str(Path(__file__).resolve().parent)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if EXPERIMENT_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENT_DIR)

import analyze_results as ar
from boreft.bo.plotting import apply_bo_axes_style, save_plot
from boreft.search_wandb import (
    DEFAULT_ENTITY,
    DEFAULT_PROJECT,
    log_analysis_directory,
)

DEFAULT_SEARCH_DIR = REPO_ROOT / "experiments" / "outputs" / "semantle" / "search"
DEFAULT_SWEEP_DIR = REPO_ROOT / "experiments" / "outputs" / "semantle" / "sweep"
DEFAULT_OUT_DIR = (
    REPO_ROOT / "experiments" / "outputs" / "semantle" / "analysis" / "ladders"
)
DEFAULT_CANONICAL_SLUG = ar.DEFAULT_BOREFT_CONFIG
DEFAULT_TRAIN_OVERRIDE = ar.DEFAULT_BOREFT_TRAIN_OVERRIDE
DEFAULT_PAPER_DIR = REPO_ROOT / "notes" / "theory" / "figures"
DEFAULT_ERANK_JSON = (
    REPO_ROOT / "data" / "semantle" / "analysis" / "qwen_train_embed_effective_rank.json"
)
DEFAULT_PAPER_N = DEFAULT_PAPER_DIR / "search_n_ladder.pdf"
DEFAULT_PAPER_RANK = DEFAULT_PAPER_DIR / "search_rank_ladder.pdf"
DEFAULT_PAPER_ERANK = DEFAULT_PAPER_DIR / "train_erank.pdf"

N_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072)
RANK_VALUES = (4, 8, 16, 32, 64, 128)
CANONICAL_N = 3072
CANONICAL_RANK = 64
COSINE_COLOR = "#0072B2"
EM_COLOR = "#D55E00"
COVERAGE_COLOR = "0.45"
ERANK_COLOR = "#0072B2"
DEFAULT_COVERAGE_JSON = (
    REPO_ROOT / "data" / "semantle" / "analysis" / "cov_interp" / "summary.json"
)


def n_method(size: int) -> str:
    return f"n{size}"


def rank_method(rank: int) -> str:
    return f"r{rank}"


def n_dir_name(size: int) -> str:
    return f"boreft_N{size}"


def rank_dir_name(rank: int) -> str:
    return f"boreft_rank{rank}"


# Sequential palettes; canonical N=3072 / rank 64 stay the paper BOReFT blue.
N_COLORS = (
    "#440154",
    "#481567",
    "#482677",
    "#3f4a8a",
    "#31688e",
    "#26828e",
    "#1f9e89",
    "#35b779",
    "#6ece58",
    "#b5de2b",
    "#fde725",
    "#dce319",
    "#0072B2",
)
RANK_COLORS = (
    "#0d0887",
    "#6a00a8",
    "#b12a90",
    "#e16462",
    "#0072B2",
    "#f0f921",
)


def register_ladder_styles() -> None:
    for size, color in zip(N_SIZES, N_COLORS):
        method = n_method(size)
        ar.DISPLAY[method] = (
            f"N={size} (main)" if size == CANONICAL_N else f"N={size}"
        )
        ar.COLORS[method] = color
    for rank, color in zip(RANK_VALUES, RANK_COLORS):
        method = rank_method(rank)
        ar.DISPLAY[method] = (
            f"rank={rank} (main)" if rank == CANONICAL_RANK else f"rank={rank}"
        )
        ar.COLORS[method] = color


register_ladder_styles()


def method_x(method: str) -> int:
    if method.startswith("n") or method.startswith("r"):
        return int(method[1:])
    raise ValueError(f"expected n*/r* method, got {method!r}")


def ordered_methods(
    table: Sequence[dict[str, Any]],
    values: Sequence[int],
    name_fn,
) -> list[str]:
    present = {row["method"] for row in table}
    return [name_fn(value) for value in values if name_fn(value) in present]


def joint_search_series(
    table: Sequence[dict[str, Any]],
    methods: Sequence[str],
) -> tuple[list[int], list[float], list[float]]:
    by_method = {row["method"]: row for row in table}
    xs: list[int] = []
    cosine: list[float] = []
    exact: list[float] = []
    for method in methods:
        row = by_method.get(method)
        if row is None:
            continue
        block = row.get("all") or {}
        if "mean_best" not in block or "success_rate" not in block:
            continue
        xs.append(method_x(method))
        cosine.append(float(block["mean_best"]))
        exact.append(float(block["success_rate"]))
    return xs, cosine, exact


def _nested_get(payload: dict[str, Any], *keys: str) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def load_coverage_vs_n(path: Path) -> dict[int, float]:
    payload = json.loads(path.read_text())
    out: dict[int, float] = {}
    for row in payload.get("conditions") or []:
        name = str(row.get("name") or "")
        if not name.startswith("n") or not name[1:].isdigit():
            continue
        median = _nested_get(
            row, "coverage", "eval_test", "summary", "vertices", "median"
        )
        if median is None:
            continue
        out[int(name[1:])] = float(median)
    return out


def coverage_series(
    xs: Sequence[int], coverage: dict[int, float]
) -> tuple[list[int], list[float]]:
    cov_xs = [x for x in xs if x in coverage]
    return cov_xs, [coverage[x] for x in cov_xs]


def load_erank_results(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = []
    for row in payload.get("results") or []:
        n = int(row.get("n_drawn") or row.get("train_n_samples") or 0)
        if n not in N_SIZES:
            continue
        rows.append(
            {
                "n_drawn": n,
                "effective_rank": float(row["effective_rank"]),
            }
        )
    rows.sort(key=lambda item: item["n_drawn"])
    return rows


def _log2_xaxis(axis, xs: Sequence[int], *, rotate: bool) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xticks(list(xs))
    axis.set_xticklabels(
        [str(x) for x in xs],
        rotation=45 if rotate else 0,
        ha="right" if rotate else "center",
    )
    if xs:
        axis.set_xlim(min(xs) / 1.15, max(xs) * 1.15)


def plot_joint_search_ladder(
    plt,
    xs: Sequence[int],
    cosine: Sequence[float],
    exact: Sequence[float],
    path: Path,
    *,
    xlabel: str,
    rotate_xticks: bool,
    coverage_xs: Optional[Sequence[int]] = None,
    coverage: Optional[Sequence[float]] = None,
) -> tuple[Path, Path]:
    fig, axis = plt.subplots(figsize=(6.6, 3.8))
    axis.plot(
        xs,
        cosine,
        color=COSINE_COLOR,
        linestyle="-",
        marker="o",
        markersize=6,
        linewidth=2.2,
        label="Mean best cosine",
        zorder=3,
    )
    axis.plot(
        xs,
        exact,
        color=EM_COLOR,
        linestyle="--",
        marker="s",
        markersize=5.5,
        linewidth=2.2,
        label="Exact match",
        zorder=3,
    )
    if coverage_xs and coverage:
        axis.plot(
            coverage_xs,
            coverage,
            color=COVERAGE_COLOR,
            alpha=0.8,
            linestyle=":",
            marker="D",
            markersize=3.5,
            linewidth=2.2,
            label="Coverage error",
            zorder=2,
        )
    _log2_xaxis(axis, xs, rotate=rotate_xticks)
    axis.set_ylim(0.0, 1.0)
    apply_bo_axes_style(axis, xlabel=xlabel, ylabel="Search performance")
    axis.legend(
        loc="lower left" if coverage_xs and coverage else "upper left",
        frameon=True,
        handlelength=2.8,
        borderaxespad=0.4,
    )
    fig.tight_layout()
    pdf_path, png_path = save_plot(fig, path)
    plt.close(fig)
    return pdf_path, png_path


def plot_train_erank(
    plt,
    results: Sequence[dict[str, Any]],
    path: Path,
) -> tuple[Path, Path]:
    xs = [int(row["n_drawn"]) for row in results]
    ys = [float(row["effective_rank"]) for row in results]
    fig, axis = plt.subplots(figsize=(6.6, 3.8))
    axis.plot(
        xs,
        ys,
        color=ERANK_COLOR,
        linestyle="-",
        marker="o",
        markersize=6,
        linewidth=2.2,
        zorder=3,
    )
    _log2_xaxis(axis, xs, rotate=True)
    y_min = min(ys) if ys else 0.0
    y_max = max(ys) if ys else 1.0
    pad = max(0.05 * (y_max - y_min), 0.25)
    axis.set_ylim(max(0.0, y_min - pad), y_max + pad)
    apply_bo_axes_style(
        axis,
        xlabel="Training set size",
        ylabel="Effective rank",
    )
    fig.tight_layout()
    pdf_path, png_path = save_plot(fig, path)
    plt.close(fig)
    return pdf_path, png_path


def copy_paper_figure(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    png_src = src.with_suffix(".png")
    if png_src.is_file():
        shutil.copy2(png_src, dest.with_suffix(".png"))


def remap_method(runs: list[dict], method: str) -> list[dict]:
    out = []
    for run in runs:
        row = dict(run)
        row["method"] = method
        out.append(row)
    return out


def load_canonical_runs(
    sweep_dir: Path,
    canonical_slug: str,
    train_override: str,
) -> tuple[list[dict], Optional[str], Optional[Path]]:
    canonical_dir = sweep_dir / canonical_slug
    if not canonical_dir.is_dir():
        return [], None, None
    runs = list(ar.iter_runs({"boreft": canonical_dir}))
    runs, train_override_used = ar.replace_boreft_train(
        runs, sweep_dir, canonical_slug, train_override
    )
    return runs, train_override_used, canonical_dir


def load_ladder_runs(
    *,
    search_dir: Path,
    sweep_dir: Path,
    canonical_slug: str,
    train_override: str,
) -> tuple[list[dict], dict[str, Path], Optional[str]]:
    method_dirs: dict[str, Path] = {}
    runs: list[dict] = []
    canonical_runs, train_override_used, canonical_dir = load_canonical_runs(
        sweep_dir, canonical_slug, train_override
    )
    if canonical_runs:
        if canonical_dir is not None:
            method_dirs[n_method(CANONICAL_N)] = canonical_dir
            method_dirs[rank_method(CANONICAL_RANK)] = canonical_dir
        runs.extend(remap_method(canonical_runs, n_method(CANONICAL_N)))
        runs.extend(remap_method(canonical_runs, rank_method(CANONICAL_RANK)))

    for size in N_SIZES:
        if size == CANONICAL_N:
            continue
        path = search_dir / n_dir_name(size)
        if not path.is_dir():
            continue
        found = list(ar.iter_runs({n_dir_name(size): path}))
        if not found:
            continue
        method_dirs[n_method(size)] = path
        runs.extend(remap_method(found, n_method(size)))

    for rank in RANK_VALUES:
        if rank == CANONICAL_RANK:
            continue
        path = search_dir / rank_dir_name(rank)
        if not path.is_dir():
            continue
        found = list(ar.iter_runs({rank_dir_name(rank): path}))
        if not found:
            continue
        method_dirs[rank_method(rank)] = path
        runs.extend(remap_method(found, rank_method(rank)))

    return runs, method_dirs, train_override_used


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-dir", type=Path, default=DEFAULT_SEARCH_DIR)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--canonical-slug", default=DEFAULT_CANONICAL_SLUG)
    parser.add_argument(
        "--canonical-train-override",
        default=DEFAULT_TRAIN_OVERRIDE,
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--from-metrics",
        type=Path,
        nargs="?",
        const=DEFAULT_OUT_DIR / "metrics.json",
        default=None,
        help="Rebuild paper line plots from a saved metrics.json "
        "(default path if flag is set with no value).",
    )
    parser.add_argument(
        "--erank-json",
        type=Path,
        default=DEFAULT_ERANK_JSON,
        help="Qwen train-embed eRank JSON for the paper eRank figure.",
    )
    parser.add_argument(
        "--coverage-json",
        type=Path,
        default=DEFAULT_COVERAGE_JSON,
        help="cov_interp summary JSON for the N-ladder coverage-error line.",
    )
    parser.add_argument(
        "--copy-paper",
        action="store_true",
        help="Copy paper PDFs into notes/theory/figures/.",
    )
    parser.add_argument("--paper-n-figure", type=Path, default=DEFAULT_PAPER_N)
    parser.add_argument("--paper-rank-figure", type=Path, default=DEFAULT_PAPER_RANK)
    parser.add_argument("--paper-erank-figure", type=Path, default=DEFAULT_PAPER_ERANK)
    parser.add_argument("--n-only", action="store_true")
    parser.add_argument("--rank-only", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-force", action="store_true")
    parser.add_argument("--wandb-entity", default=DEFAULT_ENTITY)
    parser.add_argument("--wandb-project", default=DEFAULT_PROJECT)
    parser.add_argument("--wandb-dir", type=Path, default=None)
    return parser.parse_args()


def write_ladder_figures(
    plt,
    overlay: dict,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ar.plot_anytime(
        plt,
        overlay,
        "best",
        out_dir / "anytime_best.png",
        shared_legend=True,
        suptitle="",
    )
    ar.plot_anytime(
        plt,
        overlay,
        "success",
        out_dir / "anytime_success.png",
        shared_legend=True,
        suptitle="",
    )
    ar.plot_grouped_bars(
        plt,
        overlay["table"],
        "success_rate",
        None,
        "Exact-match success (%)",
        "Fraction of runs that proposed the gold word",
        out_dir / "success_rate.png",
        ylim=(0, 105),
        percent=True,
    )
    ar.plot_grouped_bars(
        plt,
        overlay["table"],
        "mean_best",
        "sem_best",
        "Best embedding cosine",
        "Final best cosine (mean ± 1 SD across 3 seeds)",
        out_dir / "mean_best_score.png",
        ylim=(0.55, 1.02),
    )


def write_paper_ladder_plots(
    plt,
    table: Sequence[dict[str, Any]],
    out_dir: Path,
    *,
    erank_json: Optional[Path],
    n_only: bool,
    rank_only: bool,
    coverage_json: Optional[Path] = None,
) -> dict[str, Path]:
    written: dict[str, Path] = {}
    n_methods = ordered_methods(table, N_SIZES, n_method)
    rank_methods = ordered_methods(table, RANK_VALUES, rank_method)
    coverage: dict[int, float] = {}
    if coverage_json is not None and coverage_json.is_file():
        coverage = load_coverage_vs_n(coverage_json)
    if n_methods and not rank_only:
        xs, cosine, exact = joint_search_series(table, n_methods)
        cov_xs, cov_ys = coverage_series(xs, coverage)
        pdf, _png = plot_joint_search_ladder(
            plt,
            xs,
            cosine,
            exact,
            out_dir / "search_n_ladder.pdf",
            xlabel="Training set size",
            rotate_xticks=True,
            coverage_xs=cov_xs,
            coverage=cov_ys,
        )
        written["n"] = pdf
    if rank_methods and not n_only:
        xs, cosine, exact = joint_search_series(table, rank_methods)
        pdf, _png = plot_joint_search_ladder(
            plt,
            xs,
            cosine,
            exact,
            out_dir / "search_rank_ladder.pdf",
            xlabel="Rank",
            rotate_xticks=False,
        )
        written["rank"] = pdf
    if erank_json is not None and erank_json.is_file() and not rank_only:
        results = load_erank_results(erank_json)
        if results:
            pdf, _png = plot_train_erank(plt, results, out_dir / "train_erank.pdf")
            written["erank"] = pdf
    return written


def copy_paper_ladders(
    written: dict[str, Path],
    *,
    n_dest: Path,
    rank_dest: Path,
    erank_dest: Path,
) -> None:
    mapping = {"n": n_dest, "rank": rank_dest, "erank": erank_dest}
    for key, dest in mapping.items():
        src = written.get(key)
        if src is None:
            continue
        copy_paper_figure(src, dest)
        print(f"copied paper figure → {dest}")


def main() -> int:
    args = parse_args()
    if args.n_only and args.rank_only:
        print("choose at most one of --n-only / --rank-only", file=sys.stderr)
        return 1
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    erank_json = args.erank_json.expanduser().resolve() if args.erank_json else None
    coverage_json = (
        args.coverage_json.expanduser().resolve() if args.coverage_json else None
    )
    if args.copy_paper and (erank_json is None or not erank_json.is_file()) and not args.rank_only:
        print(f"missing eRank JSON for paper figure: {erank_json}", file=sys.stderr)
        return 1

    plt = ar._plt()
    if args.from_metrics is not None:
        metrics_path = args.from_metrics.expanduser().resolve()
        if not metrics_path.is_file():
            print(f"metrics not found: {metrics_path}", file=sys.stderr)
            return 1
        payload = json.loads(metrics_path.read_text())
        table = payload.get("table") or []
        if not table:
            print(f"no table in {metrics_path}", file=sys.stderr)
            return 1
        n_order = ordered_methods(table, N_SIZES, n_method)
        rank_order = ordered_methods(table, RANK_VALUES, rank_method)
        print(f"loaded metrics ← {metrics_path}")
        if n_order and not args.rank_only:
            print("N ladder")
            by_method = {row["method"]: row for row in table}
            ar.print_table([by_method[name] for name in n_order if name in by_method])
            print()
        if rank_order and not args.n_only:
            print("rank ladder")
            by_method = {row["method"]: row for row in table}
            ar.print_table([by_method[name] for name in rank_order if name in by_method])
            print()
    else:
        search_dir = args.search_dir.resolve()
        sweep_dir = args.sweep_dir.resolve()
        runs, method_dirs, train_override_used = load_ladder_runs(
            search_dir=search_dir,
            sweep_dir=sweep_dir,
            canonical_slug=str(args.canonical_slug or "").strip(),
            train_override=str(args.canonical_train_override or "").strip(),
        )
        if not runs:
            print("no ladder search runs found", file=sys.stderr)
            return 1

        n_methods = [n_method(size) for size in N_SIZES]
        rank_methods = [rank_method(rank) for rank in RANK_VALUES]
        present = {run["method"] for run in runs}
        n_order = [name for name in n_methods if name in present]
        rank_order = [name for name in rank_methods if name in present]
        if args.n_only:
            keep = set(n_order)
        elif args.rank_only:
            keep = set(rank_order)
        else:
            keep = set(n_order) | set(rank_order)
        runs = [run for run in runs if run["method"] in keep]
        target_dirs = [sweep_dir, search_dir, *method_dirs.values()]
        if train_override_used:
            target_dirs.insert(0, sweep_dir / train_override_used)
        summary = ar.summarize(runs, target_dirs)
        table = summary["table"]

        if n_order and not args.rank_only:
            n_dir = out_dir / "n"
            write_ladder_figures(plt, ar.subset_summary(summary, n_order), n_dir)
            print("N ladder")
            ar.print_table(ar.subset_summary(summary, n_order)["table"])
            print()
        if rank_order and not args.n_only:
            rank_dir = out_dir / "rank"
            write_ladder_figures(plt, ar.subset_summary(summary, rank_order), rank_dir)
            print("rank ladder")
            ar.print_table(ar.subset_summary(summary, rank_order)["table"])
            print()

        payload = {
            "search_dir": str(search_dir),
            "sweep_dir": str(sweep_dir),
            "canonical_slug": args.canonical_slug or None,
            "canonical_train_override": train_override_used,
            "method_dirs": {method: str(path) for method, path in method_dirs.items()},
            "n_methods": n_order,
            "rank_methods": rank_order,
            "budget": ar.BUDGET,
            "warmstart": ar.WARMSTART,
            **ar.json_ready(summary),
        }
        (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(f"loaded {len(runs)} runs")
        for method, path in method_dirs.items():
            if method in keep:
                print(f"{ar.label(method)} ← {path}")

    paper = write_paper_ladder_plots(
        plt,
        table,
        out_dir,
        erank_json=erank_json,
        n_only=args.n_only,
        rank_only=args.rank_only,
        coverage_json=coverage_json,
    )
    if args.copy_paper:
        missing = [
            key
            for key in (("n",) if not args.rank_only else ())
            + (("rank",) if not args.n_only else ())
            + (("erank",) if not args.rank_only else ())
            if key not in paper
        ]
        if missing:
            print(f"missing paper plots: {', '.join(missing)}", file=sys.stderr)
            return 1
        copy_paper_ladders(
            paper,
            n_dest=args.paper_n_figure.expanduser().resolve(),
            rank_dest=args.paper_rank_figure.expanduser().resolve(),
            erank_dest=args.paper_erank_figure.expanduser().resolve(),
        )
    print(f"wrote figures → {out_dir}")
    if not args.no_wandb:
        try:
            result = log_analysis_directory(
                out_dir,
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                group="semantle-analysis",
                name="semantle-ladder-comparison",
                wandb_dir=args.wandb_dir,
                force=args.wandb_force,
            )
        except Exception as exc:
            print(f"W&B comparison failed: {exc}", file=sys.stderr)
        else:
            if result.get("status") == "logged":
                print(
                    f"W&B comparison run {result.get('run_id')} "
                    f"({result.get('n_images')} plots)"
                )
            elif result.get("status") == "error":
                print(f"W&B comparison failed: {result.get('error')}", file=sys.stderr)
            else:
                print(f"W&B comparison skipped: {result.get('status')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
