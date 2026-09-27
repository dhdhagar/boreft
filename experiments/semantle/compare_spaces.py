#!/usr/bin/env python3
"""Overlay anytime curves for BOReFT search in different trained spaces.

Compares the published canonical protocol (sweep winner ``s1_t0_ard_d64``,
with the train rerun when present) against SDPO-off, no-VAE,
reconstruction-off, reconstruction-off at T=1, and no-encoder searches
written under ``search/boreft_*``. Those trees are separate from
``search/boreft/``.
The joint no-VAE + no-distillation overlay is kept in the loader but omitted
from the paper figure.

    python experiments/semantle/compare_spaces.py
    python experiments/semantle/compare_spaces.py --no-wandb
    python experiments/semantle/compare_spaces.py --from-wandb --copy-paper --no-wandb
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = str(REPO_ROOT / "src")
EXPERIMENT_DIR = str(Path(__file__).resolve().parent)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if EXPERIMENT_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENT_DIR)

import analyze_results as ar
from boreft.search_wandb import (
    DEFAULT_ENTITY,
    DEFAULT_PROJECT,
    log_analysis_directory,
)

DEFAULT_SEARCH_DIR = REPO_ROOT / "experiments" / "outputs" / "semantle" / "search"
DEFAULT_SWEEP_DIR = REPO_ROOT / "experiments" / "outputs" / "semantle" / "sweep"
DEFAULT_OUT_DIR = (
    REPO_ROOT / "experiments" / "outputs" / "semantle" / "analysis" / "spaces"
)
DEFAULT_CANONICAL_SLUG = ar.DEFAULT_BOREFT_CONFIG
DEFAULT_TRAIN_OVERRIDE = ar.DEFAULT_BOREFT_TRAIN_OVERRIDE
DEFAULT_PAPER_FIGURE = REPO_ROOT / "notes" / "theory" / "figures" / "search_cosine.pdf"

# Paper overlay: same T=0 protocol, every trained-space ablation.
PAPER_SPACE_ORDER = (
    "boreft",
    "boreft_sdpo0",
    "boreft_novae",
    # "boreft_sdpo0_novae",
    "boreft_ce0",
    "boreft_noenc",
)
SPACE_ORDER = PAPER_SPACE_ORDER + ("boreft_ce0_t1",)
WANDB_NAME_SPECS = (
    ("boreft", r"^s1_t0_ard_d64_rerun_train-"),
    ("boreft", r"^s1_t0_ard_d64_test-"),
    ("boreft_sdpo0", r"^boreft_sdpo0_(train|test)-"),
    ("boreft_novae", r"^boreft_novae_(train|test)-"),
    # ("boreft_sdpo0_novae", r"^boreft_sdpo0_novae_(train|test)-"),
    ("boreft_ce0", r"^boreft_ce0_(train|test)-"),
    ("boreft_noenc", r"^boreft_noenc_(train|test)-"),
    ("boreft_ce0_t1", r"^boreft_ce0_t1_(train|test)-"),
)
WANDB_NAME_RE = re.compile(
    r"^(?:s1_t0_ard_d64_rerun_|s1_t0_ard_d64_|"
    r"boreft_sdpo0_novae_|boreft_sdpo0_|boreft_novae_|"
    r"boreft_ce0_t1_|boreft_ce0_|boreft_noenc_)"
    r"(train|test)-(.+)_seed(\d+)$"
)
ar.DISPLAY.update(
    {
        "canonical": "BOReFT",
        "boreft": "BOReFT",
        "boreft_sdpo0": "w/o self-distillation",
        "boreft_novae": "w/o variational training",
        "boreft_sdpo0_novae": "w/o both",
        "boreft_ce0": "w/o reconstruction",
        "boreft_ce0_t1": "w/o reconstruction (T=1)",
        "boreft_noenc": "w/o shared encoder",
    }
)
ar.COLORS.update(
    {
        "canonical": "#0072B2",
        "boreft": "#0072B2",
        "boreft_sdpo0": "#D55E00",
        "boreft_novae": "#009E73",
        "boreft_sdpo0_novae": "#CC79A7",
        "boreft_ce0": "#E69F00",
        "boreft_ce0_t1": "#56B4E9",
        "boreft_noenc": "#882255",
    }
)


def remap_method(runs: list[dict], method: str) -> list[dict]:
    out = []
    for run in runs:
        row = dict(run)
        row["method"] = method
        out.append(row)
    return out


def parse_wandb_display_name(name: str) -> Optional[tuple[str, str, int]]:
    match = WANDB_NAME_RE.match(str(name or ""))
    if not match:
        return None
    split, target, seed = match.group(1), match.group(2), int(match.group(3))
    return split, target, seed


def curves_from_history(rows: list[dict], budget: int = ar.BUDGET) -> dict:
    import numpy as np

    prepared: list[tuple[int, float, float, float]] = []
    steps = [
        int(row["_step"])
        for row in rows
        if row.get("search/verifications") is None and row.get("_step") is not None
    ]
    step_base = min(steps) if steps else 0
    for row in rows:
        if row.get("search/verifications") is not None:
            verifications = int(row["search/verifications"])
        elif row.get("_step") is not None:
            step = int(row["_step"])
            verifications = step + 1 if step_base == 0 else step
        else:
            continue
        prepared.append(
            (
                verifications,
                float(row.get("search/best_so_far") or 0.0),
                float(row.get("search/found") or 0.0),
                float(row.get("search/n_unique") or 0.0),
            )
        )
    points: dict[int, tuple[float, float, float]] = {}
    for verifications, best, found, unique in prepared:
        points[verifications] = (best, found, unique)
    best_curve = np.zeros(budget, dtype=np.float64)
    found_curve = np.zeros(budget, dtype=np.float64)
    unique_curve = np.zeros(budget, dtype=np.float64)
    last = (0.0, 0.0, 0.0)
    started = False
    for i in range(budget):
        step = i + 1
        if step in points:
            last = points[step]
            started = True
        if started:
            best_curve[i], found_curve[i], unique_curve[i] = last
    found_at = next(
        (i + 1 for i, value in enumerate(found_curve) if value >= 0.5),
        None,
    )
    n_verifications = max(points) if points else 0
    return {
        "best_curve": best_curve,
        "found_curve": found_curve,
        "unique_curve": unique_curve,
        "found": found_at is not None,
        "found_at": found_at,
        "best_score": float(best_curve[-1]) if points else 0.0,
        "best_text": "",
        "n_verifications": min(int(n_verifications), budget),
        "n_observations": len(points),
        "n_search": n_verifications,
        "n_repeat_samples": 0,
        "n_repeat_proposals": 0,
        "repeat_sample_rate": float("nan"),
        "repeat_proposal_rate": float("nan"),
        "n_unique": int(unique_curve[-1]),
    }


def _latest_wandb_runs(api, entity: str, project: str, pattern: str) -> list:
    seen: dict[str, object] = {}
    runs = api.runs(
        f"{entity}/{project}",
        filters={"displayName": {"$regex": pattern}},
        order="-created_at",
        per_page=100,
    )
    for run in runs:
        name = run.display_name
        if name not in seen:
            seen[name] = run
    return list(seen.values())


def _download_wandb_history(run, cache: dict[str, list[dict]]) -> list[dict]:
    run_id = run.id
    if run_id in cache:
        return cache[run_id]
    rows = run.history(
        keys=[
            "search/best_so_far",
            "search/found",
            "search/n_unique",
        ],
        pandas=False,
        samples=1000,
    )
    if hasattr(rows, "to_dict"):
        rows = rows.to_dict("records")
    cleaned = []
    for row in rows or []:
        clean = {}
        for key, value in dict(row).items():
            if value is None:
                continue
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, float) and value != value:
                continue
            clean[key] = value
        cleaned.append(clean)
    cache[run_id] = cleaned
    return cache[run_id]


def load_wandb_space_runs(
    *,
    entity: str,
    project: str,
    cache_path: Optional[Path] = None,
    include_t1: bool = False,
) -> tuple[list[dict], dict[str, str]]:
    import wandb

    api = wandb.Api()
    cache: dict[str, list[dict]] = {}
    if cache_path and cache_path.is_file():
        cache = json.loads(cache_path.read_text())
    selected: list[tuple[str, object]] = []
    method_ids: dict[str, list[str]] = {}
    specs = WANDB_NAME_SPECS if include_t1 else WANDB_NAME_SPECS[:-1]
    for method, pattern in specs:
        for run in _latest_wandb_runs(api, entity, project, pattern):
            parsed = parse_wandb_display_name(run.display_name)
            if parsed is None:
                continue
            selected.append((method, run))
            method_ids.setdefault(method, []).append(run.id)

    runs: list[dict] = []
    for index, (method, run) in enumerate(selected, start=1):
        parsed = parse_wandb_display_name(run.display_name)
        if parsed is None:
            continue
        split, target, seed = parsed
        if index == 1 or index % 15 == 0 or index == len(selected):
            print(
                f"wandb history {index}/{len(selected)} {run.display_name}",
                flush=True,
            )
        history = _download_wandb_history(run, cache)
        if cache_path and index % 15 == 0:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache) + "\n")
        stats = curves_from_history(rows=history)
        summary = dict(run.summary or {})
        if summary.get("search/found_target") and not stats["found"]:
            stats["found"] = True
            stats["found_at"] = summary.get("search/found_at") or stats["found_at"]
        if summary.get("search/n_unique") is not None:
            stats["n_unique"] = int(summary["search/n_unique"])
        runs.append(
            {
                "method": method,
                "split": split,
                "target": target,
                "seed": seed,
                "dir": run.url,
                **stats,
            }
        )
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache) + "\n")
    method_dirs = {
        method: f"wandb://{entity}/{project} ({len(ids)} runs)"
        for method, ids in method_ids.items()
    }
    return runs, method_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-dir", type=Path, default=DEFAULT_SEARCH_DIR)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--canonical-slug", default=DEFAULT_CANONICAL_SLUG)
    parser.add_argument(
        "--canonical-train-override",
        default=DEFAULT_TRAIN_OVERRIDE,
    )
    parser.add_argument("--sdpo0-dir", type=Path, default=None)
    parser.add_argument("--novae-dir", type=Path, default=None)
    parser.add_argument("--sdpo0-novae-dir", type=Path, default=None)
    parser.add_argument("--ce0-dir", type=Path, default=None)
    parser.add_argument("--ce0-t1-dir", type=Path, default=None)
    parser.add_argument("--noenc-dir", type=Path, default=None)
    parser.add_argument(
        "--from-wandb",
        action="store_true",
        help="Rebuild overlay curves from W&B search histories "
        "(latest finished run per display name).",
    )
    parser.add_argument(
        "--include-t1",
        action="store_true",
        help="Include the reconstruction-off T=1 protocol variant.",
    )
    parser.add_argument(
        "--copy-paper",
        action="store_true",
        help="Copy anytime_best.pdf onto notes/theory/figures/search_cosine.pdf "
        "(paper methods only).",
    )
    parser.add_argument("--paper-figure", type=Path, default=DEFAULT_PAPER_FIGURE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-force", action="store_true")
    parser.add_argument("--wandb-project", default=DEFAULT_PROJECT)
    parser.add_argument("--wandb-entity", default=DEFAULT_ENTITY)
    parser.add_argument("--wandb-dir", default=None)
    return parser.parse_args()


def load_space_runs(
    *,
    search_dir: Path,
    sweep_dir: Path,
    canonical_slug: str,
    train_override: str,
    sdpo0_dir: Optional[Path],
    novae_dir: Optional[Path],
    sdpo0_novae_dir: Optional[Path] = None,
    ce0_dir: Optional[Path] = None,
    ce0_t1_dir: Optional[Path] = None,
    noenc_dir: Optional[Path] = None,
) -> tuple[list[dict], dict[str, Path], Optional[str]]:
    method_dirs: dict[str, Path] = {}
    runs: list[dict] = []
    train_override_used: Optional[str] = None

    canonical_dir = sweep_dir / canonical_slug
    if canonical_dir.is_dir():
        method_dirs["boreft"] = canonical_dir
        canonical_runs = list(ar.iter_runs({"boreft": canonical_dir}))
        canonical_runs, train_override_used = ar.replace_boreft_train(
            canonical_runs, sweep_dir, canonical_slug, train_override
        )
        runs.extend(remap_method(canonical_runs, "boreft"))
    elif (search_dir / "boreft").is_dir():
        method_dirs["boreft"] = search_dir / "boreft"
        runs.extend(remap_method(list(ar.iter_runs({"boreft": search_dir / "boreft"})), "boreft"))

    sdpo0 = sdpo0_dir or (search_dir / "boreft_sdpo0")
    if sdpo0.is_dir():
        found = list(ar.iter_runs({"boreft_sdpo0": sdpo0}))
        if found:
            method_dirs["boreft_sdpo0"] = sdpo0
            runs.extend(found)

    novae = novae_dir or (search_dir / "boreft_novae")
    if novae.is_dir():
        found = list(ar.iter_runs({"boreft_novae": novae}))
        if found:
            method_dirs["boreft_novae"] = novae
            runs.extend(found)

    joint = sdpo0_novae_dir or (search_dir / "boreft_sdpo0_novae")
    if joint.is_dir():
        found = list(ar.iter_runs({"boreft_sdpo0_novae": joint}))
        if found:
            method_dirs["boreft_sdpo0_novae"] = joint
            runs.extend(found)

    ce0 = ce0_dir or (search_dir / "boreft_ce0")
    if ce0.is_dir():
        found = list(ar.iter_runs({"boreft_ce0": ce0}))
        if found:
            method_dirs["boreft_ce0"] = ce0
            runs.extend(found)

    ce0_t1 = ce0_t1_dir or (search_dir / "boreft_ce0_t1")
    if ce0_t1.is_dir():
        found = list(ar.iter_runs({"boreft_ce0_t1": ce0_t1}))
        if found:
            method_dirs["boreft_ce0_t1"] = ce0_t1
            runs.extend(found)

    noenc = noenc_dir or (search_dir / "boreft_noenc")
    if noenc.is_dir():
        found = list(ar.iter_runs({"boreft_noenc": noenc}))
        if found:
            method_dirs["boreft_noenc"] = noenc
            runs.extend(found)

    return runs, method_dirs, train_override_used


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    space_order = SPACE_ORDER if args.include_t1 else PAPER_SPACE_ORDER
    if args.from_wandb:
        runs, method_dirs = load_wandb_space_runs(
            entity=args.wandb_entity or None,
            project=args.wandb_project,
            cache_path=out_dir / "wandb_histories.json",
            include_t1=args.include_t1,
        )
        train_override_used = args.canonical_train_override or None
        target_dirs: list[Path] = []
        search_dir = args.search_dir
        sweep_dir = args.sweep_dir
    else:
        search_dir = args.search_dir.resolve()
        sweep_dir = args.sweep_dir.resolve()
        runs, method_dirs, train_override_used = load_space_runs(
            search_dir=search_dir,
            sweep_dir=sweep_dir,
            canonical_slug=str(args.canonical_slug or "").strip(),
            train_override=str(args.canonical_train_override or "").strip(),
            sdpo0_dir=args.sdpo0_dir.resolve() if args.sdpo0_dir else None,
            novae_dir=args.novae_dir.resolve() if args.novae_dir else None,
            sdpo0_novae_dir=(
                args.sdpo0_novae_dir.resolve() if args.sdpo0_novae_dir else None
            ),
            ce0_dir=args.ce0_dir.resolve() if args.ce0_dir else None,
            ce0_t1_dir=args.ce0_t1_dir.resolve() if args.ce0_t1_dir else None,
            noenc_dir=args.noenc_dir.resolve() if args.noenc_dir else None,
        )
        target_dirs = [sweep_dir, search_dir, *method_dirs.values()]
        if train_override_used:
            target_dirs.insert(0, sweep_dir / train_override_used)
    if not runs:
        print("no space-comparison runs found", file=sys.stderr)
        return 1

    methods = [name for name in space_order if any(r["method"] == name for r in runs)]
    methods += [
        name
        for name in sorted({r["method"] for r in runs})
        if name not in methods
    ]
    if args.copy_paper:
        methods = [name for name in PAPER_SPACE_ORDER if name in methods]
    summary = ar.summarize(runs, target_dirs)
    overlay = ar.subset_summary(summary, methods)

    plt = ar._plt()
    ar.plot_anytime(
        plt,
        overlay,
        "best",
        out_dir / "anytime_best.png",
        shared_legend=True,
        suptitle="",
        panel_titles=("Train", "Test"),
        axis_labelsize=18,
        title_size=18,
        title_bold=True,
        tick_labelsize=14,
        legend_fontsize=13,
        legend_frameon=False,
        legend_single_row=True,
        line_width=2.6,
        ylabel="Best similarity",
        spine_width=2.4,
        warmstart_labelsize=11,
        bold_legend={"BOReFT"},
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

    payload = {
        "search_dir": str(search_dir),
        "sweep_dir": str(sweep_dir),
        "canonical_slug": args.canonical_slug or None,
        "canonical_train_override": train_override_used,
        "method_dirs": {method: str(path) for method, path in method_dirs.items()},
        "source": "wandb" if args.from_wandb else "disk",
        "budget": ar.BUDGET,
        "warmstart": ar.WARMSTART,
        **ar.json_ready(summary),
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"loaded {len(runs)} runs")
    for method, path in method_dirs.items():
        print(f"{ar.label(method)} ← {path}")
    print(f"wrote figures → {out_dir}")
    if args.copy_paper:
        import shutil

        src = out_dir / "anytime_best.pdf"
        dest = args.paper_figure.expanduser().resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        print(f"copied paper figure → {dest}")
    print()
    ar.print_table(overlay["table"])
    if not args.no_wandb:
        try:
            result = log_analysis_directory(
                out_dir,
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                group="semantle-analysis",
                name="semantle-space-comparison",
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
