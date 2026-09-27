#!/usr/bin/env python3
"""Upload existing Semantle search/sweep seeds and analysis artifacts to W&B.

Walks ``experiments/outputs/semantle/search``, ``.../sweep``,
``data/semantle/analysis``, and ``data/molopt/analysis`` by default. Search
seeds become ``job_type=search`` runs; analysis folders become
``job_type=analysis``. Already-uploaded items (``wandb_meta.json``) are skipped.

    python experiments/semantle/log_wandb.py
    python experiments/semantle/log_wandb.py --dry-run
    python experiments/semantle/log_wandb.py --skip-sweep
    python experiments/semantle/log_wandb.py --skip-search --skip-sweep
    python experiments/semantle/analyze_results.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = str(REPO_ROOT / "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from boreft.search_wandb import (
    DEFAULT_ENTITY,
    DEFAULT_PROJECT,
    log_analysis_dir,
    log_search_tree,
)

DEFAULT_SEARCH_DIR = REPO_ROOT / "experiments" / "outputs" / "semantle" / "search"
DEFAULT_SWEEP_DIR = REPO_ROOT / "experiments" / "outputs" / "semantle" / "sweep"
DEFAULT_ANALYSIS_DIRS = (
    REPO_ROOT / "data" / "semantle" / "analysis",
    REPO_ROOT / "data" / "molopt" / "analysis",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-dir", type=Path, default=DEFAULT_SEARCH_DIR)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        action="append",
        default=[],
        help="Offline analysis directory (repeatable). "
        "Default: data/semantle/analysis and data/molopt/analysis.",
    )
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--wandb-project", default=DEFAULT_PROJECT)
    parser.add_argument("--wandb-entity", default=DEFAULT_ENTITY)
    parser.add_argument("--wandb-dir", default=None)
    parser.add_argument(
        "--history-stride",
        type=int,
        default=1,
        help="Log every Nth verification (1 = full anytime curve).",
    )
    parser.add_argument(
        "--images",
        action="store_true",
        help="Upload per-seed best_so_far.png (off by default; search comparison "
        "figures are uploaded by analyze_results.py).",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _tally(results: list[dict], counts: dict[str, int]) -> None:
    for row in results:
        counts[str(row.get("status"))] = counts.get(str(row.get("status")), 0) + 1
        status = row.get("status")
        name = row.get("name") or row.get("seed_dir")
        if status in ("logged", "dry_run"):
            extra = row.get("n_verifications") or row.get("n_images")
            print(f"[{status}] {name}  n={extra}")
        elif status == "skipped":
            print(f"[skipped] {name}  ({row.get('run_id')})")
        elif status == "empty":
            print(f"[empty] {name}")
        elif status == "error":
            print(f"[error] {name}: {row.get('error')}")


def main() -> int:
    args = parse_args()
    roots: list[Path] = []
    if not args.skip_search:
        path = args.search_dir.resolve()
        if path.is_dir():
            roots.append(path)
        else:
            print(f"skip missing search dir: {path}", file=sys.stderr)
    if not args.skip_sweep:
        path = args.sweep_dir.resolve()
        if path.is_dir():
            roots.append(path)
        else:
            print(f"skip missing sweep dir: {path}", file=sys.stderr)
    analysis_dirs = (
        []
        if args.skip_analysis
        else [path.resolve() for path in (args.analysis_dir or DEFAULT_ANALYSIS_DIRS)]
    )
    counts: dict[str, int] = {}
    did_work = False
    if roots:
        did_work = True
        _tally(
            log_search_tree(
                roots,
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                wandb_dir=args.wandb_dir,
                history_stride=args.history_stride,
                log_images=args.images,
                force=args.force,
                dry_run=args.dry_run,
            ),
            counts,
        )
    for directory in analysis_dirs:
        if not directory.is_dir():
            print(f"skip missing analysis dir: {directory}", file=sys.stderr)
            continue
        did_work = True
        _tally(
            log_analysis_dir(
                directory,
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                wandb_dir=args.wandb_dir,
                force=args.force,
                dry_run=args.dry_run,
            ),
            counts,
        )
    if not did_work:
        print("no search/sweep/analysis directories to upload", file=sys.stderr)
        return 1
    print()
    if counts:
        print(
            "items "
            + "  ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        )
    else:
        print("items none")
    return 1 if counts.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
