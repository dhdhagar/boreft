#!/usr/bin/env python3
"""Compare the AutoDiscovery prompt-check runs against the existing full runs.

Truncates each old seed to the new run's verification count so a budget-150
check is comparable to the first 150 steps of the original budget-500 search.

    python experiments/semantle/compare_autodiscovery_prompt.py
    python experiments/semantle/compare_autodiscovery_prompt.py \\
        --new-root experiments/outputs/semantle/search/autodiscovery_prompt_check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NEW = (
    REPO_ROOT / "experiments" / "outputs" / "semantle" / "search" / "autodiscovery_prompt_check"
)
DEFAULT_OLD = (
    REPO_ROOT / "experiments" / "outputs" / "semantle" / "search" / "autodiscovery"
)
OPRO_HEADING = "Previous solutions and scores, ordered from lowest score to highest:"
OLD_BRANCH_PHRASE = "build on this branch"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _truncate(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    return rows[:n]


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no observations")
    warm = [row for row in rows if row.get("phase") == "warmstart"]
    search = [row for row in rows if row.get("phase") == "search"]
    warm_best = max((float(row["score"]) for row in warm), default=float("nan"))
    best_row = max(rows, key=lambda row: float(row["score"]))
    peak = warm_best
    n_peaks = 0
    peak_words: list[str] = []
    for row in search:
        score = float(row["score"])
        if score > peak + 1e-12:
            peak = score
            n_peaks += 1
            peak_words.append(str(row.get("solution") or ""))
    prompts = [
        str((row.get("candidate_metadata") or {}).get("prompt") or "")
        for row in search
    ]
    n_opro = sum(1 for prompt in prompts if OPRO_HEADING in prompt)
    n_old = sum(1 for prompt in prompts if OLD_BRANCH_PHRASE in prompt)
    best_sol = str(best_row.get("solution") or "")
    best_idx = int(best_row["index"])
    later = [row for row in search if int(row["index"]) > best_idx]
    later_with_best = sum(
        1
        for row in later
        if best_sol
        and best_sol
        in str((row.get("candidate_metadata") or {}).get("prompt") or "")
    )
    n_hist = [
        sum(1 for line in prompt.splitlines() if line.startswith("solution: "))
        for prompt in prompts
    ]
    sample_prompt = ""
    if prompts:
        sample_prompt = max(prompts, key=lambda prompt: prompt.count("solution: "))
    found = any(bool((row.get("components") or {}).get("exact_match")) for row in rows)
    return {
        "n_obs": len(rows),
        "n_search": len(search),
        "warm_best": warm_best,
        "best_score": float(best_row["score"]),
        "best_solution": best_sol,
        "delta": float(best_row["score"]) - warm_best,
        "n_peaks": n_peaks,
        "peak_words": peak_words[-5:],
        "found": found,
        "n_opro_prompts": n_opro,
        "n_old_prompts": n_old,
        "n_later": len(later),
        "n_later_with_best": later_with_best,
        "mean_hist": (sum(n_hist) / len(n_hist)) if n_hist else 0.0,
        "max_hist": max(n_hist) if n_hist else 0,
        "sample_prompt": sample_prompt,
    }


def _fmt(value: float) -> str:
    if value != value:  # NaN
        return "   nan"
    return f"{value:7.3f}"


def _seed_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("*/seed_*") if path.is_dir())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-root", type=Path, default=DEFAULT_NEW)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD)
    args = parser.parse_args()
    new_root = args.new_root
    old_root = args.old_root
    if not new_root.is_absolute():
        new_root = REPO_ROOT / new_root
    if not old_root.is_absolute():
        old_root = REPO_ROOT / old_root

    seeds = _seed_dirs(new_root)
    if not seeds:
        print(f"no prompt-check seeds under {new_root}")
        print("submit with experiments/semantle/autodiscovery_prompt_check.sh first")
        return 1

    print(f"new  {new_root}")
    print(f"old  {old_root}  (truncated to the new run's verification count)")
    print()
    header = (
        f"{'run':28s} {'n':>4s} {'old_best':>8s} {'new_best':>8s} "
        f"{'d_old':>7s} {'d_new':>7s} {'hist':>8s} {'pk_old':>6s} {'pk_new':>6s} "
        f"{'found':>7s} {'prompt':>8s} {'best_later':>11s}"
    )
    print(header)
    print("-" * len(header))

    deltas_old: list[float] = []
    deltas_new: list[float] = []
    bests_old: list[float] = []
    bests_new: list[float] = []
    sample_prompt = ""
    n_prompt_ok = 0
    n_compared = 0

    for new_dir in seeds:
        rel = new_dir.relative_to(new_root)
        new_obs = new_dir / "observations.jsonl"
        old_obs = old_root / rel / "observations.jsonl"
        label = str(rel)
        if not new_obs.is_file():
            print(f"{label:28s}  missing new observations.jsonl")
            continue
        new_rows = _load_jsonl(new_obs)
        new = _stats(new_rows)
        if len(new["sample_prompt"]) > len(sample_prompt):
            sample_prompt = new["sample_prompt"]
        prompt_ok = new["n_old_prompts"] == 0 and new["n_opro_prompts"] == new["n_search"]
        if prompt_ok:
            n_prompt_ok += 1
        if not old_obs.is_file():
            print(
                f"{label:28s} {new['n_obs']:4d} {'—':>8s} {_fmt(new['best_score'])} "
                f"{'—':>7s} {_fmt(new['delta'])} {new['mean_hist']:4.1f}/{new['max_hist']:<2d} "
                f"{'—':>6s} {new['n_peaks']:6d} "
                f"{'yes' if new['found'] else 'no':>7s} "
                f"{'ok' if prompt_ok else 'BAD':>8s} "
                f"{new['n_later_with_best']:3d}/{new['n_later']:<7d}  "
                f"(no old run)  best={new['best_solution']!r}"
            )
            continue
        old_rows = _truncate(_load_jsonl(old_obs), new["n_obs"])
        old = _stats(old_rows)
        n_compared += 1
        deltas_old.append(old["delta"])
        deltas_new.append(new["delta"])
        bests_old.append(old["best_score"])
        bests_new.append(new["best_score"])
        found_s = f"{'Y' if old['found'] else 'n'}/{'Y' if new['found'] else 'n'}"
        later = f"{new['n_later_with_best']:3d}/{new['n_later']:<7d}"
        hist = f"{old['mean_hist']:.1f}->{new['mean_hist']:.1f}"
        print(
            f"{label:28s} {new['n_obs']:4d} {_fmt(old['best_score'])} {_fmt(new['best_score'])} "
            f"{_fmt(old['delta'])} {_fmt(new['delta'])} {hist:>8s} {old['n_peaks']:6d} {new['n_peaks']:6d} "
            f"{found_s:>7s} "
            f"{'ok' if prompt_ok else 'BAD':>8s} {later}  "
            f"old={old['best_solution']!r}  new={new['best_solution']!r}"
        )

    print()
    if n_compared:
        mean_old = sum(deltas_old) / n_compared
        mean_new = sum(deltas_new) / n_compared
        mean_best_old = sum(bests_old) / n_compared
        mean_best_new = sum(bests_new) / n_compared
        print(
            f"mean best     old={mean_best_old:.3f}  new={mean_best_new:.3f}  "
            f"({mean_best_new - mean_best_old:+.3f})"
        )
        print(
            f"mean Δ warm   old={mean_old:+.3f}  new={mean_new:+.3f}  "
            f"({mean_new - mean_old:+.3f})"
        )
        if mean_new > mean_old + 0.02 or mean_best_new > mean_best_old + 0.02:
            print("verdict: prompt change looks helpful on this slice; worth a full rerun")
        elif mean_new + 0.02 < mean_old:
            print("verdict: no gain vs the old prompt on this slice")
        else:
            print("verdict: roughly unchanged; look at best words / later-prompt reuse")
    print(f"OPRO-style prompts: {n_prompt_ok}/{len(seeds)} seeds")
    if sample_prompt:
        print()
        print("sample new search prompt:")
        print(sample_prompt[:900])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
