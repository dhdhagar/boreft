#!/usr/bin/env python3
"""Dump labeled warmstart words from the canonical Semantle search tree.

Writes ``{"targets": {target: {seed: [word, ...]}}}`` so later N/rank search
jobs can reuse the same 10 labeled words per ``(target, seed)``.

Train targets prefer ``s1_t0_ard_d64_rerun`` when that tree exists (same source
as the paper train metrics). Test targets stay on ``s1_t0_ard_d64``.

    python experiments/semantle/dump_canonical_warmstarts.py
    python experiments/semantle/dump_canonical_warmstarts.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CELL = (
    REPO_ROOT / "experiments" / "outputs" / "semantle" / "sweep" / "s1_t0_ard_d64"
)
DEFAULT_FALLBACK = (
    REPO_ROOT
    / "experiments"
    / "outputs"
    / "semantle"
    / "sweep"
    / "s1_t0_ard_d64_rerun"
)
DEFAULT_TARGETS = (
    REPO_ROOT / "experiments" / "outputs" / "semantle" / "sweep" / "targets.json"
)
DEFAULT_OUT = (
    REPO_ROOT
    / "experiments"
    / "outputs"
    / "semantle"
    / "sweep"
    / "canonical_warmstarts.json"
)


def _norm(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def warmstart_words(observations: list[dict]) -> list[str]:
    words: list[str] = []
    for row in observations:
        kind = row.get("source") or row.get("phase") or ""
        if kind != "warmstart":
            continue
        word = (row.get("components") or {}).get("warmstart_word") or row.get(
            "decoded"
        )
        if word and str(word).strip():
            words.append(str(word).strip())
    return words


def dump_cell(cell: Path) -> dict[str, dict[str, list[str]]]:
    targets: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for run_dir in sorted(p for p in cell.iterdir() if p.is_dir()):
        if "-" not in run_dir.name:
            continue
        _, target = run_dir.name.split("-", 1)
        for seed_dir in sorted(run_dir.glob("seed_*")):
            path = seed_dir / "observations.jsonl"
            if not path.is_file():
                continue
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            words = warmstart_words(rows)
            if not words:
                continue
            seed = seed_dir.name.split("_", 1)[1]
            key = _norm(target)
            previous = targets[key].get(seed)
            if previous and previous != words:
                raise ValueError(
                    f"conflicting warmstarts for {target} seed {seed}: "
                    f"{previous} vs {words}"
                )
            targets[key][seed] = words
    return dict(targets)


def merge_protocol_warmstarts(
    primary: dict[str, dict[str, list[str]]],
    fallback: dict[str, dict[str, list[str]]],
    *,
    train_targets: set[str],
) -> dict[str, dict[str, list[str]]]:
    """Prefer fallback train lists; fill any missing ``(target, seed)`` from it."""
    merged: dict[str, dict[str, list[str]]] = {
        key: dict(seeds) for key, seeds in primary.items()
    }
    for key, seeds in fallback.items():
        dest = merged.setdefault(key, {})
        for seed, words in seeds.items():
            if key in train_targets:
                dest[seed] = words
                continue
            dest.setdefault(seed, words)
    if not merged:
        raise ValueError("no warmstart words in canonical search trees")
    return merged


def train_target_keys(targets_json: Path | None) -> set[str]:
    if targets_json is None or not targets_json.is_file():
        return set()
    payload = json.loads(targets_json.read_text(encoding="utf-8"))
    return {_norm(word) for word in payload.get("train_targets") or []}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=Path, default=DEFAULT_CELL)
    parser.add_argument("--fallback-cell", type=Path, default=DEFAULT_FALLBACK)
    parser.add_argument("--targets-json", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cell = args.cell.resolve()
    if not cell.is_dir():
        print(f"canonical cell not found: {cell}", file=sys.stderr)
        return 1
    primary = dump_cell(cell)
    fallback_dir = args.fallback_cell.resolve() if args.fallback_cell else None
    fallback = (
        dump_cell(fallback_dir)
        if fallback_dir is not None and fallback_dir.is_dir()
        else {}
    )
    targets_json = args.targets_json.resolve() if args.targets_json else None
    targets = merge_protocol_warmstarts(
        primary,
        fallback,
        train_targets=train_target_keys(targets_json),
    )
    payload = {
        "source": str(cell),
        "fallback": str(fallback_dir) if fallback else None,
        "warmstart_count": max(
            len(words) for seeds in targets.values() for words in seeds.values()
        ),
        "targets": targets,
    }
    n_lists = sum(len(seeds) for seeds in targets.values())
    print(f"dumped {len(targets)} targets × {n_lists} seed lists from {cell}")
    if args.dry_run:
        print(json.dumps(payload, indent=2)[:2000])
        return 0
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
