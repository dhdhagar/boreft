#!/usr/bin/env python3
"""Train/held-out search stats plus novelty vs the representation-training words."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEARCH = ROOT / "experiments/outputs/semantle/search"
SWEEP = ROOT / "experiments/outputs/semantle/sweep"
ITEMS = ROOT / "outputs/1784053292/items.json"

METHODS = {
    "discrete_bo": SEARCH / "discrete_bo",
    "sdpo_ttt": SEARCH / "sdpo_ttt",
    "random_sampling_k0": SEARCH / "random_sampling_k0",
    "random_sampling": SEARCH / "random_sampling",
    "autodiscovery": SEARCH / "autodiscovery",
    "migrate": SEARCH / "migrate",
    "random_sampling_lora_e9": SEARCH / "random_sampling_lora_e9",
    "bopro": SEARCH / "bopro",
    "opro": SEARCH / "opro",
    "sdpo_ttt_lora_e9": SEARCH / "sdpo_ttt_lora_e9",
    "autodiscovery_lora_e9": SEARCH / "autodiscovery_lora_e9",
    "migrate_lora_e9": SEARCH / "migrate_lora_e9",
    "bopro_lora_e9": SEARCH / "bopro_lora_e9",
    "opro_lora_e9": SEARCH / "opro_lora_e9",
}


def norm(text: object) -> str:
    return " ".join(str(text or "").strip().lower().split())


def train_words() -> set[str]:
    rows = json.loads(ITEMS.read_text())
    return {norm(row["word"]) for row in rows if norm(row.get("word"))}


def proposal_text(obs: dict) -> str:
    for key in ("solution", "decoded"):
        if obs.get(key):
            return str(obs[key])
    samples = obs.get("solution_samples") or obs.get("decoded_samples") or []
    if samples:
        return str(samples[0])
    return ""


def is_warmstart(obs: dict) -> bool:
    return (obs.get("source") or obs.get("phase") or "") == "warmstart"


def seed_level(runs: list[dict], field: str) -> float:
    grouped: dict[int, list[float]] = defaultdict(list)
    for run in runs:
        grouped[int(run["seed"])].append(float(run[field]))
    means = [sum(values) / len(values) for values in grouped.values() if values]
    return sum(means) / len(means) if means else float("nan")


def load_method(name: str, path: Path, words: set[str]) -> list[dict]:
    runs = []
    if name == "boreft":
        pairs = [
            ("train", SWEEP / "s1_t0_ard_d64_rerun", "train"),
            ("test", SWEEP / "s1_t0_ard_d64", "test"),
        ]
    else:
        pairs = [("any", path, None)]
    for forced_split, root, split_keep in pairs:
        if not root.is_dir():
            print(f"MISSING {name} {root}")
            continue
        for seed_dir in sorted(root.glob("*/seed_*")):
            summary_path = seed_dir / "summary.json"
            obs_path = seed_dir / "observations.jsonl"
            if not summary_path.is_file() or not obs_path.is_file():
                continue
            parent = seed_dir.parent.name
            parent_split, target = parent.split("-", 1)
            if split_keep is not None and parent_split != split_keep:
                continue
            split = parent_split if forced_split == "any" else forced_split
            summary = json.loads(summary_path.read_text())
            n_search = 0
            n_novel = 0
            n_repeat = 0
            n_string_repeat = 0
            seen: set[str] = set()
            with obs_path.open() as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    obs = json.loads(line)
                    text = norm(proposal_text(obs))
                    warm = is_warmstart(obs)
                    if warm:
                        if text:
                            seen.add(text)
                        continue
                    n_search += 1
                    n_repeat += int(bool(obs.get("is_repeat_proposal")))
                    if text and text in seen:
                        n_string_repeat += 1
                    if text:
                        seen.add(text)
                    if text and text not in words:
                        n_novel += 1
            runs.append(
                {
                    "split": split,
                    "target": target,
                    "seed": int(summary.get("seed") or seed_dir.name.split("_")[-1]),
                    "found": bool(summary.get("found_target")),
                    "best_score": float(summary.get("best_score") or 0.0),
                    "repeat": (n_repeat / n_search) if n_search else float("nan"),
                    "string_repeat": (
                        n_string_repeat / n_search if n_search else float("nan")
                    ),
                    "novelty": (n_novel / n_search) if n_search else float("nan"),
                    "n_search": n_search,
                    "n_novel": n_novel,
                    "n_string_repeat": n_string_repeat,
                    "n_repeat": n_repeat,
                }
            )
    return runs


def report(name: str, runs: list[dict]) -> None:
    print(f"\n== {name} n={len(runs)} ==")
    for split in ("train", "test", "all"):
        subset = runs if split == "all" else [r for r in runs if r["split"] == split]
        found = sum(1 for r in subset if r["found"])
        n_search = sum(r["n_search"] for r in subset)
        def pooled(field: str) -> float:
            return (sum(r[field] for r in subset) / n_search) if n_search else float("nan")
        print(
            f"  {split:5s} n={len(subset):2d} found={found:2d} "
            f"sim={seed_level(subset, 'best_score'):.4f} "
            f"rep={seed_level(subset, 'repeat'):.4f} "
            f"strRep={seed_level(subset, 'string_repeat'):.4f} "
            f"nov={seed_level(subset, 'novelty'):.4f} "
            f"poolRep={pooled('n_repeat'):.4f} "
            f"poolStr={pooled('n_string_repeat'):.4f} "
            f"poolNov={pooled('n_novel'):.4f}"
        )


def main() -> None:
    words = train_words()
    print(f"train words {len(words)}")
    for name, path in METHODS.items():
        report(name, load_method(name, path, words))
    report("boreft", load_method("boreft", SWEEP, words))


if __name__ == "__main__":
    main()
