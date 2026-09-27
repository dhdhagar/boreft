#!/usr/bin/env python3
"""Summarize Semantle search runs under ``experiments/outputs/semantle/search``.

Compares methods on the shared protocol (budget 500, 10 labeled checkpoint
warmstarts, seeds 1/2/3, five train + five test targets).

BOReFT is loaded from the hyperparameter-sweep winner
``experiments/outputs/semantle/sweep/s1_t0_ard_d64`` (1 sample, T=0, ARD,
projection dim 64, 1 projection layer, logEI) unless ``--boreft-config`` is
changed. Train runs prefer ``s1_t0_ard_d64_rerun`` when that folder exists
(the later train rerun: 6/15 exact matches vs 5/15 originally); test stays
on the original cell. Extra BOReFT ablations in the sweep folder are included
automatically when present: L2/L3 projection stacks and UCB / Thompson
acquisition, all on the same winning cell. Other methods stay under
``search/<method>``. Scratch AutoDiscovery prompt-check trees
(``search/autodiscovery_prompt_check*``), the superseded k=0 Random tree
(``search/random_sampling_k0``), and Random LoRA epoch 1 are ignored.
Pretrained Random is ``Random (base)``. The main paper figure plots held-out
targets only: baselines before supervised fine-tuning on the left, and the
epoch-9 LoRA-SFT baselines plus BOReFT on the right.

x-axes are **verifications**, not observation index: a multi-sample BOReFT
config charges several verifier calls per acquisition, so an
observation-indexed curve would overstate its search progress.

A run "finds" the target when any verified decode/sample equals the gold word
(whitespace-collapsed, lower-cased). Newer BOReFT summaries include
``found_target``; this script still reconstructs hits from ``observations.jsonl``
so older mean-aggregated runs stay comparable.

    python experiments/semantle/analyze_results.py
    python experiments/semantle/analyze_results.py --from-wandb --copy-paper --no-wandb
    python experiments/semantle/analyze_results.py --out-dir /tmp/semantle-analysis
    python experiments/semantle/analyze_results.py --boreft-config s5_t1_noard_d64
    python experiments/semantle/analyze_results.py --no-wandb
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = str(REPO_ROOT / "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from boreft.bo.plotting import (
    annotate_warmstart_xaxis,
    apply_bo_axes_style,
    bo_rc_params,
    iteration_ticks,
    save_plot,
)
from boreft.search_wandb import (
    DEFAULT_ENTITY,
    DEFAULT_PROJECT,
    DEFAULT_SEARCH_GROUP,
    log_analysis_directory,
)

DEFAULT_SEARCH_DIR = REPO_ROOT / "experiments" / "outputs" / "semantle" / "search"
DEFAULT_SWEEP_DIR = REPO_ROOT / "experiments" / "outputs" / "semantle" / "sweep"
DEFAULT_BOREFT_CONFIG = "s1_t0_ard_d64"
DEFAULT_BOREFT_TRAIN_OVERRIDE = "s1_t0_ard_d64_rerun"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments" / "outputs" / "semantle" / "analysis"

# Extra BOReFT cells on the winning (s1, T=0, ARD, d=64) setting.
BOREFT_VARIANT_SLUGS = {
    "boreft_L2": "s1_t0_ard_d64_L2",
    "boreft_L3": "s1_t0_ard_d64_L3",
    "boreft_ucb": "s1_t0_ard_d64_ucb",
    "boreft_thompson": "s1_t0_ard_d64_thompson",
}

SKIP_DIR_PREFIXES = (
    "autodiscovery_prompt_check",
    "random_sampling_k0",
    "random_sampling_lora_e1",
    "random_sampling_lora_e9",
    "boreft_N",
    "boreft_rank",
    "sdpo_ttt_lora",
    "autodiscovery_lora",
    "migrate_lora",
    "bopro_lora",
    "opro_lora",
)
DEFAULT_PAPER_FIGURE = REPO_ROOT / "notes" / "theory" / "figures" / "search_baselines.pdf"
# Epoch-9 LoRA-SFT trees. The paper figure's right panel uses these on
# held-out targets; the directory name maps back to the base method so both
# panels share colors and legend labels. BOReFT is drawn on both panels.
PAPER_SFT_DIRS = {
    "sdpo_ttt_lora_e9": "sdpo_ttt",
    "autodiscovery_lora_e9": "autodiscovery",
    "migrate_lora_e9": "migrate",
    "random_sampling_lora_e9": "random_sampling",
    "bopro_lora_e9": "bopro",
    "opro_lora_e9": "opro",
}
WANDB_FALLBACK_PATTERNS = {
    "random_sampling_lora_e9": re.compile(
        r"^random_sampling_lora_e9_(train|test)-(.+)_seed(\d+)$"
    ),
}
BUDGET = 500
WARMSTART = 10

METHOD_ORDER = (
    "boreft",
    "boreft_L2",
    "boreft_L3",
    "boreft_ucb",
    "boreft_thompson",
    "discrete_bo",
    "bopro",
    "opro",
    "migrate",
    "autodiscovery",
    "sdpo_ttt",
    "random_sampling",
    "random_sampling_lora_e1",
    "random_sampling_lora_e9",
)
DISPLAY = {
    "boreft": "BOReFT",
    "boreft_L2": "BOReFT L2",
    "boreft_L3": "BOReFT L3",
    "boreft_ucb": "BOReFT UCB",
    "boreft_thompson": "BOReFT Thompson",
    "discrete_bo": "Discrete BO",
    "bopro": "BOPRO",
    "opro": "OPRO",
    "migrate": "MIGRATE",
    "autodiscovery": "AutoDiscovery",
    "sdpo_ttt": "SDPO-TTT",
    "random_sampling": "Random (base)",
    "random_sampling_lora_e1": "Random (post-SFT e1)",
    "random_sampling_lora_e9": "Random (post-SFT)",
}


def skip_search_dir(name: str) -> bool:
    """Drop prompt-check, ladder, and post-SFT baseline trees."""
    return any(name.startswith(prefix) for prefix in SKIP_DIR_PREFIXES)


def label(method: str) -> str:
    return DISPLAY.get(method, method.replace("_", " ").title())


COLORS = {
    "boreft": "#0072B2",
    "boreft_L2": "#4E79A7",
    "boreft_L3": "#76B7B2",
    "boreft_ucb": "#E15759",
    "boreft_thompson": "#B07AA1",
    "discrete_bo": "#D55E00",
    "bopro": "#009E73",
    "opro": "#CC79A7",
    "migrate": "#E69F00",
    "autodiscovery": "#56B4E9",
    "sdpo_ttt": "#882255",
    "random_sampling": "#7F7F7F",
    "random_sampling_lora_e1": "#BDBDBD",
    "random_sampling_lora_e9": "#4D4D4D",
}


def color(method: str) -> str:
    return COLORS.get(method, "#333333")


def line_style(method: str) -> str:
    if method == "discrete_bo":
        return "--"
    if method == "random_sampling_lora_e9":
        return ":"
    return "-"


def order_methods_by_test(
    methods: list[str],
    table: list[dict],
    *,
    discrete_last: bool = True,
) -> list[str]:
    """Descending test exact-match, then test cosine; Discrete BO last if present."""
    by_method = {row["method"]: row for row in table}

    def rank(method: str) -> tuple:
        if discrete_last and method == "discrete_bo":
            return (1, 0.0, 0.0, method)
        test = (by_method.get(method) or {}).get("test") or {}
        success = test.get("success_rate")
        best = test.get("mean_best")
        success_key = (
            -float(success)
            if isinstance(success, (int, float)) and math.isfinite(success)
            else 1.0
        )
        best_key = (
            -float(best)
            if isinstance(best, (int, float)) and math.isfinite(best)
            else 1.0
        )
        return (0, success_key, best_key, method)

    return sorted(methods, key=rank)


def subset_summary(summary: dict, methods: list[str]) -> dict:
    wanted = set(methods)
    table = [row for row in summary["table"] if row["method"] in wanted]
    table.sort(key=lambda row: methods.index(row["method"]))
    per_target = {}
    for lab, values in summary["per_target"].items():
        per_target[lab] = {method: values[method] for method in methods if method in values}
    curves = {
        method: summary["curves"][method]
        for method in methods
        if method in summary["curves"]
    }
    return {
        **summary,
        "methods": list(methods),
        "table": table,
        "per_target": per_target,
        "curves": curves,
    }


def paper_curve_order(methods: list[str]) -> list[str]:
    """Legend order for the main search figure: Random follows SDPO-TTT."""
    ordered = list(methods)
    if "random_sampling" in ordered and "sdpo_ttt" in ordered:
        ordered.remove("random_sampling")
        ordered.insert(ordered.index("sdpo_ttt") + 1, "random_sampling")
    return ordered


def paper_before_after(
    search_dir: Path,
    base_runs: list[dict],
    before_methods: list[str],
    target_dirs: list[Path],
) -> tuple[dict, list[str], list[str], dict[str, str]]:
    """Held-out curves for the paper figure: base methods, then LoRA-SFT.

    BOReFT is drawn only on the after-training panel. Discrete BO stays on
    the before panel. ``style_key`` maps each SFT directory name back to its
    base method so the two panels share colors and legend labels.
    """
    sft_dirs = {
        dirname: search_dir / dirname
        for dirname in PAPER_SFT_DIRS
        if (search_dir / dirname).is_dir()
    }
    missing = [dirname for dirname in PAPER_SFT_DIRS if dirname not in sft_dirs]
    if missing:
        print(
            "paper figure missing LoRA-SFT dirs: " + ", ".join(missing),
            file=sys.stderr,
        )
    wanted = set(before_methods)
    summary = summarize(
        [run for run in base_runs if run["method"] in wanted] + list(iter_runs(sft_dirs)),
        target_dirs,
    )
    after_methods: list[str] = []
    base_by_sft = {base_name: dirname for dirname, base_name in PAPER_SFT_DIRS.items()}
    for method in before_methods:
        if method == "discrete_bo":
            continue
        if method == "boreft":
            after_methods.append(method)
            continue
        sft_name = base_by_sft.get(method)
        if sft_name is not None and sft_name in summary["curves"]:
            after_methods.append(sft_name)
    before_drawn = [method for method in before_methods if method != "boreft"]
    return summary, before_drawn, after_methods, dict(PAPER_SFT_DIRS)


def comparison_methods(summary: dict) -> list[str]:
    methods = [
        method
        for method in summary["methods"]
        if method not in BOREFT_VARIANT_SLUGS
    ]
    return order_methods_by_test(methods, summary["table"])


def variant_methods(summary: dict) -> list[str]:
    methods = [
        method
        for method in summary["methods"]
        if method == "boreft" or method in BOREFT_VARIANT_SLUGS
    ]
    return order_methods_by_test(methods, summary["table"], discrete_last=False)


def _norm(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            **bo_rc_params(),
            "axes.labelsize": 14,
            "axes.titlesize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 120,
            "savefig.dpi": 160,
        }
    )
    return plt


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def observation_texts(obs: dict) -> list[str]:
    texts: list[str] = []
    for key in ("decoded_samples", "solution_samples"):
        for item in obs.get(key) or []:
            if item:
                texts.append(str(item))
    for key in ("decoded", "solution"):
        item = obs.get(key)
        if item:
            texts.append(str(item))
    return texts


def sample_scores(obs: dict) -> list[float]:
    scores = obs.get("sample_scores")
    if scores:
        return [float(s) for s in scores]
    count = max(int(obs.get("sample_count") or 1), 1)
    return [float(obs.get("score") or 0.0)] * count


def observation_matches(obs: dict, target_key: str) -> Optional[int]:
    """1-based index of the first matching sample inside this observation."""
    samples = list(obs.get("decoded_samples") or obs.get("solution_samples") or [])
    if samples:
        for i, text in enumerate(samples):
            if _norm(text) == target_key:
                return i + 1
    for key in ("decoded", "solution"):
        if _norm(obs.get(key) or "") == target_key:
            return 1
    exact = (obs.get("components") or {}).get("exact_match")
    if exact is True:
        return 1
    if isinstance(exact, (int, float)) and exact > 0:
        return 1
    return None


def is_warmstart(obs: dict) -> bool:
    kind = obs.get("source") or obs.get("phase") or ""
    return kind == "warmstart"


def expand_run(
    observations: list[dict], target: str, budget: int
) -> dict[str, Any]:
    """Map a run onto a verification-indexed trajectory of length ``budget``."""
    target_key = _norm(target)
    best_curve = np.full(budget, np.nan, dtype=np.float64)
    found_curve = np.zeros(budget, dtype=np.float64)
    unique_curve = np.zeros(budget, dtype=np.float64)
    cursor = 0
    best = -math.inf
    found_at: Optional[int] = None
    best_text = ""
    best_score = -math.inf
    n_verifications = 0
    n_search = 0
    n_repeat_samples = 0
    n_repeat_proposals = 0
    seen_texts: set[str] = set()

    for obs in observations:
        scores = sample_scores(obs)
        texts = observation_texts(obs)
        if not is_warmstart(obs):
            n_search += 1
            n_repeat_samples += int(bool(obs.get("is_repeat_sample")))
            n_repeat_proposals += int(bool(obs.get("is_repeat_proposal")))
        match_at = observation_matches(obs, target_key)
        for i, score in enumerate(scores):
            if cursor >= budget:
                break
            best = max(best, score)
            best_curve[cursor] = best
            text = texts[i] if i < len(texts) else (texts[0] if texts else "")
            key = _norm(text)
            if key:
                seen_texts.add(key)
            unique_curve[cursor] = len(seen_texts)
            if score > best_score:
                best_score = score
                best_text = text
            if match_at is not None and i + 1 >= match_at and found_at is None:
                found_at = cursor + 1
            if found_at is not None:
                found_curve[cursor] = 1.0
            cursor += 1
        n_verifications = cursor
        if cursor >= budget:
            break

    if cursor == 0:
        best_curve[:] = 0.0
    else:
        last = best_curve[cursor - 1]
        best_curve[cursor:] = last
        unique_curve[cursor:] = unique_curve[cursor - 1]
        if found_at is not None:
            found_curve[cursor:] = 1.0

    return {
        "best_curve": best_curve,
        "found_curve": found_curve,
        "unique_curve": unique_curve,
        "found": found_at is not None,
        "found_at": found_at,
        "best_score": float(best_curve[-1]),
        "best_text": best_text,
        "n_verifications": n_verifications,
        "n_observations": len(observations),
        "n_search": n_search,
        "n_repeat_samples": n_repeat_samples,
        "n_repeat_proposals": n_repeat_proposals,
        "repeat_sample_rate": (
            n_repeat_samples / n_search if n_search else float("nan")
        ),
        "repeat_proposal_rate": (
            n_repeat_proposals / n_search if n_search else float("nan")
        ),
        "n_unique": len(seen_texts),
    }


def iter_runs(method_dirs: dict[str, Path]) -> Iterator[dict[str, Any]]:
    for method, method_dir in method_dirs.items():
        if not method_dir.is_dir():
            continue
        for run_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            name = run_dir.name
            if "-" not in name:
                continue
            split, target = name.split("-", 1)
            if split not in ("train", "test"):
                continue
            for seed_dir in sorted(run_dir.glob("seed_*")):
                obs_path = seed_dir / "observations.jsonl"
                if not obs_path.exists():
                    continue
                seed = int(seed_dir.name.split("_", 1)[1])
                stats = expand_run(load_jsonl(obs_path), target, BUDGET)
                yield {
                    "method": method,
                    "split": split,
                    "target": target,
                    "seed": seed,
                    "dir": str(seed_dir),
                    **stats,
                }


def parse_wandb_fallback_name(
    method: str, display_name: str
) -> Optional[tuple[str, str, int]]:
    pattern = WANDB_FALLBACK_PATTERNS.get(method)
    if pattern is None:
        return None
    match = pattern.match(str(display_name or ""))
    if not match:
        return None
    split, target, seed = match.group(1), match.group(2), int(match.group(3))
    return split, target, seed


def curves_from_history(rows: list[dict], budget: int = BUDGET) -> dict:
    """Rebuild per-verification curves from W&B search history rows."""
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


def _download_wandb_history(run, cache: dict[str, list[dict]]) -> list[dict]:
    run_id = run.id
    if run_id in cache:
        return cache[run_id]
    rows = run.history(
        keys=[
            "search/verifications",
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


def load_wandb_fallback_runs(
    method: str,
    *,
    entity: str,
    project: str,
    cache_path: Optional[Path] = None,
) -> list[dict]:
    """Latest W&B search histories for a method whose on-disk tree is absent."""
    import wandb

    pattern = WANDB_FALLBACK_PATTERNS[method]
    api = wandb.Api()
    cache: dict[str, list[dict]] = {}
    if cache_path and cache_path.is_file():
        cache = json.loads(cache_path.read_text())
    seen: dict[str, object] = {}
    for run in api.runs(
        f"{entity}/{project}",
        filters={"displayName": {"$regex": pattern.pattern}},
        order="-created_at",
        per_page=100,
    ):
        if run.display_name not in seen:
            seen[run.display_name] = run
    runs: list[dict] = []
    selected = list(seen.values())
    for index, run in enumerate(selected, start=1):
        parsed = parse_wandb_fallback_name(method, run.display_name)
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
        if summary.get("search/best_so_far") is not None:
            stats["best_score"] = float(summary["search/best_so_far"])
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
    return runs


def copy_paper_figure(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    png_src = src.with_suffix(".png")
    if png_src.is_file():
        shutil.copy2(png_src, dest.with_suffix(".png"))


def mean_std(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, 0.0
    return mean, float(np.std(values, ddof=1))


def _runs_by_seed(subset: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for run in subset:
        grouped[int(run["seed"])].append(run)
    return dict(grouped)


def seed_level_means(subset: list[dict], field: str) -> np.ndarray:
    """One scalar per seed: mean of ``field`` over that seed's targets."""
    grouped = _runs_by_seed(subset)
    means = []
    for seed in sorted(grouped):
        values = np.array(
            [float(run[field]) for run in grouped[seed]], dtype=np.float64
        )
        if values.size:
            means.append(float(np.mean(values)))
    return np.asarray(means, dtype=np.float64)


def seed_level_curve_mean_std(
    subset: list[dict], field: str
) -> tuple[np.ndarray, np.ndarray]:
    """Mean ± SD across seeds of the per-seed mean curve over targets."""
    grouped = _runs_by_seed(subset)
    seed_curves = []
    for seed in sorted(grouped):
        stacked = np.stack([run[field] for run in grouped[seed]])
        seed_curves.append(stacked.mean(axis=0))
    if not seed_curves:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty
    matrix = np.stack(seed_curves)
    mean = matrix.mean(axis=0)
    std = (
        matrix.std(axis=0, ddof=1)
        if matrix.shape[0] > 1
        else np.zeros_like(mean)
    )
    return mean, std


def target_order(candidate_dirs: list[Path], runs: list[dict]) -> list[tuple[str, str]]:
    """Prefer the sampled order in ``targets.json``; else split then name."""
    seen_files: set[Path] = set()
    for directory in candidate_dirs:
        paths = []
        if directory.is_file():
            paths.append(directory)
        elif directory.is_dir():
            paths.append(directory / "targets.json")
            paths.extend(
                sorted(p / "targets.json" for p in directory.iterdir() if p.is_dir())
            )
        for path in paths:
            resolved = path.resolve() if path.exists() else path
            if not path.exists() or resolved in seen_files:
                continue
            seen_files.add(resolved)
            payload = json.loads(path.read_text())
            ordered = [
                (str(r["split"]), str(r["target"])) for r in payload.get("runs", [])
            ]
            if ordered:
                return ordered
    return sorted(
        {(r["split"], r["target"]) for r in runs},
        key=lambda item: (0 if item[0] == "train" else 1, item[1]),
    )


def summarize(
    runs: list[dict], target_dirs: list[Path]
) -> dict[str, Any]:
    by_method: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_method[run["method"]].append(run)

    methods = [m for m in METHOD_ORDER if m in by_method]
    methods += [m for m in sorted(by_method) if m not in methods]
    table = []
    for method in methods:
        row: dict[str, Any] = {"method": method, "display": label(method)}
        for split in ("train", "test", "all"):
            subset = (
                by_method[method]
                if split == "all"
                else [r for r in by_method[method] if r["split"] == split]
            )
            found = np.array([1.0 if r["found"] else 0.0 for r in subset])
            mean, std_best = mean_std(seed_level_means(subset, "best_score"))
            found_times = [r["found_at"] for r in subset if r["found"]]
            sample_mean, sample_std = mean_std(
                seed_level_means(subset, "repeat_sample_rate")
            )
            proposal_mean, proposal_std = mean_std(
                seed_level_means(subset, "repeat_proposal_rate")
            )
            unique_mean, unique_std = mean_std(seed_level_means(subset, "n_unique"))
            row[split] = {
                "n": len(subset),
                "found": int(found.sum()),
                "success_rate": float(found.mean()) if len(subset) else float("nan"),
                "mean_best": mean,
                "sem_best": std_best,
                "median_found_at": (
                    float(np.median(found_times)) if found_times else None
                ),
                "repeat_sample_rate": sample_mean,
                "sem_repeat_sample": sample_std,
                "repeat_proposal_rate": proposal_mean,
                "sem_repeat_proposal": proposal_std,
                "mean_unique": unique_mean,
                "sem_unique": unique_std,
            }
        table.append(row)

    per_target: dict[str, dict[str, dict]] = {}
    seen = {(r["split"], r["target"]) for r in runs}
    targets = [item for item in target_order(target_dirs, runs) if item in seen]
    for split, target in targets:
        per_target[f"{split}-{target}"] = {}
        for method in methods:
            subset = [
                r
                for r in by_method[method]
                if r["split"] == split and r["target"] == target
            ]
            scores = np.array([r["best_score"] for r in subset], dtype=np.float64)
            found = sum(1 for r in subset if r["found"])
            per_target[f"{split}-{target}"][method] = {
                "n": len(subset),
                "found": found,
                "success_rate": found / len(subset) if subset else float("nan"),
                "mean_best": float(np.mean(scores)) if subset else float("nan"),
            }

    curves: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    x = np.arange(1, BUDGET + 1)
    for method in methods:
        curves[method] = {}
        for split in ("train", "test"):
            subset = [r for r in by_method[method] if r["split"] == split]
            if not subset:
                continue
            best_mean, best_std = seed_level_curve_mean_std(subset, "best_curve")
            success_mean, success_std = seed_level_curve_mean_std(
                subset, "found_curve"
            )
            unique_mean, unique_std = seed_level_curve_mean_std(
                subset, "unique_curve"
            )
            curves[method][split] = {
                "x": x,
                "best_mean": best_mean,
                "best_std": best_std,
                "success": success_mean,
                "success_std": success_std,
                "unique_mean": unique_mean,
                "unique_std": unique_std,
            }
    return {
        "n_runs": len(runs),
        "methods": methods,
        "table": table,
        "per_target": per_target,
        "curves": curves,
        "runs": runs,
    }


def plot_grouped_bars(
    plt,
    table: list[dict],
    value_key: str,
    err_key: Optional[str],
    ylabel: str,
    title: str,
    path: Path,
    ylim: Optional[tuple[float, float]] = None,
    percent: bool = False,
) -> None:
    methods = [row["method"] for row in table]
    x = np.arange(len(methods))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(9.2, 0.78 * len(methods) + 2.8), 4.6))
    for offset, split, color in ((-width / 2, "train", "#4C72B0"), (width / 2, "test", "#DD8452")):
        values = np.array([row[split][value_key] for row in table], dtype=np.float64)
        errs = (
            np.array([row[split][err_key] for row in table], dtype=np.float64)
            if err_key
            else None
        )
        drawn = values * (100.0 if percent else 1.0)
        err_drawn = None if errs is None else errs * (100.0 if percent else 1.0)
        bars = ax.bar(
            x + offset,
            drawn,
            width,
            yerr=err_drawn,
            capsize=3,
            color=color,
            edgecolor="0.2",
            linewidth=0.6,
            label=split.capitalize(),
            error_kw={"ecolor": "0.25", "elinewidth": 1.0},
        )
        for patch, method in zip(bars, methods):
            if method == "discrete_bo":
                patch.set_hatch("///")
    ax.set_xticks(x)
    ax.set_xticklabels([label(m) for m in methods], rotation=20, ha="right")
    apply_bo_axes_style(ax, ylabel=ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend()
    fig.tight_layout()
    save_plot(fig, path)
    plt.close(fig)


def _palatino_bold_file() -> Optional[Path]:
    """Palatino Bold face, extracted so the legend can embed a real bold cut."""
    cache = Path.home() / ".cache" / "boreft" / "Palatino-Bold.ttf"
    if cache.is_file():
        return cache
    collection = Path("/System/Library/Fonts/Palatino.ttc")
    if not collection.is_file():
        return None
    from fontTools.ttLib import TTCollection

    fonts = TTCollection(str(collection)).fonts
    bold = next(
        (
            font
            for font in fonts
            if font["name"].getDebugName(2) == "Bold"
        ),
        None,
    )
    if bold is None:
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    bold.save(str(cache))
    return cache


def _add_shared_top_legend(
    fig,
    axes,
    n_methods: int,
    *,
    ncol: Optional[int] = None,
    bold_labels: Optional[set[str]] = None,
    fontsize: Optional[float] = None,
    frameon: bool = True,
) -> None:
    left_handles, left_labels = axes[0].get_legend_handles_labels()
    right_handles, right_labels = axes[1].get_legend_handles_labels()
    by_label = {}
    for handle, text in list(zip(left_handles, left_labels)) + list(
        zip(right_handles, right_labels)
    ):
        by_label.setdefault(text, handle)
    # Labels that appear only on the right (BOReFT on the paper figure) keep
    # their right-panel order and precede the left-panel legend.
    labels = [text for text in right_labels if text not in left_labels] + list(left_labels)
    handles = [by_label[text] for text in labels]
    n_methods = len(labels)
    if ncol is not None:
        ncol = n_methods
    # One centered row when the labels still fit; otherwise wrap evenly.
    if ncol is None:
        if n_methods <= 5:
            ncol = max(n_methods, 1)
        elif n_methods <= 8:
            ncol = 4
        else:
            ncol = 5
    single_row = ncol >= n_methods
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncols=ncol,
        bbox_to_anchor=(0.5, 1.02),
        handlelength=1.6 if single_row else 2.4,
        columnspacing=0.8 if single_row else (1.0 if n_methods >= 5 else 1.2),
        borderaxespad=0.2,
        alignment="center",
        fontsize=fontsize,
        frameon=frameon,
    )
    if bold_labels:
        bold_file = _palatino_bold_file()
        props = None
        if bold_file is not None:
            from matplotlib.font_manager import FontProperties

            size = legend.get_texts()[0].get_fontsize()
            props = FontProperties(fname=str(bold_file), size=size)
        for text in legend.get_texts():
            if text.get_text() not in bold_labels:
                continue
            if props is not None:
                text.set_fontproperties(props)
            else:
                text.set_fontweight("bold")


def plot_anytime(
    plt,
    summary: dict,
    kind: str,
    path: Path,
    *,
    shared_legend: bool = False,
    suptitle: Optional[str] = None,
    panel_titles: Optional[tuple[str, str]] = None,
    panel_methods: Optional[tuple[list[str], list[str]]] = None,
    style_key: Optional[dict[str, str]] = None,
    axis_labelsize: Optional[float] = None,
    title_size: Optional[float] = None,
    legend_single_row: bool = False,
    legend_labels: Optional[dict[str, str]] = None,
    bold_legend: Optional[set[str]] = None,
    legend_fontsize: Optional[float] = None,
    legend_frameon: bool = True,
    line_width: float = 1.8,
    ylabel: Optional[str] = None,
    spine_width: Optional[float] = None,
    tick_labelsize: Optional[float] = None,
    title_bold: bool = False,
    warmstart_labelsize: Optional[float] = None,
) -> None:
    if panel_methods is None:
        panel_specs = [
            ("train", list(summary["methods"])),
            ("test", list(summary["methods"])),
        ]
        n_methods = len(summary["methods"])
    else:
        panel_specs = [("test", list(panel_methods[0])), ("test", list(panel_methods[1]))]
        n_methods = len(panel_methods[0])
    extra = 0.22 if shared_legend and n_methods > 4 and not legend_single_row else 0.0
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.8, (5.2 if shared_legend else 5.0) + extra),
        sharey=True,
    )
    for panel_index, (ax, (split, methods)) in enumerate(zip(axes, panel_specs)):
        for method in methods:
            style = (style_key or {}).get(method, method)
            series = summary["curves"].get(method, {}).get(split)
            if series is None:
                continue
            if kind == "best":
                y = series["best_mean"]
                band = series["best_std"]
            elif kind == "unique":
                y = series["unique_mean"]
                band = series["unique_std"]
            else:
                y = series["success"] * 100.0
                band = series["success_std"] * 100.0
            ax.plot(
                series["x"],
                y,
                color=color(style),
                lw=line_width,
                ls=line_style(style),
                label=(legend_labels or {}).get(style, label(style)),
            )
            if band is not None:
                ax.fill_between(
                    series["x"],
                    y - band,
                    y + band,
                    color=color(style),
                    alpha=0.15,
                    linewidth=0,
                )
        ax.set_xlim(1, BUDGET)
        ticks = iteration_ticks(
            BUDGET,
            ax.get_xticks(),
            warmstart_count=WARMSTART,
        )
        ax.set_xticks(ticks)
        apply_bo_axes_style(ax, xlabel="Verifications", spine_width=spine_width)
        if axis_labelsize is not None:
            ax.set_xlabel("Verifications", fontsize=axis_labelsize)
        if tick_labelsize is not None:
            ax.tick_params(axis="both", labelsize=tick_labelsize)
        if panel_titles is None:
            panel = f"{split.capitalize()} targets"
        else:
            panel = panel_titles[panel_index]
        if title_size is None:
            ax.set_title(panel)
        else:
            ax.set_title(panel, fontsize=title_size)
        if title_bold:
            bold_file = _palatino_bold_file()
            if bold_file is not None:
                from matplotlib.font_manager import FontProperties

                ax.title.set_fontproperties(
                    FontProperties(
                        fname=str(bold_file),
                        size=title_size or ax.title.get_fontsize(),
                    )
                )
            else:
                ax.title.set_fontweight("bold")
        annotate_warmstart_xaxis(fig, ax, WARMSTART, label_size=warmstart_labelsize)
    if kind == "best":
        axes[0].set_ylabel(
            "Best cosine so far" if ylabel is None else ylabel,
            fontsize=14 if axis_labelsize is None else axis_labelsize,
            labelpad=4,
        )
        axes[0].set_ylim(0.55, 1.02)
        heading = (
            "Anytime best embedding cosine (mean ± 1 SD across 3 seeds)"
            if suptitle is None
            else suptitle
        )
    elif kind == "unique":
        axes[0].set_ylabel("Unique verified strings", fontsize=14, labelpad=4)
        axes[0].set_ylim(0, BUDGET)
        heading = (
            "Anytime unique strings (mean ± 1 SD across 3 seeds; early-stopped runs stay at last count)"
            if suptitle is None
            else suptitle
        )
    else:
        axes[0].set_ylabel("Exact-match success (%)", fontsize=14, labelpad=4)
        axes[0].set_ylim(-2, 105)
        heading = (
            "Anytime exact-match rate (mean ± 1 SD across 3 seeds; any verified sample equals the gold word)"
            if suptitle is None
            else suptitle
        )
    if shared_legend:
        _add_shared_top_legend(
            fig,
            axes,
            n_methods,
            ncol=n_methods if legend_single_row else None,
            bold_labels=bold_legend,
            fontsize=legend_fontsize,
            frameon=legend_frameon,
        )
        if heading:
            fig.suptitle(heading, y=1.12)
            fig.tight_layout(rect=(0, 0, 1, 0.90))
        else:
            top = 0.91 if n_methods > 4 else 0.93
            fig.tight_layout(rect=(0, 0, 1, top))
    else:
        handles, labels = axes[1].get_legend_handles_labels()
        loc = "upper left" if kind == "unique" else "lower right"
        ncol = 3 if n_methods > 8 else 2
        axes[1].legend(handles, labels, loc=loc, ncol=ncol, handlelength=2.6)
        if heading:
            fig.suptitle(heading, y=1.06)
        fig.tight_layout()
    save_plot(fig, path)
    plt.close(fig)


def plot_heatmap(
    plt,
    summary: dict,
    field: str,
    title: str,
    path: Path,
    cmap: str,
    vmin: float,
    vmax: float,
    fmt: str,
) -> None:
    methods = summary["methods"]
    labels = list(summary["per_target"])
    matrix = np.array(
        [
            [summary["per_target"][lab][m][field] for m in methods]
            for lab in labels
        ],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(1.15 * len(methods) + 2.4, 0.42 * len(labels) + 1.8))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([label(m) for m in methods], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            ax.text(
                j,
                i,
                format(value, fmt),
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if value > (vmin + vmax) * 0.55 else "0.1",
            )
    ax.set_title(title)
    apply_bo_axes_style(ax, grid=False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_plot(fig, path)
    plt.close(fig)


def plot_found_at(plt, runs: list[dict], methods: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(9.2, 0.78 * len(methods) + 2.8), 4.6))
    drawn = False
    for i, method in enumerate(methods):
        times = [r["found_at"] for r in runs if r["method"] == method and r["found"]]
        if not times:
            continue
        drawn = True
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, size=len(times))
        ax.scatter(
            np.full(len(times), i) + jitter,
            times,
            s=28,
            color=color(method),
            edgecolor="0.2",
            linewidth=0.4,
            zorder=3,
        )
        ax.hlines(np.median(times), i - 0.25, i + 0.25, color="0.15", lw=1.6, zorder=4)
    if not drawn:
        plt.close(fig)
        return
    ax.axhline(WARMSTART, color="0.55", ls="--", lw=1.0)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([label(m) for m in methods], rotation=20, ha="right")
    apply_bo_axes_style(ax, ylabel="Verifications until first exact match")
    ax.set_title("When the gold word is first proposed (successful runs; bar = median)")
    ax.set_ylim(0, BUDGET)
    fig.tight_layout()
    save_plot(fig, path)
    plt.close(fig)


def json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: json_ready(v) for k, v in obj.items() if k != "runs"}
    if isinstance(obj, list):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def print_table(table: list[dict]) -> None:
    headers = (
        "method",
        "train found",
        "test found",
        "train best",
        "test best",
        "median t* (train)",
        "median t* (test)",
    )
    rows = []
    for row in table:
        def cell(split: str) -> tuple[str, str, str]:
            block = row[split]
            found = f"{block['found']}/{block['n']}"
            best = f"{block['mean_best']:.3f}±{block['sem_best']:.3f}"
            tstar = (
                f"{block['median_found_at']:.0f}"
                if block["median_found_at"] is not None
                else "—"
            )
            return found, best, tstar

        tr, trb, trt = cell("train")
        te, teb, tet = cell("test")
        rows.append((row["display"], tr, te, trb, teb, trt, tet))

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*row))


def print_repeat_table(table: list[dict]) -> None:
    headers = (
        "method",
        "train sample repeats",
        "test sample repeats",
        "train proposal repeats",
        "test proposal repeats",
        "train unique",
        "test unique",
    )
    rows = []
    for row in table:
        def cell(split: str) -> tuple[str, str, str]:
            block = row[split]
            sample = (
                f"{100.0 * block['repeat_sample_rate']:.1f}±"
                f"{100.0 * block['sem_repeat_sample']:.1f}%"
            )
            proposal = (
                f"{100.0 * block['repeat_proposal_rate']:.1f}±"
                f"{100.0 * block['sem_repeat_proposal']:.1f}%"
            )
            unique = (
                f"{block['mean_unique']:.0f}±{block['sem_unique']:.0f}"
            )
            return sample, proposal, unique

        trs, trp, tru = cell("train")
        tes, tep, teu = cell("test")
        rows.append((row["display"], trs, tes, trp, tep, tru, teu))

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-dir", type=Path, default=DEFAULT_SEARCH_DIR)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument(
        "--boreft-config",
        default=DEFAULT_BOREFT_CONFIG,
        help=(
            "Sweep slug to use for BOReFT (default: %(default)s). "
            "Pass empty to load search/boreft instead."
        ),
    )
    parser.add_argument(
        "--boreft-train-override",
        default=DEFAULT_BOREFT_TRAIN_OVERRIDE,
        help=(
            "Sweep slug whose train-* runs replace BOReFT train when "
            "--boreft-config is the default winner. Pass empty to disable."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--from-wandb",
        action="store_true",
        help=(
            "Load missing paper methods (Random post-SFT) from W&B search "
            "histories when the on-disk tree is absent."
        ),
    )
    parser.add_argument(
        "--copy-paper",
        action="store_true",
        help="Copy anytime_best.pdf onto notes/theory/figures/search_baselines.pdf.",
    )
    parser.add_argument(
        "--paper-figure-only",
        action="store_true",
        help=(
            "Load only the main comparison methods and write the held-out "
            "before/after paper figure."
        ),
    )
    parser.add_argument(
        "--paper-figure",
        type=Path,
        default=DEFAULT_PAPER_FIGURE,
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Skip uploading comparison figures and metrics.json to W&B.",
    )
    parser.add_argument(
        "--wandb-force",
        action="store_true",
        help="Re-upload the comparison run even if wandb_meta.json already exists.",
    )
    parser.add_argument("--wandb-project", default=DEFAULT_PROJECT)
    parser.add_argument("--wandb-entity", default=DEFAULT_ENTITY)
    parser.add_argument("--wandb-dir", default=None)
    return parser.parse_args()


def replace_boreft_train(
    runs: list[dict],
    sweep_dir: Path,
    boreft_config: str,
    override_slug: str,
) -> tuple[list[dict], Optional[str]]:
    """Swap BOReFT train runs for a later rerun; keep original test runs."""
    if (
        not override_slug
        or boreft_config != DEFAULT_BOREFT_CONFIG
        or not any(r["method"] == "boreft" for r in runs)
    ):
        return runs, None
    override_dir = sweep_dir / override_slug
    if not override_dir.is_dir():
        return runs, None
    extra_train = [
        run
        for run in iter_runs({"boreft": override_dir})
        if run["split"] == "train"
    ]
    if not extra_train:
        return runs, None
    kept = [
        run
        for run in runs
        if not (run["method"] == "boreft" and run["split"] == "train")
    ]
    return kept + extra_train, override_slug


def resolve_method_dirs(
    search_dir: Path,
    sweep_dir: Path,
    methods: list[str],
    boreft_config: str,
) -> dict[str, Path]:
    dirs: dict[str, Path] = {}
    for method in methods:
        if method == "boreft" and boreft_config:
            path = sweep_dir / boreft_config
        elif method in BOREFT_VARIANT_SLUGS:
            path = sweep_dir / BOREFT_VARIANT_SLUGS[method]
        else:
            path = search_dir / method
        if path.is_dir():
            dirs[method] = path
    return dirs


def main() -> int:
    args = parse_args()
    search_dir = args.search_dir.resolve()
    sweep_dir = args.sweep_dir.resolve()
    boreft_config = str(args.boreft_config or "").strip()
    if not search_dir.is_dir():
        print(f"search dir not found: {search_dir}", file=sys.stderr)
        return 1

    present = {
        p.name
        for p in search_dir.iterdir()
        if p.is_dir() and not skip_search_dir(p.name)
    }
    if args.from_wandb:
        for method in WANDB_FALLBACK_PATTERNS:
            if not skip_search_dir(method):
                present.add(method)
    if boreft_config and (sweep_dir / boreft_config).is_dir():
        present.add("boreft")
    elif boreft_config:
        print(
            f"BOReFT sweep config not found: {sweep_dir / boreft_config}",
            file=sys.stderr,
        )
        return 1
    for method, slug in BOREFT_VARIANT_SLUGS.items():
        if (sweep_dir / slug).is_dir():
            present.add(method)
    methods = [m for m in METHOD_ORDER if m in present]
    extra = [m for m in sorted(present) if m not in METHOD_ORDER]
    methods.extend(extra)
    if args.paper_figure_only:
        paper_keep = {
            "boreft",
            "discrete_bo",
            "bopro",
            "opro",
            "migrate",
            "autodiscovery",
            "sdpo_ttt",
            "random_sampling",
        }
        methods = [method for method in methods if method in paper_keep]
    if not methods:
        print(f"no method directories in {search_dir}", file=sys.stderr)
        return 1

    method_dirs = resolve_method_dirs(search_dir, sweep_dir, methods, boreft_config)
    runs = list(iter_runs(method_dirs))
    if args.from_wandb:
        cache_path = args.out_dir.resolve() / "wandb_history_cache.json"
        for method in WANDB_FALLBACK_PATTERNS:
            if method not in methods:
                continue
            local = method_dirs.get(method)
            if local is not None and local.is_dir():
                continue
            extra = load_wandb_fallback_runs(
                method,
                entity=args.wandb_entity or None,
                project=args.wandb_project,
                cache_path=cache_path,
            )
            if not extra:
                print(f"W&B fallback found no runs for {method}", file=sys.stderr)
                return 1
            print(f"W&B {label(method)} ← {len(extra)} runs")
            runs.extend(extra)
    train_override = str(args.boreft_train_override or "").strip()
    runs, train_override_used = replace_boreft_train(
        runs, sweep_dir, boreft_config, train_override
    )
    if not runs:
        print(f"no seed runs under {search_dir}", file=sys.stderr)
        return 1

    target_dirs = [sweep_dir, search_dir, *method_dirs.values()]
    if train_override_used:
        target_dirs.insert(0, sweep_dir / train_override_used)
    summary = summarize(runs, target_dirs)
    compare = subset_summary(summary, comparison_methods(summary))
    variants = subset_summary(summary, variant_methods(summary))
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plt = _plt()
    plot_grouped_bars(
        plt,
        compare["table"],
        "success_rate",
        None,
        "Exact-match success (%)",
        "Fraction of runs that proposed the gold word",
        out_dir / "success_rate.png",
        ylim=(0, 105),
        percent=True,
    )
    plot_grouped_bars(
        plt,
        compare["table"],
        "mean_best",
        "sem_best",
        "Best embedding cosine",
        "Final best cosine (mean ± 1 SD across 3 seeds)",
        out_dir / "mean_best_score.png",
        ylim=(0.55, 1.02),
    )
    before_methods = paper_curve_order(compare["methods"])
    paper_summary, before_methods, after_methods, style_key = paper_before_after(
        search_dir,
        runs,
        before_methods,
        target_dirs,
    )
    plot_anytime(
        plt,
        paper_summary,
        "best",
        out_dir / "anytime_best.png",
        shared_legend=True,
        suptitle="",
        panel_titles=("Before SFT", "After SFT"),
        panel_methods=(before_methods, after_methods),
        style_key=style_key,
        axis_labelsize=18,
        title_size=18,
        title_bold=True,
        tick_labelsize=14,
        legend_fontsize=13,
        legend_frameon=False,
        line_width=2.6,
        ylabel="Best similarity",
        spine_width=2.4,
        warmstart_labelsize=11,
        legend_single_row=True,
        legend_labels={"random_sampling": "Random"},
        bold_legend={"BOReFT"},
    )
    if args.copy_paper:
        src = out_dir / "anytime_best.pdf"
        dest = args.paper_figure.expanduser().resolve()
        if not src.is_file():
            print(f"missing paper plot: {src}", file=sys.stderr)
            return 1
        copy_paper_figure(src, dest)
        print(f"copied paper figure → {dest}")
    if args.paper_figure_only:
        print(f"wrote {out_dir / 'anytime_best.pdf'}")
        return 0
    plot_anytime(
        plt,
        compare,
        "success",
        out_dir / "anytime_success.png",
        shared_legend=True,
    )
    plot_heatmap(
        plt,
        compare,
        "success_rate",
        "Exact-match rate by target (fraction of 3 seeds)",
        out_dir / "per_target_found.png",
        cmap="YlGn",
        vmin=0.0,
        vmax=1.0,
        fmt=".0%",
    )
    plot_heatmap(
        plt,
        compare,
        "mean_best",
        "Mean best cosine by target (3 seeds)",
        out_dir / "per_target_best.png",
        cmap="YlOrRd",
        vmin=0.6,
        vmax=1.0,
        fmt=".2f",
    )
    plot_found_at(plt, runs, compare["methods"], out_dir / "found_at.png")
    plot_grouped_bars(
        plt,
        compare["table"],
        "repeat_sample_rate",
        "sem_repeat_sample",
        "Repeat-sample rate",
        "Fraction of search steps whose decoded text was already seen",
        out_dir / "repeat_sample_rate.png",
        ylim=(0, 1.05),
    )
    plot_grouped_bars(
        plt,
        compare["table"],
        "repeat_proposal_rate",
        "sem_repeat_proposal",
        "Repeat-proposal rate",
        "Fraction of search steps whose proposal collided in method space",
        out_dir / "repeat_proposal_rate.png",
        ylim=(0, 1.05),
    )
    plot_anytime(
        plt,
        compare,
        "unique",
        out_dir / "anytime_unique.png",
        shared_legend=True,
    )
    if len(variants["methods"]) > 1:
        plot_grouped_bars(
            plt,
            variants["table"],
            "success_rate",
            None,
            "Exact-match success (%)",
            "BOReFT variants: fraction of runs that proposed the gold word",
            out_dir / "boreft_variants_success.png",
            ylim=(0, 105),
            percent=True,
        )
        plot_anytime(
            plt,
            variants,
            "best",
            out_dir / "boreft_variants_anytime_best.png",
            shared_legend=True,
            suptitle=(
                "BOReFT variants: anytime best embedding cosine "
                "(mean ± 1 SD across 3 seeds)"
            ),
        )
        plot_anytime(
            plt,
            variants,
            "success",
            out_dir / "boreft_variants_anytime_success.png",
            shared_legend=True,
            suptitle="BOReFT variants: anytime exact-match rate (mean ± 1 SD across 3 seeds)",
        )

    payload = {
        "search_dir": str(search_dir),
        "sweep_dir": str(sweep_dir),
        "boreft_config": boreft_config or None,
        "boreft_train_override": train_override_used,
        "boreft_variants": {
            method: BOREFT_VARIANT_SLUGS[method]
            for method in methods
            if method in BOREFT_VARIANT_SLUGS
        },
        "comparison_methods": compare["methods"],
        "variant_methods": variants["methods"],
        "method_dirs": {method: str(path) for method, path in method_dirs.items()},
        "budget": BUDGET,
        "warmstart": WARMSTART,
        "skip": [],
        **json_ready(summary),
        "run_rows": [
            {
                "method": r["method"],
                "split": r["split"],
                "target": r["target"],
                "seed": r["seed"],
                "found": r["found"],
                "found_at": r["found_at"],
                "best_score": r["best_score"],
                "best_text": r["best_text"],
                "n_verifications": r["n_verifications"],
                "n_search": r["n_search"],
                "n_repeat_samples": r["n_repeat_samples"],
                "n_repeat_proposals": r["n_repeat_proposals"],
                "repeat_sample_rate": r["repeat_sample_rate"],
                "repeat_proposal_rate": r["repeat_proposal_rate"],
                "n_unique": r["n_unique"],
            }
            for r in runs
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")

    print(f"loaded {len(runs)} runs from {search_dir}")
    if boreft_config:
        print(f"BOReFT config {boreft_config} ← {sweep_dir / boreft_config}")
    if train_override_used:
        print(
            f"BOReFT train ← {sweep_dir / train_override_used} "
            "(test stays on the original cell)"
        )
    for method in methods:
        if method in BOREFT_VARIANT_SLUGS:
            slug = BOREFT_VARIANT_SLUGS[method]
            print(f"{label(method)} ← {sweep_dir / slug}")
    print(f"wrote figures → {out_dir}")
    print()
    print_table(compare["table"])
    if len(variants["methods"]) > 1:
        print()
        print("BOReFT variants")
        print_table(variants["table"])
    print()
    print_repeat_table(compare["table"])
    print()
    print("Notes")
    print("  Discrete BO searches the frozen train vocabulary, so train gold")
    print("  words are in-pool and test gold words are out-of-pool.")
    if boreft_config == DEFAULT_BOREFT_CONFIG:
        print("  BOReFT is the sweep winner s1_t0_ard_d64: 1 sample, T=0, ARD,")
        print("  projection dim 64, 1 projection layer, logEI.")
        if train_override_used:
            print(f"  Train metrics use {train_override_used}; test uses the original cell.")
    elif boreft_config:
        print(f"  BOReFT is sweep config {boreft_config}.")
    else:
        print("  BOReFT is loaded from search/boreft.")
    if any(m in BOREFT_VARIANT_SLUGS for m in methods):
        print("  Extra BOReFT cells (L2/L3, UCB, Thompson) are in")
        print("  boreft_variants_*.pdf / .png, not the main comparison figures.")
        print("  Discrete BO is hatched / dashed (non-generative pool search).")
    print("  Success = any verified sample equals the gold word.")
    print("  Repeat sample = decoded/solution text already observed.")
    print("  Repeat proposal = collision in the method's own proposal space")
    print("  (latent for BOReFT, embedding for BOPRO, text for the rest).")
    print("  Unique counts every verified decode, including multi-sample BOReFT.")
    if not args.no_wandb:
        try:
            result = log_analysis_directory(
                out_dir,
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                group=DEFAULT_SEARCH_GROUP,
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
                print(
                    f"W&B comparison failed: {result.get('error')}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"W&B comparison skipped: {result.get('status')}",
                    file=sys.stderr,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
