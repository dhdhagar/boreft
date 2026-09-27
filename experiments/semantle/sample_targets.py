"""Sample Semantle train/test search targets from a trained BOReFT run.

Train words come from ``<boreft_dir>/items.json``. Test words prefer the
checkpoint's held-out set in ``<boreft_dir>/eval/results.json``
(``recon_test_meta``, then ``genz_test_meta``). If that is missing they are
rebuilt with the same non-train pool sample ``build_test_sets`` uses (no
embedding / PCA split).

Each split is sampled independently with ``--seed`` after sorting, so the
same flags always yield the same words.

Prints JSON to stdout. With ``--output``, also writes that JSON to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from typing import Iterable, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_TEST_N_SAMPLES = 512
_SPLITS = ("train", "test")


def _normalize(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _unique_targets(words: Iterable[str]) -> list[str]:
    unique: dict[str, str] = {}
    for raw in words:
        text = str(raw).strip()
        if not text:
            continue
        key = _normalize(text)
        if key not in unique:
            unique[key] = text
    return list(unique.values())


def _read_json(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_merged_run_config(output_dir: str) -> dict:
    merged: dict = {}
    for name in ("intervention_config.json", "training_config.json"):
        path = os.path.join(output_dir, name)
        if os.path.isfile(path):
            payload = _read_json(path)
            if isinstance(payload, dict):
                merged.update(payload)
    return merged


def load_train_words(boreft_dir: str) -> list[str]:
    path = os.path.join(boreft_dir, "items.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no items.json in {boreft_dir}")
    items = _read_json(path)
    if not isinstance(items, list):
        raise ValueError(f"{path}: expected a JSON list of training items")
    words: list[str] = []
    for index, row in enumerate(items):
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{index}: expected a JSON object")
        value = row.get("word") or row.get("target")
        if value is None or not str(value).strip():
            raise ValueError(f"{path}:{index}: missing word/target")
        words.append(str(value).strip())
    return _unique_targets(words)


def _remap_csv_path(path: str) -> str:
    """Prefer an existing file; otherwise map ``.../data/...`` onto this repo."""
    if os.path.isfile(path):
        return path
    parts = path.replace("\\", "/").split("/")
    if "data" not in parts:
        return path
    candidate = os.path.join(REPO_ROOT, *parts[parts.index("data") :])
    return candidate if os.path.isfile(candidate) else path


def _meta_test_words(meta: object) -> list[str] | None:
    if not isinstance(meta, dict):
        return None
    interp = meta.get("interp_words")
    extrap = meta.get("extrap_words")
    if interp is None and extrap is None:
        return None
    words = _unique_targets([*(interp or []), *(extrap or [])])
    return words or None


def _test_words_from_eval(boreft_dir: str) -> tuple[list[str], str] | None:
    path = os.path.join(boreft_dir, "eval", "results.json")
    if not os.path.isfile(path):
        return None
    data = _read_json(path)
    if not isinstance(data, dict):
        return None
    for key in ("recon_test_meta", "genz_test_meta"):
        words = _meta_test_words(data.get(key))
        if words:
            return words, f"eval/results.json:{key}"
    return None


def _rebuild_test_words(boreft_dir: str, train_words: Sequence[str]) -> list[str]:
    src_root = os.path.join(REPO_ROOT, "src")
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    try:
        from boreft.eval.eval_suite import (  # noqa: PLC0415
            DEFAULT_TEST_N_SAMPLES as suite_default,
            build_non_train_pool,
            task_csv_paths,
        )
    except ImportError as exc:
        raise RuntimeError(
            f"{boreft_dir} has no usable eval/results.json test split, and "
            "rebuilding the held-out pool requires the boreft package. "
            "Run from the boreft conda env, or pass a checkpoint that already "
            "has eval/results.json."
        ) from exc

    saved = load_merged_run_config(boreft_dir)
    csv_paths = [_remap_csv_path(path) for path in task_csv_paths(saved, task="semantle")]
    pool = build_non_train_pool(csv_paths, train_words, task="semantle")
    if not pool:
        raise ValueError(
            f"{boreft_dir}: empty non-train pool (need eval/results.json or "
            "CSV targets outside the trained vocabulary)"
        )
    n_requested = int(saved.get("test_n_samples", suite_default))
    n_draw = min(n_requested, len(pool))
    seed = int(saved.get("seed", 42))
    return random.Random(seed).sample(sorted(pool), n_draw)


def load_test_words(
    boreft_dir: str, train_words: Sequence[str]
) -> tuple[list[str], str]:
    from_eval = _test_words_from_eval(boreft_dir)
    if from_eval is not None:
        return from_eval
    return _rebuild_test_words(boreft_dir, train_words), "rebuilt"


def sample_split(words: Sequence[str], count: int, seed: int) -> list[str]:
    if count < 0:
        raise ValueError("sample count must be nonnegative")
    if count > len(words):
        raise ValueError(
            f"requested {count} targets but the split only has {len(words)}"
        )
    if count == 0:
        return []
    ordered = sorted(words, key=lambda word: (word.lower(), word))
    return random.Random(seed).sample(ordered, count)


def dir_name(split: str, target: str) -> str:
    """Filesystem-safe ``<split>-<target>`` directory name for one run."""
    if split not in _SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if any(ch in target for ch in "\t\n\r"):
        raise ValueError(f"target must not contain whitespace controls: {target!r}")
    token = re.sub(r"[^A-Za-z0-9]+", "_", target).strip("_") or "target"
    return f"{split}-{token}"


def _runs_for(train_sample: Sequence[str], test_sample: Sequence[str]) -> list[dict]:
    runs: list[dict] = []
    used: dict[str, str] = {}
    for split, words in (("train", train_sample), ("test", test_sample)):
        for word in words:
            name = dir_name(split, word)
            previous = used.get(name)
            if previous is not None:
                raise ValueError(
                    f"run directory collision {name!r} for {previous!r} and {word!r}"
                )
            used[name] = word
            runs.append({"split": split, "target": word, "dir_name": name})
    return runs


def sample_targets(
    boreft_dir: str,
    *,
    n_train: int = 5,
    n_test: int = 5,
    seed: int = 42,
) -> dict:
    if n_train < 0 or n_test < 0:
        raise ValueError("n_train and n_test must be nonnegative")
    if n_train == 0 and n_test == 0:
        raise ValueError("nothing to launch: n_train and n_test are both 0")
    boreft_dir = os.path.abspath(os.path.expanduser(boreft_dir))
    saved = load_merged_run_config(boreft_dir)
    task = saved.get("task") or "semantle"
    if task != "semantle":
        raise ValueError(f"expected a semantle checkpoint, got task={task!r}")
    train_words = load_train_words(boreft_dir)
    if n_test == 0:
        test_words, test_source = [], "unused"
    else:
        test_words, test_source = load_test_words(boreft_dir, train_words)
    overlap = sorted(
        {_normalize(word) for word in train_words}
        & {_normalize(word) for word in test_words}
    )
    if overlap:
        preview = ", ".join(overlap[:5])
        extra = "" if len(overlap) <= 5 else f" (+{len(overlap) - 5} more)"
        raise ValueError(f"train/test pools overlap: {preview}{extra}")
    train_sample = sample_split(train_words, n_train, seed)
    test_sample = sample_split(test_words, n_test, seed)
    return {
        "boreft_dir": boreft_dir,
        "task": task,
        "seed": seed,
        "n_train": n_train,
        "n_test": n_test,
        "train_pool_size": len(train_words),
        "test_pool_size": len(test_words),
        "test_source": test_source,
        "train_targets": train_sample,
        "test_targets": test_sample,
        "runs": _runs_for(train_sample, test_sample),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boreft_dir", "--boreft-dir", required=True)
    parser.add_argument("--n_train", "--n-train", type=int, default=5)
    parser.add_argument("--n_test", "--n-test", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Optional JSON path.")
    args = parser.parse_args()
    payload = sample_targets(
        args.boreft_dir,
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
    )
    text = json.dumps(payload, indent=2)
    if args.output:
        parent = os.path.dirname(os.path.abspath(args.output))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
