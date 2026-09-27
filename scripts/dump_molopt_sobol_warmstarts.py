#!/usr/bin/env python3
"""Decode shared Sobol warmstarts for molopt property search.

Each seed writes ``seed_<n>.jsonl`` with latent ``point`` plus the T=1 decode
so every method and every oracle scores the same molecules.

  python scripts/dump_molopt_sobol_warmstarts.py \
    --reft-output-dir outputs/1789222254 \
    --output-dir experiments/outputs/molopt/warmstarts/1789222254 \
    --seeds 1 2 3 4 5 \
    --warmstart-count 10 \
    --sampling-temperature 1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from boreft.bias_tables import stack_bias_vectors
from boreft.bo import latent_bounds
from boreft.bo.runner import seed_everything
from boreft.data_utils import load_merged_run_config
from boreft.eval.semantle import load_eval_checkpoint, release_eval_checkpoint
from boreft.search import SearchConfig, checkpoint_decode, select_warmstarts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reft-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--warmstart-count", type=int, default=10)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--bounds-padding", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def dump_sobol_warmstarts(args: argparse.Namespace) -> Path:
    dest = Path(args.output_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(seed) for seed in args.seeds)
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    if args.warmstart_count < 2:
        raise ValueError("warmstart_count must be at least 2")

    existing = [dest / f"seed_{seed}.jsonl" for seed in seeds]
    if all(path.is_file() for path in existing) and not args.overwrite:
        print(f"[warmstarts] already present in {dest}; skip dump", flush=True)
        return dest

    saved = load_merged_run_config(args.reft_output_dir)
    task = saved.get("task", "semantle")
    if task != "molopt":
        raise ValueError(f"expected a molopt checkpoint, got task={task!r}")
    model_name = saved.get("model_name")
    if not model_name:
        raise ValueError("checkpoint config has no model_name")
    ckpt = load_eval_checkpoint(
        args.reft_output_dir,
        model_name,
        int(saved.get("layer", 15)),
        int(saved.get("low_rank_dim", 8)),
        saved.get("cache_dir"),
    )
    try:
        train_mu = stack_bias_vectors(
            ckpt.reft_model, list(range(len(ckpt.words)))
        ).astype(np.float32)
        bounds = latent_bounds(train_mu, padding=args.bounds_padding).astype(
            np.float32
        )
        decode = checkpoint_decode(
            ckpt,
            args.max_new_tokens,
            sampling_temperature=args.sampling_temperature,
            task=task,
        )
        config = SearchConfig(
            output_dir=args.reft_output_dir,
            target="_",
            budget=max(60, args.warmstart_count),
            warmstart_count=args.warmstart_count,
            warmstart_source="sobol",
            seeds=seeds,
            max_new_tokens=args.max_new_tokens,
            sampling_temperature=args.sampling_temperature,
            bounds_padding=args.bounds_padding,
        )
        manifest = {
            "reft_output_dir": os.path.abspath(args.reft_output_dir),
            "warmstart_count": args.warmstart_count,
            "sampling_temperature": args.sampling_temperature,
            "max_new_tokens": args.max_new_tokens,
            "seeds": list(seeds),
            "files": {},
        }
        for seed in seeds:
            seed_everything(seed)
            points, _records = select_warmstarts(
                config, ckpt, train_mu, bounds, seed
            )
            path = dest / f"seed_{seed}.jsonl"
            temporary = path.with_suffix(path.suffix + ".tmp")
            n_valid = 0
            with temporary.open("w", encoding="utf-8") as handle:
                for index, point in enumerate(points):
                    decoded = decode(np.asarray(point, dtype=np.float32))
                    if decoded.strip():
                        n_valid += 1
                    handle.write(
                        json.dumps(
                            {
                                "point": np.asarray(point, dtype=float).tolist(),
                                "decoded": decoded,
                                "seed": seed,
                                "index": index,
                            }
                        )
                        + "\n"
                    )
            os.replace(temporary, path)
            manifest["files"][str(seed)] = path.name
            print(
                f"[warmstarts] seed {seed}: {len(points)} points -> {path} "
                f"({n_valid} non-empty decodes)",
                flush=True,
            )
        (dest / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        release_eval_checkpoint(ckpt)
    print(f"[warmstarts] wrote {dest}", flush=True)
    return dest


def main(argv: list[str] | None = None) -> int:
    dump_sobol_warmstarts(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
