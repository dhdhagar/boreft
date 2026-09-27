#!/usr/bin/env python3
"""Sample reconstructed checkpoint-μ warmstarts and score them on TDC oracles.

Every search method on the same checkpoint, seed, and count draws this set.
Property-oracle names (DRD2 / GSK3B / JNK3) are not train SMILES, so they
share the same molecules. Different checkpoints can differ.

  python scripts/preview_molopt_checkpoint_warmstarts.py \\
    --reft-output-dir outputs/1789222254
  sbatch --export=ALL,REFT_OUTPUT_DIR=outputs/1789222254 \\
    scripts/preview_molopt_checkpoint_warmstarts.sh
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Callable, Mapping, Sequence

import numpy as np

from boreft.bias_tables import stack_bias_vectors
from boreft.bo import latent_bounds
from boreft.bo.runner import seed_everything
from boreft.chem import canonical_target_key
from boreft.data_utils import load_merged_run_config
from boreft.eval.semantle import load_eval_checkpoint, release_eval_checkpoint
from boreft.oracles import TDC_ORACLE_NAMES, load_oracles, score_molecules
from boreft.search import (
    SearchConfig,
    checkpoint_reconstruct_decode,
    memoize_point_decode,
    pinned_checkpoint_labels_file,
    select_warmstarts,
)

OracleMap = Mapping[str, Callable]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reft-output-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="JSON report directory (default: experiments/outputs/molopt/"
        "warmstarts/<runid>/checkpoint_recon)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--warmstart-count", type=int, default=10)
    parser.add_argument(
        "--warmstart-file",
        default=None,
        help="Pinned per-seed SMILES JSON. Lookup uses this checkpoint's μ.",
    )
    parser.add_argument("--oracles", nargs="+", default=list(TDC_ORACLE_NAMES))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--bounds-padding", type=float, default=0.0)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--load-latest", action="store_true")
    parser.add_argument(
        "--skip-oracle-check",
        action="store_true",
        help="Do not re-draw with each oracle name as --target",
    )
    return parser.parse_args(argv)


def default_output_dir(reft_output_dir: str) -> Path:
    run_id = Path(reft_output_dir).expanduser().resolve().name
    return (
        Path("experiments/outputs/molopt/warmstarts")
        / run_id
        / "checkpoint_recon"
    )


def _search_config(
    reft_output_dir: str,
    *,
    target: str,
    oracle: str,
    count: int,
    seeds: Sequence[int],
    max_new_tokens: int,
    bounds_padding: float,
    warmstart_file: str | None = None,
) -> SearchConfig:
    return SearchConfig(
        output_dir=reft_output_dir,
        target=target,
        oracle=oracle,
        budget=max(60, int(count)),
        warmstart_count=int(count),
        warmstart_source="checkpoint",
        warmstart_file=warmstart_file,
        seeds=tuple(int(seed) for seed in seeds),
        max_new_tokens=max_new_tokens,
        bounds_padding=bounds_padding,
    )


def _train_index(words: Sequence[str], label: str) -> int | None:
    key = canonical_target_key(label)
    if not key:
        return None
    for index, raw in enumerate(words):
        text = str(raw).strip()
        if not text:
            continue
        if canonical_target_key(text) == key:
            return index
    return None


def _score_row(smiles: str, oracles: OracleMap) -> dict[str, float | bool | str]:
    row: dict[str, float | bool | str] = {"smiles": smiles}
    canonical = ""
    valid = False
    for name, oracle in oracles.items():
        scored = score_molecules([smiles], oracle, include_qed=True)[0]
        row[name] = float(scored.score)
        valid = bool(scored.valid)
        canonical = scored.canonical or canonical
        if scored.qed is not None and "qed" not in row:
            row["qed"] = float(scored.qed)
    row["valid"] = valid
    row["canonical"] = canonical
    return row


def sample_seed(
    *,
    config_factory: Callable[[str], SearchConfig],
    ckpt,
    train_mu: np.ndarray,
    bounds: np.ndarray,
    seed: int,
    decode: Callable[[np.ndarray], str],
    oracle_names: Sequence[str],
    oracles: OracleMap,
    skip_oracle_check: bool,
) -> dict:
    primary = oracle_names[0]
    seed_everything(seed)
    points, records = select_warmstarts(
        config_factory(primary),
        ckpt,
        train_mu,
        bounds,
        seed,
        decode=decode,
    )
    labels = [str(record["decoded"]) for record in records]
    if not skip_oracle_check:
        for name in oracle_names[1:]:
            seed_everything(seed)
            _, other = select_warmstarts(
                config_factory(name),
                ckpt,
                train_mu,
                bounds,
                seed,
                decode=decode,
            )
            other_labels = [str(record["decoded"]) for record in other]
            if other_labels != labels:
                raise ValueError(
                    f"seed {seed}: warmstart labels for {name} differ from {primary}"
                )
    words = [str(word) for word in getattr(ckpt, "words", [])]
    rows = []
    for rank, (point, record, label) in enumerate(zip(points, records, labels)):
        gold = str(record["components"].get("warmstart_word") or label)
        greedy = str(decode(np.asarray(point, dtype=np.float32)))
        scored = _score_row(label, oracles)
        rows.append(
            {
                "rank": rank,
                "train_index": _train_index(words, gold),
                "warmstart_word": gold,
                "decoded": label,
                "greedy": greedy,
                "canonical": scored.get("canonical") or "",
                "valid": bool(scored.get("valid")),
                "reconstructed": bool(
                    record["components"].get("warmstart_reconstructed")
                ),
                "scores": {
                    name: float(scored[name])
                    for name in oracle_names
                    if name in scored
                },
                "qed": None if "qed" not in scored else float(scored["qed"]),
                "point": np.asarray(point, dtype=float).tolist(),
            }
        )
    return {
        "seed": int(seed),
        "labels": labels,
        "shared_across_oracles": not skip_oracle_check,
        "rows": rows,
    }


def format_preview(payload: dict) -> str:
    oracle_names = list(payload["oracles"])
    lines = [
        f"checkpoint  {payload['reft_output_dir']}",
        f"source      checkpoint μ + greedy reconstruct  "
        f"count={payload['warmstart_count']}",
        "shared      every method on this catalog uses this set per seed; "
        "oracles share it too",
        "",
    ]
    for block in payload["seeds"]:
        lines.append(f"seed {block['seed']}")
        header = ["i", "idx", "SMILES", *oracle_names]
        if any(row.get("qed") is not None for row in block["rows"]):
            header.append("QED")
        widths = [max(len(name), 4) for name in header]
        body: list[list[str]] = []
        for row in block["rows"]:
            smiles = str(row["decoded"])
            if len(smiles) > 48:
                smiles = smiles[:45] + "..."
            cells = [
                str(row["rank"]),
                "" if row["train_index"] is None else str(row["train_index"]),
                smiles,
                *[f"{float(row['scores'][name]):.4f}" for name in oracle_names],
            ]
            if "QED" in header:
                qed = row.get("qed")
                cells.append("" if qed is None else f"{float(qed):.3f}")
            body.append(cells)
            for i, cell in enumerate(cells):
                widths[i] = max(widths[i], len(cell))
        fmt = "  ".join(f"{{:<{width}}}" for width in widths)
        lines.append(fmt.format(*header))
        lines.extend(fmt.format(*row) for row in body)
        maxima = []
        for name in oracle_names:
            scores = [float(row["scores"][name]) for row in block["rows"]]
            maxima.append(f"{name} max={max(scores):.4f} mean={sum(scores)/len(scores):.4f}")
        lines.append("  " + "  |  ".join(maxima))
        misses = [row for row in block["rows"] if not row.get("reconstructed", True)]
        if misses:
            lines.append(
                f"  WARNING: {len(misses)} pinned μ did not greedily reconstruct; "
                "labeled means were still used"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def preview_checkpoint_warmstarts(
    *,
    ckpt,
    train_mu: np.ndarray,
    reft_output_dir: str,
    seeds: Sequence[int],
    warmstart_count: int,
    oracles: OracleMap,
    decode: Callable[[np.ndarray], str],
    bounds_padding: float = 0.0,
    max_new_tokens: int = 128,
    skip_oracle_check: bool = False,
    warmstart_file: str | None = None,
) -> dict:
    oracle_names = tuple(oracles)
    if not oracle_names:
        raise ValueError("at least one oracle is required")
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    bounds = latent_bounds(train_mu, padding=bounds_padding).astype(np.float32)

    def config_factory(oracle: str) -> SearchConfig:
        return _search_config(
            reft_output_dir,
            target=oracle,
            oracle=oracle,
            count=warmstart_count,
            seeds=seeds,
            max_new_tokens=max_new_tokens,
            bounds_padding=bounds_padding,
            warmstart_file=warmstart_file,
        )

    seed_blocks = [
        sample_seed(
            config_factory=config_factory,
            ckpt=ckpt,
            train_mu=train_mu,
            bounds=bounds,
            seed=seed,
            decode=decode,
            oracle_names=oracle_names,
            oracles=oracles,
            skip_oracle_check=skip_oracle_check,
        )
        for seed in seeds
    ]
    return {
        "reft_output_dir": os.path.abspath(reft_output_dir),
        "warmstart_source": "checkpoint",
        "warmstart_file": warmstart_file,
        "warmstart_count": int(warmstart_count),
        "oracles": list(oracle_names),
        "seeds": seed_blocks,
        "identity": "canonical SMILES; greedy decode at the training prompt, no SmiSelf",
    }


def run_preview(args: argparse.Namespace) -> dict:
    saved = load_merged_run_config(args.reft_output_dir)
    task = saved.get("task", "semantle")
    if task != "molopt":
        raise ValueError(f"expected a molopt checkpoint, got task={task!r}")
    model_name = saved.get("model_name")
    if not model_name:
        raise ValueError("checkpoint config has no model_name")
    oracle_names = tuple(
        dict.fromkeys(str(name).strip() for name in args.oracles if str(name).strip())
    )
    oracles = load_oracles(oracle_names)
    dest = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else default_output_dir(args.reft_output_dir)
    )
    dest.mkdir(parents=True, exist_ok=True)
    ckpt = load_eval_checkpoint(
        args.reft_output_dir,
        model_name,
        int(saved.get("layer", 15)),
        int(saved.get("low_rank_dim", 8)),
        args.cache_dir or saved.get("cache_dir"),
        load_latest=bool(args.load_latest),
    )
    try:
        train_mu = stack_bias_vectors(
            ckpt.reft_model, list(range(len(ckpt.words)))
        ).astype(np.float32)
        decode = memoize_point_decode(
            checkpoint_reconstruct_decode(
                ckpt, args.max_new_tokens, task=task
            )
        )
        warmstart_file = args.warmstart_file or pinned_checkpoint_labels_file(
            args.reft_output_dir, saved_cfg=saved
        )
        payload = preview_checkpoint_warmstarts(
            ckpt=ckpt,
            train_mu=train_mu,
            reft_output_dir=args.reft_output_dir,
            seeds=args.seeds,
            warmstart_count=args.warmstart_count,
            oracles=oracles,
            decode=decode,
            bounds_padding=args.bounds_padding,
            max_new_tokens=args.max_new_tokens,
            skip_oracle_check=bool(args.skip_oracle_check),
            warmstart_file=warmstart_file,
        )
    finally:
        release_eval_checkpoint(ckpt)
    text = format_preview(payload)
    print(text, flush=True)
    report = dest / "warmstarts.json"
    table = dest / "warmstarts.txt"
    report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    table.write_text(text, encoding="utf-8")
    print(f"[warmstarts] wrote {report}", flush=True)
    return payload


def main(argv: list[str] | None = None) -> int:
    run_preview(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
