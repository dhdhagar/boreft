#!/usr/bin/env python
"""Compare Qwen embedding, Qwen3-1.7B last-token, and Llama clustering on Semantle.

The strings match ``task_config["semantle"]["embedding_prompt_defn"]`` — the same
text the bias network sees with ``--use-definition-embeds`` and
``--bias-network-encoder llm_encoder``:

    The meaning of '{word}' is: {definition}

Encoders (same three as ``analyze_molopt_encoder_cluster.py``):

  qwen       ``Qwen/Qwen3-Embedding-0.6B`` via sentence-transformers
  qwen_llm   ``Qwen/Qwen3-1.7B`` last-layer last-token (causal LM)
  llama      Llama-3.2-1B-Instruct last-layer, pooled like the bias-network
             llm_encoder (``last_instruction``; ``last_token`` is an alias)

``--max-length`` (default 128, same as ``--bias-encoder-max-length``) truncates
every encoder.

  python scripts/analyze_semantle_encoder_cluster.py --n 0
  python scripts/analyze_semantle_encoder_cluster.py --reuse-previous
  sbatch scripts/analyze_semantle_encoder_cluster.sh
  sbatch --export=ALL,REUSE_PREVIOUS=1 scripts/analyze_semantle_encoder_cluster.sh
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_SRC = os.path.join(_REPO_ROOT, "src")
for _path in (_SCRIPT_DIR, _SRC):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import analyze_molopt_encoder_cluster as mec  # noqa: E402
import analyze_molopt_text_variants as mtv  # noqa: E402
from boreft.search_wandb import add_wandb_cli, maybe_log_encoder_cluster  # noqa: E402
from boreft.task_config import definition_embedding_text, task_embedding_model  # noqa: E402
from boreft.text_similarity import (  # noqa: E402
    default_definitions_path,
    definitions_row_target,
)

TASK = "semantle"
VARIANT = "word_defn"
ENCODER_NAMES = mec.ENCODER_NAMES
DEFAULT_QWEN_MODEL = task_embedding_model(TASK)
DEFAULT_QWEN_LLM_MODEL = mec.DEFAULT_QWEN_LLM_MODEL
DEFAULT_LLAMA_MODEL = mec.DEFAULT_LLAMA_MODEL
DEFAULT_CACHE_DIR = mec.DEFAULT_CACHE_DIR
DEFAULT_POOLING = mec.DEFAULT_POOLING
DEFAULT_MAX_LENGTH = mec.DEFAULT_MAX_LENGTH

selected_encoders = mec.selected_encoders
artifact_tag = mec.artifact_tag
cache_tag = mec.cache_tag
artifact_path = mec.artifact_path


@dataclass
class Word:
    target: str
    definition: str
    category: str
    category_normalized: str


def variant_text(word: Word) -> str:
    return definition_embedding_text(TASK, word.target, word.definition)


def load_words(definitions_path: str) -> list[Word]:
    rows: list[Word] = []
    with open(definitions_path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            target = definitions_row_target(row)
            definition = str(row.get("definition") or "").strip()
            if not target or not definition:
                continue
            rows.append(
                Word(
                    target=target,
                    definition=definition,
                    category=str(row.get("category") or "unknown").strip() or "unknown",
                    category_normalized=str(
                        row.get("category_normalized")
                        or row.get("category")
                        or "unknown"
                    ).strip()
                    or "unknown",
                )
            )
    if not rows:
        raise ValueError(f"{definitions_path}: no (target, definition) pairs found")
    return rows


def sample_words(
    rows: Sequence[Word],
    n: int,
    seed: int,
    *,
    label_field: str,
    drop_unknown: bool = True,
) -> list[Word]:
    if n < 0:
        raise ValueError(f"--n must be >= 0, got {n}")
    pool = list(rows)
    if drop_unknown:
        pool = [
            word
            for word in pool
            if mtv.label_of(word, label_field) not in mtv._META_CLUSTERS
        ]
    if not pool:
        raise ValueError("no words left after dropping unknown / multi-cluster labels")
    rnd = random.Random(seed)
    rnd.shuffle(pool)
    if n and n < len(pool):
        return pool[:n]
    return pool


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--definitions",
        default=default_definitions_path(TASK),
        help="definitions.jsonl with category / category_normalized fields.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=mtv.DEFAULT_N,
        help="Sample size. 0 uses every eligible word.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--label-field",
        default="category_normalized",
        choices=("category_normalized", "category"),
    )
    parser.add_argument(
        "--keep-unknown",
        action="store_true",
        help="Keep words whose label is 'unknown' / 'multi-cluster'.",
    )
    parser.add_argument(
        "--encoders",
        nargs="*",
        default=[],
        choices=list(ENCODER_NAMES),
        help="Subset of encoders to run. Default: qwen, qwen_llm, and llama.",
    )
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-llm-model", default=DEFAULT_QWEN_LLM_MODEL)
    parser.add_argument("--llama-model", default=DEFAULT_LLAMA_MODEL)
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="HuggingFace cache for gated Llama weights. "
        "Empty string uses the HF default. Qwen still uses HF_HOME.",
    )
    parser.add_argument(
        "--pooling",
        default=DEFAULT_POOLING,
        choices=["last_instruction", "instruction_mean", "last_token"],
        help="Llama pooling. last_token is a legacy alias of last_instruction.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Truncate every encoder to this many tokens "
        "(default 128, matching --bias-encoder-max-length).",
    )
    parser.add_argument(
        "--qwen-batch-size", type=int, default=mec.DEFAULT_QWEN_BATCH_SIZE
    )
    parser.add_argument(
        "--qwen-llm-batch-size", type=int, default=mec.DEFAULT_QWEN_LLM_BATCH_SIZE
    )
    parser.add_argument(
        "--llama-batch-size", type=int, default=mec.DEFAULT_LLAMA_BATCH_SIZE
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out-dir",
        default=os.path.join("data", "semantle", "analysis"),
    )
    parser.add_argument(
        "--reuse-previous",
        action="store_true",
        help="Reload compatible encoder embeddings from --out-dir instead of "
        "re-encoding. Caches are written on every run as "
        "encoder_cluster_{name}_emb*.npy.",
    )
    add_wandb_cli(parser)
    return parser.parse_args(argv)


def run(
    args: argparse.Namespace,
    encode_fns: Optional[dict[str, mec.EncodeFn]] = None,
) -> dict:
    encoders = selected_encoders(args.encoders)
    if args.n < 0:
        raise ValueError(f"--n must be >= 0, got {args.n}")

    print(f"[analyze] loading words from {args.definitions}", flush=True)
    rows = load_words(args.definitions)
    words = sample_words(
        rows,
        n=args.n,
        seed=args.seed,
        label_field=args.label_field,
        drop_unknown=not args.keep_unknown,
    )
    labels = [mtv.label_of(word, args.label_field) for word in words]
    texts = [variant_text(word) for word in words]
    targets = [word.target for word in words]
    example = texts[0]
    print(
        f"[analyze] using {len(words)} words "
        f"(requested n={args.n}, seed={args.seed}, label={args.label_field})",
        flush=True,
    )
    preview = example if len(example) <= 140 else example[:137] + "..."
    print(f"[analyze] example {VARIANT}: {preview}", flush=True)

    scores: dict[str, dict] = {}
    embeddings: dict[str, np.ndarray] = {}
    unique_clusters = sorted(set(labels))
    style_map = mtv.build_cluster_style_map(unique_clusters)
    os.makedirs(args.out_dir, exist_ok=True)
    tag = artifact_tag(args)
    ctag = cache_tag(args)
    if tag:
        print(f"[analyze] artifact suffix{tag}", flush=True)

    reused: dict[str, tuple[np.ndarray, int]] = {}
    if args.reuse_previous:
        for name in encoders:
            loaded = mec.load_reusable_encoder(
                args.out_dir,
                ctag,
                name,
                args,
                targets,
                texts,
                variant=VARIANT,
            )
            if loaded is not None:
                reused[name] = loaded

    if encode_fns is not None:
        missing = [
            name for name in encoders if name not in encode_fns and name not in reused
        ]
        if missing:
            raise ValueError(f"encode_fns missing encoder(s): {missing}")

    for name in encoders:
        encode_fn = None
        if name in reused:
            emb, n_truncated = reused[name]
        else:
            print(f"[analyze] encoding {name} ({len(texts)} strings)...", flush=True)
            encode_fn = encode_fns[name] if encode_fns is not None else None
            n_truncated = 0
            if encode_fn is None:
                if name == "qwen":
                    encode_fn = mec.load_qwen_encode_fn(
                        args.qwen_model,
                        args.device,
                        args.qwen_batch_size,
                        args.max_length,
                    )
                elif name == "qwen_llm":
                    encode_fn = mec.load_qwen_llm_encode_fn(
                        args.qwen_llm_model,
                        max_length=args.max_length,
                        device=args.device,
                        batch_size=args.qwen_llm_batch_size,
                    )
                else:
                    encode_fn = mec.load_llama_encode_fn(
                        args.llama_model,
                        pooling=args.pooling,
                        max_length=args.max_length,
                        device=args.device,
                        batch_size=args.llama_batch_size,
                        cache_dir=args.cache_dir or None,
                        task=TASK,
                    )
                n_truncated = mec._count_truncated(
                    encode_fn.tokenizer,  # type: ignore[attr-defined]
                    texts,
                    args.max_length,
                )
                if n_truncated:
                    print(
                        f"[analyze] WARNING: {n_truncated}/{len(texts)} {name} inputs "
                        f"exceed --max-length={args.max_length} and will be truncated",
                        flush=True,
                    )
            try:
                emb = mtv._l2_normalize(mec._call_encode(encode_fn, words, texts))
            finally:
                mec._close_encode_fn(encode_fn)
            emb_path, _meta_path = mec.save_encoder_cache(
                args.out_dir,
                ctag,
                name,
                args,
                emb,
                targets,
                labels,
                texts,
                n_truncated,
                variant=VARIANT,
            )
            print(f"[analyze] wrote {emb_path}", flush=True)
        embeddings[name] = emb
        scores[name] = mtv.score_variant(emb, labels, args.seed)
        scores[name]["n_truncated"] = n_truncated
        print(
            f"  {name}: knn@5={scores[name]['knn_purity_at_5']:.3f} "
            f"kmeans={scores[name]['kmeans_purity']:.3f} "
            f"sil={scores[name]['silhouette_cosine']:.3f}",
            flush=True,
        )
        mtv.plot_variant_pca(
            emb,
            labels,
            title=f"{name}  —  {VARIANT} 2-D PCA, coloured by {args.label_field}",
            save_path=artifact_path(
                args.out_dir, f"encoder_cluster_{name}_pca", tag, "png"
            ),
            style_map=style_map,
            unique_clusters=unique_clusters,
        )

    mec.print_encoder_report(
        scores,
        labels,
        example,
        heading="Semantle word+defn clustering by encoder",
        variant=VARIANT,
    )

    report = {
        "task": TASK,
        "definitions_path": os.path.abspath(args.definitions),
        "n_requested": args.n,
        "n": len(words),
        "seed": args.seed,
        "variant": VARIANT,
        "example": example,
        "label_field": args.label_field,
        "label_counts": mtv.label_counts(labels),
        "pooling": args.pooling,
        "max_length": args.max_length,
        "qwen_model": args.qwen_model,
        "qwen_llm_model": args.qwen_llm_model,
        "llama_model": args.llama_model,
        "cache_dir": args.cache_dir or None,
        "encoders": encoders,
        "reused_encoders": [name for name in encoders if name in reused],
        "artifact_tag": tag,
        "cache_tag": ctag,
        "reuse_previous": bool(args.reuse_previous),
        "scores": mtv.strip_arrays(scores),
        "targets": targets,
        "labels": labels,
    }
    json_path = artifact_path(args.out_dir, "encoder_cluster", tag, "json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"[analyze] wrote {json_path}")

    written = [
        mec.plot_encoder_purity(
            scores,
            artifact_path(args.out_dir, "encoder_cluster_purity", tag, "png"),
            title="Production word+defn clustering by encoder",
        ),
        mec.plot_encoder_metrics(
            scores,
            artifact_path(args.out_dir, "encoder_cluster_metrics", tag, "png"),
        ),
        mec.plot_encoder_pca_grid(
            embeddings,
            labels,
            unique_clusters,
            style_map,
            artifact_path(args.out_dir, "encoder_cluster_pca", tag, "png"),
            suptitle="PCA of production word+defn embeddings",
        ),
    ]
    for path in written:
        if path:
            print(f"[analyze] wrote {path}")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not os.path.isfile(args.definitions):
        print(
            f"ERROR: {args.definitions} not found",
            file=sys.stderr,
        )
        return 1
    report = run(args)
    maybe_log_encoder_cluster(
        args.out_dir,
        report,
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        group=args.wandb_group,
        name=args.wandb_run_name,
        wandb_dir=args.wandb_dir,
        no_wandb=args.no_wandb,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
