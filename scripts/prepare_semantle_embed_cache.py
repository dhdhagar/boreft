#!/usr/bin/env python3
"""
Build items.json (same vocabulary/order as boreft.train) and resolve or build
embed_cache under repo-root embed_cache/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from boreft.data import SemantleItem
from boreft.embed_cache import embed_texts_from_items, resolve_or_build_embed_cache
from boreft.text_similarity import embedding_provenance, training_cache_params


def main() -> None:
    p = argparse.ArgumentParser(
        description="items.json + SentenceTransformer embed_cache for a semantle run."
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Run folder (receives items.json; embed_cache lives under repo-root embed_cache/).",
    )
    p.add_argument(
        "--semantle_csv", nargs="+", required=True, help="Same paths as boreft.train --semantle-csv."
    )
    p.add_argument("--train_top_k", type=int, required=True, help="Same as boreft.train --train-top-k.")
    p.add_argument(
        "--use_definition_embeds",
        action="store_true",
        help="Same as boreft.train --use-definition-embeds.",
    )
    p.add_argument(
        "--definitions_path",
        default=None,
        help="Same as boreft.train --definitions-path.",
    )
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    _, words, sim_map = SemantleItem.load_csvs(args.semantle_csv, top_k=args.train_top_k)
    items = SemantleItem.load(words, sim_map)
    n = len(items)
    if n == 0:
        raise ValueError("empty vocabulary from load_csvs")

    items_path = os.path.join(args.output_dir, "items.json")
    with open(items_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"id": it.id, "prompt": it.prompt, "target": it.target} for it in items],
            f,
            indent=2,
        )
    print(f"[prepare_semantle_embed_cache] Wrote {n} entries to {items_path}")

    texts = embed_texts_from_items(items)
    train_prov, definition_lookup, _ = training_cache_params(
        use_definition_embeds=args.use_definition_embeds,
        task="semantle",
        words=texts,
        definitions_path=args.definitions_path,
    )
    sources = {
        "semantle_csv": [os.path.abspath(p) for p in args.semantle_csv],
        "train_top_k": args.train_top_k,
        "items_json": os.path.abspath(items_path),
    }
    _, train_path, train_index = resolve_or_build_embed_cache(
        texts,
        sources=sources,
        batch_size=args.batch_size,
        provenance=train_prov,
        definition_lookup=definition_lookup,
    )
    print(
        f"[prepare_semantle_embed_cache] training embed_cache at {train_path} "
        f"shape=({train_index['num_rows']}, {train_index['embed_dim']})"
    )
    if args.use_definition_embeds:
        _, eval_path, eval_index = resolve_or_build_embed_cache(
            texts,
            sources=sources,
            batch_size=args.batch_size,
            provenance=embedding_provenance(use_definition_embeds=False, task="semantle"),
            definition_lookup=None,
        )
        print(
            f"[prepare_semantle_embed_cache] eval embed_cache at {eval_path} "
            f"shape=({eval_index['num_rows']}, {eval_index['embed_dim']})"
        )


if __name__ == "__main__":
    main()
