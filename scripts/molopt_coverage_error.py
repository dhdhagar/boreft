#!/usr/bin/env python3
"""Vertex coverage error ε_cov for molopt p90 catalogs (paper Semantle analog).

A = training vertices, so each eval molecule is Euclidean distance in φ to the
nearest training-target embedding. φ is encode_texts_normalized(..., task=molopt).
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from boreft.chem import canonical_target_key, is_valid_smiles, unwrap_smiles_tags
from boreft.oracle_screen import smiles_from_items_json
from boreft.text_similarity import encode_texts_normalized

REPO = Path(__file__).resolve().parents[1]
CSV_PATH = REPO / "data/molopt/train/chebi20.csv"
OUT_PATH = REPO / "data/molopt/analysis/p90_coverage_error.json"
CKPTS = {
    1024: REPO / "outputs/1789921464",
    2048: REPO / "outputs/1789933836",
    3072: REPO / "outputs/1789933841",
}
BATCH = 64


def summarize(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
        "min": float(values.min()),
    }


def load_csv_smiles(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        cols = reader.fieldnames or []
        col = "smiles" if "smiles" in cols else cols[0]
        out = []
        for row in reader:
            text = unwrap_smiles_tags(str(row.get(col) or "").strip())
            if text:
                out.append(text)
    return out


def encode(texts: list[str]) -> np.ndarray:
    chunks = []
    for start in range(0, len(texts), BATCH):
        batch = texts[start : start + BATCH]
        print(f"[cov] encode {start + 1}-{start + len(batch)} / {len(texts)}", flush=True)
        chunks.append(encode_texts_normalized(batch, task="molopt"))
    return np.concatenate(chunks, axis=0).astype(np.float64)


def nn_l2(eval_phi: np.ndarray, train_phi: np.ndarray) -> np.ndarray:
    dots = eval_phi @ train_phi.T
    d2 = np.clip(2.0 - 2.0 * dots, 0.0, None)
    return np.sqrt(d2).min(axis=1)


def main() -> None:
    corpus_raw = load_csv_smiles(CSV_PATH)
    corpus_keys = []
    corpus_keep = []
    seen = set()
    for smiles in corpus_raw:
        if not is_valid_smiles(smiles):
            continue
        key = canonical_target_key(smiles)
        if key in seen:
            continue
        seen.add(key)
        corpus_keys.append(key)
        corpus_keep.append(smiles)
    key_to_idx = {k: i for i, k in enumerate(corpus_keys)}

    catalogs: dict[int, list[str]] = {}
    for n, path in CKPTS.items():
        smiles = smiles_from_items_json(str(path / "items.json"))
        keys = []
        for s in smiles:
            if not is_valid_smiles(s):
                continue
            keys.append(canonical_target_key(s))
        catalogs[n] = keys
        print(f"[cov] N={n} train={len(keys)} from {path}", flush=True)

    split = json.loads((CKPTS[1024] / "oracle_split.json").read_text())
    test_keys = []
    for s in split.get("test_smiles") or []:
        if s and is_valid_smiles(s):
            test_keys.append(canonical_target_key(s))
    test_keys = [k for k in dict.fromkeys(test_keys) if k in key_to_idx]
    print(
        f"[cov] corpus={len(corpus_keys)} high-tail={len(test_keys)} "
        f"n_train_eligible={split.get('n_train_eligible')}",
        flush=True,
    )

    phi = encode(corpus_keep)
    union_train = set().union(*catalogs.values())
    p90_eligible = [k for k in corpus_keys if k not in set(test_keys)]
    shared_unused = [k for k in p90_eligible if k not in union_train]

    payload = {
        "protocol": (
            "vertices A: nearest-train Euclidean distance in L2-normalized "
            "Qwen3-Embedding-0.6B molopt φ (same as Semantle paper coverage line)"
        ),
        "n_corpus": len(corpus_keys),
        "n_high_tail": len(test_keys),
        "n_shared_unused_p90": len(shared_unused),
        "by_n": {},
    }
    for n, train_keys in catalogs.items():
        train_idx = [key_to_idx[k] for k in train_keys if k in key_to_idx]
        train_phi = phi[np.array(train_idx)]
        train_set = set(train_keys)

        def stats_for(keys: list[str]) -> dict:
            idx = [key_to_idx[k] for k in keys if k in key_to_idx and k not in train_set]
            if not idx:
                return {"n": 0}
            dists = nn_l2(phi[np.array(idx)], train_phi)
            return summarize(dists)

        leftover_p90 = [k for k in p90_eligible if k not in train_set]
        leftover_corpus = [k for k in corpus_keys if k not in train_set]
        block = {
            "n_train": len(train_idx),
            "high_tail": stats_for(test_keys),
            "unused_p90": stats_for(leftover_p90),
            "corpus_heldout": stats_for(leftover_corpus),
            "shared_unused_p90": stats_for(shared_unused),
        }
        payload["by_n"][str(n)] = block
        print(json.dumps({"n": n, **{k: v.get("median") if isinstance(v, dict) else v for k, v in block.items()}}), flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[cov] wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
