#!/usr/bin/env python3
"""
Ranking / geometry metrics between learned low-rank word vectors and reference embeddings.

Compares:
  - Learned: rows from word_mu (VAE) or word_bias (linear), same as MM loss / interpolate.
  - Reference: embed_cache.pt if present; otherwise targets from items.json are encoded with
    the SentenceTransformer for the checkpoint's task
    (:func:`~boreft.text_similarity.embedding_model_name`).

Metrics:
  1. Spearman ρ on upper-triangular pairwise cosine similarities
  2. kNN overlap@k (and Recall@k — same value when both neighbor sets have size k)
  3. MRR: reciprocal rank of the first hit among top reference neighbors in learned ranking
  4. NDCG@k with reference cosine as relevance, learned cosine as score

Usage:
  python -m boreft.eval.semantle_ranking \\
    --output_dir outputs/out_... \\
    --model-name meta-llama/Llama-3.2-1B \\
    --cache_dir ~/.cache/huggingface \\
    --layer 13 --low_rank_dim 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

from boreft.data_utils import add_load_latest_argument, load_merged_run_config
from boreft.embed_cache import (
    embed_cache_tensor_to_numpy,
    expected_texts_for_rows,
    load_embed_cache_tensor,
    resolve_eval_embed_cache_path,
    resolve_or_build_embed_cache,
)
from boreft.bias_tables import get_bias_vector
from boreft.eval.semantle import _load_model_and_items
from boreft.text_similarity import embedding_model_name, eval_embed_cache_provenance


def load_ref_embeddings_np(
    path: str,
    words: Sequence[str],
    *,
    num_words: Optional[int] = None,
    indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Load reference rows from embed_cache.pt (first ``num_words`` or explicit ``indices``)."""
    expected = expected_texts_for_rows(
        words, num_rows=num_words, indices=indices
    )
    t = load_embed_cache_tensor(
        path,
        num_rows=num_words,
        indices=indices,
        expected_texts=expected,
        require_index=True,
    )
    return t.cpu().numpy().astype(np.float64)


def learned_matrix_from_intervention(reft_model, n_words: int) -> np.ndarray:
    """Stack per-word vectors [N, d] (word_mu or word_bias) for ids 0..n-1."""
    rows = [get_bias_vector(reft_model, i) for i in range(n_words)]
    return np.stack(rows, axis=0).astype(np.float64)


def learned_matrix_from_word_ids(reft_model, word_ids: Sequence[int]) -> np.ndarray:
    """Stack per-word vectors [N, d] for explicit training word ids."""
    rows = [get_bias_vector(reft_model, int(wid)) for wid in word_ids]
    return np.stack(rows, axis=0).astype(np.float64)


def encode_reference_embeddings_from_targets(
    targets: List[str],
    batch_size: int = 64,
    *,
    task: str = "semantle",
) -> np.ndarray:
    """Row i = embedding(targets[i]); same convention as prepare_semantle_embed_cache.py."""
    return encode_reference_embeddings(
        targets, batch_size=batch_size, task=task
    )


def pairwise_cosine_similarity(Z: np.ndarray) -> np.ndarray:
    """Z: [N, d] → S_ij = cos(z_i, z_j), shape [N, N]."""
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    zn = Z / np.maximum(norms, 1e-12)
    return zn @ zn.T


def upper_tri_vec(S: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(S.shape[0], k=1)
    return S[i, j].astype(np.float64)


def spearman_pairwise_similarity_rho(
    S_learned: np.ndarray,
    S_ref: np.ndarray,
    k: int = 10,
) -> Dict[str, Any]:
    """Average Spearman correlation over anchors, restricted to top‑k reference neighbors.

    Return the mean ρ across anchors (ignoring anchors with < 2 neighbors).
    """
    n = S_learned.shape[0]
    if n < 2:
        return {"rho": float("nan"), "n_anchors": 0, "k_effective": 0}

    k_eff = min(k, max(0, n - 1))
    if k_eff == 0:
        return {"rho": float("nan"), "n_anchors": 0, "k_effective": 0}

    rhos: List[float] = []
    for w in range(n):
        ref_sims = S_ref[w].copy()
        ref_sims[w] = -np.inf
        order_ref = np.argsort(-ref_sims)[:k_eff]
        if order_ref.size < 2:
            continue
        r_vals = ref_sims[order_ref].astype(np.float64)
        z_vals = S_learned[w, order_ref].astype(np.float64)

        r = spearmanr(r_vals, z_vals, nan_policy="omit")
        rho = getattr(r, "statistic", getattr(r, "correlation", np.nan))
        try:
            rho_f = (
                float(rho) if rho is not None and not np.isnan(rho) else float("nan")
            )
        except (TypeError, ValueError):
            rho_f = float("nan")
        if not np.isnan(rho_f):
            rhos.append(rho_f)

    if not rhos:
        return {"rho": float("nan"), "n_anchors": 0, "k_effective": k_eff}

    return {
        "rho": float(np.mean(rhos)),
        "n_anchors": len(rhos),
        "k_effective": k_eff,
    }


def knn_sets_from_similarity(S: np.ndarray, k: int) -> List[set]:
    n = S.shape[0]
    if n < 2:
        return [set() for _ in range(n)]
    k_eff = min(k, n - 1)
    out: List[set] = []
    for i in range(n):
        sims = S[i].copy()
        sims[i] = -np.inf
        idx = np.argsort(-sims)[:k_eff]
        out.append(set(idx.tolist()))
    return out


def knn_overlap_and_recall(
    S_learned: np.ndarray,
    S_ref: np.ndarray,
    ks: Sequence[int],
) -> Dict[str, Any]:
    n = S_learned.shape[0]
    result: Dict[str, Any] = {"n_words": n, "per_k": {}}
    for k in ks:
        k_eff = min(k, max(0, n - 1))
        if k_eff == 0:
            result["per_k"][str(k)] = {
                "k_effective": 0,
                "knn_overlap": float("nan"),
                "recall_at_k": float("nan"),
            }
            continue
        nz = knn_sets_from_similarity(S_learned, k)
        ne = knn_sets_from_similarity(S_ref, k)
        overlaps = []
        recalls = []
        for i in range(n):
            inter = nz[i] & ne[i]
            overlaps.append(len(inter) / k_eff)
            recalls.append(len(inter) / k_eff)
        result["per_k"][str(k)] = {
            "k_effective": k_eff,
            "knn_overlap": float(np.mean(overlaps)),
            "recall_at_k": float(np.mean(recalls)),
        }
    return result


def mean_reciprocal_rank_ref_neighbors(
    S_learned: np.ndarray,
    S_ref: np.ndarray,
    n_positives: int = 1,
) -> Dict[str, Any]:
    n = S_learned.shape[0]
    if n < 2:
        return {"mrr": float("nan"), "n_positives": n_positives, "n_words": n}

    ranks: List[float] = []
    for w in range(n):
        ref_sims = S_ref[w].copy()
        ref_sims[w] = -np.inf
        order_ref = np.argsort(-ref_sims)
        positives = set(order_ref[: min(n_positives, n - 1)].tolist())

        learned_sims = S_learned[w].copy()
        learned_sims[w] = -np.inf
        order_learned = np.argsort(-learned_sims)

        best_rank: Optional[int] = None
        for rank1, j in enumerate(order_learned, start=1):
            if j in positives:
                best_rank = rank1
                break
        if best_rank is None:
            ranks.append(0.0)
        else:
            ranks.append(1.0 / float(best_rank))

    return {
        "mrr": float(np.mean(ranks)),
        "n_positives": n_positives,
        "n_words": n,
    }


def ndcg_at_k_per_anchor(
    S_learned: np.ndarray,
    S_ref: np.ndarray,
    ks: Sequence[int],
    relevance_shift: str = "nonneg",
) -> Dict[str, Any]:
    n = S_learned.shape[0]
    ndcg_means: Dict[str, float] = {}

    for k in ks:
        scores: List[float] = []
        for w in range(n):
            mask = np.ones(n, dtype=bool)
            mask[w] = False
            rel = S_ref[w, mask].astype(np.float64)
            pred = S_learned[w, mask].astype(np.float64)

            if relevance_shift == "nonneg":
                rel = rel - rel.min() + 1e-8

            k_eff = min(k, rel.size)
            if k_eff < 1 or rel.size < 2:
                scores.append(0.0)
                continue

            y_true = rel.reshape(1, -1)
            y_score = pred.reshape(1, -1)
            try:
                s = ndcg_score(y_true, y_score, k=k_eff)
            except ValueError:
                s = 0.0
            scores.append(float(s))

        ndcg_means[str(k)] = float(np.mean(scores))

    return {
        "ndcg_mean_per_anchor": ndcg_means,
        "n_words": n,
        "relevance_shift": relevance_shift,
    }


def compute_all_metrics(
    Z: np.ndarray,
    E: np.ndarray,
    knn_ks: Sequence[int] = (5, 10, 20),
    ndcg_ks: Sequence[int] = (5, 10, 20),
    mrr_positives: int = 1,
) -> Dict[str, Any]:
    if Z.shape[0] != E.shape[0]:
        raise ValueError(f"Row mismatch: Z {Z.shape[0]} vs E {E.shape[0]}")
    sz = pairwise_cosine_similarity(Z)
    se = pairwise_cosine_similarity(E)

    return {
        # Use the largest k from knn_ks for the Spearman@k calculation
        "spearman_pairwise": spearman_pairwise_similarity_rho(sz, se, 10),
        "knn_recall": knn_overlap_and_recall(sz, se, knn_ks),
        "mrr": mean_reciprocal_rank_ref_neighbors(sz, se, n_positives=mrr_positives),
        "ndcg": ndcg_at_k_per_anchor(sz, se, ndcg_ks),
    }


def run_from_checkpoint(
    output_dir: str,
    model_name: str,
    layer: int,
    low_rank_dim: int,
    cache_dir: Optional[str],
    embed_cache_path: Optional[str] = None,
    knn_ks: Sequence[int] = (5, 10, 20),
    ndcg_ks: Sequence[int] = (5, 10, 20),
    mrr_positives: int = 1,
    auto_encode_ref: bool = True,
    encode_ref_batch_size: int = 64,
    *,
    reft_model=None,
    words: Optional[Sequence[str]] = None,
    word_ids: Optional[Sequence[int]] = None,
    load_latest: bool = False,
) -> Dict[str, Any]:
    if reft_model is None or words is None:
        reft_model, _tokenizer, words, _prompt, _items = _load_model_and_items(
            output_dir,
            model_name,
            layer,
            low_rank_dim,
            cache_dir,
            load_latest=load_latest,
        )
    n = len(words)
    if word_ids is not None:
        z = learned_matrix_from_word_ids(reft_model, word_ids)
    else:
        z = learned_matrix_from_intervention(reft_model, n)

    ref_path = resolve_eval_embed_cache_path(output_dir, embed_cache_path)

    if ref_path:
        if word_ids is not None:
            e = load_ref_embeddings_np(ref_path, words, indices=word_ids)
        else:
            e = load_ref_embeddings_np(ref_path, words, num_words=n)
        ref_meta: Dict[str, Any] = {
            "ref_source": "embed_cache_file",
            "embed_cache_path": ref_path,
            "sentence_transformer_model": None,
        }
    elif auto_encode_ref:
        saved_cfg = load_merged_run_config(output_dir)
        embed_task = str(saved_cfg.get("task", "semantle"))
        embed_model = embedding_model_name(embed_task)
        print(
            f"[semantle_ranking] No embed_cache file — encoding {n} targets with "
            f"SentenceTransformer({embed_model!r}) ...",
            file=sys.stderr,
        )
        tensor, pt_path, index = resolve_or_build_embed_cache(
            list(words),
            batch_size=encode_ref_batch_size,
            provenance=eval_embed_cache_provenance(task=embed_task),
        )
        if word_ids is not None:
            e = embed_cache_tensor_to_numpy(tensor, indices=word_ids)
        else:
            e = embed_cache_tensor_to_numpy(tensor, num_rows=n)
        ref_meta = {
            "ref_source": "embed_cache_built",
            "embed_cache_path": index["cache_pt"],
            "sentence_transformer_model": embed_model,
        }
    else:
        raise FileNotFoundError(
            "No embed_cache found and --no_auto_encode_ref was set. Pass --embed_cache_path, "
            "place embed_cache.pt in output_dir, set eval_embed_cache_path in "
            "intervention_config.json, or allow auto-encoding (default)."
        )

    metrics = compute_all_metrics(
        z, e, knn_ks=knn_ks, ndcg_ks=ndcg_ks, mrr_positives=mrr_positives
    )
    metrics["meta"] = {
        "output_dir": os.path.abspath(output_dir),
        "n_words": n,
        "learned_dim": int(z.shape[1]),
        "ref_dim": int(e.shape[1]),
        **ref_meta,
    }
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(
        description="Semantle learned vs reference ranking metrics"
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Run dir with items.json + intervenable_model/",
    )
    add_load_latest_argument(p)
    p.add_argument("--model-name", dest="model_name", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--layer", type=int, default=13)
    p.add_argument("--low_rank_dim", type=int, default=64)
    p.add_argument(
        "--embed_cache_path",
        default=None,
        help="Override embed_cache.pt (else intervention_config or output_dir/embed_cache.pt).",
    )
    p.add_argument(
        "--no_auto_encode_ref",
        action="store_true",
        help="Fail if no embed_cache file (do not encode items.json with SentenceTransformer).",
    )
    p.add_argument(
        "--encode_ref_batch_size",
        type=int,
        default=64,
        help="Batch size when auto-encoding reference targets (no embed_cache.pt).",
    )
    p.add_argument(
        "--knn_ks",
        default="5,10,20",
        help="Comma-separated k for kNN overlap / recall.",
    )
    p.add_argument("--ndcg_ks", default="5,10,20", help="Comma-separated k for NDCG.")
    p.add_argument(
        "--mrr_positives",
        type=int,
        default=1,
        help="Top reference neighbors treated as positives for MRR.",
    )
    p.add_argument("--out_json", default=None, help="Write metrics JSON to this path.")
    args = p.parse_args()

    knn_ks = [int(x.strip()) for x in args.knn_ks.split(",") if x.strip()]
    ndcg_ks = [int(x.strip()) for x in args.ndcg_ks.split(",") if x.strip()]

    metrics = run_from_checkpoint(
        output_dir=args.output_dir,
        model_name=args.model_name,
        layer=args.layer,
        low_rank_dim=args.low_rank_dim,
        cache_dir=args.cache_dir,
        embed_cache_path=args.embed_cache_path,
        knn_ks=knn_ks,
        ndcg_ks=ndcg_ks,
        mrr_positives=args.mrr_positives,
        auto_encode_ref=not args.no_auto_encode_ref,
        encode_ref_batch_size=args.encode_ref_batch_size,
        load_latest=bool(args.load_latest),
    )

    print(json.dumps(metrics, indent=2))
    if args.out_json:
        d = os.path.dirname(os.path.abspath(args.out_json))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[semantle_ranking] Wrote {args.out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
