"""Shared decode + recall-hit helpers for bias-recovery experiments.

These are the canonical implementations reused by both
``experiments/recall_recovery_learn.py`` and ``boreft.extend_subspace`` so the
test-set loading, bias prediction, greedy/temperature decoding, and recall@n hit
accounting stay in one place.
"""

from __future__ import annotations

import json
import os
import random
from typing import Optional, Sequence

import numpy as np
import torch

from boreft.bias_tables import bias_predict_kwargs, predict_bias_vectors_for_words
from boreft.eval.eval_suite import (
    DEFAULT_BBOX_PCA_VAR,
    DEFAULT_TEST_N_SAMPLES,
    build_test_sets,
    target_normalizer,
    task_csv_paths,
)
from boreft.eval.semantle import (
    _run_greedy_decode_experiment,
    _run_sampling_experiment,
)
from boreft.text_similarity import definition_lookup_for_cfg


# ─────────────────────────────────────────────────────────────────────────────
# Test set
# ─────────────────────────────────────────────────────────────────────────────


def load_test_set(output_dir: str, ckpt, saved_cfg: dict) -> tuple[list[str], list[str], str]:
    """Return ``(interp, extrap, source)`` — reuse eval's split when available."""
    task = saved_cfg.get("task", "semantle")
    results_path = os.path.join(output_dir, "eval", "results.json")
    if os.path.isfile(results_path):
        with open(results_path, encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("recon_test_meta", {})
        interp = meta.get("interp_words")
        extrap = meta.get("extrap_words")
        if interp is not None and extrap is not None:
            return list(interp), list(extrap), "eval/results.json"

    # Lazy import avoids a module-load cycle (learn_bias imports eval.semantle).
    from boreft.learn_bias import get_train_embeddings

    train_emb = get_train_embeddings(ckpt, task=task)
    from boreft.molopt_split import load_oracle_split

    oracle_split = load_oracle_split(output_dir, saved_cfg)
    high_tail = (oracle_split or {}).get("test_smiles") if oracle_split else None
    interp, extrap, _meta = build_test_sets(
        train_targets=ckpt.words,
        csv_paths=task_csv_paths(saved_cfg, task=task),
        test_n_samples=int(saved_cfg.get("test_n_samples", DEFAULT_TEST_N_SAMPLES)),
        seed=int(saved_cfg.get("seed", 42)),
        pca_var=float(saved_cfg.get("eval_bbox_pca_var", DEFAULT_BBOX_PCA_VAR)),
        task=task,
        train_embeddings=train_emb,
        pool=high_tail,
        pool_source="oracle_high_tail" if high_tail else None,
    )
    return list(interp), list(extrap), "rebuilt"


# ─────────────────────────────────────────────────────────────────────────────
# Bias prediction
# ─────────────────────────────────────────────────────────────────────────────


def predict_test_bias_vectors(ckpt, words: Sequence[str], batch_size: int) -> list[np.ndarray]:
    """Bias-network predicted μ for each word (training provenance)."""
    saved_cfg = ckpt.saved_cfg or {}
    task = saved_cfg.get("task", "semantle")
    def_lookup = definition_lookup_for_cfg(saved_cfg)
    enc_kwargs = bias_predict_kwargs(saved_cfg, tokenizer=ckpt.tokenizer)
    return predict_bias_vectors_for_words(
        ckpt.reft_model,
        list(words),
        definition_lookup=def_lookup,
        batch_size=batch_size,
        task=task,
        **enc_kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Decoding
# ─────────────────────────────────────────────────────────────────────────────


def _decode_common(ckpt) -> dict:
    saved_cfg = ckpt.saved_cfg or {}
    return dict(
        position=saved_cfg.get("position", "l1"),
        assistant_suffix=ckpt.assistant_suffix,
        from_chat_template=ckpt.from_chat_template,
        intervention_token_id=ckpt.intervention_token_id,
        content_span=ckpt.content_span,
    )


def decode_targets(
    ckpt,
    words: Sequence[str],
    *,
    word_ids: Optional[Sequence[int]] = None,
    bias_vectors: Optional[Sequence[np.ndarray]] = None,
    temps: Sequence[float],
    n_samples: int,
    top_p: float,
    batch_size: int,
    max_new_tokens: int,
    decode_seed: Optional[int] = None,
) -> tuple[list[str], dict[float, list[dict]]]:
    """Greedy + per-temperature sampled decodes, via ``word_ids`` or raw bias."""
    if decode_seed is not None:
        random.seed(decode_seed)
        np.random.seed(decode_seed)
        torch.manual_seed(decode_seed)
    saved_cfg = ckpt.saved_cfg or {}
    task = saved_cfg.get("task", "semantle")
    common = dict(max_new_tokens=max_new_tokens, **_decode_common(ckpt))
    greedy = _run_greedy_decode_experiment(
        ckpt.reft_model,
        ckpt.tokenizer,
        list(words),
        ckpt.prompt,
        word_ids=list(word_ids) if word_ids is not None else None,
        bias_vectors=list(bias_vectors) if bias_vectors is not None else None,
        batch_size=batch_size,
        **common,
    )
    temp_results: dict[float, list[dict]] = {}
    for t in temps:
        temp_results[t] = _run_sampling_experiment(
            ckpt.reft_model,
            ckpt.tokenizer,
            list(words),
            ckpt.prompt,
            n_samples,
            word_ids=list(word_ids) if word_ids is not None else None,
            bias_vectors=list(bias_vectors) if bias_vectors is not None else None,
            use_sample=True,
            temperature=t,
            top_p=top_p,
            batch_size=batch_size,
            task=task,
            **common,
        )
    return greedy, temp_results


# ─────────────────────────────────────────────────────────────────────────────
# Hit accounting
# ─────────────────────────────────────────────────────────────────────────────


def greedy_hits(
    words: Sequence[str], greedy: Sequence[str], *, task: str = "semantle"
) -> dict[str, bool]:
    norm = target_normalizer(task)
    return {w: norm(d) == norm(w) for w, d in zip(words, greedy)}


def recall_hits_at(
    words: Sequence[str],
    temp_results: dict[float, list[dict]],
    temp: float,
    *,
    task: str = "semantle",
) -> dict[str, bool]:
    """recall@n hit map for a single temperature."""
    norm = target_normalizer(task)
    per: dict[str, bool] = {}
    for w, res in zip(words, temp_results[temp]):
        samples = {norm(x) for x in res["samples"]}
        per[w] = norm(w) in samples
    return per


def recall_hits(
    words: Sequence[str],
    temp_results: dict[float, list[dict]],
    temps: Sequence[float],
    *,
    task: str = "semantle",
) -> dict[float, dict[str, bool]]:
    """recall@n hit maps keyed by temperature."""
    return {
        t: recall_hits_at(words, temp_results, t, task=task) for t in temps
    }
