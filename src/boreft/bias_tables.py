"""Materialize and load per-word bias tables exported from a trained bias network."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from boreft.pyreft.losses import clamp_logvar
from boreft.text_similarity import encode_reference_embeddings, encode_texts_as_is

BIAS_TABLES_FILENAME = "bias_tables.pt"


def _base_model_from_reft(reft_model):
    """Return the wrapped base causal LM from a pyvene IntervenableModel."""
    base = getattr(reft_model, "model", None)
    if base is None:
        raise AttributeError("reft_model has no .model (base causal LM)")
    return base


def _predict_bias_vectors_from_encoder_texts(
    reft_model,
    intervention,
    texts: Sequence[str],
    *,
    tokenizer,
    encoder_max_length: int,
    encoder_layer_index: Optional[int],
    batch_size: int,
    task: Optional[str] = None,
    word_definition_pairs: Optional[Sequence[tuple[str, str]]] = None,
) -> list[np.ndarray]:
    """LLM-encoder path: tokenized text -> penultimate -> encoder -> bias mu."""
    from boreft.pyreft.semantic_encoder import (
        POOLING_LAST_INSTRUCTION,
        capture_penultimate,
        encode_definition_inputs,
        normalize_encoder_pooling,
        pooling_requires_instruction_mask,
        pooling_uses_instruction_mask,
    )

    encoder = intervention.get_semantic_encoder()
    if encoder is None:
        raise RuntimeError(
            "llm_encoder mode requires a semantic encoder attached to the intervention"
        )
    if tokenizer is None:
        raise ValueError("llm_encoder predict requires tokenizer")

    pooling = normalize_encoder_pooling(
        getattr(encoder, "pooling", POOLING_LAST_INSTRUCTION)
    )
    raw_text_mode = (
        pooling_requires_instruction_mask(pooling) and word_definition_pairs is None
    )
    if pooling_requires_instruction_mask(pooling) and not raw_text_mode:
        if task is None or word_definition_pairs is None:
            raise ValueError(
                f"llm_encoder pooling={pooling!r} requires task and "
                "word_definition_pairs to build instruction masks"
            )
    build_instr = (
        word_definition_pairs is not None and pooling_uses_instruction_mask(pooling)
    )
    if build_instr and len(word_definition_pairs) != len(texts):
        raise ValueError("word_definition_pairs length must match texts")

    base_model = _base_model_from_reft(reft_model)
    device = _intervention_device(intervention)
    text_list = list(texts)
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(text_list), batch_size):
            batch_texts = text_list[start : start + batch_size]
            batch_pairs = (
                list(word_definition_pairs[start : start + batch_size])
                if word_definition_pairs is not None
                else None
            )
            input_ids, attn, instr = encode_definition_inputs(
                tokenizer,
                batch_texts,
                device=device,
                max_length=encoder_max_length,
                task=task,
                word_definition_pairs=batch_pairs,
                build_instruction_mask=build_instr,
            )
            penult = capture_penultimate(
                base_model, input_ids, attn, layer_index=encoder_layer_index
            )
            if raw_text_mode:
                instr = attn.long()
            feats = encoder(penult, attn, instruction_mask=instr)
            mu = intervention.bias_network.mu(feats)
            chunks.append(mu.detach().float().cpu().numpy())
    return [row for row in np.concatenate(chunks, axis=0)]


def _predict_bias_vectors_via_encoder(
    reft_model,
    intervention,
    words: Sequence[str],
    *,
    tokenizer,
    task: str,
    raw_definition_lookup: dict[str, str],
    encoder_max_length: int,
    encoder_layer_index: Optional[int],
    batch_size: int,
) -> list[np.ndarray]:
    """LLM-encoder path: definition prompt text -> penultimate -> encoder -> bias mu."""
    from boreft.task_config import definition_embedding_text

    if raw_definition_lookup is None:
        raise ValueError(
            "llm_encoder predict requires tokenizer and raw_definition_lookup"
        )

    missing = [w for w in words if str(w).strip() not in raw_definition_lookup]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} held-out words missing definitions for the encoder "
            f"(e.g. {preview})"
        )

    texts = [
        definition_embedding_text(task, str(w).strip(), raw_definition_lookup[str(w).strip()])
        for w in words
    ]
    pairs = [
        (str(w).strip(), raw_definition_lookup[str(w).strip()]) for w in words
    ]
    return _predict_bias_vectors_from_encoder_texts(
        reft_model,
        intervention,
        texts,
        tokenizer=tokenizer,
        encoder_max_length=encoder_max_length,
        encoder_layer_index=encoder_layer_index,
        batch_size=batch_size,
        task=task,
        word_definition_pairs=pairs,
    )


def encoder_definitions_path(saved_cfg: dict) -> str:
    """Resolve the JSONL definitions file for llm_encoder bias-network predict."""
    from boreft.text_similarity import default_definitions_path

    return (
        saved_cfg.get("bias_encoder_definitions_path")
        or saved_cfg.get("definitions_path")
        or default_definitions_path(str(saved_cfg.get("task", "semantle")))
    )


def _encoder_predict_kwargs(
    saved_cfg: dict,
    *,
    tokenizer,
    raw_definition_lookup: Optional[dict[str, str]] = None,
) -> dict:
    """Extra kwargs for :func:`predict_bias_vectors_for_words` in llm_encoder mode."""
    from boreft.text_similarity import (
        definition_text_for_cfg,
        rdkit_definition_lookup_for_cfg,
    )

    if raw_definition_lookup is None:
        from boreft.text_similarity import load_raw_definitions

        raw_definition_lookup = load_raw_definitions(encoder_definitions_path(saved_cfg))
    if saved_cfg.get("append_rdkit_definitions"):
        rdkit_lookup = rdkit_definition_lookup_for_cfg(saved_cfg)
        raw_definition_lookup = {
            target: definition_text_for_cfg(
                saved_cfg,
                target,
                definition,
                rdkit_lookup=rdkit_lookup,
                # Checkpoint-corpus rows must match the saved sidecar. Ad-hoc
                # valid SMILES supplied by viz/extend are computed on demand.
                require_lookup=False,
            )
            for target, definition in raw_definition_lookup.items()
        }
    return {
        "tokenizer": tokenizer,
        "raw_definition_lookup": raw_definition_lookup,
        "encoder_max_length": int(saved_cfg.get("bias_encoder_max_length", 64)),
        "encoder_layer_index": saved_cfg.get("bias_encoder_layer_index"),
    }


def bias_predict_kwargs(
    saved_cfg: dict,
    *,
    tokenizer,
    raw_definition_lookup: Optional[dict[str, str]] = None,
) -> dict:
    """Checkpoint-derived kwargs for the bias-network predict helpers.

    Covers both input sources: llm_encoder needs the tokenizer and definition text,
    while a sentence-transformer bias network needs the encoder it was trained with
    whenever that is not the task's own (``--bias-network-embed-model``). Returns
    empty for a plain sentence-transformer run on the task's model.
    """
    if saved_cfg.get("bias_input_source") == "llm_encoder":
        return _encoder_predict_kwargs(
            saved_cfg,
            tokenizer=tokenizer,
            raw_definition_lookup=raw_definition_lookup,
        )
    from boreft.text_similarity import bias_network_embed_model_from_cfg

    model = bias_network_embed_model_from_cfg(saved_cfg)
    return {"embed_model": model} if model else {}


def _intervention_from_reft_model(reft_model):
    iv = list(reft_model.interventions.values())[0]
    return iv[0] if isinstance(iv, (list, tuple)) else iv


def _intervention_device(intervention) -> torch.device:
    return intervention.rotate_layer.weight.device


def _bias_lookup_device(intervention) -> torch.device:
    if getattr(intervention, "embed_cache", None) is not None:
        return intervention.embed_cache.device
    return _intervention_device(intervention)


def get_bias_vector(reft_model, word_id: int) -> np.ndarray:
    """Return the per-word mean bias vector for ``word_id``."""
    iv = _intervention_from_reft_model(reft_model)
    device = _bias_lookup_device(iv)
    ids = torch.tensor([int(word_id)], dtype=torch.long, device=device)
    return iv.get_bias_mu(ids).detach().float().cpu().numpy()[0]


def stack_bias_vectors(reft_model, word_ids: List[int]) -> np.ndarray:
    """Stack per-word bias vectors ``[N, rank]`` for the given word ids."""
    iv = _intervention_from_reft_model(reft_model)
    device = _bias_lookup_device(iv)
    ids = torch.tensor([int(w) for w in word_ids], dtype=torch.long, device=device)
    return iv.get_bias_mu(ids).detach().float().cpu().numpy()


def stack_bias_mu_std(
    reft_model, word_ids: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack per-word posterior means and stds ``[N, rank]``.

    Std is ``exp(0.5 * clamp(logvar))``, matching the sampling distribution.
    """
    if not word_ids:
        raise ValueError("word_ids must be non-empty")
    iv = _intervention_from_reft_model(reft_model)
    device = _bias_lookup_device(iv)
    ids = torch.tensor([int(w) for w in word_ids], dtype=torch.long, device=device)
    was_training = iv.training
    iv.eval()
    try:
        with torch.no_grad():
            try:
                mu, logvar = iv.get_bias_mu_logvar(ids)
            except (AttributeError, TypeError) as exc:
                raise ValueError(
                    "checkpoint has no per-target posterior log-variance; "
                    "--aabb-std-k requires learned or fixed logvar"
                ) from exc
            if logvar is None:
                raise ValueError(
                    "checkpoint has no per-target posterior log-variance; "
                    "--aabb-std-k requires learned or fixed logvar"
                )
            std = torch.exp(0.5 * clamp_logvar(logvar.float()))
            mu_np = mu.detach().float().cpu().numpy()
            std_np = std.detach().float().cpu().numpy()
    finally:
        iv.train(was_training)
    if std_np.shape != mu_np.shape:
        raise ValueError("posterior std shape does not match means")
    if not np.isfinite(std_np).all() or np.any(std_np < 0):
        raise ValueError("posterior std must be finite and nonnegative")
    return mu_np, std_np


def training_search_region(
    reft_model,
    word_ids: List[int],
    *,
    padding: float = 0.0,
    aabb_std_k: float = 0.0,
    search_domain: str = "aabb",
) -> Tuple[np.ndarray, np.ndarray, Optional[Any]]:
    """Return ``(mu, bounds, ellipsoid)`` for the training-target search region.

    ``search_domain="aabb"`` is the axis-aligned box of posterior means
    (expanded by ``aabb_std_k``). ``search_domain="ellipsoid"`` is the
    covering Mahalanobis ellipsoid of those same generators (sample
    covariance, radius scaled to the means and, if ``aabb_std_k > 0``,
    the per-axis vertices ``μ_i ± k σ_{ij} e_j``); ``bounds`` is then that
    ellipsoid's axis-aligned bounding box, which the GP still uses for
    input scaling.
    """
    from boreft.bo.acquisition import LatentEllipsoid, latent_bounds, latent_ellipsoid

    if search_domain not in ("aabb", "ellipsoid"):
        raise ValueError("search_domain must be 'aabb' or 'ellipsoid'")
    if not np.isfinite(aabb_std_k) or aabb_std_k < 0:
        raise ValueError("aabb_std_k must be finite and nonnegative")
    ids = [int(w) for w in word_ids]
    if aabb_std_k > 0:
        mu, std = stack_bias_mu_std(reft_model, ids)
        mu = np.asarray(mu, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
    else:
        mu = np.asarray(stack_bias_vectors(reft_model, ids), dtype=np.float32)
        std = None
    if search_domain == "ellipsoid":
        ellipsoid: Optional[LatentEllipsoid] = latent_ellipsoid(
            mu, padding=padding, std=std, std_k=aabb_std_k
        )
        bounds = ellipsoid.aabb().astype(np.float32)
        return mu, bounds, ellipsoid
    bounds = latent_bounds(
        mu, padding=padding, std=std, std_k=aabb_std_k
    ).astype(np.float32)
    return mu, bounds, None


def training_search_domain(
    reft_model,
    word_ids: List[int],
    *,
    padding: float = 0.0,
    aabb_std_k: float = 0.0,
    search_domain: str = "aabb",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(mu, bounds)`` for the training-target search box.

    ``aabb_std_k == 0`` is the axis-aligned box of posterior means.
    ``aabb_std_k > 0`` expands by ``k`` times each target's posterior std.
    ``search_domain="ellipsoid"`` returns that ellipsoid's enclosing box.
    """
    mu, bounds, _ = training_search_region(
        reft_model,
        word_ids,
        padding=padding,
        aabb_std_k=aabb_std_k,
        search_domain=search_domain,
    )
    return mu, bounds


def predict_bias_vectors_for_words(
    reft_model,
    words: Sequence[str],
    *,
    definition_lookup: Optional[dict[str, str]] = None,
    batch_size: int = 64,
    task: str = "semantle",
    tokenizer=None,
    raw_definition_lookup: Optional[dict[str, str]] = None,
    encoder_max_length: int = 64,
    encoder_layer_index: Optional[int] = None,
    embed_model: Optional[str] = None,
) -> list[np.ndarray]:
    """Predict mean bias vectors for arbitrary words via the live bias network.

    Sentence-transformer mode (default): uses ``encode_reference_embeddings``
    (training embed-cache provenance) as input to ``bias_network.mu``, with
    ``embed_model`` overriding the task's encoder for runs whose bias network was
    trained in another embedding space.

    LLM-encoder mode (``intervention.bias_input_source == "llm_encoder"``):
    encodes each word's definition text with the base model, runs the copied
    final block + LoRA, and pools the last token before ``bias_network.mu``.
    Requires ``tokenizer`` and ``raw_definition_lookup``.

    Either way, materialized bias tables are ignored so OOV test targets are
    scored from the network, not frozen train rows.
    """
    intervention = _intervention_from_reft_model(reft_model)
    if getattr(intervention, "bias_network", None) is None:
        raise ValueError("predict_bias_vectors_for_words requires add_bias_network=True")

    word_list = list(words)
    if not word_list:
        return []

    if getattr(intervention, "bias_input_source", "embed_cache") == "llm_encoder":
        return _predict_bias_vectors_via_encoder(
            reft_model,
            intervention,
            word_list,
            tokenizer=tokenizer,
            task=task,
            raw_definition_lookup=raw_definition_lookup,
            encoder_max_length=encoder_max_length,
            encoder_layer_index=encoder_layer_index,
            batch_size=batch_size,
        )

    emb = encode_reference_embeddings(
        word_list,
        definition_lookup=definition_lookup,
        batch_size=batch_size,
        task=task,
        model_name=embed_model,
    )
    device = _intervention_device(intervention)
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(word_list), batch_size):
            e = torch.from_numpy(emb[start : start + batch_size]).to(
                device=device, dtype=torch.float32
            )
            chunks.append(intervention.bias_network.mu(e).detach().cpu().numpy())
    return [row for row in np.concatenate(chunks, axis=0)]


def predict_bias_vectors_from_raw_texts(
    reft_model,
    texts: Sequence[str],
    *,
    batch_size: int = 64,
    task: str = "semantle",
    tokenizer=None,
    encoder_max_length: int = 64,
    encoder_layer_index: Optional[int] = None,
    embed_model: Optional[str] = None,
) -> list[np.ndarray]:
    """Predict mean bias vectors from raw strings.

    Sentence-transformer mode (default): encodes strings as-is with ``embed_model``,
    or ``task``'s embedding model. Either way it must be the one the bias network was
    trained against — a different model need not even agree on the input dimension.

    LLM-encoder mode: tokenizes strings as-is, runs the base model through the
    copied final block + LoRA, then pools per ``bias_encoder_pooling`` (last
    instruction token by default; ``instruction_mean`` mean-pools all encoded
    tokens for raw /target-text input). Requires ``tokenizer``.
    """
    intervention = _intervention_from_reft_model(reft_model)
    if getattr(intervention, "bias_network", None) is None:
        raise ValueError(
            "predict_bias_vectors_from_raw_texts requires add_bias_network=True"
        )

    text_list = list(texts)
    if not text_list:
        return []

    if getattr(intervention, "bias_input_source", "embed_cache") == "llm_encoder":
        return _predict_bias_vectors_from_encoder_texts(
            reft_model,
            intervention,
            text_list,
            tokenizer=tokenizer,
            encoder_max_length=encoder_max_length,
            encoder_layer_index=encoder_layer_index,
            batch_size=batch_size,
        )

    emb = encode_texts_as_is(text_list, task=task, model_name=embed_model)
    device = _intervention_device(intervention)
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(text_list), batch_size):
            e = torch.from_numpy(emb[start : start + batch_size]).to(
                device=device, dtype=torch.float32
            )
            chunks.append(intervention.bias_network.mu(e).detach().cpu().numpy())
    return [row for row in np.concatenate(chunks, axis=0)]


@torch.no_grad()
def materialize_bias_tables(
    intervention,
    num_words: int,
    *,
    batch_size: int = 256,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    """Run the bias network over word ids ``0 .. num_words-1``.

    Always uses the live ``bias_network`` (ignores any in-memory materialized
    tables). Returns CPU float32 ``(mu, logvar, metadata)``. ``logvar`` is
    omitted when variance is fixed; ``metadata`` may contain ``fixed_logvar``.
    """
    if not getattr(intervention, "add_bias_network", False):
        raise ValueError("materialize_bias_tables requires add_bias_network=True")
    if getattr(intervention, "bias_network", None) is None:
        raise ValueError("intervention has no bias_network")

    device = _intervention_device(intervention)
    fixed_logvar = getattr(intervention, "fixed_logvar", None)
    learnable_logvar = fixed_logvar is None and getattr(
        intervention.bias_network, "learnable_logvar", False
    )

    # Materialization must be deterministic: run the intervention (and its encoder
    # submodule) in eval so LoRA/bias-network dropout is disabled, then restore.
    was_training = intervention.training
    intervention.eval()

    mu_chunks: list[torch.Tensor] = []
    logvar_chunks: list[torch.Tensor] = []

    try:
        for start in range(0, num_words, batch_size):
            end = min(start + batch_size, num_words)
            word_ids = torch.arange(start, end, dtype=torch.long, device=device)
            e = intervention.bias_features(word_ids)
            if learnable_logvar:
                mu, logvar = intervention.bias_network(e)
                mu_chunks.append(mu.detach().float().cpu())
                logvar_chunks.append(logvar.detach().float().cpu())
            else:
                mu = intervention.bias_network.mu(e)
                mu_chunks.append(mu.detach().float().cpu())
    finally:
        intervention.train(was_training)

    mu_all = torch.cat(mu_chunks, dim=0)
    if mu_all.shape[0] != num_words:
        raise RuntimeError(
            f"Expected {num_words} bias rows, got {mu_all.shape[0]}"
        )
    logvar_all = torch.cat(logvar_chunks, dim=0) if logvar_chunks else None
    metadata: Dict[str, Any] = {}
    if fixed_logvar is not None:
        metadata["fixed_logvar"] = float(fixed_logvar)
    if getattr(intervention, "embed_cache", None) is not None:
        metadata["target_embed_dim"] = int(intervention.embed_cache.shape[1])
    elif getattr(intervention.bias_network, "fc1", None) is not None:
        metadata["target_embed_dim"] = int(intervention.bias_network.fc1.in_features)
    return mu_all, logvar_all, metadata


def save_bias_tables(
    path: str,
    mu: torch.Tensor,
    logvar: Optional[torch.Tensor] = None,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Write ``bias_tables.pt`` and return its absolute path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload: Dict[str, Any] = {
        "mu": mu.detach().float().cpu(),
        "num_words": int(mu.shape[0]),
        "bias_dim": int(mu.shape[1]),
        "metadata": metadata or {},
    }
    if logvar is not None:
        payload["logvar"] = logvar.detach().float().cpu()
    torch.save(payload, path)
    return os.path.abspath(path)


def load_bias_tables(
    path: str,
    *,
    num_words: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    """Load ``(mu, logvar, metadata)`` from ``bias_tables.pt``."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"bias tables not found: {path}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "mu" in blob:
        mu = blob["mu"].float()
        logvar = blob.get("logvar")
        if logvar is not None:
            logvar = logvar.float()
        meta = dict(blob.get("metadata") or {})
        meta.setdefault("num_words", int(blob.get("num_words", mu.shape[0])))
        meta.setdefault("bias_dim", int(blob.get("bias_dim", mu.shape[1])))
    else:
        raise ValueError(f"Unrecognized bias_tables format in {path}")

    if num_words is not None and mu.shape[0] < num_words:
        raise ValueError(
            f"bias_tables has {mu.shape[0]} rows but expected at least {num_words}"
        )
    return mu, logvar, meta


def mean_per_dim_std(intervention, num_words: Optional[int] = None) -> Optional[float]:
    """Mean per-dimension sampling std ``exp(0.5 * logvar)`` for annotation plots."""
    fixed = getattr(intervention, "fixed_logvar", None)
    if fixed is not None:
        return float(math.exp(0.5 * fixed))

    tables = getattr(intervention, "materialized_logvar", None)
    if tables is not None:
        return float(torch.exp(0.5 * tables.float()).mean().item())

    if getattr(intervention, "word_logvar", None) is not None:
        w = intervention.word_logvar.weight.detach().float()
        return float(torch.exp(0.5 * w).mean().item())

    if (
        getattr(intervention, "bias_network", None) is not None
        and getattr(intervention.bias_network, "learnable_logvar", False)
        and num_words is not None
        and getattr(intervention, "embed_cache", None) is not None
    ):
        device = _intervention_device(intervention)
        chunks: list[torch.Tensor] = []
        batch_size = 256
        with torch.no_grad():
            for start in range(0, num_words, batch_size):
                end = min(start + batch_size, num_words)
                word_ids = torch.arange(start, end, dtype=torch.long, device=device)
                e = intervention.embed_cache[word_ids]
                _, logvar = intervention.bias_network(e)
                chunks.append(torch.exp(0.5 * logvar.float()).mean(dim=-1))
        return float(torch.cat(chunks).mean().item())

    return None


def typical_sampling_reach(
    intervention, bias_dim: int, *, num_words: Optional[int] = None
) -> Optional[float]:
    """Typical radius ``mean_sigma * sqrt(bias_dim)`` for silhouette histograms."""
    mean_sigma = mean_per_dim_std(intervention, num_words=num_words)
    if mean_sigma is None:
        return None
    return mean_sigma * math.sqrt(bias_dim)


def resolve_bias_tables_path(
    output_dir: str,
    explicit: Optional[str] = None,
    saved_cfg: Optional[Dict] = None,
) -> Optional[str]:
    """Resolve bias_tables.pt: explicit path, config, then ``output_dir/bias_tables.pt``."""
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"bias_tables path does not exist: {explicit}")
        return os.path.abspath(explicit)
    cfg = saved_cfg
    if cfg is None:
        cfg_path = os.path.join(output_dir, "intervention_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
    if cfg:
        p = cfg.get("bias_tables_path")
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    fallback = os.path.join(output_dir, BIAS_TABLES_FILENAME)
    if os.path.isfile(fallback):
        return os.path.abspath(fallback)
    return None


def detach_materialized_bias_tables(intervention) -> None:
    """Remove materialized lookup buffers (e.g. after export during training)."""
    for name in ("materialized_mu", "materialized_logvar"):
        if hasattr(intervention, name):
            delattr(intervention, name)


def attach_materialized_bias_tables(
    intervention,
    mu: torch.Tensor,
    logvar: Optional[torch.Tensor] = None,
) -> None:
    """Register materialized tables on the intervention for O(1) lookup."""
    device = _intervention_device(intervention)
    intervention.register_buffer(
        "materialized_mu", mu.detach().to(device=device, dtype=torch.float32)
    )
    if logvar is not None:
        intervention.register_buffer(
            "materialized_logvar",
            logvar.detach().to(device=device, dtype=torch.float32),
        )
    else:
        intervention.materialized_logvar = None


def export_bias_tables_for_checkpoint(
    reft_model,
    output_dir: str,
    num_words: int,
    *,
    batch_size: int = 256,
    update_config: bool = True,
    attach_in_memory: bool = False,
) -> Optional[str]:
    """Materialize bias tables from ``reft_model`` and write ``bias_tables.pt``.

    By default does **not** attach tables to the in-memory intervention so
    mid-training checkpoint export cannot freeze biases for subsequent steps.

    Returns the absolute path when exported, else ``None``.
    """
    intervention = _intervention_from_reft_model(reft_model)
    if not getattr(intervention, "add_bias_network", False):
        return None
    if getattr(intervention, "bias_network", None) is None:
        return None

    mu, logvar, meta = materialize_bias_tables(
        intervention, num_words, batch_size=batch_size
    )
    tables_path = os.path.join(output_dir, BIAS_TABLES_FILENAME)
    save_bias_tables(tables_path, mu, logvar, metadata=meta)
    if attach_in_memory:
        attach_materialized_bias_tables(intervention, mu, logvar)

    if update_config:
        cfg_path = os.path.join(output_dir, "intervention_config.json")
        cfg: Dict[str, Any] = {}
        if os.path.isfile(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
        cfg["bias_materialized"] = True
        cfg["bias_tables_path"] = os.path.abspath(tables_path)
        if "target_embed_dim" in meta:
            cfg["bias_network_embed_dim"] = meta["target_embed_dim"]
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    print(
        f"[bias_tables] Saved materialized tables ({num_words} words, "
        f"rank={mu.shape[1]}, logvar={'yes' if logvar is not None else 'fixed'}) "
        f"to {tables_path}",
        flush=True,
    )
    return os.path.abspath(tables_path)
