#!/usr/bin/env python3
"""
Standalone RECON / RECON_TEST / GENZ / DIST evaluation for semantle-trained models.

Loads a saved checkpoint and computes the end-of-training eval-suite metrics:
  - RECON: greedy reconstruction of train targets (b=μ) + recall under
    temperature sampling at ``EVAL_TEMPERATURES``.
  - RECON_TEST: same RECON metrics on held-out test targets when
    ``add_bias_network`` is set (bias predicted from target embeddings).
  - DIST:  output-distribution stats (modal-match, sample-sim mean/std) from the
    same temperature sample bags.
  - GENZ:  Sobol space sampling (greedy + temperature) over the μ bounding box,
    plus a non-train test set split into interp/extrap subsets for recall
    coverage. Runs only when ``--n_uniform`` is set.

LIPZ (interpolation continuity) is computed by ``boreft.eval.run_full_eval``.
Results are written to ``output_dir/eval/results.json`` (and
``output_dir/eval/sobol_results.json`` when GENZ runs).

Usage:
  python -m boreft.eval.semantle --output_dir outputs/out_semantle-xxx --model-name meta-llama/Llama-3.2-1B --layer 13 --low_rank_dim 64 --n_samples 25 --n_uniform 100 --full_eval_n_samples 20
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from boreft.bias_tables import (
    attach_materialized_bias_tables,
    bias_predict_kwargs,
    load_bias_tables,
    predict_bias_vectors_for_words,
    resolve_bias_tables_path,
    stack_bias_vectors,
)
from boreft.pyreft import (
    DistributionalWordIntervention,
    LoreftPerWordBiasIntervention,
    ReftConfig,
    build_reft_representation,
    get_reft_model,
)
from boreft.task_config import task_instruction
from boreft.embed_cache import (
    bias_network_input_dim_from_cfg,
    resolve_embed_cache_path,
    resolve_or_build_embed_cache,
)
from boreft.data_utils import (
    add_load_latest_argument,
    decode_generated_text,
    load_checkpoint_tokenizer,
    load_merged_run_config,
    resolve_assistant_suffix,
    infer_torch_dtype_name,
    resolve_torch_dtype,
    prompt_tokenization_from_cfg,
    tokenize_model_text,
    system_prompt_from_cfg,
    TORCH_DTYPE_NAMES,
)
from boreft.text_similarity import (
    definition_lookup_for_cfg,
    embedding_model_name,
    embedding_sim_per_text,
    encode_texts_normalized,
    rdkit_map_path_for_cfg,
    training_embed_cache_provenance_from_cfg,
    warn_if_embedding_provenance_stale,
)
from boreft.intervention_marker import (
    build_eval_prompt,
    content_span_from_cfg,
    intervention_position_list,
    intervention_token_id_from_cfg,
    validate_intervention_position,
)
from boreft.eval.eval_suite import (
    DEFAULT_BBOX_PCA_VAR,
    DEFAULT_EMBED_SIM_TAU,
    DEFAULT_TEST_N_SAMPLES,
    EVAL_TEMPERATURES,
    build_test_sets,
    dist_metrics,
    edit_dist_per_target,
    genz_metrics,
    map_geary_c,
    recon_metrics,
    recon_test_metrics,
    rdkit_sim_per_target,
    semantic_dispersion,
    sobol_geary_c,
    target_normalizer,
    task_csv_paths,
    temp_key,
    tfs_per_target,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────


def _load_model_and_items(
    output_dir: str,
    model_name: str,
    layer: int,
    low_rank_dim: int,
    cache_dir: str | None,
    use_word_bias: bool | None = None,
    variance=None,
    torch_dtype: str | None = None,
    *,
    prefer_materialized_bias: bool = True,
    load_latest: bool = False,
    skip_embed_cache: bool = False,
):
    """Load items.json, tokenizer, base model, and intervenable model."""
    from boreft.data_utils import resolve_weight_dir_and_run_config

    output_dir, saved_cfg = resolve_weight_dir_and_run_config(
        output_dir, load_latest=load_latest
    )
    items_path = os.path.join(output_dir, "items.json")
    if not os.path.exists(items_path):
        raise FileNotFoundError(f"No items.json in {output_dir}")
    with open(items_path, encoding="utf-8") as f:
        items = json.load(f)

    warn_if_embedding_provenance_stale(saved_cfg, context="eval")

    effective_use_word_bias = (
        use_word_bias
        if use_word_bias is not None
        else saved_cfg.get("use_word_bias", True)
    )
    effective_variance = (
        variance if variance is not None else saved_cfg.get("variance", "learnable")
    )
    bias_type = saved_cfg.get("bias_type", "vae")
    add_bias_network = bool(saved_cfg.get("add_bias_network", False))
    bias_network_residual = bool(saved_cfg.get("bias_network_residual", False))
    effective_layer = saved_cfg.get("layer", layer)

    lambda_mm = float(saved_cfg.get("lambda_mm", 0.0))
    lambda_rank = float(saved_cfg.get("lambda_rank", 0.0))
    mm_exclude_diagonal = bool(saved_cfg.get("mm_exclude_diagonal", True))

    task = saved_cfg.get("task", "semantle")
    words = [it.get("word", it["target"]) for it in items]

    tokenizer = load_checkpoint_tokenizer(output_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_chat_template = bool(saved_cfg.get("use_chat_template"))
    prompt = build_eval_prompt(
        tokenizer,
        task,
        use_chat_template=use_chat_template,
        chat_instruction=saved_cfg.get("chat_instruction"),
        intervention_inject=saved_cfg.get("intervention_inject", "none"),
        intervention_token=saved_cfg.get("intervention_token"),
        intervention_token_id=saved_cfg.get("intervention_token_id"),
        system_prompt=system_prompt_from_cfg(saved_cfg),
        mist_smiles_tags=bool(saved_cfg.get("mist_smiles_tags")),
    )

    dtype_name = infer_torch_dtype_name(
        override=torch_dtype,
        saved_cfg=saved_cfg,
    )
    if torch_dtype is not None:
        print(f"[eval] torch_dtype={dtype_name} (override={torch_dtype!r})", flush=True)
    elif saved_cfg.get("torch_dtype") in TORCH_DTYPE_NAMES:
        print(f"[eval] torch_dtype={dtype_name} (checkpoint)", flush=True)
    else:
        print(f"[eval] torch_dtype={dtype_name} (auto)", flush=True)
    dtype = resolve_torch_dtype(dtype_name)
    kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="auto" if DEVICE == "cuda" else None,
        **kwargs,
    )

    hidden_size = base_model.config.hidden_size

    bias_tables_path = (
        resolve_bias_tables_path(output_dir, saved_cfg=saved_cfg)
        if prefer_materialized_bias
        else None
    )

    # LLM-encoder bias network reads from a copied final block over definition
    # text, not the sentence-transformer embed_cache.
    bias_input_source = saved_cfg.get("bias_input_source", "embed_cache")
    encoder_mode = add_bias_network and bias_input_source == "llm_encoder"

    embed_cache_tensor = None
    need_embed_cache = (
        not skip_embed_cache
        and (
            lambda_mm > 0.0
            or lambda_rank > 0.0
            or (add_bias_network and not encoder_mode)
        )
    )
    if need_embed_cache:
        embed_cache_tensor, _, _ = resolve_or_build_embed_cache(
            words,
            explicit_path=resolve_embed_cache_path(output_dir, saved_cfg=saved_cfg),
            provenance=training_embed_cache_provenance_from_cfg(saved_cfg),
            definition_lookup=definition_lookup_for_cfg(saved_cfg),
        )

    bias_network_kwargs = dict(
        add_bias_network=add_bias_network,
        bias_network_residual=bias_network_residual,
        embed_cache=embed_cache_tensor,
    )
    if encoder_mode:
        bias_network_kwargs["bias_input_source"] = "llm_encoder"
        bias_network_kwargs["bias_network_input_dim"] = int(
            saved_cfg.get("bias_network_input_dim", hidden_size)
        )
    elif skip_embed_cache and add_bias_network and embed_cache_tensor is None:
        dim = bias_network_input_dim_from_cfg(saved_cfg)
        if dim is None:
            dim = bias_network_input_dim_from_cfg(
                saved_cfg,
                embed_cache_path=resolve_embed_cache_path(
                    output_dir, saved_cfg=saved_cfg
                ),
            )
        if dim is None:
            raise ValueError(
                "skip_embed_cache requires bias_network_input_dim, "
                "bias_network_embed_dim, or an on-disk embed_cache when the "
                "bias network reads ST rows"
            )
        bias_network_kwargs["bias_network_input_dim"] = int(dim)
    mm_kwargs = dict(
        lambda_mm=lambda_mm,
        mm_exclude_diagonal=mm_exclude_diagonal,
        **bias_network_kwargs,
    )
    dropout = float(saved_cfg.get("dropout", 0.0))
    dropout_on_b = float(saved_cfg.get("dropout_on_b", 0.0))

    if bias_type == "linear":
        intervention = LoreftPerWordBiasIntervention(
            embed_dim=hidden_size,
            low_rank_dimension=low_rank_dim,
            num_words=len(words),
            dtype=dtype,
            device=DEVICE,
            act_fn="linear",
            dropout=dropout,
            dropout_on_b=dropout_on_b,
            **mm_kwargs,
        )
    else:
        intervention = DistributionalWordIntervention(
            embed_dim=hidden_size,
            low_rank_dimension=low_rank_dim,
            num_words=len(words),
            beta=0.1,
            dtype=dtype,
            device=DEVICE,
            use_word_bias=effective_use_word_bias,
            variance=effective_variance,
            dropout=dropout,
            dropout_on_b=dropout_on_b,
            **mm_kwargs,
        )

    # Build the encoder as a submodule *before* wrapping/loading so pyvene's
    # load_intervention populates its LoRA tensors. The frozen copied block is
    # reconstructed here from the (clean, not-yet-hooked) base model.
    if encoder_mode:
        from boreft.pyreft.semantic_encoder import (
            DEFAULT_LORA_TARGETS,
            build_semantic_encoder,
        )

        enc_targets = saved_cfg.get("bias_encoder_lora_targets")
        encoder = build_semantic_encoder(
            base_model,
            lora_rank=int(saved_cfg.get("bias_encoder_lora_rank", 8)),
            lora_alpha=float(saved_cfg.get("bias_encoder_lora_alpha", 16.0)),
            lora_dropout=float(saved_cfg.get("bias_encoder_lora_dropout", 0.0)),
            lora_targets=tuple(enc_targets) if enc_targets else DEFAULT_LORA_TARGETS,
            pooling=saved_cfg.get("bias_encoder_pooling", "last_instruction"),
            layer_index=saved_cfg.get("bias_encoder_layer_index"),
        )
        intervention.set_semantic_encoder(encoder)

    reft_config = ReftConfig(
        representations=[
            build_reft_representation(effective_layer, low_rank_dim, intervention)
        ]
    )
    reft_model = get_reft_model(base_model, reft_config, set_device=True)

    intervenable_path = os.path.join(output_dir, "intervenable_model")
    if not os.path.exists(intervenable_path):
        raise FileNotFoundError(
            f"No intervenable_model/ in {output_dir}. Run training first."
        )
    reft_model.load_intervention(intervenable_path, include_model=False)
    reft_model.eval()

    if encoder_mode:
        print("[eval] Attached llm_encoder bias network (LoRA loaded)", flush=True)

    if bias_tables_path is not None:
        mu_tables, logvar_tables, _ = load_bias_tables(
            bias_tables_path, num_words=len(words)
        )
        attach_materialized_bias_tables(
            _get_intervention(reft_model), mu_tables, logvar_tables
        )
        print(
            f"[eval] Using materialized bias tables from {bias_tables_path}",
            flush=True,
        )
    elif add_bias_network:
        print("[eval] No bias_tables.pt — running bias network at lookup time", flush=True)

    return reft_model, tokenizer, words, prompt, items


@dataclass
class EvalCheckpoint:
    """In-memory checkpoint bundle shared across full-eval steps."""

    reft_model: Any
    tokenizer: Any
    words: List[str]
    prompt: str
    items: List[dict]
    assistant_suffix: Optional[str] = None
    from_chat_template: bool = False
    saved_cfg: Optional[dict] = None
    intervention_token_id: Optional[int] = None
    content_span: Optional[tuple[int, int]] = None
    # Set when ``subset_eval_checkpoint`` narrows the eval vocabulary; full vocab
    # is retained for Sobol nearest-μ lookup and unseen-rate denominators.
    full_words: Optional[List[str]] = None
    full_items: Optional[List[dict]] = None


# Shared generation cap for Semantle eval (greedy, sampling, Sobol, interpolation).
DEFAULT_MAX_NEW_TOKENS = 128

# Default seed for the Sobol scramble so GENZ b-vector draws are reproducible.
DEFAULT_SOBOL_SEED = 42


@dataclass
class SemantleGenerationEvalArgs:
    """Typed args for :func:`run_semantle_generation_eval` (CLI + orchestrator)."""

    output_dir: str
    model_name: str = "meta-llama/Llama-3.2-1B"
    layer: int = 13
    low_rank_dim: int = 64
    cache_dir: Optional[str] = None
    n_samples: int = 25
    eval_batch_size: int = 32
    top_p: float = 1.0
    use_word_bias: Optional[bool] = None
    variance: Union[str, float, None] = None
    torch_dtype: Optional[str] = None
    seed: int = 42
    full_eval_n_samples: Optional[int] = None
    n_uniform: Optional[int] = None
    position: Optional[str] = None
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    embed_sim_tau: float = DEFAULT_EMBED_SIM_TAU
    test_n_samples: int = DEFAULT_TEST_N_SAMPLES
    bbox_pca_var: float = DEFAULT_BBOX_PCA_VAR
    load_latest: bool = False


def load_eval_checkpoint(
    output_dir: str,
    model_name: str,
    layer: int,
    low_rank_dim: int,
    cache_dir: Optional[str] = None,
    *,
    use_word_bias: Optional[bool] = None,
    variance=None,
    torch_dtype: Optional[str] = None,
    load_latest: bool = False,
    skip_embed_cache: bool = False,
) -> EvalCheckpoint:
    """Load model + items once for reuse across multiple eval steps.

    When ``load_latest`` is true and ``output_dir/latest/`` has weights, intervenable
    params are loaded from that subdirectory. Run-level ``training_config.json`` is
    always overlaid from the run root so metadata is not lost.

    ``skip_embed_cache`` skips sentence-transformer encode/rebuild. Use it when
    only decoding from materialized μ tables (Qwen3-Embedding needs
    ``transformers>=4.51``).
    """
    from boreft.data_utils import resolve_weight_dir_and_run_config

    reft_model, tokenizer, words, prompt, items = _load_model_and_items(
        output_dir,
        model_name,
        layer,
        low_rank_dim,
        cache_dir,
        use_word_bias=use_word_bias,
        variance=variance,
        torch_dtype=torch_dtype,
        load_latest=load_latest,
        skip_embed_cache=skip_embed_cache,
    )
    _, saved_cfg = resolve_weight_dir_and_run_config(
        output_dir, load_latest=load_latest
    )
    assistant_suffix = resolve_assistant_suffix(
        tokenizer,
        saved_cfg,
        task_instruction(saved_cfg.get("task", "semantle"), use_chat_template=True),
    )
    return EvalCheckpoint(
        reft_model=reft_model,
        tokenizer=tokenizer,
        words=words,
        prompt=prompt,
        items=items,
        assistant_suffix=assistant_suffix,
        from_chat_template=prompt_tokenization_from_cfg(saved_cfg),
        saved_cfg=saved_cfg,
        intervention_token_id=intervention_token_id_from_cfg(saved_cfg),
        content_span=content_span_from_cfg(tokenizer, saved_cfg, model_name),
    )


def subset_eval_checkpoint(
    checkpoint: EvalCheckpoint,
    n: Optional[int],
    seed: int,
) -> EvalCheckpoint:
    """Narrow ``words``/``items`` to a fixed subset for all full-eval steps."""
    if n is None or n >= len(checkpoint.items):
        return checkpoint
    checkpoint.full_items = list(checkpoint.items)
    checkpoint.full_words = list(checkpoint.words)
    sampled = random.Random(seed).sample(checkpoint.items, n)
    checkpoint.items = sampled
    checkpoint.words = [it.get("word", it["target"]) for it in sampled]
    print(
        f"[full_eval] Eval word subset: {len(checkpoint.words)}/{len(checkpoint.full_words)} "
        f"words (seed={seed})",
        flush=True,
    )
    return checkpoint


def release_eval_checkpoint(checkpoint: Optional[EvalCheckpoint]) -> None:
    """Explicitly drop a loaded eval checkpoint and free GPU memory."""
    if checkpoint is None:
        return
    checkpoint.reft_model = None
    checkpoint.tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Intervention helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_intervention(reft_model):
    iv = list(reft_model.interventions.values())[0]
    return iv[0] if isinstance(iv, (list, tuple)) else iv


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────


def generate_text(
    reft_model,
    tokenizer,
    prompt: str,
    word_idx,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    use_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 0,
    use_stochastic_intervention: bool = False,
    position: str = "l1",
    assistant_suffix: str | None = None,
    from_chat_template: bool = False,
    intervention_token_id: int | None = None,
    content_span: tuple[int, int] | None = None,
) -> str:
    """Generate text for a given word index or raw bias vector.

    word_idx : int  → looks up μ_w from the embedding table.
               ndarray/list → treats as a raw low-rank bias vector b(t).
    use_stochastic_intervention : put the VAE intervention in train mode so it
               samples b ~ N(μ,σ²) rather than using μ deterministically.
    position : intervention position string (e.g. 'l1', 'f1', 'f2+l2').
               Must match the position used during training.
    from_chat_template : when True, do not prepend BOS again (chat strings are
               pre-rendered). Set from checkpoint ``use_chat_template`` config.
    """
    iv = _get_intervention(reft_model)
    iv.train() if use_stochastic_intervention else iv.eval()

    enc = tokenize_model_text(
        tokenizer, prompt, from_chat_template=from_chat_template, return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(reft_model.get_device())
    attn_mask = enc["attention_mask"].to(reft_model.get_device())
    pos_list = intervention_position_list(
        position,
        input_ids[0].tolist(),
        prompt_len=input_ids.shape[1],
        intervention_token_id=intervention_token_id,
        content_span=content_span,
    )

    if isinstance(word_idx, (int, np.integer)):
        subspaces = [[[int(word_idx)]]]
    else:
        subspaces = [[[np.asarray(word_idx, dtype=np.float32).tolist()]]]

    gen_args: dict = {
        "base": {"input_ids": input_ids, "attention_mask": attn_mask},
        "unit_locations": {"sources->base": (None, [[[p for p in pos_list]]])},
        "intervene_on_prompt": True,
        "subspaces": subspaces,
        "max_new_tokens": max_new_tokens,
        "do_sample": use_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if use_sample:
        gen_args["temperature"] = temperature
        gen_args["top_p"] = top_p
        if top_k > 0:
            gen_args["top_k"] = top_k

    with torch.no_grad():
        _, output_ids = reft_model.generate(**gen_args)

    iv.eval()  # always restore eval mode

    return decode_generated_text(
        tokenizer,
        output_ids,
        input_ids.shape[1],
        assistant_suffix=assistant_suffix,
    )


def generate_texts_batch(
    reft_model,
    tokenizer,
    prompt: str,
    word_idx,
    n_samples: int,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    use_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 0,
    use_stochastic_intervention: bool = False,
    position: str = "l1",
    assistant_suffix: str | None = None,
    from_chat_template: bool = False,
    intervention_token_id: int | None = None,
    content_span: tuple[int, int] | None = None,
) -> list[str]:
    """Generate n_samples outputs for a single word_idx in one batched forward pass.

    Repeats the same prompt × n_samples with the same subspace (word_id or b vector),
    using the identical subspaces format as the single-item generate_text() call so
    pyvene sees exactly the same intervention structure.

    Returns a list of n_samples decoded strings.
    """
    iv = _get_intervention(reft_model)
    iv.train() if use_stochastic_intervention else iv.eval()

    enc = tokenize_model_text(
        tokenizer, prompt, from_chat_template=from_chat_template, return_tensors="pt"
    )
    prompt_len = enc["input_ids"].shape[1]
    input_ids = (
        enc["input_ids"].to(reft_model.get_device()).repeat(n_samples, 1)
    )  # [n_samples, seq]
    attn_mask = enc["attention_mask"].to(reft_model.get_device()).repeat(n_samples, 1)
    pos_list = intervention_position_list(
        position,
        input_ids[0].tolist(),
        prompt_len=prompt_len,
        intervention_token_id=intervention_token_id,
        content_span=content_span,
    )

    # pyvene subspaces format: [num_interventions][batch_size][value]
    # One intervention, n_samples batch items — NOT n_samples copies of a 1-item intervention.
    if isinstance(word_idx, (int, np.integer)):
        subspaces = [[[int(word_idx)]] * n_samples]
    else:
        subspaces = [[[np.asarray(word_idx, dtype=np.float32).tolist()]] * n_samples]

    unit_locations = {"sources->base": (None, [[pos_list]] * n_samples)}

    gen_args: dict = {
        "base": {"input_ids": input_ids, "attention_mask": attn_mask},
        "unit_locations": unit_locations,
        "intervene_on_prompt": True,
        "subspaces": subspaces,
        "max_new_tokens": max_new_tokens,
        "do_sample": use_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if use_sample:
        gen_args["temperature"] = temperature
        gen_args["top_p"] = top_p
        if top_k > 0:
            gen_args["top_k"] = top_k

    with torch.no_grad():
        _, output_ids = reft_model.generate(**gen_args)

    iv.eval()

    results = []
    for i in range(n_samples):
        results.append(
            decode_generated_text(
                tokenizer,
                output_ids,
                prompt_len,
                assistant_suffix=assistant_suffix,
                batch_idx=i,
            )
        )
    return results


def generate_texts_multi_batch(
    reft_model,
    tokenizer,
    prompt: str,
    word_indices: list,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    use_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 0,
    use_stochastic_intervention: bool = False,
    position: str = "l1",
    assistant_suffix: str | None = None,
    from_chat_template: bool = False,
    intervention_token_id: int | None = None,
    content_span: tuple[int, int] | None = None,
) -> list[str]:
    """Greedy or sampled decode for a batch of distinct word ids / bias vectors.

    ``word_indices[i]`` is an int word id or a raw low-rank bias vector, parallel
    to the batch dimension of a single ``reft_model.generate`` call.
    """
    n = len(word_indices)
    if n == 0:
        return []

    iv = _get_intervention(reft_model)
    iv.train() if use_stochastic_intervention else iv.eval()

    enc = tokenize_model_text(
        tokenizer, prompt, from_chat_template=from_chat_template, return_tensors="pt"
    )
    prompt_len = enc["input_ids"].shape[1]
    input_ids = enc["input_ids"].to(reft_model.get_device()).repeat(n, 1)
    attn_mask = enc["attention_mask"].to(reft_model.get_device()).repeat(n, 1)
    pos_list = intervention_position_list(
        position,
        input_ids[0].tolist(),
        prompt_len=prompt_len,
        intervention_token_id=intervention_token_id,
        content_span=content_span,
    )

    subspace_row: list = []
    for wi in word_indices:
        if isinstance(wi, (int, np.integer)):
            subspace_row.append([int(wi)])
        else:
            subspace_row.append(np.asarray(wi, dtype=np.float32).tolist())
    subspaces = [subspace_row]

    unit_locations = {"sources->base": (None, [[pos_list]] * n)}

    gen_args: dict = {
        "base": {"input_ids": input_ids, "attention_mask": attn_mask},
        "unit_locations": unit_locations,
        "intervene_on_prompt": True,
        "subspaces": subspaces,
        "max_new_tokens": max_new_tokens,
        "do_sample": use_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if use_sample:
        gen_args["temperature"] = temperature
        gen_args["top_p"] = top_p
        if top_k > 0:
            gen_args["top_k"] = top_k

    with torch.no_grad():
        _, output_ids = reft_model.generate(**gen_args)

    iv.eval()

    return [
        decode_generated_text(
            tokenizer,
            output_ids,
            prompt_len,
            assistant_suffix=assistant_suffix,
            batch_idx=i,
        )
        for i in range(n)
    ]


def _run_greedy_decode_experiment(
    reft_model,
    tokenizer,
    words: list,
    prompt: str,
    *,
    word_ids: list[int] | None = None,
    bias_vectors: list | None = None,
    batch_size: int = 32,
    position: str = "l1",
    assistant_suffix: str | None = None,
    from_chat_template: bool = False,
    intervention_token_id: int | None = None,
    content_span: tuple[int, int] | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> list[str]:
    """Greedy-decode one string per word using multi-word batched forwards."""
    if bias_vectors is not None:
        if len(bias_vectors) != len(words):
            raise ValueError("bias_vectors must be parallel to words")
        subspaces_src = bias_vectors
    elif word_ids is not None:
        if len(word_ids) != len(words):
            raise ValueError("word_ids must be parallel to words")
        subspaces_src = word_ids
    else:
        subspaces_src = list(range(len(words)))

    w = len(words)
    n_batches = (w + batch_size - 1) // batch_size
    print(
        f"  batched greedy decode: {w} words  (batch_size={batch_size}, "
        f"{n_batches} forwards)",
        flush=True,
    )

    decodes: list[str] = []
    for start in range(0, w, batch_size):
        end = min(start + batch_size, w)
        batch_no = start // batch_size + 1
        print(f"  batch {batch_no}/{n_batches}: words {start + 1}-{end}/{w}", flush=True)
        decodes.extend(
            generate_texts_multi_batch(
                reft_model,
                tokenizer,
                prompt,
                subspaces_src[start:end],
                max_new_tokens=max_new_tokens,
                use_sample=False,
                position=position,
                assistant_suffix=assistant_suffix,
                from_chat_template=from_chat_template,
                intervention_token_id=intervention_token_id,
                content_span=content_span,
            )
        )
    return decodes


# ─────────────────────────────────────────────────────────────────────────────
# Sampling experiment engine
# ─────────────────────────────────────────────────────────────────────────────


def _run_sampling_experiment(
    reft_model,
    tokenizer,
    words: list,
    prompt: str,
    n_samples: int,
    word_ids: list[int] | None = None,
    bias_vectors: list | None = None,
    train_vocab: set | None = None,
    use_stochastic_intervention: bool = False,
    use_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 0,
    batch_size: int = 32,
    position: str = "l1",
    assistant_suffix: str | None = None,
    from_chat_template: bool = False,
    intervention_token_id: int | None = None,
    content_span: tuple[int, int] | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    task: str = "semantle",
) -> list[dict]:
    """Run n_samples generations per word using batched inference.

    All words × n_samples combinations are chunked into batches of `batch_size`
    and processed in a single generate() call each — much faster than one call
    per (word, run) pair.

    Similarity is computed over unique predictions only.

    Pass ``word_ids`` for per-word embedding lookup, or ``bias_vectors`` (one
    low-rank vector per word) for raw-bias decoding (e.g. RECON_TEST).
    """
    if bias_vectors is not None and len(bias_vectors) != len(words):
        raise ValueError("bias_vectors must be parallel to words")
    if bias_vectors is None:
        ids = word_ids if word_ids is not None else list(range(len(words)))
    W = len(words)

    print(
        f"  batched generation: {W} words × {n_samples} samples  (batch_size={batch_size})",
        flush=True,
    )

    # Generate n_samples outputs per word using one batched call per word, then
    # batch-encode each distinct target once and all unique predictions once.
    per_target_samples: list[list[str]] = []
    per_target_unique: list[list[str]] = []
    all_unique_gens: list[str] = []

    for wi, word in enumerate(words):
        subspace = bias_vectors[wi] if bias_vectors is not None else ids[wi]
        print(f"  word {wi + 1}/{W}: {word!r}...", flush=True)
        runs: list[str] = []
        for start in range(0, n_samples, batch_size):
            chunk = min(batch_size, n_samples - start)
            runs.extend(
                generate_texts_batch(
                    reft_model,
                    tokenizer,
                    prompt,
                    subspace,
                    n_samples=chunk,
                    use_sample=use_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    use_stochastic_intervention=use_stochastic_intervention,
                    position=position,
                    assistant_suffix=assistant_suffix,
                    from_chat_template=from_chat_template,
                    intervention_token_id=intervention_token_id,
                    content_span=content_span,
                    max_new_tokens=max_new_tokens,
                )
            )
        unique_samples = list(dict.fromkeys(runs))  # stable dedup
        per_target_samples.append(runs)
        per_target_unique.append(unique_samples)
        all_unique_gens.extend(unique_samples)

    # Encode each distinct target once (indexed by raw text), plus all unique gens.
    distinct_targets = list(dict.fromkeys(words))
    target_emb = dict(
        zip(distinct_targets, encode_texts_normalized(distinct_targets, task=task))
    )
    emb_gen_all = (
        encode_texts_normalized(all_unique_gens, task=task)
        if all_unique_gens
        else np.zeros((0, 0), dtype=np.float64)
    )

    results = []
    ptr = 0
    for target, samples, unique_samples in zip(
        words, per_target_samples, per_target_unique
    ):
        n_u = len(unique_samples)
        block = emb_gen_all[ptr : ptr + n_u]
        ptr += n_u
        per_sample_sims = (
            (block @ target_emb[target]).tolist() if n_u else []
        )
        n_unseen = sum(
            1 for r in samples if train_vocab is not None and r not in train_vocab
        )
        results.append(
            {
                "target": target,
                "samples": samples,
                "unique_samples": unique_samples,
                "n_unique": n_u,
                "n_unseen": n_unseen,
                "unseen_rate": n_unseen,
                "per_sample_sims": per_sample_sims,
                "mean_sim": float(np.mean(per_sample_sims)) if per_sample_sims else 0.0,
            }
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Sobol space sampling (shared utilities)
# ─────────────────────────────────────────────────────────────────────────────


def _nearest_mu_indices(mu_all: np.ndarray, sampled_bs: np.ndarray) -> list[int]:
    indices: list[int] = []
    for b in sampled_bs:
        dists = np.linalg.norm(mu_all - b[None, :], axis=1)
        indices.append(int(np.argmin(dists)))
    return indices


def _sample_sim_to_target_stats(
    sample_texts: list[str], nearest_train_target: str, *, task: str = "semantle"
) -> tuple[dict[str, float], list[str]]:
    """Embedding sim of each sample vs target; return stats and samples sorted by sim desc."""
    if not sample_texts:
        return {"max": 0.0, "min": 0.0, "mean": 0.0, "std": 0.0}, []

    emb_tgt = encode_texts_normalized([nearest_train_target], task=task)
    emb_gen = encode_texts_normalized(sample_texts, task=task)
    sims = np.sum(emb_gen * emb_tgt, axis=1)

    order = np.argsort(-sims)
    sorted_samples = [sample_texts[int(i)] for i in order]
    sorted_sims = sims[order]

    return (
        {
            "max": float(sorted_sims[0]),
            "min": float(sorted_sims[-1]),
            "mean": float(sorted_sims.mean()),
            "std": float(sorted_sims.std()) if len(sorted_sims) > 1 else 0.0,
        },
        sorted_samples,
    )


def _build_sobol_per_sample_row(
    *,
    idx: int,
    b: np.ndarray,
    nearest_train_target: str,
    nearest_idx: int,
    mu_all: np.ndarray,
    mu_norms: np.ndarray,
    sample_texts: list[str],
    train_vocab: set | None,
    task: str = "semantle",
) -> dict:
    """One unified per-Sobol-point row for greedy and temperature sections."""
    b_norm = float(np.linalg.norm(b))
    denom = b_norm * float(mu_norms[nearest_idx]) + 1e-12
    bias_sim = float(np.dot(b, mu_all[nearest_idx]) / denom)

    sim_stats, sorted_samples = _sample_sim_to_target_stats(
        sample_texts, nearest_train_target, task=task
    )
    sample_mode = Counter(sample_texts).most_common(1)[0][0] if sample_texts else ""
    mode_sim_stats, _ = (
        _sample_sim_to_target_stats(sample_texts, sample_mode, task=task)
        if sample_mode
        else ({"max": 0.0, "min": 0.0, "mean": 0.0, "std": 0.0}, [])
    )
    unique_texts = set(sample_texts)
    n_unique = len(unique_texts)
    n_unseen = (
        sum(1 for t in unique_texts if t not in train_vocab)
        if train_vocab is not None
        else None
    )

    return {
        "idx": idx,
        "b_norm": b_norm,
        "nearest_train_target": nearest_train_target,
        "n_unique": n_unique,
        "n_unseen": n_unseen,
        "bias_sim_to_nearest_train_target": bias_sim,
        "sample_sim_to_nearest_train_target": sim_stats,
        "sample_mode": sample_mode,
        "sample_sim_to_mode": mode_sim_stats,
        "samples": sorted_samples,
    }


def _sobol_per_sample_legend() -> dict[str, str]:
    return {
        "nearest_train_target": "training target whose learned μ is closest to b (Euclidean)",
        "n_unique": "number of distinct generated texts for this Sobol point",
        "n_unseen": "number of distinct generated texts not in the training vocabulary",
        "bias_sim_to_nearest_train_target": "cosine similarity between b and nearest μ",
        "sample_sim_to_nearest_train_target": "embed sim of each decode vs nearest_train_target "
        "(max/min/mean/std); samples sorted by sim descending",
        "sample_mode": "most frequently decoded text for this Sobol point",
        "sample_sim_to_mode": "embed sim of each decode vs sample_mode "
        "(max/min/mean/std)",
        "samples": "decode outputs for this Sobol point, highest embed sim first",
    }


def sample_sobol_bias_vectors(
    reft_model,
    words: list,
    n_sobol_points: int,
    word_ids: list[int] | None = None,
    mu_all: np.ndarray | None = None,
    seed: int | None = DEFAULT_SOBOL_SEED,
) -> tuple[np.ndarray, list[str], list[int], np.ndarray]:
    """Draw Sobol b-vectors over the μ bounding box.

    Returns ``(sampled_bs, nearest_words, nearest_indices, mu_all)``.
    Pass ``mu_all`` to avoid reloading bias vectors when sampling repeatedly.

    ``seed`` seeds the Sobol scramble so the draw is reproducible across runs;
    pass ``None`` for a nondeterministic scramble.

    **words / word_ids alignment:** ``stack_bias_vectors`` builds ``mu_all`` with
    row ``j`` = bias for ``word_ids[j]``. ``nearest_indices`` are row indices into
    that matrix, so ``words`` must be parallel: ``words[j]`` is the label for
    ``word_ids[j]`` (not necessarily ``words[word_ids[j]]`` unless
    ``word_ids == list(range(len(words)))``).
    """
    ids = word_ids if word_ids is not None else list(range(len(words)))
    if mu_all is None:
        mu_all = stack_bias_vectors(reft_model, [int(w) for w in ids])

    lo = mu_all.min(axis=0)
    hi = mu_all.max(axis=0)
    rank = mu_all.shape[1]

    from torch.quasirandom import SobolEngine

    engine = SobolEngine(dimension=rank, scramble=True, seed=seed)
    samples_01 = engine.draw(n_sobol_points).numpy()  # [N, rank] in [0, 1]
    sampled_bs = lo + samples_01 * (hi - lo)  # [N, rank] scaled

    print(
        f"\n[Sobol] Sampled {n_sobol_points} bias vectors  rank={rank}  "
        f"bbox=[{lo.min():.3f}, {hi.max():.3f}]",
        flush=True,
    )

    nearest_indices = _nearest_mu_indices(mu_all, sampled_bs)
    nearest_words = [words[i] for i in nearest_indices]

    return sampled_bs, nearest_words, nearest_indices, mu_all


def _sobol_sample_centroids(
    sample_bags: list[list[str]], *, task: str = "semantle"
) -> np.ndarray:
    """L2-normalized decode-embedding centroid per Sobol point.

    ``sample_bags[i]`` is the raw temperature sample bag (duplicates kept) for
    point ``i``. All decodes are encoded in one batched call, then averaged per
    point (frequency-weighted via duplicates) and renormalized so that cosine
    distance identities hold. Empty bags yield a zero row.
    """
    flat: list[str] = []
    spans: list[tuple[int, int]] = []
    for bag in sample_bags:
        start = len(flat)
        flat.extend(bag)
        spans.append((start, len(flat)))
    if not flat:
        return np.zeros((0, 0), dtype=np.float64)

    emb = encode_texts_normalized(flat, task=task).astype(np.float64)  # [sum_S, D]
    cents = np.zeros((len(sample_bags), emb.shape[1]), dtype=np.float64)
    for i, (a, z) in enumerate(spans):
        if z > a:
            c = emb[a:z].mean(axis=0)
            cents[i] = c / (np.linalg.norm(c) + 1e-12)
    return cents


# ─────────────────────────────────────────────────────────────────────────────
# Case 4 — Sobol space sampling (greedy)
# ─────────────────────────────────────────────────────────────────────────────


def run_uniform_space_sampling(
    reft_model,
    tokenizer,
    words: list,
    prompt: str,
    n_sobol_points: int = 200,
    word_ids: list[int] | None = None,
    train_vocab: set | None = None,
    save_path: str | None = None,
    position: str = "l1",
    assistant_suffix: str | None = None,
    from_chat_template: bool = False,
    intervention_token_id: int | None = None,
    content_span: tuple[int, int] | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    sampled_bs: np.ndarray | None = None,
    nearest_words: list[str] | None = None,
    nearest_indices: list[int] | None = None,
    mu_all: np.ndarray | None = None,
    seed: int | None = DEFAULT_SOBOL_SEED,
    task: str = "semantle",
) -> dict:
    """Case 4: draw b via Sobol quasi-random sampling over the per-dim bounding box of all μ vectors.

    Sobol sequences give deterministic, evenly-spaced coverage of the bbox —
    far better than independent uniform draws in high dimensions (rank=64).

    For each sampled b:
      - generate text greedily (b used as raw bias vector)
      - find the nearest learned μ_w by Euclidean distance
      - record embedding similarity between generated text and that nearest target

    Pass ``sampled_bs``, ``nearest_words``, ``nearest_indices``, and ``mu_all`` to
    reuse the same Sobol draw as temperature decoding variants.

    Returns a dict with aggregate stats and per-sample details.
    """
    ids = word_ids if word_ids is not None else list(range(len(words)))
    if mu_all is None:
        mu_all = stack_bias_vectors(reft_model, [int(w) for w in ids])
    mu_norms = np.linalg.norm(mu_all, axis=1)  # [W]

    if sampled_bs is None:
        sampled_bs, nearest_words, nearest_indices, mu_all = sample_sobol_bias_vectors(
            reft_model, words, n_sobol_points, word_ids=word_ids, mu_all=mu_all, seed=seed
        )
    else:
        n_sobol_points = len(sampled_bs)
        if nearest_indices is None or nearest_words is None:
            nearest_indices = _nearest_mu_indices(mu_all, sampled_bs)
            nearest_words = [words[i] for i in nearest_indices]
        print(
            f"\n[Case 4] Sobol space sampling (greedy)  n_sobol_points={n_sobol_points}  "
            f"(shared Sobol points)",
            flush=True,
        )

    generated_texts: list[str] = []

    for i, b in enumerate(sampled_bs):
        gen = generate_text(
            reft_model,
            tokenizer,
            prompt,
            b.tolist(),
            max_new_tokens=max_new_tokens,
            position=position,
            assistant_suffix=assistant_suffix,
            from_chat_template=from_chat_template,
            intervention_token_id=intervention_token_id,
            content_span=content_span,
        )
        generated_texts.append(gen)

    per_sample = [
        _build_sobol_per_sample_row(
            idx=i,
            b=b,
            nearest_train_target=nearest_words[i],
            nearest_idx=nearest_indices[i],
            mu_all=mu_all,
            mu_norms=mu_norms,
            sample_texts=[generated_texts[i]],
            train_vocab=train_vocab,
            task=task,
        )
        for i, b in enumerate(sampled_bs)
    ]

    n_unique = len(set(generated_texts))
    unique_frac = n_unique / n_sobol_points
    n_unseen = sum(p["n_unseen"] or 0 for p in per_sample)
    unseen_rate = n_unseen / n_sobol_points if train_vocab is not None else None
    avg_word_sim = float(
        np.mean([p["sample_sim_to_nearest_train_target"]["mean"] for p in per_sample])
    )
    avg_bias_sim = float(
        np.mean([p["bias_sim_to_nearest_train_target"] for p in per_sample])
    )

    legend = _sobol_per_sample_legend()

    result = {
        "n_sobol_points": n_sobol_points,
        "legend": legend,
        "avg_word_sim": avg_word_sim,
        "avg_bias_sim": avg_bias_sim,
        "n_unique": n_unique,
        "unique_frac": unique_frac,
        "n_unseen": n_unseen,
        "unseen_rate": unseen_rate,
        "per_sample": per_sample,
    }

    print(
        f"  avg word_sim (gen vs nearest word) : {avg_word_sim:.4f}\n"
        f"  avg bias_sim (b vs nearest mu)     : {avg_bias_sim:.4f}\n"
        f"  unique words : {n_unique}/{n_sobol_points}  ({unique_frac:.1%})"
        + (
            f"\n  unseen rate  : {unseen_rate:.1%}  (not in training vocabulary)"
            if unseen_rate is not None
            else ""
        ),
        flush=True,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as _f:
            json.dump(result, _f, indent=2)
        print(f"  sobol results saved → {save_path}", flush=True)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Case 5 — Sobol space sampling + temperature (stochastic token) decoding
# ─────────────────────────────────────────────────────────────────────────────


def run_sobol_temperature_sampling(
    reft_model,
    tokenizer,
    words: list,
    prompt: str,
    n_sobol_points: int = 200,
    n_samples: int = 25,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    word_ids: list[int] | None = None,
    train_vocab: set | None = None,
    batch_size: int = 25,
    position: str = "l1",
    assistant_suffix: str | None = None,
    from_chat_template: bool = False,
    intervention_token_id: int | None = None,
    content_span: tuple[int, int] | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    sampled_bs: np.ndarray | None = None,
    nearest_words: list[str] | None = None,
    nearest_indices: list[int] | None = None,
    mu_all: np.ndarray | None = None,
    seed: int | None = DEFAULT_SOBOL_SEED,
    task: str = "semantle",
) -> dict:
    """Case 5: Sobol b-vectors + temperature (stochastic token) decoding.

    For each Sobol-sampled b-vector, run n_samples stochastic generations in one
    batched GPU call (generate_texts_batch with use_sample=True).  Reports
    diversity metrics (unique, unseen) analogous to the combined-sampling case,
    but probing the full b-vector bounding box rather than per-word μ.

    The b-vector is injected as a raw ndarray — same path as the greedy Sobol
    case; no word-id lookup is performed.

    Pass ``sampled_bs``, ``nearest_words``, ``nearest_indices``, and ``mu_all`` to
    reuse the same Sobol draw across multiple temperature settings (paired comparison).
    """
    if sampled_bs is not None:
        n_sobol_points = len(sampled_bs)
        if nearest_words is None or nearest_indices is None:
            ids = word_ids if word_ids is not None else list(range(len(words)))
            if mu_all is None:
                mu_all = stack_bias_vectors(reft_model, [int(w) for w in ids])
            nearest_indices = _nearest_mu_indices(mu_all, sampled_bs)
            nearest_words = [words[i] for i in nearest_indices]
        print(
            f"\n[Case 5] Sobol+Temperature sampling  n_sobol_points={n_sobol_points}  "
            f"n_samples={n_samples}  T={temperature}  (shared Sobol points)",
            flush=True,
        )
    else:
        sampled_bs, nearest_words, nearest_indices, mu_all = sample_sobol_bias_vectors(
            reft_model, words, n_sobol_points, word_ids=word_ids, mu_all=mu_all, seed=seed
        )
        print(
            f"\n[Case 5] Sobol+Temperature sampling  n_sobol_points={n_sobol_points}  "
            f"n_samples={n_samples}  T={temperature}",
            flush=True,
        )

    ids = word_ids if word_ids is not None else list(range(len(words)))
    if mu_all is None:
        mu_all = stack_bias_vectors(reft_model, [int(w) for w in ids])
    mu_norms = np.linalg.norm(mu_all, axis=1)

    all_generated: list[list[str]] = []  # [n_sobol_points][n_samples]

    for i, b in enumerate(sampled_bs):
        # Batched temperature generation — n_samples outputs per b-vector
        runs: list[str] = []
        for start in range(0, n_samples, batch_size):
            chunk = min(batch_size, n_samples - start)
            runs.extend(
                generate_texts_batch(
                    reft_model,
                    tokenizer,
                    prompt,
                    b,
                    n_samples=chunk,
                    use_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    use_stochastic_intervention=False,
                    position=position,
                    assistant_suffix=assistant_suffix,
                    from_chat_template=from_chat_template,
                    intervention_token_id=intervention_token_id,
                    content_span=content_span,
                    max_new_tokens=max_new_tokens,
                )
            )
        all_generated.append(runs)

        if (i + 1) % max(1, n_sobol_points // 10) == 0:
            print(f"  {i + 1}/{n_sobol_points} Sobol points done", flush=True)

    # Flatten all generated texts for aggregate stats
    flat_generated = [w for runs in all_generated for w in runs]
    total = len(flat_generated)

    unique_generated = set(flat_generated)
    n_unique = len(unique_generated)
    unique_frac = n_unique / total

    n_unseen = (
        sum(1 for g in unique_generated if g not in train_vocab)
        if train_vocab is not None
        else None
    )
    unseen_rate = n_unseen / n_unique if n_unseen is not None else None

    per_sample = [
        _build_sobol_per_sample_row(
            idx=i,
            b=b,
            nearest_train_target=nearest_words[i],
            nearest_idx=nearest_indices[i],
            mu_all=mu_all,
            mu_norms=mu_norms,
            sample_texts=runs,
            train_vocab=train_vocab,
            task=task,
        )
        for i, (b, runs) in enumerate(zip(sampled_bs, all_generated))
    ]

    print(
        f"  total words generated: {total}  "
        f"unique: {n_unique} ({unique_frac:.1%})"
        + (
            f"  unseen: {n_unseen} ({unseen_rate:.1%})"
            if unseen_rate is not None
            else ""
        ),
        flush=True,
    )

    return {
        "n_sobol_points": n_sobol_points,
        "n_samples": n_samples,
        "temperature": temperature,
        "n_unique": n_unique,
        "unique_frac": unique_frac,
        "n_unseen": n_unseen,
        "unseen_rate": unseen_rate,
        "avg_unique_per_point": float(np.mean([p["n_unique"] for p in per_sample])),
        "avg_unseen_per_point": (
            float(
                np.mean(
                    [p["n_unseen"] for p in per_sample if p["n_unseen"] is not None]
                )
            )
            if train_vocab is not None
            else None
        ),
        "per_sample": per_sample,
    }


def _temperature_sobol_legend() -> dict[str, str]:
    return _sobol_per_sample_legend()


SOBOL_SECTION_GREEDY = "greedy"
SOBOL_SECTION_TEMPERATURE_1_0 = "temperature_1.0"
SOBOL_SECTION_TEMPERATURE_1_5 = "temperature_1.5"

_SOBOL_STRUCTURED_KEYS = frozenset(
    {
        SOBOL_SECTION_GREEDY,
        "temperature",
        SOBOL_SECTION_TEMPERATURE_1_0,
        SOBOL_SECTION_TEMPERATURE_1_5,
    }
)


def _normalize_sobol_results(data: dict) -> dict:
    """Map legacy section keys to canonical names."""
    out = dict(data)
    if "temperature" in out and "temperature_1.0" not in out:
        out["temperature_1.0"] = out["temperature"]
    return out


def load_sobol_results_json(path: str) -> dict:
    """Load sobol_results.json, normalizing legacy flat greedy-only files."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if _SOBOL_STRUCTURED_KEYS.intersection(data):
        return _normalize_sobol_results(data)
    return {"greedy": data}


def _apply_temperature_sobol_section(
    payload: dict,
    legend: dict,
    existing: dict,
    key: str,
    section: dict | None,
) -> None:
    if section is not None:
        payload[key] = {**section, "mode": key, "do_sample": True}
        legend.update(_temperature_sobol_legend())
    elif existing.get(key):
        payload[key] = existing[key]
        legend.update(_temperature_sobol_legend())


def write_sobol_results_json(
    path: str,
    *,
    greedy: dict | None = None,
    temperature_1_0: dict | None = None,
    temperature_1_5: dict | None = None,
) -> None:
    """Write greedy + temperature Sobol sections to sobol_results.json.

    Sections use keys ``greedy``, ``temperature_1.0``, and ``temperature_1.5``.
    Legacy files with a ``temperature`` section are read via
    :func:`load_sobol_results_json` but rewritten under ``temperature_1.0``.
    """
    existing: dict = {}
    if os.path.exists(path):
        existing = load_sobol_results_json(path)

    payload: dict[str, Any] = {}
    legend: dict[str, str] = {}

    if greedy is not None:
        payload["greedy"] = {**greedy, "mode": "greedy", "do_sample": False}
        legend.update(greedy.get("legend", {}))
    elif existing.get("greedy"):
        payload["greedy"] = existing["greedy"]
        legend.update(existing.get("legend", {}))

    _apply_temperature_sobol_section(
        payload, legend, existing, SOBOL_SECTION_TEMPERATURE_1_0, temperature_1_0
    )
    _apply_temperature_sobol_section(
        payload, legend, existing, SOBOL_SECTION_TEMPERATURE_1_5, temperature_1_5
    )

    if legend:
        payload["legend"] = legend

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  sobol results saved → {path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description="Semantle evaluation: similarity, diversity, PCA."
    )
    p.add_argument("--output_dir", required=True)
    add_load_latest_argument(p)
    p.add_argument("--model-name", dest="model_name", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--layer", type=int, default=13)
    p.add_argument("--low_rank_dim", type=int, default=64)
    p.add_argument("--cache_dir", default=None)
    p.add_argument(
        "--n_samples", type=int, default=25, help="Samples per word for Cases 1 & 2."
    )
    p.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size for batched generation in Cases 1 & 2 (GPU memory vs speed).",
    )
    p.add_argument(
        "--top-p",
        dest="top_p",
        type=float,
        default=1.0,
        help="Nucleus top-p for all temperature-sampling passes (RECON/DIST/GENZ).",
    )
    p.add_argument(
        "--use-word-bias",
        dest="use_word_bias",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override use_word_bias from checkpoint (default: load from checkpoint).",
    )
    p.add_argument("--variance", default=None, help="'learnable' or a positive float.")
    p.add_argument(
        "--torch-dtype",
        dest="torch_dtype",
        choices=TORCH_DTYPE_NAMES,
        default=None,
        help="Override model/intervention dtype (default: checkpoint config, else auto).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (word subsampling, Sobol scramble, etc.).",
    )
    p.add_argument(
        "--full_eval_n_samples",
        type=int,
        default=None,
        help="Evaluate a random subset of training words.",
    )
    p.add_argument(
        "--n_uniform",
        type=int,
        default=None,
        help="Sobol samples for Cases 4–5 (greedy + temperature over shared b-vectors). "
        "0 or omitted = skip.",
    )
    p.add_argument(
        "--position",
        default=None,
        help="Intervention position string (e.g. l1, f1, f2+l2, marker, content_f1, content_l1). "
        "Defaults to checkpoint intervention_config.json.",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Max tokens to generate per decode (greedy and sampling evals).",
    )
    p.add_argument(
        "--embed-sim-tau",
        dest="embed_sim_tau",
        type=float,
        default=DEFAULT_EMBED_SIM_TAU,
        help="RECON threshold for recon/embed_sim_gte_tau.",
    )
    p.add_argument(
        "--test-n-samples",
        dest="test_n_samples",
        type=int,
        default=DEFAULT_TEST_N_SAMPLES,
        help="GENZ: number of non-train test words to sample.",
    )
    p.add_argument(
        "--bbox-pca-var",
        dest="bbox_pca_var",
        type=float,
        default=DEFAULT_BBOX_PCA_VAR,
        help="GENZ: PCA variance ratio for the train-embedding bounding box.",
    )
    args = p.parse_args()
    if args.position is not None:
        validate_intervention_position(args.position)
    eval_args = SemantleGenerationEvalArgs(
        output_dir=args.output_dir,
        model_name=args.model_name,
        layer=args.layer,
        low_rank_dim=args.low_rank_dim,
        cache_dir=args.cache_dir,
        n_samples=args.n_samples,
        eval_batch_size=args.eval_batch_size,
        top_p=args.top_p,
        use_word_bias=args.use_word_bias,
        variance=args.variance,
        torch_dtype=args.torch_dtype,
        seed=args.seed,
        full_eval_n_samples=args.full_eval_n_samples,
        n_uniform=args.n_uniform,
        position=args.position,
        max_new_tokens=args.max_new_tokens,
        embed_sim_tau=args.embed_sim_tau,
        test_n_samples=args.test_n_samples,
        bbox_pca_var=args.bbox_pca_var,
        load_latest=bool(args.load_latest),
    )
    return run_semantle_generation_eval(eval_args)


def run_semantle_generation_eval(
    args: Union[SemantleGenerationEvalArgs, argparse.Namespace],
    *,
    checkpoint: Optional[EvalCheckpoint] = None,
    release_model: bool = True,
) -> dict:
    """RECON / DIST (and GENZ when ``n_uniform`` is set) metrics for a checkpoint.

    RECON/DIST use train-target greedy + temperature decodes (b=μ, temps in
    ``EVAL_TEMPERATURES``). When ``n_uniform`` is set, GENZ adds Sobol greedy +
    temperature sampling (writing ``output_dir/eval/sobol_results.json``) and a
    non-train test set split into interp/extrap subsets for recall coverage.
    LIPZ (interpolation continuity) is computed separately by ``run_full_eval``.

    ``args`` may be a :class:`SemantleGenerationEvalArgs` or argparse ``Namespace``.
    Pass ``checkpoint`` to reuse a model already loaded by the orchestrator; when
    ``release_model`` is True and no shared checkpoint was passed, the locally
    loaded model is explicitly released before returning. Writes
    ``output_dir/eval/results.json`` and returns the results dict.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    variance_override = args.variance
    if variance_override is not None:
        try:
            variance_override = float(variance_override)
        except ValueError:
            pass

    owned_checkpoint = checkpoint is None
    if checkpoint is None:
        print(f"[eval] Loading from {args.output_dir}", flush=True)
        checkpoint = load_eval_checkpoint(
            args.output_dir,
            args.model_name,
            args.layer,
            args.low_rank_dim,
            args.cache_dir,
            use_word_bias=args.use_word_bias,
            variance=variance_override,
            torch_dtype=args.torch_dtype,
            load_latest=bool(args.load_latest),
        )
    else:
        print(f"[eval] Using pre-loaded checkpoint from {args.output_dir}", flush=True)

    reft_model = checkpoint.reft_model
    tokenizer = checkpoint.tokenizer
    words = list(checkpoint.words)
    prompt = checkpoint.prompt
    items = list(checkpoint.items)
    assistant_suffix = checkpoint.assistant_suffix
    from_chat_template = checkpoint.from_chat_template
    saved_cfg = checkpoint.saved_cfg or {}
    intervention_token_id = checkpoint.intervention_token_id
    content_span = checkpoint.content_span
    position = validate_intervention_position(
        args.position if args.position is not None else saved_cfg.get("position", "l1")
    )
    max_new_tokens = getattr(args, "max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
    print(f"[eval] intervention position: {position}", flush=True)
    print(f"[eval] max_new_tokens: {max_new_tokens}", flush=True)

    print(f"[eval] {len(words)} words loaded.", flush=True)

    if checkpoint.full_items is not None:
        # Subset already applied by run_full_eval (or a prior subset_eval_checkpoint call).
        train_vocab = set(checkpoint.full_words or words)
        all_words = list(checkpoint.full_words or words)
    else:
        train_vocab = set(words)
        all_words = list(words)
        if args.full_eval_n_samples is not None and args.full_eval_n_samples < len(
            items
        ):
            items = random.Random(args.seed).sample(items, args.full_eval_n_samples)
            words = [it.get("word", it["target"]) for it in items]
            print(f"[eval] Sampled {len(words)} words.", flush=True)

    word_ids = [it["id"] for it in items]

    eval_dir = os.path.join(args.output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    embed_sim_tau = float(getattr(args, "embed_sim_tau", DEFAULT_EMBED_SIM_TAU))
    embed_task = str(saved_cfg.get("task", "semantle"))
    rdkit_map_path = rdkit_map_path_for_cfg(saved_cfg)
    results: dict = {
        "n_words": len(words),
        "max_new_tokens": max_new_tokens,
        "embedding_model": embedding_model_name(embed_task),
    }

    # ── RECON: greedy reconstruction (b=μ) ────────────────────────────────
    print("\n[RECON] Greedy reconstruction (b=μ)...", flush=True)
    targets, greedy_decodes = [], []
    for i, (word, wid) in enumerate(zip(words, word_ids)):
        print(f"  {i + 1}/{len(words)}: {word!r}...", flush=True)
        targets.append(word)
        greedy_decodes.append(
            generate_text(
                reft_model,
                tokenizer,
                prompt,
                wid,
                max_new_tokens=max_new_tokens,
                position=position,
                assistant_suffix=assistant_suffix,
                from_chat_template=from_chat_template,
                intervention_token_id=intervention_token_id,
                content_span=content_span,
            )
        )
    greedy_sims = embedding_sim_per_text(targets, greedy_decodes, task=embed_task)
    greedy_tfs = tfs_per_target(targets, greedy_decodes, task=embed_task)
    greedy_rdkit_sim = rdkit_sim_per_target(
        targets,
        greedy_decodes,
        task=embed_task,
        map_path=rdkit_map_path,
    )
    per_target = [
        {"target": t, "generated": g, "sim": float(s)}
        for t, g, s in zip(targets, greedy_decodes, greedy_sims.tolist())
    ]
    results["embedding_sim"] = {
        "mean_sim": float(np.mean(greedy_sims)) if len(greedy_sims) else 0.0,
        "per_target": per_target,
    }
    if greedy_tfs is not None:
        # Carried alongside each target's embed_sim so the RECON similarity plot
        # can scatter the two against each other from one place.
        for row, tfs_value in zip(per_target, greedy_tfs.tolist()):
            row["tfs"] = float(tfs_value)
        results["embedding_sim"]["mean_tfs"] = (
            float(np.mean(greedy_tfs)) if greedy_tfs.size else 0.0
        )
    if greedy_rdkit_sim is not None:
        for row, value in zip(per_target, greedy_rdkit_sim.tolist()):
            row["rdkit_sim"] = float(value)
        results["embedding_sim"]["mean_rdkit_sim"] = (
            float(np.mean(greedy_rdkit_sim))
            if greedy_rdkit_sim.size
            else 0.0
        )
    greedy_edit_dist = edit_dist_per_target(
        targets, greedy_decodes, task=embed_task
    )
    if greedy_edit_dist is not None:
        for row, value in zip(per_target, greedy_edit_dist.tolist()):
            row["edit_dist"] = float(value)
        results["embedding_sim"]["mean_edit_dist"] = (
            float(np.mean(greedy_edit_dist))
            if greedy_edit_dist.size
            else 0.0
        )

    # ── RECON + DIST: per-temperature sample bags (b=μ) ───────────────────
    temp_results: dict[float, list[dict]] = {}
    for temp in EVAL_TEMPERATURES:
        print(
            f"\n[RECON/DIST] Temperature sampling b=μ  T={temp}, top_p={args.top_p}, "
            f"n_samples={args.n_samples} × {len(words)} words",
            flush=True,
        )
        temp_results[temp] = _run_sampling_experiment(
            reft_model,
            tokenizer,
            words,
            prompt,
            args.n_samples,
            word_ids=word_ids,
            train_vocab=train_vocab,
            use_stochastic_intervention=False,
            use_sample=True,
            temperature=temp,
            top_p=args.top_p,
            batch_size=args.eval_batch_size,
            position=position,
            assistant_suffix=assistant_suffix,
            from_chat_template=from_chat_template,
            intervention_token_id=intervention_token_id,
            content_span=content_span,
            max_new_tokens=max_new_tokens,
            task=embed_task,
        )

    recon = recon_metrics(
        targets=targets,
        greedy_decodes=greedy_decodes,
        greedy_embed_sims=greedy_sims,
        temp_results=temp_results,
        tau=embed_sim_tau,
        task=embed_task,
        rdkit_map_path=rdkit_map_path,
    )
    dist = dist_metrics(
        targets=targets,
        temp_results=temp_results,
        task=embed_task,
        rdkit_map_path=rdkit_map_path,
    )
    # Geometry of the trained bias->semantics map over the μ points: semantic
    # dispersion (spread of per-word decode centroids) and Geary's C (local
    # smoothness). Greedy uses the single greedy decode per word; each temperature
    # uses that word's sample bag. Both are None (not NaN) when degenerate, so
    # they drop out of the JSON/WandB logging rather than emitting NaN.
    mu_train = stack_bias_vectors(reft_model, word_ids)
    greedy_centroids = _sobol_sample_centroids(
        [[g] for g in greedy_decodes], task=embed_task
    )
    dist["semdisp_greedy"] = semantic_dispersion(greedy_centroids)
    dist["gearyC_greedy"] = map_geary_c(mu_train, greedy_centroids)
    for temp in EVAL_TEMPERATURES:
        train_centroids = _sobol_sample_centroids(
            [r["samples"] for r in temp_results[temp]], task=embed_task
        )
        dist[f"semdisp_{temp_key(temp)}"] = semantic_dispersion(train_centroids)
        dist[f"gearyC_{temp_key(temp)}"] = map_geary_c(mu_train, train_centroids)
    results["recon"] = recon
    results["dist"] = dist

    add_bias_network = bool(saved_cfg.get("add_bias_network"))
    interp_set: list[str] = []
    extrap_set: list[str] = []
    test_meta: dict = {}
    if add_bias_network or args.n_uniform:
        from boreft.molopt_split import load_oracle_split

        print("\n[eval] Building non-train test set (interp/extrap split)...", flush=True)
        oracle_split = load_oracle_split(args.output_dir, saved_cfg)
        high_tail = (oracle_split or {}).get("test_smiles") if oracle_split else None
        interp_set, extrap_set, test_meta = build_test_sets(
            train_targets=all_words,
            csv_paths=task_csv_paths(saved_cfg, task=embed_task),
            test_n_samples=int(getattr(args, "test_n_samples", DEFAULT_TEST_N_SAMPLES)),
            seed=args.seed,
            pca_var=float(getattr(args, "bbox_pca_var", DEFAULT_BBOX_PCA_VAR)),
            task=embed_task,
            pool=high_tail,
            pool_source="oracle_high_tail" if high_tail else None,
        )
        print(
            f"  test set: {test_meta.get('n_test', 0)} words "
            f"(interp={test_meta.get('n_interp', 0)}, "
            f"extrap={test_meta.get('n_extrap', 0)}, "
            f"pool={test_meta.get('n_pool', 0)})",
            flush=True,
        )

    # ── RECON_TEST: bias-network reconstruction of held-out test targets ──
    recon_test: dict = {}
    if add_bias_network:
        test_words = list(interp_set) + list(extrap_set)
        if test_words:
            print(
                "\n[RECON_TEST] Greedy reconstruction via bias network "
                f"({len(test_words)} held-out targets)...",
                flush=True,
            )
            def_lookup = definition_lookup_for_cfg(saved_cfg)
            enc_predict_kwargs = bias_predict_kwargs(saved_cfg, tokenizer=tokenizer)
            bias_vecs = predict_bias_vectors_for_words(
                reft_model,
                test_words,
                definition_lookup=def_lookup,
                batch_size=args.eval_batch_size,
                task=embed_task,
                **enc_predict_kwargs,
            )
            rt_greedy_decodes = _run_greedy_decode_experiment(
                reft_model,
                tokenizer,
                test_words,
                prompt,
                bias_vectors=bias_vecs,
                batch_size=args.eval_batch_size,
                position=position,
                assistant_suffix=assistant_suffix,
                from_chat_template=from_chat_template,
                intervention_token_id=intervention_token_id,
                content_span=content_span,
                max_new_tokens=max_new_tokens,
            )
            rt_greedy_sims = embedding_sim_per_text(
                test_words, rt_greedy_decodes, task=embed_task
            )

            rt_temp_results: dict[float, list[dict]] = {}
            for temp in EVAL_TEMPERATURES:
                print(
                    f"\n[RECON_TEST] Temperature sampling  T={temp}, "
                    f"top_p={args.top_p}, n_samples={args.n_samples} × "
                    f"{len(test_words)} words",
                    flush=True,
                )
                rt_temp_results[temp] = _run_sampling_experiment(
                    reft_model,
                    tokenizer,
                    test_words,
                    prompt,
                    args.n_samples,
                    bias_vectors=bias_vecs,
                    train_vocab=train_vocab,
                    use_stochastic_intervention=False,
                    use_sample=True,
                    temperature=temp,
                    top_p=args.top_p,
                    batch_size=args.eval_batch_size,
                    position=position,
                    assistant_suffix=assistant_suffix,
                    from_chat_template=from_chat_template,
                    intervention_token_id=intervention_token_id,
                    content_span=content_span,
                    max_new_tokens=max_new_tokens,
                    task=embed_task,
                )

            recon_test = recon_test_metrics(
                full_targets=test_words,
                interp_targets=interp_set,
                extrap_targets=extrap_set,
                greedy_decodes=rt_greedy_decodes,
                greedy_embed_sims=rt_greedy_sims,
                temp_results=rt_temp_results,
                tau=embed_sim_tau,
                task=embed_task,
                rdkit_map_path=rdkit_map_path,
            )
            results["recon_test"] = recon_test
            results["recon_test_meta"] = test_meta
            rt_per_target = [
                {"target": t, "generated": g, "sim": float(s)}
                for t, g, s in zip(
                    test_words, rt_greedy_decodes, rt_greedy_sims.tolist()
                )
            ]
            rt_edit = edit_dist_per_target(
                test_words, rt_greedy_decodes, task=embed_task
            )
            if rt_edit is not None:
                for row, value in zip(rt_per_target, rt_edit.tolist()):
                    row["edit_dist"] = float(value)
            interp_norm = {target_normalizer(embed_task)(w) for w in interp_set}
            extrap_norm = {target_normalizer(embed_task)(w) for w in extrap_set}
            norm = target_normalizer(embed_task)
            for row in rt_per_target:
                key = norm(row["target"])
                if key in interp_norm:
                    row["split"] = "interp"
                elif key in extrap_norm:
                    row["split"] = "extrap"
            results["recon_test_per_target"] = rt_per_target
        else:
            print(
                "[RECON_TEST] Empty test set — skipping bias-network test eval.",
                flush=True,
            )

    # ── GENZ: Sobol space sampling + test-set recovery ────────────────────
    genz: dict = {}
    if args.n_uniform:
        word_id_list = list(range(len(all_words)))
        mu_all = stack_bias_vectors(reft_model, word_id_list)
        (
            shared_sobol_bs,
            shared_nearest_words,
            shared_nearest_indices,
            mu_all,
        ) = sample_sobol_bias_vectors(
            reft_model,
            all_words,
            args.n_uniform,
            word_ids=word_id_list,
            mu_all=mu_all,
            seed=args.seed,
        )
        print("\n[GENZ] Sobol space sampling (greedy)...", flush=True)
        c4_results = run_uniform_space_sampling(
            reft_model,
            tokenizer,
            all_words,
            prompt,
            n_sobol_points=args.n_uniform,
            word_ids=word_id_list,
            train_vocab=train_vocab,
            position=position,
            assistant_suffix=assistant_suffix,
            from_chat_template=from_chat_template,
            intervention_token_id=intervention_token_id,
            content_span=content_span,
            max_new_tokens=max_new_tokens,
            sampled_bs=shared_sobol_bs,
            nearest_words=shared_nearest_words,
            nearest_indices=shared_nearest_indices,
            mu_all=mu_all,
            task=embed_task,
        )
        greedy_sobol = [
            p["samples"][0] for p in c4_results["per_sample"] if p["samples"]
        ]

        sobol_temp_results: dict[float, dict] = {}
        temp_sobol: dict[float, list[str]] = {}
        for temp in EVAL_TEMPERATURES:
            print(
                f"\n[GENZ] Sobol space sampling + temperature={temp}, "
                f"top_p={args.top_p}...",
                flush=True,
            )
            res = run_sobol_temperature_sampling(
                reft_model,
                tokenizer,
                all_words,
                prompt,
                n_sobol_points=args.n_uniform,
                n_samples=args.n_samples,
                temperature=temp,
                top_p=args.top_p,
                word_ids=word_id_list,
                train_vocab=train_vocab,
                batch_size=args.eval_batch_size,
                position=position,
                assistant_suffix=assistant_suffix,
                from_chat_template=from_chat_template,
                intervention_token_id=intervention_token_id,
                content_span=content_span,
                max_new_tokens=max_new_tokens,
                sampled_bs=shared_sobol_bs,
                nearest_words=shared_nearest_words,
                nearest_indices=shared_nearest_indices,
                mu_all=mu_all,
                task=embed_task,
            )
            sobol_temp_results[temp] = res
            temp_sobol[temp] = [s for p in res["per_sample"] for s in p["samples"]]

        write_sobol_results_json(
            os.path.join(eval_dir, "sobol_results.json"),
            greedy=c4_results,
            temperature_1_0=sobol_temp_results.get(1.0),
            temperature_1_5=sobol_temp_results.get(1.5),
        )

        genz = genz_metrics(
            train_targets=all_words,
            interp_set=interp_set,
            extrap_set=extrap_set,
            greedy_decodes=greedy_sobol,
            temp_decodes=temp_sobol,
            task=embed_task,
            rdkit_map_path=rdkit_map_path,
        )
        # LIPZ-style smoothness of the bias->semantics map over the Sobol points.
        # Greedy uses the single greedy decode per point; temp1.0 uses the
        # count-weighted centroid of the temperature=1.0 sample bag. None (not
        # NaN) when degenerate, so it serializes as null and is skipped for WandB.
        greedy_centroids = _sobol_sample_centroids(
            [p["samples"][:1] for p in c4_results["per_sample"]], task=embed_task
        )
        genz["sobol_gearyC_greedy"] = sobol_geary_c(shared_sobol_bs, greedy_centroids)
        genz["sobol_semdisp_greedy"] = semantic_dispersion(greedy_centroids)
        if 1.0 in sobol_temp_results:
            temp1_centroids = _sobol_sample_centroids(
                [p["samples"] for p in sobol_temp_results[1.0]["per_sample"]],
                task=embed_task,
            )
            genz[f"sobol_gearyC_{temp_key(1.0)}"] = sobol_geary_c(
                shared_sobol_bs, temp1_centroids
            )
            genz[f"sobol_semdisp_{temp_key(1.0)}"] = semantic_dispersion(
                temp1_centroids
            )
        results["genz"] = genz
        results["genz_test_meta"] = test_meta
    else:
        print(
            "[GENZ] --n_uniform not set — skipping Sobol/test-set GENZ metrics.",
            flush=True,
        )

    # ── Summary ───────────────────────────────────────────────────────────
    def _f(v, fmt=".4f"):
        return format(v, fmt) if v is not None else "N/A"

    print("\n[eval] RECON:", flush=True)
    print(
        f"  embed_sim={_f(recon['embed_sim'])}  "
        f"embed_sim_gte_tau(@{embed_sim_tau:g})={_f(recon['embed_sim_gte_tau'])}  "
        f"recall_greedy={_f(recon['recall_greedy'])}",
        flush=True,
    )
    if recon.get("edit_dist") is not None:
        print(f"  edit_dist={_f(recon['edit_dist'], '.2f')}", flush=True)
    for temp in EVAL_TEMPERATURES:
        print(
            f"  recall_at_n_{temp_key(temp)}="
            f"{_f(recon.get(f'recall_at_n_{temp_key(temp)}'))}",
            flush=True,
        )
    print("[eval] DIST:", flush=True)
    for k in sorted(dist):
        print(f"  {k}={_f(dist[k])}", flush=True)
    if genz:
        print("[eval] GENZ:", flush=True)
        for k in sorted(genz):
            print(f"  {k}={_f(genz[k])}", flush=True)
    if recon_test:
        print("[eval] RECON_TEST:", flush=True)
        for k in sorted(recon_test):
            print(f"  {k}={_f(recon_test[k])}", flush=True)

    results_path = os.path.join(eval_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] Saved: {results_path}")

    if owned_checkpoint and release_model:
        release_eval_checkpoint(checkpoint)
    return results


if __name__ == "__main__":
    main()
