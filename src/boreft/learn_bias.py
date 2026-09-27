"""Bias-vector learning for bias-network REFT checkpoints.

Given an unseen ``{target, definition}`` pair, the trained bias network predicts a
bias vector ``mu_pred``. When that prediction fails to reconstruct the target, this
module freezes *everything else* (base LM, rotation ``R``, ``learned_source`` ``W``,
the bias network / encoder) and directly optimizes a single low-rank bias vector to
reconstruct the target — using the **same training objective the checkpoint was
trained with**:

    total = lambda_ce * LM objective (CE or margin loss)
          + kl_beta   * KL(N(mu, sigma^2) || N(0, tau^2))     # VAE checkpoints
          + lambda_sdpo * SDPO(teacher = base + definition)   # SDPO checkpoints
          + anchor_weight * || mu - mu_pred ||^2              # stay on-manifold

The learnable vector is injected through the intervention's normal integer-word-id
forward path via :meth:`TargetBiasNetwork`-agnostic ``set_forced_bias`` (see
``interventions._EncoderBiasMixin``), so CE, VAE reparameterization sharing, and
SDPO student/on-policy scoring all reuse the exact training code paths.

The result is persisted to ``<checkpoint_dir>/learned_biases.jsonl``.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from transformers import TrainerCallback

from boreft.chem import (
    generation_smiles,
    maybe_append_mist_smiles_open_tag,
    mist_open_tag_in_prompt,
)
from boreft.data_utils import (
    ReftDataCollator,
    build_reft_row,
    chat_assistant_target,
    load_merged_run_config,
    tokenize_model_text,
    system_prompt_from_cfg,
)
from boreft.data.eval_callback import (
    EVAL_SELECTION_METRICS,
    embed_sim_frac_at_or_above,
    embed_sim_stop_triggered,
)
from boreft.eval.eval_suite import (
    DEFAULT_BBOX_PCA_VAR,
    DEFAULT_EMBED_SIM_TAU,
    normalize_text,
    split_interp_extrap,
    target_normalizer,
)
from boreft.eval.semantle import (
    DEFAULT_MAX_NEW_TOKENS,
    _get_intervention,
    generate_text,
    generate_texts_multi_batch,
)
from boreft.pyreft.losses import (
    LinearAnnealMap,
    align_linear_annealing_map,
    kl_divergence,
    legacy_kl_annealing_map_from_saved_cfg,
    load_linear_annealing_map,
    resolve_linear_annealing_map,
    serialize_linear_annealing_map,
    token_greedy_margin_loss,
)
from boreft.pyreft.sdpo import SDPOConfig, compute_sdpo_loss
from boreft.task_config import (
    sdpo_teacher_instruction,
    task_instruction,
    task_supports_fingerprints,
    task_supports_sdpo,
)
from boreft.text_similarity import (
    embedding_sim_per_text,
    encode_texts_normalized,
    rdkit_map_path_for_cfg,
)

LEARNED_BIASES_FILENAME = "learned_biases.jsonl"


@dataclass
class LearnBiasConfig:
    """Hyperparameters for :func:`learn_bias_for_pair`.

    Loss coefficients (``lambda_ce``, ``lambda_sdpo``, ``kl_beta`` …) default to the
    values read from the checkpoint config so per-instance fitting matches training.
    ``anchor_weight`` is the L2 pull toward the bias network's ``mu_pred`` — the
    on-manifold regularizer that replaces training's origin L2.
    """

    max_steps: int = 200
    lr: float = 1e-2
    anchor_weight: float = 0.1
    sim_every: int = 5
    sim_tol: float = DEFAULT_EMBED_SIM_TAU
    loss_tol: Optional[float] = None
    exact_match: bool = True
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    seed: int = 42


@dataclass
class LearnBiasResult:
    """Outcome of a per-instance bias-learning run."""

    target: str
    definition: str
    mu: np.ndarray
    mu_pred: np.ndarray
    logvar: Optional[np.ndarray]
    bias_dim: int
    steps: int
    converged: bool
    final_loss: float
    final_ce: float
    final_sdpo: Optional[float]
    final_kl: Optional[float]
    final_anchor: float
    final_sim: float
    decode: str
    # Baseline reconstruction from the bias network's prediction (pre-training).
    pred_decode: str = ""
    pred_sim: float = 0.0
    # Bias-space / embedding-space placement relative to the train set.
    classification: Optional[str] = None  # "interp" | "extrap"
    closest_train_word: Optional[str] = None
    bias_distance: Optional[float] = None  # L2 to closest train mu in bias space
    embed_sim_to_closest: Optional[float] = None  # output-embedding cosine
    persist_path: Optional[str] = None
    loss_config: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint-config extraction
# ─────────────────────────────────────────────────────────────────────────────


def _loss_config_from_ckpt(saved_cfg: dict) -> Dict[str, Any]:
    """Read the training loss coefficients that carry over to per-instance fitting."""
    return {
        "bias_type": saved_cfg.get("bias_type", "vae"),
        "lambda_ce": float(saved_cfg.get("lambda_ce", 1.0)),
        "use_margin_loss": bool(saved_cfg.get("use_margin_loss", False)),
        "margin_loss_margin": float(saved_cfg.get("margin_loss_margin", 0.1)),
        "lambda_sdpo": float(saved_cfg.get("lambda_sdpo", 0.0)),
        "kl_beta": float(saved_cfg.get("kl_beta", 0.0)),
        "kl_prior_var": float(saved_cfg.get("kl_prior_var", 1.0)),
        "vae_free_bits_lambda": float(saved_cfg.get("vae_free_bits_lambda", 0.0)),
        "lambda_l2": float(saved_cfg.get("lambda_l2", 0.0)),
    }


def _sdpo_cfg_from_ckpt(
    ckpt, *, lambda_sdpo: Optional[float] = None
) -> Optional[SDPOConfig]:
    """Rebuild the checkpoint's :class:`SDPOConfig`, or ``None`` if SDPO is off.

    When ``lambda_sdpo`` is provided it gates activation (so callers can disable
    SDPO even when the checkpoint was trained with it, or enable it on SDPO-capable
    tasks). Otherwise the checkpoint's saved ``lambda_sdpo`` is used.
    """
    cfg = ckpt.saved_cfg or {}
    task = cfg.get("task", "semantle")
    effective = (
        float(cfg.get("lambda_sdpo", 0.0))
        if lambda_sdpo is None
        else float(lambda_sdpo)
    )
    if effective <= 0.0:
        return None
    if not task_supports_sdpo(task):
        return None
    n_offpolicy = int(cfg.get("sdpo_n_offpolicy", 0))
    offpolicy_pool = int(cfg.get("sdpo_offpolicy_pool") or 0) or (2 * n_offpolicy)
    return SDPOConfig(
        divergence=cfg.get("sdpo_divergence", "forward_kl"),
        temperature=float(cfg.get("sdpo_temperature", 1.0)),
        sample_temperature=float(cfg.get("sdpo_sample_temperature", 1.0)),
        sample_top_p=float(cfg.get("sdpo_sample_top_p", 1.0)),
        max_new_tokens=int(cfg.get("sdpo_max_new_tokens", 8)),
        n_onpolicy=int(cfg.get("sdpo_n_onpolicy", 4)),
        n_offpolicy=n_offpolicy,
        offpolicy_pool=offpolicy_pool,
        include_gold=bool(cfg.get("sdpo_include_gold", True)),
        position=cfg.get("position", "l1"),
        intervention_token_id=ckpt.intervention_token_id,
        content_span=ckpt.content_span,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt / target / batch construction (mirrors ReftDataset + train.py SDPO setup)
# ─────────────────────────────────────────────────────────────────────────────


def _build_target_texts(ckpt, target: str) -> tuple[str, str]:
    """Return ``(prompt_text, full_input_text)`` for one target, as in training.

    ``prompt_text`` is the checkpoint's canonical prompt (marker already injected
    where applicable). ``full_input_text`` appends the target continuation exactly
    as :class:`ReftDataset` does, so tokenization/labels align with training.
    """
    tok = ckpt.tokenizer
    use_chat = ckpt.from_chat_template
    cfg = ckpt.saved_cfg or {}
    task = cfg.get("task", "semantle")
    smiles_tags = bool(cfg.get("smiles_tags")) and task == "molopt"
    mist_smiles_tags = bool(cfg.get("mist_smiles_tags")) and task == "molopt"
    raw_word = generation_smiles(
        target.strip(),
        smiles_tags=smiles_tags,
        mist_smiles_tags=mist_smiles_tags,
        mist_open_in_target=mist_smiles_tags and use_chat,
    )
    from boreft.intervention_marker import resolve_instruction_for_prompt

    prompt_text = ckpt.prompt
    if use_chat:
        instruction = task_instruction(
            task,
            use_chat_template=True,
            override=cfg.get("chat_instruction"),
        )
        instruction = resolve_instruction_for_prompt(
            instruction,
            cfg.get("intervention_inject", "none"),
            cfg.get("intervention_token"),
            tokenizer=tok,
            intervention_token_id=getattr(ckpt, "intervention_token_id", None),
        )
        instruction = maybe_append_mist_smiles_open_tag(
            instruction,
            mist_open_tag_in_prompt(
                mist_smiles_tags=mist_smiles_tags,
                use_chat_template=True,
            ),
        )
        item_target = chat_assistant_target(
            tok,
            instruction,
            raw_word,
            system_prompt=system_prompt_from_cfg(cfg),
        )
        full_input = prompt_text + item_target.strip()
    else:
        eos = tok.eos_token or ""
        full_input = prompt_text + " " + raw_word + eos
    return prompt_text, full_input


def _tokenize_ids(ckpt, text: str) -> List[int]:
    return list(
        tokenize_model_text(
            ckpt.tokenizer, text, from_chat_template=ckpt.from_chat_template
        )["input_ids"]
    )


def _build_ce_batch(ckpt, target: str, device) -> Dict[str, torch.Tensor]:
    """Collate a single [prompt; target] CE row with a dummy word id (0)."""
    prompt_text, full_input = _build_target_texts(ckpt, target)
    tok = ckpt.tokenizer
    use_chat = ckpt.from_chat_template
    prompt_ids = tokenize_model_text(
        tok, prompt_text, from_chat_template=use_chat, return_tensors="pt"
    )["input_ids"][0]
    input_ids = tokenize_model_text(
        tok, full_input, from_chat_template=use_chat, return_tensors="pt"
    )["input_ids"][0]
    position = (ckpt.saved_cfg or {}).get("position", "l1")
    row = build_reft_row(
        input_ids=input_ids,
        prompt_len=len(prompt_ids),
        prompt_ids=prompt_ids.tolist(),
        word_id=0,
        pad_token_id=tok.pad_token_id,
        position=position,
        intervention_token_id=ckpt.intervention_token_id,
        content_span=ckpt.content_span,
    )
    collator = ReftDataCollator(tokenizer=tok, model=ckpt.reft_model.model)
    batch = collator([row])
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _lm_loss(
    reft_model,
    batch: Dict[str, torch.Tensor],
    *,
    use_margin_loss: bool,
    margin_loss_margin: float,
) -> torch.Tensor:
    """Teacher-forced checkpoint LM objective over the target continuation."""
    unit_locations = {
        "sources->base": (
            None,
            batch["intervention_locations"].permute(1, 0, 2).tolist(),
        )
    }
    base_out, cf_out = reft_model(
        {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]},
        unit_locations=unit_locations,
        labels=batch["labels"],
        subspaces=batch["subspaces"].permute(1, 0, 2).tolist(),
    )
    out = cf_out if cf_out is not None else base_out
    if use_margin_loss:
        if out.logits is None:
            raise RuntimeError("Margin loss requires logits from intervenable forward")
        return token_greedy_margin_loss(
            out.logits,
            batch["labels"],
            margin=margin_loss_margin,
        )
    return out.loss


# ─────────────────────────────────────────────────────────────────────────────
# Bias-network prior (mu_pred warm-start + anchor target)
# ─────────────────────────────────────────────────────────────────────────────


def predict_prior_mu(ckpt, target: str, definition: str) -> np.ndarray:
    """Predict ``mu_pred`` for ``{target, definition}`` via the frozen bias network.

    Respects the checkpoint's bias-network input provenance:
      - ``llm_encoder``: the definition text is run through the copied final block.
      - ``embed_cache`` + ``use_definition_embeds``: definition-augmented embedding.
      - ``embed_cache`` word-only: the target word embedding.
    """
    from boreft.bias_tables import predict_bias_vectors_for_words

    cfg = ckpt.saved_cfg or {}
    task = cfg.get("task", "semantle")
    raw_word = target.strip()
    from boreft.text_similarity import definition_text_for_cfg

    definition = definition_text_for_cfg(cfg, raw_word, definition)
    if cfg.get("bias_input_source", "embed_cache") == "llm_encoder":
        return predict_bias_vectors_for_words(
            ckpt.reft_model,
            [raw_word],
            task=task,
            tokenizer=ckpt.tokenizer,
            raw_definition_lookup={raw_word: definition},
            encoder_max_length=int(cfg.get("bias_encoder_max_length", 64)),
            encoder_layer_index=cfg.get("bias_encoder_layer_index"),
        )[0]
    from boreft.text_similarity import bias_network_embed_model_from_cfg

    use_def = bool(cfg.get("use_definition_embeds", False))
    def_lookup = {raw_word: definition} if use_def else None
    return predict_bias_vectors_for_words(
        ckpt.reft_model,
        [raw_word],
        definition_lookup=def_lookup,
        task=task,
        embed_model=bias_network_embed_model_from_cfg(cfg),
    )[0]


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────


def learned_biases_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, LEARNED_BIASES_FILENAME)


def save_learned_bias(checkpoint_dir: str, record: Dict[str, Any]) -> str:
    """Append one learned-bias record (JSON line) and return the file path."""
    path = learned_biases_path(checkpoint_dir)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return os.path.abspath(path)


def load_learned_biases(checkpoint_dir: str) -> List[Dict[str, Any]]:
    """Read all persisted learned-bias records (empty list if none)."""
    path = learned_biases_path(checkpoint_dir)
    if not os.path.isfile(path):
        return []
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _record_normalizer(checkpoint_dir: str, task: Optional[str]) -> Callable[[str], str]:
    """Target-identity key for a checkpoint's learned-bias records.

    ``task`` is read from the checkpoint's own config when the caller does not know
    it, so a molopt run matches records by canonical SMILES rather than lower-casing
    them (which would merge aromatic and aliphatic atoms).
    """
    if task is None:
        task = str(load_merged_run_config(checkpoint_dir).get("task", "semantle"))
    return target_normalizer(task)


def delete_learned_biases(
    checkpoint_dir: str,
    targets: List[str] | tuple[str, ...],
    *,
    task: Optional[str] = None,
) -> int:
    """Remove every learned-bias record whose target matches ``targets``.

    Matching uses the task's exact-match key (see :func:`_record_normalizer`).
    Returns the number of JSONL rows removed. Rewrites ``learned_biases.jsonl`` in
    place; if no rows remain the file is deleted.
    """
    norm = _record_normalizer(checkpoint_dir, task)
    keys = {norm(target) for target in targets if str(target).strip()}
    if not keys:
        return 0
    path = learned_biases_path(checkpoint_dir)
    if not os.path.isfile(path):
        return 0
    records = load_learned_biases(checkpoint_dir)
    kept = [
        record for record in records if norm(str(record.get("target", ""))) not in keys
    ]
    removed = len(records) - len(kept)
    if removed <= 0:
        return 0
    if not kept:
        os.remove(path)
        return removed
    with open(path, "w", encoding="utf-8") as file:
        for record in kept:
            file.write(json.dumps(record) + "\n")
    return removed


def find_learned_bias(
    checkpoint_dir: str, target: str, *, task: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Return the most recently learned record whose target matches (normalized)."""
    norm = _record_normalizer(checkpoint_dir, task)
    key = norm(target)
    matches = [
        r for r in load_learned_biases(checkpoint_dir)
        if norm(str(r.get("target", ""))) == key
    ]
    if not matches:
        return None
    return max(matches, key=lambda r: r.get("learned_at", ""))


# ─────────────────────────────────────────────────────────────────────────────
# Train-set placement analysis (interp/extrap + nearest train target)
# ─────────────────────────────────────────────────────────────────────────────


def get_train_embeddings(ckpt, *, task: Optional[str] = None) -> Optional[np.ndarray]:
    """Return the checkpoint's train-target output embeddings, encoded once.

    The embeddings (rows aligned to ``ckpt.words``) are static for a loaded
    checkpoint, so they are cached on ``ckpt._learn_train_emb`` and shared with
    the REPL's ``build_test_sets`` call to avoid encoding the train vocabulary
    twice. Returns ``None`` for an empty vocabulary.
    """
    cached = getattr(ckpt, "_learn_train_emb", None)
    if cached is not None:
        return cached
    if task is None:
        task = (ckpt.saved_cfg or {}).get("task", "semantle")
    words = list(ckpt.words)
    emb = encode_texts_normalized(words, task=task) if words else None
    ckpt._learn_train_emb = emb
    return emb


def _train_reference(ckpt) -> Dict[str, Any]:
    """Cache the train words, their output embeddings, and their bias vectors.

    These are static for a loaded checkpoint (frozen bias network / materialized
    tables), so we compute them once and stash them on ``ckpt`` for reuse across
    repeated ``/learn`` calls. Train embeddings come from
    :func:`get_train_embeddings` (shared with the REPL's test-set build).
    ``mu_train`` is best-effort: for checkpoints whose bias lookup needs runtime
    encoder inputs (and no materialized tables), the stack is left as ``None``.
    """
    cache = getattr(ckpt, "_learn_train_ref", None)
    if cache is not None:
        return cache

    from boreft.bias_tables import stack_bias_vectors

    task = (ckpt.saved_cfg or {}).get("task", "semantle")
    ids = [int(it["id"]) for it in ckpt.items]
    words = list(ckpt.words)
    train_emb = get_train_embeddings(ckpt, task=task)
    try:
        mu_train = stack_bias_vectors(ckpt.reft_model, ids) if ids else None
    except (RuntimeError, ValueError, KeyError, AttributeError):
        # e.g. llm_encoder lookup without runtime encoder inputs / tables.
        mu_train = None

    cache = {
        "ids": ids,
        "words": words,
        "train_emb": train_emb,
        "mu_train": np.asarray(mu_train, dtype=np.float32) if mu_train is not None else None,
    }
    ckpt._learn_train_ref = cache
    return cache


def analyze_learned_bias(
    ckpt,
    target: str,
    mu: np.ndarray,
    *,
    pca_var: Optional[float] = None,
) -> Dict[str, Any]:
    """Place a learned bias relative to the train set.

    Returns ``classification`` (interp/extrap w.r.t. the train-embedding PCA
    bounding box), the ``closest_train_word`` in bias space, its ``bias_distance``
    (L2 between mean bias vectors), and ``embed_sim_to_closest`` (output-embedding
    cosine between the learned target and that closest train word).
    """
    saved_cfg = ckpt.saved_cfg or {}
    task = saved_cfg.get("task", "semantle")
    if pca_var is None:
        pca_var = float(saved_cfg.get("eval_bbox_pca_var", DEFAULT_BBOX_PCA_VAR))

    ref = _train_reference(ckpt)
    words = ref["words"]
    out: Dict[str, Any] = {
        "classification": None,
        "closest_train_word": None,
        "bias_distance": None,
        "embed_sim_to_closest": None,
    }

    # Interp/extrap: robust — only needs output embeddings (always available).
    if ref["train_emb"] is not None and len(words) >= 2:
        tgt_emb = encode_texts_normalized([target], task=task)
        interp, _extrap = split_interp_extrap(
            ref["train_emb"], tgt_emb, [target], pca_var=pca_var
        )
        out["classification"] = "interp" if interp else "extrap"

    # Nearest train target in bias space (best-effort; needs stacked train mu).
    mu_train = ref["mu_train"]
    if mu_train is not None and len(mu_train) > 0:
        mu_vec = np.asarray(mu, dtype=np.float32).reshape(1, -1)
        dists = np.linalg.norm(mu_train - mu_vec, axis=1)
        j = int(np.argmin(dists))
        closest = words[j]
        out["closest_train_word"] = closest
        out["bias_distance"] = float(dists[j])
        out["embed_sim_to_closest"] = float(
            embedding_sim_per_text([target], [closest], task=task)[0]
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Core learning loop
# ─────────────────────────────────────────────────────────────────────────────


def _greedy_decode_and_sim(ckpt, mu_np: np.ndarray, target: str, task: str, max_new_tokens: int):
    """Greedy-decode with the current mean bias, return (decode, embed_sim).

    Forces the base LM to eval so the reconstruction check is deterministic even
    when the optimization step ran the model in train mode (dropout active).
    """
    ckpt.reft_model.model.eval()
    decode = generate_text(
        ckpt.reft_model,
        ckpt.tokenizer,
        ckpt.prompt,
        mu_np,
        max_new_tokens=max_new_tokens,
        use_sample=False,
        position=(ckpt.saved_cfg or {}).get("position", "l1"),
        assistant_suffix=ckpt.assistant_suffix,
        from_chat_template=ckpt.from_chat_template,
        intervention_token_id=ckpt.intervention_token_id,
        content_span=ckpt.content_span,
    )
    sim = float(embedding_sim_per_text([target], [decode], task=task)[0])
    return decode, sim


def learn_bias_for_pair(
    ckpt,
    *,
    target: str,
    definition: str,
    checkpoint_dir: Optional[str] = None,
    config: Optional[LearnBiasConfig] = None,
    show_progress: bool = True,
    persist: bool = True,
) -> LearnBiasResult:
    """Learn a bias vector for one ``{target, definition}`` pair (see module docstring).

    ``ckpt`` is a loaded :class:`boreft.eval.semantle.EvalCheckpoint`. Requires an
    ``--add-bias-network`` checkpoint (a ``mu_pred`` prior is needed for warm-start
    and the anchor term). ``checkpoint_dir`` is where ``learned_biases.jsonl`` is
    written when ``persist=True``.
    """
    config = config or LearnBiasConfig()
    target = target.strip()
    definition = definition.strip()
    if not target or not definition:
        raise ValueError("learn_bias_for_pair requires non-empty target and definition")

    saved_cfg = ckpt.saved_cfg or {}
    task = saved_cfg.get("task", "semantle")
    intervention = _get_intervention(ckpt.reft_model)
    if getattr(intervention, "bias_network", None) is None:
        raise ValueError(
            "learn_bias_for_pair requires an --add-bias-network checkpoint"
        )

    loss_cfg = _loss_config_from_ckpt(saved_cfg)
    is_vae = loss_cfg["bias_type"] == "vae"
    # Align with the generation path (generate_text uses reft_model.get_device())
    # so the CE/SDPO batches and the sim-eval decode all live on one device.
    device = ckpt.reft_model.get_device()

    # Freeze the whole model; only the injected bias vector(s) train.
    for p in ckpt.reft_model.parameters():
        p.requires_grad_(False)

    mu_pred = np.asarray(predict_prior_mu(ckpt, target, definition), dtype=np.float32)
    mu_pred_t = torch.tensor(mu_pred, dtype=torch.float32, device=device)
    bias_dim = int(mu_pred.shape[0])

    # Baseline decode from the predicted bias, before any optimization. Done before
    # set_forced_bias so generate_text uses mu_pred directly via the raw-vector path.
    pred_decode, pred_sim = _greedy_decode_and_sim(
        ckpt, mu_pred, target, task, config.max_new_tokens
    )
    if show_progress:
        print(f"[learn] mu_pred decode={pred_decode!r} sim={pred_sim:.3f}")

    mu = torch.nn.Parameter(mu_pred_t.clone())
    params: List[torch.nn.Parameter] = [mu]
    learn_logvar = is_vae and getattr(intervention, "fixed_logvar", None) is None
    logvar: Optional[torch.nn.Parameter] = None
    if learn_logvar:
        logvar = torch.nn.Parameter(torch.zeros(bias_dim, dtype=torch.float32, device=device))
        params.append(logvar)

    optimizer = torch.optim.Adam(params, lr=config.lr)

    # SDPO setup (built once; teacher distribution is fixed for this pair).
    sdpo_cfg = _sdpo_cfg_from_ckpt(ckpt)
    sdpo_active = sdpo_cfg is not None and loss_cfg["lambda_sdpo"] > 0.0
    sdpo_state: Dict[str, Any] = {}
    if sdpo_active:
        from boreft.text_similarity import definition_text_for_cfg

        prompt_text, full_input = _build_target_texts(ckpt, target)
        student_prompt_ids = _tokenize_ids(ckpt, prompt_text)
        full_ids = _tokenize_ids(ckpt, full_input)
        gold_target_ids = full_ids[len(student_prompt_ids):]
        teacher_text = sdpo_teacher_instruction(
            task,
            definition_text_for_cfg(saved_cfg, target, definition),
            use_chat_template=ckpt.from_chat_template,
        )
        teacher_text = maybe_append_mist_smiles_open_tag(
            teacher_text,
            mist_open_tag_in_prompt(
                mist_smiles_tags=bool(saved_cfg.get("mist_smiles_tags")),
                use_chat_template=ckpt.from_chat_template,
            ),
        )
        if ckpt.from_chat_template:
            from boreft.data_utils import chat_prompt

            teacher_text = chat_prompt(
                ckpt.tokenizer,
                teacher_text,
                system_prompt=system_prompt_from_cfg(saved_cfg),
            )
        sdpo_state = {
            "student_prompt_ids": student_prompt_ids,
            "teacher_prompt_ids": [_tokenize_ids(ckpt, teacher_text)],
            "target_ids": [gold_target_ids],
            "collator": ReftDataCollator(tokenizer=ckpt.tokenizer, model=ckpt.reft_model.model),
            "base_model": ckpt.reft_model.model,
            "offpolicy_cache": {},
        }

    intervention.set_forced_bias(mu, logvar)
    ce_batch = _build_ce_batch(ckpt, target, device)

    result_stats: Dict[str, Any] = {}
    decode, sim = "", 0.0
    converged = False
    steps_done = 0

    pbar = None
    if show_progress:
        from tqdm import tqdm

        pbar = tqdm(
            range(config.max_steps),
            desc=f"learn {target!r}",
            unit="step",
            dynamic_ncols=True,
            leave=True,
        )
    step_iter = pbar if pbar is not None else range(config.max_steps)

    try:
        for step in step_iter:
            # Match the training regime: run base LM + intervention in train mode so
            # every dropout the checkpoint trained with (base-model dropout,
            # intervention output dropout, dropout_on_b) and VAE sampling are active.
            # The frozen params still receive no updates (requires_grad=False); only
            # the injected bias vector(s) train. The sim-eval decode forces eval for
            # a deterministic reconstruction check.
            ckpt.reft_model.model.train()
            intervention.train()
            if sdpo_active:
                intervention._shared_b = {}
                intervention._shared_bias = {}
            try:
                optimizer.zero_grad(set_to_none=True)
                lm_loss = _lm_loss(
                    ckpt.reft_model,
                    ce_batch,
                    use_margin_loss=loss_cfg["use_margin_loss"],
                    margin_loss_margin=loss_cfg["margin_loss_margin"],
                )
                total = loss_cfg["lambda_ce"] * lm_loss

                kl_val: Optional[torch.Tensor] = None
                if is_vae and loss_cfg["kl_beta"] > 0.0:
                    if logvar is not None:
                        lv = logvar
                    else:
                        fixed = getattr(intervention, "fixed_logvar", None)
                        lv = torch.full_like(mu, fixed if fixed is not None else 0.0)
                    kl_val = kl_divergence(
                        mu.unsqueeze(0),
                        lv.unsqueeze(0),
                        prior_var=loss_cfg["kl_prior_var"],
                        free_bits=loss_cfg["vae_free_bits_lambda"],
                    )
                    total = total + loss_cfg["kl_beta"] * kl_val

                anchor = (mu - mu_pred_t).pow(2).sum()
                total = total + config.anchor_weight * anchor

                sdpo_val: Optional[torch.Tensor] = None
                if sdpo_active:
                    sdpo_val, _ = compute_sdpo_loss(
                        intervenable=ckpt.reft_model,
                        base_model=sdpo_state["base_model"],
                        tokenizer=ckpt.tokenizer,
                        word_ids=[0],
                        student_prompt_ids=sdpo_state["student_prompt_ids"],
                        teacher_prompt_ids=sdpo_state["teacher_prompt_ids"],
                        target_ids=sdpo_state["target_ids"],
                        cfg=sdpo_cfg,
                        collator=sdpo_state["collator"],
                        device=device,
                        offpolicy_cache=sdpo_state["offpolicy_cache"],
                        draw_seed=config.seed * 1_000_003 + step,
                    )
                    total = total + loss_cfg["lambda_sdpo"] * sdpo_val

                total.backward()
                optimizer.step()
            finally:
                if sdpo_active:
                    intervention._shared_b = None
                    intervention._shared_bias = None

            steps_done = step + 1
            result_stats = {
                "loss": float(total.detach()),
                # Keep the legacy key/result field for persisted-result compatibility;
                # for margin checkpoints this is the primary margin loss.
                "ce": float(lm_loss.detach()),
                "anchor": float(anchor.detach()),
                "kl": float(kl_val.detach()) if kl_val is not None else None,
                "sdpo": float(sdpo_val.detach()) if sdpo_val is not None else None,
            }

            do_eval = (step % max(1, config.sim_every) == 0) or (step == config.max_steps - 1)
            if do_eval:
                mu_np = mu.detach().float().cpu().numpy()
                decode, sim = _greedy_decode_and_sim(
                    ckpt, mu_np, target, task, config.max_new_tokens
                )

            if pbar is not None:
                postfix = {"loss": f"{result_stats['loss']:.4f}", "ce": f"{result_stats['ce']:.4f}"}
                if result_stats["sdpo"] is not None:
                    postfix["sdpo"] = f"{result_stats['sdpo']:.3f}"
                if result_stats["kl"] is not None:
                    postfix["kl"] = f"{result_stats['kl']:.3f}"
                postfix["sim"] = f"{sim:.3f}"
                postfix["decode"] = decode[:16]
                pbar.set_postfix(postfix, refresh=True)

            hit_exact = config.exact_match and decode.strip() == target
            hit_sim = sim >= config.sim_tol
            hit_loss = config.loss_tol is not None and result_stats["loss"] <= config.loss_tol
            if do_eval and (hit_exact or hit_sim or hit_loss):
                converged = True
                if pbar is not None:
                    pbar.update(config.max_steps - steps_done)
                break
    finally:
        intervention.clear_forced_bias()
        intervention.eval()
        ckpt.reft_model.model.eval()  # restore eval for subsequent REPL generation
        if pbar is not None:
            pbar.close()

    # Final deterministic evaluation of the learned mean bias.
    mu_np = mu.detach().float().cpu().numpy()
    decode, sim = _greedy_decode_and_sim(ckpt, mu_np, target, task, config.max_new_tokens)
    logvar_np = logvar.detach().float().cpu().numpy() if logvar is not None else None

    # Placement relative to the train set (forced bias already cleared above, so
    # the cached train mu reflect the frozen network, not the learned vector).
    try:
        analysis = analyze_learned_bias(ckpt, target, mu_np)
    except (RuntimeError, ValueError, KeyError, AttributeError):
        analysis = {}

    result = LearnBiasResult(
        target=target,
        definition=definition,
        mu=mu_np,
        mu_pred=mu_pred,
        logvar=logvar_np,
        bias_dim=bias_dim,
        steps=steps_done,
        converged=converged,
        final_loss=float(result_stats.get("loss", float("nan"))),
        final_ce=float(result_stats.get("ce", float("nan"))),
        final_sdpo=result_stats.get("sdpo"),
        final_kl=result_stats.get("kl"),
        final_anchor=float(result_stats.get("anchor", float("nan"))),
        final_sim=sim,
        decode=decode,
        pred_decode=pred_decode,
        pred_sim=pred_sim,
        classification=analysis.get("classification"),
        closest_train_word=analysis.get("closest_train_word"),
        bias_distance=analysis.get("bias_distance"),
        embed_sim_to_closest=analysis.get("embed_sim_to_closest"),
        loss_config={**loss_cfg, "anchor_weight": config.anchor_weight},
    )

    if persist:
        persist_dir = checkpoint_dir or (ckpt.saved_cfg or {}).get("output_dir")
        if not persist_dir or not os.path.isdir(persist_dir):
            raise ValueError(
                "persist=True requires a valid checkpoint_dir to write "
                f"{LEARNED_BIASES_FILENAME}"
            )
        record = {
            "target": target,
            "definition": definition,
            "target_norm": normalize_text(target, task=task),
            "mu": mu_np.tolist(),
            "mu_pred": mu_pred.tolist(),
            "logvar": logvar_np.tolist() if logvar_np is not None else None,
            "bias_dim": bias_dim,
            "bias_type": loss_cfg["bias_type"],
            "steps": steps_done,
            "converged": converged,
            "final_loss": result.final_loss,
            "final_sim": sim,
            "decode": decode,
            "pred_decode": pred_decode,
            "pred_sim": pred_sim,
            "classification": result.classification,
            "closest_train_word": result.closest_train_word,
            "bias_distance": result.bias_distance,
            "embed_sim_to_closest": result.embed_sim_to_closest,
            "loss_config": result.loss_config,
            "learned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        result.persist_path = save_learned_bias(persist_dir, record)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Batched learning through the main ReftTrainer code path
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class BatchLearnConfig:
    """Hyperparameters for :func:`learn_biases_batched`.

    Every ``None`` value inherits the corresponding saved training parameter.
    The learnable bias table is warm-started from the bias network's ``mu_pred``
    (no anchor term). Loss weights (``lambda_ce``, ``lambda_sdpo``, ``kl_beta``)
    and ``weight_decay_mode`` can be overridden modularly; ``0`` / ``"none"``
    disables the corresponding term. ``linear_annealing_map`` inherits the
    checkpoint map when ``None``; an explicit string/dict overrides it (empty
    string disables annealing for this run).
    """

    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    grad_acc_steps: Optional[int] = None
    lr: Optional[float] = None
    kl_beta: Optional[float] = None
    lambda_ce: Optional[float] = None
    lambda_sdpo: Optional[float] = None
    # None = inherit checkpoint map; str/dict = override; "" = disable.
    linear_annealing_map: Optional[str | dict] = None
    seed: Optional[int] = None
    max_grad_norm: Optional[float] = None
    lr_scheduler_type: Optional[str] = None
    warmup_ratio: Optional[float] = None
    weight_decay_mode: Optional[str] = None
    wd_W: Optional[float] = None
    wd_b: Optional[float] = None
    torch_dtype: Optional[str] = None
    eval_epochs: Optional[int] = None
    eval_batch_size: Optional[int] = None
    eval_max_new_tokens: Optional[int] = None
    # Metric that drives early stopping: embed_sim / rdkit_sim / tfs. None
    # inherits the checkpoint's eval_selection_metric, else embed_sim. Unlike
    # the stop thresholds this is never "off", so it ignores
    # inherit_stop_threshold.
    eval_selection_metric: Optional[str] = None
    stop_threshold: Optional[float] = None
    stop_threshold_min: Optional[float] = None
    # Fraction of eval targets that must clear stop_threshold_min (default 1.0).
    # None inherits the checkpoint value when inherit_stop_threshold is True.
    stop_threshold_frac: Optional[float] = None
    # When True (default), None stop_* values fall back to the checkpoint config.
    # Viz recovery sets this False so a blank UI field means "off", not inherit.
    inherit_stop_threshold: bool = True
    learn_W: bool = False
    learn_R: bool = False
    learn_bias_network: bool = False


@dataclass(frozen=True)
class GroupStopSpec:
    """Per-eval-group early-stop bars (AND-combined with the global bars)."""

    stop_threshold: Optional[float] = None
    stop_threshold_min: Optional[float] = None
    stop_threshold_frac: float = 1.0

    def validate(self) -> None:
        if self.stop_threshold is None and self.stop_threshold_min is None:
            raise ValueError(
                "GroupStopSpec requires stop_threshold and/or stop_threshold_min"
            )
        frac = float(self.stop_threshold_frac)
        if not (0.0 < frac <= 1.0):
            raise ValueError("stop_threshold_frac must be in (0, 1]")
        if frac < 1.0 - 1e-12 and self.stop_threshold_min is None:
            raise ValueError("stop_threshold_frac < 1 requires stop_threshold_min")


def grouped_stop_triggered(
    *,
    global_mean: float,
    global_min: float,
    global_sims: Optional[Sequence[float]],
    stop_threshold: Optional[float],
    stop_threshold_min: Optional[float],
    stop_threshold_frac: float,
    group_slices: Optional[Dict[str, tuple[float, float, Sequence[float]]]] = None,
    stop_groups: Optional[Dict[str, GroupStopSpec]] = None,
) -> bool:
    """AND global thresholds with named group thresholds.

    A mean of 1.0 on the global slice still always stops. When only group
    specs are set, stopping requires every named group to pass.
    """
    if float(global_mean) >= 1.0 - 1e-12:
        return True
    global_configured = stop_threshold is not None or stop_threshold_min is not None
    groups_configured = bool(stop_groups)
    if not global_configured and not groups_configured:
        return False
    global_ok = True
    if global_configured:
        global_ok = embed_sim_stop_triggered(
            global_mean,
            global_min,
            stop_threshold=stop_threshold,
            stop_threshold_min=stop_threshold_min,
            stop_threshold_frac=stop_threshold_frac,
            sims=global_sims,
        )
    groups_ok = True
    if groups_configured:
        if not group_slices:
            raise ValueError("stop_groups requires group_slices")
        for name, spec in stop_groups.items():
            if name not in group_slices:
                raise ValueError(f"stop group {name!r} missing from eval slices")
            spec.validate()
            mean, min_sim, sims = group_slices[name]
            groups_ok = groups_ok and embed_sim_stop_triggered(
                mean,
                min_sim,
                stop_threshold=spec.stop_threshold,
                stop_threshold_min=spec.stop_threshold_min,
                stop_threshold_frac=spec.stop_threshold_frac,
                sims=sims,
            )
    return bool(global_ok and groups_ok)


def resolve_batch_linear_annealing_map(
    config: BatchLearnConfig,
    saved_cfg: dict,
    *,
    kl_beta: float,
    lambda_ce: float,
    lambda_sdpo: float,
) -> LinearAnnealMap:
    """Resolve the annealing map for a batched-learning run.

    Priority:
      1. Explicit ``config.linear_annealing_map`` (empty string → no anneal)
      2. Checkpoint ``linear_annealing_map``
      3. Legacy checkpoint ``kl_beta_anneal*`` fields

    Disabled terms (static coeff ``<= 0``) are dropped. When inheriting a
    checkpoint map, anneal ends are rewritten only for coefficients that were
    explicitly overridden on ``config``; otherwise the persisted start/end is
    kept (so cool-downs survive when the static flag is a training gate).
    Explicit map overrides own their ends (shorthand already fills from the
    current coeffs).
    """
    explicit_map = config.linear_annealing_map is not None
    if explicit_map:
        raw = config.linear_annealing_map
        if isinstance(raw, str) and not raw.strip():
            anneal_map: LinearAnnealMap = {}
        else:
            anneal_map = resolve_linear_annealing_map(
                raw_map=raw,
                defaults={
                    "kl_beta": float(kl_beta),
                    "lambda_ce": float(lambda_ce),
                    "lambda_sdpo": float(lambda_sdpo),
                },
            )
    elif "linear_annealing_map" in saved_cfg:
        # Explicit key (including null/empty) means "use this map"; do not fall
        # back to legacy kl_beta_anneal* after a recovery that disabled annealing.
        raw = saved_cfg["linear_annealing_map"]
        anneal_map = load_linear_annealing_map(raw) if raw else {}
    else:
        anneal_map = legacy_kl_annealing_map_from_saved_cfg(
            saved_cfg, kl_beta=kl_beta
        )
    override_ends: set[str] = set()
    if not explicit_map:
        if config.kl_beta is not None:
            override_ends.add("kl_beta")
        if config.lambda_ce is not None:
            override_ends.add("lambda_ce")
        if config.lambda_sdpo is not None:
            override_ends.add("lambda_sdpo")
    return align_linear_annealing_map(
        anneal_map,
        kl_beta=kl_beta,
        lambda_ce=lambda_ce,
        lambda_sdpo=lambda_sdpo,
        override_ends=override_ends,
    )


@dataclass
class BatchLearnResult:
    """Outcome of a batched bias-learning run over many targets."""

    targets: List[str]
    mu: np.ndarray  # [N, rank] learned means (row i ↔ targets[i])
    mu_pred: np.ndarray  # [N, rank] warm-start predictions
    logvar: Optional[np.ndarray]  # [N, rank] learned logvar, or None
    bias_dim: int
    steps: int
    mean_train_loss: float  # mean loss over all training steps (HF training_loss)
    loss_config: Dict[str, Any] = field(default_factory=dict)
    resolved_train_config: Dict[str, Any] = field(default_factory=dict)
    eval_history: List[Dict[str, Any]] = field(default_factory=list)
    stopped_early: bool = False


def batch_learn_stopped_early(
    *,
    eval_history: List[Dict[str, Any]],
    training_control: Optional["BatchLearnControl"],
) -> bool:
    """True when the user requested stop or embed-sim early-stop fired."""
    if training_control is not None and training_control.stop_requested:
        return True
    return any(bool(entry.get("early_stopped")) for entry in eval_history)


class _RowDataset(torch.utils.data.Dataset):
    """In-memory dataset of pre-built REFT rows (row i carries word id i)."""

    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return deepcopy(self._rows[idx])


def _build_learn_row(ckpt, target: str, word_id: int) -> Dict[str, Any]:
    """Build one CE row for ``target`` with subspace word id ``word_id``.

    Mirrors :func:`_build_ce_batch` (and training's :class:`ReftDataset`) but keeps
    the row un-collated so the trainer's dataloader + collator batch it, and tags a
    distinct id so each row indexes its own learnable bias-table row.
    """
    prompt_text, full_input = _build_target_texts(ckpt, target)
    tok = ckpt.tokenizer
    use_chat = ckpt.from_chat_template
    prompt_ids = tokenize_model_text(
        tok, prompt_text, from_chat_template=use_chat, return_tensors="pt"
    )["input_ids"][0]
    input_ids = tokenize_model_text(
        tok, full_input, from_chat_template=use_chat, return_tensors="pt"
    )["input_ids"][0]
    position = (ckpt.saved_cfg or {}).get("position", "l1")
    return build_reft_row(
        input_ids=input_ids,
        prompt_len=len(prompt_ids),
        prompt_ids=prompt_ids.tolist(),
        word_id=word_id,
        pad_token_id=tok.pad_token_id,
        position=position,
        intervention_token_id=ckpt.intervention_token_id,
        content_span=ckpt.content_span,
    )


def _build_batched_learn_rows(
    ckpt,
    targets: List[str],
    *,
    nbr_lambda: float,
    nbr_top_k: int,
) -> tuple[List[Dict[str, Any]], Optional[int]]:
    """Build direct-learning rows, including training-style neighborhood CE.

    Each neighbor row uses the anchor's learnable-bias id but the neighbor's target
    text, matching :meth:`SemantleItem.build_augmented_items`. Rows are emitted in
    fixed anchor blocks so the main trainer's block sampler can be reused.
    """
    use_nbr = (
        nbr_lambda > 0.0
        and (ckpt.saved_cfg or {}).get("task", "semantle") == "semantle"
        and len(targets) > 1
    )
    if not use_nbr:
        return [_build_learn_row(ckpt, word, i) for i, word in enumerate(targets)], None

    from boreft.text_similarity import pairwise_embedding_similarity_matrix

    task = (ckpt.saved_cfg or {}).get("task", "semantle")
    sim = pairwise_embedding_similarity_matrix(targets, task=task)
    k_eff = min(int(nbr_top_k), len(targets) - 1)
    rows: List[Dict[str, Any]] = []
    for i, word in enumerate(targets):
        anchor = _build_learn_row(ckpt, word, i)
        anchor["weight"] = torch.tensor(1.0, dtype=torch.float32)
        rows.append(anchor)
        row_sim = sim[i].clone()
        row_sim[i] = float("-inf")
        values, indices = torch.topk(row_sim, k_eff)
        for value, index in zip(values, indices):
            j = int(index.item())
            row = _build_learn_row(ckpt, targets[j], i)
            weight = nbr_lambda * max(0.0, (float(value.item()) + 1.0) / 2.0)
            row["weight"] = torch.tensor(weight, dtype=torch.float32)
            rows.append(row)
    return rows, 1 + k_eff


def _build_batch_sdpo_kwargs(
    ckpt,
    targets: List[str],
    definitions: Dict[str, str],
    sdpo_cfg: SDPOConfig,
    *,
    lambda_sdpo: float,
) -> Dict[str, Any]:
    """Assemble per-(dummy-id) SDPO prompt/target ids for the trainer.

    Parallels :func:`boreft.train._build_sdpo_trainer_kwargs` but indexes by batch
    position (0..N-1) instead of the training vocabulary id.
    """
    tok = ckpt.tokenizer
    saved_cfg = ckpt.saved_cfg or {}
    task = saved_cfg.get("task", "semantle")
    from boreft.text_similarity import (
        definition_text_for_cfg,
        rdkit_definition_lookup_for_cfg,
    )

    rdkit_lookup = rdkit_definition_lookup_for_cfg(saved_cfg)
    student_prompt_ids = _tokenize_ids(ckpt, ckpt.prompt)
    teacher_prompt_ids: List[List[int]] = []
    target_ids: List[List[int]] = []
    for word in targets:
        prompt_text, full_input = _build_target_texts(ckpt, word)
        p_ids = _tokenize_ids(ckpt, prompt_text)
        target_ids.append(_tokenize_ids(ckpt, full_input)[len(p_ids):])
        definition = definition_text_for_cfg(
            saved_cfg,
            word,
            definitions[word],
            rdkit_lookup=rdkit_lookup,
        )
        teacher_text = sdpo_teacher_instruction(
            task, definition, use_chat_template=ckpt.from_chat_template
        )
        teacher_text = maybe_append_mist_smiles_open_tag(
            teacher_text,
            mist_open_tag_in_prompt(
                mist_smiles_tags=bool(saved_cfg.get("mist_smiles_tags")),
                use_chat_template=ckpt.from_chat_template,
            ),
        )
        if ckpt.from_chat_template:
            from boreft.data_utils import chat_prompt

            teacher_text = chat_prompt(
                tok, teacher_text, system_prompt=system_prompt_from_cfg(saved_cfg)
            )
        teacher_prompt_ids.append(_tokenize_ids(ckpt, teacher_text))
    return dict(
        lambda_sdpo=float(lambda_sdpo),
        sdpo_cfg=sdpo_cfg,
        sdpo_base_model=ckpt.reft_model.model,
        sdpo_student_prompt_ids=student_prompt_ids,
        sdpo_teacher_prompt_ids=teacher_prompt_ids,
        sdpo_target_ids=target_ids,
    )


def learned_set_slice_metrics(
    targets: Sequence[str],
    decodes: Sequence[str],
    similarities: Sequence[float],
    *,
    embed_sim_tau: float,
    stop_threshold_min: Optional[float] = None,
    task: str = "semantle",
    selection_metric: str = "embed_sim",
    selection_similarities: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Scalar greedy-eval metrics for one target slice (unit-testable).

    ``embed_sim`` keys always describe the embedding similarities in
    ``similarities``. The ``selection_*`` keys describe whichever metric drives
    early stopping, passed as ``selection_similarities`` (defaults to the
    embedding similarities, i.e. ``selection_metric='embed_sim'``).
    """
    if not targets:
        raise ValueError("learned_set_slice_metrics requires a non-empty slice")
    norm = target_normalizer(task)
    sims = np.asarray(similarities, dtype=np.float64)
    sel = (
        sims
        if selection_similarities is None
        else np.asarray(selection_similarities, dtype=np.float64)
    )
    if sel.shape != sims.shape:
        raise ValueError(
            "selection_similarities must be parallel to similarities "
            f"({sel.shape} vs {sims.shape})"
        )
    n_recovered = sum(
        norm(target) == norm(decode) for target, decode in zip(targets, decodes)
    )
    metrics: Dict[str, Any] = {
        "avg_embed_sim": float(np.mean(sims)),
        "min_embed_sim": float(np.min(sims)),
        "n_recovered": int(n_recovered),
        "n_targets": len(targets),
        "greedy_decodes": sorted_greedy_decode_rows(
            targets, decodes, sims, task=task
        ),
        "embed_sim_tau": float(embed_sim_tau),
        "embed_sim_gte_tau": embed_sim_frac_at_or_above(sims, embed_sim_tau),
        "selection_metric": str(selection_metric),
        "avg_selection_sim": float(np.mean(sel)),
        "min_selection_sim": float(np.min(sel)),
    }
    if stop_threshold_min is not None:
        metrics["embed_sim_gte_stop_min"] = embed_sim_frac_at_or_above(
            sims, stop_threshold_min
        )
        metrics["selection_gte_stop_min"] = embed_sim_frac_at_or_above(
            sel, stop_threshold_min
        )
    return metrics


class _LearnedSetEvalCallback(TrainerCallback):
    """Greedily evaluate every currently learned bias at epoch boundaries."""

    def __init__(
        self,
        ckpt,
        targets: List[str],
        *,
        eval_epochs: int,
        batch_size: int,
        max_new_tokens: int,
        stop_threshold: Optional[float],
        stop_threshold_min: Optional[float],
        stop_threshold_frac: float = 1.0,
        embed_sim_tau: float = DEFAULT_EMBED_SIM_TAU,
        selection_metric: str = "embed_sim",
        eval_indices: Optional[List[int]] = None,
        eval_groups: Optional[Dict[str, Sequence[int]]] = None,
        stop_eval_indices: Optional[List[int]] = None,
        stop_groups: Optional[Dict[str, GroupStopSpec]] = None,
        result_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.ckpt = ckpt
        self.all_targets = list(targets)
        # Named subsets of ``all_targets`` row indices (e.g. previous / new /
        # combined). When set, ``eval_indices`` defaults to their union.
        self.eval_groups: Dict[str, List[int]] = {
            str(name): [int(i) for i in idxs]
            for name, idxs in (eval_groups or {}).items()
            if idxs
        }
        if eval_indices is not None:
            self.eval_indices = list(eval_indices)
        elif self.eval_groups:
            seen: set[int] = set()
            union: List[int] = []
            for idxs in self.eval_groups.values():
                for i in idxs:
                    if i not in seen:
                        seen.add(i)
                        union.append(i)
            self.eval_indices = union
        else:
            self.eval_indices = list(range(len(self.all_targets)))
        # Early-stop / top-level metrics use this subset (defaults to eval set;
        # ``extend_subspace`` points it at the new absorption targets).
        self.stop_eval_indices = (
            list(stop_eval_indices)
            if stop_eval_indices is not None
            else list(self.eval_indices)
        )
        self.stop_groups: Dict[str, GroupStopSpec] = dict(stop_groups or {})
        for name, spec in self.stop_groups.items():
            spec.validate()
            if name not in self.eval_groups:
                raise ValueError(
                    f"stop group {name!r} is not in eval_groups "
                    f"{tuple(self.eval_groups)}"
                )
        self.targets = [self.all_targets[i] for i in self.eval_indices]
        self.eval_epochs = eval_epochs
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.stop_threshold = stop_threshold
        self.stop_threshold_min = stop_threshold_min
        self.stop_threshold_frac = float(stop_threshold_frac)
        if not (0.0 < self.stop_threshold_frac <= 1.0):
            raise ValueError("stop_threshold_frac must be in (0, 1]")
        if (
            self.stop_threshold_frac < 1.0 - 1e-12
            and self.stop_threshold_min is None
        ):
            raise ValueError(
                "stop_threshold_frac < 1 requires stop_threshold_min"
            )
        self.embed_sim_tau = float(embed_sim_tau)
        if not (0.0 <= self.embed_sim_tau <= 1.0):
            raise ValueError("embed_sim_tau must be in [0, 1]")
        self.task = (ckpt.saved_cfg or {}).get("task", "semantle")
        self.selection_metric = str(selection_metric)
        if self.selection_metric not in EVAL_SELECTION_METRICS:
            raise ValueError(
                f"selection_metric must be one of {EVAL_SELECTION_METRICS}, "
                f"got {selection_metric!r}"
            )
        if self.selection_metric != "embed_sim" and not task_supports_fingerprints(
            self.task
        ):
            raise ValueError(
                f"selection_metric={self.selection_metric!r} is unavailable for "
                f"task={self.task!r}"
            )
        self.rdkit_map_path = (
            rdkit_map_path_for_cfg(ckpt.saved_cfg or {})
            if self.selection_metric == "rdkit_sim"
            else None
        )
        self.target_embeddings = encode_texts_normalized(self.targets, task=self.task)
        self.history: List[Dict[str, Any]] = []
        self.trainer = None
        self.result_callback = result_callback

    def _selection_sims(self, decodes: Sequence[str], embed_sims) -> np.ndarray:
        """Per-target values of the metric that drives early stopping."""
        if self.selection_metric == "embed_sim":
            return np.asarray(embed_sims, dtype=np.float64)
        if self.selection_metric == "tfs":
            from boreft.chem import tanimoto_sim_per_text

            return tanimoto_sim_per_text(self.targets, list(decodes))
        from boreft.chem import rdkit_sim_per_text

        return rdkit_sim_per_text(
            self.targets, list(decodes), map_path=self.rdkit_map_path
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(round(float(state.epoch or 0.0)))
        if epoch <= 0 or epoch % self.eval_epochs != 0:
            return control
        if not self.targets:
            return control

        intervention = _get_intervention(self.ckpt.reft_model)
        base_was_training = self.ckpt.reft_model.model.training
        intervention_was_training = intervention.training
        try:
            self.ckpt.reft_model.model.eval()
            intervention.eval()
            if intervention._learnable_bias_active():
                device = intervention._learn_bias_mu.device
                idx = torch.tensor(
                    self.eval_indices, dtype=torch.long, device=device
                )
                mu_tensor = intervention._learn_bias_mu[idx]
            else:
                device = intervention.rotate_layer.weight.device
                word_ids = torch.tensor(
                    self.eval_indices, dtype=torch.long, device=device
                )
                mu_tensor = intervention._bias_network_mu(word_ids)
            mu = mu_tensor.detach().float().cpu().numpy()
            decodes: List[str] = []
            for start in range(0, len(self.targets), self.batch_size):
                end = min(start + self.batch_size, len(self.targets))
                decodes.extend(
                    generate_texts_multi_batch(
                        self.ckpt.reft_model,
                        self.ckpt.tokenizer,
                        self.ckpt.prompt,
                        list(mu[start:end]),
                        max_new_tokens=self.max_new_tokens,
                        use_sample=False,
                        position=(self.ckpt.saved_cfg or {}).get("position", "l1"),
                        assistant_suffix=self.ckpt.assistant_suffix,
                        from_chat_template=self.ckpt.from_chat_template,
                        intervention_token_id=self.ckpt.intervention_token_id,
                        content_span=self.ckpt.content_span,
                    )
                )
        finally:
            self.ckpt.reft_model.model.train(base_was_training)
            intervention.train(intervention_was_training)

        decode_embeddings = encode_texts_normalized(decodes, task=self.task)
        similarities = np.sum(self.target_embeddings * decode_embeddings, axis=1)
        selection_sims = self._selection_sims(decodes, similarities)
        pos_by_all_idx = {
            all_idx: pos for pos, all_idx in enumerate(self.eval_indices)
        }

        def _slice_for(indices: Sequence[int]) -> Dict[str, Any]:
            local = [pos_by_all_idx[i] for i in indices if i in pos_by_all_idx]
            if not local:
                raise ValueError("eval slice resolved to an empty index set")
            return learned_set_slice_metrics(
                [self.targets[i] for i in local],
                [decodes[i] for i in local],
                [float(similarities[i]) for i in local],
                embed_sim_tau=self.embed_sim_tau,
                stop_threshold_min=self.stop_threshold_min,
                task=self.task,
                selection_metric=self.selection_metric,
                selection_similarities=[float(selection_sims[i]) for i in local],
            )

        metrics = {
            "epoch": epoch,
            "step": int(state.global_step),
            **_slice_for(self.stop_eval_indices),
        }
        if self.eval_groups:
            groups: Dict[str, Dict[str, Any]] = {}
            for name, idxs in self.eval_groups.items():
                groups[name] = _slice_for(idxs)
            metrics["groups"] = groups
        stop_sims = np.asarray(
            [
                float(selection_sims[pos_by_all_idx[i]])
                for i in self.stop_eval_indices
                if i in pos_by_all_idx
            ],
            dtype=np.float64,
        )
        group_slices: Dict[str, tuple[float, float, Sequence[float]]] = {}
        if self.eval_groups:
            for name, idxs in self.eval_groups.items():
                slice_metrics = metrics.get("groups", {}).get(name) or {}
                local = [pos_by_all_idx[i] for i in idxs if i in pos_by_all_idx]
                group_slices[name] = (
                    float(slice_metrics["avg_selection_sim"]),
                    float(slice_metrics["min_selection_sim"]),
                    [float(selection_sims[i]) for i in local],
                )
        should_stop = grouped_stop_triggered(
            global_mean=metrics["avg_selection_sim"],
            global_min=metrics["min_selection_sim"],
            global_sims=stop_sims,
            stop_threshold=self.stop_threshold,
            stop_threshold_min=self.stop_threshold_min,
            stop_threshold_frac=self.stop_threshold_frac,
            group_slices=group_slices or None,
            stop_groups=self.stop_groups or None,
        )
        metrics["early_stopped"] = should_stop
        self.history.append(metrics)
        if self.result_callback is not None:
            self.result_callback({"event": "eval_result", **metrics})
        if self.trainer is not None:
            log_payload = {
                "eval_avg_embed_sim": metrics["avg_embed_sim"],
                "eval_min_embed_sim": metrics["min_embed_sim"],
                "eval_n_recovered": metrics["n_recovered"],
                "eval_embed_sim_gte_tau": metrics["embed_sim_gte_tau"],
                "eval_embed_sim_tau": metrics["embed_sim_tau"],
                "eval_selection_sim": metrics["avg_selection_sim"],
                "eval_selection_sim_min": metrics["min_selection_sim"],
            }
            if "embed_sim_gte_stop_min" in metrics:
                log_payload["eval_embed_sim_gte_stop_min"] = metrics[
                    "embed_sim_gte_stop_min"
                ]
            if "selection_gte_stop_min" in metrics:
                log_payload["eval_selection_gte_stop_min"] = metrics[
                    "selection_gte_stop_min"
                ]
            self.trainer.log(log_payload)
        if should_stop:
            control.should_training_stop = True
        return control


def sorted_greedy_decode_rows(
    targets: Sequence[str],
    decodes: Sequence[str],
    similarities: Sequence[float],
    *,
    task: str = "semantle",
) -> List[Dict[str, Any]]:
    """Per-target greedy decodes sorted by descending embedding similarity."""
    norm = target_normalizer(task)
    sorted_rows = sorted(
        zip(targets, decodes, similarities),
        key=lambda row: float(row[2]),
        reverse=True,
    )
    return [
        {
            "rank": rank,
            "target": target,
            "greedy_decode": decode,
            "embed_similarity": float(similarity),
            "exact_match": norm(target) == norm(decode),
        }
        for rank, (target, decode, similarity) in enumerate(sorted_rows, start=1)
    ]


class _BatchLearnProgressCallback(TrainerCallback):
    """Forward trainer logs to non-terminal clients such as the visualization UI."""

    def __init__(self, callback: Callable[[Dict[str, Any]], None]):
        self.callback = callback

    def on_train_begin(self, args, state, control, **kwargs):
        self.callback(
            {
                "event": "train_begin",
                "step": int(state.global_step),
                "max_steps": int(state.max_steps),
                "epoch": float(state.epoch or 0.0),
            }
        )
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        payload: Dict[str, Any] = {
            "event": "log",
            "step": int(state.global_step),
            "max_steps": int(state.max_steps),
            "epoch": float(state.epoch or 0.0),
        }
        for key, value in (logs or {}).items():
            if isinstance(value, (int, float)):
                payload[key] = float(value)
        self.callback(payload)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.callback(
            {
                "event": "train_end",
                "step": int(state.global_step),
                "max_steps": int(state.max_steps),
                "epoch": float(state.epoch or 0.0),
            }
        )
        return control


class BatchLearnControl:
    """Thread-safe cooperative pause and stop signals for batched learning."""

    def __init__(self) -> None:
        self._pause = threading.Event()
        self._stop = threading.Event()

    @property
    def paused(self) -> bool:
        return self._pause.is_set()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def stop(self) -> None:
        self._stop.set()
        self._pause.clear()


class _BatchLearnControlCallback(TrainerCallback):
    """Apply pause/stop requests at optimizer-step boundaries."""

    def __init__(
        self,
        training_control: BatchLearnControl,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    ):
        self.training_control = training_control
        self.progress_callback = progress_callback

    def _report(self, event: str, state) -> None:
        if self.progress_callback is not None:
            self.progress_callback(
                {
                    "event": event,
                    "step": int(state.global_step),
                    "max_steps": int(state.max_steps),
                    "epoch": float(state.epoch or 0.0),
                }
            )

    def on_step_begin(self, args, state, control, **kwargs):
        reported_pause = False
        while (
            self.training_control.paused
            and not self.training_control.stop_requested
        ):
            if not reported_pause:
                self._report("paused", state)
                reported_pause = True
            time.sleep(0.2)
        if reported_pause and not self.training_control.stop_requested:
            self._report("resumed", state)
        if self.training_control.stop_requested:
            self._report("stopping", state)
            control.should_training_stop = True
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.training_control.stop_requested:
            self._report("stopping", state)
            control.should_training_stop = True
        return control


def _install_batch_bias_network_inputs(
    ckpt,
    intervention,
    targets: List[str],
    definitions: Optional[Dict[str, str]],
) -> Callable[[], None]:
    """Install target-aligned bias-network inputs and return a restore callback."""
    saved_cfg = ckpt.saved_cfg or {}
    device = intervention.rotate_layer.weight.device
    if definitions:
        from boreft.text_similarity import (
            definition_text_for_cfg,
            rdkit_definition_lookup_for_cfg,
        )

        rdkit_lookup = rdkit_definition_lookup_for_cfg(saved_cfg)
        definitions = {
            target: definition_text_for_cfg(
                saved_cfg,
                target,
                definition,
                rdkit_lookup=rdkit_lookup,
            )
            for target, definition in definitions.items()
        }

    if getattr(intervention, "bias_input_source", "embed_cache") == "llm_encoder":
        from boreft.pyreft.semantic_encoder import (
            capture_penultimate,
            encode_definition_inputs,
            pooling_uses_instruction_mask,
        )
        from boreft.task_config import definition_embedding_text

        if not definitions or any(not definitions.get(word) for word in targets):
            raise ValueError(
                "llm_encoder bias-network training requires a definition "
                "for every target"
            )
        texts = [
            definition_embedding_text(
                str(saved_cfg.get("task", "semantle")),
                word,
                definitions[word],
            )
            for word in targets
        ]
        pairs = [(word, definitions[word]) for word in targets]
        encoder = intervention.get_semantic_encoder()
        build_instruction_mask = pooling_uses_instruction_mask(encoder.pooling)
        input_ids, attn, instruction_mask = encode_definition_inputs(
            ckpt.tokenizer,
            texts,
            device=device,
            max_length=int(saved_cfg.get("bias_encoder_max_length", 64)),
            task=str(saved_cfg.get("task", "semantle")),
            word_definition_pairs=pairs,
            build_instruction_mask=build_instruction_mask,
        )
        chunks = []
        for start in range(0, len(targets), 64):
            end = min(start + 64, len(targets))
            chunks.append(
                capture_penultimate(
                    ckpt.reft_model.model,
                    input_ids[start:end],
                    attn[start:end],
                    layer_index=saved_cfg.get("bias_encoder_layer_index"),
                )
            )
        penult = torch.cat(chunks, dim=0)
        names = (
            "encoder_penult",
            "encoder_attn",
            "encoder_instruction_mask",
        )
        previous = {
            name: getattr(intervention, name)
            for name in names
            if hasattr(intervention, name)
        }
        intervention.set_encoder_inputs(penult, attn, instruction_mask)

        def restore() -> None:
            for name in names:
                if hasattr(intervention, name):
                    delattr(intervention, name)
            for name, value in previous.items():
                if isinstance(value, torch.Tensor):
                    intervention.register_buffer(
                        name, value, persistent=False
                    )
                else:
                    setattr(intervention, name, value)

        return restore

    from boreft.task_config import definition_embedding_text
    from boreft.text_similarity import (
        bias_network_embed_model_from_cfg,
        definition_lookup_for_cfg,
        encode_reference_embeddings,
    )

    definition_lookup = definition_lookup_for_cfg(saved_cfg)
    if saved_cfg.get("use_definition_embeds"):
        definition_lookup = dict(definition_lookup or {})
        for word in targets:
            if definitions and definitions.get(word):
                definition_lookup[word] = definition_embedding_text(
                    str(saved_cfg.get("task", "semantle")),
                    word,
                    definitions[word],
                )
    features = encode_reference_embeddings(
        targets,
        definition_lookup=definition_lookup,
        task=str(saved_cfg.get("task", "semantle")),
        model_name=bias_network_embed_model_from_cfg(saved_cfg),
    )
    previous_embed_cache = getattr(intervention, "embed_cache", None)
    intervention.embed_cache = torch.as_tensor(
        features, dtype=torch.float32, device=device
    )

    def restore() -> None:
        intervention.embed_cache = previous_embed_cache

    return restore


def predict_bias_network_rows(
    ckpt,
    targets: List[str],
    definitions: Optional[Dict[str, str]] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Run arbitrary targets through the live bias network without training."""
    intervention = _get_intervention(ckpt.reft_model)
    if getattr(intervention, "bias_network", None) is None:
        raise ValueError(
            "predict_bias_network_rows requires an --add-bias-network checkpoint"
        )
    target_list = [target.strip() for target in targets]
    if not target_list:
        return np.zeros((0, 0), dtype=np.float32), None

    restore_inputs = _install_batch_bias_network_inputs(
        ckpt, intervention, target_list, definitions
    )
    materialized_tables: Dict[str, torch.Tensor] = {}
    for name in ("materialized_mu", "materialized_logvar"):
        if hasattr(intervention, name):
            materialized_tables[name] = getattr(intervention, name)
            delattr(intervention, name)

    was_training = intervention.training
    try:
        intervention.eval()
        device = intervention.rotate_layer.weight.device
        word_ids = torch.arange(
            len(target_list), dtype=torch.long, device=device
        )
        with torch.no_grad():
            if (
                getattr(intervention, "fixed_logvar", None) is None
                and getattr(
                    intervention.bias_network, "learnable_logvar", False
                )
            ):
                mu, logvar = intervention._bias_network_mu_logvar(word_ids)
                logvar_out = logvar.detach().float().cpu().numpy()
            else:
                mu = intervention._bias_network_mu(word_ids)
                logvar_out = None
            return mu.detach().float().cpu().numpy(), logvar_out
    finally:
        intervention.train(was_training)
        restore_inputs()
        for name, value in materialized_tables.items():
            intervention.register_buffer(name, value)


def learn_biases_batched(
    ckpt,
    *,
    targets: List[str],
    warm_start_mu: np.ndarray,
    definitions: Optional[Dict[str, str]] = None,
    config: Optional[BatchLearnConfig] = None,
    show_progress: bool = True,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    eval_result_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    training_control: Optional[BatchLearnControl] = None,
    eval_indices: Optional[List[int]] = None,
    eval_groups: Optional[Dict[str, Sequence[int]]] = None,
    stop_eval_indices: Optional[List[int]] = None,
    stop_groups: Optional[Dict[str, GroupStopSpec]] = None,
) -> BatchLearnResult:
    """Learn bias vectors for many targets at once via the main training loop.

    Bias provenance is controlled by ``learn_bias_network`` alone: when it is set,
    rows flow through the live bias network (which becomes trainable). Otherwise a
    trainable ``[N, low_rank]`` bias table is warm-started from ``warm_start_mu``
    and used directly — ``learn_W`` / ``learn_R`` then additionally unfreeze the
    shared rotation ``R`` / source ``W`` while biases still come from the direct
    table. The exact main training recipe (weighted CE / VAE KL+L2 / SDPO with
    shared-b) is reused. ``definitions`` is required for SDPO and LLM-encoder
    bias-network checkpoints.

    ``eval_indices`` restricts the during-training :class:`_LearnedSetEvalCallback`
    to a subset of ``targets`` (row indices); ``None`` evaluates all of them (or
    the union of ``eval_groups`` when provided). ``eval_groups`` optionally
    partitions those rows into named metric slices (e.g. previous / new /
    combined). ``stop_eval_indices`` selects which slice drives the global early
    stop and the top-level eval scalars; ``stop_groups`` optionally AND-combines
    per-group bars (each name must appear in ``eval_groups``).
    ``config.eval_selection_metric`` selects which similarity metric those
    decisions read.

    Returns the learned means (and logvar) aligned to ``targets``.
    """
    from transformers import TrainingArguments

    from boreft.data_utils import infer_torch_dtype_name, trainer_amp_flags
    from boreft.pyreft.reft_trainer import ReftTrainerForCausalLM
    from boreft.train import _build_lr_scheduler, _build_optimizer

    config = config or BatchLearnConfig()
    targets = [t.strip() for t in targets]
    if not targets:
        raise ValueError("learn_biases_batched requires a non-empty targets list")

    saved_cfg = ckpt.saved_cfg or {}
    intervention = _get_intervention(ckpt.reft_model)
    if getattr(intervention, "bias_network", None) is None:
        raise ValueError("learn_biases_batched requires an --add-bias-network checkpoint")

    warm = np.asarray(warm_start_mu, dtype=np.float32)
    if warm.ndim != 2 or warm.shape[0] != len(targets):
        raise ValueError("warm_start_mu must have shape [len(targets), low_rank]")
    bias_dim = int(warm.shape[1])

    loss_cfg = _loss_config_from_ckpt(saved_cfg)
    lambda_ce = float(
        config.lambda_ce if config.lambda_ce is not None else loss_cfg["lambda_ce"]
    )
    if lambda_ce < 0:
        raise ValueError("lambda_ce must be >= 0")
    lambda_sdpo = float(
        config.lambda_sdpo
        if config.lambda_sdpo is not None
        else loss_cfg["lambda_sdpo"]
    )
    if lambda_sdpo < 0:
        raise ValueError("lambda_sdpo must be >= 0")
    loss_cfg = {
        **loss_cfg,
        "lambda_ce": lambda_ce,
        "lambda_sdpo": lambda_sdpo,
    }
    is_vae = loss_cfg["bias_type"] == "vae"
    has_learnable_logvar = (
        is_vae and getattr(intervention, "fixed_logvar", None) is None
    )
    # Bias provenance is decoupled from the shared-component flags: only
    # ``learn_bias_network`` routes biases through the live network. ``learn_W`` /
    # ``learn_R`` unfreeze the shared rotation/source while biases still come from
    # the direct learnable table (warm-started from ``warm_start_mu``).
    through_bias_network = bool(config.learn_bias_network)
    device = ckpt.reft_model.get_device()

    sdpo_cfg = _sdpo_cfg_from_ckpt(ckpt, lambda_sdpo=lambda_sdpo)
    sdpo_active = sdpo_cfg is not None and lambda_sdpo > 0.0
    sdpo_kwargs: Dict[str, Any] = {}
    if sdpo_active:
        if not definitions or any(w not in definitions for w in targets):
            raise ValueError(
                "SDPO checkpoint requires a definition for every target "
                "(pass definitions={word: raw_definition})"
            )
        sdpo_kwargs = _build_batch_sdpo_kwargs(
            ckpt, targets, definitions, sdpo_cfg, lambda_sdpo=lambda_sdpo
        )

    epochs = int(config.epochs if config.epochs is not None else saved_cfg.get("epochs", 50))
    batch_size = int(
        config.batch_size
        if config.batch_size is not None
        else saved_cfg.get("batch_size", 32)
    )
    grad_acc_steps = int(
        config.grad_acc_steps
        if config.grad_acc_steps is not None
        else saved_cfg.get("grad_acc_steps", 1)
    )
    if grad_acc_steps < 1:
        raise ValueError("grad_acc_steps must be a positive integer")
    lr = float(config.lr if config.lr is not None else saved_cfg.get("lr", 1e-2))
    seed = int(config.seed if config.seed is not None else saved_cfg.get("seed", 42))
    max_grad_norm = float(
        config.max_grad_norm
        if config.max_grad_norm is not None
        else saved_cfg.get("max_grad_norm", 1.0)
    )
    lr_scheduler_type = str(
        config.lr_scheduler_type
        if config.lr_scheduler_type is not None
        else saved_cfg.get("lr_scheduler_type", "linear")
    )
    warmup_ratio = float(
        config.warmup_ratio
        if config.warmup_ratio is not None
        else saved_cfg.get("warmup_ratio", 0.0)
    )
    weight_decay_mode = str(
        config.weight_decay_mode
        if config.weight_decay_mode is not None
        else saved_cfg.get("weight_decay_mode", "none")
    )
    wd_W = float(config.wd_W if config.wd_W is not None else saved_cfg.get("wd_W", 0.0))
    wd_b = float(config.wd_b if config.wd_b is not None else saved_cfg.get("wd_b", 0.0))
    nbr_lambda = float(saved_cfg.get("nbr_lambda", 0.0))
    nbr_top_k = int(saved_cfg.get("nbr_top_k", 0))
    aug_batch_mode = str(saved_cfg.get("aug_batch_mode", "shuffle"))
    dtype_name = infer_torch_dtype_name(
        override=config.torch_dtype, saved_cfg=saved_cfg
    )
    train_bf16, train_fp16 = trainer_amp_flags(
        dtype_name, use_cuda=torch.cuda.is_available()
    )
    eval_epochs_raw = (
        config.eval_epochs
        if config.eval_epochs is not None
        else saved_cfg.get("eval_epochs")
    )
    eval_epochs = int(eval_epochs_raw) if eval_epochs_raw else None
    if eval_epochs is not None and eval_epochs <= 0:
        raise ValueError("eval_epochs must be positive when enabled")
    eval_batch_size = int(
        config.eval_batch_size
        if config.eval_batch_size is not None
        else saved_cfg.get("full_eval_batch_size", 64)
    )
    eval_max_new_tokens = int(
        config.eval_max_new_tokens
        if config.eval_max_new_tokens is not None
        else saved_cfg.get("full_eval_max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
    )
    if config.stop_threshold is not None:
        stop_threshold = float(config.stop_threshold)
    elif config.inherit_stop_threshold and saved_cfg.get("stop_threshold") is not None:
        stop_threshold = float(saved_cfg["stop_threshold"])
    else:
        stop_threshold = None
    if config.stop_threshold_min is not None:
        stop_threshold_min = float(config.stop_threshold_min)
    elif (
        config.inherit_stop_threshold
        and saved_cfg.get("stop_threshold_min") is not None
    ):
        stop_threshold_min = float(saved_cfg["stop_threshold_min"])
    else:
        stop_threshold_min = None
    if config.stop_threshold_frac is not None:
        stop_threshold_frac = float(config.stop_threshold_frac)
    elif (
        config.inherit_stop_threshold
        and saved_cfg.get("stop_threshold_frac") is not None
    ):
        stop_threshold_frac = float(saved_cfg["stop_threshold_frac"])
    else:
        stop_threshold_frac = 1.0
    if not (0.0 < stop_threshold_frac <= 1.0):
        raise ValueError("stop_threshold_frac must be in (0, 1]")
    if stop_threshold_frac < 1.0 - 1e-12 and stop_threshold_min is None:
        raise ValueError(
            "stop_threshold_frac < 1 requires stop_threshold_min (per-target tau)"
        )
    if (
        stop_threshold is not None
        or stop_threshold_min is not None
        or stop_groups
    ) and eval_epochs is None:
        raise ValueError(
            "stop_threshold / stop_threshold_min / stop_groups require "
            "eval_epochs to be set"
        )
    embed_sim_tau = float(
        saved_cfg.get("eval_embed_sim_tau", DEFAULT_EMBED_SIM_TAU)
    )
    if not (0.0 <= embed_sim_tau <= 1.0):
        raise ValueError("eval_embed_sim_tau must be in [0, 1]")
    selection_metric = str(
        config.eval_selection_metric
        or saved_cfg.get("eval_selection_metric")
        or "embed_sim"
    )
    if selection_metric not in EVAL_SELECTION_METRICS:
        raise ValueError(
            f"eval_selection_metric must be one of {EVAL_SELECTION_METRICS}, "
            f"got {selection_metric!r}"
        )
    run_task = str(saved_cfg.get("task", "semantle"))
    if selection_metric != "embed_sim" and not task_supports_fingerprints(run_task):
        raise ValueError(
            f"eval_selection_metric={selection_metric!r} requires a task whose "
            f"targets are molecules (task={run_task!r}); only embed_sim is "
            f"available there"
        )

    rows, nbr_block_size = _build_batched_learn_rows(
        ckpt,
        targets,
        nbr_lambda=nbr_lambda,
        nbr_top_k=nbr_top_k,
    )
    dataset = _RowDataset(rows)
    collator = ReftDataCollator(tokenizer=ckpt.tokenizer, model=ckpt.reft_model.model)

    kl_beta = float(
        config.kl_beta if config.kl_beta is not None else loss_cfg["kl_beta"]
    )
    if kl_beta < 0:
        raise ValueError("kl_beta must be >= 0")
    loss_cfg = {**loss_cfg, "kl_beta": kl_beta}
    anneal_map = resolve_batch_linear_annealing_map(
        config,
        saved_cfg,
        kl_beta=kl_beta,
        lambda_ce=lambda_ce,
        lambda_sdpo=lambda_sdpo,
    )
    if show_progress:
        print(
            "[learn-batch] inherited config: "
            f"epochs={epochs} batch_size={batch_size} "
            f"grad_acc_steps={grad_acc_steps} lr={lr:g} "
            f"scheduler={lr_scheduler_type} warmup_ratio={warmup_ratio:g} "
            f"max_grad_norm={max_grad_norm:g} "
            f"weight_decay={weight_decay_mode}(W={wd_W:g},b={wd_b:g}) "
            f"lambda_ce={lambda_ce:g} lambda_sdpo={lambda_sdpo:g} "
            f"kl_beta={kl_beta:g} "
            f"linear_annealing_map={serialize_linear_annealing_map(anneal_map)} "
            f"amp=bf16:{train_bf16}/fp16:{train_fp16} "
            f"nbr=lambda:{nbr_lambda:g}/top_k:{nbr_top_k}/{aug_batch_mode} "
            f"eval_epochs={eval_epochs or 'off'} "
            f"eval_selection_metric={selection_metric} "
            f"learn_W={config.learn_W} learn_R={config.learn_R} "
            f"learn_bias_network={config.learn_bias_network} "
            f"mode={'bias_network' if through_bias_network else 'direct_bias'}",
            flush=True,
        )

    restore_bias_inputs: Optional[Callable[[], None]] = None
    materialized_tables: Dict[str, torch.Tensor] = {}
    if through_bias_network:
        restore_bias_inputs = _install_batch_bias_network_inputs(
            ckpt, intervention, targets, definitions
        )
        for name in ("materialized_mu", "materialized_logvar"):
            if hasattr(intervention, name):
                materialized_tables[name] = getattr(intervention, name)
                delattr(intervention, name)
    original_requires_grad = {
        id(param): param.requires_grad for param in ckpt.reft_model.parameters()
    }
    # Freeze everything. The temporary bias table and explicitly requested shared
    # components are enabled below.
    for param in ckpt.reft_model.parameters():
        param.requires_grad_(False)
    if not through_bias_network:
        mu_init = torch.tensor(warm, dtype=torch.float32, device=device)
        logvar_init = (
            torch.zeros(
                len(targets), bias_dim, dtype=torch.float32, device=device
            )
            if has_learnable_logvar
            else None
        )
        intervention.set_learnable_bias_table(mu_init, logvar_init)
    if config.learn_W:
        for param in intervention.learned_source.parameters():
            param.requires_grad_(True)
    if config.learn_R:
        for param in intervention.rotate_layer.parameters():
            param.requires_grad_(True)
    encoder_trainable_ids: set[int] = set()
    if config.learn_bias_network:
        for param in intervention.bias_network.parameters():
            param.requires_grad_(True)
        encoder = intervention.get_semantic_encoder()
        if encoder is not None:
            for param in encoder.parameters():
                if original_requires_grad.get(id(param), False):
                    param.requires_grad_(True)
                    encoder_trainable_ids.add(id(param))
    shared_snapshots = []
    if config.learn_W:
        shared_snapshots.append(
            (
                intervention.learned_source,
                {
                    key: value.detach().clone()
                    for key, value in intervention.learned_source.state_dict().items()
                },
            )
        )
    if config.learn_R:
        shared_snapshots.append(
            (
                intervention.rotate_layer,
                {
                    key: value.detach().clone()
                    for key, value in intervention.rotate_layer.state_dict().items()
                },
            )
        )
    if config.learn_bias_network:
        shared_snapshots.append(
            (
                intervention.bias_network,
                {
                    key: value.detach().clone()
                    for key, value in intervention.bias_network.state_dict().items()
                },
            )
        )
    encoder_parameter_snapshots = [
        (param, param.detach().clone())
        for param in ckpt.reft_model.parameters()
        if id(param) in encoder_trainable_ids
    ]

    steps = 0
    mean_train_loss = float("nan")
    learned_mu = warm.copy()
    learned_logvar: Optional[np.ndarray] = None
    eval_callback: Optional[_LearnedSetEvalCallback] = None
    succeeded = False
    try:
        with tempfile.TemporaryDirectory() as tmp_out:
            logging_steps = 1_000_000
            if show_progress:
                logging_steps = 10
            elif progress_callback is not None:
                logging_steps = 1
            training_args = TrainingArguments(
                output_dir=tmp_out,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_acc_steps,
                max_grad_norm=max_grad_norm,
                logging_steps=logging_steps,
                save_strategy="no",
                remove_unused_columns=False,
                report_to=[],
                seed=seed,
                disable_tqdm=not show_progress,
                bf16=train_bf16,
                fp16=train_fp16,
            )
            if eval_epochs is not None:
                eval_callback = _LearnedSetEvalCallback(
                    ckpt,
                    targets,
                    eval_epochs=eval_epochs,
                    batch_size=eval_batch_size,
                    max_new_tokens=eval_max_new_tokens,
                    stop_threshold=stop_threshold,
                    stop_threshold_min=stop_threshold_min,
                    stop_threshold_frac=stop_threshold_frac,
                    embed_sim_tau=embed_sim_tau,
                    selection_metric=selection_metric,
                    eval_indices=eval_indices,
                    eval_groups=eval_groups,
                    stop_eval_indices=stop_eval_indices,
                    stop_groups=stop_groups,
                    result_callback=eval_result_callback,
                )
            callbacks: List[TrainerCallback] = []
            if eval_callback is not None:
                callbacks.append(eval_callback)
            if progress_callback is not None:
                callbacks.append(_BatchLearnProgressCallback(progress_callback))
            if training_control is not None:
                callbacks.append(
                    _BatchLearnControlCallback(
                        training_control, progress_callback
                    )
                )
            trainer = ReftTrainerForCausalLM(
                model=ckpt.reft_model,
                tokenizer=ckpt.tokenizer,
                args=training_args,
                train_dataset=dataset,
                data_collator=collator,
                kl_beta=kl_beta,
                linear_annealing_map=anneal_map,
                lambda_ce=lambda_ce,
                use_margin_loss=loss_cfg["use_margin_loss"],
                margin_loss_margin=loss_cfg["margin_loss_margin"],
                callbacks=callbacks or None,
                neighborhood_block_size=nbr_block_size
                if nbr_block_size is not None and aug_batch_mode == "block"
                else None,
                **sdpo_kwargs,
            )
            if eval_callback is not None:
                eval_callback.trainer = trainer
            trainable = [
                p for p in ckpt.reft_model.parameters() if p.requires_grad
            ]
            allowed_ids: set[int] = set()
            if not through_bias_network:
                allowed_ids.add(id(intervention._learn_bias_mu))
                if has_learnable_logvar:
                    allowed_ids.add(id(intervention._learn_bias_logvar))
            if config.learn_W:
                allowed_ids.update(
                    id(param) for param in intervention.learned_source.parameters()
                )
            if config.learn_R:
                allowed_ids.update(
                    id(param) for param in intervention.rotate_layer.parameters()
                )
            if config.learn_bias_network:
                allowed_ids.update(
                    id(param) for param in intervention.bias_network.parameters()
                )
                allowed_ids.update(encoder_trainable_ids)
            trainable_ids = {id(param) for param in trainable}
            if trainable_ids != allowed_ids:
                raise RuntimeError(
                    "unexpected trainable parameter set during batched learning: "
                    f"expected {len(allowed_ids)} tensors, got "
                    f"{len(trainable_ids)}"
                )
            # HF Trainer.state.global_step counts optimizer updates, not micro-batches.
            micro_batches = len(trainer.get_train_dataloader())
            steps_per_epoch = max(1, micro_batches // max(1, grad_acc_steps))
            total_steps = steps_per_epoch * max(1, epochs)
            warmup_steps = int(warmup_ratio * total_steps)
            trainer.steps_per_epoch = steps_per_epoch
            trainer.optimizer = _build_optimizer(
                ckpt.reft_model,
                lr,
                weight_decay_mode,
                wd_W,
                wd_b,
            )
            trainer.lr_scheduler = _build_lr_scheduler(
                trainer.optimizer,
                lr_scheduler_type,
                warmup_steps,
                total_steps,
            )
            try:
                ckpt.reft_model.model.train()
                intervention.train()
                train_out = trainer.train()
                mean_train_loss = float(train_out.training_loss)
                steps = int(trainer.state.global_step)
            finally:
                ckpt.reft_model.model.eval()
                intervention.eval()

        if through_bias_network:
            word_ids = torch.arange(
                len(targets), dtype=torch.long, device=device
            )
            with torch.no_grad():
                if has_learnable_logvar:
                    mu_tensor, logvar_tensor = (
                        intervention._bias_network_mu_logvar(word_ids)
                    )
                    learned_logvar = (
                        logvar_tensor.detach().float().cpu().numpy()
                    )
                else:
                    mu_tensor = intervention._bias_network_mu(word_ids)
                learned_mu = mu_tensor.detach().float().cpu().numpy()
        else:
            learned_mu = (
                intervention._learn_bias_mu.detach().float().cpu().numpy()
            )
            learned_logvar = (
                intervention._learn_bias_logvar.detach().float().cpu().numpy()
                if has_learnable_logvar
                else None
            )
        succeeded = True
    finally:
        # Always remove the table so a failed/aborted fit can never leave the
        # intervention resolving lookups to stale learnable rows.
        intervention.clear_learnable_bias_table()
        if restore_bias_inputs is not None:
            restore_bias_inputs()
        for name, value in materialized_tables.items():
            intervention.register_buffer(name, value)
        if not succeeded:
            for module, state_dict in shared_snapshots:
                module.load_state_dict(state_dict)
            for param, value in encoder_parameter_snapshots:
                param.data.copy_(value)
        for param in ckpt.reft_model.parameters():
            param.requires_grad_(original_requires_grad.get(id(param), False))

    return BatchLearnResult(
        targets=targets,
        mu=learned_mu,
        mu_pred=warm,
        logvar=learned_logvar,
        bias_dim=bias_dim,
        steps=steps,
        mean_train_loss=mean_train_loss,
        loss_config=loss_cfg,
        resolved_train_config={
            "epochs": epochs,
            "batch_size": int(trainer._train_batch_size),
            "grad_acc_steps": grad_acc_steps,
            "lr": lr,
            "seed": seed,
            "max_grad_norm": max_grad_norm,
            "lr_scheduler_type": lr_scheduler_type,
            "warmup_ratio": warmup_ratio,
            "warmup_steps": warmup_steps,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "weight_decay_mode": weight_decay_mode,
            "wd_W": wd_W,
            "wd_b": wd_b,
            "torch_dtype": dtype_name,
            "train_bf16": train_bf16,
            "train_fp16": train_fp16,
            "kl_beta": kl_beta,
            "linear_annealing_map": serialize_linear_annealing_map(anneal_map),
            "lambda_ce": lambda_ce,
            "lambda_sdpo": lambda_sdpo,
            "nbr_lambda": nbr_lambda,
            "nbr_top_k": nbr_top_k,
            "aug_batch_mode": aug_batch_mode,
            "num_training_examples": len(rows),
            "eval_epochs": eval_epochs,
            "eval_batch_size": eval_batch_size,
            "eval_max_new_tokens": eval_max_new_tokens,
            "stop_threshold": stop_threshold,
            "stop_threshold_min": stop_threshold_min,
            "stop_threshold_frac": stop_threshold_frac,
            "eval_embed_sim_tau": embed_sim_tau,
            "eval_selection_metric": selection_metric,
            "learn_W": config.learn_W,
            "learn_R": config.learn_R,
            "learn_bias_network": config.learn_bias_network,
            "bias_learning_mode": (
                "bias_network" if through_bias_network else "direct_bias"
            ),
        },
        eval_history=eval_callback.history if eval_callback is not None else [],
        stopped_early=batch_learn_stopped_early(
            eval_history=(
                eval_callback.history if eval_callback is not None else []
            ),
            training_control=training_control,
        ),
    )
