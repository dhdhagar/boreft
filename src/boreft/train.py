#!/usr/bin/env python3
"""
REFT training. Run: python -m boreft.train --task semantle --semantle-csv path.csv ...
"""

import gc
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
)
import wandb


from boreft.pyreft import (
    DEFAULT_LORA_TARGETS,
    DistributionalWordIntervention,
    EMBED_LAYER,
    LoreftPerWordBiasIntervention,
    ReftConfig,
    ReftTrainerForCausalLM,
    build_reft_representation,
    build_semantic_encoder,
    capture_penultimate,
    encode_definition_inputs,
    get_reft_model,
    intervention_component,
)
from boreft.pyreft.semantic_encoder import (
    pooling_requires_instruction_mask,
    pooling_uses_instruction_mask,
    resolve_encoder_layer_index,
)
from boreft.data import ArcItem, HypoGenItem, MolOptItem, ReftItem, SemantleItem
from boreft.train_args import (
    TrainConfig,
    parse_train_config,
    resolved_linear_annealing_map,
)
from boreft.pyreft.losses import serialize_linear_annealing_map
from boreft.chem import (
    RDKIT_DESCRIPTOR_SCHEMA_VERSION,
    file_sha256,
    load_rdkit_descriptor_map,
    maybe_append_mist_smiles_open_tag,
    mist_open_tag_in_prompt,
)
from boreft.text_similarity import (
    default_definitions_path,
    default_rdkit_definitions_map_path,
    default_rdkit_definitions_path,
    definition_text_for_cfg,
    embedding_model_name,
    embedding_provenance,
    load_rdkit_definition_lookup,
    load_raw_definitions,
    training_cache_params,
)
from boreft.task_config import definition_embedding_text
from boreft.embed_cache import (
    embed_texts_from_items,
    resolve_or_build_embed_cache,
)
from boreft.data_utils import (
    ReftDataCollator,
    ReftDataset,
    TRAINING_CONFIG_NAME,
    apply_chat_format,
    sample_reft_items,
    reft_items_to_json,
    semantle_csv_abspaths,
    EpochIntervalCheckpoint,
    resolve_torch_dtype,
    infer_torch_dtype_name,
    infer_training_load_dtype_name,
    trainer_amp_flags,
    init_learnable_parameters_from_run,
    has_best_checkpoint,
    latest_checkpoint_dir,
    load_intervention_config,
    promote_best_checkpoint_to_root,
    save_reft_checkpoint_dir,
)
from boreft.task_config import task_instruction, task_system_prompt
from boreft.intervention_marker import (
    init_intervention_token_embedding,
    instruction_content_span,
    is_content_position,
    is_marker_position,
    prepare_instruction,
    resolve_intervention_token,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _write_json_config(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _seed_run_root_metadata(
    output_dir: str,
    *,
    tokenizer,
    items,
    intervention_config: Dict,
) -> None:
    """Write tokenizer / items / intervention config at the run root (no weights)."""
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "items.json"), "w", encoding="utf-8") as f:
        json.dump(reft_items_to_json(items), f, indent=2)
    _write_json_config(
        os.path.join(output_dir, "intervention_config.json"),
        intervention_config,
    )


def _finalize_training_checkpoint(
    *,
    output_dir: str,
    reft_model,
    tokenizer,
    items,
    intervention_config: Dict,
    save_best_params: bool,
) -> Dict:
    """Write latest and promote best-to-root (or save final to root).

    When ``save_best_params`` and a ``best/`` tree exist: write current weights
    to ``latest/``, then copy best weights into the run root. Otherwise write
    current weights to the root only (legacy layout).
    """
    os.makedirs(output_dir, exist_ok=True)
    train_cfg_src = os.path.join(output_dir, TRAINING_CONFIG_NAME)
    train_cfg_arg = train_cfg_src if os.path.isfile(train_cfg_src) else None
    use_best = bool(save_best_params) and has_best_checkpoint(output_dir)
    if use_best:
        latest_dir = latest_checkpoint_dir(output_dir)
        save_reft_checkpoint_dir(
            latest_dir,
            reft_model=reft_model,
            tokenizer=tokenizer,
            items=items,
            intervention_config=intervention_config,
            training_config_src=train_cfg_arg,
        )
        _seed_run_root_metadata(
            output_dir,
            tokenizer=tokenizer,
            items=items,
            intervention_config=intervention_config,
        )
        info = promote_best_checkpoint_to_root(output_dir)
        if info is not None:
            print(
                f"[train] promoted best {info.get('selection_metric', 'metric')} "
                f"mean={info.get('mean_sim_at_save')} "
                f"(step={info.get('global_step')}) -> {output_dir}",
                flush=True,
            )
        print(f"[train] latest params saved to {latest_dir}", flush=True)
        return load_intervention_config(output_dir) or dict(intervention_config)

    return save_reft_checkpoint_dir(
        output_dir,
        reft_model=reft_model,
        tokenizer=tokenizer,
        items=items,
        intervention_config=intervention_config,
        training_config_src=train_cfg_arg,
    )

def _build_training_config_dict(
    config: TrainConfig,
    *,
    batch_size: int,
    dtype_name: str,
    num_anchors: int,
    num_training_examples: int,
    semantle_csv_paths: Optional[List[str]] = None,
    chat_instruction_resolved: Optional[str] = None,
    system_prompt_resolved: Optional[str] = None,
    train_bf16: bool = False,
    train_fp16: bool = False,
    steps_per_epoch: Optional[int] = None,
    total_steps: Optional[int] = None,
    warmup_steps: Optional[int] = None,
) -> Dict:
    """Training-loop hyperparameters and run metadata (not intervention weights).

    ``num_anchors`` is the anchor vocabulary size (``len(items)``).
    ``num_training_examples`` is the dataset length passed to the trainer
    (equals ``num_anchors`` unless neighborhood augmentation expands items).
    """
    from boreft.molopt_split import oracle_split_for_config

    payload: Dict = {
        "task": config.task,
        "model_name": config.model_name,
        "model": os.path.basename(config.model_name),
        "cache_dir": config.cache_dir,
        "low_rank_dim": config.low_rank_dim,
        "layer": config.layer,
        "position": config.position,
        "sigma_b": config.sigma_b,
        "epochs": config.epochs,
        "batch_size": batch_size,
        "grad_acc_steps": config.grad_acc_steps,
        "lr": config.lr,
        "lr_scheduler_type": config.lr_scheduler_type,
        "warmup_ratio": config.warmup_ratio,
        "max_grad_norm": config.max_grad_norm,
        "seed": config.seed,
        "init_from": (
            os.path.abspath(config.init_from) if config.init_from else None
        ),
        "num_anchors": num_anchors,
        "num_training_examples": num_training_examples,
        "train_n_samples": config.train_n_samples,
        "train_top_k": config.train_top_k,
        "semantle_csv": semantle_csv_paths,
        # Absolute so standalone eval can rebuild the held-out pool from any cwd.
        "molopt_csv": (
            os.path.abspath(config.molopt_csv) if config.molopt_csv else None
        ),
        "molopt_oracle_cap_percentile": config.molopt_oracle_cap_percentile,
        "molopt_oracle_scores_path": config.molopt_oracle_scores_path,
        "full_eval_oracle_screen": config.full_eval_oracle_screen,
        "full_eval_oracle_screen_n_sobol": config.full_eval_oracle_screen_n_sobol,
        "oracle_split": oracle_split_for_config(getattr(config, "_oracle_split", None)),
        "hypogen_csv": (
            os.path.abspath(config.hypogen_csv) if config.hypogen_csv else None
        ),
        "arc_dir": config.arc_dir,
        "arc_task_filter": config.arc_task_filter,
        "eval_steps": config.eval_steps,
        "eval_epochs": config.eval_epochs,
        "eval_n_samples": config.eval_n_samples,
        "eval_selection_metric": config.eval_selection_metric,
        "stop_threshold": config.stop_threshold,
        "stop_threshold_min": config.stop_threshold_min,
        "stop_threshold_frac": config.stop_threshold_frac,
        "checkpoint_embed_sim_thresholds": config.checkpoint_embed_sim_thresholds_list,
        "checkpoint_embed_sim_thresholds_min": config.checkpoint_embed_sim_thresholds_min_list,
        "checkpoint_epoch_interval": config.checkpoint_epoch_interval,
        "save_best_params": config.save_best_params,
        "run_full_eval": config.run_full_eval,
        "full_eval_n_samples": config.full_eval_n_samples,
        "full_eval_gen_samples": config.full_eval_gen_samples,
        "full_eval_top_p": config.full_eval_top_p,
        "full_eval_interp": config.full_eval_interp_list,
        "full_eval_n_uniform": config.full_eval_n_uniform,
        "full_eval_batch_size": config.full_eval_batch_size,
        "full_eval_interp_n_samples": config.full_eval_interp_n_samples,
        "full_eval_interp_t_steps": config.full_eval_interp_t_steps,
        "interp_method": config.interp_method,
        "full_eval_max_new_tokens": config.full_eval_max_new_tokens,
        "eval_embed_sim_tau": config.eval_embed_sim_tau,
        "test_n_samples": config.test_n_samples,
        "eval_bbox_pca_var": config.eval_bbox_pca_var,
        "semantle_dir": config.semantle_dir,
        "nbr_lambda": config.nbr_lambda,
        "nbr_top_k": config.nbr_top_k,
        "aug_batch_mode": config.aug_batch_mode,
        "lambda_ce": config.lambda_ce,
        "use_margin_loss": config.use_margin_loss,
        "margin_loss_margin": config.margin_loss_margin,
        "lambda_sdpo": config.lambda_sdpo,
        "linear_annealing_map": serialize_linear_annealing_map(
            resolved_linear_annealing_map(config)
        ),
        "linear_annealing_map_raw": config.linear_annealing_map,
        "sdpo_divergence": config.sdpo_divergence,
        "sdpo_temperature": config.sdpo_temperature,
        "sdpo_sample_temperature": config.sdpo_sample_temperature,
        "sdpo_sample_top_p": config.sdpo_sample_top_p,
        "sdpo_max_new_tokens": config.sdpo_max_new_tokens,
        "sdpo_n_onpolicy": config.sdpo_n_onpolicy,
        "sdpo_n_offpolicy": config.sdpo_n_offpolicy,
        "sdpo_offpolicy_pool": config.sdpo_offpolicy_pool,
        "sdpo_include_gold": config.sdpo_include_gold,
        "sdpo_definitions_path": config.sdpo_definitions_path,
        "weight_decay_mode": config.weight_decay_mode,
        "wd_W": config.wd_W,
        "wd_b": config.wd_b,
        "use_chat_template": config.use_chat_template,
        "chat_instruction": chat_instruction_resolved if config.use_chat_template else None,
        "system_prompt": (
            (
                system_prompt_resolved
                if system_prompt_resolved is not None
                else task_system_prompt(config.task)
            )
            if config.use_chat_template
            else None
        ),
        "bias_network_encoder": config.bias_network_encoder,
        "bias_network_embed_model": config.bias_network_embed_model,
        "use_definition_embeds": config.use_definition_embeds,
        "append_rdkit_definitions": config.append_rdkit_definitions,
        "omit_molt5_definitions": config.omit_molt5_definitions,
        "smiles_tags": config.smiles_tags,
        "mist_smiles_tags": config.mist_smiles_tags,
        "rdkit_definitions_path": (
            os.path.abspath(
                config.rdkit_definitions_path or default_rdkit_definitions_path()
            )
            if config.append_rdkit_definitions
            else ""
        ),
        "rdkit_definitions_sha256": (
            file_sha256(
                os.path.abspath(
                    config.rdkit_definitions_path
                    or default_rdkit_definitions_path()
                )
            )
            if config.append_rdkit_definitions
            else ""
        ),
        "rdkit_definitions_map_path": (
            os.path.abspath(
                config.rdkit_definitions_map_path
                or default_rdkit_definitions_map_path()
            )
            if config.task == "molopt"
            else ""
        ),
        "rdkit_definitions_map_sha256": (
            file_sha256(
                os.path.abspath(
                    config.rdkit_definitions_map_path
                    or default_rdkit_definitions_map_path()
                )
            )
            if config.task == "molopt"
            else ""
        ),
        "rdkit_descriptor_schema_version": (
            RDKIT_DESCRIPTOR_SCHEMA_VERSION
            if config.task == "molopt"
            else None
        ),
        "definitions_path": (
            os.path.abspath(
                config.definitions_path or default_definitions_path(config.task)
            )
            if config.use_definition_embeds
            else ""
        ),
        "torch_dtype": dtype_name,
        "torch_dtype_override": config.torch_dtype,
        "train_bf16": train_bf16,
        "train_fp16": train_fp16,
        "wandb_project": config.wandb_project,
        "wandb_entity": config.wandb_entity,
        "wandb_run_name": config.wandb_run_name,
        "wandb_group": config.wandb_group,
        "wandb_dir": config.wandb_dir,
        "cli_args": sys.argv[1:],
    }
    if steps_per_epoch is not None:
        payload["steps_per_epoch"] = steps_per_epoch
        payload["total_steps"] = total_steps
        payload["warmup_steps"] = warmup_steps
    return payload


def _wandb_config_dict(
    training_config: Dict,
    intervention_config: Dict,
) -> Dict:
    """W&B run config: training artifact plus intervention fields used for UI filtering."""
    return {
        **training_config,
        "size": training_config.get(
            "num_anchors", training_config.get("num_items")
        ),
        "lr_scheduler": training_config["lr_scheduler_type"],
        "layer": intervention_config["layer"],
        "position": intervention_config["position"],
        "bias_type": intervention_config["bias_type"],
        "use_word_bias": intervention_config["use_word_bias"],
        "variance": intervention_config["variance"],
        "dropout": intervention_config["dropout"],
        "dropout_on_b": intervention_config["dropout_on_b"],
        "add_bias_network": intervention_config["add_bias_network"],
        "bias_network_residual": intervention_config["bias_network_residual"],
        "bias_materialized": intervention_config.get("bias_materialized", False),
        "bias_tables_path": intervention_config.get("bias_tables_path"),
        "kl_beta": intervention_config["kl_beta"],
        "linear_annealing_map": intervention_config.get("linear_annealing_map"),
        "kl_prior_var": intervention_config["kl_prior_var"],
        "vae_free_bits_lambda": intervention_config["vae_free_bits_lambda"],
        "lambda_l2": intervention_config["lambda_l2"],
        "lambda_mm": intervention_config["lambda_mm"],
        "mm_topk_neighbors": intervention_config["mm_topk_neighbors"],
        "mm_anchor_samples": intervention_config["mm_anchor_samples"],
        "lambda_rank": intervention_config["lambda_rank"],
        "rank_temperature": intervention_config["rank_temperature"],
        "rank_min_ref_gap": intervention_config["rank_min_ref_gap"],
        "rank_topk_neighbors": intervention_config["rank_topk_neighbors"],
        "rank_anchor_samples": intervention_config["rank_anchor_samples"],
        "checkpoint_thresholds": training_config["checkpoint_embed_sim_thresholds"],
        "checkpoint_thresholds_min": training_config["checkpoint_embed_sim_thresholds_min"],
    }


def _run_post_training_full_eval(config: TrainConfig, *, output_dir: str) -> bool:
    """Run the post-training eval pipeline in-process (resumes W&B from output_dir).

    Returns True when all eval steps succeed. The checkpoint is already saved
    before this runs; failures are recorded in ``eval/eval_status.json``.
    """
    from boreft.eval.run_full_eval import FullEvalConfig, run_full_eval

    cfg_kwargs: Dict = dict(
        output_dir=output_dir,
        model_name=config.model_name,
        cache_dir=config.cache_dir,
        layer=config.layer,
        low_rank_dim=config.low_rank_dim,
        seed=config.seed,
        position=config.position,
        semantle_dir=config.semantle_dir,
        full_eval_interp=config.full_eval_interp_list,
        wandb_project=config.wandb_project,
        wandb_entity=config.wandb_entity,
        wandb_run_name=config.wandb_run_name,
        wandb_group=config.wandb_group,
        wandb_dir=config.wandb_dir,
        top_k=config.train_top_k,
        full_eval_n_samples=config.full_eval_n_samples,
        n_samples=config.full_eval_gen_samples,
        top_p=config.full_eval_top_p,
        n_uniform=config.full_eval_n_uniform,
        eval_batch_size=config.full_eval_batch_size,
        oracle_screen=config.full_eval_oracle_screen,
        oracle_screen_n_sobol=config.full_eval_oracle_screen_n_sobol,
        interp_n_samples=config.full_eval_interp_n_samples,
        interp_t_steps=config.full_eval_interp_t_steps,
        interp_method=config.interp_method,
        max_new_tokens=config.full_eval_max_new_tokens,
        torch_dtype=config.torch_dtype,
        embed_sim_tau=config.eval_embed_sim_tau,
        test_n_samples=config.test_n_samples,
        bbox_pca_var=config.eval_bbox_pca_var,
    )

    print(
        f"[train] Running post-training full eval (in-process) on {output_dir}",
        flush=True,
    )
    results = run_full_eval(FullEvalConfig(**cfg_kwargs))
    success = bool(results.get("meta", {}).get("success", False))
    if not success:
        failed = results.get("meta", {}).get("failed_steps", [])
        print(
            f"[train] WARNING: post-training full eval failed (steps: {failed})",
            flush=True,
        )
    return success


def _build_sdpo_trainer_kwargs(
    config: TrainConfig,
    items: List[ReftItem],
    tokenizer,
    base_model,
    *,
    intervention_token_id: Optional[int],
    content_span: Optional[Tuple[int, int]],
) -> Dict:
    """Precompute SDPO teacher/student prompt ids and build trainer kwargs.

    Returns an empty dict when SDPO is disabled. Otherwise returns the kwargs
    consumed by :class:`ReftTrainerForCausalLM` (lambda, config, base model, and
    per-word-id tokenized prompts/targets).
    """
    if config.lambda_sdpo <= 0:
        return {}

    from boreft.task_config import sdpo_teacher_instruction
    from boreft.text_similarity import load_raw_definitions
    from boreft.data_utils import chat_prompt, tokenize_model_text
    from boreft.pyreft.sdpo import SDPOConfig

    def_path = os.path.abspath(
        config.sdpo_definitions_path
        or config.definitions_path
        or default_definitions_path(config.task)
    )
    raw_defs = load_raw_definitions(def_path)
    rdkit_lookup = (
        load_rdkit_definition_lookup(
            os.path.abspath(
                config.rdkit_definitions_path or default_rdkit_definitions_path()
            )
        )
        if config.append_rdkit_definitions
        else None
    )

    def _word(it: ReftItem) -> str:
        return (getattr(it, "_raw_word", None) or it.target).strip()

    missing = sorted({_word(it) for it in items if _word(it) not in raw_defs})
    if missing:
        raise ValueError(
            f"[sdpo] {len(missing)} training words missing definitions in "
            f"{def_path} (e.g. {missing[:5]})"
        )
    if rdkit_lookup is not None:
        missing_rdkit = sorted(
            {_word(it) for it in items if _word(it) not in rdkit_lookup}
        )
        if missing_rdkit:
            raise ValueError(
                f"[sdpo] {len(missing_rdkit)} training targets missing RDKit "
                f"definitions (e.g. {missing_rdkit[:5]})"
            )

    def _ids(text: str) -> List[int]:
        return list(
            tokenize_model_text(
                tokenizer, text, from_chat_template=config.use_chat_template
            )["input_ids"]
        )

    n = len(items)
    student_prompt_ids = _ids(items[0].prompt)
    teacher_prompt_ids: List[Optional[List[int]]] = [None] * n
    target_ids: List[Optional[List[int]]] = [None] * n
    for it in items:
        teacher_text = sdpo_teacher_instruction(
            config.task,
            definition_text_for_cfg(
                vars(config),
                _word(it),
                raw_defs[_word(it)],
                rdkit_lookup=rdkit_lookup,
                require_lookup=True,
            ),
            use_chat_template=config.use_chat_template,
        )
        teacher_text = maybe_append_mist_smiles_open_tag(
            teacher_text,
            mist_open_tag_in_prompt(
                mist_smiles_tags=config.mist_smiles_tags,
                use_chat_template=config.use_chat_template,
            ),
        )
        if config.use_chat_template:
            teacher_text = chat_prompt(
                tokenizer, teacher_text, system_prompt=task_system_prompt(config.task)
            )
        teacher_prompt_ids[it.id] = _ids(teacher_text)

        if config.use_chat_template:
            full_input = it.prompt + it.target.strip()
        else:
            full_input = it.prompt + " " + it.target.strip() + tokenizer.eos_token
        prompt_ids = _ids(it.prompt)
        target_ids[it.id] = _ids(full_input)[len(prompt_ids):]

    offpolicy_pool = config.sdpo_offpolicy_pool or (2 * config.sdpo_n_offpolicy)
    sdpo_cfg = SDPOConfig(
        divergence=config.sdpo_divergence,
        temperature=config.sdpo_temperature,
        sample_temperature=config.sdpo_sample_temperature,
        sample_top_p=config.sdpo_sample_top_p,
        max_new_tokens=config.sdpo_max_new_tokens,
        n_onpolicy=config.sdpo_n_onpolicy,
        n_offpolicy=config.sdpo_n_offpolicy,
        offpolicy_pool=offpolicy_pool,
        include_gold=config.sdpo_include_gold,
        position=config.position,
        intervention_token_id=intervention_token_id,
        content_span=content_span,
    )
    print(
        f"[train] SDPO enabled: lambda={config.lambda_sdpo} "
        f"divergence={config.sdpo_divergence} on_policy={config.sdpo_n_onpolicy} "
        f"off_policy={config.sdpo_n_offpolicy} (pool={offpolicy_pool}) "
        f"gold={config.sdpo_include_gold} defs={def_path}",
        flush=True,
    )
    return dict(
        lambda_sdpo=config.lambda_sdpo,
        sdpo_cfg=sdpo_cfg,
        sdpo_base_model=base_model,
        sdpo_student_prompt_ids=student_prompt_ids,
        sdpo_teacher_prompt_ids=teacher_prompt_ids,
        sdpo_target_ids=target_ids,
    )


def _dump_sdpo_offpolicy_samples(trainer, items, tokenizer, output_dir: str) -> None:
    """Write the cached off-policy teacher samples to ``output_dir`` for inspection.

    No-op unless the lazy off-policy cache was populated (SDPO with
    ``sdpo_n_offpolicy > 0``). Keyed by the target word (stable, unlike the
    run-specific ``word_id``); stores both raw token ids and decoded text. Words
    that produced no usable continuation appear with an empty ``samples`` list.
    """
    cache = getattr(trainer, "_sdpo_offpolicy_cache", None)
    if not cache:
        return
    id_to_word = {
        it.id: (getattr(it, "_raw_word", None) or it.target).strip() for it in items
    }
    dump: Dict[str, Dict] = {}
    for wid, seqs in cache.items():
        word = id_to_word.get(int(wid), str(wid))
        dump[word] = {
            "word_id": int(wid),
            "samples": [
                {
                    "token_ids": [int(t) for t in seq],
                    "text": tokenizer.decode(seq, skip_special_tokens=True),
                }
                for seq in seqs
            ],
        }
    path = os.path.join(output_dir, "sdpo_offpolicy_samples.json")
    _write_json_config(path, dump)
    print(
        f"[train] Wrote off-policy SDPO samples for {len(dump)} words to {path}",
        flush=True,
    )


class SDPOOffpolicyDumpCallback(TrainerCallback):
    """Dump cached off-policy teacher samples once, after the first epoch.

    By the end of one epoch every training target has been processed once, so the
    lazy off-policy cache is fully populated; we dump it then (and only then).
    """

    def __init__(self, trainer, items, tokenizer, output_dir: str):
        self._trainer = trainer
        self._items = items
        self._tokenizer = tokenizer
        self._output_dir = output_dir
        self._done = False

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._done:
            return
        _dump_sdpo_offpolicy_samples(
            self._trainer, self._items, self._tokenizer, self._output_dir
        )
        self._done = True


def _resolve_encoder_definitions_path(config: TrainConfig) -> str:
    return os.path.abspath(
        config.definitions_path or default_definitions_path(config.task)
    )


def _build_bias_encoder_inputs(
    config: TrainConfig,
    base_model,
    tokenizer,
    items: List[ReftItem],
    *,
    defs_path: str,
):
    """Precompute penultimate hidden states for each training word's definition.

    Runs on the *clean* base model (before pyvene installs the layer-0 hook) so
    captured states are free of the intervention. Returns ``(penult [N, L, H],
    attn [N, L][, instruction_mask [N, L]])`` aligned to item ids ``0 .. N-1``.
    """
    raw_defs = load_raw_definitions(defs_path)
    words = embed_texts_from_items(items)
    rdkit_lookup = (
        load_rdkit_definition_lookup(
            os.path.abspath(
                config.rdkit_definitions_path or default_rdkit_definitions_path()
            )
        )
        if config.append_rdkit_definitions
        else None
    )
    missing = sorted({w for w in words if w not in raw_defs})
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"--bias-network-encoder llm_encoder needs definitions for all training "
            f"words; {len(missing)} missing from {defs_path} (e.g. {preview})"
        )
    if rdkit_lookup is not None:
        missing_rdkit = sorted({w for w in words if w not in rdkit_lookup})
        if missing_rdkit:
            raise ValueError(
                f"--append-rdkit-definitions needs RDKit values for all training "
                f"targets; {len(missing_rdkit)} missing (e.g. {missing_rdkit[:5]})"
            )
    definitions = [
        definition_text_for_cfg(
            vars(config),
            w,
            raw_defs[w],
            rdkit_lookup=rdkit_lookup,
            require_lookup=True,
        )
        for w in words
    ]
    texts = [
        definition_embedding_text(config.task, w, definition)
        for w, definition in zip(words, definitions)
    ]
    pairs = list(zip(words, definitions))
    build_instr_mask = pooling_uses_instruction_mask(config.bias_encoder_pooling)
    # Warn if any definition is longer than the cap: pooling may read truncated text.
    n_truncated = sum(
        1
        for t in texts
        if len(tokenizer(t, truncation=False)["input_ids"]) > config.bias_encoder_max_length
    )
    if n_truncated:
        extra = ""
        if pooling_requires_instruction_mask(config.bias_encoder_pooling):
            extra = (
                " instruction_mean is especially sensitive to truncation "
                "(mean pools only the retained prefix of each definition)."
            )
        print(
            f"[train] WARNING: {n_truncated}/{len(texts)} encoder definition inputs "
            f"exceed --bias-encoder-max-length={config.bias_encoder_max_length} and "
            f"will be truncated (instruction pooling may miss tail tokens).{extra} "
            f"Consider raising --bias-encoder-max-length.",
            flush=True,
        )
    input_ids, attn, instr_mask = encode_definition_inputs(
        tokenizer,
        texts,
        device=base_model.device,
        max_length=config.bias_encoder_max_length,
        task=config.task,
        word_definition_pairs=pairs,
        build_instruction_mask=build_instr_mask,
    )
    chunk = 64
    penult_chunks: List[torch.Tensor] = []
    for start in range(0, input_ids.shape[0], chunk):
        end = start + chunk
        penult_chunks.append(
            capture_penultimate(
                base_model,
                input_ids[start:end],
                attn[start:end],
                layer_index=config.bias_encoder_layer,
            )
        )
    penult = torch.cat(penult_chunks, dim=0)
    return penult, attn, instr_mask


def _build_optimizer(
    reft_model,
    lr: float,
    weight_decay_mode: str,
    wd_W: float,
    wd_b: float,
):
    """Build AdamW for all trainable intervention params.

    When ``weight_decay_mode`` is ``"none"``, every param gets ``weight_decay=0``.
    Otherwise decay is applied selectively:

      "W_only"  — wd_W on learned_source (W); 0 elsewhere
      "W_and_b" — wd_W on W; wd_b on bias parameters/tables; 0 elsewhere
      "b_only"  — wd_b on bias parameters/tables; 0 on W and elsewhere

    R (rotate_layer) is always excluded from decay — kept orthogonal by parametrization.
    Empty param groups are dropped before constructing AdamW.
    """
    if weight_decay_mode == "none":
        trainable = [p for p in reft_model.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

    W_params, b_params, other_params = [], [], []
    for name, param in reft_model.named_parameters():
        if not param.requires_grad:
            continue
        if "learned_source" in name:
            W_params.append(param)
        elif any(
            k in name
            for k in (
                "word_mu",
                "word_bias",
                "bias_network",
                "_learn_bias_mu",
                "_learn_bias_logvar",
            )
        ):
            b_params.append(param)
        else:
            # Encoder LoRA adapters (semantic_encoder.*.lora_*) land here (wd=0).
            other_params.append(param)

    wd_W_effective = wd_W if weight_decay_mode in ("W_only", "W_and_b") else 0.0
    wd_b_effective = wd_b if weight_decay_mode in ("W_and_b", "b_only") else 0.0
    param_groups = [
        {"params": other_params, "weight_decay": 0.0, "name": "other"},
        {"params": W_params, "weight_decay": wd_W_effective, "name": "W"},
        {"params": b_params, "weight_decay": wd_b_effective, "name": "b"},
    ]
    param_groups = [g for g in param_groups if g["params"]]

    print(
        f"[train] weight_decay_mode={weight_decay_mode}  "
        f"wd_W={wd_W_effective}  wd_b={wd_b_effective}  "
        f"(W params={len(W_params)}, b params={len(b_params)}, other={len(other_params)})"
    )

    return torch.optim.AdamW(param_groups, lr=lr)


def _build_lr_scheduler(
    optimizer,
    lr_scheduler_type: str,
    warmup_steps: int,
    total_steps: int,
):
    """Build an LR scheduler from dataloader-derived step counts."""
    from transformers import get_scheduler

    return get_scheduler(
        lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def load_training_items(
    config: TrainConfig,
) -> tuple[List[ReftItem], Optional[List[int]]]:
    """Load task data and apply optional ``train_n_samples`` subsampling."""
    if config.task == "semantle":
        _, words, sim_map = SemantleItem.load_csvs(
            list(config.semantle_csv),
            top_k=config.train_top_k,
        )
        items = SemantleItem.load(words, sim_map)
    elif config.task == "molopt":
        from boreft.data.molopt import apply_mist_smiles_tags, apply_smiles_tags
        from boreft.molopt_split import DEFAULT_ORACLE_CAP_PERCENTILE

        cap = config.molopt_oracle_cap_percentile
        if cap is None:
            cap = DEFAULT_ORACLE_CAP_PERCENTILE
        # Percentiles and the high-tail test pool are over the full CSV.
        # --train-top-k is a prefix only when the cap is off.
        items = MolOptItem.load_csv(
            config.molopt_csv,
            top_k=None if cap > 0 else config.train_top_k,
        )
        if config.smiles_tags:
            apply_smiles_tags(items)
        if config.mist_smiles_tags:
            apply_mist_smiles_tags(
                items, include_open=config.use_chat_template
            )
        if cap > 0:
            from boreft.molopt_split import (
                apply_molopt_oracle_cap,
                default_oracle_scores_path,
                write_oracle_split,
            )

            cache_path = os.path.abspath(
                config.molopt_oracle_scores_path
                or default_oracle_scores_path(str(config.molopt_csv))
            )
            object.__setattr__(config, "molopt_oracle_scores_path", cache_path)
            n_before = len(items)
            n_train = (
                config.train_n_samples
                if config.train_n_samples is not None
                else config.train_top_k
            )
            items, embed_cache_indices, split_meta = apply_molopt_oracle_cap(
                items,
                percentile=float(cap),
                n_train=n_train,
                seed=config.seed,
                cache_path=cache_path,
                strict_n_train=config.train_n_samples is not None,
            )
            write_oracle_split(
                os.path.join(config.output_dir, "oracle_split.json"),
                split_meta,
            )
            object.__setattr__(config, "_oracle_split", split_meta)
            print(
                f"[train] molopt oracle cap p{cap:g}: "
                f"train {len(items)} / {split_meta['n_train_eligible']} eligible "
                f"(pool={n_before}, high-tail={split_meta['n_test_eligible']}, "
                f"seed={config.seed})",
                flush=True,
            )
            return items, embed_cache_indices
    elif config.task == "hypogen":
        items = HypoGenItem.load_csv(config.hypogen_csv, top_k=config.train_top_k)
    else:
        items = ArcItem.load(config.arc_dir, task_filter=config.arc_task_filter)

    embed_cache_indices: Optional[List[int]] = None
    if config.train_n_samples is not None:
        n_before = len(items)
        items, embed_cache_indices = sample_reft_items(
            items,
            config.train_n_samples,
            seed=config.seed,
        )
        if (
            config.lambda_mm <= 0
            and config.lambda_rank <= 0
            and not config.add_bias_network
        ):
            embed_cache_indices = None
        print(
            f"[train] --train-n-samples {config.train_n_samples}: "
            f"using {len(items)} / {n_before} items (seed={config.seed})",
            flush=True,
        )
    return items, embed_cache_indices


def train_reft(
    config: TrainConfig,
    items: List[ReftItem],
    *,
    embed_cache_indices: Optional[List[int]] = None,
    word_sim_matrix: Optional[torch.Tensor] = None,
) -> str:
    """
    Train REFT. When task="semantle" and eval_steps or eval_epochs is set, runs Semantle eval during training.

    ``word_sim_matrix``: optional precomputed [N, N] pairwise similarities S_ij aligned
    with ``items`` order. If None and ``nbr_lambda`` > 0 (semantle), S_ij is computed
    from sentence-transformer embeddings of the target words.

    Neighborhood augmentation and weighted CE are enabled only when
    ``nbr_lambda > 0`` and ``task == "semantle"``. Otherwise training uses the
    standard LM loss (``output.loss``) and, for VAE, ``+ aux`` as before.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    if config.task == "molopt":
        load_rdkit_descriptor_map(
            os.path.abspath(
                config.rdkit_definitions_map_path
                or default_rdkit_definitions_map_path()
            )
        )

    batch_size = config.batch_size

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        padding_side="right",
        use_fast=True,
        cache_dir=config.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_name = infer_torch_dtype_name(override=config.torch_dtype)
    load_dtype_name = infer_training_load_dtype_name(override=config.torch_dtype)
    load_dtype = resolve_torch_dtype(load_dtype_name)
    if config.torch_dtype is not None:
        print(
            f"[train] torch_dtype={dtype_name} (override={config.torch_dtype!r})",
            flush=True,
        )
    else:
        print(f"[train] torch_dtype={dtype_name} (auto)", flush=True)
    if load_dtype_name != dtype_name:
        print(
            f"[train] loading base model as {load_dtype_name} for AMP training "
            f"(checkpoint/eval dtype remains {dtype_name})",
            flush=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=load_dtype,
        device_map="auto" if DEVICE == "cuda" else None,
        cache_dir=config.cache_dir,
    )

    hidden_size = model.config.hidden_size

    if config.layer == EMBED_LAYER:
        print("[train] layer=-1: intervening on model.embed_tokens", flush=True)
    else:
        num_layers = getattr(model.config, "num_hidden_layers", None)
        if num_layers is not None and not (0 <= config.layer < num_layers):
            raise ValueError(
                f"--layer must be {EMBED_LAYER} (embed_tokens) or in "
                f"[0, {num_layers - 1}], got {config.layer}"
            )

    # When the bias network reads from the LLM encoder, no sentence-transformer
    # embed_cache is needed (the bias-network input comes from copied-block LoRA
    # features over each word's definition text).
    encoder_mode = (
        config.add_bias_network
        and config.bias_network_encoder == "llm_encoder"
    )
    embed_cache_tensor: Optional[torch.Tensor] = None
    effective_embed_cache_path: Optional[str] = config.embed_cache_path
    effective_eval_embed_cache_path: Optional[str] = None
    need_embed_cache = (
        config.lambda_mm > 0.0
        or config.lambda_rank > 0.0
        or (config.add_bias_network and not encoder_mode)
    )
    embed_vocab = embed_texts_from_items(items)
    train_provenance, definition_lookup, definitions_path_abs = training_cache_params(
        use_definition_embeds=config.use_definition_embeds,
        task=config.task,
        words=embed_vocab,
        definitions_path=config.definitions_path,
        model_name=config.bias_network_embed_model,
        append_rdkit_definitions=config.append_rdkit_definitions,
        omit_molt5_definitions=config.omit_molt5_definitions,
        rdkit_definitions_path=config.rdkit_definitions_path,
        rdkit_definitions_map_path=config.rdkit_definitions_map_path,
    )
    if need_embed_cache:
        embed_sources: Dict[str, Any] = {
            "task": config.task,
            "seed": config.seed,
        }
        # Descriptive only: cache identity comes from the target list + provenance,
        # so this just records which data produced an entry.
        if config.task == "semantle":
            embed_sources["semantle_csv"] = semantle_csv_abspaths(config.semantle_csv)
        elif config.task == "molopt":
            embed_sources["molopt_csv"] = os.path.abspath(config.molopt_csv)
        elif config.task == "hypogen":
            embed_sources["hypogen_csv"] = os.path.abspath(config.hypogen_csv)
        if config.task in ("semantle", "molopt", "hypogen"):
            embed_sources["train_top_k"] = config.train_top_k
            if config.train_n_samples is not None:
                embed_sources["train_n_samples"] = config.train_n_samples
        embed_cache_tensor, effective_embed_cache_path, _ = resolve_or_build_embed_cache(
            embed_vocab,
            explicit_path=config.embed_cache_path,
            indices=embed_cache_indices,
            sources=embed_sources,
            provenance=train_provenance,
            definition_lookup=definition_lookup,
        )
        # The eval reference cache is always the task's own model over undecorated
        # prompts, so ranking/silhouette stay comparable across runs. It can only be
        # reused as-is when the training cache was built that same way.
        if config.use_definition_embeds or config.bias_network_embed_model is not None:
            _, effective_eval_embed_cache_path, _ = resolve_or_build_embed_cache(
                embed_vocab,
                indices=embed_cache_indices,
                sources=embed_sources,
                provenance=embedding_provenance(
                    use_definition_embeds=False, task=config.task
                ),
                definition_lookup=None,
            )
        else:
            effective_eval_embed_cache_path = effective_embed_cache_path

    bias_network_kwargs = dict(
        add_bias_network=config.add_bias_network,
        bias_network_residual=config.bias_network_residual,
        embed_cache=embed_cache_tensor,
    )
    if encoder_mode:
        bias_network_kwargs["bias_input_source"] = "llm_encoder"
        bias_network_kwargs["bias_network_input_dim"] = hidden_size

    if config.bias_type == "linear":
        intervention = LoreftPerWordBiasIntervention(
            embed_dim=hidden_size,
            low_rank_dimension=config.low_rank_dim,
            dropout=config.dropout,
            dropout_on_b=config.dropout_on_b,
            dtype=load_dtype,
            device=DEVICE,
            num_words=len(items),
            act_fn="linear",
            sigma_b=config.sigma_b,
            lambda_mm=config.lambda_mm,
            mm_exclude_diagonal=config.mm_exclude_diagonal,
            mm_topk_neighbors=config.mm_topk_neighbors,
            mm_anchor_samples=config.mm_anchor_samples,
            lambda_rank=config.lambda_rank,
            rank_temperature=config.rank_temperature,
            rank_min_ref_gap=config.rank_min_ref_gap,
            rank_topk_neighbors=config.rank_topk_neighbors,
            rank_anchor_samples=config.rank_anchor_samples,
            **bias_network_kwargs,
        )
    else:
        intervention = DistributionalWordIntervention(
            embed_dim=hidden_size,
            low_rank_dimension=config.low_rank_dim,
            dropout=config.dropout,
            dropout_on_b=config.dropout_on_b,
            dtype=load_dtype,
            device=DEVICE,
            num_words=len(items),
            kl_beta=config.kl_beta,
            kl_prior_var=config.kl_prior_var,
            vae_free_bits_lambda=config.vae_free_bits_lambda,
            lambda_l2=config.lambda_l2,
            use_word_bias=config.use_word_bias,
            variance=config.variance,
            sigma_b=config.sigma_b,
            lambda_mm=config.lambda_mm,
            mm_exclude_diagonal=config.mm_exclude_diagonal,
            mm_topk_neighbors=config.mm_topk_neighbors,
            mm_anchor_samples=config.mm_anchor_samples,
            lambda_rank=config.lambda_rank,
            rank_temperature=config.rank_temperature,
            rank_min_ref_gap=config.rank_min_ref_gap,
            rank_topk_neighbors=config.rank_topk_neighbors,
            rank_anchor_samples=config.rank_anchor_samples,
            **bias_network_kwargs,
        )

    # LLM-encoder bias network: build the copied final block + LoRA and precompute
    # each training word's penultimate hidden states on the *clean* base model
    # (before pyvene installs the layer-0 hook), then attach to the intervention.
    semantic_encoder = None
    encoder_defs_path: Optional[str] = None
    encoder_layer_index: Optional[int] = None
    if encoder_mode:
        encoder_defs_path = _resolve_encoder_definitions_path(config)
        encoder_layer_index = resolve_encoder_layer_index(
            model, config.bias_encoder_layer
        )
        semantic_encoder = build_semantic_encoder(
            model,
            lora_rank=config.bias_encoder_lora_rank,
            lora_alpha=config.bias_encoder_lora_alpha,
            lora_dropout=config.bias_encoder_lora_dropout,
            lora_targets=DEFAULT_LORA_TARGETS,
            pooling=config.bias_encoder_pooling,
            layer_index=config.bias_encoder_layer,
        )
        enc_penult, enc_attn, enc_instr = _build_bias_encoder_inputs(
            config, model, tokenizer, items, defs_path=encoder_defs_path
        )
        intervention.set_semantic_encoder(semantic_encoder)
        intervention.set_encoder_inputs(enc_penult, enc_attn, enc_instr)
        print(
            f"[train] bias-network encoder: llm_encoder layer={encoder_layer_index} "
            f"lora_rank={config.bias_encoder_lora_rank} "
            f"pooling={config.bias_encoder_pooling} "
            f"inputs={enc_penult.shape[0]}x{enc_penult.shape[1]} "
            f"(trainable LoRA params="
            f"{sum(p.numel() for p in semantic_encoder.trainable_parameters()):,d})",
            flush=True,
        )
    elif config.add_bias_network and config.bias_network_embed_model is not None:
        print(
            f"[train] bias-network encoder: sentence_transformer "
            f"{config.bias_network_embed_model!r} (embed_sim metrics stay on "
            f"{embedding_model_name(config.task)!r})",
            flush=True,
        )

    reft_config = ReftConfig(
        representations=[
            build_reft_representation(config.layer, config.low_rank_dim, intervention)
        ]
    )
    reft_model = get_reft_model(model, reft_config, set_device=True)
    if semantic_encoder is not None:
        # Keep the encoder on the intervention device (set_device moved buffers).
        semantic_encoder.to(model.device)
    reft_model.print_trainable_parameters()
    if config.init_from:
        n_loaded = init_learnable_parameters_from_run(
            reft_model, config.init_from, load_latest=config.load_latest
        )
        print(
            f"[train] initialized {n_loaded} learnable parameter tensors from "
            f"{config.init_from}",
            flush=True,
        )

    intervention_token_str: Optional[str] = None
    intervention_token_id: Optional[int] = None
    span_probe_token: Optional[str] = None
    content_span: Optional[Tuple[int, int]] = None
    if is_marker_position(config.position) or is_content_position(config.position):
        intervention_token_str, ensured_id = resolve_intervention_token(
            tokenizer,
            config.model_name,
            config.intervention_token,
            model,
        )
        if is_marker_position(config.position):
            intervention_token_id = ensured_id
            init_intervention_token_embedding(
                model,
                tokenizer,
                intervention_token_id,
                config.intervention_token_init,
            )
            print(
                f"[train] intervention marker: {intervention_token_str!r} "
                f"(id={intervention_token_id}, inject={config.intervention_inject}, "
                f"init={config.intervention_token_init})",
                flush=True,
            )
        else:
            span_probe_token = intervention_token_str

    # --- Neighborhood augmentation (Idea 1): only if nbr_lambda > 0 and semantle ---
    # Then dataset carries per-example weights and ReftTrainer uses weighted CE (+ aux for VAE).
    training_items = items
    nbr_block_size: Optional[int] = None
    word_sim_matrix_used: Optional[torch.Tensor] = word_sim_matrix
    use_neighborhood_aug = config.nbr_lambda > 0.0 and config.task == "semantle"
    if config.task == "semantle" and word_sim_matrix_used is None:
        from boreft.text_similarity import pairwise_embedding_similarity_matrix

        words = [it.target for it in items]
        print(
            f"[train] Computing pairwise word similarity matrix ({len(words)}×{len(words)})...",
            flush=True,
        )
        word_sim_matrix_used = pairwise_embedding_similarity_matrix(words)
    elif word_sim_matrix_used is not None and word_sim_matrix_used.shape != (
        len(items),
        len(items),
    ):
        raise ValueError(
            f"word_sim_matrix must be [{len(items)}, {len(items)}], "
            f"got {tuple(word_sim_matrix_used.shape)}"
        )

    if use_neighborhood_aug:
        from boreft.data.semantle import SemantleItem

        training_items, nbr_block_size = SemantleItem.build_augmented_items(
            items,
            word_sim_matrix=word_sim_matrix_used,
            nbr_top_k=config.nbr_top_k,
            nbr_lambda=config.nbr_lambda,
        )
        if batch_size % nbr_block_size != 0:
            old_bs = batch_size
            batch_size = max(
                nbr_block_size,
                ((batch_size + nbr_block_size - 1) // nbr_block_size) * nbr_block_size,
            )
            print(
                f"[train] Rounded per_device_train_batch_size {old_bs} up to {batch_size} "
                f"(smallest multiple of neighborhood block_size={nbr_block_size}).",
                flush=True,
            )
        _nb = len(items)
        _bpb = batch_size // nbr_block_size
        if config.aug_batch_mode == "block" and _bpb > _nb:
            old_bs = batch_size
            batch_size = _nb * nbr_block_size
            _bpb = _nb
            print(
                f"[train] Capped per_device_train_batch_size {old_bs} down to {batch_size} "
                f"({_bpb} anchor blocks × block_size={nbr_block_size}).",
                flush=True,
            )
        elif config.aug_batch_mode == "block" and _nb % _bpb != 0:
            print(
                f"[train] Warning: num anchor blocks {_nb} is not divisible by blocks_per_batch={_bpb}; "
                f"each epoch drops the last {_nb % _bpb} blocks (drop_last). "
                f"Prefer batch_size so {_nb} % ({batch_size}//{nbr_block_size}) == 0.",
                flush=True,
            )
        if config.aug_batch_mode == "block":
            print(
                f"[train] Neighborhood augmentation (block mode): {len(items)} anchors × block_size={nbr_block_size} "
                f"= {len(training_items)} items; batches group {batch_size // nbr_block_size} "
                f"blocks per step (shuffled block order).",
                flush=True,
            )
        else:
            print(
                f"[train] Neighborhood augmentation (shuffle mode): {len(items)} anchors × block_size={nbr_block_size} "
                f"= {len(training_items)} items; standard shuffled DataLoader (no block grouping).",
                flush=True,
            )

    assistant_suffix: Optional[str] = None
    instruction_raw = task_instruction(
        config.task,
        use_chat_template=config.use_chat_template,
        override=config.chat_instruction,
    )
    chat_instruction_for_cfg = instruction_raw.strip() if config.use_chat_template else None
    wrapped_instruction = prepare_instruction(
        instruction_raw,
        config.intervention_inject,
        intervention_token_str,
    )
    wrapped_instruction = maybe_append_mist_smiles_open_tag(
        wrapped_instruction,
        mist_open_tag_in_prompt(
            mist_smiles_tags=config.mist_smiles_tags,
            use_chat_template=config.use_chat_template,
        ),
    )
    system_prompt = (
        task_system_prompt(config.task) if config.use_chat_template else None
    )
    if is_content_position(config.position):
        if not span_probe_token:
            raise RuntimeError("content position requires a resolved span probe token")
        content_span = instruction_content_span(
            tokenizer,
            wrapped_instruction,
            use_chat_template=config.use_chat_template,
            span_probe_token=span_probe_token,
            system_prompt=system_prompt,
        )
        print(
            f"[train] content position {config.position!r}: span={content_span} "
            f"(probe={span_probe_token!r})",
            flush=True,
        )
    if config.mist_smiles_tags:
        print(
            "[train] MiST SMILES tags: prompt ends with [START_SMILES] "
            "(after marker inject); gold is 'SMILES [END_SMILES]'"
            if not config.use_chat_template
            else "[train] MiST SMILES tags: gold is '[START_SMILES] SMILES [END_SMILES]'",
            flush=True,
        )
    if config.use_chat_template:
        if system_prompt:
            print("[train] chat system prompt from task_config", flush=True)
        assistant_suffix = apply_chat_format(
            training_items,
            tokenizer,
            wrapped_instruction,
            system_prompt=system_prompt,
        )
    else:
        for it in training_items:
            it.prompt = wrapped_instruction

    train_dataset = ReftDataset(
        items=training_items,
        tokenizer=tokenizer,
        position=config.position,
        num_interventions=1,
        share_weights=False,
        seed=config.seed,
        use_sample_weights=use_neighborhood_aug,
        use_chat_template=config.use_chat_template,
        intervention_token_id=intervention_token_id,
        content_span=content_span,
    )

    report_to = "wandb" if config.wandb_project else "none"

    train_bf16, train_fp16 = trainer_amp_flags(load_dtype_name, use_cuda=DEVICE == "cuda")
    if DEVICE == "cuda" and (train_bf16 or train_fp16):
        amp_mode = "bf16" if train_bf16 else "fp16-amp"
        print(
            f"[train] Trainer AMP: {amp_mode} (load dtype={load_dtype_name})",
            flush=True,
        )
    elif DEVICE == "cuda" and load_dtype_name in ("float16", "bfloat16"):
        print(
            f"[train] Trainer AMP: off (native {load_dtype_name} weights)",
            flush=True,
        )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=config.grad_acc_steps,
        # learning_rate=config.lr,  # set later in optimizer
        # lr_scheduler_type=config.lr_scheduler_type,  # set later in scheduler
        max_grad_norm=config.max_grad_norm,
        logging_steps=10,
        save_strategy="no",
        remove_unused_columns=False,
        report_to=report_to,
        run_name=config.wandb_run_name,
        bf16=train_bf16,
        fp16=train_fp16,
    )

    ckpt_th = list(config.checkpoint_embed_sim_thresholds_list or [])
    ckpt_th_min = list(config.checkpoint_embed_sim_thresholds_min_list or [])

    if config.checkpoint_epoch_interval is not None:
        print(
            f"[train] Epoch checkpoints every {config.checkpoint_epoch_interval} epoch(s) "
            f"-> {config.output_dir}/checkpoints_by_epoch/",
            flush=True,
        )

    intervention_config_dict: Dict = {
        "task": config.task,
        "use_word_bias": config.use_word_bias,
        "bias_type": config.bias_type,
        "variance": config.variance
        if isinstance(config.variance, str)
        else float(config.variance),
        "kl_beta": config.kl_beta,
        "linear_annealing_map": serialize_linear_annealing_map(
            resolved_linear_annealing_map(config)
        ),
        "kl_prior_var": config.kl_prior_var,
        "vae_free_bits_lambda": config.vae_free_bits_lambda,
        "lambda_l2": config.lambda_l2,
        "lambda_mm": config.lambda_mm,
        "mm_topk_neighbors": config.mm_topk_neighbors,
        "mm_anchor_samples": config.mm_anchor_samples,
        "embed_cache_path": effective_embed_cache_path,
        "eval_embed_cache_path": effective_eval_embed_cache_path,
        "mm_exclude_diagonal": config.mm_exclude_diagonal,
        "lambda_rank": config.lambda_rank,
        "rank_temperature": config.rank_temperature,
        "rank_min_ref_gap": config.rank_min_ref_gap,
        "rank_topk_neighbors": config.rank_topk_neighbors,
        "rank_anchor_samples": config.rank_anchor_samples,
        "low_rank_dim": config.low_rank_dim,
        "layer": config.layer,
        "component": intervention_component(config.layer),
        "position": config.position,
        "intervention_inject": config.intervention_inject,
        "intervention_token": intervention_token_str,
        "intervention_token_id": intervention_token_id,
        "intervention_token_init": config.intervention_token_init,
        "span_probe_token": span_probe_token,
        "dropout": config.dropout,
        "dropout_on_b": config.dropout_on_b,
        "add_bias_network": config.add_bias_network,
        "bias_network_residual": config.bias_network_residual,
        "torch_dtype": dtype_name,
        "use_chat_template": config.use_chat_template,
        "chat_instruction": chat_instruction_for_cfg,
        "system_prompt": system_prompt,
        "use_definition_embeds": config.use_definition_embeds,
        "definitions_path": definitions_path_abs,
        "smiles_tags": config.smiles_tags,
        "mist_smiles_tags": config.mist_smiles_tags,
        **train_provenance,
    }
    if config.task == "molopt":
        intervention_config_dict.update(
            {
                "rdkit_definitions_map_path": os.path.abspath(
                    config.rdkit_definitions_map_path
                    or default_rdkit_definitions_map_path()
                ),
                "rdkit_definitions_map_sha256": file_sha256(
                    os.path.abspath(
                        config.rdkit_definitions_map_path
                        or default_rdkit_definitions_map_path()
                    )
                ),
                "rdkit_descriptor_schema_version": (
                    RDKIT_DESCRIPTOR_SCHEMA_VERSION
                ),
            }
        )
    if config.append_rdkit_definitions:
        intervention_config_dict.update(
            {
                "append_rdkit_definitions": True,
                "omit_molt5_definitions": config.omit_molt5_definitions,
                "rdkit_definitions_path": os.path.abspath(
                    config.rdkit_definitions_path
                    or default_rdkit_definitions_path()
                ),
                "rdkit_definitions_sha256": file_sha256(
                    os.path.abspath(
                        config.rdkit_definitions_path
                        or default_rdkit_definitions_path()
                    )
                ),
            }
        )
    if config.add_bias_network and embed_cache_tensor is not None:
        intervention_config_dict["bias_network_embed_dim"] = int(
            embed_cache_tensor.shape[1]
        )
    if config.bias_network_embed_model is not None:
        # Recorded separately from ``sentence_transformer_model`` so eval can tell a
        # deliberate override apart from the task default it should check for drift.
        intervention_config_dict["bias_network_embed_model"] = str(
            config.bias_network_embed_model
        )
    if encoder_mode:
        # The encoder LoRA is persisted inside intervenable_model/ (pyvene); the
        # frozen copied block is rebuilt from the base model at load time using
        # this provenance.
        intervention_config_dict.update(
            {
                "bias_input_source": "llm_encoder",
                "bias_network_encoder": "llm_encoder",
                "bias_network_input_dim": int(hidden_size),
                "bias_network_embed_dim": int(hidden_size),
                "bias_encoder_lora_rank": int(config.bias_encoder_lora_rank),
                "bias_encoder_lora_alpha": float(config.bias_encoder_lora_alpha),
                "bias_encoder_lora_dropout": float(config.bias_encoder_lora_dropout),
                "bias_encoder_lora_targets": list(DEFAULT_LORA_TARGETS),
                "bias_encoder_layer_index": int(encoder_layer_index),
                "bias_encoder_pooling": str(config.bias_encoder_pooling),
                "bias_encoder_max_length": int(config.bias_encoder_max_length),
                "bias_encoder_definitions_path": encoder_defs_path,
            }
        )

    semantle_csv_paths = semantle_csv_abspaths(config.semantle_csv)

    os.makedirs(config.output_dir, exist_ok=True)
    tokenizer.save_pretrained(config.output_dir)
    with open(
        os.path.join(config.output_dir, "items.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(reft_items_to_json(items), f, indent=2)
    _write_json_config(
        os.path.join(config.output_dir, "intervention_config.json"),
        intervention_config_dict,
    )

    training_config_dict = _build_training_config_dict(
        config,
        batch_size=batch_size,
        dtype_name=dtype_name,
        num_anchors=len(items),
        num_training_examples=len(training_items),
        semantle_csv_paths=semantle_csv_paths,
        chat_instruction_resolved=chat_instruction_for_cfg,
        system_prompt_resolved=system_prompt,
        train_bf16=train_bf16,
        train_fp16=train_fp16,
    )
    training_config_path = os.path.join(config.output_dir, TRAINING_CONFIG_NAME)
    _write_json_config(training_config_path, training_config_dict)

    callbacks = []
    if config.eval_steps or config.eval_epochs:
        from boreft.data import TargetReconEval

        callbacks.append(
            TargetReconEval(
                reft_model,
                tokenizer,
                items,
                eval_steps=config.eval_steps,
                eval_every_epochs=config.eval_epochs,
                eval_n_samples=config.eval_n_samples,
                eval_sample_seed=config.seed,
                stop_threshold=config.stop_threshold,
                stop_threshold_min=config.stop_threshold_min,
                stop_threshold_frac=config.stop_threshold_frac,
                embed_sim_tau=config.eval_embed_sim_tau,
                selection_metric=config.eval_selection_metric,
                save_best_params=config.save_best_params,
                output_dir=config.output_dir,
                intervention_config=intervention_config_dict,
                checkpoint_embed_sim_thresholds=ckpt_th or None,
                checkpoint_embed_sim_thresholds_min=ckpt_th_min or None,
                position=config.position,
                assistant_suffix=assistant_suffix,
                from_chat_template=config.use_chat_template,
                eval_prompt=training_items[0].prompt if training_items else wrapped_instruction,
                intervention_token_id=intervention_token_id,
                content_span=content_span,
                task=config.task,
            )
        )
    if config.checkpoint_epoch_interval is not None:
        callbacks.append(
            EpochIntervalCheckpoint(
                reft_model,
                tokenizer,
                items,
                output_dir=config.output_dir,
                intervention_config=intervention_config_dict,
                interval_epochs=config.checkpoint_epoch_interval,
            )
        )

    sdpo_kwargs = _build_sdpo_trainer_kwargs(
        config,
        items,
        tokenizer,
        model,
        intervention_token_id=intervention_token_id,
        content_span=content_span,
    )

    anneal_map = resolved_linear_annealing_map(config)
    trainer = ReftTrainerForCausalLM(
        model=reft_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=ReftDataCollator(tokenizer=tokenizer, model=model),
        word_sim_matrix=word_sim_matrix_used,
        neighborhood_block_size=nbr_block_size
        if use_neighborhood_aug and config.aug_batch_mode == "block"
        else None,
        kl_beta=config.kl_beta,
        linear_annealing_map=anneal_map,
        lambda_ce=config.lambda_ce,
        use_margin_loss=config.use_margin_loss,
        margin_loss_margin=config.margin_loss_margin,
        callbacks=callbacks,
        **sdpo_kwargs,
    )
    # HF Trainer.state.global_step counts optimizer updates, not micro-batches.
    micro_batches = len(trainer.get_train_dataloader())
    steps_per_epoch = max(1, micro_batches // max(1, config.grad_acc_steps))
    total_steps = steps_per_epoch * max(1, config.epochs)
    warmup_steps = int(config.warmup_ratio * total_steps)
    trainer.steps_per_epoch = steps_per_epoch
    training_config_dict = _build_training_config_dict(
        config,
        batch_size=batch_size,
        dtype_name=dtype_name,
        num_anchors=len(items),
        num_training_examples=len(training_items),
        semantle_csv_paths=semantle_csv_paths,
        chat_instruction_resolved=chat_instruction_for_cfg,
        system_prompt_resolved=system_prompt,
        train_bf16=train_bf16,
        train_fp16=train_fp16,
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )
    _write_json_config(training_config_path, training_config_dict)

    if config.wandb_project:
        os.environ["WANDB_PROJECT"] = config.wandb_project
        if config.wandb_entity:
            os.environ["WANDB_ENTITY"] = config.wandb_entity
        if config.wandb_group:
            os.environ["WANDB_GROUP"] = config.wandb_group
        if config.wandb_dir:
            os.environ["WANDB_DIR"] = config.wandb_dir
            os.makedirs(config.wandb_dir, exist_ok=True)
        _wb_init: Dict = {
            "project": config.wandb_project,
            "name": config.wandb_run_name,
            "dir": config.wandb_dir,
            "config": _wandb_config_dict(
                training_config_dict, intervention_config_dict
            ),
        }
        if config.wandb_entity:
            _wb_init["entity"] = config.wandb_entity
        if config.wandb_group:
            _wb_init["group"] = config.wandb_group
        wandb.init(**_wb_init)

    print(
        f"[train] steps/epoch={steps_per_epoch}  total_steps={total_steps}  "
        f"warmup_steps={warmup_steps}  grad_acc_steps={config.grad_acc_steps}",
        flush=True,
    )
    trainer.optimizer = _build_optimizer(
        reft_model,
        config.lr,
        config.weight_decay_mode,
        config.wd_W,
        config.wd_b,
    )
    trainer.lr_scheduler = _build_lr_scheduler(
        trainer.optimizer,
        config.lr_scheduler_type,
        warmup_steps,
        total_steps,
    )
    if anneal_map:
        for key, (start, end, epochs) in sorted(anneal_map.items()):
            anneal_steps = int(epochs) * steps_per_epoch
            print(
                f"[train] linear anneal {key}: {start} → {end} "
                f"over {epochs} epoch(s) "
                f"({anneal_steps} steps, {steps_per_epoch} steps/epoch)",
                flush=True,
            )
    if config.lambda_sdpo > 0 and config.sdpo_n_offpolicy > 0:
        # After the first epoch the lazy off-policy cache is fully populated.
        trainer.add_callback(
            SDPOOffpolicyDumpCallback(trainer, items, tokenizer, config.output_dir)
        )
    trainer.train()

    os.makedirs(config.output_dir, exist_ok=True)
    if use_neighborhood_aug and word_sim_matrix_used is not None:
        torch.save(
            word_sim_matrix_used.cpu(),
            os.path.join(config.output_dir, "pairwise_word_sim.pt"),
        )

    intervention_config_dict = _finalize_training_checkpoint(
        output_dir=config.output_dir,
        reft_model=reft_model,
        tokenizer=tokenizer,
        items=items,
        intervention_config=intervention_config_dict,
        save_best_params=config.save_best_params,
    )

    print(f"[train] Saved to {config.output_dir}")
    if config.wandb_project:
        if wandb.run is not None:
            meta = {
                "run_id": wandb.run.id,
                "project": config.wandb_project,
                "entity": config.wandb_entity or getattr(wandb.run, "entity", None),
                "group": config.wandb_group or getattr(wandb.run, "group", None),
                "program": sys.argv[0],
                "args": sys.argv[1:],
            }
            with open(
                os.path.join(config.output_dir, "wandb_meta.json"),
                "w",
                encoding="utf-8",
            ) as _f:
                json.dump(meta, _f, indent=2)
            # Backward compat for older eval scripts
            with open(
                os.path.join(config.output_dir, "wandb_run_id.txt"),
                "w",
                encoding="utf-8",
            ) as _f:
                _f.write(wandb.run.id)
        # Keep the run open through in-process post-training eval so WandB does not
        # mark the run crashed between train.finish() and eval resume. Eval calls
        # wandb.finish() when the pipeline completes.
        if not config.run_full_eval:
            wandb.finish()

    # No task check here: _validate_train_config already restricts run_full_eval to
    # RECON_EVAL_TASKS, and re-testing the task is how molopt got silently skipped.
    if config.run_full_eval:
        # Release the in-memory training model before the eval pipeline reloads the
        # checkpoint from disk, so the training copy and the eval copy never sit in
        # (GPU) memory at the same time.
        del trainer, reft_model, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _run_post_training_full_eval(config, output_dir=config.output_dir)

    return config.output_dir


def main():
    config = parse_train_config()
    items, embed_cache_indices = load_training_items(config)
    output_dir = train_reft(config, items, embed_cache_indices=embed_cache_indices)

    if config.run_full_eval:
        status_path = os.path.join(output_dir, "eval", "eval_status.json")
        if os.path.isfile(status_path):
            with open(status_path, encoding="utf-8") as f:
                status = json.load(f)
            if not status.get("success", False):
                print(
                    "[train] Post-training eval failed — see eval/eval_output.log",
                    file=sys.stderr,
                )
                sys.exit(1)


if __name__ == "__main__":
    main()
