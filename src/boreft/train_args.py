"""Training configuration: single source of truth for CLI and train_reft."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Annotated, List, Literal, Optional, Tuple, Union

import tyro

from boreft.eval.semantle import DEFAULT_MAX_NEW_TOKENS
from boreft.pyreft.losses import (
    LinearAnnealMap,
    resolve_linear_annealing_map,
)


def resolved_linear_annealing_map(config: "TrainConfig") -> LinearAnnealMap:
    """Return the normalized annealing map from ``--linear-annealing-map``."""
    cached = getattr(config, "_linear_annealing_map_resolved", None)
    if cached is not None:
        return dict(cached)
    return resolve_linear_annealing_map(
        raw_map=config.linear_annealing_map,
        defaults={
            "kl_beta": float(config.kl_beta),
            "lambda_ce": float(config.lambda_ce),
            "lambda_sdpo": float(config.lambda_sdpo),
        },
    )


# Tasks with one bias vector per target and an embedding model that can score a
# decode against its target — everything the reconstruction eval suite needs.
RECON_EVAL_TASKS = ("semantle", "molopt", "hypogen")


def _validate_train_config(config: "TrainConfig") -> None:
    from boreft.task_config import (
        task_supports_chat_template,
        task_supports_definition_embeds,
        task_supports_fingerprints,
    )

    def require_recon_eval_task(flag: str) -> None:
        if config.task not in RECON_EVAL_TASKS:
            raise ValueError(
                f"{flag} requires a task with target-reconstruction eval "
                f"({' or '.join(RECON_EVAL_TASKS)}); task={config.task!r}"
            )

    if config.eval_steps is not None and config.eval_epochs is not None:
        raise ValueError("pass only one of --eval-steps and --eval-epochs")
    if config.eval_steps is not None or config.eval_epochs is not None:
        require_recon_eval_task("--eval-steps/--eval-epochs")

    if config.task == "semantle" and not config.semantle_csv:
        raise ValueError("--semantle-csv required for task=semantle")
    if config.task == "molopt" and not config.molopt_csv:
        raise ValueError("--molopt-csv required for task=molopt")
    if config.task == "hypogen" and not config.hypogen_csv:
        raise ValueError("--hypogen-csv required for task=hypogen")
    if config.task == "arc" and not config.arc_dir:
        raise ValueError("--arc-dir required for task=arc")

    # Dataset-specific folders (data/hypogen/<dataset>/) keep definitions next to
    # the CSV. Fill that sibling in so SDPO / definition-embeds do not look at
    # the unused data/hypogen/train/ path.
    if (
        config.task == "hypogen"
        and not config.definitions_path
        and config.hypogen_csv
    ):
        sibling = os.path.join(
            os.path.dirname(os.path.abspath(config.hypogen_csv)),
            "definitions.jsonl",
        )
        if os.path.isfile(sibling):
            object.__setattr__(config, "definitions_path", sibling)

    has_eval = bool(config.eval_steps or config.eval_epochs)

    if config.eval_selection_metric != "embed_sim":
        if not has_eval:
            raise ValueError(
                "--eval-selection-metric requires --eval-steps or --eval-epochs"
            )
        if not task_supports_fingerprints(config.task):
            raise ValueError(
                f"--eval-selection-metric={config.eval_selection_metric} requires a "
                f"task whose targets are molecules (task={config.task!r} has none); "
                f"only embed_sim is available there"
            )

    if config.stop_threshold is not None and not has_eval:
        raise ValueError("--stop-threshold requires --eval-steps or --eval-epochs")

    if config.stop_threshold_min is not None and not has_eval:
        raise ValueError("--stop-threshold-min requires --eval-steps or --eval-epochs")

    if config.stop_threshold_min is not None and not (
        0.0 <= config.stop_threshold_min <= 1.0
    ):
        raise ValueError("--stop-threshold-min must be in [0, 1]")

    if not (0.0 < config.stop_threshold_frac <= 1.0):
        raise ValueError("--stop-threshold-frac must be in (0, 1]")

    if (
        config.stop_threshold_frac < 1.0 - 1e-12
        and config.stop_threshold_min is None
    ):
        raise ValueError(
            "--stop-threshold-frac < 1 requires --stop-threshold-min (the per-target tau)"
        )

    if config.stop_threshold_frac < 1.0 - 1e-12 and not has_eval:
        raise ValueError("--stop-threshold-frac requires --eval-steps or --eval-epochs")

    if config.run_full_eval:
        require_recon_eval_task("--run-full-eval")

    if config.checkpoint_embed_sim_thresholds:
        require_recon_eval_task("--checkpoint-embed-sim-thresholds")
        if not has_eval:
            raise ValueError(
                "--checkpoint-embed-sim-thresholds requires --eval-steps or --eval-epochs"
            )

    if config.checkpoint_embed_sim_thresholds_min:
        require_recon_eval_task("--checkpoint-embed-sim-thresholds-min")
        if not has_eval:
            raise ValueError(
                "--checkpoint-embed-sim-thresholds-min requires --eval-steps or --eval-epochs"
            )
        for t in config.checkpoint_embed_sim_thresholds_min:
            if not (0.0 <= t <= 1.0):
                raise ValueError(
                    "--checkpoint-embed-sim-thresholds-min values must be in [0, 1]"
                )

    if (
        config.checkpoint_epoch_interval is not None
        and config.checkpoint_epoch_interval <= 0
    ):
        raise ValueError("--checkpoint-epoch-interval must be a positive integer")

    if config.use_chat_template and not task_supports_chat_template(config.task):
        raise ValueError(
            f"--use-chat-template requires a task with a 'chat_prompt' in task_config "
            f"(task={config.task!r} has none)"
        )

    # Definition embeds need one written definition per target: Semantle words
    # get dictionary glosses, molopt SMILES get their ChEBI descriptions (both
    # shipped as definitions.jsonl).
    if config.use_definition_embeds and not task_supports_definition_embeds(
        config.task
    ):
        raise ValueError(
            f"--use-definition-embeds requires a task with an 'embedding_prompt_defn' "
            f"(task={config.task!r} has none)"
        )

    if config.append_rdkit_definitions:
        if config.task != "molopt":
            raise ValueError(
                "--append-rdkit-definitions is only supported for task=molopt"
            )
        consumes_definitions = (
            config.use_definition_embeds
            or config.lambda_sdpo > 0
            or (
                config.add_bias_network
                and config.bias_network_encoder == "llm_encoder"
            )
        )
        if not consumes_definitions:
            raise ValueError(
                "--append-rdkit-definitions requires --use-definition-embeds, "
                "--lambda-sdpo > 0, or "
                "--bias-network-encoder llm_encoder with --add-bias-network"
            )
    if config.omit_molt5_definitions and not config.append_rdkit_definitions:
        raise ValueError(
            "--omit-molt5-definitions requires --append-rdkit-definitions"
        )

    if config.smiles_tags and config.mist_smiles_tags:
        raise ValueError("pass only one of --smiles-tags and --mist-smiles-tags")
    if config.smiles_tags and config.task != "molopt":
        raise ValueError("--smiles-tags is only supported for task=molopt")
    if config.mist_smiles_tags and config.task != "molopt":
        raise ValueError("--mist-smiles-tags is only supported for task=molopt")

    # Neighborhood augmentation needs the Semantle CSV's similarity column.
    if config.nbr_lambda > 0 and config.task != "semantle":
        raise ValueError("--nbr-lambda > 0 is only supported for task=semantle")

    if (config.lambda_mm > 0 or config.lambda_rank > 0) and config.add_bias_network:
        raise ValueError(
            "--add-bias-network is incompatible with --lambda-mm > 0 or --lambda-rank > 0"
        )

    if config.add_bias_network and not config.use_word_bias:
        raise ValueError("--add-bias-network requires use_word_bias=True")

    if config.bias_network_encoder == "llm_encoder":
        if not config.add_bias_network:
            raise ValueError(
                "--bias-network-encoder llm_encoder requires --add-bias-network"
            )
        # The LLM encoder reads each target's definition text.
        if not task_supports_definition_embeds(config.task):
            raise ValueError(
                f"--bias-network-encoder llm_encoder requires a task with an "
                f"'embedding_prompt_defn' (task={config.task!r} has none)"
            )
        if config.bias_encoder_lora_rank <= 0:
            raise ValueError("--bias-encoder-lora-rank must be a positive integer")
        if config.bias_encoder_max_length <= 0:
            raise ValueError("--bias-encoder-max-length must be a positive integer")
        pooling = config.bias_encoder_pooling
        if pooling not in (
            "last_instruction",
            "instruction_mean",
            "last_token",
        ):
            raise ValueError(
                "--bias-encoder-pooling must be one of: "
                "last_instruction, instruction_mean, last_token"
            )

    if config.bias_network_embed_model is not None:
        if not config.add_bias_network:
            raise ValueError(
                "--bias-network-embed-model requires --add-bias-network"
            )
        # llm_encoder builds its inputs from the base model's own hidden states, so
        # there is no sentence-transformer to swap out.
        if config.bias_network_encoder != "sentence_transformer":
            raise ValueError(
                "--bias-network-embed-model requires "
                "--bias-network-encoder sentence_transformer "
                f"(got {config.bias_network_encoder!r})"
            )
        if not str(config.bias_network_embed_model).strip():
            raise ValueError("--bias-network-embed-model must not be empty")

    if config.train_n_samples is not None and config.train_n_samples <= 0:
        raise ValueError("--train-n-samples must be a positive integer")

    if config.molopt_oracle_cap_percentile is not None and config.task != "molopt":
        raise ValueError("--molopt-oracle-cap-percentile is only supported for task=molopt")
    if config.task == "molopt":
        cap = config.molopt_oracle_cap_percentile
        if cap is None:
            from boreft.molopt_split import DEFAULT_ORACLE_CAP_PERCENTILE

            object.__setattr__(
                config, "molopt_oracle_cap_percentile", DEFAULT_ORACLE_CAP_PERCENTILE
            )
            cap = DEFAULT_ORACLE_CAP_PERCENTILE
        if cap < 0 or cap > 100:
            raise ValueError(
                "--molopt-oracle-cap-percentile must be in [0, 100] "
                "(0 disables the property split)"
            )
    if config.full_eval_oracle_screen_n_sobol < 0:
        raise ValueError("--full-eval-oracle-screen-n-sobol must be >= 0")

    if not (0.0 <= config.eval_embed_sim_tau <= 1.0):
        raise ValueError("--eval-embed-sim-tau must be in [0, 1]")
    if config.test_n_samples <= 0:
        raise ValueError("--test-n-samples must be a positive integer")
    if not (0.0 < config.eval_bbox_pca_var <= 1.0):
        raise ValueError("--eval-bbox-pca-var must be in (0, 1]")

    if config.max_grad_norm < 0:
        raise ValueError("--max-grad-norm must be >= 0")

    if config.grad_acc_steps < 1:
        raise ValueError("--grad-acc-steps must be a positive integer")

    if config.lambda_ce < 0:
        raise ValueError("--lambda-ce must be >= 0")
    if config.margin_loss_margin < 0:
        raise ValueError("--margin-loss-margin must be >= 0")

    anneal_map = resolve_linear_annealing_map(
        raw_map=config.linear_annealing_map,
        defaults={
            "kl_beta": float(config.kl_beta),
            "lambda_ce": float(config.lambda_ce),
            "lambda_sdpo": float(config.lambda_sdpo),
        },
    )
    object.__setattr__(config, "_linear_annealing_map_resolved", anneal_map)

    if "kl_beta" in anneal_map:
        if config.bias_type != "vae":
            raise ValueError(
                "--linear-annealing-map kl_beta requires --bias-type vae"
            )
        kl_start, kl_end, _kl_epochs = anneal_map["kl_beta"]
        if config.kl_beta <= 0:
            # Require a positive static --kl-beta so intervention KL is
            # enabled at construction time (trainer writes the annealed
            # value onto iv.kl_beta each step).
            raise ValueError(
                "--linear-annealing-map kl_beta requires --kl-beta > 0 "
                "(intervention KL gate)"
            )
        if max(float(kl_start), float(kl_end)) <= 0:
            raise ValueError(
                "--linear-annealing-map kl_beta requires a positive start "
                "or end value"
            )

    if config.lambda_sdpo < 0:
        raise ValueError("--lambda-sdpo must be >= 0")
    if "lambda_sdpo" in anneal_map:
        sdpo_start, sdpo_end, _sdpo_epochs = anneal_map["lambda_sdpo"]
        if max(float(sdpo_start), float(sdpo_end)) > 0 and config.lambda_sdpo <= 0:
            raise ValueError(
                "--linear-annealing-map lambda_sdpo with a positive start/end "
                "requires --lambda-sdpo > 0 (SDPO setup gate; use shorthand "
                "lambda_sdpo=START:EPOCHS with --lambda-sdpo set to the target end)"
            )
    if config.lambda_sdpo > 0:
        from boreft.task_config import task_supports_sdpo

        if not task_supports_sdpo(config.task):
            raise ValueError(
                f"--lambda-sdpo > 0 requires a task with an 'sdpo_teacher_prompt' "
                f"(task={config.task!r} has none)"
            )
        if config.use_chat_template:
            from boreft.task_config import task_config

            if "chat_sdpo_teacher_prompt" not in task_config.get(config.task, {}):
                raise ValueError(
                    f"--lambda-sdpo > 0 with --use-chat-template requires a task "
                    f"with a 'chat_sdpo_teacher_prompt' (task={config.task!r} has none)"
                )
        if not config.use_word_bias:
            raise ValueError("--lambda-sdpo > 0 requires --use-word-bias")
        if config.nbr_lambda > 0:
            raise ValueError(
                "--lambda-sdpo is not yet supported with neighborhood augmentation "
                "(--nbr-lambda > 0)"
            )
        if config.sdpo_n_onpolicy < 0 or config.sdpo_n_offpolicy < 0:
            raise ValueError("--sdpo-n-onpolicy/--sdpo-n-offpolicy must be >= 0")
        if config.sdpo_offpolicy_pool < 0:
            raise ValueError("--sdpo-offpolicy-pool must be >= 0")
        if 0 < config.sdpo_offpolicy_pool < config.sdpo_n_offpolicy:
            raise ValueError(
                "--sdpo-offpolicy-pool must be >= --sdpo-n-offpolicy"
            )
        if (
            config.sdpo_n_onpolicy == 0
            and config.sdpo_n_offpolicy == 0
            and not config.sdpo_include_gold
        ):
            raise ValueError(
                "--lambda-sdpo > 0 requires at least one trajectory source "
                "(--sdpo-n-onpolicy, --sdpo-n-offpolicy, or --sdpo-include-gold)"
            )
        if config.sdpo_temperature <= 0 or config.sdpo_sample_temperature <= 0:
            raise ValueError("SDPO temperatures must be positive")
        if config.sdpo_max_new_tokens <= 0:
            raise ValueError("--sdpo-max-new-tokens must be positive")

    if config.vae_free_bits_lambda < 0:
        raise ValueError("--vae-free-bits-lambda must be >= 0")
    if config.vae_free_bits_lambda > 0:
        if config.bias_type != "vae":
            raise ValueError("--vae-free-bits-lambda requires --bias-type vae")
        if config.kl_beta <= 0:
            raise ValueError("--vae-free-bits-lambda requires --kl-beta > 0")

    from boreft.intervention_marker import (
        is_content_position,
        is_marker_position,
        validate_intervention_position,
    )

    object.__setattr__(
        config, "position", validate_intervention_position(config.position)
    )
    marker_pos = is_marker_position(config.position)
    content_pos = is_content_position(config.position)
    inject = config.intervention_inject
    if inject != "none" and not marker_pos:
        raise ValueError(
            "--intervention-inject prefix|suffix requires --position marker"
        )
    if marker_pos and inject == "none":
        raise ValueError(
            "--position marker requires --intervention-inject prefix or suffix"
        )
    if content_pos and inject != "none":
        raise ValueError(
            "content_f1/content_l1 require --intervention-inject none"
        )

    if config.init_from is not None:
        init_from = str(config.init_from).strip()
        if not init_from:
            raise ValueError("--init-from must be a non-empty directory path")
        init_from = os.path.abspath(init_from)
        object.__setattr__(config, "init_from", init_from)
        if not os.path.isdir(init_from):
            raise ValueError(f"--init-from is not a directory: {init_from}")
        from boreft.data_utils import (
            list_intervenable_weight_files,
            resolve_checkpoint_dir,
        )

        resolved = resolve_checkpoint_dir(
            init_from, load_latest=bool(config.load_latest)
        )
        intervenable = os.path.join(resolved, "intervenable_model")
        if not os.path.isdir(intervenable):
            raise ValueError(
                f"--init-from {resolved} has no intervenable_model/ "
                "(pass a previous run's output_dir"
                + (" or --load-latest when latest/ exists)" if config.load_latest else ")")
            )
        bins = list_intervenable_weight_files(resolved)
        if not bins:
            raise ValueError(
                f"--init-from {resolved}/intervenable_model has no intkey_*.bin "
                "(pass a previous run's output_dir)"
            )
        if len(bins) > 1:
            names = ", ".join(os.path.basename(path) for path in bins)
            raise ValueError(
                f"--init-from {resolved}/intervenable_model: expected one "
                f"intkey_*.bin, found {len(bins)} ({names})"
            )


def _normalize_variance(variance: Union[str, float]) -> Union[str, float]:
    if isinstance(variance, float):
        if variance <= 0:
            raise ValueError(
                f"--variance must be positive when numeric, got {variance}"
            )
        return variance
    if variance == "learnable":
        return variance
    try:
        value = float(variance)
    except ValueError as exc:
        raise ValueError(
            f"--variance must be 'learnable' or a positive float, got {variance!r}"
        ) from exc
    if value <= 0:
        raise ValueError(f"--variance must be positive when numeric, got {value}")
    return value


@dataclass
class TrainConfig:
    """All training hyperparameters and data paths (CLI + programmatic entry point)."""

    task: Annotated[
        Literal["semantle", "molopt", "hypogen", "arc"],
        tyro.conf.arg(help="Training task."),
    ]

    # Data
    semantle_csv: Annotated[
        Optional[Tuple[str, ...]],
        tyro.conf.arg(help="Semantle CSV paths (required for task=semantle)."),
    ] = None
    train_top_k: Annotated[
        int,
        tyro.conf.arg(
            help="CSV prefix used as the training vocabulary. For molopt with "
            "the oracle cap on, this is the draw size from the p90-capped "
            "pool unless --train-n-samples is set."
        ),
    ] = 100
    train_n_samples: Annotated[
        Optional[int],
        tyro.conf.arg(help="Random subsample of training items (optional)."),
    ] = None
    molopt_csv: Annotated[
        Optional[str],
        tyro.conf.arg(help="MolOpt CSV path (required for task=molopt)."),
    ] = None
    molopt_oracle_cap_percentile: Annotated[
        Optional[float],
        tyro.conf.arg(
            help="MolOpt: drop molecules above this per-oracle percentile "
            "(DRD2 / GSK3B / JNK3) from the train pool. Default 90. Test "
            "targets are drawn from the complementary high tail of the full "
            "CSV. 0 disables and restores --train-top-k prefix sampling."
        ),
    ] = None
    molopt_oracle_scores_path: Annotated[
        Optional[str],
        tyro.conf.arg(
            help="JSON cache of TDC oracle scores keyed by canonical SMILES. "
            "Default: <molopt-csv-dir>/oracle_scores.json."
        ),
    ] = None
    hypogen_csv: Annotated[
        Optional[str],
        tyro.conf.arg(help="HypoGen CSV path (required for task=hypogen)."),
    ] = None
    smiles_tags: Annotated[
        bool,
        tyro.conf.arg(
            help="Wrap molopt generation targets in <SMILES>...</SMILES> tags. "
            "Identity, embeddings, and definitions stay on the bare SMILES; "
            "decodes unwrap the tags before metrics.",
        ),
    ] = False
    mist_smiles_tags: Annotated[
        bool,
        tyro.conf.arg(
            help="MiST SMILES markup for molopt. Completion: append "
            "[START_SMILES] after marker inject and wrap gold as "
            "'SMILES [END_SMILES]'. Chat: wrap gold as "
            "'[START_SMILES] SMILES [END_SMILES]' (open tag is not in the "
            "user turn). Identity, embeddings, and definitions stay on the "
            "bare SMILES; decodes unwrap the tags before metrics. Mutually "
            "exclusive with --smiles-tags.",
        ),
    ] = False
    arc_dir: Annotated[
        Optional[str],
        tyro.conf.arg(help="ARC data directory (required for task=arc)."),
    ] = None
    arc_task_filter: Annotated[
        Optional[str],
        tyro.conf.arg(help="Optional ARC task name filter."),
    ] = None

    # Model
    model_name: Annotated[
        str,
        tyro.conf.arg(help="HuggingFace model ID."),
    ] = "meta-llama/Llama-3.2-1B"
    cache_dir: Annotated[
        Optional[str],
        tyro.conf.arg(help="HuggingFace cache directory."),
    ] = None
    torch_dtype: Annotated[
        Optional[Literal["bfloat16", "float16", "float32"]],
        tyro.conf.arg(
            help="Model/intervention dtype. Omit for auto: float32 on CPU; on CUDA, "
            "bfloat16 if compute capability >= 8.0 (A100+), else float16 (V100).",
        ),
    ] = None

    # Intervention / training
    low_rank_dim: Annotated[int, tyro.conf.arg(help="LoReFT rank.")] = 8
    layer: Annotated[
        int,
        tyro.conf.arg(help="Intervention layer (-1 = embed_tokens)."),
    ] = 15
    position: Annotated[
        str,
        tyro.conf.arg(
            help="Intervention position: l1, f1, f2+l2, content_f1, content_l1, marker (used to add an unused token to the user instruction).",
        ),
    ] = "l1"
    intervention_inject: Annotated[
        Literal["none", "prefix", "suffix"],
        tyro.conf.arg(
            help="Inject intervention token at start/end of user instruction (requires position=marker)."
        ),
    ] = "none"
    intervention_token: Annotated[
        Optional[str],
        tyro.conf.arg(
            help="Intervention marker string (default: model entry in config/intervention_tokens.json)."
        ),
    ] = None
    intervention_token_init: Annotated[
        Literal["none", "newline", "random"],
        tyro.conf.arg(
            help="Initialize intervention token embedding before training (when using marker)."
        ),
    ] = "none"
    epochs: Annotated[int, tyro.conf.arg(help="Training epochs.")] = 30
    batch_size: Annotated[int, tyro.conf.arg(help="Per-device train batch size.")] = 4
    grad_acc_steps: Annotated[
        int,
        tyro.conf.arg(
            help="Gradient accumulation steps (HuggingFace TrainingArguments "
            "gradient_accumulation_steps). Effective batch size is "
            "batch_size * grad_acc_steps."
        ),
    ] = 1
    lr: Annotated[float, tyro.conf.arg(help="Learning rate.")] = 1e-4
    lr_scheduler_type: Annotated[
        str,
        tyro.conf.arg(help="LR scheduler type (e.g. linear, cosine)."),
    ] = "linear"
    warmup_ratio: Annotated[
        float, tyro.conf.arg(help="Warmup ratio of total steps. Accepts values in [0, 1].")
    ] = 0.0
    max_grad_norm: Annotated[
        float,
        tyro.conf.arg(
            help="Max gradient norm for Trainer clipping (0 disables clipping). "
            "Default matches HuggingFace TrainingArguments (1.0).",
        ),
    ] = 1.0
    kl_beta: Annotated[float, tyro.conf.arg(help="KL penalty weight (VAE bias).")] = 0.0
    linear_annealing_map: Annotated[
        Optional[str],
        tyro.conf.arg(
            help="Comma-separated linear anneals for loss coefficients. "
            "Format: KEY=START:END:EPOCHS or KEY=START:EPOCHS (end from the "
            "matching --kl-beta / --lambda-ce / --lambda-sdpo flag). "
            "Allowed keys: kl_beta, lambda_ce, lambda_sdpo. "
            "Example: kl_beta=0:0.1:50,lambda_sdpo=0:20",
        ),
    ] = None
    kl_prior_var: Annotated[
        float, tyro.conf.arg(help="KL prior variance (VAE bias).")
    ] = 1.0
    vae_free_bits_lambda: Annotated[
        float,
        tyro.conf.arg(
            help="Free-bits floor (nats per latent dim) for the VAE KL term. Each "
            "dimension's batch-averaged KL is clamped at this value before summing, "
            "so KL below the floor is not penalized (mitigates posterior collapse). "
            "0 disables free-bits (standard KL).",
        ),
    ] = 0.0
    lambda_l2: Annotated[
        float,
        tyro.conf.arg(
            help="Loss-term L2 penalty on batch bias mu (not AdamW weight decay)."
        ),
    ] = 0.0
    lambda_mm: Annotated[float, tyro.conf.arg(help="Mismatch loss weight.")] = 0.0
    embed_cache_path: Annotated[
        Optional[str],
        tyro.conf.arg(
            help="Optional path to embed_cache.pt; when unset, cache is resolved or "
            "built under repo-root embed_cache/ from training vocabulary.",
        ),
    ] = None
    use_definition_embeds: Annotated[
        bool,
        tyro.conf.arg(
            help="Use definition-based embed_cache for training losses (MM/rank/bias-network). "
            "Also builds a separate prompt-based eval_embed_cache. embed_sim always uses "
            "embedding_prompt.",
        ),
    ] = False
    definitions_path: Annotated[
        Optional[str],
        tyro.conf.arg(
            help="JSONL target definitions (default: data/<task>/train/definitions.jsonl; "
            "for hypogen, definitions.jsonl next to --hypogen-csv when present).",
        ),
    ] = None
    append_rdkit_definitions: Annotated[
        bool,
        tyro.conf.arg(
            help="Append the target's semicolon-separated RDKit descriptor values "
            "to every definition-consuming molopt input (definition embeddings, "
            "SDPO teacher, and llm_encoder bias network).",
        ),
    ] = False
    omit_molt5_definitions: Annotated[
        bool,
        tyro.conf.arg(
            help="With --append-rdkit-definitions, use only the labeled 2D "
            "properties text and omit the MolT5/ChEBI natural-language definition.",
        ),
    ] = False
    rdkit_definitions_path: Annotated[
        Optional[str],
        tyro.conf.arg(
            help="RDKit descriptor-vector JSONL used by "
            "--append-rdkit-definitions (default: "
            "data/molopt/train/definitions_rdkit.jsonl).",
        ),
    ] = None
    rdkit_definitions_map_path: Annotated[
        Optional[str],
        tyro.conf.arg(
            help="RDKit descriptor schema/normalization JSON (default: "
            "data/molopt/train/definitions_rdkit_map.json).",
        ),
    ] = None
    mm_exclude_diagonal: Annotated[
        bool,
        tyro.conf.arg(help="Exclude diagonal from mismatch loss."),
    ] = True
    mm_topk_neighbors: Annotated[
        int, tyro.conf.arg(help="Top-k neighbors for MM loss.")
    ] = 0
    mm_anchor_samples: Annotated[
        int, tyro.conf.arg(help="Anchor samples for MM loss.")
    ] = 0
    lambda_rank: Annotated[float, tyro.conf.arg(help="Ranking loss weight.")] = 0.0
    rank_temperature: Annotated[
        float, tyro.conf.arg(help="Ranking loss temperature.")
    ] = 0.1
    rank_min_ref_gap: Annotated[
        float, tyro.conf.arg(help="Min reference gap for ranking loss.")
    ] = 0.0
    rank_topk_neighbors: Annotated[
        int, tyro.conf.arg(help="Top-k neighbors for ranking loss.")
    ] = 0
    rank_anchor_samples: Annotated[
        int, tyro.conf.arg(help="Anchor samples for ranking loss.")
    ] = 0
    lambda_ce: Annotated[
        float,
        tyro.conf.arg(
            help="Weight on the primary LM term (cross-entropy, or margin loss when "
            "--use-margin-loss is set). Set < 1 to down-weight it relative to "
            "aux/SDPO; 0 disables it."
        ),
    ] = 1.0
    use_margin_loss: Annotated[
        bool,
        tyro.conf.arg(
            help="Replace cross-entropy with token-level strongest-competitor hinge "
            "loss over the supervised target continuation."
        ),
    ] = False
    margin_loss_margin: Annotated[
        float,
        tyro.conf.arg(
            help="Required target-logit advantage for --use-margin-loss. The token "
            "loss is max(0, margin + strongest competitor - target logit)."
        ),
    ] = 0.

    # SDPO self-distillation (definition-in-context teacher → intervention student)
    lambda_sdpo: Annotated[
        float,
        tyro.conf.arg(
            help="SDPO self-distillation loss weight (>0 enables). Distills a base "
            "model conditioned on the target's definition into the intervention.",
        ),
    ] = 0.0
    sdpo_divergence: Annotated[
        Literal["forward_kl", "reverse_kl", "js"],
        tyro.conf.arg(
            help="SDPO per-token divergence: forward_kl (mass-covering KD), "
            "reverse_kl (mode-seeking, SDPO Eq. 1), or js (symmetric)."
        ),
    ] = "forward_kl"
    sdpo_temperature: Annotated[
        float, tyro.conf.arg(help="SDPO distillation softmax temperature.")
    ] = 1.0
    sdpo_sample_temperature: Annotated[
        float, tyro.conf.arg(help="Token-sampling temperature for SDPO rollouts.")
    ] = 1.0
    sdpo_sample_top_p: Annotated[
        float, tyro.conf.arg(help="Top-p for SDPO rollout sampling.")
    ] = 1.0
    sdpo_max_new_tokens: Annotated[
        int, tyro.conf.arg(help="Max new tokens per SDPO rollout.")
    ] = 128
    sdpo_n_onpolicy: Annotated[
        int, tyro.conf.arg(help="On-policy (student) samples per word for SDPO.")
    ] = 4
    sdpo_n_offpolicy: Annotated[
        int, tyro.conf.arg(help="Off-policy (teacher) samples per word for SDPO.")
    ] = 4
    sdpo_offpolicy_pool: Annotated[
        int,
        tyro.conf.arg(
            help="Cached teacher continuations per word for off-policy SDPO "
            "(lazily sampled once, then reused). 0 = auto (2 * --sdpo-n-offpolicy). "
            "Must be >= --sdpo-n-offpolicy.",
        ),
    ] = 0
    sdpo_include_gold: Annotated[
        bool,
        tyro.conf.arg(
            help="Also distill over the gold target sequence in SDPO "
            "(--no-sdpo-include-gold to disable)."
        ),
    ] = True
    sdpo_definitions_path: Annotated[
        Optional[str],
        tyro.conf.arg(
            help="JSONL word definitions for the SDPO teacher (default: "
            "--definitions-path or data/semantle/train/definitions.jsonl)."
        ),
    ] = None
    output_dir: Annotated[str, tyro.conf.arg(help="Run output directory.")] = (
        "./reft_out"
    )
    init_from: Annotated[
        Optional[str],
        tyro.conf.arg(
            help="Initialize all learnable intervention parameters from a previous "
            "run's output_dir (intervenable_model/). Errors if that run's learnable "
            "parameter set does not match this run."
        ),
    ] = None
    load_latest: Annotated[
        bool,
        tyro.conf.arg(
            help="With --init-from, load weights from <dir>/latest/ when present "
            "(default: load the run root / best checkpoint)."
        ),
    ] = False
    seed: Annotated[int, tyro.conf.arg(help="Random seed.")] = 42

    # W&B
    wandb_project: Annotated[Optional[str], tyro.conf.arg(help="W&B project name.")] = (
        None
    )
    wandb_entity: Annotated[Optional[str], tyro.conf.arg(help="W&B entity/team.")] = (
        None
    )
    wandb_run_name: Annotated[
        Optional[str], tyro.conf.arg(help="W&B run display name.")
    ] = None
    wandb_group: Annotated[
        Optional[str], tyro.conf.arg(help="W&B run group (groups related runs in the UI).")
    ] = None
    wandb_dir: Annotated[Optional[str], tyro.conf.arg(help="W&B file directory.")] = (
        None
    )

    # During-training eval
    eval_steps: Annotated[
        Optional[int],
        tyro.conf.arg(help="Run Semantle/MolOpt eval every N optimizer steps."),
    ] = None
    eval_epochs: Annotated[
        Optional[int],
        tyro.conf.arg(help="Run Semantle eval every N epochs (semantle only)."),
    ] = None
    eval_n_samples: Annotated[
        Optional[int],
        tyro.conf.arg(help="Words to subsample for during-training eval."),
    ] = None
    eval_selection_metric: Annotated[
        Literal["embed_sim", "rdkit_sim", "tfs"],
        tyro.conf.arg(
            help="Which during-training eval metric drives early stopping and "
            "threshold checkpoints: embed_sim (embedding cosine, default), "
            "rdkit_sim (descriptor RBF) or tfs (Morgan Tanimoto). The two "
            "molecular metrics require a molopt-style task. Every metric is "
            "logged either way; this only picks the one --stop-threshold* and "
            "--checkpoint-embed-sim-thresholds* compare against."
        ),
    ] = "embed_sim"
    stop_threshold: Annotated[
        Optional[float],
        tyro.conf.arg(
            help="Stop when the mean eval --eval-selection-metric "
            "(embed_sim by default) exceeds this threshold."
        ),
    ] = None
    stop_threshold_min: Annotated[
        Optional[float],
        tyro.conf.arg(
            help="Per-target --eval-selection-metric bar (tau) for early stop. Combined with "
            "--stop-threshold-frac: stop when at least that fraction of eval "
            "targets score >= tau (frac=1 means every target / min >= tau). "
            "Combined with --stop-threshold when both are set (both must pass).",
        ),
    ] = None
    stop_threshold_frac: Annotated[
        float,
        tyro.conf.arg(
            help="Minimum fraction of eval targets that must have "
            "embed_sim >= --stop-threshold-min. Default 1.0 (all targets). "
            "Values < 1 require --stop-threshold-min.",
        ),
    ] = 1.0
    checkpoint_embed_sim_thresholds: Annotated[
        Optional[Tuple[float, ...]],
        tyro.conf.arg(
            help="Save checkpoint when mean eval embed_sim crosses each threshold."
        ),
    ] = None
    checkpoint_embed_sim_thresholds_min: Annotated[
        Optional[Tuple[float, ...]],
        tyro.conf.arg(
            help="Save checkpoint when min eval embed_sim (worst word in the eval "
            "subset) crosses each threshold. Checkpoints are saved independently of "
            "--checkpoint-embed-sim-thresholds.",
        ),
    ] = None
    checkpoint_epoch_interval: Annotated[
        Optional[int],
        tyro.conf.arg(help="Save eval-ready checkpoint every N epochs."),
    ] = None
    save_best_params: Annotated[
        bool,
        tyro.conf.arg(
            help="After each during-train eval, keep the best params so far under "
            "output_dir/best/ (by mean --eval-selection-metric). At end of training, "
            "promote best weights to the run root and write the final step to "
            "output_dir/latest/. With no eval, the root still gets the final weights. "
            "Use --no-save-best-params to disable."
        ),
    ] = True

    # Post-training full eval (semantle)
    run_full_eval: Annotated[
        bool,
        tyro.conf.arg(help="Run full post-training eval pipeline after training."),
    ] = False
    full_eval_n_samples: Annotated[
        Optional[int],
        tyro.conf.arg(help="Word subset size for post-training eval (all if unset)."),
    ] = None
    full_eval_gen_samples: Annotated[
        int,
        tyro.conf.arg(help="Stochastic generations per word in post-training eval."),
    ] = 25
    full_eval_top_p: Annotated[
        float,
        tyro.conf.arg(
            help="Nucleus top-p for all post-training temperature-sampling passes "
            "(RECON/DIST/GENZ/LIPZ)."
        ),
    ] = 1.0
    full_eval_interp: Annotated[
        Optional[Tuple[str, ...]],
        tyro.conf.arg(
            help="Interpolation (LIPZ) eval pairs: a single integer N to sample N "
            "random pairs from the eval vocabulary, or explicit alternating word "
            "pairs (word1 word2 ...). Default: 8 random pairs."
        ),
    ] = ("8",)
    full_eval_n_uniform: Annotated[
        int,
        tyro.conf.arg(help="Sobol samples for space-coverage eval."),
    ] = 1000
    full_eval_batch_size: Annotated[
        int,
        tyro.conf.arg(help="Batch size for post-training generation eval."),
    ] = 64
    full_eval_interp_n_samples: Annotated[
        int,
        tyro.conf.arg(help="Generations per interpolation step for each T variant."),
    ] = 10
    full_eval_interp_t_steps: Annotated[
        int,
        tyro.conf.arg(help="Number of interpolation steps for LIPZ eval."),
    ] = 101
    interp_method: Annotated[
        Literal["lerp", "slerp"],
        tyro.conf.arg(
            help="Bias interpolation method for LIPZ eval: lerp (linear) or slerp."
        ),
    ] = "lerp"
    full_eval_max_new_tokens: Annotated[
        int,
        tyro.conf.arg(
            help="Max tokens per decode in post-training generation and interpolation eval."
        ),
    ] = DEFAULT_MAX_NEW_TOKENS
    eval_embed_sim_tau: Annotated[
        float,
        tyro.conf.arg(
            help="RECON threshold for recon/embed_sim_gte_tau (fraction of train "
            "targets whose greedy embed_sim >= tau). Independent of --stop-threshold."
        ),
    ] = 0.8
    test_n_samples: Annotated[
        int,
        tyro.conf.arg(
            help="RECON_TEST / GENZ: number of held-out targets. For molopt with "
            "an oracle cap, sampled from the high-tail (p90–100) pool."
        ),
    ] = 512
    full_eval_oracle_screen: Annotated[
        bool,
        tyro.conf.arg(
            help="MolOpt full eval: score train / high-tail test / Sobol-decoded "
            "B on DRD2, GSK3B, and JNK3."
        ),
    ] = True
    full_eval_oracle_screen_n_sobol: Annotated[
        int,
        tyro.conf.arg(
            help="Sobol codes to greedy-decode for the molopt oracle screen "
            "(0 skips decode and scores GENZ SMILES when present)."
        ),
    ] = 2048
    eval_bbox_pca_var: Annotated[
        float,
        tyro.conf.arg(
            help="GENZ: PCA variance ratio retained when fitting the train-embedding "
            "bounding box that splits test targets into interp vs extrap sets."
        ),
    ] = 0.9
    semantle_dir: Annotated[
        Optional[str],
        tyro.conf.arg(help="Semantle CSV directory for cluster PCA/silhouette eval."),
    ] = None

    # Intervention bias
    use_word_bias: Annotated[
        bool,
        tyro.conf.arg(help="Use per-word bias vectors (VAE/linear intervention)."),
    ] = True
    bias_type: Annotated[
        Literal["vae", "linear"],
        tyro.conf.arg(help="Intervention type: vae or linear."),
    ] = "vae"
    variance: Annotated[
        Union[str, float],
        tyro.conf.arg(help="Bias variance: 'learnable' or a positive float."),
    ] = "learnable"
    sigma_b: Annotated[float, tyro.conf.arg(help="Initial bias scale (sigma_b).")] = 0.1
    add_bias_network: Annotated[
        bool,
        tyro.conf.arg(
            help="Map frozen target embeddings (embed_cache) to bias vectors via a learned MLP "
            "instead of per-word embedding tables. Incompatible with lambda_mm/lambda_rank. "
            "Exports bias_tables.pt after training for fast eval lookup.",
        ),
    ] = False
    bias_network_residual: Annotated[
        bool,
        tyro.conf.arg(
            help="Add linear skip on mu: mu = W@e + f(e) (requires --add-bias-network).",
        ),
    ] = False
    bias_network_encoder: Annotated[
        Literal["sentence_transformer", "llm_encoder"],
        tyro.conf.arg(
            help="Bias-network input source (requires --add-bias-network). "
            "'sentence_transformer' (default): frozen sentence-transformer "
            "embed_cache rows. 'llm_encoder': a copied final decoder block with "
            "LoRA over the frozen base model, pooled over each word's definition "
            "text (--bias-encoder-pooling). The encoder LoRA is used only to compute bias "
            "vectors; text generation never uses it.",
        ),
    ] = "sentence_transformer"
    bias_network_embed_model: Annotated[
        Optional[str],
        tyro.conf.arg(
            help="sentence-transformers model for the bias network's inputs "
            "(requires --bias-network-encoder sentence_transformer). Defaults to the "
            "task's own embedding model. Set this to train the bias network in a "
            "different embedding space (e.g. a chemistry encoder for molopt) while "
            "embed_sim and every reported similarity metric stay on the task model.",
        ),
    ] = None
    bias_encoder_lora_rank: Annotated[
        int,
        tyro.conf.arg(help="LoRA rank for the llm_encoder bias network."),
    ] = 8
    bias_encoder_lora_alpha: Annotated[
        float,
        tyro.conf.arg(help="LoRA alpha (scaling = alpha / rank) for the llm_encoder."),
    ] = 16.0
    bias_encoder_lora_dropout: Annotated[
        float,
        tyro.conf.arg(help="LoRA input dropout for the llm_encoder."),
    ] = 0.0
    bias_encoder_layer: Annotated[
        Optional[int],
        tyro.conf.arg(
            help="Base-model decoder layer to copy for the llm_encoder "
            "(default: final layer). Negative indexes from the end.",
        ),
    ] = None
    bias_encoder_max_length: Annotated[
        int,
        tyro.conf.arg(
            help="Max token length for llm_encoder definition inputs.",
        ),
    ] = 128
    bias_encoder_pooling: Annotated[
        Literal["last_instruction", "instruction_mean", "last_token"],
        tyro.conf.arg(
            help="LLM-encoder pooling over definition inputs (requires llm_encoder). "
            "'last_instruction' (default): final non-template instruction token "
            "(legacy alias: last_token). 'instruction_mean': mean over all "
            "non-template instruction tokens (definition span).",
        ),
    ] = "last_instruction"
    dropout: Annotated[float, tyro.conf.arg(help="Intervention dropout.")] = 0.0
    dropout_on_b: Annotated[
        float,
        tyro.conf.arg(
            help="Dropout on per-word bias vectors in low-rank space (0 = off). "
            "Applied only for embedding lookups, not raw interpolation vectors.",
        ),
    ] = 0.0

    # Neighborhood augmentation (semantle)
    nbr_lambda: Annotated[
        float,
        tyro.conf.arg(
            help="Neighborhood augmentation weight (>0 enables weighted CE)."
        ),
    ] = 0.0
    nbr_top_k: Annotated[
        int, tyro.conf.arg(help="Top-k neighbors for augmentation.")
    ] = 10
    aug_batch_mode: Annotated[
        Literal["block", "shuffle"],
        tyro.conf.arg(help="Neighborhood batching: block or shuffle."),
    ] = "block"

    # Chat template (semantle)
    use_chat_template: Annotated[
        bool,
        tyro.conf.arg(help="Wrap prompts with the tokenizer chat template. Requires an instruct-model (e.g., meta-llama/Llama-3.2-1B-Instruct)"),
    ] = False
    chat_instruction: Annotated[
        Optional[str],
        tyro.conf.arg(
            help="User instruction for chat template. "
            "Defaults to task_config chat_prompt for the current task."
        ),
    ] = None

    # AdamW weight decay (optimizer; see lambda_l2 for loss-term L2 on batch mu).
    weight_decay_mode: Annotated[
        Literal["none", "W_only", "b_only", "W_and_b"],
        tyro.conf.arg(help="AdamW weight decay mode for W and/or bias params."),
    ] = "none"
    wd_W: Annotated[
        float, tyro.conf.arg(help="Weight decay on learned_source (W).")
    ] = 1e-4
    wd_b: Annotated[
        float,
        tyro.conf.arg(help="Weight decay on word_mu / word_bias / bias_network (not word_logvar)."),
    ] = 1e-4

    def __post_init__(self) -> None:
        object.__setattr__(self, "variance", _normalize_variance(self.variance))
        _validate_train_config(self)

    @property
    def full_eval_interp_list(self) -> Optional[List[str]]:
        if self.full_eval_interp is None:
            return None
        return list(self.full_eval_interp)

    @property
    def checkpoint_embed_sim_thresholds_list(self) -> Optional[List[float]]:
        if self.checkpoint_embed_sim_thresholds is None:
            return None
        return list(self.checkpoint_embed_sim_thresholds)

    @property
    def checkpoint_embed_sim_thresholds_min_list(self) -> Optional[List[float]]:
        if self.checkpoint_embed_sim_thresholds_min is None:
            return None
        return list(self.checkpoint_embed_sim_thresholds_min)


def parse_train_config() -> TrainConfig:
    """Parse CLI arguments into a validated TrainConfig (via tyro)."""
    return tyro.cli(TrainConfig)
