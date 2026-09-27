"""
Generic dataset primitives for dataset-agnostic REFT training.

ReftDataset       — tokenizes ReftItem list for REFT training
ReftDataCollator  — batches ReftDataset samples
"""

import argparse
import json
import math
import os
import random
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq, TrainerCallback

from boreft.chem import unwrap_smiles_tags
from boreft.data import ReftItem
from boreft.intervention_marker import (
    intervention_locations_for_prompt,
    is_content_position,
    is_marker_position,
)
from boreft.data.base import parse_positions

IGNORE_INDEX = -100

TORCH_DTYPE_NAMES = ("bfloat16", "float16", "float32")

TRAINING_CONFIG_NAME = "training_config.json"
INTERVENTION_CONFIG_NAME = "intervention_config.json"
LATEST_CHECKPOINT_DIRNAME = "latest"
BEST_CHECKPOINT_DIRNAME = "best"
BEST_CHECKPOINT_INFO_NAME = "best_checkpoint_info.json"
LOAD_LATEST_HELP = (
    "Load intervenable weights from <dir>/latest/ when present "
    "(default: run root / best)."
)


def add_load_latest_argument(
    parser: argparse.ArgumentParser,
    *,
    help: str = LOAD_LATEST_HELP,
) -> None:
    """Add ``--load-latest`` / ``--no-load-latest`` (dest: ``load_latest``)."""
    parser.add_argument(
        "--load-latest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=help,
    )


def checkpoint_dir_has_intervenable_weights(path: str) -> bool:
    """True when ``path/intervenable_model/`` looks like a saved intervention."""
    return bool(list_intervenable_weight_files(path))


def resolve_checkpoint_dir(path: str, *, load_latest: bool = False) -> str:
    """Resolve a run directory to the tree that should be loaded.

    By default returns ``path`` (best / canonical root after training). When
    ``load_latest`` is true and ``path/latest/`` contains intervenable weights,
    returns that subdirectory instead.
    """
    root = os.path.abspath(path)
    if not load_latest:
        return root
    latest = os.path.join(root, LATEST_CHECKPOINT_DIRNAME)
    if checkpoint_dir_has_intervenable_weights(latest):
        return latest
    return root


def best_checkpoint_dir(output_dir: str) -> str:
    return os.path.join(os.path.abspath(output_dir), BEST_CHECKPOINT_DIRNAME)


def latest_checkpoint_dir(output_dir: str) -> str:
    return os.path.join(os.path.abspath(output_dir), LATEST_CHECKPOINT_DIRNAME)


def has_best_checkpoint(output_dir: str) -> bool:
    return checkpoint_dir_has_intervenable_weights(best_checkpoint_dir(output_dir))


def resolve_weight_dir_and_run_config(
    path: str, *, load_latest: bool = False
) -> Tuple[str, dict]:
    """Return ``(weight_dir, merged_cfg)`` for a run root.

    ``weight_dir`` is ``path`` or ``path/latest`` when ``load_latest`` and that
    subtree has intervenable weights. Merged config always overlays the run-root
    ``training_config.json`` when weights come from a subdirectory, so loaders
    keep canonical run metadata even if ``latest/`` lacks a training config copy.
    """
    root = os.path.abspath(path)
    weight_dir = resolve_checkpoint_dir(root, load_latest=load_latest)
    cfg = load_merged_run_config(weight_dir)
    if weight_dir != root:
        root_train = load_training_config(root)
        if root_train:
            cfg = {**cfg, **root_train}
    return weight_dir, cfg


def _atomic_replace_dir(src_dir: str, dst_dir: str) -> None:
    """Replace ``dst_dir`` with a copy of ``src_dir`` (same-filesystem swap)."""
    parent = os.path.dirname(os.path.abspath(dst_dir)) or "."
    staging_parent = tempfile.mkdtemp(prefix=".ckpt_promote_", dir=parent)
    staged = os.path.join(staging_parent, "new")
    trash = os.path.join(staging_parent, "old")
    try:
        shutil.copytree(src_dir, staged)
        if os.path.lexists(dst_dir):
            os.replace(dst_dir, trash)
        os.replace(staged, dst_dir)
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


def _atomic_replace_file(src_file: str, dst_file: str) -> None:
    """Replace ``dst_file`` with a copy of ``src_file`` via a same-dir temp."""
    parent = os.path.dirname(os.path.abspath(dst_file)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".ckpt_promote_", dir=parent)
    os.close(fd)
    try:
        shutil.copy2(src_file, tmp)
        os.replace(tmp, dst_file)
    except Exception:
        if os.path.isfile(tmp):
            os.remove(tmp)
        raise


def promote_best_checkpoint_to_root(output_dir: str) -> Optional[dict]:
    """Copy ``best/`` weights into the run root; return best checkpoint_info if any.

    Atomically replaces root ``intervenable_model/`` and ``bias_tables.pt`` when
    present. Updates root ``intervention_config.json`` ``bias_tables_path`` when
    tables are promoted. Leaves tokenizer / items / training configs at the root
    untouched.
    """
    root = os.path.abspath(output_dir)
    best = best_checkpoint_dir(root)
    if not checkpoint_dir_has_intervenable_weights(best):
        return None

    _atomic_replace_dir(
        os.path.join(best, "intervenable_model"),
        os.path.join(root, "intervenable_model"),
    )

    src_tables = os.path.join(best, "bias_tables.pt")
    dst_tables = os.path.join(root, "bias_tables.pt")
    if os.path.isfile(src_tables):
        _atomic_replace_file(src_tables, dst_tables)
        cfg_path = os.path.join(root, INTERVENTION_CONFIG_NAME)
        if os.path.isfile(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["bias_materialized"] = True
            cfg["bias_tables_path"] = os.path.abspath(dst_tables)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
                f.write("\n")
    elif os.path.isfile(dst_tables):
        # Best had no tables; drop a stale root tables file from a prior write.
        os.remove(dst_tables)

    info = None
    info_path = os.path.join(best, "checkpoint_info.json")
    if os.path.isfile(info_path):
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        with open(
            os.path.join(root, BEST_CHECKPOINT_INFO_NAME), "w", encoding="utf-8"
        ) as f:
            json.dump(info, f, indent=2)
            f.write("\n")
    return info


def semantle_csv_abspaths(
    csv_paths: Optional[Union[Tuple[str, ...], List[str]]],
) -> Optional[List[str]]:
    if not csv_paths:
        return None
    return [os.path.abspath(p) for p in csv_paths]


def load_checkpoint_tokenizer(output_dir: str, **kwargs):
    """Load a tokenizer saved with a training run.

    Prefer the fast tokenizer only when ``tokenizer.json`` is present so older
    slow-only Llama checkpoints keep their original encoding. Passing
    ``use_fast=True`` unconditionally would convert slow saves and risk subtle
    tokenization drift vs training. ``use_fast`` in ``kwargs`` is ignored.
    """
    from transformers import AutoTokenizer

    use_fast = os.path.isfile(os.path.join(output_dir, "tokenizer.json"))
    kwargs.pop("use_fast", None)
    return AutoTokenizer.from_pretrained(output_dir, use_fast=use_fast, **kwargs)


def load_intervention_config(output_dir: str) -> dict:
    path = os.path.join(output_dir, INTERVENTION_CONFIG_NAME)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_training_config(output_dir: str) -> dict:
    path = os.path.join(output_dir, TRAINING_CONFIG_NAME)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_merged_run_config(output_dir: str) -> dict:
    """Intervention config overlaid with training config (training wins on duplicate keys)."""
    intervention = load_intervention_config(output_dir)
    training = load_training_config(output_dir)
    return {**intervention, **training}


# Runtime / data buffers that a saved intervention may contain but that are not
# learnable parameters. Copied ``embed_cache`` would overwrite the current run's
# frozen target embeddings; encoder penultimate inputs are rebuilt at train time.
_INIT_FROM_SKIP_BUFFERS = frozenset(
    {
        "embed_cache",
        "encoder_penult",
        "encoder_attn",
        "encoder_instruction_mask",
    }
)


def _intervention_module(reft_model):
    iv = list(reft_model.interventions.values())[0]
    return iv[0] if isinstance(iv, (list, tuple)) else iv


def learnable_parameter_spec(module) -> Dict[str, Tuple[int, ...]]:
    """Name → shape for parameters with ``requires_grad=True``."""
    return {
        name: tuple(param.shape)
        for name, param in module.named_parameters()
        if param.requires_grad
    }


def list_intervenable_weight_files(output_dir: str) -> List[str]:
    """Return ``intkey_*.bin`` paths under ``output_dir/intervenable_model/``."""
    intervenable = os.path.join(output_dir, "intervenable_model")
    if not os.path.isdir(intervenable):
        return []
    return sorted(
        os.path.join(intervenable, name)
        for name in os.listdir(intervenable)
        if name.startswith("intkey_") and name.endswith(".bin")
    )


def _load_torch_state_dict(path: str):
    try:
        saved = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        saved = torch.load(path, map_location="cpu")
    return saved


def load_intervenable_state_dict(output_dir: str) -> dict:
    """Load the single ``intkey_*.bin`` saved under ``output_dir/intervenable_model/``."""
    intervenable = os.path.join(output_dir, "intervenable_model")
    if not os.path.isdir(intervenable):
        raise FileNotFoundError(
            f"{output_dir}: no intervenable_model/ directory"
        )
    bins = list_intervenable_weight_files(output_dir)
    if not bins:
        raise FileNotFoundError(
            f"{output_dir}/intervenable_model: no intkey_*.bin intervention weights"
        )
    if len(bins) > 1:
        names = ", ".join(os.path.basename(path) for path in bins)
        raise ValueError(
            f"{output_dir}/intervenable_model: expected one intkey_*.bin, found {len(bins)} "
            f"({names})"
        )
    saved = _load_torch_state_dict(bins[0])
    if not isinstance(saved, dict):
        raise ValueError(
            f"{bins[0]}: expected a state_dict, got {type(saved).__name__}"
        )
    return saved


def _is_encoder_frozen_saved_key(key: str) -> bool:
    return "semantic_encoder." in key and ".lora_" not in key


def _parametrization_buffers(module) -> Dict[str, torch.Tensor]:
    """Orthogonal-parametrization buffers required to reconstruct ``R``."""
    return {
        name: buffer
        for name, buffer in module.named_buffers()
        if "parametrizations" in name
    }


def _format_init_from_incompatibility(
    *,
    source: str,
    missing: Sequence[str],
    extra: Sequence[str],
    shape_mismatches: Sequence[str],
) -> str:
    lines = [
        f"--init-from is incompatible with this run: learnable parameters do not match "
        f"({source})."
    ]
    if missing:
        lines.append(
            f"  missing from source ({len(missing)}): {', '.join(missing)}"
        )
    if extra:
        lines.append(f"  extra in source ({len(extra)}): {', '.join(extra)}")
    if shape_mismatches:
        lines.append(
            f"  shape mismatches ({len(shape_mismatches)}): {', '.join(shape_mismatches)}"
        )
    return "\n".join(lines)


def apply_init_from_state_dict(
    intervention,
    saved_sd: dict,
    *,
    source: str,
) -> int:
    """Copy learnable tensors from ``saved_sd`` onto ``intervention``.

    Compatibility is the set of learnable parameter names and shapes, plus the
    orthogonal parametrization buffers needed so ``R`` matches after load. Frozen
    data buffers such as ``embed_cache`` are left on the current module.

    Returns the number of learnable parameters copied.
    """
    learnable = {
        name: param
        for name, param in intervention.named_parameters()
        if param.requires_grad
    }
    rot_buffers = _parametrization_buffers(intervention)
    learnable_keys = set(learnable)
    saved_keys = set(saved_sd)

    dest_nonlearnable = set(_INIT_FROM_SKIP_BUFFERS)
    dest_nonlearnable.update(name for name, _ in intervention.named_buffers())
    dest_nonlearnable.update(
        name
        for name, param in intervention.named_parameters()
        if not param.requires_grad
    )

    missing = sorted(learnable_keys - saved_keys)
    missing.extend(
        sorted(
            name
            for name in rot_buffers
            if name not in saved_keys and name not in missing
        )
    )
    extra = sorted(
        key
        for key in saved_keys - learnable_keys
        if key not in dest_nonlearnable and not _is_encoder_frozen_saved_key(key)
    )
    shape_mismatches = []
    for name in sorted(learnable_keys & saved_keys):
        current_shape = tuple(learnable[name].shape)
        source_shape = tuple(saved_sd[name].shape)
        if current_shape != source_shape:
            shape_mismatches.append(
                f"{name} current={tuple(current_shape)} source={tuple(source_shape)}"
            )
    for name in sorted(set(rot_buffers) & saved_keys):
        current_shape = tuple(rot_buffers[name].shape)
        source_shape = tuple(saved_sd[name].shape)
        if current_shape != source_shape:
            shape_mismatches.append(
                f"{name} current={tuple(current_shape)} source={tuple(source_shape)}"
            )

    if missing or extra or shape_mismatches:
        raise ValueError(
            _format_init_from_incompatibility(
                source=source,
                missing=missing,
                extra=extra,
                shape_mismatches=shape_mismatches,
            )
        )

    with torch.no_grad():
        for name, param in learnable.items():
            param.copy_(
                saved_sd[name].to(device=param.device, dtype=param.dtype)
            )
        for name, buffer in rot_buffers.items():
            buffer.copy_(
                saved_sd[name].to(device=buffer.device, dtype=buffer.dtype)
            )
    return len(learnable)


def init_learnable_parameters_from_run(
    reft_model, init_from_dir: str, *, load_latest: bool = False
) -> int:
    """Load a previous run's intervention weights onto ``reft_model``."""
    resolved = resolve_checkpoint_dir(init_from_dir, load_latest=load_latest)
    saved_sd = load_intervenable_state_dict(resolved)
    intervention = _intervention_module(reft_model)
    return apply_init_from_state_dict(
        intervention, saved_sd, source=os.path.abspath(resolved)
    )


def resolve_torch_dtype(name: str) -> torch.dtype:
    """Map a dtype name string to ``torch.dtype``."""
    if name not in TORCH_DTYPE_NAMES:
        raise ValueError(
            f"torch_dtype must be one of {TORCH_DTYPE_NAMES}, got {name!r}"
        )
    return getattr(torch, name)


def infer_optimal_precision() -> str:
    """Pick the best default dtype for the current hardware."""
    if not torch.cuda.is_available():
        return "float32"
    device_id = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device_id)
    compute_capability = major + (minor / 10)
    if compute_capability >= 8.0:
        return "bfloat16"
    return "float16"


def infer_torch_dtype_name(
    *,
    override: Optional[str] = None,
    saved_cfg: Optional[dict] = None,
) -> str:
    """Resolve dtype name: CLI override > checkpoint config > hardware auto."""
    if override is not None:
        return override
    if saved_cfg and saved_cfg.get("torch_dtype") in TORCH_DTYPE_NAMES:
        return saved_cfg["torch_dtype"]
    return infer_optimal_precision()


def infer_training_load_dtype_name(
    *,
    override: Optional[str] = None,
    saved_cfg: Optional[dict] = None,
) -> str:
    """Dtype for ``from_pretrained`` during training.

    Auto on V100-class GPUs resolves to float16 for inference checkpoints, but
    training loads float32 master weights and uses fp16 AMP — native float16
    weights without AMP produce NaN gradients during backward through the LM.
    """
    name = infer_torch_dtype_name(override=override, saved_cfg=saved_cfg)
    if name == "float16" and torch.cuda.is_available():
        return "float32"
    return name


def trainer_amp_flags(load_dtype_name: str, *, use_cuda: bool) -> tuple[bool, bool]:
    """Return ``(bf16, fp16)`` for :class:`TrainingArguments`.

    ``load_dtype_name`` is the dtype passed to ``from_pretrained`` (see
    :func:`infer_training_load_dtype_name`). float32 weights on CUDA use
    hardware-appropriate mixed precision; native half-precision loads only
    enable bf16 autocast (never fp16 GradScaler, which requires fp32 params).
    """
    if not use_cuda:
        return False, False
    if load_dtype_name == "bfloat16":
        return True, False
    if load_dtype_name == "float32":
        auto = infer_optimal_precision()
        return auto == "bfloat16", auto == "float16"
    if load_dtype_name == "float16":
        return False, False
    return False, False


# Rust tokenizers overflow on absurd model_max_length sentinels (e.g. Gemma3).
_TRUNCATION_LENGTH_CAP = 10**7
_TRUNCATION_LENGTH_FALLBACK = 131072


def _tokenizer_max_length(tokenizer) -> int:
    raw = getattr(tokenizer, "model_max_length", None)
    if (
        raw is None
        or not isinstance(raw, int)
        or raw <= 0
        or raw > _TRUNCATION_LENGTH_CAP
    ):
        return _TRUNCATION_LENGTH_FALLBACK
    return raw


_CHAT_PLACEHOLDER = "<<<BOReFT>>>"
# Llama 3.x templates use strftime_now when date_string is unset; pin for reproducibility.
FIXED_CHAT_DATE = "26 Jul 2024"


def _require_chat_template(tokenizer) -> None:
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            f"Tokenizer {getattr(tokenizer, 'name_or_path', tokenizer)!r} has no chat_template; "
            "cannot use --use_chat_template."
        )


def prepend_system_message(
    messages: list[dict[str, str]],
    system_prompt: Optional[str],
) -> list[dict[str, str]]:
    """Put a system turn in front of ``messages`` when ``system_prompt`` is set."""
    text = (system_prompt or "").strip()
    if not text:
        return messages
    return [{"role": "system", "content": text}, *messages]


def system_prompt_from_cfg(saved_cfg: Optional[dict]) -> Optional[str]:
    """System message recorded on a checkpoint, or ``None`` if none was used."""
    if not saved_cfg:
        return None
    raw = saved_cfg.get("system_prompt")
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _apply_chat_template(tokenizer, conversation, **kwargs):
    """Render chat template with a fixed date (avoids dynamic strftime_now).

    ``enable_thinking`` defaults to False. For Qwen3, omitting it is treated as
    thinking-enabled; False prefills an empty ``<think>`` block so SFT targets
    are the answer only. Callers may pass True for free-form thinking generation.
    """
    _require_chat_template(tokenizer)
    kwargs.setdefault("date_string", FIXED_CHAT_DATE)
    kwargs.setdefault("enable_thinking", False)
    return tokenizer.apply_chat_template(conversation, **kwargs)


def chat_prompt(
    tokenizer,
    instruction: str,
    *,
    enable_thinking: bool = False,
    system_prompt: Optional[str] = None,
) -> str:
    return _apply_chat_template(
        tokenizer,
        prepend_system_message(
            [{"role": "user", "content": instruction.strip()}],
            system_prompt,
        ),
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=enable_thinking,
    )


def _assistant_tail(
    tokenizer,
    instruction: str,
    content: str,
    *,
    system_prompt: Optional[str] = None,
) -> str:
    # Always pair with enable_thinking=False so the generation prompt (empty
    # think prefill on Qwen3) matches the supervised target suffix.
    prompt = chat_prompt(
        tokenizer,
        instruction,
        enable_thinking=False,
        system_prompt=system_prompt,
    )
    full = _apply_chat_template(
        tokenizer,
        prepend_system_message(
            [
                {"role": "user", "content": instruction.strip()},
                {"role": "assistant", "content": content},
            ],
            system_prompt,
        ),
        add_generation_prompt=False,
        tokenize=False,
        enable_thinking=False,
    )
    if not full.startswith(prompt):
        raise ValueError(
            "Chat template prefix mismatch between prompt and full conversation."
        )
    return full[len(prompt) :]


def chat_assistant_suffix(
    tokenizer, instruction: str, *, system_prompt: Optional[str] = None
) -> str:
    return _assistant_tail(
        tokenizer, instruction, _CHAT_PLACEHOLDER, system_prompt=system_prompt
    ).replace(_CHAT_PLACEHOLDER, "", 1)


def _maybe_append_eos(tokenizer, text: str) -> str:
    eos = tokenizer.eos_token
    if eos and eos not in text:
        return text + eos
    return text


def chat_assistant_target(
    tokenizer,
    instruction: str,
    word: str,
    *,
    system_prompt: Optional[str] = None,
) -> str:
    tail = _assistant_tail(
        tokenizer, instruction, _CHAT_PLACEHOLDER, system_prompt=system_prompt
    )
    return _maybe_append_eos(tokenizer, tail.replace(_CHAT_PLACEHOLDER, word, 1))


def prompt_tokenization_from_cfg(saved_cfg: dict | None) -> bool:
    """True when prompts were pre-rendered with ``apply_chat_template`` (see ``use_chat_template`` in config)."""
    return bool(saved_cfg and saved_cfg.get("use_chat_template"))


def tokenize_model_text(
    tokenizer, text: str, *, from_chat_template: bool = False, **kwargs
):
    """Tokenize *text*; chat-template strings already include BOS/role markers."""
    return tokenizer(text, add_special_tokens=not from_chat_template, **kwargs)


def apply_chat_format(
    items: List[ReftItem],
    tokenizer,
    instruction: str,
    *,
    system_prompt: Optional[str] = None,
) -> str:
    """Rewrite items in place for chat-template training. Returns assistant suffix for decoding."""
    prompt = chat_prompt(tokenizer, instruction, system_prompt=system_prompt)
    for item in items:
        gen = item.target.strip()
        identity = getattr(item, "_raw_word", None) or gen
        item._raw_word = str(identity).strip()
        item.prompt = prompt
        item.target = chat_assistant_target(
            tokenizer, instruction, gen, system_prompt=system_prompt
        )
    return chat_assistant_suffix(tokenizer, instruction, system_prompt=system_prompt)


def sample_reft_items(
    items: List[ReftItem], n: int, seed: int
) -> Tuple[List[ReftItem], Optional[List[int]]]:
    """Randomly sample ``n`` items without replacement; reassign ``id`` to 0..n-1.

    Returns ``(sampled_items, original_indices)``. ``original_indices`` is ``None`` when
    no subsampling occurred (for aligning precomputed embed_cache rows).
    """
    if n <= 0:
        raise ValueError(f"sample size must be positive, got {n}")
    if n >= len(items):
        return list(items), None
    rng = random.Random(seed)
    chosen = rng.sample(list(enumerate(items)), n)
    sampled: List[ReftItem] = []
    original_indices: List[int] = []
    for new_id, (orig_id, item) in enumerate(chosen):
        item.id = new_id
        sampled.append(item)
        original_indices.append(orig_id)
    return sampled, original_indices


def reft_items_to_json(items: List[ReftItem]) -> List[Dict]:
    rows = []
    for it in items:
        row = {"id": it.id, "prompt": it.prompt, "target": it.target}
        raw = getattr(it, "_raw_word", None)
        if raw is not None:
            row["word"] = raw
        rows.append(row)
    return rows


def save_reft_checkpoint_dir(
    save_dir: str,
    *,
    reft_model,
    tokenizer,
    items: List[ReftItem],
    intervention_config: dict,
    checkpoint_info: Optional[dict] = None,
    training_config_src: Optional[str] = None,
) -> dict:
    """Write an eval-ready checkpoint tree under ``save_dir``.

    Returns the intervention config that was written (may include
    ``bias_materialized`` / ``bias_tables_path`` when a bias network is exported).
    Does not attach materialized tables to the in-memory intervention.
    When ``training_config_src`` is a file path, copies it as
    ``training_config.json`` into ``save_dir``.
    """
    os.makedirs(save_dir, exist_ok=True)
    reft_model.save_intervention(
        save_directory=os.path.join(save_dir, "intervenable_model"),
        include_model=False,
    )
    cfg = dict(intervention_config)
    with open(os.path.join(save_dir, "items.json"), "w", encoding="utf-8") as f:
        json.dump(reft_items_to_json(items), f, indent=2)
    tokenizer.save_pretrained(save_dir)
    if cfg.get("add_bias_network"):
        from boreft.bias_tables import export_bias_tables_for_checkpoint

        tables_path = export_bias_tables_for_checkpoint(
            reft_model,
            save_dir,
            len(items),
            update_config=False,
            attach_in_memory=False,
        )
        if tables_path is not None:
            cfg["bias_materialized"] = True
            cfg["bias_tables_path"] = tables_path
    with open(
        os.path.join(save_dir, "intervention_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    if checkpoint_info is not None:
        with open(
            os.path.join(save_dir, "checkpoint_info.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(checkpoint_info, f, indent=2)
            f.write("\n")
    if training_config_src and os.path.isfile(training_config_src):
        shutil.copy2(
            training_config_src, os.path.join(save_dir, TRAINING_CONFIG_NAME)
        )
    return cfg


class EpochIntervalCheckpoint(TrainerCallback):
    """Save eval-ready intervention checkpoints every N completed epochs."""

    def __init__(
        self,
        reft_model,
        tokenizer,
        items: List[ReftItem],
        output_dir: str,
        intervention_config: dict,
        interval_epochs: int,
    ):
        if interval_epochs <= 0:
            raise ValueError(
                f"checkpoint_epoch_interval must be positive, got {interval_epochs}"
            )
        self.reft_model = reft_model
        self.tokenizer = tokenizer
        self.items = list(items)
        self.output_dir = output_dir
        self.intervention_config = intervention_config
        self.interval_epochs = interval_epochs

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        if epoch == 0 or epoch % self.interval_epochs != 0:
            return
        sub = os.path.join(
            self.output_dir, "checkpoints_by_epoch", f"epoch_{epoch:04d}"
        )
        save_reft_checkpoint_dir(
            sub,
            reft_model=self.reft_model,
            tokenizer=self.tokenizer,
            items=self.items,
            intervention_config=self.intervention_config,
            checkpoint_info={"epoch": epoch, "global_step": state.global_step},
        )
        print(f"[checkpoint] epoch {epoch} -> saved {sub}", flush=True)


def resolve_assistant_suffix(
    tokenizer, saved_cfg: dict, default_instruction: str
) -> Optional[str]:
    if not saved_cfg.get("use_chat_template"):
        return None
    from boreft.intervention_marker import resolve_instruction_for_prompt

    instruction = saved_cfg.get("chat_instruction") or default_instruction
    from boreft.chem import maybe_append_mist_smiles_open_tag, mist_open_tag_in_prompt

    token_id = saved_cfg.get("intervention_token_id")
    if token_id is not None:
        token_id = int(token_id)
    instruction = resolve_instruction_for_prompt(
        instruction,
        saved_cfg.get("intervention_inject", "none"),
        saved_cfg.get("intervention_token"),
        tokenizer=tokenizer,
        intervention_token_id=token_id,
    )
    instruction = maybe_append_mist_smiles_open_tag(
        instruction,
        mist_open_tag_in_prompt(
            mist_smiles_tags=bool(saved_cfg.get("mist_smiles_tags")),
            use_chat_template=bool(saved_cfg.get("use_chat_template")),
        ),
    )
    return chat_assistant_suffix(
        tokenizer, instruction, system_prompt=system_prompt_from_cfg(saved_cfg)
    )


def decode_generated_text(
    tokenizer,
    output_ids,
    prompt_len: int,
    *,
    assistant_suffix: Optional[str] = None,
    batch_idx: int = 0,
) -> str:
    """Decode generated tokens after ``prompt_len``.

    Truncates at the first ``eos_token_id``, then decodes with special tokens
    stripped. This avoids chat-template trailer mismatches where
    ``assistant_suffix`` includes a trailing newline (Qwen/Gemma) that
    ``generate`` never emits because it stops at EOS.

    ``assistant_suffix`` is accepted for call-site compatibility but unused.
    """
    del assistant_suffix  # kept for API compatibility with chat-template callers
    new_ids = output_ids[batch_idx, prompt_len:]
    ids = new_ids.tolist() if hasattr(new_ids, "tolist") else list(new_ids)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None and eos_id in ids:
        ids = ids[: ids.index(eos_id)]
    return unwrap_smiles_tags(tokenizer.decode(ids, skip_special_tokens=True).strip())


def build_reft_row(
    *,
    input_ids: Union[List[int], torch.Tensor],
    prompt_len: int,
    prompt_ids: Sequence[int],
    word_id: int,
    pad_token_id: int,
    position: str,
    intervention_token_id: Optional[int] = None,
    content_span: Optional[Tuple[int, int]] = None,
    num_interventions: int = 1,
    share_weights: bool = False,
    pad_mode: str = "first",
) -> Dict:
    """Build one REFT training/scoring row from token ids.

    Shared by :class:`ReftDataset` and the SDPO loss so on-/off-policy sampled
    sequences get the exact same leading-pad insertion, ``+1`` location shift,
    intervention-location computation, and subspace word-id encoding that normal
    training uses. ``input_ids`` is the full ``[prompt ; target]`` sequence;
    ``prompt_ids`` are the prompt-only ids (used to locate marker/content spans).
    Returns a dict with tensor ``input_ids``/``labels``/``attention_mask``, the
    nested ``intervention_locations`` list, and ``subspaces`` ``[[word_id]]``.
    """
    input_ids = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else list(input_ids)
    labels = list(input_ids)
    for i in range(min(prompt_len, len(labels))):
        labels[i] = IGNORE_INDEX

    intervention_locations = intervention_locations_for_prompt(
        position,
        prompt_len,
        list(prompt_ids),
        intervention_token_id=intervention_token_id,
        num_interventions=num_interventions,
        share_weights=share_weights,
        pad_mode=pad_mode,
        content_span=content_span,
    )

    if pad_mode == "first":
        input_ids = [pad_token_id] + input_ids
        labels = [IGNORE_INDEX] + labels
        intervention_locations = (
            torch.tensor(intervention_locations, dtype=torch.int) + 1
        ).tolist()

    input_ids_t = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = (input_ids_t != pad_token_id).int()
    return {
        "input_ids": input_ids_t,
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": attention_mask,
        "intervention_locations": intervention_locations,
        "subspaces": torch.tensor([[word_id]], dtype=torch.long),
    }


class ReftDataset(Dataset):
    """
    Generic dataset that works with any List[ReftItem].

    Tokenises `prompt + " " + target + eos_token`, masks the prompt portion
    in labels with IGNORE_INDEX, computes intervention locations inside the
    prompt, and stores item.id as the subspace integer (word ID) for
    DistributionalWordIntervention.

    When ``use_sample_weights`` is True (neighborhood-augmented Semantle training),
    each sample includes a ``weight`` field for weighted CE in the trainer.

    **Leading pad (``pad_mode="first"``)** — inherited from pyreft: each example
    prepends one ``pad_token_id`` at index 0 before batch collation, then
    ``intervention_locations`` are shifted by +1 so hooks stay on the same
    semantic tokens. That dummy pad is masked out of the loss (``IGNORE_INDEX``)
    and attention (``attention_mask`` 0). Eval/generation paths do *not* insert
    this pad, so training indices are offset by one relative to eval for the same
    prompt text. The pad also serves as the target for dummy intervention slots
    (``get_intervention_locations`` sentinel ``-1`` → index 0 after the shift)
    when prefix/suffix location lists are padded to equal length.
    """

    def __init__(
        self,
        items: List[ReftItem],
        tokenizer,
        position: str = "l1",
        num_interventions: int = 1,
        share_weights: bool = False,
        seed: int = 42,
        use_sample_weights: bool = False,
        use_chat_template: bool = False,
        intervention_token_id: Optional[int] = None,
        content_span: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.items = list(items)
        self.tokenizer = tokenizer
        self.position = position
        self.is_marker = is_marker_position(position)
        self.is_content = is_content_position(position)
        if not self.is_marker and not self.is_content:
            self.first_n, self.last_n = parse_positions(position)
        self.intervention_token_id = intervention_token_id
        self.content_span = content_span
        self.num_interventions = num_interventions
        self.share_weights = share_weights
        self.pad_mode = "first"
        self.seed = seed
        self.use_sample_weights = use_sample_weights
        self.use_chat_template = use_chat_template
        self.samples = self._build()

    def _build(self) -> List[Dict]:
        samples = []
        for item in self.items:
            if self.use_chat_template:
                full_input = item.prompt + item.target.strip()
            else:
                full_input = (
                    item.prompt + " " + item.target.strip() + self.tokenizer.eos_token
                )

            prompt_ids = tokenize_model_text(
                self.tokenizer,
                item.prompt,
                from_chat_template=self.use_chat_template,
                return_tensors="pt",
            )["input_ids"][0]

            input_ids = tokenize_model_text(
                self.tokenizer,
                full_input,
                from_chat_template=self.use_chat_template,
                return_tensors="pt",
            )["input_ids"][0]

            prompt_len = len(prompt_ids)
            built = build_reft_row(
                input_ids=input_ids,
                prompt_len=prompt_len,
                prompt_ids=prompt_ids.tolist(),
                word_id=item.id,
                pad_token_id=self.tokenizer.pad_token_id,
                position=self.position,
                intervention_token_id=self.intervention_token_id,
                content_span=self.content_span,
                num_interventions=self.num_interventions,
                share_weights=self.share_weights,
                pad_mode=self.pad_mode,
            )

            row = {"id": item.id, **built}
            # Only when neighborhood augmentation is enabled (see train.py): per-example CE weights.
            if self.use_sample_weights:
                row["weight"] = torch.tensor(
                    getattr(item, "weight", 1.0), dtype=torch.float32
                )
            samples.append(row)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return deepcopy(self.samples[idx])


@dataclass
class ReftDataCollator:
    """
    Batches ReftDataset samples.  Strips non-standard keys before passing to
    DataCollatorForSeq2Seq, then re-attaches subspaces and intervention_locations
    as tensors.
    """

    tokenizer: object
    model: object

    def __post_init__(self):
        self.inner = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=IGNORE_INDEX,
            padding="longest",
        )

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        instances = [deepcopy(inst) for inst in instances]
        for inst in instances:
            for key in ("input_ids", "labels", "attention_mask"):
                if key in inst and isinstance(inst[key], torch.Tensor):
                    inst[key] = inst[key].tolist()
        subspaces_list = [
            inst.pop("subspaces") for inst in instances if "subspaces" in inst
        ]
        intloc_list = [
            inst.pop("intervention_locations")
            for inst in instances
            if "intervention_locations" in inst
        ]
        weights_list = [inst.pop("weight") for inst in instances if "weight" in inst]
        _ = [inst.pop("id", None) for inst in instances]

        batch = self.inner(instances)
        max_seq_len = batch["input_ids"].shape[-1]

        if subspaces_list:
            stacked = [
                s if isinstance(s, torch.Tensor) else torch.tensor(s)
                for s in subspaces_list
            ]
            batch["subspaces"] = torch.stack(stacked, dim=0)

        if intloc_list:
            loc_tensor = torch.tensor(intloc_list, dtype=torch.long)
            batch["intervention_locations"] = loc_tensor[..., :max_seq_len]

        if weights_list:
            batch["weights"] = torch.stack(weights_list, dim=0)  # [batch]

        return batch


class NeighborhoodBlockBatchSampler:
    """Batches dataset indices so each batch is ``blocks_per_batch`` contiguous anchor blocks.

    Dataset layout must be: repeated blocks of length ``block_size`` (anchor row + k
    neighbor rows), as produced by ``SemantleItem.build_augmented_items``. Each yielded
    batch is ``blocks_per_batch * block_size`` indices: ``c`` whole blocks, shuffled
    at the block level when ``shuffle=True``.

    Requires ``dataset_len == num_blocks * block_size``. Each batch has fixed size
    ``blocks_per_batch * block_size`` (``drop_last`` drops a final partial group of blocks).
    """

    def __init__(
        self,
        dataset_len: int,
        block_size: int,
        blocks_per_batch: int,
        *,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if dataset_len % block_size != 0:
            raise ValueError(
                f"dataset length {dataset_len} must be divisible by block_size {block_size}"
            )
        self.dataset_len = dataset_len
        self.block_size = block_size
        self.num_blocks = dataset_len // block_size
        self.blocks_per_batch = blocks_per_batch
        if self.blocks_per_batch <= 0:
            raise ValueError("blocks_per_batch must be positive")
        self.shuffle = shuffle
        self.seed = int(seed)
        self.drop_last = drop_last
        self._epoch = 0

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_blocks // self.blocks_per_batch
        return math.ceil(self.num_blocks / self.blocks_per_batch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        block_ids = list(range(self.num_blocks))
        if self.shuffle:
            rng.shuffle(block_ids)
        step = self.blocks_per_batch
        for start in range(0, len(block_ids), step):
            chunk = block_ids[start : start + step]
            if len(chunk) < step and self.drop_last:
                break
            batch: List[int] = []
            for b in chunk:
                lo = b * self.block_size
                batch.extend(range(lo, lo + self.block_size))
            yield batch
