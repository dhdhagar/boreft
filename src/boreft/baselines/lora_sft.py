"""LoRA SFT so Random samples from the BOReFT train-set support.

Semantle draws the same ``--train-top-k`` / ``--train-n-samples`` / ``--seed``
subset that ``boreft.train`` uses (canonical: top 4000 of ``computer.csv``, then
3072 items at seed 42). Molopt uses the checkpoint ``items.json`` SMILES (the
exact reconstruction set) unless ``--molopt-csv`` resamples. Supervises
prompt → target CE on a task-local LoRA attached to the frozen base LM. For
MiST molopt the prompt is the completion prefix plus ``[START_SMILES]`` and
the gold is ``SMILES [END_SMILES]``, matching BOReFT. The adapter is the
proposal model for a later Random search run; it does not reproduce ReFT
interventions.

``--epochs`` defaults to 10. After LoRA inject (B=0, base model) and every
``--eval-epochs`` (default 1, plus the last epoch), ``--n-eval`` (default 500)
samples the SFT prompt and reports train-set hit / uniqueness / coverage
in ``eval.json``. ``--save-epochs`` (default 1) writes ``adapters/epochNNN.pt``
so a later search run can pick a specific epoch. ``adapter.pt`` is still the
best ``shift_score`` so far; ``adapter_last.pt`` is the final epoch.

  python -m boreft.baselines.lora_sft --reft-output-dir outputs/1784053292
  python -m boreft.baselines.lora_sft --reft-output-dir outputs/1789222254 --from-checkpoint-items
  python -m boreft.baselines.lora_sft --eval-only --output-dir experiments/outputs/semantle/lora_sft_random
"""


from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq

from boreft.baselines.sdpo_ttt import LoRALinear, _inject_lora
from boreft.chem import (
    canonical_target_key,
    maybe_append_mist_smiles_open_tag,
    mist_open_tag_in_prompt,
    unwrap_smiles_tags,
    validity_rate,
)
from boreft.data.base import item_target
from boreft.data.molopt import MolOptItem, apply_mist_smiles_tags, apply_smiles_tags
from boreft.data.semantle import SemantleItem
from boreft.data_utils import (
    IGNORE_INDEX,
    apply_chat_format,
    chat_prompt,
    infer_training_load_dtype_name,
    load_merged_run_config,
    resolve_torch_dtype,
    sample_reft_items,
    system_prompt_from_cfg,
    tokenize_model_text,
)
from boreft.search import normalize_search_text
from boreft.task_config import task_instruction, task_system_prompt

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV = REPO_ROOT / "data" / "semantle" / "train" / "computer.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "outputs" / "semantle" / "lora_sft_random_e10"
DEFAULT_MOLOPT_OUTPUT_DIR = (
    REPO_ROOT / "experiments" / "outputs" / "molopt" / "lora_sft_random"
)
DEFAULT_TRAIN_TOP_K = 4000
DEFAULT_TRAIN_N_SAMPLES = 3072
DEFAULT_SEED = 42
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
SEARCH_TASK_DESCRIPTION = (
    "Generate an English word as a guess to find the hidden word "
    "(only the word, without any decoration or formatting)."
)
ADAPTER_NAME = "adapter.pt"
ADAPTER_LAST_NAME = "adapter_last.pt"
ADAPTERS_DIR = "adapters"
WORDS_NAME = "words.json"
CONFIG_NAME = "config.json"
METRICS_NAME = "metrics.json"
EVAL_NAME = "eval.json"
DEFAULT_EPOCHS = 10
DEFAULT_N_EVAL = 500
DEFAULT_EVAL_EPOCHS = 1
DEFAULT_SAVE_EPOCHS = 1
EVAL_BATCH_SIZE = 32
EVAL_MAX_NEW_TOKENS = 16
MOLOPT_EVAL_MAX_NEW_TOKENS = 128


def resolve_semantle_csv(path: str, *, repo_root: Path = REPO_ROOT) -> str:
    """Return an existing CSV path; fall back to ``data/semantle/train/<name>``."""
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    local = repo_root / "data" / "semantle" / "train" / candidate.name
    if local.is_file():
        return str(local.resolve())
    raise FileNotFoundError(f"Semantle CSV not found: {path}")


def load_semantle_train_items(
    csv_paths: Sequence[str],
    *,
    train_top_k: int,
    train_n_samples: int | None,
    seed: int,
) -> list[SemantleItem]:
    """Same draw as ``boreft.train.load_training_items`` for task=semantle."""
    resolved = [resolve_semantle_csv(path) for path in csv_paths]
    _, words, sim_map = SemantleItem.load_csvs(list(resolved), top_k=train_top_k)
    items = SemantleItem.load(words, sim_map)
    if train_n_samples is None:
        return items
    sampled, _ = sample_reft_items(items, train_n_samples, seed=seed)
    return sampled


def load_semantle_train_words(
    csv_paths: Sequence[str],
    *,
    train_top_k: int,
    train_n_samples: int | None,
    seed: int,
) -> list[str]:
    return [item_target(item) for item in load_semantle_train_items(
        csv_paths,
        train_top_k=train_top_k,
        train_n_samples=train_n_samples,
        seed=seed,
    )]


def words_from_items_json(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a JSON list of training items")
    words: list[str] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{index}: expected a JSON object")
        if "word" in row and row["word"] is not None and str(row["word"]).strip():
            value = row["word"]
        else:
            value = row.get("target")
        if value is None or not str(value).strip():
            raise ValueError(f"{path}:{index}: JSON object must contain word or target")
        words.append(str(value).strip())
    return words


def resolve_task(task: str | None, saved: dict[str, Any]) -> str:
    if task:
        return str(task)
    saved_task = saved.get("task")
    if saved_task:
        return str(saved_task)
    return "semantle"


def resolved_prompt_source(prompt_source: str | None, task: str) -> str:
    if prompt_source:
        return prompt_source
    return "train" if task == "molopt" else "search"


def resolved_task_description(
    task_description: str | None,
    *,
    task: str,
    use_chat_template: bool,
) -> str:
    if task_description:
        return task_description
    if task == "molopt":
        return task_instruction("molopt", use_chat_template=use_chat_template)
    return SEARCH_TASK_DESCRIPTION


def resolved_eval_max_new_tokens(
    max_new_tokens: int | None, *, task: str, saved: dict[str, Any]
) -> int:
    if max_new_tokens is not None:
        return int(max_new_tokens)
    if saved.get("full_eval_max_new_tokens") is not None:
        return int(saved["full_eval_max_new_tokens"])
    return MOLOPT_EVAL_MAX_NEW_TOKENS if task == "molopt" else EVAL_MAX_NEW_TOKENS


def resolved_output_dir(output_dir: str | None, *, task: str) -> Path:
    if output_dir:
        return Path(output_dir).expanduser()
    if task == "molopt":
        return DEFAULT_MOLOPT_OUTPUT_DIR
    return DEFAULT_OUTPUT_DIR


def _csv_from_saved(saved: dict[str, Any]) -> list[str]:
    raw = saved.get("semantle_csv")
    if isinstance(raw, str) and raw.strip():
        return [raw]
    if isinstance(raw, (list, tuple)):
        return [str(path) for path in raw if str(path).strip()]
    return []


@dataclass(frozen=True)
class TrainWordDraw:
    words: list[str]
    csv_paths: list[str]
    train_top_k: int
    train_n_samples: int | None
    seed: int
    source: str
    matched_checkpoint_items: bool | None
    task: str = "semantle"


def load_molopt_train_words(
    csv_path: str,
    *,
    train_top_k: int | None,
    train_n_samples: int | None,
    seed: int,
) -> list[str]:
    items = MolOptItem.load_csv(csv_path, top_k=train_top_k)
    if train_n_samples is not None:
        items, _ = sample_reft_items(items, train_n_samples, seed=seed)
    return [unwrap_smiles_tags(item_target(item)) for item in items]


def resolve_train_word_draw(
    *,
    semantle_csv: Sequence[str] | None,
    train_top_k: int | None,
    train_n_samples: int | None,
    seed: int | None,
    reft_output_dir: str | None,
    from_checkpoint_items: bool,
    skip_items_check: bool,
    task: str | None = None,
    molopt_csv: str | None = None,
) -> TrainWordDraw:
    saved: dict[str, Any] = {}
    checkpoint: Path | None = None
    if reft_output_dir:
        checkpoint = Path(reft_output_dir).expanduser()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint}")
        saved = load_merged_run_config(str(checkpoint))
    task_name = resolve_task(task, saved)

    if saved:
        if train_top_k is None and saved.get("train_top_k") is not None:
            train_top_k = int(saved["train_top_k"])
        if train_n_samples is None and saved.get("train_n_samples") is not None:
            train_n_samples = int(saved["train_n_samples"])
        if seed is None and saved.get("seed") is not None:
            seed = int(saved["seed"])
    if seed is None:
        seed = DEFAULT_SEED

    items_path = checkpoint / "items.json" if checkpoint is not None else None
    checkpoint_words = (
        words_from_items_json(items_path)
        if items_path is not None and items_path.is_file()
        else None
    )
    if task_name == "molopt":
        return _resolve_molopt_draw(
            molopt_csv=molopt_csv or (str(saved["molopt_csv"]) if saved.get("molopt_csv") else None),
            train_top_k=train_top_k,
            train_n_samples=train_n_samples,
            seed=seed,
            from_checkpoint_items=from_checkpoint_items,
            skip_items_check=skip_items_check,
            checkpoint_words=checkpoint_words,
            items_path=items_path,
        )

    if train_top_k is None:
        train_top_k = DEFAULT_TRAIN_TOP_K
    if train_n_samples is None:
        train_n_samples = DEFAULT_TRAIN_N_SAMPLES

    csv_paths = list(semantle_csv or ())
    if not csv_paths:
        csv_paths = _csv_from_saved(saved)
    if not csv_paths:
        csv_paths = [str(DEFAULT_CSV)]
    csv_paths = [resolve_semantle_csv(path) for path in csv_paths]

    if from_checkpoint_items:
        if checkpoint_words is None:
            raise FileNotFoundError(
                "--from-checkpoint-items requires items.json in --reft-output-dir"
            )
        return TrainWordDraw(
            words=list(checkpoint_words),
            csv_paths=csv_paths,
            train_top_k=train_top_k,
            train_n_samples=len(checkpoint_words),
            seed=seed,
            source="items.json",
            matched_checkpoint_items=True,
            task=task_name,
        )

    words = load_semantle_train_words(
        csv_paths,
        train_top_k=train_top_k,
        train_n_samples=train_n_samples,
        seed=seed,
    )
    matched: bool | None = None
    if checkpoint_words is not None and not skip_items_check:
        if words != checkpoint_words:
            raise ValueError(
                "sampled train words do not match checkpoint items.json "
                f"({len(words)} sampled vs {len(checkpoint_words)} in {items_path}). "
                "Pass --from-checkpoint-items to use items.json, or "
                "--skip-items-check to ignore the mismatch."
            )
        matched = True
    elif checkpoint_words is not None:
        matched = words == checkpoint_words
    return TrainWordDraw(
        words=words,
        csv_paths=csv_paths,
        train_top_k=train_top_k,
        train_n_samples=train_n_samples,
        seed=seed,
        source="sample_reft_items",
        matched_checkpoint_items=matched,
        task=task_name,
    )


def _resolve_molopt_draw(
    *,
    molopt_csv: str | None,
    train_top_k: int | None,
    train_n_samples: int | None,
    seed: int,
    from_checkpoint_items: bool,
    skip_items_check: bool,
    checkpoint_words: list[str] | None,
    items_path: Path | None,
) -> TrainWordDraw:
    csv_paths = [molopt_csv] if molopt_csv else []
    use_items = from_checkpoint_items or not csv_paths
    if use_items:
        if checkpoint_words is None:
            raise FileNotFoundError(
                "molopt LoRA SFT needs items.json in --reft-output-dir "
                "(or pass --molopt-csv to resample)"
            )
        return TrainWordDraw(
            words=[unwrap_smiles_tags(text) for text in checkpoint_words],
            csv_paths=csv_paths,
            train_top_k=train_top_k if train_top_k is not None else len(checkpoint_words),
            train_n_samples=len(checkpoint_words),
            seed=seed,
            source="items.json",
            matched_checkpoint_items=True,
            task="molopt",
        )
    if not molopt_csv:
        raise ValueError("molopt LoRA SFT requires --reft-output-dir or --molopt-csv")
    words = load_molopt_train_words(
        molopt_csv,
        train_top_k=train_top_k,
        train_n_samples=train_n_samples,
        seed=seed,
    )
    matched: bool | None = None
    if checkpoint_words is not None and not skip_items_check:
        expected = [unwrap_smiles_tags(text) for text in checkpoint_words]
        if words != expected:
            raise ValueError(
                "sampled molopt SMILES do not match checkpoint items.json "
                f"({len(words)} sampled vs {len(expected)} in {items_path}). "
                "Pass --from-checkpoint-items to use items.json, or "
                "--skip-items-check to ignore the mismatch."
            )
        matched = True
    elif checkpoint_words is not None:
        matched = words == [unwrap_smiles_tags(text) for text in checkpoint_words]
    return TrainWordDraw(
        words=words,
        csv_paths=csv_paths,
        train_top_k=train_top_k if train_top_k is not None else len(words),
        train_n_samples=train_n_samples if train_n_samples is not None else len(words),
        seed=seed,
        source="molopt_csv",
        matched_checkpoint_items=matched,
        task="molopt",
    )


def user_instruction(
    *,
    prompt_source: str,
    task_description: str,
    use_chat_template: bool,
    task: str = "semantle",
) -> str:
    if prompt_source == "train":
        return task_instruction(task, use_chat_template=use_chat_template)
    if prompt_source == "search":
        return task_description.strip()
    raise ValueError(f"unknown prompt source: {prompt_source!r}")


def resolve_sft_instruction(
    *,
    prompt_source: str,
    task_description: str,
    use_chat_template: bool,
    task: str,
    mist_smiles_tags: bool,
) -> str:
    text = user_instruction(
        prompt_source=prompt_source,
        task_description=task_description,
        use_chat_template=use_chat_template,
        task=task,
    )
    return maybe_append_mist_smiles_open_tag(
        text,
        mist_open_tag_in_prompt(
            mist_smiles_tags=mist_smiles_tags,
            use_chat_template=use_chat_template,
        ),
    )


class WordSFTDataset(Dataset):
    def __init__(self, rows: list[dict[str, torch.Tensor]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.rows[index]


def prepare_sft_items(
    words: Sequence[str],
    instruction: str,
    tokenizer=None,
    *,
    use_chat_template: bool,
    system_prompt: str | None,
    task: str = "semantle",
    mist_smiles_tags: bool = False,
    smiles_tags: bool = False,
) -> list:
    """Prompt → target items. ``SemantleItem.load`` uses the train prompt; replace it."""
    if task == "molopt":
        items = [
            MolOptItem(id=index, prompt=instruction, target=unwrap_smiles_tags(word))
            for index, word in enumerate(words)
        ]
        if smiles_tags and mist_smiles_tags:
            raise ValueError("smiles_tags and mist_smiles_tags are mutually exclusive")
        if mist_smiles_tags:
            apply_mist_smiles_tags(items, include_open=use_chat_template)
        elif smiles_tags:
            apply_smiles_tags(items)
        if use_chat_template:
            if tokenizer is None:
                raise ValueError("tokenizer is required when use_chat_template is true")
            apply_chat_format(
                items, tokenizer, instruction, system_prompt=system_prompt
            )
        else:
            for item in items:
                item.prompt = instruction
        return items
    items = SemantleItem.load(list(words))
    if use_chat_template:
        if tokenizer is None:
            raise ValueError("tokenizer is required when use_chat_template is true")
        apply_chat_format(
            items, tokenizer, instruction, system_prompt=system_prompt
        )
    else:
        for item in items:
            item.prompt = instruction
    return items


def build_sft_rows(
    words: Sequence[str],
    tokenizer,
    instruction: str,
    *,
    use_chat_template: bool,
    system_prompt: str | None,
    task: str = "semantle",
    mist_smiles_tags: bool = False,
    smiles_tags: bool = False,
) -> list[dict[str, torch.Tensor]]:
    items = prepare_sft_items(
        words,
        instruction,
        tokenizer,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        task=task,
        mist_smiles_tags=mist_smiles_tags,
        smiles_tags=smiles_tags,
    )
    rows: list[dict[str, torch.Tensor]] = []
    for item in items:
        if use_chat_template:
            full_input = item.prompt + item.target.strip()
        else:
            eos = tokenizer.eos_token or ""
            full_input = item.prompt + " " + item.target.strip() + eos
        prompt_ids = tokenize_model_text(
            tokenizer,
            item.prompt,
            from_chat_template=use_chat_template,
            return_tensors="pt",
        )["input_ids"][0]
        input_ids = tokenize_model_text(
            tokenizer,
            full_input,
            from_chat_template=use_chat_template,
            return_tensors="pt",
        )["input_ids"][0]
        prompt_len = int(prompt_ids.shape[0])
        if prompt_len >= int(input_ids.shape[0]):
            raise ValueError(
                f"SFT row for {item_target(item)!r} has no target tokens "
                f"(prompt_len={prompt_len}, seq_len={int(input_ids.shape[0])})"
            )
        if not torch.equal(input_ids[:prompt_len], prompt_ids):
            raise ValueError(
                f"SFT prompt tokens are not a prefix of the full sequence for "
                f"{item_target(item)!r}"
            )
        labels = input_ids.clone()
        labels[:prompt_len] = IGNORE_INDEX
        rows.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": torch.ones_like(input_ids),
            }
        )
    return rows


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    payload: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            payload[f"{name}.lora_A"] = module.lora_A.detach().cpu().clone()
            payload[f"{name}.lora_B"] = module.lora_B.detach().cpu().clone()
    if not payload:
        raise ValueError("no LoRA adapters found on the model")
    return payload


def should_eval_epoch(epoch: int, *, epochs: int, eval_epochs: int) -> bool:
    """Run at multiples of ``eval_epochs`` and at the last epoch."""
    if eval_epochs <= 0 or epoch < 1:
        return False
    return epoch == epochs or epoch % eval_epochs == 0


def epoch_adapter_path(out_dir: Path, epoch: int) -> Path:
    return Path(out_dir) / ADAPTERS_DIR / f"epoch{epoch:03d}.pt"


def save_adapter_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    lora_rank: int,
    lora_alpha: float,
    model_name: str,
    use_chat_template: bool,
    instruction: str,
    prompt_source: str,
    epoch: int | None = None,
    label: str | None = None,
    shift_score: float | None = None,
    task: str = "semantle",
    mist_smiles_tags: bool = False,
    smiles_tags: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "model_name": model_name,
            "use_chat_template": use_chat_template,
            "instruction": instruction,
            "prompt_source": prompt_source,
            "epoch": epoch,
            "label": label,
            "shift_score": shift_score,
            "task": task,
            "mist_smiles_tags": mist_smiles_tags,
            "smiles_tags": smiles_tags,
            "state_dict": adapter_state_dict(model),
        },
        path,
    )


def load_adapter_state_dict(
    model: nn.Module,
    state: dict[str, torch.Tensor],
    *,
    strict: bool = True,
) -> None:
    adapters = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    }
    if not adapters:
        raise ValueError("no LoRA adapters attached to load into")
    loaded = 0
    for name, module in adapters.items():
        key_a = f"{name}.lora_A"
        key_b = f"{name}.lora_B"
        if key_a not in state or key_b not in state:
            if strict:
                raise ValueError(f"adapter state is missing {key_a} / {key_b}")
            continue
        module.lora_A.data.copy_(state[key_a].to(device=module.lora_A.device, dtype=module.lora_A.dtype))
        module.lora_B.data.copy_(state[key_b].to(device=module.lora_B.device, dtype=module.lora_B.dtype))
        if hasattr(module, "ema_A"):
            module.ema_A.copy_(module.lora_A.detach())
            module.ema_B.copy_(module.lora_B.detach())
        loaded += 1
    unmatched = [
        key[: -len(".lora_A")]
        for key in state
        if key.endswith(".lora_A") and key[: -len(".lora_A")] not in adapters
    ]
    if unmatched:
        raise ValueError(
            "adapter state has modules that are not on the model: "
            + ", ".join(unmatched[:5])
        )
    if loaded == 0:
        raise ValueError("no LoRA weights were loaded from the adapter file")


def attach_lora_sft_adapter(model: nn.Module, adapter_path: str | Path) -> dict[str, Any]:
    """Inject task-local LoRA if needed and load ``adapter.pt`` / epoch weights."""
    payload = load_adapter_file(adapter_path)
    rank = int(payload.get("lora_rank") or 16)
    alpha = float(payload.get("lora_alpha") or rank)
    _inject_lora(model, rank, alpha)
    load_adapter_state_dict(model, payload["state_dict"], strict=False)
    return payload


def _proposal_key(text: str, *, task: str) -> str:
    if task == "molopt":
        return canonical_target_key(text)
    return normalize_search_text(text, task=task)


def proposal_eval_stats(
    samples: Sequence[str],
    train_words: Sequence[str],
    *,
    task: str = "semantle",
    top_k: int = 10,
) -> dict[str, Any]:
    """Compare decoded proposals to the SFT train-word support.

    ``train_hit_rate`` is P(sample in the train set). ``unique_rate`` and
    ``unigram_entropy`` fall when longer SFT collapses onto a few memorized
    words. ``shift_score`` is hit × unique, so both collapse and "no shift"
    score low.
    """
    train_keys = {
        _proposal_key(word, task=task)
        for word in train_words
        if str(word).strip()
    }
    normalized = [_proposal_key(text, task=task) for text in samples]
    n = len(normalized)
    nonempty = [text for text in normalized if text]
    in_train = [text for text in nonempty if text in train_keys]
    unique = list(dict.fromkeys(nonempty))
    unique_in_train = [text for text in unique if text in train_keys]
    counts = Counter(nonempty)
    entropy = 0.0
    if nonempty:
        total = len(nonempty)
        entropy = -sum(
            (count / total) * math.log(count / total) for count in counts.values()
        )
    hit_rate = len(in_train) / n if n else 0.0
    unique_rate = len(unique) / n if n else 0.0
    payload: dict[str, Any] = {
        "n": n,
        "n_nonempty": len(nonempty),
        "blank_rate": 1.0 - (len(nonempty) / n if n else 0.0),
        "train_hit_rate": hit_rate,
        "unique_rate": unique_rate,
        "n_unique": len(unique),
        "n_unique_in_train": len(unique_in_train),
        "train_coverage": (
            len(unique_in_train) / len(train_keys) if train_keys else 0.0
        ),
        "unigram_entropy": entropy,
        "shift_score": hit_rate * unique_rate,
        "top": counts.most_common(top_k),
        "samples": list(samples),
        "normalized": normalized,
        "task": task,
    }
    if task == "molopt":
        payload["valid_rate"] = validity_rate(list(samples))
    return payload


def sample_proposals(
    model: nn.Module,
    tokenizer,
    prompt: str,
    *,
    n: int,
    device: torch.device,
    from_chat_template: bool,
    max_new_tokens: int = EVAL_MAX_NEW_TOKENS,
    temperature: float = 1.0,
    top_p: float = 1.0,
    batch_size: int = EVAL_BATCH_SIZE,
) -> list[str]:
    """Sample task-only completions (same prompt Random uses with last-k=0)."""
    if n < 1:
        return []
    encoded = tokenize_model_text(
        tokenizer,
        prompt,
        from_chat_template=from_chat_template,
        return_tensors="pt",
    )
    prompt_ids = encoded["input_ids"].to(device)
    prompt_mask = encoded["attention_mask"].to(device)
    prompt_len = prompt_ids.shape[1]
    was_training = model.training
    cache = getattr(model.config, "use_cache", True)
    model.eval()
    model.config.use_cache = True
    texts: list[str] = []
    try:
        with torch.no_grad():
            remaining = n
            while remaining > 0:
                chunk = min(batch_size, remaining)
                input_ids = prompt_ids.repeat(chunk, 1)
                attention_mask = prompt_mask.repeat(chunk, 1)
                gen_kwargs: dict[str, Any] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": temperature > 0,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                if temperature > 0:
                    gen_kwargs["temperature"] = max(temperature, 1e-8)
                    gen_kwargs["top_p"] = top_p
                output = model.generate(**gen_kwargs)
                for row in output:
                    texts.append(
                        tokenizer.decode(
                            row[prompt_len:], skip_special_tokens=True
                        ).strip()
                    )
                remaining -= chunk
    finally:
        model.config.use_cache = cache
        model.train(was_training)
    return texts


def _compact_eval(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in stats.items()
        if key not in ("samples", "normalized")
    }


def _format_eval(stats: dict[str, Any]) -> str:
    text = (
        f"hit={stats['train_hit_rate']:.3f} unique={stats['unique_rate']:.3f} "
        f"cover={stats['train_coverage']:.3f} H={stats['unigram_entropy']:.2f} "
        f"shift={stats['shift_score']:.3f}"
    )
    if "valid_rate" in stats:
        text += f" valid={stats['valid_rate']:.3f}"
    return text


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--reft-output-dir",
        default=None,
        help="BOReFT checkpoint. Copies omitted train_top_k / train_n_samples / seed / "
        "model / chat flags from training_config.json and checks items.json.",
    )
    parser.add_argument(
        "--task",
        default=None,
        choices=("semantle", "molopt"),
        help="Defaults to the checkpoint task, else semantle.",
    )
    parser.add_argument(
        "--semantle-csv",
        nargs="+",
        default=None,
        help="Same as boreft.train --semantle-csv. Default: computer.csv, or the "
        "checkpoint's CSV list when --reft-output-dir is set.",
    )
    parser.add_argument(
        "--molopt-csv",
        default=None,
        help="Molopt SMILES CSV. Default: checkpoint items.json (not a resample).",
    )
    parser.add_argument(
        "--train-top-k",
        type=int,
        default=None,
        help=f"CSV head size. Default {DEFAULT_TRAIN_TOP_K}, or the checkpoint value when omitted.",
    )
    parser.add_argument(
        "--train-n-samples",
        type=int,
        default=None,
        help=f"Subsample size. Default {DEFAULT_TRAIN_N_SAMPLES}, or the checkpoint value when omitted.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"sample_reft_items seed. Default {DEFAULT_SEED}, or the checkpoint value when omitted.",
    )
    parser.add_argument(
        "--from-checkpoint-items",
        action="store_true",
        help="Use items.json words instead of resampling CSVs. Molopt default "
        "when --molopt-csv is omitted.",
    )
    parser.add_argument(
        "--skip-items-check",
        action="store_true",
        help="Do not require the sampled draw to match checkpoint items.json.",
    )
    parser.add_argument(
        "--task-description",
        default=None,
        help="Search-time Random prompt (prompt-source=search). Semantle default "
        "is the English-word instruction; molopt default is the completion prefix.",
    )
    parser.add_argument(
        "--prompt-source",
        choices=("search", "train"),
        default=None,
        help="search: --task-description. train: task_config completion/chat prompt. "
        "Default search for Semantle, train for molopt.",
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"SFT epochs (default {DEFAULT_EPOCHS}).",
    )
    parser.add_argument(
        "--eval-epochs",
        type=int,
        default=DEFAULT_EVAL_EPOCHS,
        help=f"Run proposal eval every N epochs and at the last epoch "
        f"(default {DEFAULT_EVAL_EPOCHS}; 0 skips periodic eval).",
    )
    parser.add_argument(
        "--save-epochs",
        type=int,
        default=DEFAULT_SAVE_EPOCHS,
        help=f"Write adapters/epochNNN.pt every N epochs and at the last epoch "
        f"(default {DEFAULT_SAVE_EPOCHS}; 0 skips per-epoch adapters).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Proposal-eval decode length. Default 16 for Semantle, 128 for molopt "
        "(or checkpoint full_eval_max_new_tokens).",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write words.json / config.json and exit without loading the LM.",
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=DEFAULT_N_EVAL,
        help="Sample this many task-only proposals at base (LoRA B=0) and every "
        "--eval-epochs. 0 skips. Writes eval.json (train hit / unique / coverage).",
    )
    parser.add_argument(
        "--n-probe",
        type=int,
        default=8,
        help="Print this many eval samples from the last snapshot (0 to skip printing).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load adapter.pt from --output-dir, score base vs adapter proposals, "
        "and resume the original W&B run. Does not train.",
    )
    parser.add_argument(
        "--allow-new-wandb-run",
        action="store_true",
        help="With --eval-only, create a new W&B run if the output dir has no saved id.",
    )
    parser.add_argument("--wandb-run-id", default=None, help="Override W&B run id to resume.")
    parser.add_argument("--wandb-project", default="boreft")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args(argv)


def _resolved_wandb_group(group: str | None, *, task: str) -> str:
    if group:
        return group
    return "molopt-lora-sft" if task == "molopt" else "semantle-lora-sft"


def _resolved_model_flags(
    args: argparse.Namespace, saved: dict[str, Any], *, task: str
) -> dict[str, Any]:
    model_name = args.model_name or saved.get("model_name") or DEFAULT_MODEL
    if args.use_chat_template is None:
        if "use_chat_template" in saved:
            use_chat = bool(saved["use_chat_template"])
        else:
            use_chat = task != "molopt"
    else:
        use_chat = bool(args.use_chat_template)
    cache_dir = args.cache_dir or saved.get("cache_dir")
    if saved:
        system_prompt = system_prompt_from_cfg(saved)
        mist_smiles_tags = bool(saved.get("mist_smiles_tags", True))
        smiles_tags = bool(saved.get("smiles_tags", False))
    else:
        system_prompt = task_system_prompt(task)
        mist_smiles_tags = True
        smiles_tags = False
    if task != "molopt":
        mist_smiles_tags = False
        smiles_tags = False
    elif mist_smiles_tags:
        smiles_tags = False
    return {
        "model_name": str(model_name),
        "use_chat_template": use_chat,
        "cache_dir": cache_dir,
        "torch_dtype": saved.get("torch_dtype"),
        "system_prompt": system_prompt,
        "mist_smiles_tags": mist_smiles_tags,
        "smiles_tags": smiles_tags,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def load_adapter_file(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"{path}: expected adapter.pt with a state_dict")
    return payload


_WANDB_URL_RE = re.compile(
    r"wandb\.ai/(?P<entity>[^/\s]+)/(?P<project>[^/\s]+)/runs/(?P<run_id>[A-Za-z0-9]+)"
)


def _run_id_from_wandb_dirname(name: str) -> str | None:
    """Parse ``run-YYYYMMDD_HHMMSS-<id>`` / ``offline-run-...`` directory names."""
    for prefix in ("offline-run-", "run-"):
        if name.startswith(prefix):
            rest = name[len(prefix) :]
            if "-" in rest:
                run_id = rest.split("-", 1)[1].strip()
                return run_id or None
            return None
    return None


def _wandb_identity_from_text(text: str) -> dict[str, str]:
    match = _WANDB_URL_RE.search(text)
    if not match:
        return {}
    return {
        "run_id": match.group("run_id"),
        "project": match.group("project"),
        "entity": match.group("entity"),
    }


def _wandb_metadata_identity(run_dir: Path) -> dict[str, str]:
    identity: dict[str, str] = {}
    for meta_path in (
        run_dir / "files" / "wandb-metadata.json",
        run_dir / "wandb-metadata.json",
    ):
        if not meta_path.is_file():
            continue
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        run_id = payload.get("id") or payload.get("run_id")
        if run_id:
            identity["run_id"] = str(run_id)
        if payload.get("project"):
            identity["project"] = str(payload["project"])
        entity = payload.get("entity") or payload.get("entityName")
        if entity:
            identity["entity"] = str(entity)
        break
    if not identity.get("run_id"):
        run_id = _run_id_from_wandb_dirname(run_dir.name)
        if run_id:
            identity["run_id"] = run_id
    if not identity.get("run_id"):
        for log_name in ("output.log", "debug.log", "debug-internal.log"):
            for log_path in (run_dir / "files" / log_name, run_dir / log_name):
                if not log_path.is_file():
                    continue
                found = _wandb_identity_from_text(log_path.read_text(encoding="utf-8", errors="ignore"))
                if found.get("run_id"):
                    identity.update(found)
                    return identity
    return identity


def discover_wandb_identity(out_dir: Path) -> dict[str, str]:
    """Find a W&B run id saved by training (meta file or wandb/ run dir)."""
    meta_path = out_dir / "wandb_meta.json"
    if meta_path.is_file():
        payload = _load_json(meta_path)
        identity = {
            key: str(payload[key])
            for key in ("run_id", "project", "entity")
            if payload.get(key)
        }
        if identity.get("run_id"):
            return identity
    run_id_path = out_dir / "wandb_run_id.txt"
    if run_id_path.is_file():
        run_id = run_id_path.read_text(encoding="utf-8").strip()
        if run_id:
            return {"run_id": run_id}
    wandb_dir = out_dir / "wandb"
    if not wandb_dir.is_dir():
        return {}
    candidates: list[Path] = []
    latest = wandb_dir / "latest-run"
    if latest.exists():
        candidates.append(latest.resolve() if latest.is_symlink() else latest)
    candidates.extend(sorted(wandb_dir.glob("run-*"), reverse=True))
    candidates.extend(sorted(wandb_dir.glob("offline-run-*"), reverse=True))
    seen: set[Path] = set()
    for run_dir in candidates:
        resolved = run_dir.resolve() if run_dir.exists() else run_dir
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        identity = _wandb_metadata_identity(resolved)
        if identity.get("run_id"):
            return identity
    return {}


def _write_wandb_meta(out_dir: Path, run: Any) -> None:
    payload = {
        "run_id": run.id,
        "project": getattr(run, "project", None),
        "entity": getattr(run, "entity", None),
        "name": getattr(run, "name", None),
    }
    _write_json(out_dir / "wandb_meta.json", payload)
    (out_dir / "wandb_run_id.txt").write_text(f"{run.id}\n", encoding="utf-8")


def init_lora_sft_wandb(
    out_dir: Path,
    *,
    no_wandb: bool,
    allow_new: bool,
    config_payload: dict[str, Any],
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_group: str | None,
    wandb_run_name: str | None,
    wandb_run_id: str | None,
):
    if no_wandb:
        return None
    import wandb

    identity = discover_wandb_identity(out_dir)
    run_id = wandb_run_id or identity.get("run_id")
    project = wandb_project or identity.get("project") or "boreft"
    entity = wandb_entity or identity.get("entity") or None
    os.environ["WANDB_CONSOLE"] = "off"
    if run_id:
        print(
            f"[lora_sft] resuming W&B run {run_id} (entity={entity!r}, project={project!r})",
            flush=True,
        )
        run = wandb.init(
            id=run_id,
            project=project,
            entity=entity,
            resume="allow",
            dir=str(out_dir),
        )
        _write_wandb_meta(out_dir, run)
        return run
    if not allow_new:
        raise FileNotFoundError(
            f"{out_dir} has no wandb_meta.json / wandb/ run id; refusing to "
            "create a new W&B run. Pass --no-wandb or --allow-new-wandb-run."
        )
    run = wandb.init(
        project=project,
        entity=entity,
        group=wandb_group,
        name=wandb_run_name or out_dir.name,
        config=config_payload,
        dir=str(out_dir),
    )
    _write_wandb_meta(out_dir, run)
    return run


def _log_eval_wandb(wandb_run, stats: dict[str, Any], *, prefix: str | None = None) -> None:
    if wandb_run is None:
        return
    payload = {
        "epoch": stats["epoch"],
        "eval/train_hit_rate": stats["train_hit_rate"],
        "eval/unique_rate": stats["unique_rate"],
        "eval/train_coverage": stats["train_coverage"],
        "eval/unigram_entropy": stats["unigram_entropy"],
        "eval/shift_score": stats["shift_score"],
        "eval/blank_rate": stats["blank_rate"],
    }
    if "valid_rate" in stats:
        payload["eval/valid_rate"] = stats["valid_rate"]
    if prefix:
        for key in (
            "train_hit_rate",
            "unique_rate",
            "train_coverage",
            "unigram_entropy",
            "shift_score",
            "blank_rate",
        ):
            payload[f"eval/{prefix}/{key}"] = stats[key]
            wandb_run.summary[f"eval/{prefix}/{key}"] = stats[key]
        if "valid_rate" in stats:
            payload[f"eval/{prefix}/valid_rate"] = stats["valid_rate"]
            wandb_run.summary[f"eval/{prefix}/valid_rate"] = stats["valid_rate"]
    wandb_run.log(payload)


def _best_trained_eval(eval_snapshots: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    trained = [item for item in eval_snapshots if item.get("label") != "base"]
    if not trained:
        return None
    return max(trained, key=lambda item: item["shift_score"])


def _write_eval_outputs(
    out_dir: Path,
    *,
    metrics: dict[str, Any],
    eval_snapshots: list[dict[str, Any]],
    n_eval: int,
    announce_best: bool = True,
) -> None:
    compact = [_compact_eval(item) for item in eval_snapshots]
    best = max(eval_snapshots, key=lambda item: item["shift_score"]) if eval_snapshots else None
    checkpoint = _best_trained_eval(eval_snapshots)
    metrics["eval"] = compact
    metrics["best_eval_label"] = None if best is None else best["label"]
    metrics["best_shift_score"] = None if best is None else best["shift_score"]
    metrics["best_checkpoint_label"] = None if checkpoint is None else checkpoint["label"]
    metrics["best_checkpoint_epoch"] = None if checkpoint is None else checkpoint["epoch"]
    metrics["best_checkpoint_shift_score"] = (
        None if checkpoint is None else checkpoint["shift_score"]
    )
    _write_json(out_dir / METRICS_NAME, metrics)
    if eval_snapshots:
        _write_json(
            out_dir / EVAL_NAME,
            {
                "n_eval": n_eval,
                "snapshots": eval_snapshots,
                "best_label": best["label"] if best is not None else None,
                "best_checkpoint_label": None if checkpoint is None else checkpoint["label"],
            },
        )
    if announce_best and checkpoint is not None:
        print(
            f"[lora_sft] best checkpoint {checkpoint['label']} {_format_eval(checkpoint)}",
            flush=True,
        )


def eval_existing(args: argparse.Namespace) -> Path:
    """Score base vs saved adapter proposals and resume the training W&B run."""
    if args.n_eval < 1:
        raise ValueError("--eval-only requires --n-eval > 0")
    if not args.output_dir:
        raise ValueError("--eval-only requires --output-dir")
    out_dir = Path(args.output_dir).expanduser()
    adapter_path = out_dir / ADAPTER_NAME
    words_path = out_dir / WORDS_NAME
    if not adapter_path.is_file():
        raise FileNotFoundError(f"--eval-only requires {adapter_path}")
    if not words_path.is_file():
        raise FileNotFoundError(f"--eval-only requires {words_path}")
    words_payload = _load_json(words_path)
    words = [str(word) for word in words_payload.get("words") or ()]
    if not words:
        raise ValueError(f"{words_path}: words list is empty")
    if not args.no_wandb:
        identity = discover_wandb_identity(out_dir)
        if not (args.wandb_run_id or identity.get("run_id") or args.allow_new_wandb_run):
            raise FileNotFoundError(
                f"{out_dir} has no wandb_meta.json / wandb/ run id; refusing to "
                "create a new W&B run. Pass --wandb-run-id, --no-wandb, or "
                "--allow-new-wandb-run."
            )
    saved_cfg = _load_json(out_dir / CONFIG_NAME) if (out_dir / CONFIG_NAME).is_file() else {}
    adapter = load_adapter_file(adapter_path)
    saved_reft: dict[str, Any] = {}
    reft_dir = args.reft_output_dir or saved_cfg.get("reft_output_dir")
    if reft_dir and Path(str(reft_dir)).expanduser().is_dir():
        saved_reft = load_merged_run_config(str(reft_dir))
    args.model_name = args.model_name or adapter.get("model_name") or saved_cfg.get("model_name")
    if args.use_chat_template is None and "use_chat_template" in adapter:
        args.use_chat_template = bool(adapter["use_chat_template"])
    elif args.use_chat_template is None and "use_chat_template" in saved_cfg:
        args.use_chat_template = bool(saved_cfg["use_chat_template"])
    task = resolve_task(
        args.task or adapter.get("task") or saved_cfg.get("task"),
        saved_reft,
    )
    flags = _resolved_model_flags(args, saved_reft or saved_cfg, task=task)
    prompt_source = resolved_prompt_source(
        adapter.get("prompt_source") or saved_cfg.get("prompt_source") or args.prompt_source,
        task,
    )
    task_description = resolved_task_description(
        saved_cfg.get("task_description") or args.task_description,
        task=task,
        use_chat_template=flags["use_chat_template"],
    )
    instruction = str(
        adapter.get("instruction")
        or saved_cfg.get("instruction")
        or resolve_sft_instruction(
            prompt_source=prompt_source,
            task_description=task_description,
            use_chat_template=flags["use_chat_template"],
            task=task,
            mist_smiles_tags=flags["mist_smiles_tags"],
        )
    )
    max_new_tokens = resolved_eval_max_new_tokens(
        args.max_new_tokens, task=task, saved=saved_reft or saved_cfg
    )
    lora_rank = int(adapter.get("lora_rank") or args.lora_rank)
    lora_alpha = float(adapter.get("lora_alpha") or (lora_rank if args.lora_alpha == 0 else args.lora_alpha))
    existing_metrics = (
        _load_json(out_dir / METRICS_NAME) if (out_dir / METRICS_NAME).is_file() else {}
    )
    adapter_epoch = 0
    for row in existing_metrics.get("epochs") or ():
        if isinstance(row, dict) and row.get("epoch") is not None:
            adapter_epoch = max(adapter_epoch, int(row["epoch"]))
    adapter_epoch = adapter_epoch or int(saved_cfg.get("epochs") or args.epochs or 0)

    _seed_everything(int(words_payload.get("seed") or saved_cfg.get("seed") or DEFAULT_SEED))
    load_dtype_name = infer_training_load_dtype_name(saved_cfg=saved_reft or saved_cfg)
    load_dtype = resolve_torch_dtype(load_dtype_name)
    print(
        f"[lora_sft] eval-only {len(words)} words  adapter={adapter_path} "
        f"model={flags['model_name']} n_eval={args.n_eval}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        flags["model_name"],
        padding_side="right",
        use_fast=True,
        cache_dir=args.cache_dir or flags["cache_dir"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        flags["model_name"],
        dtype=load_dtype,
        cache_dir=args.cache_dir or flags["cache_dir"],
    )
    model.to(device)
    _inject_lora(model, lora_rank, lora_alpha)
    eval_prompt = instruction
    if flags["use_chat_template"]:
        eval_prompt = chat_prompt(
            tokenizer, instruction, system_prompt=flags["system_prompt"]
        )
    eval_snapshots: list[dict[str, Any]] = []
    wandb_run = None
    try:
        wandb_run = init_lora_sft_wandb(
            out_dir,
            no_wandb=args.no_wandb,
            allow_new=args.allow_new_wandb_run,
            config_payload=saved_cfg,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity or None,
            wandb_group=_resolved_wandb_group(args.wandb_group, task=task),
            wandb_run_name=args.wandb_run_name,
            wandb_run_id=args.wandb_run_id,
        )

        def run_eval(label: str, epoch: int, *, prefix: str) -> dict[str, Any]:
            samples = sample_proposals(
                model,
                tokenizer,
                eval_prompt,
                n=args.n_eval,
                device=device,
                from_chat_template=flags["use_chat_template"],
                max_new_tokens=max_new_tokens,
            )
            stats = proposal_eval_stats(samples, words, task=task)
            stats["label"] = label
            stats["epoch"] = epoch
            eval_snapshots.append(stats)
            print(f"[lora_sft] eval {label} {_format_eval(stats)}", flush=True)
            if stats["top"]:
                preview = ", ".join(f"{word}×{count}" for word, count in stats["top"][:5])
                print(f"[lora_sft] eval {label} top: {preview}", flush=True)
            _log_eval_wandb(wandb_run, stats, prefix=prefix)
            return stats

        run_eval("base", 0, prefix="base")
        load_adapter_state_dict(model, adapter["state_dict"])
        run_eval("adapter", adapter_epoch, prefix="adapter")
        existing_metrics["eval_only"] = True
        _write_eval_outputs(
            out_dir,
            metrics=existing_metrics,
            eval_snapshots=eval_snapshots,
            n_eval=args.n_eval,
        )
        if args.n_probe > 0 and eval_snapshots:
            print("[lora_sft] last-eval samples:", flush=True)
            for index, text in enumerate(eval_snapshots[-1]["samples"][: args.n_probe], start=1):
                print(f"  {index}. {text}", flush=True)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    return out_dir


def train(args: argparse.Namespace) -> Path:
    if args.eval_only:
        return eval_existing(args)
    if args.lora_rank < 1:
        raise ValueError(f"--lora-rank must be positive, got {args.lora_rank}")
    if args.epochs < 1 and not args.dry_run:
        raise ValueError(f"--epochs must be positive, got {args.epochs}")
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    if args.n_eval < 0:
        raise ValueError(f"--n-eval must be nonnegative, got {args.n_eval}")
    if args.eval_epochs < 0:
        raise ValueError(f"--eval-epochs must be nonnegative, got {args.eval_epochs}")
    if args.save_epochs < 0:
        raise ValueError(f"--save-epochs must be nonnegative, got {args.save_epochs}")

    saved: dict[str, Any] = {}
    if args.reft_output_dir:
        saved = load_merged_run_config(args.reft_output_dir)
    task = resolve_task(args.task, saved)
    out_dir = resolved_output_dir(args.output_dir, task=task)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_dir} is not empty; pass --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    draw = resolve_train_word_draw(
        semantle_csv=args.semantle_csv,
        train_top_k=args.train_top_k,
        train_n_samples=args.train_n_samples,
        seed=args.seed,
        reft_output_dir=args.reft_output_dir,
        from_checkpoint_items=args.from_checkpoint_items,
        skip_items_check=args.skip_items_check,
        task=task,
        molopt_csv=args.molopt_csv,
    )
    flags = _resolved_model_flags(args, saved, task=task)
    prompt_source = resolved_prompt_source(args.prompt_source, task)
    task_description = resolved_task_description(
        args.task_description,
        task=task,
        use_chat_template=flags["use_chat_template"],
    )
    instruction = resolve_sft_instruction(
        prompt_source=prompt_source,
        task_description=task_description,
        use_chat_template=flags["use_chat_template"],
        task=task,
        mist_smiles_tags=flags["mist_smiles_tags"],
    )
    max_new_tokens = resolved_eval_max_new_tokens(
        args.max_new_tokens, task=task, saved=saved
    )
    lora_alpha = float(args.lora_rank if args.lora_alpha == 0 else args.lora_alpha)
    config_payload = {
        "reft_output_dir": args.reft_output_dir,
        "task": task,
        "semantle_csv": draw.csv_paths,
        "molopt_csv": args.molopt_csv,
        "train_top_k": draw.train_top_k,
        "train_n_samples": draw.train_n_samples,
        "seed": draw.seed,
        "word_source": draw.source,
        "matched_checkpoint_items": draw.matched_checkpoint_items,
        "n_words": len(draw.words),
        "task_description": task_description,
        "prompt_source": prompt_source,
        "instruction": instruction,
        "mist_smiles_tags": flags["mist_smiles_tags"],
        "smiles_tags": flags["smiles_tags"],
        "model_name": flags["model_name"],
        "cache_dir": flags["cache_dir"],
        "use_chat_template": flags["use_chat_template"],
        "lora_rank": args.lora_rank,
        "lora_alpha": lora_alpha,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "max_new_tokens": max_new_tokens,
        "n_eval": args.n_eval,
        "eval_epochs": args.eval_epochs,
        "save_epochs": args.save_epochs,
    }
    words_payload = {
        "words": draw.words,
        "n": len(draw.words),
        "seed": draw.seed,
        "train_top_k": draw.train_top_k,
        "train_n_samples": draw.train_n_samples,
        "semantle_csv": draw.csv_paths,
        "source": draw.source,
        "matched_checkpoint_items": draw.matched_checkpoint_items,
        "task": task,
    }
    _write_json(out_dir / CONFIG_NAME, config_payload)
    _write_json(out_dir / WORDS_NAME, words_payload)
    print(
        f"[lora_sft] task={task}  {len(draw.words)} targets  top_k={draw.train_top_k} "
        f"n={draw.train_n_samples} seed={draw.seed} source={draw.source}",
        flush=True,
    )
    print(f"[lora_sft] csv={draw.csv_paths}", flush=True)
    print(
        f"[lora_sft] prompt_source={prompt_source}  mist_tags={flags['mist_smiles_tags']}",
        flush=True,
    )
    if args.dry_run:
        print(f"[lora_sft] dry-run wrote {out_dir / WORDS_NAME}", flush=True)
        return out_dir

    _seed_everything(draw.seed)
    load_dtype_name = infer_training_load_dtype_name(saved_cfg=saved)
    load_dtype = resolve_torch_dtype(load_dtype_name)
    print(
        f"[lora_sft] model={flags['model_name']} dtype={load_dtype_name} "
        f"chat={flags['use_chat_template']} rank={args.lora_rank}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        flags["model_name"],
        padding_side="right",
        use_fast=True,
        cache_dir=flags["cache_dir"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        flags["model_name"],
        dtype=load_dtype,
        cache_dir=flags["cache_dir"],
    )
    model.to(device)
    model.config.use_cache = False
    adapters = _inject_lora(model, args.lora_rank, lora_alpha)
    rows = build_sft_rows(
        draw.words,
        tokenizer,
        instruction,
        use_chat_template=flags["use_chat_template"],
        system_prompt=flags["system_prompt"],
        task=task,
        mist_smiles_tags=flags["mist_smiles_tags"],
        smiles_tags=flags["smiles_tags"],
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer, padding=True, label_pad_token_id=IGNORE_INDEX
    )
    loader = DataLoader(
        WordSFTDataset(rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(draw.seed),
    )
    params = [adapter.lora_A for adapter in adapters] + [
        adapter.lora_B for adapter in adapters
    ]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    eval_prompt = instruction
    if flags["use_chat_template"]:
        eval_prompt = chat_prompt(
            tokenizer, instruction, system_prompt=flags["system_prompt"]
        )
    eval_snapshots: list[dict[str, Any]] = []
    epoch_losses: list[dict[str, Any]] = []
    epoch_adapters: list[dict[str, Any]] = []
    global_step = 0
    best_checkpoint: dict[str, Any] | None = None

    def persist_eval(*, announce_best: bool) -> None:
        _write_eval_outputs(
            out_dir,
            metrics={
                "epochs": epoch_losses,
                "n_words": len(draw.words),
                "steps": global_step,
                "epoch_adapters": epoch_adapters,
            },
            eval_snapshots=eval_snapshots,
            n_eval=args.n_eval,
            announce_best=announce_best,
        )
        if epoch_adapters:
            _write_json(out_dir / ADAPTERS_DIR / "index.json", {"adapters": epoch_adapters})

    def write_adapter(
        path: Path,
        *,
        epoch: int | None,
        label: str | None,
        shift_score: float | None,
    ) -> None:
        save_adapter_checkpoint(
            path,
            model=model,
            lora_rank=args.lora_rank,
            lora_alpha=lora_alpha,
            model_name=flags["model_name"],
            use_chat_template=flags["use_chat_template"],
            instruction=instruction,
            prompt_source=prompt_source,
            epoch=epoch,
            label=label,
            shift_score=shift_score,
            task=task,
            mist_smiles_tags=flags["mist_smiles_tags"],
            smiles_tags=flags["smiles_tags"],
        )

    def run_eval(label: str, epoch: int) -> dict[str, Any] | None:
        if args.n_eval <= 0:
            return None
        samples = sample_proposals(
            model,
            tokenizer,
            eval_prompt,
            n=args.n_eval,
            device=device,
            from_chat_template=flags["use_chat_template"],
            max_new_tokens=max_new_tokens,
        )
        stats = proposal_eval_stats(samples, draw.words, task=task)
        stats["label"] = label
        stats["epoch"] = epoch
        eval_snapshots.append(stats)
        print(f"[lora_sft] eval {label} {_format_eval(stats)}", flush=True)
        if stats["top"]:
            preview = ", ".join(f"{word}×{count}" for word, count in stats["top"][:5])
            print(f"[lora_sft] eval {label} top: {preview}", flush=True)
        _log_eval_wandb(wandb_run, stats)
        return stats

    def maybe_save_best(stats: dict[str, Any] | None) -> None:
        nonlocal best_checkpoint
        persist_eval(announce_best=False)
        if stats is None or stats.get("label") == "base":
            return
        if (
            best_checkpoint is not None
            and stats["shift_score"] <= best_checkpoint["shift_score"]
        ):
            return
        write_adapter(
            out_dir / ADAPTER_NAME,
            epoch=stats["epoch"],
            label=stats["label"],
            shift_score=stats["shift_score"],
        )
        best_checkpoint = stats
        print(
            f"[lora_sft] wrote best {out_dir / ADAPTER_NAME} {stats['label']} "
            f"{_format_eval(stats)}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.summary["eval/best_shift_score"] = stats["shift_score"]
            wandb_run.summary["eval/best_epoch"] = stats["epoch"]
            wandb_run.summary["eval/best_label"] = stats["label"]

    wandb_run = None
    try:
        if not args.no_wandb:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                group=_resolved_wandb_group(args.wandb_group, task=task),
                name=args.wandb_run_name or out_dir.name,
                config=config_payload,
                dir=str(out_dir),
            )
            _write_wandb_meta(out_dir, wandb_run)

        # LoRA B is zeros at inject, so this is the pretrained Random prior.
        maybe_save_best(run_eval("base", 0))
        model.train()
        for epoch in range(args.epochs):
            running = 0.0
            n_batches = 0
            for batch in loader:
                batch = {
                    key: value.to(device) if hasattr(value, "to") else value
                    for key, value in batch.items()
                }
                optimizer.zero_grad(set_to_none=True)
                loss = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                ).loss
                loss.backward()
                optimizer.step()
                running += float(loss.detach())
                n_batches += 1
                global_step += 1
                if args.max_steps is not None and global_step >= args.max_steps:
                    break
            mean_loss = running / max(n_batches, 1)
            epoch_idx = epoch + 1
            epoch_losses.append({"epoch": epoch_idx, "loss": mean_loss, "steps": n_batches})
            print(f"[lora_sft] epoch {epoch_idx}/{args.epochs} loss={mean_loss:.4f}", flush=True)
            if wandb_run is not None:
                wandb_run.log({"epoch": epoch_idx, "train/loss": mean_loss, "step": global_step})
            stopped_early = args.max_steps is not None and global_step >= args.max_steps
            stats: dict[str, Any] | None = None
            if should_eval_epoch(
                epoch_idx, epochs=args.epochs, eval_epochs=args.eval_epochs
            ) or stopped_early:
                stats = run_eval(f"epoch{epoch_idx}", epoch_idx)
                maybe_save_best(stats)
            if should_eval_epoch(
                epoch_idx, epochs=args.epochs, eval_epochs=args.save_epochs
            ) or stopped_early:
                adapter_path = epoch_adapter_path(out_dir, epoch_idx)
                write_adapter(
                    adapter_path,
                    epoch=epoch_idx,
                    label=f"epoch{epoch_idx}",
                    shift_score=None if stats is None else stats.get("shift_score"),
                )
                rel = str(adapter_path.relative_to(out_dir))
                epoch_adapters.append(
                    {
                        "epoch": epoch_idx,
                        "path": rel,
                        "shift_score": None if stats is None else stats.get("shift_score"),
                    }
                )
                print(f"[lora_sft] wrote {adapter_path}", flush=True)
                persist_eval(announce_best=False)
            if stopped_early:
                break

        last_epoch = epoch_losses[-1]["epoch"] if epoch_losses else 0
        last_eval = next(
            (item for item in reversed(eval_snapshots) if item.get("label") != "base"),
            None,
        )
        write_adapter(
            out_dir / ADAPTER_LAST_NAME,
            epoch=last_epoch,
            label="last",
            shift_score=None if last_eval is None else last_eval.get("shift_score"),
        )
        print(f"[lora_sft] wrote {out_dir / ADAPTER_LAST_NAME}", flush=True)
        if best_checkpoint is None:
            write_adapter(
                out_dir / ADAPTER_NAME,
                epoch=last_epoch,
                label="last",
                shift_score=None if last_eval is None else last_eval.get("shift_score"),
            )
            print(f"[lora_sft] wrote {out_dir / ADAPTER_NAME} (last epoch; no trained eval)", flush=True)
        persist_eval(announce_best=True)
        if args.n_probe > 0 and eval_snapshots:
            print("[lora_sft] last-eval samples:", flush=True)
            for index, text in enumerate(eval_snapshots[-1]["samples"][: args.n_probe], start=1):
                print(f"  {index}. {text}", flush=True)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
