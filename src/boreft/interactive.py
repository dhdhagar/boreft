"""Shared prompt and generation helpers for interactive BOReFT clients."""

from __future__ import annotations

import re
from typing import Literal

import torch

from boreft.chem import maybe_append_mist_smiles_open_tag, mist_open_tag_in_prompt
from boreft.data.base import parse_positions
from boreft.data_utils import (
    _apply_chat_template,
    decode_generated_text,
    prepend_system_message,
    tokenize_model_text,
)
from boreft.intervention_marker import (
    instruction_content_span,
    is_content_position,
    is_marker_position,
    resolve_instruction_for_prompt,
    resolve_span_probe_token,
)

# Qwen3 (and similar) emit ``<think>...</think>`` when thinking is enabled.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

GenerationMode = Literal["intervention", "zero_bias", "base"]


def strip_thinking_text(text: str) -> str:
    """Drop ``<think>...</think>`` blocks; keep the final answer for matching."""
    return _THINK_BLOCK_RE.sub("", text).strip()


def end_relative_intervention_position(position: str) -> bool:
    """True when intervention sites depend on the prompt suffix (e.g. ``l1``)."""
    if is_marker_position(position) or is_content_position(position):
        return False
    _, last_n = parse_positions(position)
    return last_n > 0


def build_chat_prompt(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool = False,
) -> str:
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer has no chat_template.")
    return _apply_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=enable_thinking,
    )


def _decode_base_response(tokenizer, output_ids, prompt_len: int) -> str:
    new_ids = output_ids[0, prompt_len:].tolist()
    eos_id = tokenizer.eos_token_id
    if eos_id is not None and eos_id in new_ids:
        new_ids = new_ids[: new_ids.index(eos_id) + 1]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


@torch.inference_mode()
def generate_base(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    from_chat_template: bool = False,
    assistant_suffix: str | None = None,
) -> str:
    """Generate without intervention hooks, matching the terminal REPL."""
    enc = tokenize_model_text(
        tokenizer, prompt, from_chat_template=from_chat_template, return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(model.device)
    attn_mask = enc["attention_mask"].to(model.device)
    prompt_len = input_ids.shape[1]

    gen_kwargs: dict = {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    output_ids = model.generate(**gen_kwargs)
    if assistant_suffix is not None:
        return decode_generated_text(
            tokenizer,
            output_ids,
            prompt_len,
            assistant_suffix=assistant_suffix,
        )
    return _decode_base_response(tokenizer, output_ids, prompt_len)


def maybe_append_decode_smiles_open_tag(ckpt, instruction: str) -> str:
    """Append MiST ``[START_SMILES]`` for completion-style molopt decoding.

    Intervention mode already does this via
    :func:`prepare_instruction_for_intervention`. Base-mode LLM baselines
    must call this so the open tag is the last prompt token, matching
    training.
    """
    cfg = getattr(ckpt, "saved_cfg", None) or {}
    return maybe_append_mist_smiles_open_tag(
        instruction,
        mist_open_tag_in_prompt(
            mist_smiles_tags=bool(cfg.get("mist_smiles_tags")),
            use_chat_template=bool(cfg.get("use_chat_template")),
        ),
    )


def prepare_instruction_for_intervention(ckpt, instruction: str) -> str:
    """Inject the marker (if used) and append MiST ``[START_SMILES]`` last."""
    cfg = ckpt.saved_cfg or {}
    text = instruction.strip()
    if is_marker_position(cfg.get("position", "l1")):
        text = resolve_instruction_for_prompt(
            instruction,
            cfg.get("intervention_inject", "none"),
            cfg.get("intervention_token"),
            tokenizer=ckpt.tokenizer,
            intervention_token_id=ckpt.intervention_token_id,
        )
    return maybe_append_mist_smiles_open_tag(
        text,
        mist_open_tag_in_prompt(
            mist_smiles_tags=bool(cfg.get("mist_smiles_tags")),
            use_chat_template=bool(cfg.get("use_chat_template")),
        ),
    )


def _content_span_for_instruction(
    ckpt,
    instruction: str,
    *,
    from_chat_template: bool,
    model_name: str,
    enable_thinking: bool = False,
    system_prompt: str | None = None,
):
    cfg = ckpt.saved_cfg or {}
    if not is_content_position(cfg.get("position", "l1")):
        return ckpt.content_span
    probe = cfg.get("span_probe_token") or resolve_span_probe_token(
        cfg.get("model_name") or model_name,
        cfg.get("intervention_token"),
    )
    return instruction_content_span(
        ckpt.tokenizer,
        instruction,
        use_chat_template=from_chat_template,
        span_probe_token=probe,
        enable_thinking=enable_thinking,
        system_prompt=system_prompt,
    )


def build_checkpoint_prompt(
    ckpt,
    *,
    user_text: str,
    history: list[dict[str, str]],
    accumulate_history: bool,
    use_checkpoint_prompt: bool,
    generation_mode: GenerationMode,
    model_name: str,
    enable_thinking: bool = False,
    system_prompt: str | None = None,
) -> tuple[str, bool, tuple[int, int] | None]:
    """Build a checkpoint prompt exactly as the terminal REPL does."""
    from_chat = ckpt.from_chat_template

    if use_checkpoint_prompt:
        return ckpt.prompt, from_chat, ckpt.content_span

    use_intervention = generation_mode in ("intervention", "zero_bias")
    position = (ckpt.saved_cfg or {}).get("position", "l1")
    if (
        enable_thinking
        and use_intervention
        and end_relative_intervention_position(position)
    ):
        raise ValueError(
            f"enable_thinking is incompatible with position={position!r}: "
            "thinking changes the chat-template suffix and moves end-relative "
            "intervention sites (e.g. l1). Use base mode, or a content/marker "
            "position."
        )

    user_content = (
        prepare_instruction_for_intervention(ckpt, user_text)
        if use_intervention
        else user_text
    )

    if from_chat:
        messages = (
            [*history, {"role": "user", "content": user_content}]
            if accumulate_history
            else [{"role": "user", "content": user_content}]
        )
        messages = prepend_system_message(messages, system_prompt)
        prompt = build_chat_prompt(
            ckpt.tokenizer, messages, enable_thinking=enable_thinking
        )
    else:
        prompt = user_content

    content_span = (
        _content_span_for_instruction(
            ckpt,
            user_content,
            from_chat_template=from_chat,
            model_name=model_name,
            enable_thinking=enable_thinking,
            system_prompt=system_prompt,
        )
        if use_intervention
        else ckpt.content_span
    )
    return prompt, from_chat, content_span


def target_intervention_flags(
    generated: str,
    target: str | None,
    word_to_id: dict[str, int],
) -> tuple[bool, bool]:
    """Whether a decode is in the training vocabulary and equals the target."""
    gen = strip_thinking_text(generated)
    return gen in word_to_id, target is not None and gen == target.strip()
