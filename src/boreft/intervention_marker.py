"""Intervention marker injection and position=marker resolution."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple

import torch

from boreft.data.base import get_intervention_locations, parse_positions

InjectMode = Literal["none", "prefix", "suffix"]
InitMode = Literal["none", "newline", "random"]
POSITION_MARKER = "marker"
POSITION_CONTENT_F1 = "content_f1"
POSITION_CONTENT_L1 = "content_l1"
CONTENT_POSITIONS = frozenset({POSITION_CONTENT_F1, POSITION_CONTENT_L1})

_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "intervention_tokens.json"


def is_marker_position(position: str) -> bool:
    return position.strip().lower() == POSITION_MARKER


def is_content_position(position: str) -> bool:
    return position.strip().lower() in CONTENT_POSITIONS


_LEGACY_POSITION_RE = re.compile(r"^f\d+$|^l\d+$|^f\d+\+l\d+$", re.IGNORECASE)


def validate_intervention_position(position: str) -> str:
    """Validate and normalize an intervention position string."""
    pos = position.strip()
    lower = pos.lower()
    if lower == POSITION_MARKER:
        return POSITION_MARKER
    if lower in CONTENT_POSITIONS:
        return lower
    if _LEGACY_POSITION_RE.match(pos):
        return pos.lower()
    raise ValueError(
        f"invalid intervention position {position!r}. "
        "Expected marker, content_f1, content_l1, or legacy forms "
        "f<N>, l<N>, f<N>+l<M> (e.g. l1, f1, f2+l2)."
    )


def _load_token_map() -> dict[str, str]:
    if not _CONFIG_PATH.is_file():
        return {}
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{_CONFIG_PATH} must be a JSON object mapping model names to tokens")
    return {str(k): str(v) for k, v in data.items()}


def resolve_intervention_token_string(model_name: str, override: Optional[str]) -> str:
    if override is not None and override.strip():
        return override.strip()
    mapping = _load_token_map()
    if model_name in mapping:
        return mapping[model_name]
    raise ValueError(
        f"No intervention token for model {model_name!r}. "
        f"Pass --intervention-token or add an entry to {_CONFIG_PATH}"
    )


def encode_single_intervention_token(tokenizer, token_str: str) -> int:
    ids = tokenizer.encode(token_str, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"intervention token {token_str!r} must encode to exactly one id, got {ids}"
        )
    return int(ids[0])


def ensure_intervention_token(tokenizer, model, token_str: str) -> int:
    """Ensure ``token_str`` is a single special token inside the embedding table.

    Idempotently ``add_tokens`` when needed. Does **not** resize embeddings: the
    resolved id must already satisfy ``0 <= id < num_embeddings`` (spare rows).
    """
    from tokenizers import AddedToken

    tokenizer.add_tokens([AddedToken(token_str, normalized=False, special=True)])
    token_id = int(tokenizer.convert_tokens_to_ids(token_str))
    n_emb = int(model.get_input_embeddings().weight.shape[0])
    if token_id < 0 or token_id >= n_emb:
        raise ValueError(
            f"intervention token {token_str!r} id={token_id} is outside the "
            f"embedding table (size {n_emb}); refuse to resize"
        )
    ids = tokenizer.encode(token_str, add_special_tokens=False)
    if ids != [token_id]:
        raise ValueError(
            f"intervention token {token_str!r} must encode to [{token_id}], got {ids}"
        )
    return token_id


def resolve_intervention_token(
    tokenizer,
    model_name: str,
    token_override: Optional[str],
    model,
) -> Tuple[str, int]:
    token_str = resolve_intervention_token_string(model_name, token_override)
    token_id = ensure_intervention_token(tokenizer, model, token_str)
    return token_str, token_id


def resolve_span_probe_token(model_name: str, token_override: Optional[str]) -> str:
    """Single-token probe for chat instruction span detection (from intervention_tokens.json)."""
    return resolve_intervention_token_string(model_name, token_override)


def _flatten_tokenizer_ids(raw) -> List[int]:
    if raw and isinstance(raw[0], list):
        return list(raw[0])
    return list(raw)


def _unique_probe_index(ids: Sequence[int], probe_id: int, *, label: str) -> int:
    hits = [i for i, tid in enumerate(ids) if tid == probe_id]
    if len(hits) != 1:
        raise ValueError(
            f"content span {label} probe: expected 1 occurrence of id={probe_id}, "
            f"found {len(hits)}"
        )
    return hits[0]


def _chat_prompt_ids(
    tokenizer,
    instruction: str,
    *,
    enable_thinking: bool = False,
    system_prompt: Optional[str] = None,
) -> List[int]:
    from boreft.data_utils import chat_prompt, tokenize_model_text

    raw = tokenize_model_text(
        tokenizer,
        chat_prompt(
            tokenizer,
            instruction.strip(),
            enable_thinking=enable_thinking,
            system_prompt=system_prompt,
        ),
        from_chat_template=True,
    )["input_ids"]
    return _flatten_tokenizer_ids(raw)


def instruction_content_span(
    tokenizer,
    instruction: str,
    *,
    use_chat_template: bool,
    span_probe_token: str,
    enable_thinking: bool = False,
    system_prompt: Optional[str] = None,
) -> Tuple[int, int]:
    """``(content_start, content_end)`` in the rendered prompt; end is exclusive."""
    instruction = instruction.strip()
    encode_single_intervention_token(tokenizer, span_probe_token)

    if not use_chat_template:
        from boreft.data_utils import tokenize_model_text

        ids = _flatten_tokenizer_ids(
            tokenize_model_text(tokenizer, instruction, from_chat_template=False)[
                "input_ids"
            ]
        )
        bos = getattr(tokenizer, "bos_token_id", None)
        start = 1 if bos is not None and ids and ids[0] == bos else 0
        end = len(ids)
        if start >= end:
            raise ValueError("instruction token span is empty")
        return start, end

    probe_id = encode_single_intervention_token(tokenizer, span_probe_token)
    ids_kwargs = dict(enable_thinking=enable_thinking, system_prompt=system_prompt)
    base = _chat_prompt_ids(tokenizer, instruction, **ids_kwargs)
    if probe_id in base:
        raise ValueError(
            f"content span probe id={probe_id} already present in base prompt"
        )

    marked_prefix = _chat_prompt_ids(
        tokenizer,
        f"{span_probe_token} {instruction}",
        **ids_kwargs,
    )
    marked_suffix = _chat_prompt_ids(
        tokenizer,
        f"{instruction} {span_probe_token}",
        **ids_kwargs,
    )

    # Probe marks instruction boundaries in diff prompts; indices map to ``base``.
    content_start = _unique_probe_index(marked_prefix, probe_id, label="prefix")
    probe_suffix_idx = _unique_probe_index(marked_suffix, probe_id, label="suffix")
    tail_shift = len(marked_suffix) - len(base)
    # Suffix probe sits after instruction in the diff prompt (+tail_shift tokens);
    # map back to ``base`` exclusive end (before eot / assistant header).
    content_end = probe_suffix_idx - tail_shift + 1

    if base[:content_start] != marked_prefix[:content_start]:
        raise ValueError(
            "content span prefix probe: prompt prefix mismatch before instruction"
        )

    if base[content_end + 1 :] != marked_suffix[probe_suffix_idx + tail_shift :]:
        raise ValueError(
            "content span suffix probe: prompt suffix mismatch after instruction"
        )

    if content_start >= content_end:
        raise ValueError(
            f"invalid instruction span [{content_start}, {content_end})"
        )
    return content_start, content_end


def content_position_indices(
    position: str, content_start: int, content_end: int
) -> List[int]:
    kind = position.strip().lower()
    if kind == POSITION_CONTENT_F1:
        return [content_start]
    if kind == POSITION_CONTENT_L1:
        return [content_end - 1]
    raise ValueError(f"not a content position: {position!r}")


def content_span_from_cfg(
    tokenizer,
    saved_cfg: dict,
    model_name: str,
) -> Optional[Tuple[int, int]]:
    position = saved_cfg.get("position", "l1")
    if not is_content_position(position):
        return None
    probe_model = saved_cfg.get("model_name") or model_name
    probe = saved_cfg.get("span_probe_token") or resolve_span_probe_token(
        probe_model, saved_cfg.get("intervention_token")
    )
    from boreft.data_utils import system_prompt_from_cfg
    from boreft.chem import maybe_append_mist_smiles_open_tag, mist_open_tag_in_prompt
    from boreft.task_config import task_instruction

    instruction = task_instruction(
        saved_cfg.get("task", "semantle"),
        use_chat_template=bool(saved_cfg.get("use_chat_template")),
        override=saved_cfg.get("chat_instruction"),
    )
    # Same user string as ``build_eval_prompt`` (inject, then optional MiST open).
    instruction = resolve_instruction_for_prompt(
        instruction,
        saved_cfg.get("intervention_inject", "none"),
        saved_cfg.get("intervention_token"),
        tokenizer=tokenizer,
        intervention_token_id=intervention_token_id_from_cfg(saved_cfg),
    )
    instruction = maybe_append_mist_smiles_open_tag(
        instruction,
        mist_open_tag_in_prompt(
            mist_smiles_tags=bool(saved_cfg.get("mist_smiles_tags")),
            use_chat_template=bool(saved_cfg.get("use_chat_template")),
        ),
    )
    return instruction_content_span(
        tokenizer,
        instruction,
        use_chat_template=bool(saved_cfg.get("use_chat_template")),
        span_probe_token=probe,
        system_prompt=system_prompt_from_cfg(saved_cfg),
    )


def apply_injection(instruction: str, inject: InjectMode, token: str) -> str:
    text = instruction.strip()
    if inject == "none":
        return text
    if inject == "prefix":
        return f"{token} {text}"
    if inject == "suffix":
        return f"{text} {token}"
    raise ValueError(f"unknown intervention_inject {inject!r}")


def prepare_instruction(
    instruction: str,
    inject: InjectMode,
    token_str: Optional[str],
) -> str:
    if inject == "none":
        return instruction.strip()
    if not token_str:
        raise ValueError("intervention_inject prefix|suffix requires an intervention token")
    return apply_injection(instruction, inject, token_str)


def resolve_instruction_for_prompt(
    instruction: str,
    inject: InjectMode,
    token_str: Optional[str],
    *,
    tokenizer=None,
    intervention_token_id: Optional[int] = None,
) -> str:
    """Apply injection once for eval/training prompts.

    ``chat_instruction`` in checkpoints is stored **without** the marker. Legacy runs
    that saved a wrapped instruction are accepted when exactly one marker is already
    present (re-injection is skipped).
    """
    instruction = instruction.strip()
    if inject == "none":
        return instruction
    if not token_str:
        raise ValueError("intervention_inject prefix|suffix requires an intervention token")

    if tokenizer is not None and intervention_token_id is not None:
        inst_ids = tokenizer.encode(instruction, add_special_tokens=False)
        n_markers = sum(1 for tid in inst_ids if tid == intervention_token_id)
        if n_markers == 1:
            return instruction
        if n_markers > 1:
            raise ValueError(
                f"chat_instruction contains {n_markers} intervention markers "
                f"(id={intervention_token_id}); expected raw instruction in checkpoint config"
            )

    return apply_injection(instruction, inject, token_str)


def marker_indices(
    prompt_ids: Sequence[int],
    token_id: int,
    prompt_len: Optional[int] = None,
) -> List[int]:
    end = prompt_len if prompt_len is not None else len(prompt_ids)
    indices = [i for i, tid in enumerate(prompt_ids[:end]) if tid == token_id]
    if len(indices) != 1:
        raise ValueError(
            f"expected exactly one intervention marker (id={token_id}) in prompt, "
            f"found {len(indices)}"
        )
    return indices


def intervention_locations_for_prompt(
    position: str,
    prompt_len: int,
    prompt_ids: Sequence[int],
    *,
    intervention_token_id: Optional[int] = None,
    num_interventions: int = 1,
    share_weights: bool = False,
    pad_mode: str = "first",
    content_span: Optional[Tuple[int, int]] = None,
) -> List[List[int]]:
    if is_marker_position(position):
        if intervention_token_id is None:
            raise ValueError("position=marker requires intervention_token_id")
        locs = marker_indices(prompt_ids, intervention_token_id, prompt_len)
        return [locs] * num_interventions

    if is_content_position(position):
        if content_span is None:
            raise ValueError("position=content_f1|content_l1 requires content_span")
        locs = content_position_indices(position, *content_span)
        return [locs] * num_interventions

    first_n, last_n = parse_positions(position)
    return get_intervention_locations(
        last_position=prompt_len,
        first_n=first_n,
        last_n=last_n,
        num_interventions=num_interventions,
        share_weights=share_weights,
        pad_mode=pad_mode,
    )


def intervention_position_list(
    position: str,
    prompt_ids: Sequence[int],
    *,
    prompt_len: Optional[int] = None,
    intervention_token_id: Optional[int] = None,
    content_span: Optional[Tuple[int, int]] = None,
) -> List[int]:
    plen = prompt_len if prompt_len is not None else len(prompt_ids)
    nested = intervention_locations_for_prompt(
        position,
        plen,
        prompt_ids,
        intervention_token_id=intervention_token_id,
        num_interventions=1,
        content_span=content_span,
    )
    return nested[0]


def init_intervention_token_embedding(
    model,
    tokenizer,
    token_id: int,
    init: InitMode,
) -> None:
    if init == "none":
        return
    embed = model.get_input_embeddings()
    weight = embed.weight
    if token_id < 0 or token_id >= weight.shape[0]:
        raise ValueError(f"intervention_token_id {token_id} out of embedding range")

    if init == "newline":
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if not newline_ids:
            raise ValueError("tokenizer could not encode newline for intervention_token_init")
        ref = weight[newline_ids].mean(dim=0)
        weight.data[token_id] = ref.to(dtype=weight.dtype, device=weight.device)
        return

    if init == "random":
        std = float(weight.data.std().item())
        if std <= 0:
            std = 0.02
        weight.data[token_id] = torch.randn(
            weight.shape[1],
            dtype=weight.dtype,
            device=weight.device,
        ) * std
        return

    raise ValueError(f"unknown intervention_token_init {init!r}")


def build_eval_prompt(
    tokenizer,
    task: str,
    *,
    use_chat_template: bool,
    chat_instruction: Optional[str] = None,
    intervention_inject: InjectMode = "none",
    intervention_token: Optional[str] = None,
    intervention_token_id: Optional[int] = None,
    system_prompt: Optional[str] = None,
    mist_smiles_tags: bool = False,
) -> str:
    from boreft.chem import maybe_append_mist_smiles_open_tag, mist_open_tag_in_prompt
    from boreft.data_utils import chat_prompt
    from boreft.task_config import task_instruction

    instruction = task_instruction(
        task,
        use_chat_template=use_chat_template,
        override=chat_instruction,
    )
    if intervention_token_id is None and intervention_token:
        try:
            intervention_token_id = encode_single_intervention_token(
                tokenizer, intervention_token
            )
        except ValueError:
            intervention_token_id = None
    instruction = resolve_instruction_for_prompt(
        instruction,
        intervention_inject,
        intervention_token,
        tokenizer=tokenizer,
        intervention_token_id=intervention_token_id,
    )
    instruction = maybe_append_mist_smiles_open_tag(
        instruction,
        mist_open_tag_in_prompt(
            mist_smiles_tags=mist_smiles_tags,
            use_chat_template=use_chat_template,
        ),
    )
    if use_chat_template:
        return chat_prompt(tokenizer, instruction, system_prompt=system_prompt)
    return instruction


def intervention_token_id_from_cfg(saved_cfg: Optional[dict]) -> Optional[int]:
    if not saved_cfg:
        return None
    raw = saved_cfg.get("intervention_token_id")
    if raw is None:
        return None
    return int(raw)
