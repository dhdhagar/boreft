"""LLM-based encoder for the bias network.

Instead of feeding frozen sentence-transformer embeddings into
:class:`TargetBiasNetwork`, this module derives the bias-network input from the
same causal LM used for generation:

  definition text -> frozen base model up to the penultimate layer
                  -> a *copied* final decoder block with LoRA adapters
                  -> final RMSNorm -> pooling -> feature vector

The copied final block + LoRA are the only trainable parts here (the base
backbone stays frozen). This encoder is used **only** to compute bias vectors
(mu / logvar). Text generation never routes through these LoRA adapters — the
base model's own final layer is left untouched.

Design notes:
  - The final block is a ``copy.deepcopy`` of the base model's last decoder
    layer (Option 2: a dedicated copied block), so generation is structurally
    incapable of using the encoder LoRA.
  - Position embeddings, the attention mask, and the final norm are computed
    with the base model's own machinery. Decoder forward signatures are
    inspected so both single-RoPE and multi-RoPE layers receive the arguments
    they require, including sliding-window masks where applicable.
  - The penultimate hidden states (input to the final layer) are captured with a
    forward pre-hook so the encoder never re-enters the base model forward. This
    lets the encoder run inside the intervention forward using cached penultimate
    states during training, and directly from the base model during eval for
    held-out words.
"""

from __future__ import annotations

import copy
import inspect
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)

from boreft.task_config import definition_instruction_char_span


# Default LoRA target submodules within a Llama-style decoder layer.
DEFAULT_LORA_TARGETS: Tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

POOLING_LAST_INSTRUCTION = "last_instruction"
POOLING_INSTRUCTION_MEAN = "instruction_mean"
# Backward-compatible alias saved in older checkpoints / configs.
POOLING_LAST_TOKEN = "last_token"

ENCODER_POOLING_CHOICES: Tuple[str, ...] = (
    POOLING_LAST_INSTRUCTION,
    POOLING_INSTRUCTION_MEAN,
    POOLING_LAST_TOKEN,
)


def normalize_encoder_pooling(pooling: str) -> str:
    """Map legacy ``last_token`` to ``last_instruction``."""
    if pooling == POOLING_LAST_TOKEN:
        return POOLING_LAST_INSTRUCTION
    return pooling


def pooling_uses_instruction_mask(pooling: str) -> bool:
    """True when pooling can use non-template instruction token positions."""
    return normalize_encoder_pooling(pooling) in (
        POOLING_LAST_INSTRUCTION,
        POOLING_INSTRUCTION_MEAN,
    )


def pooling_requires_instruction_mask(pooling: str) -> bool:
    """True when pooling fails without an instruction-span mask."""
    return normalize_encoder_pooling(pooling) == POOLING_INSTRUCTION_MEAN


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` with an additive low-rank (LoRA) update.

    ``y = base(x) + scaling * (dropout(x) @ A^T) @ B^T`` where ``B`` is
    zero-initialized so the module is an identity at start of training. The
    wrapped ``base`` linear is frozen; only ``lora_A`` / ``lora_B`` train.
    """

    def __init__(
        self, base: nn.Linear, *, r: int, alpha: float, dropout: float
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        weight = base.weight
        self.lora_A = nn.Parameter(
            torch.zeros(r, base.in_features, dtype=weight.dtype, device=weight.device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, r, dtype=weight.dtype, device=weight.device)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B left at zero -> initial delta is exactly zero.
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora = (self.lora_dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return out + self.scaling * lora


def inject_lora(
    module: nn.Module,
    target_suffixes: Sequence[str],
    *,
    r: int,
    alpha: float,
    dropout: float,
) -> List[str]:
    """Replace matching ``nn.Linear`` submodules with :class:`LoRALinear` in place.

    ``target_suffixes`` are dotted suffixes relative to ``module`` (e.g.
    ``"self_attn.q_proj"``). Returns the list of replaced submodule paths.
    """
    matches: List[str] = []
    for name, child in module.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        if any(name == s or name.endswith("." + s) for s in target_suffixes):
            matches.append(name)

    for name in matches:
        parent_path, _, attr = name.rpartition(".")
        parent = module.get_submodule(parent_path) if parent_path else module
        base_linear = getattr(parent, attr)
        setattr(
            parent,
            attr,
            LoRALinear(base_linear, r=r, alpha=alpha, dropout=dropout),
        )
    return matches


def _pool_hidden_states(
    out: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    pooling: str,
    instruction_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reduce ``[B, L, H]`` final-layer states to ``[B, H]``."""
    pooling = normalize_encoder_pooling(pooling)
    bsz, seq_len, _ = out.shape
    device = out.device

    if pooling == POOLING_INSTRUCTION_MEAN:
        if instruction_mask is None:
            raise RuntimeError(
                f"pooling={POOLING_INSTRUCTION_MEAN!r} requires instruction_mask"
            )
        pool_mask = instruction_mask.to(dtype=out.dtype) * attention_mask.to(
            dtype=out.dtype
        )
        denom = pool_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (out * pool_mask.unsqueeze(-1)).sum(dim=1) / denom

    if pooling == POOLING_LAST_INSTRUCTION:
        if instruction_mask is not None:
            combined = (
                instruction_mask.to(dtype=torch.long)
                * attention_mask.to(dtype=torch.long)
            )
            has_instr = combined.sum(dim=1) > 0
            flipped = combined.flip(dims=[1])
            from_end = flipped.argmax(dim=1)
            last_instr = seq_len - 1 - from_end
            lengths = attention_mask.long().sum(dim=1) - 1
            lengths = lengths.clamp(min=0)
            last_idx = torch.where(has_instr, last_instr, lengths)
        else:
            lengths = attention_mask.long().sum(dim=1) - 1
            last_idx = lengths.clamp(min=0)
        return out[torch.arange(bsz, device=device), last_idx]

    raise ValueError(f"unsupported encoder pooling: {pooling!r}")


class SemanticEncoder(nn.Module):
    """Copied final decoder block + LoRA producing a pooled feature per input.

    The base backbone is frozen and lives outside this module; only the copied
    final block's LoRA adapters train. ``rotary_emb`` and ``config`` are held as
    non-registered references so they are not part of this module's ``state_dict``
    (they belong to the frozen base model and are reconstructed at load time).
    """

    def __init__(
        self,
        decoder_layer: nn.Module,
        norm: nn.Module,
        rotary_emb: nn.Module,
        config,
        *,
        position_embedding_modules: Optional[Dict[str, nn.Module]] = None,
        pooling: str = POOLING_LAST_INSTRUCTION,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_targets: Sequence[str] = DEFAULT_LORA_TARGETS,
        layer_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        pooling = normalize_encoder_pooling(pooling)
        if pooling not in (POOLING_LAST_INSTRUCTION, POOLING_INSTRUCTION_MEAN):
            raise ValueError(
                f"pooling must be one of {POOLING_LAST_INSTRUCTION!r}, "
                f"{POOLING_INSTRUCTION_MEAN!r} (or legacy {POOLING_LAST_TOKEN!r}), "
                f"got {pooling!r}"
            )
        self.pooling = pooling
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_targets = list(lora_targets)
        self.layer_index = layer_index

        # Dedicated copy of the final decoder block (Option 2). Freeze the copied
        # base weights; only injected LoRA adapters train.
        self.layer = copy.deepcopy(decoder_layer)
        for p in self.layer.parameters():
            p.requires_grad_(False)
        self.injected_targets = inject_lora(
            self.layer,
            self.lora_targets,
            r=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
        )

        # Frozen copy of the final norm (RMSNorm) applied before pooling.
        self.norm = copy.deepcopy(norm)
        for p in self.norm.parameters():
            p.requires_grad_(False)

        # Non-registered references to frozen base machinery (kept out of
        # state_dict); wrapped in lists so nn.Module does not register them.
        self._rotary_ref = [rotary_emb]
        if position_embedding_modules is None:
            position_embedding_modules = {"position_embeddings": rotary_emb}
        self._position_embedding_refs = {
            name: [module] for name, module in position_embedding_modules.items()
        }
        self._config = config

    @property
    def rotary_emb(self) -> nn.Module:
        return self._rotary_ref[0]

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def _position_embedding_kwargs(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> Dict[str, object]:
        return {
            name: ref[0](hidden_states, position_ids)
            for name, ref in self._position_embedding_refs.items()
        }

    def _attention_mask(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        mask_kwargs = {
            "config": self._config,
            "input_embeds": hidden_states,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        attention_type = getattr(self.layer, "attention_type", None)
        if attention_type == "sliding_attention":
            return create_sliding_window_causal_mask(**mask_kwargs)
        return create_causal_mask(**mask_kwargs)

    def forward(
        self,
        penultimate: torch.Tensor,
        attention_mask: torch.Tensor,
        instruction_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Map penultimate hidden states to a pooled feature vector.

        Args:
            penultimate:        ``[B, L, H]`` input to the final decoder layer
                                (right-padded), base-model dtype.
            attention_mask:     ``[B, L]`` with 1 for real tokens, 0 for padding.
            instruction_mask:   Optional ``[B, L]`` with 1 for non-template
                                instruction tokens (definition span). Required for
                                ``instruction_mean``; recommended for
                                ``last_instruction`` when templates include suffix
                                text after the definition.

        Returns:
            ``[B, H]`` pooled, final-normed hidden state.
        """
        if penultimate.dim() != 3:
            raise ValueError(
                f"penultimate must be [B, L, H], got {tuple(penultimate.shape)}"
            )
        bsz, seq_len, _ = penultimate.shape
        device = penultimate.device

        position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)
        )
        cache_position = torch.arange(seq_len, device=device)
        position_embedding_kwargs = self._position_embedding_kwargs(
            penultimate, position_ids
        )
        causal_mask = self._attention_mask(
            penultimate, attention_mask, cache_position, position_ids
        )

        out = self.layer(
            penultimate,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            **position_embedding_kwargs,
        )
        if isinstance(out, tuple):
            out = out[0]
        out = self.norm(out)

        return _pool_hidden_states(
            out,
            attention_mask,
            pooling=self.pooling,
            instruction_mask=instruction_mask,
        )


def _base_inner_model(base_model: nn.Module) -> nn.Module:
    """Return the inner transformer (``LlamaModel``) from a CausalLM wrapper."""
    inner = getattr(base_model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner
    if hasattr(base_model, "layers"):
        return base_model
    raise AttributeError("could not locate the inner transformer with .layers")


def resolve_encoder_layer_index(base_model: nn.Module, layer_index: Optional[int]) -> int:
    inner = _base_inner_model(base_model)
    n_layers = len(inner.layers)
    if layer_index is None:
        return n_layers - 1
    if layer_index < 0:
        layer_index = n_layers + layer_index
    if not (0 <= layer_index < n_layers):
        raise ValueError(
            f"encoder layer_index {layer_index} out of range for {n_layers} layers"
        )
    return layer_index


def build_semantic_encoder(
    base_model: nn.Module,
    *,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    lora_targets: Sequence[str] = DEFAULT_LORA_TARGETS,
    pooling: str = POOLING_LAST_INSTRUCTION,
    layer_index: Optional[int] = None,
) -> SemanticEncoder:
    """Construct a :class:`SemanticEncoder` from a frozen causal LM."""
    inner = _base_inner_model(base_model)
    idx = resolve_encoder_layer_index(base_model, layer_index)
    n_layers = len(inner.layers)
    if idx != n_layers - 1:
        import warnings

        warnings.warn(
            f"SemanticEncoder copies decoder layer {idx} but applies the model's "
            f"final norm (meant for layer {n_layers - 1}). The pooled features may "
            f"be mis-normalized; prefer the final layer unless you know why.",
            stacklevel=2,
        )

    layer = inner.layers[idx]
    forward_parameters = inspect.signature(layer.forward).parameters
    position_embedding_modules: Dict[str, nn.Module] = {}
    for name in forward_parameters:
        if not name.startswith("position_embeddings"):
            continue
        suffix = name.removeprefix("position_embeddings")
        rotary_name = f"rotary_emb{suffix}"
        if suffix in ("", "_global"):
            rotary_name = "rotary_emb"
        rotary = getattr(inner, rotary_name, None)
        if rotary is None:
            raise AttributeError(
                f"{type(layer).__name__}.forward requires {name!r}, but "
                f"{type(inner).__name__} has no {rotary_name!r} module"
            )
        position_embedding_modules[name] = rotary
    if not position_embedding_modules:
        raise TypeError(
            f"{type(layer).__name__}.forward has no supported "
            "position_embeddings parameter"
        )

    encoder = SemanticEncoder(
        layer,
        inner.norm,
        inner.rotary_emb,
        inner.config,
        position_embedding_modules=position_embedding_modules,
        pooling=pooling,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_targets=lora_targets,
        layer_index=idx,
    )
    return encoder


@torch.no_grad()
def capture_penultimate(
    base_model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    layer_index: Optional[int] = None,
) -> torch.Tensor:
    """Return the hidden states fed into the final (or ``layer_index``) layer.

    Runs the frozen inner transformer once and captures the input to the target
    decoder layer with a forward pre-hook. Shape ``[B, L, H]``.
    """
    inner = _base_inner_model(base_model)
    idx = resolve_encoder_layer_index(base_model, layer_index)
    captured: dict = {}

    def _pre_hook(module, args, kwargs):
        hs = kwargs.get("hidden_states")
        if hs is None and args:
            hs = args[0]
        captured["hidden_states"] = hs.detach()

    handle = inner.layers[idx].register_forward_pre_hook(_pre_hook, with_kwargs=True)
    try:
        inner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    finally:
        handle.remove()

    if "hidden_states" not in captured:
        raise RuntimeError("failed to capture penultimate hidden states")
    return captured["hidden_states"]


def instruction_mask_from_offset_mapping(
    offset_mapping: Sequence[Tuple[Optional[int], Optional[int]]],
    char_start: int,
    char_end: int,
) -> List[int]:
    """Mark tokens whose character offset starts inside ``[char_start, char_end)``."""
    mask: List[int] = []
    for start, _end in offset_mapping:
        if start is None:
            mask.append(0)
        elif char_start <= start < char_end:
            mask.append(1)
        else:
            mask.append(0)
    return mask


def instruction_masks_from_batched_encoding(
    offset_mapping_batch: Sequence[Sequence[Tuple[Optional[int], Optional[int]]]],
    attention_mask: torch.Tensor,
    char_spans: Sequence[Tuple[int, int]],
) -> torch.Tensor:
    """Build instruction masks aligned to a batched tokenizer output."""
    if len(offset_mapping_batch) != len(char_spans):
        raise ValueError("offset_mapping_batch and char_spans length mismatch")
    rows: List[List[int]] = []
    for offsets, (char_start, char_end) in zip(offset_mapping_batch, char_spans):
        rows.append(
            instruction_mask_from_offset_mapping(offsets, char_start, char_end)
        )
    mask = torch.tensor(rows, dtype=torch.long, device=attention_mask.device)
    return mask * attention_mask.long()


def encode_definition_inputs(
    tokenizer,
    texts: Sequence[str],
    *,
    device,
    max_length: int = 64,
    task: Optional[str] = None,
    word_definition_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    build_instruction_mask: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Right-pad tokenize encoder input texts.

    Returns ``(input_ids, attn_mask[, instruction_mask])``. Padding side is forced
    to ``right`` so pooling reads real tokens left-to-right; the tokenizer's
    original padding side is restored afterwards.

    When ``build_instruction_mask=True``, pass ``task`` and ``word_definition_pairs``
    (same length as ``texts``). Masks are derived from the batched encoding's
    ``offset_mapping`` so they stay aligned with ``input_ids``.
    """
    if build_instruction_mask:
        if task is None or word_definition_pairs is None:
            raise ValueError(
                "build_instruction_mask requires task and word_definition_pairs"
            )
        if len(word_definition_pairs) != len(texts):
            raise ValueError(
                "word_definition_pairs length must match texts "
                f"({len(word_definition_pairs)} vs {len(texts)})"
            )
    if build_instruction_mask and not getattr(tokenizer, "is_fast", False):
        raise TypeError(
            "build_instruction_mask requires a fast tokenizer "
            f"(use_fast=True); got {type(tokenizer).__name__}"
        )
    prev_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "right"
    try:
        enc = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=build_instruction_mask,
        )
    finally:
        tokenizer.padding_side = prev_side
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    instruction_mask_tensor: Optional[torch.Tensor] = None
    if build_instruction_mask:
        assert task is not None and word_definition_pairs is not None
        char_spans = [
            definition_instruction_char_span(task, word, definition)
            for word, definition in word_definition_pairs
        ]
        instruction_mask_tensor = instruction_masks_from_batched_encoding(
            enc["offset_mapping"],
            attention_mask,
            char_spans,
        )
    return input_ids, attention_mask, instruction_mask_tensor
