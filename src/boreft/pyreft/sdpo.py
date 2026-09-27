"""SDPO-style self-distillation auxiliary loss for REFT interventions.

The "teacher" is the frozen base model with the target's definition placed
in-context (privileged info); the "student" is the intervened model (per-word
bias ``b_w``) seeing only the bare prompt. We distill the teacher's next-token
distribution into the student over a mixture of trajectory sources:

  - on-policy : sequences sampled from the student (bias fixed to ``mu``),
  - off-policy: sequences sampled from the teacher (base model + definition),
  - gold      : the training target (optional).

Sampling happens once per step; the resulting sequences are reused for the
single student forward (grad) and single teacher forward (detached). The
per-token divergence (forward/reverse KL or JS) is in
:func:`boreft.pyreft.losses.sdpo_distillation_loss`.

In VAE mode the SDPO student scoring forward reuses the *same* reparameterized
bias ``b`` sampled by the CE forward this step (shared via the intervention's
per-step ``_shared_b`` store), so both objectives backprop through one
reparameterization node. On-policy rollouts, in contrast, are generated from the
deterministic mean bias ``mu`` (interventions are switched to eval during
generation).
"""

from __future__ import annotations

import contextlib
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from boreft.data_utils import IGNORE_INDEX, build_reft_row
from boreft.intervention_marker import intervention_position_list
from boreft.pyreft.losses import sdpo_distillation_loss


@dataclass
class SDPOConfig:
    """SDPO loss hyperparameters (lambda applied by the caller)."""

    divergence: str = "forward_kl"
    temperature: float = 1.0
    sample_temperature: float = 1.0
    sample_top_p: float = 1.0
    max_new_tokens: int = 8
    n_onpolicy: int = 4
    n_offpolicy: int = 0
    offpolicy_pool: int = 0  # cached teacher samples per word (lazy); >= n_offpolicy
    include_gold: bool = True
    position: str = "l1"
    intervention_token_id: Optional[int] = None
    content_span: Optional[Tuple[int, int]] = None

    @property
    def has_source(self) -> bool:
        return self.n_onpolicy > 0 or self.n_offpolicy > 0 or self.include_gold


def _iter_interventions(intervenable) -> List:
    """Flat list of intervention modules on a pyvene IntervenableModel."""
    out = []
    for v in intervenable.interventions.values():
        out.append(v[0] if isinstance(v, (list, tuple)) else v)
    return out


@contextlib.contextmanager
def _eval_mode(module):
    """Temporarily put ``module`` (and its submodules) in eval mode, then restore."""
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


@torch.no_grad()
def _teacher_generate(base_model, input_ids, attn, cfg: "SDPOConfig", tokenizer):
    """Sample from the base (teacher) model with it forced to eval mode."""
    with _eval_mode(base_model):
        return base_model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            temperature=cfg.sample_temperature,
            top_p=cfg.sample_top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )


def _trim_generated(ids: Sequence[int], eos_id: Optional[int], pad_id: int) -> List[int]:
    """Cut a generated continuation at the first EOS (inclusive) / pad."""
    out: List[int] = []
    for t in ids:
        t = int(t)
        if t == pad_id and t != eos_id:
            break
        out.append(t)
        if eos_id is not None and t == eos_id:
            break
    if not out:
        # Empty after trimming (continuation began with a pad token). Rare; the
        # sequence is dropped. ``warnings`` dedupes by call site, so this fires
        # at most once per run by default.
        warnings.warn(
            "SDPO: a generated continuation was empty after trimming; skipping it.",
            RuntimeWarning,
            stacklevel=2,
        )
    return out


def _dedupe_records(
    records: List[Tuple[int, List[int]]],
) -> List[Tuple[int, List[int]]]:
    """Drop duplicate ``(word_id, sequence)`` pairs, preserving first occurrence."""
    seen: set = set()
    unique: List[Tuple[int, List[int]]] = []
    for w, y in records:
        key = (w, tuple(y))
        if key not in seen:
            seen.add(key)
            unique.append((w, y))
    return unique


def _left_pad(seqs: List[List[int]], pad_id: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Left-pad variable-length id lists for batched base-model generation."""
    max_len = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, max_len - len(s):] = torch.tensor(s, dtype=torch.long)
        attn[i, max_len - len(s):] = 1
    return input_ids.to(device), attn.to(device)


@torch.no_grad()
def _sample_onpolicy(
    intervenable,
    prompt_ids: List[int],
    word_ids: List[int],
    cfg: SDPOConfig,
    tokenizer,
    device,
) -> List[Tuple[int, List[int]]]:
    """Sample ``n_onpolicy`` continuations per word from the intervened student.

    Rollouts use the deterministic mean bias ``mu`` (interventions are switched to
    eval for generation), independent of the sampled bias used for scoring.
    """
    if cfg.n_onpolicy <= 0:
        return []
    n = cfg.n_onpolicy
    R = len(word_ids) * n
    L = len(prompt_ids)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device).repeat(R, 1)
    attn = torch.ones_like(input_ids)
    pos_list = intervention_position_list(
        cfg.position,
        prompt_ids,
        prompt_len=L,
        intervention_token_id=cfg.intervention_token_id,
        content_span=cfg.content_span,
    )
    row_wids = [int(w) for w in word_ids for _ in range(n)]  # item-major
    subspaces = [[[w] for w in row_wids]]  # [num_interventions=1][R][value]
    unit_locations = {"sources->base": (None, [[pos_list]] * R)}

    ivs = _iter_interventions(intervenable)
    prev_modes = [iv.training for iv in ivs]
    for iv in ivs:
        iv.eval()  # mu (deterministic) bias for on-policy rollouts
    try:
        _, output_ids = intervenable.generate(
            base={"input_ids": input_ids, "attention_mask": attn},
            unit_locations=unit_locations,
            intervene_on_prompt=True,
            subspaces=subspaces,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            temperature=cfg.sample_temperature,
            top_p=cfg.sample_top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    finally:
        for iv, mode in zip(ivs, prev_modes):
            iv.train(mode)
    records: List[Tuple[int, List[int]]] = []
    for r in range(R):
        y = _trim_generated(
            output_ids[r, L:].tolist(), tokenizer.eos_token_id, tokenizer.pad_token_id
        )
        if y:
            records.append((row_wids[r], y))
    return records


@torch.no_grad()
def _sample_offpolicy(
    base_model,
    teacher_prompt_ids: List[List[int]],
    word_ids: List[int],
    cfg: SDPOConfig,
    tokenizer,
    device,
) -> List[Tuple[int, List[int]]]:
    """Sample ``n_offpolicy`` continuations per word from the base+definition teacher."""
    if cfg.n_offpolicy <= 0:
        return []
    n = cfg.n_offpolicy
    row_wids = [int(w) for w in word_ids for _ in range(n)]
    prompts = [teacher_prompt_ids[w] for w in row_wids]
    input_ids, attn = _left_pad(prompts, tokenizer.pad_token_id, device)
    L = input_ids.shape[1]
    output_ids = _teacher_generate(base_model, input_ids, attn, cfg, tokenizer)
    records: List[Tuple[int, List[int]]] = []
    for r in range(output_ids.shape[0]):
        y = _trim_generated(
            output_ids[r, L:].tolist(), tokenizer.eos_token_id, tokenizer.pad_token_id
        )
        if y:
            records.append((row_wids[r], y))
    return records


@torch.no_grad()
def _generate_offpolicy_pool(
    base_model,
    teacher_prompt_ids: List[List[int]],
    wids: List[int],
    n: int,
    cfg: SDPOConfig,
    tokenizer,
    device,
) -> Dict[int, List[List[int]]]:
    """Sample ``n`` teacher continuations for each (unique) word id."""
    pool: Dict[int, List[List[int]]] = {int(w): [] for w in wids}
    if not wids or n <= 0:
        return pool
    row_wids = [int(w) for w in wids for _ in range(n)]
    prompts = [teacher_prompt_ids[w] for w in row_wids]
    input_ids, attn = _left_pad(prompts, tokenizer.pad_token_id, device)
    L = input_ids.shape[1]
    output_ids = _teacher_generate(base_model, input_ids, attn, cfg, tokenizer)
    for r in range(output_ids.shape[0]):
        y = _trim_generated(
            output_ids[r, L:].tolist(), tokenizer.eos_token_id, tokenizer.pad_token_id
        )
        if y:
            pool[row_wids[r]].append(y)
    return pool


def _ensure_offpolicy_cached(
    base_model,
    teacher_prompt_ids: List[List[int]],
    word_ids: Sequence[int],
    cache: Dict[int, List[List[int]]],
    cfg: SDPOConfig,
    tokenizer,
    device,
) -> None:
    """Populate ``cache`` for any batch word ids not yet sampled (lazy, once each).

    The teacher distribution ``pi_base(.|x, f_w)`` is fixed (frozen base, fixed
    definition context), so a word's pool is sampled exactly once and reused for
    the rest of training. Words that yield no usable continuation are cached as
    an empty list so they are not retried every step.
    """
    missing = [w for w in dict.fromkeys(int(w) for w in word_ids) if w not in cache]
    if not missing:
        return
    pool = _generate_offpolicy_pool(
        base_model, teacher_prompt_ids, missing, cfg.offpolicy_pool, cfg, tokenizer, device
    )
    for w in missing:
        cache[w] = pool.get(w, [])


def _draw_offpolicy(
    word_ids: Sequence[int],
    cache: Dict[int, List[List[int]]],
    n: int,
    seed: int,
) -> List[Tuple[int, List[int]]]:
    """Draw up to ``n`` cached teacher continuations per batch word (no replacement)."""
    rng = random.Random(seed)
    records: List[Tuple[int, List[int]]] = []
    for w in word_ids:
        w = int(w)
        pool = cache.get(w) or []
        if not pool:
            continue
        for y in rng.sample(pool, min(n, len(pool))):
            records.append((w, list(y)))
    return records


def _collate_teacher(
    rows: List[Dict], pad_id: int, device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad teacher [prefix; y] rows into (input_ids, attention_mask, labels)."""
    max_len = max(len(r["input_ids"]) for r in rows)
    n = len(rows)
    input_ids = torch.full((n, max_len), pad_id, dtype=torch.long)
    labels = torch.full((n, max_len), IGNORE_INDEX, dtype=torch.long)
    attn = torch.zeros((n, max_len), dtype=torch.long)
    for i, r in enumerate(rows):
        ln = len(r["input_ids"])
        input_ids[i, :ln] = torch.tensor(r["input_ids"], dtype=torch.long)
        labels[i, :ln] = torch.tensor(r["labels"], dtype=torch.long)
        attn[i, :ln] = 1
    return input_ids.to(device), attn.to(device), labels.to(device)


def _build_teacher_row(prefix_ids: List[int], y_ids: List[int]) -> Dict:
    return {
        "input_ids": list(prefix_ids) + list(y_ids),
        "labels": [IGNORE_INDEX] * len(prefix_ids) + list(y_ids),
    }


def _gather_next_token_logits(
    logits: torch.Tensor, labels: torch.Tensor
) -> List[Optional[torch.Tensor]]:
    """Per row, return logits that *predict* the labeled (target) tokens.

    For a labeled position ``j`` (``labels[j] != IGNORE``), the predictive
    distribution is ``logits[j - 1]`` (standard next-token shift). Returns one
    ``[n_j, V]`` tensor per row (or ``None`` if the row has no labels).
    """
    out: List[Optional[torch.Tensor]] = []
    for r in range(labels.shape[0]):
        idx = (labels[r] != IGNORE_INDEX).nonzero(as_tuple=True)[0]
        idx = idx[idx > 0]
        out.append(logits[r, idx - 1, :] if idx.numel() > 0 else None)
    return out


def compute_sdpo_loss(
    *,
    intervenable,
    base_model,
    tokenizer,
    word_ids: Sequence[int],
    student_prompt_ids: List[int],
    teacher_prompt_ids: List[List[int]],
    target_ids: List[List[int]],
    cfg: SDPOConfig,
    collator,
    device,
    offpolicy_cache: Optional[Dict[int, List[List[int]]]] = None,
    draw_seed: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the (token-mean, lambda-free) SDPO distillation loss for a batch.

    ``word_ids`` are the per-example target ids (from ``subspaces``).
    ``student_prompt_ids`` is the constant bare prompt; ``teacher_prompt_ids`` /
    ``target_ids`` are indexed by word id. Returns ``(loss, metrics)``.
    """
    word_ids = [int(w) for w in word_ids]

    metrics: Dict[str, float] = {"sdpo_n_seq": 0.0, "sdpo_n_tokens": 0.0}
    zero = torch.zeros((), device=device)
    if not cfg.has_source:
        return zero, metrics

    # In VAE mode the student scoring forward (below) reuses the bias sampled by
    # the CE forward this step (shared grad path via _shared_b and cached
    # bias-network rows via _shared_bias). On-policy rollouts use mu (see
    # _sample_onpolicy).
    records: List[Tuple[int, List[int]]] = []
    records += _sample_onpolicy(
        intervenable, student_prompt_ids, word_ids, cfg, tokenizer, device
    )
    if cfg.n_offpolicy > 0:
        if offpolicy_cache is not None:
            # Lazy cache: sample each word's teacher pool once, then reuse.
            _ensure_offpolicy_cached(
                base_model,
                teacher_prompt_ids,
                word_ids,
                offpolicy_cache,
                cfg,
                tokenizer,
                device,
            )
            records += _draw_offpolicy(
                word_ids, offpolicy_cache, cfg.n_offpolicy, draw_seed
            )
        else:
            records += _sample_offpolicy(
                base_model, teacher_prompt_ids, word_ids, cfg, tokenizer, device
            )
    if cfg.include_gold:
        for w in word_ids:
            y = list(target_ids[w])
            if y:
                records.append((w, y))

    # Deduplicate: distill each distinct (word_id, sequence) once even if the
    # same continuation shows up across on-policy / off-policy / gold sources.
    records = _dedupe_records(records)
    if not records:
        return zero, metrics

    # --- build student ([prompt; y]) and teacher ([prefix; y]) batches ---
    student_rows: List[Dict] = []
    teacher_rows: List[Dict] = []
    for w, y in records:
        student_rows.append(
            build_reft_row(
                input_ids=list(student_prompt_ids) + y,
                prompt_len=len(student_prompt_ids),
                prompt_ids=student_prompt_ids,
                word_id=w,
                pad_token_id=tokenizer.pad_token_id,
                position=cfg.position,
                intervention_token_id=cfg.intervention_token_id,
                content_span=cfg.content_span,
            )
        )
        teacher_rows.append(_build_teacher_row(teacher_prompt_ids[w], y))

    # --- score: student (grad, sampled bias) and teacher (detached, base) ---
    student_batch = collator(student_rows)
    s_input_ids = student_batch["input_ids"].to(device)
    s_attn = student_batch["attention_mask"].to(device)
    s_labels = student_batch["labels"].to(device)
    unit_locations = {
        "sources->base": (
            None,
            student_batch["intervention_locations"].permute(1, 0, 2).tolist(),
        )
    }
    base_out, cf_out = intervenable(
        {"input_ids": s_input_ids, "attention_mask": s_attn},
        unit_locations=unit_locations,
        labels=s_labels,
        subspaces=student_batch["subspaces"].permute(1, 0, 2).tolist(),
    )
    student_logits = (cf_out if cf_out is not None else base_out).logits

    t_input_ids, t_attn, t_labels = _collate_teacher(
        teacher_rows, tokenizer.pad_token_id, device
    )
    with torch.no_grad(), _eval_mode(base_model):
        teacher_logits = base_model(
            input_ids=t_input_ids, attention_mask=t_attn
        ).logits

    # --- gather aligned next-token logits, then per-token divergence ---
    s_sel = _gather_next_token_logits(student_logits, s_labels)
    t_sel = _gather_next_token_logits(teacher_logits, t_labels)
    s_parts, t_parts = [], []
    for s_part, t_part in zip(s_sel, t_sel):
        if s_part is None or t_part is None:
            continue
        if s_part.shape[0] != t_part.shape[0]:
            continue  # tokenization boundary mismatch — skip this row
        s_parts.append(s_part)
        t_parts.append(t_part.detach())

    if not s_parts:
        return zero, metrics

    student_sel = torch.cat(s_parts, dim=0)
    teacher_sel = torch.cat(t_parts, dim=0)
    loss = sdpo_distillation_loss(
        student_sel,
        teacher_sel,
        temperature=cfg.temperature,
        divergence=cfg.divergence,
    )
    metrics["sdpo_n_seq"] = float(len(s_parts))
    metrics["sdpo_n_tokens"] = float(student_sel.shape[0])
    return loss, metrics
