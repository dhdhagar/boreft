"""Reconstruction audit helpers for molopt (advisor diagnostics 1–2).

Separates string recall from molecular identity, scores teacher-forced
likelihood under own / shuffled / no intervention, and supports
differentiable code inversion with the frozen intervention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from boreft.bias_tables import stack_bias_mu_std, stack_bias_vectors
from boreft.catalog_decode import catalog_maxima
from boreft.chem import (
    canonical_smiles,
    generation_smiles,
    is_valid_smiles,
    maybe_repair_invalid_smiles,
    tanimoto_similarity,
    unwrap_smiles_tags,
)
from boreft.data_utils import (
    IGNORE_INDEX,
    ReftDataCollator,
    build_reft_row,
    tokenize_model_text,
)
from boreft.eval.semantle import _get_intervention, generate_text
from boreft.learn_bias import _build_ce_batch, _lm_loss
from boreft.oracles import TDC_ORACLE_NAMES


def items_words(items: Sequence) -> list[str]:
    """Same ``word``/``target`` rule as ``load_eval_checkpoint``."""
    words: list[str] = []
    for row in items:
        if not isinstance(row, dict):
            raise TypeError("items.json must be a list of objects")
        if "word" in row:
            words.append(str(row["word"]))
        else:
            words.append(str(row["target"]))
    return words


def try_stack_bias_mu_std(reft_model, word_ids: Sequence[int]):
    """``(mu, std)`` or ``(mu, None)`` when the checkpoint has no logvar."""
    ids = [int(i) for i in word_ids]
    try:
        return stack_bias_mu_std(reft_model, ids)
    except ValueError:
        mu = stack_bias_vectors(reft_model, ids)
        return mu, None


@dataclass(frozen=True)
class IdentityMatch:
    decoded: str
    gold: str
    raw_exact: bool
    canonical_exact: bool
    graph_exact: bool
    valid: bool
    tanimoto: float | None


def bare_smiles(text: str) -> str:
    return unwrap_smiles_tags(str(text).strip())


def identity_match(gold: str, decoded: str) -> IdentityMatch:
    """String, stereo-preserving, and stereo-stripped molecular identity."""
    gold_raw = bare_smiles(gold)
    dec_raw = bare_smiles(decoded)
    gold_iso = canonical_smiles(gold_raw, isomeric=True)
    dec_iso = canonical_smiles(dec_raw, isomeric=True)
    gold_graph = canonical_smiles(gold_raw, isomeric=False)
    dec_graph = canonical_smiles(dec_raw, isomeric=False)
    valid = is_valid_smiles(dec_raw)
    tani = tanimoto_similarity(gold_raw, dec_raw) if valid else None
    return IdentityMatch(
        decoded=dec_raw,
        gold=gold_raw,
        raw_exact=bool(gold_raw) and gold_raw == dec_raw,
        canonical_exact=bool(gold_iso) and gold_iso == dec_iso,
        graph_exact=bool(gold_graph) and gold_graph == dec_graph,
        valid=valid,
        tanimoto=None if tani is None else float(tani),
    )


def identity_payload(row: IdentityMatch) -> dict[str, Any]:
    return asdict(row)


def select_audit_panel(
    smiles: Sequence[str],
    *,
    catalog_indices: Sequence[int] = (),
    n: int = 32,
    seed: int = 42,
) -> list[int]:
    """Catalog maxima plus a length-stratified sample (unique, stable)."""
    if n < 1:
        raise ValueError("panel size must be positive")
    total = len(smiles)
    if total < 1:
        raise ValueError("need at least one train SMILES")
    chosen: list[int] = []
    seen: set[int] = set()
    for raw in catalog_indices:
        index = int(raw)
        if index < 0 or index >= total:
            raise ValueError(f"catalog index {index} out of range for {total} SMILES")
        if index not in seen:
            seen.add(index)
            chosen.append(index)
    remaining = [i for i in range(total) if i not in seen]
    need = max(0, min(n, total) - len(chosen))
    if need == 0 or not remaining:
        return chosen
    remaining.sort(key=lambda i: (len(smiles[i]), i))
    rng = np.random.default_rng(seed)
    bins = 5
    per_bin = max(1, need // bins)
    picked: list[int] = []
    for start in range(bins):
        lo = int(round(start * len(remaining) / bins))
        hi = int(round((start + 1) * len(remaining) / bins))
        bucket = remaining[lo:hi]
        if not bucket:
            continue
        take = min(per_bin, len(bucket), need - len(picked))
        if take <= 0:
            break
        if take >= len(bucket):
            picked.extend(bucket)
        else:
            chosen_pos = rng.choice(len(bucket), size=take, replace=False)
            picked.extend(bucket[int(p)] for p in chosen_pos)
    leftover = [i for i in remaining if i not in set(picked)]
    while len(picked) < need and leftover:
        extra = leftover[int(rng.integers(0, len(leftover)))]
        leftover.remove(extra)
        picked.append(extra)
    chosen.extend(picked[:need])
    return chosen  # catalog maxima are never dropped when n is smaller


def catalog_panel_indices(
    smiles: Sequence[str],
    scores_by_oracle: dict[str, Sequence[float]],
    *,
    n: int = 32,
    seed: int = 42,
    oracles: Sequence[str] = TDC_ORACLE_NAMES,
) -> tuple[list[int], list[dict[str, Any]]]:
    maxima = catalog_maxima(smiles, scores_by_oracle, oracles=oracles)
    catalog = [row.index for row in maxima]
    panel = select_audit_panel(
        smiles, catalog_indices=catalog, n=n, seed=seed
    )
    meta = [
        {
            "oracle": row.oracle,
            "index": row.index,
            "smiles": row.smiles,
            "score": row.score,
        }
        for row in maxima
    ]
    return panel, meta


def gold_generation_text(ckpt, smiles: str) -> str:
    cfg = getattr(ckpt, "saved_cfg", None) or {}
    task = cfg.get("task", "molopt")
    return generation_smiles(
        bare_smiles(smiles),
        smiles_tags=bool(cfg.get("smiles_tags")) and task == "molopt",
        mist_smiles_tags=bool(cfg.get("mist_smiles_tags")) and task == "molopt",
        mist_open_in_target=bool(cfg.get("mist_smiles_tags"))
        and bool(getattr(ckpt, "from_chat_template", False)),
    )


def build_teacher_forced_batch(ckpt, smiles: str, prompt: str | None, device):
    """``[prompt ; gold]`` CE row. Checkpoint prompt reuses training concat."""
    bare = bare_smiles(smiles)
    if prompt is None or prompt == getattr(ckpt, "prompt", None):
        return _build_ce_batch(ckpt, bare, device)
    tok = ckpt.tokenizer
    gold = gold_generation_text(ckpt, bare)
    eos = tok.eos_token or ""
    if ckpt.from_chat_template:
        full = prompt + gold
    else:
        full = prompt + " " + gold + eos
    prompt_ids = tokenize_model_text(
        tok, prompt, from_chat_template=ckpt.from_chat_template, return_tensors="pt"
    )["input_ids"][0]
    input_ids = tokenize_model_text(
        tok, full, from_chat_template=ckpt.from_chat_template, return_tensors="pt"
    )["input_ids"][0]
    saved = ckpt.saved_cfg or {}
    row = build_reft_row(
        input_ids=input_ids,
        prompt_len=len(prompt_ids),
        prompt_ids=prompt_ids.tolist(),
        word_id=0,
        pad_token_id=tok.pad_token_id,
        position=saved.get("position", "l1"),
        intervention_token_id=ckpt.intervention_token_id,
        content_span=ckpt.content_span,
    )
    collator = ReftDataCollator(tokenizer=tok, model=ckpt.reft_model.model)
    batch = collator([row])
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def teacher_forced_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> dict[str, Any]:
    """NLL and greedy-prefix stats on teacher-forced target tokens."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels.ne(ignore_index)
    n_tokens = int(valid.sum().item())
    if n_tokens < 1:
        return {
            "nll": None,
            "token_acc": None,
            "greedy_prefix": False,
            "n_tokens": 0,
        }
    logp = F.log_softmax(shift_logits, dim=-1)
    safe = shift_labels.masked_fill(~valid, 0)
    token_nll = -logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    pred = shift_logits.argmax(dim=-1)
    hits = pred.eq(shift_labels) & valid
    return {
        "nll": float((token_nll * valid).sum().item() / n_tokens),
        "token_acc": float(hits.sum().item() / n_tokens),
        "greedy_prefix": bool(hits[valid].all().item()),
        "n_tokens": n_tokens,
    }


def _with_word_id(batch: dict, word_id: int) -> dict:
    """Keep the collated subspace layout; only replace the looked-up word id."""
    out = dict(batch)
    subspaces = batch["subspaces"].clone()
    subspaces.fill_(int(word_id))
    out["subspaces"] = subspaces
    return out


def generation_word_idx(code):
    """Pass ints through so ``generate_text`` looks up ``μ``; arrays stay raw."""
    if isinstance(code, (int, np.integer)):
        return int(code)
    return np.asarray(code, dtype=np.float32)


@torch.no_grad()
def intervened_logits(ckpt, batch: dict, *, word_id: int):
    row = _with_word_id(batch, word_id)
    unit_locations = {
        "sources->base": (
            None,
            row["intervention_locations"].permute(1, 0, 2).tolist(),
        )
    }
    _base, cf = ckpt.reft_model(
        {"input_ids": row["input_ids"], "attention_mask": row["attention_mask"]},
        unit_locations=unit_locations,
        labels=row["labels"],
        subspaces=row["subspaces"].permute(1, 0, 2).tolist(),
    )
    out = cf if cf is not None else _base
    return out.logits, out.loss


@torch.no_grad()
def base_logits(ckpt, batch: dict):
    """Teacher-forced logits with the frozen base LM (no intervention)."""
    out = ckpt.reft_model.model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    return out.logits, out.loss


def forced_bias_logits(ckpt, batch: dict, bias: torch.Tensor):
    iv = _get_intervention(ckpt.reft_model)
    iv.set_forced_bias(bias)
    try:
        unit_locations = {
            "sources->base": (
                None,
                batch["intervention_locations"].permute(1, 0, 2).tolist(),
            )
        }
        _base, cf = ckpt.reft_model(
            {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]},
            unit_locations=unit_locations,
            labels=batch["labels"],
            subspaces=batch["subspaces"].permute(1, 0, 2).tolist(),
        )
        out = cf if cf is not None else _base
        return out.logits, out.loss
    finally:
        iv.clear_forced_bias()


def teacher_forced_bundle(ckpt, batch: dict, *, word_id: int) -> dict[str, Any]:
    ckpt.reft_model.eval()
    logits, _loss = intervened_logits(ckpt, batch, word_id=word_id)
    return teacher_forced_from_logits(logits, batch["labels"])


def teacher_forced_base_bundle(ckpt, batch: dict) -> dict[str, Any]:
    ckpt.reft_model.eval()
    logits, _loss = base_logits(ckpt, batch)
    return teacher_forced_from_logits(logits, batch["labels"])


@torch.no_grad()
def teacher_forced_bias_bundle(ckpt, batch: dict, bias: np.ndarray) -> dict[str, Any]:
    ckpt.reft_model.eval()
    device = ckpt.reft_model.get_device()
    tensor = torch.as_tensor(bias, dtype=torch.float32, device=device)
    logits, _loss = forced_bias_logits(ckpt, batch, tensor)
    return teacher_forced_from_logits(logits, batch["labels"])


def sample_posterior_codes(
    reft_model,
    word_id: int,
    *,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    mu, std = stack_bias_mu_std(reft_model, [int(word_id)])
    noise = rng.standard_normal((n, mu.shape[1])).astype(np.float32)
    return mu + std * noise


def decode_at_code(
    ckpt,
    code,
    *,
    prompt: str,
    from_chat: bool,
    span,
    temperature: float,
    max_new_tokens: int,
) -> str:
    saved = ckpt.saved_cfg or {}
    use_sample = temperature > 0
    text = generate_text(
        ckpt.reft_model,
        ckpt.tokenizer,
        prompt,
        generation_word_idx(code),
        max_new_tokens=max_new_tokens,
        use_sample=use_sample,
        temperature=max(temperature, 1e-8),
        top_p=1.0,
        position=saved.get("position", "l1"),
        assistant_suffix=ckpt.assistant_suffix,
        from_chat_template=from_chat,
        intervention_token_id=ckpt.intervention_token_id,
        content_span=span,
    )
    smiles, _repaired = maybe_repair_invalid_smiles(text)
    return smiles


def project_to_bounds(point: torch.Tensor, bounds: np.ndarray) -> torch.Tensor:
    lo = torch.as_tensor(bounds[0], dtype=point.dtype, device=point.device)
    hi = torch.as_tensor(bounds[1], dtype=point.dtype, device=point.device)
    return point.clamp(min=lo, max=hi)


def invert_code(
    ckpt,
    batch: dict,
    *,
    init: np.ndarray,
    bounds: np.ndarray | None,
    steps: int,
    lr: float,
) -> dict[str, Any]:
    """Minimize teacher-forced NLL over a single code, model frozen."""
    if steps < 1:
        raise ValueError("inversion steps must be positive")
    device = ckpt.reft_model.get_device()
    ckpt.reft_model.eval()
    for param in ckpt.reft_model.parameters():
        param.requires_grad_(False)
    start = np.asarray(init, dtype=np.float32)
    if bounds is not None:
        start = np.clip(start, bounds[0], bounds[1]).astype(np.float32)
    code = torch.nn.Parameter(torch.as_tensor(start, dtype=torch.float32, device=device))
    opt = torch.optim.Adam([code], lr=lr)
    iv = _get_intervention(ckpt.reft_model)
    iv.eval()
    history: list[dict[str, Any]] = []

    def _snapshot() -> np.ndarray:
        return code.detach().float().cpu().numpy().copy()

    def _eval_nll() -> tuple[float, np.ndarray]:
        with torch.no_grad():
            loss = _lm_loss(
                ckpt.reft_model,
                batch,
                use_margin_loss=False,
                margin_loss_margin=0.0,
            )
        return float(loss.item()), _snapshot()

    iv.set_forced_bias(code)
    try:
        init_nll, init_code = _eval_nll()
        best = {"nll": init_nll, "code": init_code}
        history.append({"step": 0, "nll": init_nll})
        for step in range(1, steps + 1):
            loss = _lm_loss(
                ckpt.reft_model,
                batch,
                use_margin_loss=False,
                margin_loss_margin=0.0,
            )
            nll = float(loss.detach().item())
            if nll < best["nll"]:
                best = {"nll": nll, "code": _snapshot()}
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if bounds is not None:
                with torch.no_grad():
                    code.copy_(project_to_bounds(code, bounds))
            if step == steps or step % max(1, steps // 5) == 0:
                history.append({"step": step, "nll": nll})
        final_nll, final_code = _eval_nll()
        if history[-1]["step"] != steps:
            history.append({"step": steps, "nll": final_nll})
        else:
            history[-1]["nll"] = final_nll
        if final_nll < best["nll"]:
            best = {"nll": final_nll, "code": final_code}
    finally:
        iv.clear_forced_bias()
    return {
        "nll": best["nll"],
        "code": best["code"],
        "init_nll": init_nll,
        "history": history,
        "constrained": bounds is not None,
    }


def inversion_inits(
    mu: np.ndarray,
    *,
    std: np.ndarray | None,
    bounds: np.ndarray,
    n_restarts: int,
    seed: int,
) -> list[np.ndarray]:
    if n_restarts < 1:
        raise ValueError("n_restarts must be positive")
    rng = np.random.default_rng(seed)
    inits = [np.asarray(mu, dtype=np.float32)]
    dim = int(mu.shape[0])
    for _restart in range(1, n_restarts):
        if std is not None:
            draw = mu + 0.25 * std * rng.standard_normal(dim).astype(np.float32)
        else:
            unit = rng.random(dim).astype(np.float32)
            draw = bounds[0] + unit * (bounds[1] - bounds[0])
        inits.append(np.clip(draw, bounds[0], bounds[1]).astype(np.float32))
    return inits


def invert_with_restarts(
    ckpt,
    batch: dict,
    *,
    mu: np.ndarray,
    std: np.ndarray | None,
    bounds: np.ndarray | None,
    n_restarts: int,
    steps: int,
    lr: float,
    seed: int,
) -> dict[str, Any]:
    """Best NLL over ``μ`` plus restarts. ``bounds=None`` is unconstrained."""
    box = bounds if bounds is not None else np.stack(
        [mu - 4.0, mu + 4.0], axis=0
    ).astype(np.float32)
    inits = inversion_inits(
        mu, std=std, bounds=box, n_restarts=n_restarts, seed=seed
    )
    best: dict[str, Any] | None = None
    runs = []
    for init in inits:
        result = invert_code(
            ckpt,
            batch,
            init=init,
            bounds=bounds,
            steps=steps,
            lr=lr,
        )
        runs.append({"init_nll": result["init_nll"], "nll": result["nll"]})
        if best is None or result["nll"] < best["nll"]:
            best = result
    assert best is not None
    best = dict(best)
    best["n_restarts"] = n_restarts
    best["restart_nlls"] = runs
    return best


def stack_train_mu(reft_model, n_words: int) -> np.ndarray:
    return stack_bias_vectors(reft_model, list(range(n_words)))


def code_in_bounds(code: np.ndarray, bounds: np.ndarray, *, atol: float = 1e-5) -> bool:
    point = np.asarray(code, dtype=np.float32)
    lo = np.asarray(bounds[0], dtype=np.float32)
    hi = np.asarray(bounds[1], dtype=np.float32)
    return bool(np.all(point >= lo - atol) and np.all(point <= hi + atol))


def shuffled_index(index: int, panel: Sequence[int], n_train: int) -> int:
    """Rotate to the next panel member, else another train index."""
    panel = [int(i) for i in panel]
    if index in panel and len(panel) > 1:
        return panel[(panel.index(index) + 1) % len(panel)]
    others = [i for i in panel if i != index]
    if others:
        return int(others[0])
    if n_train < 2:
        raise ValueError("need at least two train molecules to shuffle codes")
    return 0 if index != 0 else 1


def mean_finite(values: Sequence[float | None]) -> float | None:
    nums = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not nums:
        return None
    return float(sum(nums) / len(nums))


def rate(flags: Sequence[bool]) -> float | None:
    if not flags:
        return None
    return float(sum(bool(x) for x in flags) / len(flags))


def checkpoint_metadata(ckpt) -> dict[str, Any]:
    saved = getattr(ckpt, "saved_cfg", None) or {}
    prompt = str(getattr(ckpt, "prompt", "") or "")
    return {
        "task": saved.get("task"),
        "model_name": saved.get("model_name"),
        "layer": saved.get("layer"),
        "low_rank_dim": saved.get("low_rank_dim"),
        "position": saved.get("position", "l1"),
        "smiles_tags": bool(saved.get("smiles_tags")),
        "mist_smiles_tags": bool(saved.get("mist_smiles_tags")),
        "use_chat_template": bool(saved.get("use_chat_template")),
        "from_chat_template": bool(getattr(ckpt, "from_chat_template", False)),
        "intervention_token": saved.get("intervention_token"),
        "intervention_inject": saved.get("intervention_inject"),
        "n_words": len(getattr(ckpt, "words", []) or []),
        "prompt_chars": len(prompt),
        "prompt_head": prompt[:120],
        "prompt_tail": prompt[-80:] if prompt else "",
    }


def decode_at_word_id(
    ckpt,
    word_id: int,
    *,
    prompt: str,
    from_chat: bool,
    span,
    temperature: float,
    max_new_tokens: int,
) -> str:
    return decode_at_code(
        ckpt,
        int(word_id),
        prompt=prompt,
        from_chat=from_chat,
        span=span,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )


def interpret_inversion(
    *,
    recovered_in_bounds: bool,
    recovered_unconstrained: bool,
    nll_improved: bool,
    inverted_in_bounds: bool,
) -> str:
    """Advisor table: encoder vs bounds vs capacity vs search."""
    if recovered_in_bounds or (
        recovered_unconstrained and inverted_in_bounds
    ):
        return "encoder"
    if recovered_unconstrained:
        return "bounds"
    if nll_improved:
        return "capacity_or_decoding"
    return "no_improvement"
