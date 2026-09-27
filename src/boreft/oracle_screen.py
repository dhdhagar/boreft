"""Molopt oracle screen: train vs high-tail test vs Sobol-decoded B.

Library scoring is CPU-only. Sobol decoding needs a checkpoint. See
``notes/molopt-experiments-plan.md``.
"""

from __future__ import annotations

from dataclasses import replace
import json
import os
import random
from typing import Mapping, Sequence

import numpy as np

from boreft.chem import (
    canonical_target_key,
    is_valid_smiles,
    qed_value,
    repair_smiles,
    unwrap_smiles_tags,
)
from boreft.data.molopt import read_csv_smiles
from boreft.oracles import (
    INVALID_SCORE,
    SCORE_THRESHOLDS,
    TDC_ORACLE_NAMES,
    MoleculeScore,
    OracleFn,
    oracle_score_lists,
    score_molecules,
    summarize_scores,
)
from boreft.task_config import task_instruction, task_system_prompt

SPLIT_ORDER: tuple[str, ...] = ("train", "test", "held_out", "sobol")
SPLIT_LABELS = {
    "train": "train targets",
    "test": "high-tail test (p90–100)",
    "held_out": "held-out ChEBI",
    "sobol": "Sobol-decoded B",
}


def sobol_generation_kwargs(
    temperature: float, top_p: float = 1.0
) -> dict[str, float | bool]:
    """``generate_texts_multi_batch`` kwargs. ``temperature=0`` is greedy."""
    temp = float(temperature)
    p = float(top_p)
    if temp < 0.0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if not (0.0 < p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if temp == 0.0:
        return {"use_sample": False}
    return {"use_sample": True, "temperature": temp, "top_p": p}


def sobol_decode_tag(*, temperature: float, top_p: float = 1.0) -> str:
    """Filesystem / W&B suffix: empty for greedy, else ``temp1`` / ``temp1_topp0.9``."""
    kwargs = sobol_generation_kwargs(temperature, top_p)
    if not kwargs.get("use_sample"):
        return ""
    tag = f"temp{float(kwargs['temperature']):g}"
    p = float(kwargs["top_p"])
    if p != 1.0:
        tag = f"{tag}_topp{p:g}"
    return tag


def overlay_generation_prompt_cfg(
    saved_cfg: Mapping,
    *,
    generation_prompt: str | None = None,
    generation_prompt_from_task_config: bool = False,
    system_prompt_from_task_config: bool = False,
) -> dict:
    """Copy a checkpoint config with a decode-time user/system prompt overlay.

    ``generation_prompt_from_task_config`` replaces the checkpoint's saved
    ``chat_instruction`` with the current ``task_config`` chat user message.
    ``system_prompt_from_task_config`` replaces ``system_prompt`` the same way
    (or clears it if the task has none). A custom ``generation_prompt``
    replaces only the user instruction and is mutually exclusive with
    ``generation_prompt_from_task_config``. Chat wrapping still follows the
    checkpoint's ``use_chat_template``.
    """
    custom = (generation_prompt or "").strip() or None
    if generation_prompt_from_task_config and custom:
        raise ValueError(
            "pass only one of generation_prompt and "
            "generation_prompt_from_task_config"
        )
    if (
        not generation_prompt_from_task_config
        and custom is None
        and not system_prompt_from_task_config
    ):
        return dict(saved_cfg)
    cfg = dict(saved_cfg)
    task = str(cfg.get("task") or "molopt")
    if generation_prompt_from_task_config:
        cfg["chat_instruction"] = task_instruction(
            task, use_chat_template=True
        )
    elif custom is not None:
        cfg["chat_instruction"] = custom
    if system_prompt_from_task_config:
        cfg["system_prompt"] = task_system_prompt(task)
    return cfg


def load_library_splits(
    csv_path: str,
    *,
    train_smiles: Sequence[str] | None = None,
    train_top_k: int = 1024,
    held_out_n: int | None = None,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Train prefix (or explicit vocabulary) and the non-train CSV pool."""
    if train_smiles is not None:
        train = [str(s).strip() for s in train_smiles if str(s).strip()]
    else:
        if train_top_k < 1:
            raise ValueError("train_top_k must be positive")
        train = read_csv_smiles(csv_path, top_k=train_top_k)
    train_keys = {canonical_target_key(s) for s in train}
    held: list[str] = []
    seen = set(train_keys)
    for smiles in read_csv_smiles(csv_path):
        key = canonical_target_key(smiles)
        if key in seen:
            continue
        seen.add(key)
        held.append(smiles)
    if held_out_n is not None:
        if held_out_n < 0:
            raise ValueError("held_out_n must be nonnegative")
        if held_out_n < len(held):
            held = random.Random(seed).sample(sorted(held), held_out_n)
    return {"train": train, "held_out": held}


def smiles_from_items_json(path: str) -> list[str]:
    """SMILES targets from a checkpoint ``items.json`` (a JSON list of rows)."""
    with open(path, encoding="utf-8") as handle:
        items = json.load(handle)
    if not isinstance(items, list):
        raise ValueError(f"{path}: expected a JSON list")
    smiles: list[str] = []
    for row in items:
        if not isinstance(row, dict):
            continue
        value = row.get("word") or row.get("target")
        text = unwrap_smiles_tags(str(value or "").strip())
        if text:
            smiles.append(text)
    if not smiles:
        raise ValueError(f"{path}: no SMILES targets")
    return smiles


_SOBOL_SECTION_KEYS = frozenset(
    {"greedy", "temperature", "temperature_1.0", "temperature_1.5"}
)


def _load_sobol_results_payload(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    if _SOBOL_SECTION_KEYS.intersection(data):
        if "temperature" in data and "temperature_1.0" not in data:
            data = dict(data)
            data["temperature_1.0"] = data["temperature"]
        return data
    return {"greedy": data}


def smiles_from_sobol_results(
    path: str, *, section: str = "greedy"
) -> list[str]:
    """Greedy (or named) decode strings from a GENZ dump or a JSON SMILES list."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return [unwrap_smiles_tags(str(s)) for s in data]
    payload = _load_sobol_results_payload(path)
    block = payload.get(section) if section in payload else payload
    if not isinstance(block, dict):
        return []
    texts: list[str] = []
    for row in block.get("per_sample") or []:
        if not isinstance(row, dict):
            continue
        samples = row.get("samples") or []
        if samples:
            texts.append(unwrap_smiles_tags(str(samples[0])))
            continue
        mode = row.get("sample_mode")
        if mode:
            texts.append(unwrap_smiles_tags(str(mode)))
    return texts


def decode_sobol_smiles(
    *,
    reft_model,
    tokenizer,
    words: Sequence[str],
    prompt: str,
    n_sobol: int,
    seed: int,
    batch_size: int,
    max_new_tokens: int,
    position: str,
    assistant_suffix: str | None,
    from_chat_template: bool,
    intervention_token_id: int | None,
    content_span: tuple[int, int] | None,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> list[str]:
    """Greedy (or sampled) Sobol decodes from an already-loaded checkpoint."""
    from boreft.eval.semantle import (
        generate_texts_multi_batch,
        sample_sobol_bias_vectors,
    )

    if n_sobol < 1:
        raise ValueError("n_sobol must be positive")
    if batch_size < 1:
        raise ValueError("eval_batch_size must be positive")
    gen_kwargs = sobol_generation_kwargs(temperature, top_p)
    sampled_bs, _nearest, _idx, _mu = sample_sobol_bias_vectors(
        reft_model,
        list(words),
        n_sobol,
        seed=seed,
    )
    texts: list[str] = []
    for start in range(0, len(sampled_bs), batch_size):
        batch = [
            np.asarray(row, dtype=np.float32)
            for row in sampled_bs[start : start + batch_size]
        ]
        decoded = generate_texts_multi_batch(
            reft_model,
            tokenizer,
            prompt,
            batch,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
            position=position,
            assistant_suffix=assistant_suffix,
            from_chat_template=from_chat_template,
            intervention_token_id=intervention_token_id,
            content_span=content_span,
        )
        texts.extend(unwrap_smiles_tags(text) for text in decoded)
        done = min(start + batch_size, len(sampled_bs))
        print(f"[oracle_screen] sobol decode {done}/{len(sampled_bs)}", flush=True)
    return texts


def screen_metrics_for_wandb(splits: Mapping[str, dict]) -> dict[str, float]:
    """Flatten split × oracle scalars onto ``oracle_screen/...`` W&B keys."""
    flat: dict[str, float] = {}
    for split, block in splits.items():
        if not isinstance(block, Mapping):
            continue
        for key in ("n", "n_valid", "validity"):
            value = block.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                flat[f"oracle_screen/{split}/{key}"] = float(value)
        oracles = block.get("oracles") or {}
        if not isinstance(oracles, Mapping):
            continue
        for oracle, stats in oracles.items():
            if not isinstance(stats, Mapping):
                continue
            for key, value in stats.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    flat[f"oracle_screen/{split}/{oracle}/{key}"] = float(value)
    return flat


def write_decoded_smiles(path: str, texts: Sequence[str]) -> None:
    """Write decoded (unrepaired) SMILES as a JSON list for later ``--sobol-results``."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([str(s) for s in texts], handle)


def require_smiself() -> None:
    """Raise if SmiSelf is not installed (needed for ``--repair-smiles``)."""
    try:
        import smiself  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "SmiSelf is required for --repair-smiles. "
            "Install with: bash scripts/install_smiself.sh"
        ) from e


def repair_invalid_smiles(texts: Sequence[str]) -> tuple[list[str], dict[str, int]]:
    """Replace invalid strings with SmiSelf repairs when those parse.

    Already-valid SMILES are left unchanged. Failed repairs keep the original
    string (still invalid, still scored as 0). Requires SmiSelf.
    """
    require_smiself()
    out: list[str] = []
    n_invalid = 0
    n_repaired = 0
    for text in texts:
        if is_valid_smiles(text):
            out.append(str(text))
            continue
        n_invalid += 1
        repaired = repair_smiles(text)
        if repaired is not None:
            out.append(repaired)
            n_repaired += 1
        else:
            out.append(str(text))
    return out, {
        "n": len(texts),
        "n_invalid": n_invalid,
        "n_repaired": n_repaired,
        "n_still_invalid": n_invalid - n_repaired,
    }


def score_split_with_cache(
    smiles: Sequence[str],
    cache: Mapping[str, Mapping[str, float]],
    oracle_fns: Mapping[str, OracleFn] | None,
    *,
    oracles: Sequence[str] = TDC_ORACLE_NAMES,
    include_qed: bool = True,
) -> dict[str, list[MoleculeScore]]:
    """Score a split from the catalog cache, live-scoring only cache misses."""
    names = tuple(oracles) or TDC_ORACLE_NAMES
    texts = [str(s) for s in smiles]
    keys = [canonical_target_key(unwrap_smiles_tags(text) or text) for text in texts]
    missing_idx = [
        i
        for i, key in enumerate(keys)
        if not all(name in (cache.get(key) or {}) for name in names)
    ]
    live: dict[str, list[MoleculeScore]] = {name: [] for name in names}
    if missing_idx:
        missing_texts = [texts[i] for i in missing_idx]
        if oracle_fns:
            live = score_split(
                missing_texts,
                {name: oracle_fns[name] for name in names if name in oracle_fns},
                include_qed=include_qed,
            )
        else:
            qeds = [qed_value(s) for s in missing_texts] if include_qed else None
            for name in names:
                live[name] = [
                    MoleculeScore(
                        smiles=text,
                        canonical=canonical_target_key(text)
                        if is_valid_smiles(text)
                        else None,
                        valid=is_valid_smiles(text),
                        score=INVALID_SCORE,
                        qed=None if qeds is None else qeds[j],
                    )
                    for j, text in enumerate(missing_texts)
                ]
    live_by_index = {idx: j for j, idx in enumerate(missing_idx)}
    qeds = [qed_value(s) for s in texts] if include_qed else None
    out: dict[str, list[MoleculeScore]] = {}
    for name in names:
        rows: list[MoleculeScore] = []
        for i, text in enumerate(texts):
            if i in live_by_index:
                rows.append(live[name][live_by_index[i]])
                continue
            block = cache.get(keys[i]) or {}
            valid = is_valid_smiles(text)
            rows.append(
                MoleculeScore(
                    smiles=text,
                    canonical=keys[i] if valid else None,
                    valid=valid,
                    score=float(block.get(name, INVALID_SCORE)),
                    qed=None if qeds is None else qeds[i],
                )
            )
        out[name] = rows
    return out


def score_split(
    smiles: Sequence[str],
    oracles: Mapping[str, OracleFn],
    *,
    include_qed: bool = True,
) -> dict[str, list[MoleculeScore]]:
    qeds = [qed_value(s) for s in smiles] if include_qed else None
    out: dict[str, list[MoleculeScore]] = {}
    for name, oracle in oracles.items():
        rows = score_molecules(smiles, oracle, include_qed=False)
        if qeds is not None:
            rows = [replace(row, qed=qeds[i]) for i, row in enumerate(rows)]
        out[name] = rows
    return out


def summarize_split(
    scored: Mapping[str, Sequence[MoleculeScore]],
    *,
    thresholds: Sequence[float] = SCORE_THRESHOLDS,
) -> dict:
    if not scored:
        return {"n": 0, "n_valid": 0, "validity": 0.0, "oracles": {}}
    first = next(iter(scored.values()))
    n = len(first)
    n_valid = sum(1 for row in first if row.valid)
    return {
        "n": n,
        "n_valid": n_valid,
        "validity": (n_valid / n) if n else 0.0,
        "oracles": {
            name: summarize_scores(rows, thresholds=thresholds)
            for name, rows in scored.items()
        },
    }


def score_payload(
    scored: Mapping[str, Sequence[MoleculeScore]],
) -> dict[str, dict[str, list[float]]]:
    return {name: oracle_score_lists(rows) for name, rows in scored.items()}


def plot_oracle_histograms(
    scores: Mapping[str, Mapping[str, Mapping[str, list[float]]]],
    *,
    oracle_names: Sequence[str],
    out_path: str,
    split_order: Sequence[str] = SPLIT_ORDER,
    bins: int = 40,
    valid_only: bool = True,
) -> None:
    """One panel per oracle: overlaid histograms of train / high-tail / Sobol."""
    import matplotlib.pyplot as plt

    from boreft.bo.plotting import apply_bo_axes_style, bo_rc_params, save_plot

    names = [n for n in oracle_names if any(n in (scores.get(s) or {}) for s in split_order)]
    if not names:
        return
    present_splits = [
        split
        for split in split_order
        if split in scores and any(name in scores[split] for name in names)
    ]
    colors = {
        "train": "#2c7bb6",
        "test": "#31a354",
        "held_out": "#fdae61",
        "sobol": "#d7191c",
    }
    key = "valid" if valid_only else "all"
    with plt.rc_context(bo_rc_params()):
        fig, axes = plt.subplots(
            1,
            len(names),
            figsize=(4.2 * len(names), 3.6),
            sharey=True,
            squeeze=False,
        )
        for ax, oracle in zip(axes[0], names):
            for split in present_splits:
                values = (scores.get(split) or {}).get(oracle, {}).get(key) or []
                if not values:
                    continue
                ax.hist(
                    values,
                    bins=np.linspace(0.0, 1.0, bins + 1),
                    density=True,
                    histtype="step",
                    linewidth=1.8,
                    color=colors.get(split, "0.3"),
                    label=SPLIT_LABELS.get(split, split),
                )
            apply_bo_axes_style(
                ax,
                xlabel=oracle,
                ylabel="density" if ax is axes[0, 0] else None,
            )
            ax.set_xlim(0.0, 1.0)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=len(present_splits),
                frameon=False,
                fontsize=11,
                bbox_to_anchor=(0.5, 1.08),
            )
        fig.tight_layout()
        save_plot(fig, out_path)
        plt.close(fig)


def write_screen_report(
    path: str,
    *,
    meta: dict,
    splits: Mapping[str, dict],
    scores: Mapping[str, Mapping[str, Mapping[str, list[float]]]],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {**meta, "splits": dict(splits), "scores": dict(scores)}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_three_way_screen(
    split_smiles: Mapping[str, Sequence[str]],
    oracles: Mapping[str, OracleFn] | None,
    *,
    oracle_names: Sequence[str] | None = None,
    cache: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Score named SMILES splits. Returns ``(summaries, score_lists)``."""
    names = list(oracle_names) if oracle_names is not None else list(oracles or ())
    if not names:
        names = list(TDC_ORACLE_NAMES)
    summaries: dict[str, dict] = {}
    scores: dict[str, dict] = {}
    cache = cache or {}
    for split, texts in split_smiles.items():
        if not texts:
            continue
        if cache:
            scored = score_split_with_cache(
                texts, cache, oracles, oracles=names
            )
        elif oracles:
            scored = score_split(texts, {n: oracles[n] for n in names if n in oracles})
        else:
            continue
        summaries[split] = summarize_split(scored)
        scores[split] = score_payload(scored)
    return summaries, scores


def format_screen_table(splits: Mapping[str, dict], oracle_names: Sequence[str]) -> str:
    """Plain-text go/no-go summary: max and validity per split × oracle."""
    lines = []
    header = f"{'split':<12} {'n':>6} {'valid':>8} " + " ".join(
        f"{name + ' max_v':>14}" for name in oracle_names
    )
    lines.append(header)
    for split in SPLIT_ORDER:
        block = splits.get(split)
        if not block:
            continue
        cells = [
            f"{split:<12}",
            f"{int(block.get('n') or 0):>6d}",
            f"{float(block.get('validity') or 0.0):>8.1%}",
        ]
        oracles = block.get("oracles") or {}
        for name in oracle_names:
            stats = oracles.get(name) or {}
            value = stats.get("max_valid")
            cells.append(f"{'—':>14}" if value is None else f"{float(value):>14.4f}")
        lines.append(" ".join(cells))
    return "\n".join(lines)
