"""MolOpt train/test split by TDC oracle percentiles.

Train draws from the pool at or below the per-oracle percentile (default p90).
The held-out test pool is the complement: any molecule above p90 on DRD2,
GSK3B, or JNK3. Scores are cached next to the CSV so training does not
re-download TDC pickles on every run.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from typing import Mapping, Optional, Sequence

import numpy as np

from boreft.chem import canonical_target_key, unwrap_smiles_tags
from boreft.data.base import ReftItem, item_target
from boreft.oracles import TDC_ORACLE_NAMES, load_oracles, score_molecules

ORACLE_SPLIT_FILENAME = "oracle_split.json"
DEFAULT_ORACLE_CAP_PERCENTILE = 90.0
SCORES_FILENAME = "oracle_scores.json"


def default_oracle_scores_path(csv_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(csv_path)), SCORES_FILENAME)


def oracle_split_path(output_dir: str) -> str:
    return os.path.join(output_dir, ORACLE_SPLIT_FILENAME)


def load_oracle_score_cache(path: str) -> dict[str, dict[str, float]]:
    """Canonical-SMILES key → ``{oracle: score, ...}``."""
    if not path or not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("scores") if isinstance(payload, dict) else None
    if not isinstance(rows, dict):
        return {}
    out: dict[str, dict[str, float]] = {}
    for key, block in rows.items():
        if not isinstance(block, Mapping):
            continue
        out[str(key)] = {
            name: float(block[name])
            for name in TDC_ORACLE_NAMES
            if name in block
        }
    return out


def save_oracle_score_cache(
    path: str,
    scores: Mapping[str, Mapping[str, float]],
    *,
    oracles: Sequence[str] = TDC_ORACLE_NAMES,
) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {
        "oracles": list(oracles),
        "scores": {
            key: {name: float(block[name]) for name in oracles if name in block}
            for key, block in scores.items()
        },
    }
    fd, tmp_path = tempfile.mkstemp(dir=directory or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def ensure_oracle_scores(
    smiles: Sequence[str],
    *,
    cache_path: str,
    oracles: Sequence[str] = TDC_ORACLE_NAMES,
) -> dict[str, dict[str, float]]:
    """Return per-canonical scores, scoring and caching any missing rows."""
    names = tuple(oracles) or TDC_ORACLE_NAMES
    cache = load_oracle_score_cache(cache_path)
    missing: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in smiles:
        text = unwrap_smiles_tags(str(raw).strip())
        if not text:
            continue
        key = canonical_target_key(text)
        if key in seen:
            continue
        seen.add(key)
        block = cache.get(key) or {}
        if all(name in block for name in names):
            continue
        missing.append((key, text))
    if missing:
        print(
            f"[molopt] scoring {len(missing)} molecules on {', '.join(names)} "
            f"(cache={cache_path})",
            flush=True,
        )
        fns = load_oracles(names)
        texts = [text for _key, text in missing]
        for name, oracle in fns.items():
            print(f"[molopt]   {name} ...", flush=True)
            rows = score_molecules(texts, oracle, include_qed=False)
            for (key, _text), row in zip(missing, rows):
                cache.setdefault(key, {})[name] = float(row.score)
        save_oracle_score_cache(cache_path, cache, oracles=names)
        print(f"[molopt] wrote {len(cache)} scored molecules → {cache_path}", flush=True)
    return cache


def oracle_percentile_thresholds(
    scores: Mapping[str, Mapping[str, float]],
    percentile: float,
    *,
    oracles: Sequence[str] = TDC_ORACLE_NAMES,
) -> dict[str, float]:
    if not 0.0 < percentile <= 100.0:
        raise ValueError(f"percentile must be in (0, 100], got {percentile}")
    thresholds: dict[str, float] = {}
    for name in oracles:
        values = [float(block[name]) for block in scores.values() if name in block]
        if not values:
            raise ValueError(f"no {name} scores to compute a percentile")
        thresholds[name] = float(np.percentile(values, percentile))
    return thresholds


def split_smiles_by_oracle_percentiles(
    smiles: Sequence[str],
    scores: Mapping[str, Mapping[str, float]],
    *,
    percentile: float,
    oracles: Sequence[str] = TDC_ORACLE_NAMES,
) -> tuple[list[int], list[int], dict]:
    """Index split of ``smiles`` into train-eligible (≤ p) vs high-tail (> p).

    A row is train-eligible when **every** oracle is at or below its percentile.
    It is high-tail when **any** oracle exceeds its percentile. Rows with no
    score key still go to train (treated as 0).
    """
    names = tuple(oracles) or TDC_ORACLE_NAMES
    keys = [canonical_target_key(unwrap_smiles_tags(str(s).strip()) or str(s)) for s in smiles]
    keyed = {key: scores[key] for key in keys if key in scores}
    thresholds = oracle_percentile_thresholds(keyed or scores, percentile, oracles=names)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for i, key in enumerate(keys):
        block = scores.get(key) or {}
        high = any(float(block.get(name, 0.0)) > thresholds[name] for name in names)
        (test_idx if high else train_idx).append(i)
    meta = {
        "percentile": float(percentile),
        "oracles": list(names),
        "thresholds": thresholds,
        "n_pool": len(smiles),
        "n_train_eligible": len(train_idx),
        "n_test_eligible": len(test_idx),
    }
    return train_idx, test_idx, meta


def sample_indices(indices: Sequence[int], n: Optional[int], seed: int) -> list[int]:
    chosen = list(indices)
    if n is None or n >= len(chosen):
        return chosen
    if n < 1:
        raise ValueError(f"sample size must be positive, got {n}")
    return random.Random(seed).sample(chosen, n)


def take_items_at(
    items: Sequence[ReftItem], indices: Sequence[int]
) -> tuple[list[ReftItem], list[int]]:
    sampled: list[ReftItem] = []
    original: list[int] = []
    for new_id, orig_id in enumerate(indices):
        item = items[orig_id]
        item.id = new_id
        sampled.append(item)
        original.append(int(orig_id))
    return sampled, original


def apply_molopt_oracle_cap(
    items: Sequence[ReftItem],
    *,
    percentile: float,
    n_train: Optional[int],
    seed: int,
    cache_path: str,
    oracles: Sequence[str] = TDC_ORACLE_NAMES,
    strict_n_train: bool = True,
) -> tuple[list[ReftItem], Optional[list[int]], dict]:
    """Filter molopt items to the p-capped pool, then subsample ``n_train``."""
    smiles = [unwrap_smiles_tags(item_target(item)) for item in items]
    scores = ensure_oracle_scores(smiles, cache_path=cache_path, oracles=oracles)
    train_idx, test_idx, meta = split_smiles_by_oracle_percentiles(
        smiles, scores, percentile=percentile, oracles=oracles
    )
    if n_train is not None and n_train > len(train_idx):
        if strict_n_train:
            raise ValueError(
                f"--train-n-samples {n_train} exceeds the p{percentile:g}-capped "
                f"pool ({len(train_idx)} molecules). Lower N or raise "
                f"--molopt-oracle-cap-percentile."
            )
        n_train = len(train_idx)
    drawn = sample_indices(train_idx, n_train, seed)
    sampled, original = take_items_at(items, drawn)
    test_smiles = [smiles[i] for i in test_idx]
    meta.update(
        {
            "n_train": len(sampled),
            "train_pool_indices": list(original),
            "test_smiles": test_smiles,
            "scores_path": os.path.abspath(cache_path),
        }
    )
    embed_indices: Optional[list[int]] = original if len(original) < len(items) else None
    return sampled, embed_indices, meta


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(val) for val in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def oracle_split_for_config(meta: Mapping | None) -> dict | None:
    """Drop bulky row indices from the copy stored in training_config.json."""
    if not meta:
        return None
    return {key: value for key, value in meta.items() if key != "train_pool_indices"}


def write_oracle_split(path: str, meta: Mapping) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(dict(meta)), handle, indent=2)


def load_oracle_split(output_dir: str | None, saved_cfg: Mapping | None = None) -> dict | None:
    """Checkpoint ``oracle_split.json``, else the copy embedded in training_config."""
    if output_dir:
        path = oracle_split_path(output_dir)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict) and payload.get("test_smiles"):
                return payload
    if saved_cfg:
        embedded = saved_cfg.get("oracle_split")
        if isinstance(embedded, dict) and embedded.get("test_smiles"):
            return embedded
    return None
