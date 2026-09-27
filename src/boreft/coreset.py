"""Geometry-based coreset index selectors (herding, k-center greedy)."""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

import numpy as np

SAMPLE_STRATEGIES = ("all", "random", "herding", "k-center-greedy")
GEOMETRY_STRATEGIES = ("herding", "k-center-greedy")


def _as_feature_matrix(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {x.shape}")
    if x.size and not np.isfinite(x).all():
        raise ValueError("features must be finite")
    return x


def _l2_normalize_rows(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def herding_indices(features: np.ndarray, k: int) -> List[int]:
    """Greedy herding coreset indices into ``features``.

    L2-normalize rows, let ``mu`` be the dataset mean, and iteratively pick the
    unselected point maximizing ``<x, mu - mean(selected)>`` (matching-pursuit
    herding). Ties break toward the lowest index.
    """
    x = _l2_normalize_rows(_as_feature_matrix(features))
    n = int(x.shape[0])
    k = int(k)
    if k < 0:
        raise ValueError("k must be >= 0")
    if k == 0 or n == 0:
        return []
    if k >= n:
        return list(range(n))

    mu = x.mean(axis=0)
    selected: List[int] = []
    selected_mask = np.zeros(n, dtype=bool)
    selected_sum = np.zeros(x.shape[1], dtype=np.float64)

    for t in range(k):
        if t == 0:
            residual = mu
        else:
            residual = mu - (selected_sum / t)
        scores = x @ residual
        scores[selected_mask] = -np.inf
        # argmax with lowest-index tie-break (numpy argmax is first-max).
        idx = int(np.argmax(scores))
        selected.append(idx)
        selected_mask[idx] = True
        selected_sum = selected_sum + x[idx]
    return selected


def k_center_greedy_indices(features: np.ndarray, k: int) -> List[int]:
    """Gonzalez k-center greedy coreset indices into ``features``.

    Uses raw Euclidean distance in feature space (bias-mean geometry keeps
    magnitude). The first center is the point farthest from the dataset mean;
    each subsequent center maximizes min-distance to the current center set.
    Ties break toward the lowest index.
    """
    x = _as_feature_matrix(features)
    n = int(x.shape[0])
    k = int(k)
    if k < 0:
        raise ValueError("k must be >= 0")
    if k == 0 or n == 0:
        return []
    if k >= n:
        return list(range(n))

    mean = x.mean(axis=0)
    dist_to_mean = np.linalg.norm(x - mean, axis=1)
    first = int(np.argmax(dist_to_mean))
    selected: List[int] = [first]
    # min squared distance from each point to the closest selected center
    min_sq = np.sum((x - x[first]) ** 2, axis=1)
    selected_mask = np.zeros(n, dtype=bool)
    selected_mask[first] = True

    for _ in range(1, k):
        scores = min_sq.copy()
        scores[selected_mask] = -np.inf
        nxt = int(np.argmax(scores))
        selected.append(nxt)
        selected_mask[nxt] = True
        new_sq = np.sum((x - x[nxt]) ** 2, axis=1)
        min_sq = np.minimum(min_sq, new_sq)
    return selected


def resolve_sample_count(
    n_available: int,
    *,
    n: Optional[int] = None,
    prop: Optional[float] = None,
    n_name: str = "n",
    prop_name: str = "prop",
) -> Optional[int]:
    """Resolve an absolute count or a fraction to a sample size.

    Returns ``None`` when neither ``n`` nor ``prop`` is set (keep all).
    """
    if n is not None and prop is not None:
        raise ValueError(f"specify at most one of {n_name} / {prop_name}")
    if n is None and prop is None:
        return None
    if prop is not None:
        value = float(prop)
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{prop_name} must be in [0, 1]")
        return int(round(value * max(int(n_available), 0)))
    count = int(n)
    if count < 0:
        raise ValueError(f"{n_name} must be >= 0")
    return count


def select_subset_indices(
    n_items: int,
    *,
    n: Optional[int] = None,
    prop: Optional[float] = None,
    strategy: str = "all",
    features: Optional[np.ndarray] = None,
    seed: int = 0,
    allowed: Sequence[str] = SAMPLE_STRATEGIES,
    n_name: str = "n",
    prop_name: str = "prop",
    strategy_name: str = "strategy",
) -> List[int]:
    """Return sorted indices for a coreset of ``n_items`` rows.

    ``strategy='all'`` keeps every row (``n`` / ``prop`` must be unset).
    Any other allowed strategy keeps all rows when ``n`` and ``prop`` are
    both unset; otherwise it samples the resolved count.
    """
    if n_items < 0:
        raise ValueError("n_items must be >= 0")
    if n_items == 0:
        return []
    strategy_key = str(strategy or "all").strip().lower()
    allowed_keys = tuple(
        str(item).strip().lower() for item in (allowed or SAMPLE_STRATEGIES)
    )
    if strategy_key not in allowed_keys:
        raise ValueError(
            f"unknown {strategy_name} {strategy!r}; expected one of {tuple(allowed)}"
        )
    if strategy_key == "all":
        if n is not None or prop is not None:
            raise ValueError(
                f"{strategy_name}='all' cannot be combined with {n_name} / {prop_name}"
            )
        return list(range(n_items))

    count = resolve_sample_count(
        n_items, n=n, prop=prop, n_name=n_name, prop_name=prop_name
    )
    if count is None or count >= n_items:
        return list(range(n_items))
    if count == 0:
        return []

    if strategy_key == "random":
        idxs = random.Random(int(seed)).sample(range(n_items), count)
        return sorted(idxs)

    if strategy_key not in GEOMETRY_STRATEGIES:
        raise ValueError(
            f"unknown {strategy_name} {strategy!r}; expected one of {tuple(allowed)}"
        )
    if features is None:
        raise ValueError(f"{strategy_name}={strategy_key} requires bias-mean features")
    feat = np.asarray(features)
    if feat.ndim != 2 or feat.shape[0] != n_items:
        raise ValueError(
            f"features must have shape [n_items, D]=[{n_items}, D], got {feat.shape}"
        )
    if feat.size and not np.isfinite(feat).all():
        raise ValueError("features must be finite")
    if strategy_key == "herding":
        idxs = herding_indices(feat, count)
    else:
        idxs = k_center_greedy_indices(feat, count)
    return sorted(idxs)
