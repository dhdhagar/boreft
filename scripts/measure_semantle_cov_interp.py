#!/usr/bin/env python3
"""Coverage and interpolation errors on Semantle checkpoints.

Implements the protocol in ``notes/measure_cov_interp_notes.md``:

  ε_int  = max_α  E[ ||φ(S) - x_α||_2 ]     (needs decoding)
  ε_cov  = min_α  ||φ(s) - x_α||_2          (embeddings only)

Coverage is a property of the training-target embeddings and the weight set A,
so it can be compared across training-set sizes without loading the LLM.
Interpolation compares BOReFT to its training ablations and needs generation.

    python scripts/measure_semantle_cov_interp.py --dry-run
    python scripts/measure_semantle_cov_interp.py \\
        --condition canonical=outputs/1784053292 \\
        --condition sdpo0=outputs/1788622157
    python scripts/measure_semantle_cov_interp.py \\
        --coverage-only --from-ladder experiments/semantle/ladder_checkpoints.json
    python scripts/measure_semantle_cov_interp.py --make-table --out-dir data/semantle/analysis/cov_interp
    sbatch scripts/measure_semantle_cov_interp.sh
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
from typing import Any, Optional, Sequence

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

TASK = "semantle"
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "data", "semantle", "analysis", "cov_interp")
DEFAULT_CACHE_DIR = None
DEFAULT_LADDER = os.path.join(
    REPO_ROOT, "experiments", "semantle", "ladder_checkpoints.json"
)
DEFAULT_TARGETS_JSON = os.path.join(
    REPO_ROOT, "experiments", "outputs", "semantle", "sweep", "targets.json"
)
CANONICAL_DIR = os.path.join(REPO_ROOT, "outputs", "1784053292")
EPS = 1e-12
HULL_ITERS = 400
HULL_INTERIOR = 1e-3
PAIR_GRID = tuple(round(0.1 * k, 1) for k in range(1, 10))
MIDPATH_T_LO = 0.4
MIDPATH_T_HI = 0.6
NESTED_FAMILIES = ("vertices", "pairs", "mix3", "mix5")

DEFAULT_CONDITIONS: tuple[dict[str, str], ...] = (
    {
        "name": "canonical",
        "wandb_id": "jn0knp44",
        "output_dir": os.path.join(REPO_ROOT, "outputs", "1784053292"),
    },
    {
        "name": "sdpo0",
        "wandb_id": "yt8pnmyh",
        "output_dir": os.path.join(REPO_ROOT, "outputs", "1788622157"),
    },
    {
        "name": "novae",
        "wandb_id": "vz6h0vba",
        "output_dir": os.path.join(REPO_ROOT, "outputs", "1789101817"),
    },
    {
        "name": "joint",
        "wandb_id": "gsuy3cnv",
        "output_dir": os.path.join(REPO_ROOT, "outputs", "1789021668"),
    },
)
CONDITION_LABELS = {
    "canonical": "BOReFT",
    "sdpo0": "w/o self-distillation",
    "novae": "w/o variational training",
    "joint": "w/o both",
    "sdpo0_novae": "w/o both",
    "ce0": "w/o reconstruction",
    "noenc": "w/o shared encoder",
}


def condition_label(name: str) -> str:
    key = str(name or "").strip().lower()
    if key in CONDITION_LABELS:
        return CONDITION_LABELS[key]
    if key.startswith("n") and key[1:].isdigit():
        return f"$n={int(key[1:])}$"
    return str(name)


def item_word(item: dict[str, Any]) -> str:
    return str(item.get("word") or item["target"]).strip()


def item_word_id(item: dict[str, Any]) -> int:
    if "id" not in item or item["id"] is None:
        raise ValueError(f"items.json row is missing id: {item!r}")
    return int(item["id"])


def remap_repo_path(path: str) -> str:
    if os.path.isfile(path):
        return path
    parts = path.replace("\\", "/").split("/")
    if "data" not in parts:
        return path
    candidate = os.path.join(REPO_ROOT, *parts[parts.index("data") :])
    return candidate if os.path.isfile(candidate) else path


def load_items(output_dir: str) -> tuple[list[str], list[int], list[dict[str, Any]]]:
    path = os.path.join(output_dir, "items.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no items.json in {output_dir}")
    with open(path, encoding="utf-8") as handle:
        items = json.load(handle)
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path}: expected a nonempty JSON list")
    words = [item_word(row) for row in items]
    ids = [item_word_id(row) for row in items]
    return words, ids, items


def canonical_words(words: Sequence[str]) -> list[str]:
    return sorted({str(w).strip().lower() for w in words if str(w).strip()})


def word_set_key(words: Sequence[str]) -> str:
    payload = "\n".join(canonical_words(words))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def local_index_map(words: Sequence[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for i, word in enumerate(words):
        key = str(word).strip().lower()
        if key and key not in mapping:
            mapping[key] = i
    return mapping


def remap_weight_supports(
    weights: Sequence[dict[str, Any]],
    canon: Sequence[str],
    local_words: Sequence[str],
) -> list[dict[str, Any]]:
    loc = local_index_map(local_words)
    remapped: list[dict[str, Any]] = []
    for weight in weights:
        support = []
        for idx in weight["support"]:
            name = canon[int(idx)]
            if name not in loc:
                raise KeyError(f"canonical word {name!r} missing from local items")
            support.append(loc[name])
        row = dict(weight)
        row["support"] = support
        remapped.append(row)
    return remapped


def l2_normalize(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        n = float(np.linalg.norm(arr))
        return arr / max(n, EPS)
    n = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(n, EPS)


def interpolation_floor(x_alpha: np.ndarray) -> float:
    return float(max(0.0, 1.0 - np.linalg.norm(np.asarray(x_alpha, dtype=np.float64))))


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto ``{a >= 0, sum a = 1}`` (Duchi et al.)."""
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = int(v.size)
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1.0))[0]
    if rho.size == 0:
        out = np.zeros(n, dtype=np.float64)
        out[int(np.argmax(v))] = 1.0
        return out
    theta = (cssv[int(rho[-1])] - 1.0) / float(rho[-1] + 1)
    w = np.maximum(v - theta, 0.0)
    s = float(w.sum())
    if s <= EPS:
        out = np.zeros(n, dtype=np.float64)
        out[int(np.argmax(v))] = 1.0
        return out
    return w / s


def gram_lipschitz(points: np.ndarray) -> float:
    """Squared operator norm of ``points``, i.e. Lipschitz constant of the hull gradient."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return 1.0
    return float(max(np.linalg.norm(pts, ord=2) ** 2, EPS))


def project_onto_hull(
    x: np.ndarray,
    points: np.ndarray,
    *,
    n_iter: int = HULL_ITERS,
    lipschitz: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    """Project ``x`` onto conv(rows of ``points``). Returns ``(alpha, distance)``."""
    pts = np.asarray(points, dtype=np.float64)
    target = np.asarray(x, dtype=np.float64).reshape(-1)
    if pts.ndim != 2 or pts.shape[0] == 0:
        raise ValueError("points must be a nonempty [n, d] array")
    if pts.shape[1] != target.size:
        raise ValueError("x and points have different dimensions")
    n = pts.shape[0]
    lr = 1.0 / max(
        float(lipschitz) if lipschitz is not None else gram_lipschitz(pts),
        EPS,
    )
    nearest = int(np.argmin(np.linalg.norm(pts - target, axis=1)))
    alpha = np.zeros(n, dtype=np.float64)
    alpha[nearest] = 1.0
    for _ in range(int(n_iter)):
        residual = pts.T @ alpha - target
        grad = pts @ residual
        alpha = project_simplex(alpha - lr * grad)
    hat = pts.T @ alpha
    return alpha, float(np.linalg.norm(target - hat))


def split_half_displacement(emb: np.ndarray, x_alpha: np.ndarray) -> Optional[float]:
    """Unbiased split-half estimator of ``||m_τ(b_α) - x_α||_2``."""
    u = np.asarray(emb, dtype=np.float64)
    x = np.asarray(x_alpha, dtype=np.float64).reshape(-1)
    if u.ndim != 2 or u.shape[0] < 2:
        return None
    mid = u.shape[0] // 2
    a = u[:mid].mean(axis=0) - x
    b = u[mid:].mean(axis=0) - x
    return float(np.sqrt(max(0.0, float(np.dot(a, b)))))


def plugin_displacement(emb: np.ndarray, x_alpha: np.ndarray) -> float:
    u = np.asarray(emb, dtype=np.float64)
    x = np.asarray(x_alpha, dtype=np.float64).reshape(-1)
    mean = u.mean(axis=0) if u.ndim == 2 else u
    return float(np.linalg.norm(mean - x))


def direction_only_distance(emb: np.ndarray, x_alpha: np.ndarray) -> float:
    """Mean Euclidean distance from unit embeddings to the renormalized interpolant."""
    u = np.asarray(emb, dtype=np.float64)
    if u.ndim == 1:
        u = u.reshape(1, -1)
    direction = l2_normalize(np.asarray(x_alpha, dtype=np.float64).reshape(-1))
    unit = l2_normalize(u)
    return float(np.mean(np.linalg.norm(unit - direction, axis=1)))


def expected_l2(emb: np.ndarray, x_alpha: np.ndarray) -> tuple[float, float]:
    u = np.asarray(emb, dtype=np.float64)
    if u.ndim == 1:
        u = u.reshape(1, -1)
    x = np.asarray(x_alpha, dtype=np.float64).reshape(-1)
    d = np.linalg.norm(u - x, axis=1)
    mean = float(d.mean())
    if d.size <= 1:
        return mean, 0.0
    return mean, float(d.std(ddof=1) / np.sqrt(d.size))


def coverage_min(phi_s: np.ndarray, interpolants: np.ndarray) -> tuple[float, int]:
    pts = np.asarray(interpolants, dtype=np.float64)
    x = np.asarray(phi_s, dtype=np.float64).reshape(-1)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return float("nan"), -1
    d = np.linalg.norm(pts - x, axis=1)
    idx = int(np.argmin(d))
    return float(d[idx]), idx


def summarize_values(
    values: Sequence[float],
    rng: np.random.Generator,
    *,
    n_boot: int = 1000,
) -> dict[str, Any]:
    v = np.asarray(list(values), dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "q90": None,
            "max": None,
            "max_ci95": [None, None],
        }
    mx = float(v.max())
    boots = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        samp = rng.choice(v, size=v.size, replace=True)
        boots[i] = float(samp.max())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "q90": float(np.quantile(v, 0.9)),
        "max": mx,
        "max_ci95": [float(lo), float(hi)],
    }


def _dirichlet(rng: np.random.Generator, k: int) -> np.ndarray:
    w = rng.gamma(1.0, 1.0, size=int(k))
    return w / max(float(w.sum()), EPS)


def nearest_indices(phi: np.ndarray, index: int, k: int) -> list[int]:
    """``k`` nearest neighbors of ``index`` (excluding itself), by cosine."""
    row = np.asarray(phi[index], dtype=np.float64)
    cos = np.asarray(phi, dtype=np.float64) @ row
    cos[index] = -np.inf
    k = min(int(k), int(phi.shape[0]) - 1)
    if k <= 0:
        return []
    return [int(i) for i in np.argpartition(-cos, kth=k - 1)[:k]]


def cosine_tertiles(phi: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    n = int(phi.shape[0])
    if n < 2:
        return {"near": [], "medium": [], "far": []}
    cos = np.asarray(phi, dtype=np.float64) @ np.asarray(phi, dtype=np.float64).T
    ii, jj = np.triu_indices(n, k=1)
    vals = cos[ii, jj]
    cuts = np.quantile(vals, [1.0 / 3.0, 2.0 / 3.0])
    bins: dict[str, list[tuple[int, int]]] = {"far": [], "medium": [], "near": []}
    for i, j, c in zip(ii.tolist(), jj.tolist(), vals.tolist()):
        pair = (int(i), int(j))
        if c <= cuts[0]:
            bins["far"].append(pair)
        elif c <= cuts[1]:
            bins["medium"].append(pair)
        else:
            bins["near"].append(pair)
    return bins


def _sample_pairs(
    bins: dict[str, list[tuple[int, int]]],
    n_pairs: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, str]]:
    order = ("near", "medium", "far")
    per = max(1, n_pairs // 3)
    chosen: list[tuple[int, int, str]] = []
    used: set[tuple[int, int]] = set()
    for name in order:
        pool = [p for p in bins.get(name, []) if p not in used]
        rng.shuffle(pool)
        take = pool[:per]
        for i, j in take:
            used.add((i, j))
            chosen.append((i, j, name))
    leftover = n_pairs - len(chosen)
    if leftover > 0:
        rest = [
            (i, j, name)
            for name in order
            for i, j in bins.get(name, [])
            if (i, j) not in used
        ]
        rng.shuffle(rest)
        chosen.extend(rest[:leftover])
    return chosen[:n_pairs]


def build_weight_set(
    phi: np.ndarray,
    rng: np.random.Generator,
    *,
    n_pairs: int = 40,
    pair_grid: Sequence[float] = PAIR_GRID,
    n_mix3: int = 100,
    n_mix5: int = 100,
    include_all_vertices: bool = True,
) -> list[dict[str, Any]]:
    """Sparse convex combinations over training-target embeddings."""
    n = int(phi.shape[0])
    weights: list[dict[str, Any]] = []
    if include_all_vertices:
        for i in range(n):
            weights.append(
                {
                    "family": "vertices",
                    "support": [i],
                    "weights": [1.0],
                    "kind": "vertex",
                    "decode": False,
                }
            )
    if n < 2 or n_pairs <= 0:
        return weights
    bins = cosine_tertiles(phi)
    pairs = _sample_pairs(bins, min(int(n_pairs), n * (n - 1) // 2), rng)
    decode_vertices: set[int] = set()
    for i, j, kind in pairs:
        decode_vertices.add(i)
        decode_vertices.add(j)
        for t in pair_grid:
            tt = float(t)
            weights.append(
                {
                    "family": "pairs",
                    "support": [i, j],
                    "weights": [1.0 - tt, tt],
                    "kind": f"pair_{kind}",
                    "t": tt,
                    "decode": True,
                }
            )
    for i in sorted(decode_vertices):
        weights.append(
            {
                "family": "vertices",
                "support": [i],
                "weights": [1.0],
                "kind": "vertex_anchor",
                "decode": True,
            }
        )

    def _mix(k: int, n_draw: int, family: str) -> None:
        if n < k or n_draw <= 0:
            return
        n_nn = n_draw // 2
        n_unif = n_draw - n_nn
        for which, count, kind in (
            ("nn", n_nn, "mix_nn"),
            ("uniform", n_unif, "mix_uniform"),
        ):
            for _ in range(count):
                if which == "nn":
                    center = int(rng.integers(0, n))
                    nbrs = nearest_indices(phi, center, k - 1)
                    support = [center] + nbrs
                else:
                    support = [int(x) for x in rng.choice(n, size=k, replace=False)]
                weights.append(
                    {
                        "family": family,
                        "support": support,
                        "weights": [float(x) for x in _dirichlet(rng, k)],
                        "kind": kind,
                        "decode": True,
                    }
                )

    _mix(3, int(n_mix3), "mix3")
    _mix(5, int(n_mix5), "mix5")
    return weights


def interpolant(phi: np.ndarray, weight: dict[str, Any]) -> np.ndarray:
    idx = np.asarray(weight["support"], dtype=np.int64)
    w = np.asarray(weight["weights"], dtype=np.float64)
    return w @ np.asarray(phi, dtype=np.float64)[idx]


def code_interpolant(mu: np.ndarray, weight: dict[str, Any]) -> np.ndarray:
    idx = np.asarray(weight["support"], dtype=np.int64)
    w = np.asarray(weight["weights"], dtype=np.float64)
    return w @ np.asarray(mu, dtype=np.float64)[idx]


def in_bounding_box(code: np.ndarray, mu: np.ndarray, *, atol: float = 1e-5) -> bool:
    lo = np.asarray(mu, dtype=np.float64).min(axis=0)
    hi = np.asarray(mu, dtype=np.float64).max(axis=0)
    b = np.asarray(code, dtype=np.float64)
    return bool(np.all(b >= lo - atol) and np.all(b <= hi + atol))


def jump_strawman(phi: np.ndarray, weight: dict[str, Any]) -> Optional[float]:
    if weight.get("family") != "pairs" or len(weight.get("support") or []) != 2:
        return None
    i, j = (int(x) for x in weight["support"])
    t = float(weight.get("t", 0.5))
    d = float(np.linalg.norm(phi[i] - phi[j]))
    return float(t * d if t < 0.5 else (1.0 - t) * d)


def nested_mask(weights: Sequence[dict[str, Any]], family: str) -> np.ndarray:
    rank = {name: i for i, name in enumerate(NESTED_FAMILIES)}
    cap = rank[family]
    return np.array(
        [rank.get(str(w.get("family")), 99) <= cap for w in weights],
        dtype=bool,
    )


def load_search_targets(path: Optional[str]) -> tuple[list[str], list[str], str]:
    if not path or not os.path.isfile(path):
        return [], [], "missing"
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    train = [str(x).strip() for x in (payload.get("train_targets") or []) if str(x).strip()]
    test = [str(x).strip() for x in (payload.get("test_targets") or []) if str(x).strip()]
    return train, test, path


def fallback_search_targets(canonical_dir: str) -> tuple[list[str], list[str], str]:
    """Rebuild the 5+5 search targets if the sweep JSON is not on disk."""
    path = os.path.abspath(os.path.expanduser(canonical_dir))
    if not os.path.isfile(os.path.join(path, "items.json")):
        print(
            "[cov_interp] no search-target JSON and no canonical items.json; "
            "held-out coverage will be empty",
            flush=True,
        )
        return [], [], "missing"
    spec = importlib.util.spec_from_file_location(
        "semantle_sample_targets",
        os.path.join(REPO_ROOT, "experiments", "semantle", "sample_targets.py"),
    )
    if spec is None or spec.loader is None:
        return [], [], "missing"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = module.sample_targets(path, n_train=5, n_test=5, seed=42)
    train = [str(x).strip() for x in (payload.get("train_targets") or []) if str(x).strip()]
    test = [str(x).strip() for x in (payload.get("test_targets") or []) if str(x).strip()]
    print(
        f"[cov_interp] rebuilt search targets from {path} "
        f"({len(train)} train, {len(test)} held-out)",
        flush=True,
    )
    return train, test, f"sample_targets:{path}"


def load_eval_test_words(output_dir: str) -> tuple[list[str], str]:
    results_path = os.path.join(output_dir, "eval", "results.json")
    if not os.path.isfile(results_path):
        return [], "missing"
    with open(results_path, encoding="utf-8") as handle:
        data = json.load(handle)
    meta = data.get("recon_test_meta") or data.get("genz_test_meta") or {}
    interp = list(meta.get("interp_words") or [])
    extrap = list(meta.get("extrap_words") or [])
    words = [str(w).strip() for w in interp + extrap if str(w).strip()]
    if not words:
        return [], "empty"
    return words, results_path


def rebuild_test_words(
    output_dir: str,
    train_words: Sequence[str],
    saved: dict[str, Any],
    train_phi: np.ndarray,
) -> tuple[list[str], str]:
    from boreft.eval.eval_suite import (
        DEFAULT_BBOX_PCA_VAR,
        DEFAULT_TEST_N_SAMPLES,
        build_test_sets,
        task_csv_paths,
    )

    csv_paths = [remap_repo_path(p) for p in task_csv_paths(saved, task=TASK)]
    interp, extrap, meta = build_test_sets(
        train_targets=list(train_words),
        csv_paths=csv_paths,
        test_n_samples=int(saved.get("test_n_samples", DEFAULT_TEST_N_SAMPLES)),
        seed=int(saved.get("seed", 42)),
        pca_var=float(saved.get("eval_bbox_pca_var", DEFAULT_BBOX_PCA_VAR)),
        task=TASK,
        train_embeddings=train_phi,
    )
    words = [str(w).strip() for w in list(interp) + list(extrap) if str(w).strip()]
    return words, f"rebuilt:{meta.get('n_test', len(words))}"


def encode_texts(texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
    from boreft.text_similarity import encode_texts_normalized

    payload = [str(t) for t in texts]
    if not payload:
        return np.zeros((0, 0), dtype=np.float64)
    chunks: list[np.ndarray] = []
    for start in range(0, len(payload), batch_size):
        chunks.append(
            encode_texts_normalized(payload[start : start + batch_size], task=TASK)
        )
    return np.concatenate(chunks, axis=0).astype(np.float64)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
        handle.write("\n")


def conditions_from_ladder(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    rows: list[dict[str, str]] = []
    for n, spec in (data.get("n_ladder") or {}).items():
        output_dir = spec.get("output_dir")
        if not output_dir:
            continue
        rows.append(
            {
                "name": f"n{n}",
                "wandb_id": str(spec.get("wandb") or ""),
                "output_dir": str(output_dir),
                "coverage_only": "1",
            }
        )
    return rows


def parse_condition(raw: str) -> dict[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"condition must be name=path, got {raw!r}")
    name, path = raw.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError(f"condition must be name=path, got {raw!r}")
    known = {row["name"]: row for row in DEFAULT_CONDITIONS}
    wandb_id = known[name]["wandb_id"] if name in known else ""
    return {"name": name, "wandb_id": wandb_id, "output_dir": path}


def resolve_specs(args: argparse.Namespace) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    seen: set[str] = set()

    def _add(row: dict[str, str]) -> None:
        name = row["name"]
        if name in seen:
            return
        seen.add(name)
        specs.append(dict(row))

    if args.condition:
        for row in args.condition:
            _add(row)
    elif not (args.from_ladder and args.coverage_only):
        for row in DEFAULT_CONDITIONS:
            _add(row)
    if args.from_ladder:
        ladder_path = args.ladder_json or DEFAULT_LADDER
        for row in conditions_from_ladder(ladder_path):
            _add(row)
    if args.coverage_only:
        for row in specs:
            row["coverage_only"] = "1"
    return specs


def coverage_block(
    *,
    weights: Sequence[dict[str, Any]],
    interpolants: np.ndarray,
    target_names: Sequence[str],
    target_phi: np.ndarray,
    train_phi: np.ndarray,
    rng: np.random.Generator,
    compute_hull: bool,
) -> dict[str, Any]:
    per_target: list[dict[str, Any]] = []
    family_values: dict[str, list[float]] = {name: [] for name in NESTED_FAMILIES}
    hull_dists: list[float] = []
    hull_supports: list[int] = []
    family_masks = {name: nested_mask(weights, name) for name in NESTED_FAMILIES}
    lipschitz = gram_lipschitz(train_phi) if compute_hull else None
    for name, vec in zip(target_names, target_phi):
        row: dict[str, Any] = {"target": name}
        for family in NESTED_FAMILIES:
            mask = family_masks[family]
            if not np.any(mask):
                row[f"epscov_{family}"] = None
                continue
            dist, idx = coverage_min(vec, interpolants[mask])
            row[f"epscov_{family}"] = dist
            row[f"epscov_{family}_alpha"] = int(np.flatnonzero(mask)[idx]) if idx >= 0 else None
            family_values[family].append(dist)
        if compute_hull:
            alpha, dist = project_onto_hull(vec, train_phi, lipschitz=lipschitz)
            n_interior = int(np.sum(alpha > HULL_INTERIOR))
            row["epscov_hull"] = dist
            row["hull_support"] = n_interior
            row["hull_is_vertex"] = bool(n_interior <= 1)
            hull_dists.append(dist)
            hull_supports.append(n_interior)
        per_target.append(row)
    summary: dict[str, Any] = {}
    for family in NESTED_FAMILIES:
        summary[family] = summarize_values(family_values[family], rng)
    if compute_hull:
        summary["hull"] = summarize_values(hull_dists, rng)
        if hull_supports:
            summary["hull"]["mean_support"] = float(np.mean(hull_supports))
            summary["hull"]["frac_interior"] = float(
                np.mean([s > 1 for s in hull_supports])
            )
            summary["hull"]["frac_vertex"] = float(
                np.mean([s <= 1 for s in hull_supports])
            )
    return {"summary": summary, "per_target": per_target}


def set_decode_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def decode_at_code(
    ckpt,
    saved: dict[str, Any],
    code: np.ndarray,
    *,
    n_samples: int,
    temperature: float,
    top_p: float,
    greedy: bool,
) -> dict[str, list[str]]:
    from boreft.eval.decode_utils import _decode_common
    from boreft.eval.semantle import DEFAULT_MAX_NEW_TOKENS, generate_text, generate_texts_batch

    decode_kw = dict(
        max_new_tokens=int(
            saved.get("full_eval_max_new_tokens") or DEFAULT_MAX_NEW_TOKENS
        ),
        **_decode_common(ckpt),
    )
    b = np.asarray(code, dtype=np.float32)
    out: dict[str, list[str]] = {}
    if greedy:
        out["greedy"] = [
            generate_text(
                ckpt.reft_model,
                ckpt.tokenizer,
                ckpt.prompt,
                word_idx=b,
                use_sample=False,
                **decode_kw,
            )
        ]
    if n_samples > 0:
        out["temperature_1.0"] = generate_texts_batch(
            ckpt.reft_model,
            ckpt.tokenizer,
            ckpt.prompt,
            word_idx=b,
            n_samples=int(n_samples),
            use_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            **decode_kw,
        )
    return out


def estimators_from_embeddings(
    emb: np.ndarray,
    x_alpha: np.ndarray,
) -> dict[str, Any]:
    floor = interpolation_floor(x_alpha)
    mean, se = expected_l2(emb, x_alpha)
    return {
        "epsint": mean,
        "epsint_se": se,
        "floor": floor,
        "excess": mean - floor,
        "displacement_plugin": plugin_displacement(emb, x_alpha),
        "displacement_split": split_half_displacement(emb, x_alpha),
        "direction_only": direction_only_distance(emb, x_alpha),
    }


def aggregate_interp(
    records: Sequence[dict[str, Any]],
    weights: Sequence[dict[str, Any]],
    variant: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for family in NESTED_FAMILIES:
        mask = nested_mask(weights, family)
        values: dict[str, list[float]] = {
            "epsint": [],
            "excess": [],
            "displacement_plugin": [],
            "displacement_split": [],
            "direction_only": [],
            "floor": [],
        }
        for rec, keep in zip(records, mask):
            if not keep or not rec.get("decode"):
                continue
            block = (rec.get("variants") or {}).get(variant) or {}
            for key in values:
                val = block.get(key)
                if val is not None and np.isfinite(val):
                    values[key].append(float(val))
        out[family] = {key: summarize_values(vals, rng) for key, vals in values.items()}
    return out


def pair_weight(rec: dict[str, Any]) -> Optional[float]:
    if rec.get("t") is not None:
        return float(rec["t"])
    weights = rec.get("weights") or []
    if rec.get("family") == "pairs" and len(weights) == 2:
        return float(weights[1])
    return None


def pair_records(
    records: Sequence[dict[str, Any]],
    *,
    t_lo: float = 0.0,
    t_hi: float = 1.0,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in records:
        if rec.get("family") != "pairs" or not rec.get("decode"):
            continue
        t = pair_weight(rec)
        if t is None or t + 1e-9 < t_lo or t - 1e-9 > t_hi:
            continue
        out.append(rec)
    return out


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def midpath_stats(
    records: Sequence[dict[str, Any]],
    variant: str,
    rng: np.random.Generator,
    *,
    t_lo: float = MIDPATH_T_LO,
    t_hi: float = MIDPATH_T_HI,
) -> dict[str, Any]:
    """Un-nested pair interpolants in ``[t_lo, t_hi]``, with a paired jump comparison."""
    recs = pair_records(records, t_lo=t_lo, t_hi=t_hi)
    epsint: list[float] = []
    excess: list[float] = []
    disp: list[float] = []
    floors: list[float] = []
    jumps: list[float] = []
    eps_minus: list[float] = []
    disp_minus: list[float] = []
    beat_eps: list[float] = []
    beat_disp: list[float] = []
    for rec in recs:
        block = (rec.get("variants") or {}).get(variant) or {}
        e = _finite(block.get("epsint"))
        x = _finite(block.get("excess"))
        d = _finite(block.get("displacement_split"))
        f = _finite(block.get("floor"))
        j = _finite(rec.get("jump_strawman"))
        if e is not None:
            epsint.append(e)
        if x is not None:
            excess.append(x)
        if d is not None:
            disp.append(d)
        if f is not None:
            floors.append(f)
        if j is not None:
            jumps.append(j)
            if e is not None:
                eps_minus.append(e - j)
                beat_eps.append(float(e < j))
            if d is not None:
                disp_minus.append(d - j)
                beat_disp.append(float(d < j))
    return {
        "t_lo": t_lo,
        "t_hi": t_hi,
        "n": len(recs),
        "epsint": summarize_values(epsint, rng),
        "excess": summarize_values(excess, rng),
        "displacement_split": summarize_values(disp, rng),
        "floor": summarize_values(floors, rng),
        "jump": summarize_values(jumps, rng),
        "epsint_minus_jump": summarize_values(eps_minus, rng),
        "disp_minus_jump": summarize_values(disp_minus, rng),
        "beat_jump_epsint": None if not beat_eps else float(np.mean(beat_eps)),
        "beat_jump_disp": None if not beat_disp else float(np.mean(beat_disp)),
    }


def run_coverage_condition(
    *,
    spec: dict[str, str],
    args: argparse.Namespace,
    search_test: Sequence[str],
    eval_test: Sequence[str],
    a_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    output_dir = os.path.abspath(os.path.expanduser(spec["output_dir"]))
    from boreft.data_utils import load_merged_run_config

    saved = load_merged_run_config(output_dir)
    words, word_ids, _items = load_items(output_dir)
    print(
        f"[cov_interp] [{spec['name']}] n_train={len(words)}  {output_dir}",
        flush=True,
    )
    phi = encode_texts(words, batch_size=args.batch_size)
    key = word_set_key(words)
    canon = canonical_words(words)
    rng = np.random.default_rng(args.seed)
    if key in a_cache and not args.rebuild_a:
        packed = a_cache[key]
        weights = remap_weight_supports(packed["weights_canon"], packed["canon_words"], words)
        interpolants = np.asarray(packed["interpolants"], dtype=np.float64)
        print(
            f"[cov_interp] [{spec['name']}] reuse A ({key}, {len(weights)} weights)",
            flush=True,
        )
    else:
        loc = local_index_map(words)
        canon_idx = [loc[name] for name in canon]
        phi_canon = phi[np.asarray(canon_idx, dtype=np.int64)]
        weights_canon = build_weight_set(
            phi_canon,
            rng,
            n_pairs=args.n_pairs,
            pair_grid=args.pair_grid,
            n_mix3=args.n_mix3,
            n_mix5=args.n_mix5,
        )
        interpolants = np.stack(
            [interpolant(phi_canon, w) for w in weights_canon], axis=0
        )
        a_cache[key] = {
            "word_set_key": key,
            "canon_words": canon,
            "weights_canon": weights_canon,
            "interpolants": interpolants,
        }
        weights = remap_weight_supports(weights_canon, canon, words)
        print(
            f"[cov_interp] [{spec['name']}] built A: "
            f"{sum(1 for w in weights if w['family']=='vertices')} vertices, "
            f"{sum(1 for w in weights if w['family']=='pairs')} pairs, "
            f"{sum(1 for w in weights if w['family']=='mix3')} mix3, "
            f"{sum(1 for w in weights if w['family']=='mix5')} mix5",
            flush=True,
        )

    search_keep = [w for w in search_test if w.strip().lower() not in {x.lower() for x in words}]
    eval_keep = [w for w in eval_test if w.strip().lower() not in {x.lower() for x in words}]
    search_phi = encode_texts(search_keep, batch_size=args.batch_size) if search_keep else np.zeros((0, phi.shape[1]))
    eval_phi = encode_texts(eval_keep, batch_size=args.batch_size) if eval_keep else np.zeros((0, phi.shape[1]))
    cov_rng = np.random.default_rng(args.seed + 1)
    coverage = {
        "search_heldout": coverage_block(
            weights=weights,
            interpolants=interpolants,
            target_names=search_keep,
            target_phi=search_phi,
            train_phi=phi,
            rng=cov_rng,
            compute_hull=not args.skip_hull,
        ),
        "eval_test": coverage_block(
            weights=weights,
            interpolants=interpolants,
            target_names=eval_keep,
            target_phi=eval_phi,
            train_phi=phi,
            rng=cov_rng,
            compute_hull=not args.skip_hull,
        ),
    }
    jump_vals = [jump_strawman(phi, w) for w in weights]
    jump_vals = [v for v in jump_vals if v is not None]
    payload: dict[str, Any] = {
        "name": spec["name"],
        "wandb_id": spec.get("wandb_id", ""),
        "output_dir": output_dir,
        "coverage_only": True,
        "n_train": len(words),
        "word_set_key": key,
        "words": words,
        "word_ids": word_ids,
        "bias_type": saved.get("bias_type"),
        "lambda_sdpo": saved.get("lambda_sdpo"),
        "kl_beta": saved.get("kl_beta"),
        "layer": saved.get("layer"),
        "low_rank_dim": saved.get("low_rank_dim"),
        "seed": args.seed,
        "n_pairs": args.n_pairs,
        "n_mix3": args.n_mix3,
        "n_mix5": args.n_mix5,
        "pair_grid": list(args.pair_grid),
        "n_weights": len(weights),
        "n_decode_weights": sum(1 for w in weights if w.get("decode")),
        "weights": weights,
        "coverage": coverage,
        "references": {
            "jump_strawman": summarize_values(jump_vals, cov_rng),
        },
    }
    return payload


def run_interp_condition(
    *,
    spec: dict[str, str],
    args: argparse.Namespace,
    base: dict[str, Any],
    out_path: Optional[str] = None,
) -> dict[str, Any]:
    from boreft.bias_tables import stack_bias_vectors
    from boreft.data_utils import load_merged_run_config
    from boreft.eval.semantle import (
        load_eval_checkpoint,
        release_eval_checkpoint,
        sample_sobol_bias_vectors,
    )

    output_dir = base["output_dir"]
    saved = load_merged_run_config(output_dir)
    model_name = saved.get("model_name")
    if not model_name:
        raise ValueError(f"{output_dir}: training_config.json is missing model_name")
    layer = int(saved.get("layer", 13))
    rank = int(saved.get("low_rank_dim", 64))
    top_p = float(
        saved.get("full_eval_top_p")
        if saved.get("full_eval_top_p") is not None
        else args.top_p
    )
    print(
        f"[cov_interp] [{spec['name']}] load LLM {model_name} layer={layer} rank={rank}",
        flush=True,
    )
    ckpt = load_eval_checkpoint(
        output_dir,
        model_name,
        layer,
        rank,
        args.cache_dir or saved.get("cache_dir") or DEFAULT_CACHE_DIR,
        skip_embed_cache=True,
    )
    try:
        mu = stack_bias_vectors(ckpt.reft_model, list(base["word_ids"]))
        phi = encode_texts(base["words"], batch_size=args.batch_size)
        weights: list[dict[str, Any]] = list(base["weights"])
        n_decode = sum(1 for w in weights if w.get("decode"))
        prev_records = {
            int(rec["index"]): rec
            for rec in ((base.get("interpolation") or {}).get("records") or [])
            if rec.get("decode") and rec.get("variants")
        }
        print(
            f"[cov_interp] [{spec['name']}] decode {n_decode} codes × "
            f"({args.n_samples} T=1 + greedy) × {args.n_seeds} seeds"
            + (f"  resume={len(prev_records)}" if prev_records else ""),
            flush=True,
        )
        records: list[dict[str, Any]] = []
        done = 0
        for i, w in enumerate(weights):
            if i in prev_records:
                rec = prev_records[i]
                records.append(rec)
                if rec.get("decode"):
                    done += 1
                continue
            x_a = interpolant(phi, w)
            rec = {
                "index": i,
                "family": w["family"],
                "kind": w.get("kind"),
                "support": w["support"],
                "weights": w["weights"],
                "x_norm": float(np.linalg.norm(x_a)),
                "floor": interpolation_floor(x_a),
                "decode": bool(w.get("decode")),
                "jump_strawman": jump_strawman(phi, w),
                "variants": {},
            }
            if not w.get("decode"):
                records.append(rec)
                continue
            b = code_interpolant(mu, w)
            if not in_bounding_box(b, mu):
                raise RuntimeError(
                    f"{spec['name']}: interpolated code left the μ bounding box "
                    f"(family={w['family']}, support={w['support']})"
                )
            rec["b_norm"] = float(np.linalg.norm(b))
            bags: dict[str, list[str]] = {"greedy": [], "temperature_1.0": []}
            for seed_i, seed in enumerate(args.decode_seeds):
                set_decode_seed(int(seed))
                decoded = decode_at_code(
                    ckpt,
                    saved,
                    b,
                    n_samples=args.n_samples,
                    temperature=args.temperature,
                    top_p=top_p,
                    greedy=(seed_i == 0),
                )
                for variant, texts in decoded.items():
                    bags[variant].extend(texts)
            rec["decodes"] = bags
            for variant, texts in bags.items():
                if not texts:
                    continue
                rec["variants"][variant] = estimators_from_embeddings(
                    encode_texts(texts, batch_size=args.batch_size),
                    x_a,
                )
                rec["variants"][variant]["n_decodes"] = len(texts)
            records.append(rec)
            done += 1
            if done == 1 or done % 20 == 0 or done == n_decode:
                print(
                    f"[cov_interp] [{spec['name']}] {done}/{n_decode} {w['family']}",
                    flush=True,
                )
            if out_path and (done % 50 == 0 or done == n_decode):
                base["interpolation"] = {
                    "n_samples": args.n_samples,
                    "n_seeds": args.n_seeds,
                    "decode_seeds": list(args.decode_seeds),
                    "temperature": args.temperature,
                    "top_p": top_p,
                    "partial": done < n_decode,
                    "records": records,
                }
                write_json(out_path, base)

        rng = np.random.default_rng(args.seed + 2)
        interp_summary = {
            "greedy": aggregate_interp(records, weights, "greedy", rng),
            "temperature_1.0": aggregate_interp(
                records, weights, "temperature_1.0", rng
            ),
            "midpath": {
                "greedy": midpath_stats(records, "greedy", rng),
                "temperature_1.0": midpath_stats(records, "temperature_1.0", rng),
            },
        }

        n_random = int(args.n_random)
        random_block: Optional[dict[str, Any]] = None
        if n_random > 0:
            sampled_bs, _, _, _ = sample_sobol_bias_vectors(
                ckpt.reft_model,
                list(base["words"]),
                n_random,
                word_ids=list(base["word_ids"]),
                mu_all=mu,
                seed=args.seed,
            )
            interp_pts = np.stack([interpolant(phi, w) for w in weights], axis=0)
            dists: list[float] = []
            for b in sampled_bs:
                set_decode_seed(args.seed)
                texts = decode_at_code(
                    ckpt,
                    saved,
                    b,
                    n_samples=max(args.n_samples, 1),
                    temperature=args.temperature,
                    top_p=top_p,
                    greedy=False,
                )["temperature_1.0"]
                emb = encode_texts(texts, batch_size=args.batch_size)
                for vec in emb:
                    dist, _ = coverage_min(vec, interp_pts)
                    dists.append(dist)
            random_block = summarize_values(dists, rng)

        vertex_floor = [
            rec["variants"]["temperature_1.0"]["epsint"]
            for rec in records
            if rec.get("kind") == "vertex_anchor"
            and rec.get("variants", {}).get("temperature_1.0")
        ]
        base["interpolation"] = {
            "n_samples": args.n_samples,
            "n_seeds": args.n_seeds,
            "decode_seeds": list(args.decode_seeds),
            "temperature": args.temperature,
            "top_p": top_p,
            "partial": False,
            "summary": interp_summary,
            "records": records,
        }
        base["references"]["reconstruction_floor"] = summarize_values(vertex_floor, rng)
        base["references"]["random_code"] = random_block
        base["coverage_only"] = False
        base["model_name"] = model_name
        base["layer"] = layer
        base["rank"] = rank
        return base
    finally:
        release_eval_checkpoint(ckpt)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def _nested_get(payload: dict[str, Any], *keys: str) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def ensure_midpath(payload: dict[str, Any], rng: Optional[np.random.Generator] = None) -> dict[str, Any]:
    existing = _nested_get(payload, "interpolation", "summary", "midpath")
    t1 = (existing or {}).get("temperature_1.0") or {}
    if (
        existing
        and abs(float(t1.get("t_lo", -1)) - MIDPATH_T_LO) < 1e-9
        and abs(float(t1.get("t_hi", -1)) - MIDPATH_T_HI) < 1e-9
    ):
        return existing
    records = _nested_get(payload, "interpolation", "records") or []
    if not records:
        return {}
    rng = rng or np.random.default_rng(42)
    mid = {
        "greedy": midpath_stats(records, "greedy", rng),
        "temperature_1.0": midpath_stats(records, "temperature_1.0", rng),
    }
    summary = payload.setdefault("interpolation", {}).setdefault("summary", {})
    summary["midpath"] = mid
    return mid


def latex_tables(payloads: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    abl = [p for p in payloads if p.get("interpolation")]
    rng = np.random.default_rng(42)
    if abl:
        lines.append(
            "% Table 1. Mid-path distance of the mean to the interpolant at T=1 (t in [0.4, 0.6])."
        )
        lines.append(r"\begin{tabular}{lc}")
        lines.append(r"\toprule")
        lines.append(r"Condition & Distance to interpolant \\")
        lines.append(r"\midrule")
        jump_med = None
        for row in abl:
            mid = ensure_midpath(row, rng).get("temperature_1.0") or {}
            disp = mid.get("displacement_split") or {}
            if jump_med is None:
                jump_med = (mid.get("jump") or {}).get("median")
            lines.append(
                f"{condition_label(row['name'])} & {_fmt(disp.get('median'))} \\\\"
            )
        lines.append(f"No interpolation & {_fmt(jump_med)} \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append("")

        canon = next((p for p in abl if p["name"] == "canonical"), abl[0])
        lines.append("% Table 2. Nested $\\mathcal{A}$ trade-off on BOReFT.")
        lines.append(r"\begin{tabular}{lcccc}")
        lines.append(r"\toprule")
        lines.append(
            r"$\mathcal{A}$ & median $\epscov$ (held-out) & q90 $\epsint$ & max $\epsint$ & $\epscov+\epsint$ \\"
        )
        lines.append(r"\midrule")
        for family, label in (
            ("vertices", "vertices"),
            ("pairs", "+ pairs"),
            ("mix3", "+ mix3"),
            ("mix5", "+ mix5"),
        ):
            cov = _nested_get(
                canon, "coverage", "search_heldout", "summary", family
            ) or {}
            inn = _nested_get(
                canon, "interpolation", "summary", "temperature_1.0", family, "epsint"
            ) or {}
            bound = None
            if cov.get("median") is not None and inn.get("max") is not None:
                bound = float(cov["median"]) + float(inn["max"])
            lines.append(
                f"{label} & {_fmt(cov.get('median'))} & {_fmt(inn.get('q90'))} & "
                f"{_fmt(inn.get('max'))} & {_fmt(bound)} \\\\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append("")

    n_rows = [
        p
        for p in payloads
        if str(p.get("name", "")).startswith("n") and str(p.get("name", ""))[1:].isdigit()
    ]
    n_rows.sort(key=lambda p: int(str(p["name"])[1:]))
    if n_rows:
        lines.append("% Table 3. Coverage vs training-set size (eval test set).")
        lines.append(r"\begin{tabular}{lccccc}")
        lines.append(r"\toprule")
        lines.append(
            r"$n$ & vertices & +pairs & +mix3 & +mix5 & hull \\"
        )
        lines.append(r"\midrule")
        for row in n_rows:
            n = str(row["name"])[1:]
            cells = [
                _fmt(
                    _nested_get(
                        row, "coverage", "eval_test", "summary", family, "median"
                    )
                )
                for family in NESTED_FAMILIES
            ]
            hull = _fmt(
                _nested_get(row, "coverage", "eval_test", "summary", "hull", "median")
            )
            lines.append(f"{n} & " + " & ".join(cells) + f" & {hull} \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def slim_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": payload.get("name"),
        "wandb_id": payload.get("wandb_id"),
        "output_dir": payload.get("output_dir"),
        "coverage_only": payload.get("coverage_only"),
        "n_train": payload.get("n_train"),
        "word_set_key": payload.get("word_set_key"),
        "bias_type": payload.get("bias_type"),
        "lambda_sdpo": payload.get("lambda_sdpo"),
        "n_weights": payload.get("n_weights"),
        "n_decode_weights": payload.get("n_decode_weights"),
        "coverage": {
            split: {
                "summary": (payload.get("coverage") or {}).get(split, {}).get("summary"),
                "n_targets": len(
                    (payload.get("coverage") or {}).get(split, {}).get("per_target") or []
                ),
            }
            for split in ("search_heldout", "eval_test")
        },
        "interpolation": {
            "summary": (payload.get("interpolation") or {}).get("summary"),
            "n_samples": (payload.get("interpolation") or {}).get("n_samples"),
        }
        if payload.get("interpolation")
        else None,
        "references": payload.get("references"),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--condition",
        action="append",
        type=parse_condition,
        default=None,
        help="name=output_dir (repeatable). Default: the four paper ablations.",
    )
    p.add_argument(
        "--from-ladder",
        action="store_true",
        help="Add n-ladder checkpoints from --ladder-json (coverage-only).",
    )
    p.add_argument("--ladder-json", default=DEFAULT_LADDER)
    p.add_argument(
        "--coverage-only",
        action="store_true",
        help="Skip LLM loading and decoding. Interpolation is omitted.",
    )
    p.add_argument("--n-pairs", type=int, default=40)
    p.add_argument("--n-mix3", type=int, default=100)
    p.add_argument("--n-mix5", type=int, default=100)
    p.add_argument(
        "--pair-grid",
        type=float,
        nargs="+",
        default=list(PAIR_GRID),
        help="Interior t values on each pair segment (default: 0.1 0.2 ... 0.9).",
    )
    p.add_argument("--n-samples", type=int, default=16, help="T=1 decodes per code.")
    p.add_argument("--n-seeds", type=int, default=2, help="Independent decode seeds.")
    p.add_argument("--decode-seed", type=int, default=0, help="Base decode seed.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--n-random", type=int, default=32, help="Sobol-box reference codes.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--skip-hull", action="store_true")
    p.add_argument("--rebuild-a", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--make-table", action="store_true")
    p.add_argument(
        "--search-targets-json",
        default=DEFAULT_TARGETS_JSON,
        help="JSON with train_targets / test_targets from sample_targets.py.",
    )
    p.add_argument(
        "--canonical-dir",
        default=CANONICAL_DIR,
        help="Checkpoint used to recover the 512-word eval test set if needed.",
    )
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--wandb-project", default="boreft")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-dir", default=None)
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args(argv)


def load_existing_payloads(out_dir: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not os.path.isdir(out_dir):
        return rows
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".json") or name in {"summary.json", "tables.tex"}:
            continue
        path = os.path.join(out_dir, name)
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and payload.get("name"):
            rows.append(payload)
    return rows


def resolve_eval_test(args: argparse.Namespace, specs: Sequence[dict[str, str]]) -> tuple[list[str], str]:
    canonical = os.path.abspath(os.path.expanduser(args.canonical_dir))
    words, source = load_eval_test_words(canonical)
    if words:
        return words, source
    for spec in specs:
        path = os.path.abspath(os.path.expanduser(spec["output_dir"]))
        words, source = load_eval_test_words(path)
        if words:
            return words, source
    if os.path.isdir(canonical) and os.path.isfile(os.path.join(canonical, "items.json")):
        from boreft.data_utils import load_merged_run_config

        saved = load_merged_run_config(canonical)
        train_words, _, _ = load_items(canonical)
        phi = encode_texts(train_words, batch_size=args.batch_size)
        words, source = rebuild_test_words(canonical, train_words, saved, phi)
        return words, source
    return [], "missing"


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args.decode_seeds = [int(args.decode_seed) + i for i in range(int(args.n_seeds))]
    os.makedirs(args.out_dir, exist_ok=True)

    if args.make_table:
        payloads = load_existing_payloads(args.out_dir)
        text = latex_tables(payloads)
        tex_path = os.path.join(args.out_dir, "tables.tex")
        with open(tex_path, "w", encoding="utf-8") as handle:
            handle.write(text)
        print(text, end="")
        print(f"[cov_interp] wrote {tex_path}", flush=True)
        return

    specs = resolve_specs(args)
    if not specs:
        raise SystemExit("no conditions to run")

    if args.dry_run:
        for spec in specs:
            path = os.path.abspath(os.path.expanduser(spec["output_dir"]))
            n = "?"
            n_int: Optional[int] = None
            if os.path.isfile(os.path.join(path, "items.json")):
                words, _, _ = load_items(path)
                n = str(len(words))
                n_int = len(words)
            n_grid = len(args.pair_grid)
            if n_int is not None and n_int < 2:
                n_decode = 0
            else:
                n_decode = (
                    int(args.n_pairs) * n_grid
                    + int(args.n_mix3)
                    + int(args.n_mix5)
                    + 2 * int(args.n_pairs)
                )
            coverage_only = args.coverage_only or spec.get("coverage_only") == "1"
            gens = 0 if coverage_only else n_decode * (int(args.n_samples) + 1) * int(args.n_seeds)
            print(
                f"[dry-run] {spec['name']:12s} n_train={n:4s}  "
                f"decode_weights≈{n_decode:<4d}  gens≈{gens:<6d}  "
                f"{'coverage-only' if coverage_only else 'full'}  {path}",
                flush=True,
            )
        return

    search_train, search_test, search_src = load_search_targets(args.search_targets_json)
    if not search_test:
        search_train, search_test, search_src = fallback_search_targets(args.canonical_dir)
    eval_test, eval_src = resolve_eval_test(args, specs)
    print(
        f"[cov_interp] search held-out={len(search_test)} from {search_src}; "
        f"eval test={len(eval_test)} from {eval_src}; "
        f"search train listed={len(search_train)}",
        flush=True,
    )

    a_cache: dict[str, dict[str, Any]] = {}
    payloads: list[dict[str, Any]] = []
    for spec in specs:
        out_path = os.path.join(args.out_dir, f"{spec['name']}.json")
        coverage_only = args.coverage_only or spec.get("coverage_only") == "1"
        existing = None
        if os.path.isfile(out_path) and not args.overwrite:
            with open(out_path, encoding="utf-8") as handle:
                existing = json.load(handle)
            interp = (existing or {}).get("interpolation") if isinstance(existing, dict) else None
            finished = bool(interp) and not interp.get("partial")
            if coverage_only or finished:
                print(f"[cov_interp] skip existing {out_path}", flush=True)
                payloads.append(existing)
                continue
        if existing is not None and existing.get("weights"):
            print(f"[cov_interp] resume {out_path}", flush=True)
            base = existing
        else:
            base = run_coverage_condition(
                spec=spec,
                args=args,
                search_test=search_test,
                eval_test=eval_test,
                a_cache=a_cache,
            )
            write_json(out_path, base)
        if not coverage_only:
            base = run_interp_condition(
                spec=spec, args=args, base=base, out_path=out_path
            )
        write_json(out_path, base)
        print(f"[cov_interp] wrote {out_path}", flush=True)
        payloads.append(base)

    summary = {
        "seed": args.seed,
        "n_pairs": args.n_pairs,
        "n_mix3": args.n_mix3,
        "n_mix5": args.n_mix5,
        "pair_grid": list(args.pair_grid),
        "n_samples": args.n_samples,
        "n_seeds": args.n_seeds,
        "search_targets_json": args.search_targets_json,
        "conditions": [slim_summary(p) for p in payloads],
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    write_json(summary_path, summary)
    tex = latex_tables(payloads)
    tex_path = os.path.join(args.out_dir, "tables.tex")
    with open(tex_path, "w", encoding="utf-8") as handle:
        handle.write(tex)
    print(tex, end="")
    print(f"[cov_interp] wrote {summary_path}", flush=True)
    from boreft.search_wandb import maybe_log_named_analysis

    maybe_log_named_analysis(
        args.out_dir,
        summary_path,
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        group=args.wandb_group,
        name=args.wandb_run_name,
        wandb_dir=args.wandb_dir,
        no_wandb=args.no_wandb,
    )


if __name__ == "__main__":
    main()
