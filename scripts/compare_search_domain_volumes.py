#!/usr/bin/env python3
"""Compare Lebesgue volumes of Semantle BOReFT search domains.

Rebuilds the same regions ``boreft.search`` uses: the axis-aligned box of
training posterior means expanded by ``k σ``, and the covering Mahalanobis
ellipsoid of those means. In 64D the raw volumes overflow, so the table
reports log-volume and the side of an equal-volume cube.

Default domains match the search-domain ablation: AABB k=0, 0.1, 0.5, and
the covering ellipsoid (plus that ellipsoid's enclosing AABB, which the GP
uses for input scaling).

Prefers ``bias_tables.pt`` (CPU). Pass ``--from-checkpoint`` only if tables
are missing.

    python scripts/compare_search_domain_volumes.py
    python scripts/compare_search_domain_volumes.py --output-dir outputs/1784053292
    sbatch scripts/compare_search_domain_volumes.sh
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Optional

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from boreft.bo.acquisition import (  # noqa: E402
    box_log_volume,
    latent_bounds,
    latent_ellipsoid,
)
from boreft.data_utils import (  # noqa: E402
    add_load_latest_argument,
    load_merged_run_config,
    resolve_weight_dir_and_run_config,
)
from boreft.pyreft.losses import clamp_logvar  # noqa: E402

DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "1784053292")
DEFAULT_JSON_OUT = os.path.join(
    REPO_ROOT, "data", "semantle", "analysis", "search_domain_volumes.json"
)
DEFAULT_K = (0.0, 0.1, 0.5)
DEFAULT_CACHE_DIR = None


def _json_float(value: float) -> float:
    return float(value)


def equivalent_cube_side(log_volume: float, dim: int) -> float:
    if dim < 1:
        raise ValueError("dim must be positive")
    return math.exp(log_volume / dim)


def _span_stats(sides: np.ndarray) -> dict[str, float]:
    sides = np.asarray(sides, dtype=np.float64).reshape(-1)
    if sides.size == 0 or np.any(sides <= 0) or not np.isfinite(sides).all():
        raise ValueError("side lengths must be positive and finite")
    return {
        "mean_span": float(sides.mean()),
        "median_span": float(np.median(sides)),
        "min_span": float(sides.min()),
        "max_span": float(sides.max()),
        "geo_mean_span": float(np.exp(np.log(sides).mean())),
    }


def _box_record(name: str, bounds: np.ndarray, *, dim: int) -> dict[str, Any]:
    log_vol = box_log_volume(bounds)
    sides = np.asarray(bounds[1] - bounds[0], dtype=np.float64)
    record = {
        "name": name,
        "kind": "aabb",
        "log_volume": _json_float(log_vol),
        "equivalent_cube_side": _json_float(equivalent_cube_side(log_vol, dim)),
    }
    record.update(_span_stats(sides))
    return record


def load_items(weight_dir: str, run_root: str) -> list[Any]:
    for path in (
        os.path.join(weight_dir, "items.json"),
        os.path.join(run_root, "items.json"),
    ):
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as handle:
                items = json.load(handle)
            if not isinstance(items, list) or not items:
                raise ValueError(f"{path}: expected a non-empty list")
            return items
    raise FileNotFoundError(f"No items.json in {weight_dir} or {run_root}")


def load_mu_std_from_tables(
    output_dir: str, *, load_latest: bool
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from boreft.bias_tables import load_bias_tables, resolve_bias_tables_path

    run_root = os.path.abspath(os.path.expanduser(output_dir))
    weight_dir, saved = resolve_weight_dir_and_run_config(
        run_root, load_latest=load_latest
    )
    items = load_items(weight_dir, run_root)
    n_words = len(items)
    tables_path = resolve_bias_tables_path(
        weight_dir, saved_cfg=saved
    ) or resolve_bias_tables_path(run_root, saved_cfg=saved)
    if tables_path is None:
        raise FileNotFoundError(
            f"no bias_tables.pt under {weight_dir} or {run_root}; "
            "pass --from-checkpoint to load the intervention"
        )
    mu_t, logvar_t, _meta = load_bias_tables(tables_path, num_words=n_words)
    if logvar_t is None:
        raise ValueError(
            f"{tables_path}: no logvar; AABB k>0 needs a learned posterior std"
        )
    mu = mu_t[:n_words].detach().float().cpu().numpy().astype(np.float32)
    std = (
        torch.exp(0.5 * clamp_logvar(logvar_t[:n_words].float()))
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    if mu.shape[0] != n_words:
        raise ValueError(f"μ has {mu.shape[0]} rows, items.json has {n_words}")
    return mu, std, {
        "source": "bias_tables",
        "tables_path": os.path.abspath(tables_path),
        "weight_dir": weight_dir,
        "n_words": n_words,
        "rank": int(mu.shape[1]),
        "saved_cfg": saved,
    }


def load_mu_std_from_checkpoint(
    output_dir: str, *, load_latest: bool, cache_dir: Optional[str]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from boreft.bias_tables import stack_bias_mu_std
    from boreft.eval.semantle import load_eval_checkpoint, release_eval_checkpoint

    run_root = os.path.abspath(os.path.expanduser(output_dir))
    saved = load_merged_run_config(run_root)
    model_name = saved.get("model_name")
    if not model_name:
        raise ValueError(f"{run_root}: training_config.json is missing model_name")
    layer = int(saved.get("layer", 13))
    rank = int(saved.get("low_rank_dim", 64))
    ckpt = load_eval_checkpoint(
        run_root,
        model_name,
        layer,
        rank,
        cache_dir or saved.get("cache_dir") or DEFAULT_CACHE_DIR,
        load_latest=load_latest,
        skip_embed_cache=True,
    )
    try:
        ids = list(range(len(ckpt.words)))
        mu, std = stack_bias_mu_std(ckpt.reft_model, ids)
        mu = np.asarray(mu, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        return mu, std, {
            "source": "checkpoint",
            "tables_path": None,
            "weight_dir": run_root,
            "n_words": int(mu.shape[0]),
            "rank": int(mu.shape[1]),
            "saved_cfg": saved,
        }
    finally:
        release_eval_checkpoint(ckpt)


def compare_domains(
    mu: np.ndarray,
    std: np.ndarray,
    *,
    aabb_std_k: tuple[float, ...],
    bounds_padding: float,
) -> dict[str, Any]:
    mu = np.asarray(mu, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mu.ndim != 2 or std.shape != mu.shape:
        raise ValueError("mu and std must be matching [N, D] arrays")
    dim = int(mu.shape[1])
    domains: list[dict[str, Any]] = []
    for k in aabb_std_k:
        if not np.isfinite(k) or k < 0:
            raise ValueError(f"aabb-std-k must be finite and nonnegative, got {k}")
        std_arg = None if k == 0 else std
        bounds = latent_bounds(
            mu, padding=bounds_padding, std=std_arg, std_k=k
        )
        record = _box_record(f"AABB k={k:g}", bounds, dim=dim)
        record["aabb_std_k"] = float(k)
        domains.append(record)

    ellipsoid = latent_ellipsoid(mu, padding=bounds_padding)
    ell_log_vol = ellipsoid.log_volume()
    ell_aabb = ellipsoid.aabb()
    mahalanobis = ellipsoid.mahalanobis(mu)
    ell_record = {
        "name": "covering ellipsoid",
        "kind": "ellipsoid",
        "log_volume": _json_float(ell_log_vol),
        "equivalent_cube_side": _json_float(
            equivalent_cube_side(ell_log_vol, dim)
        ),
        "max_mahalanobis": float(mahalanobis.max()),
        "n_means_outside": int(np.sum(~ellipsoid.contains(mu))),
        "log_abs_det_L": float(np.linalg.slogdet(ellipsoid.chol)[1]),
    }
    ell_record.update(_span_stats(ell_aabb[1] - ell_aabb[0]))
    domains.append(ell_record)

    enclosing = _box_record("ellipsoid enclosing AABB", ell_aabb, dim=dim)
    enclosing["kind"] = "ellipsoid_aabb"
    domains.append(enclosing)

    base_log = domains[0]["log_volume"]
    for record in domains:
        delta = record["log_volume"] - base_log
        record["log_volume_ratio_vs_k0"] = _json_float(delta)
        record["volume_ratio_vs_k0"] = _json_float(math.exp(delta))

    mean_sigma = float(std.mean())
    k0_span = domains[0]["mean_span"]
    return {
        "n_words": int(mu.shape[0]),
        "rank": dim,
        "bounds_padding": float(bounds_padding),
        "mean_posterior_std": mean_sigma,
        "mean_k0_span": k0_span,
        "mean_std_over_mean_k0_span": (
            mean_sigma / k0_span if k0_span > 0 else float("nan")
        ),
        "domains": domains,
    }


def format_table(payload: dict[str, Any]) -> str:
    rows = payload["domains"]
    header = (
        f"{'domain':<28} {'logVol':>10} {'eq.cube':>10} {'mean span':>10} "
        f"{'Vol / B0':>12}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['name']:<28} {row['log_volume']:10.2f} "
            f"{row['equivalent_cube_side']:10.4f} {row['mean_span']:10.4f} "
            f"{row['volume_ratio_vs_k0']:12.3g}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare AABB vs covering-ellipsoid search-domain volumes."
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="BOReFT run directory (default: canonical Semantle checkpoint).",
    )
    parser.add_argument(
        "--aabb-std-k",
        type=float,
        nargs="+",
        default=list(DEFAULT_K),
        help="AABB expansion k values (default: 0 0.1 0.5).",
    )
    parser.add_argument(
        "--bounds-padding",
        type=float,
        default=0.0,
        help="Match search --bounds-padding (default 0).",
    )
    parser.add_argument(
        "--json-out",
        default=DEFAULT_JSON_OUT,
        help="Write the comparison JSON here.",
    )
    parser.add_argument(
        "--from-checkpoint",
        action="store_true",
        help="Load μ/σ from the intervention instead of bias_tables.pt.",
    )
    parser.add_argument("--cache-dir", default=None)
    add_load_latest_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bounds_padding < 0:
        raise SystemExit("--bounds-padding must be nonnegative")
    requested: list[float] = []
    for raw in args.aabb_std_k:
        k = float(raw)
        if not np.isfinite(k) or k < 0:
            raise SystemExit(f"--aabb-std-k must be finite and nonnegative, got {raw}")
        if k not in requested:
            requested.append(k)
    if not requested:
        raise SystemExit("--aabb-std-k requires at least one value")
    ks = tuple(dict.fromkeys((0.0, *requested)))

    if args.from_checkpoint:
        mu, std, meta = load_mu_std_from_checkpoint(
            args.output_dir,
            load_latest=bool(args.load_latest),
            cache_dir=args.cache_dir,
        )
    else:
        mu, std, meta = load_mu_std_from_tables(
            args.output_dir, load_latest=bool(args.load_latest)
        )

    payload = compare_domains(
        mu, std, aabb_std_k=ks, bounds_padding=args.bounds_padding
    )
    payload["output_dir"] = os.path.abspath(os.path.expanduser(args.output_dir))
    payload["source"] = meta["source"]
    payload["tables_path"] = meta["tables_path"]
    payload["weight_dir"] = meta["weight_dir"]
    payload["aabb_std_k"] = [float(k) for k in ks]

    print(
        f"[domain-vol] {payload['output_dir']}  n={payload['n_words']}  "
        f"r={payload['rank']}  source={payload['source']}",
        flush=True,
    )
    if payload["tables_path"]:
        print(f"[domain-vol] tables={payload['tables_path']}", flush=True)
    print(
        f"[domain-vol] mean σ={payload['mean_posterior_std']:.4f}  "
        f"mean B0 span={payload['mean_k0_span']:.4f}  "
        f"σ/span={payload['mean_std_over_mean_k0_span']:.3f}",
        flush=True,
    )
    print(format_table(payload), flush=True)
    print(
        "[domain-vol] eq.cube is the side of a cube with the same volume; "
        "mean span for the ellipsoid rows is the enclosing AABB.",
        flush=True,
    )
    ell = next(row for row in payload["domains"] if row["kind"] == "ellipsoid")
    print(
        f"[domain-vol] ellipsoid max Mahalanobis of μ="
        f"{ell['max_mahalanobis']:.4f}  "
        f"means outside={ell['n_means_outside']}",
        flush=True,
    )

    json_path = os.path.abspath(args.json_out)
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[domain-vol] wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
