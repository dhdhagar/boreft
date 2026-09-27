#!/usr/bin/env python3
"""Latent utilization and semantic–code Spearman on Semantle checkpoints.

For each training target \(s_i\):

  φ(s_i)  Qwen embedding (definition-decorated when the run used it)
  μ_i     learned posterior mean (bias table or word_mu)

Reports, matching notes/theory-notes/shared-encoder-bias-experiments.md:

  utilization  eRank({μ_i}) / min(n-1, r)   (centered posterior means)
  Spearman     ρ of pairwise 1-cos(φ_i, φ_j) vs ||μ_i-μ_j||_2

Default comparison is the shared encoder vs independent per-target codes.

    python scripts/analyze_semantle_code_geometry.py
    python scripts/analyze_semantle_code_geometry.py \\
        --condition canonical=outputs/1784053292 \\
        --condition noenc=outputs/1788894671
    python scripts/analyze_semantle_code_geometry.py \\
        --from-json data/semantle/analysis/code_geometry.json
    sbatch scripts/analyze_semantle_code_geometry.sh
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional, Sequence

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
SCRIPTS_ROOT = os.path.dirname(os.path.abspath(__file__))
for _path in (SRC_ROOT, SCRIPTS_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import analyze_semantle_train_embed_rank as ter  # noqa: E402
from boreft.bo.plotting import apply_bo_axes_style, bo_rc_params, save_plot  # noqa: E402
from boreft.search_wandb import add_wandb_cli, maybe_log_named_analysis  # noqa: E402
from boreft.task_config import task_embedding_model  # noqa: E402

TASK = "semantle"
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "data", "semantle", "analysis")
DEFAULT_CACHE_DIR = None
DEFAULT_BATCH_SIZE = 64
DEFAULT_SCATTER_PAIRS = 30000
EPS = 1e-12
JSON_NAME = "code_geometry.json"
SPECTRUM_PNG = "code_geometry_spectrum.png"
SCATTER_PNG = "code_geometry_scatter.png"

DEFAULT_CONDITIONS: tuple[dict[str, str], ...] = (
    {
        "name": "canonical",
        "wandb_id": "jn0knp44",
        "output_dir": os.path.join(REPO_ROOT, "outputs", "1784053292"),
    },
    {
        "name": "noenc",
        "wandb_id": "6yv75wzj",
        "output_dir": os.path.join(REPO_ROOT, "outputs", "1788894671"),
    },
)
CONDITION_LABELS = {
    "canonical": "shared encoder",
    "noenc": "w/o shared encoder",
}
CONDITION_COLORS = {
    "canonical": "#0072B2",
    "noenc": "#D55E00",
}


def condition_label(name: str) -> str:
    key = str(name or "").strip().lower()
    return CONDITION_LABELS.get(key, str(name))


def item_word(item: dict[str, Any]) -> str:
    return str(item.get("word") or item["target"]).strip()


def item_word_id(item: dict[str, Any]) -> int:
    if "id" not in item or item["id"] is None:
        raise ValueError(f"items.json row is missing id: {item!r}")
    return int(item["id"])


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, EPS)


def pairwise_semantic_distances(phi: np.ndarray) -> np.ndarray:
    """D^sem_ij = 1 - cos(φ_i, φ_j). Diagonal is 0."""
    zn = l2_normalize(np.asarray(phi, dtype=np.float64))
    cos = np.clip(zn @ zn.T, -1.0, 1.0)
    d = 1.0 - cos
    np.fill_diagonal(d, 0.0)
    return d


def pairwise_code_distances(mu: np.ndarray) -> np.ndarray:
    """D^code_ij = ||μ_i - μ_j||_2. Diagonal is 0."""
    x = np.asarray(mu, dtype=np.float64)
    sq = np.sum(x * x, axis=1, keepdims=True)
    d2 = sq + sq.T - 2.0 * (x @ x.T)
    d2 = np.maximum(d2, 0.0)
    np.fill_diagonal(d2, 0.0)
    return np.sqrt(d2)


def upper_tri(mat: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(mat.shape[0], k=1)
    return mat[i, j]


def spearman_geometry(phi: np.ndarray, mu: np.ndarray) -> dict[str, Any]:
    """Spearman ρ of all pairwise (D^sem, D^code)."""
    from scipy.stats import spearmanr

    if phi.shape[0] != mu.shape[0]:
        raise ValueError(
            f"phi and mu must have the same n, got {phi.shape[0]} vs {mu.shape[0]}"
        )
    n = int(phi.shape[0])
    if n < 3:
        raise ValueError(f"need at least 3 targets for pairwise Spearman, got {n}")
    d_sem = pairwise_semantic_distances(phi)
    d_code = pairwise_code_distances(mu)
    sem_vec = upper_tri(d_sem)
    code_vec = upper_tri(d_code)
    result = spearmanr(sem_vec, code_vec, nan_policy="omit")
    rho = getattr(result, "statistic", getattr(result, "correlation", np.nan))
    return {
        "spearman_rho": float(rho),
        "n_pairs": int(sem_vec.size),
        "d_sem_mean": float(sem_vec.mean()),
        "d_code_mean": float(code_vec.mean()),
        "d_sem": d_sem,
        "d_code": d_code,
    }


def utilization_from_mu(mu: np.ndarray) -> dict[str, Any]:
    """Centered Roy–Vetterli eRank of {μ_i} and latent utilization."""
    x = np.asarray(mu, dtype=np.float64)
    n, rank = int(x.shape[0]), int(x.shape[1])
    sigma = ter.singular_values(x, center=True)
    stats = ter.spectrum_stats(sigma)
    erank = float(stats["effective_rank"])
    denom = max(min(n - 1, rank), 1)
    return {
        "n": n,
        "rank": rank,
        "erank": erank,
        "utilization": float(erank / denom),
        "utilization_denom": int(denom),
        "stable_rank": float(stats["stable_rank"]),
        "participation_ratio": float(stats["participation_ratio"]),
        "sigma": [float(v) for v in sigma],
        "mu_norm_mean": float(np.linalg.norm(x, axis=1).mean()),
        "mu_norm_max": float(np.linalg.norm(x, axis=1).max()),
    }


def subsample_pairs(
    d_sem: np.ndarray,
    d_code: np.ndarray,
    n_pairs: int,
    seed: int,
) -> dict[str, list[float]]:
    vec_s = upper_tri(d_sem)
    vec_c = upper_tri(d_code)
    n = int(vec_s.size)
    take = min(int(n_pairs), n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=take, replace=False)
    idx.sort()
    return {
        "d_sem": [float(v) for v in vec_s[idx]],
        "d_code": [float(v) for v in vec_c[idx]],
        "n": take,
    }


def encode_target_phi(
    words: Sequence[str],
    saved: dict[str, Any],
    *,
    batch_size: int,
) -> np.ndarray:
    """Qwen φ(s_i) in the same space as RECON/DIST, definition-decorated if trained so."""
    from boreft.text_similarity import (
        encode_reference_embeddings,
        training_cache_params,
    )

    use_def = bool(saved.get("use_definition_embeds"))
    _, definition_lookup, _ = training_cache_params(
        use_definition_embeds=use_def,
        task=saved.get("task") or TASK,
        words=list(words),
        definitions_path=saved.get("definitions_path"),
    )
    return encode_reference_embeddings(
        list(words),
        batch_size=batch_size,
        definition_lookup=definition_lookup if use_def else None,
        task=saved.get("task") or TASK,
    )


def run_condition(
    *,
    name: str,
    output_dir: str,
    wandb_id: str,
    cache_dir: Optional[str],
    batch_size: int,
    scatter_pairs: int,
    seed: int,
) -> dict[str, Any]:
    from boreft.bias_tables import stack_bias_vectors
    from boreft.data_utils import load_merged_run_config
    from boreft.eval.semantle import load_eval_checkpoint, release_eval_checkpoint

    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    saved = load_merged_run_config(output_dir)
    model_name = saved.get("model_name")
    if not model_name:
        raise ValueError(f"{output_dir}: training_config.json is missing model_name")
    layer = int(saved.get("layer", 13))
    rank = int(saved.get("low_rank_dim", 64))
    print(
        f"[code_geometry] [{name}] load {output_dir}  "
        f"model={model_name} layer={layer} rank={rank} "
        f"add_bias_network={bool(saved.get('add_bias_network'))}",
        flush=True,
    )
    ckpt = load_eval_checkpoint(
        output_dir,
        model_name,
        layer,
        rank,
        cache_dir or saved.get("cache_dir") or DEFAULT_CACHE_DIR,
    )
    try:
        items = list(ckpt.items)
        words = [item_word(it) for it in items]
        word_ids = [item_word_id(it) for it in items]
        mu = stack_bias_vectors(ckpt.reft_model, word_ids)
        if mu.shape[0] != len(words):
            raise ValueError(
                f"{output_dir}: stacked μ has {mu.shape[0]} rows, items has {len(words)}"
            )
        print(
            f"[code_geometry] [{name}] μ {tuple(mu.shape)}  "
            f"encode φ for {len(words)} targets",
            flush=True,
        )
        phi = encode_target_phi(words, saved, batch_size=batch_size)
        if phi.shape[0] != mu.shape[0]:
            raise ValueError(
                f"{output_dir}: φ has {phi.shape[0]} rows, μ has {mu.shape[0]}"
            )
        util = utilization_from_mu(mu)
        geom = spearman_geometry(phi, mu)
        scatter = subsample_pairs(
            geom["d_sem"], geom["d_code"], scatter_pairs, seed
        )
        print(
            f"[code_geometry] [{name}] erank={util['erank']:.3f}  "
            f"utilization={util['utilization']:.3f}  "
            f"Spearman ρ={geom['spearman_rho']:.3f}  "
            f"n={util['n']} r={util['rank']}",
            flush=True,
        )
        return {
            "name": name,
            "wandb_id": wandb_id,
            "output_dir": output_dir,
            "add_bias_network": bool(saved.get("add_bias_network")),
            "use_definition_embeds": bool(saved.get("use_definition_embeds")),
            "bias_type": saved.get("bias_type"),
            "lambda_sdpo": saved.get("lambda_sdpo"),
            "lambda_ce": saved.get("lambda_ce"),
            "n": util["n"],
            "rank": util["rank"],
            "erank": util["erank"],
            "utilization": util["utilization"],
            "utilization_denom": util["utilization_denom"],
            "stable_rank": util["stable_rank"],
            "participation_ratio": util["participation_ratio"],
            "sigma": util["sigma"],
            "mu_norm_mean": util["mu_norm_mean"],
            "mu_norm_max": util["mu_norm_max"],
            "spearman_rho": geom["spearman_rho"],
            "n_pairs": geom["n_pairs"],
            "d_sem_mean": geom["d_sem_mean"],
            "d_code_mean": geom["d_code_mean"],
            "embed_dim": int(phi.shape[1]),
            "embed_model": task_embedding_model(saved.get("task") or TASK),
            "scatter": scatter,
        }
    finally:
        release_eval_checkpoint(ckpt)


def plot_spectra(conditions: Sequence[dict[str, Any]], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(bo_rc_params()):
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for row in conditions:
            sigma = np.asarray(row.get("sigma") or [], dtype=np.float64)
            if sigma.size == 0:
                continue
            ys = sigma / max(float(sigma[0]), EPS)
            xs = np.arange(1, sigma.size + 1)
            ax.plot(
                xs,
                ys,
                color=CONDITION_COLORS.get(row["name"], "0.3"),
                label=condition_label(row["name"]),
                lw=1.8,
            )
        ax.set_xlabel("singular value index")
        ax.set_ylabel(r"$\sigma_k / \sigma_1$")
        ax.set_title("Posterior-mean spectrum")
        apply_bo_axes_style(ax)
        ax.legend(frameon=False)
        fig.tight_layout()
        save_plot(fig, path, dpi=200)
        plt.close(fig)


def plot_scatter(conditions: Sequence[dict[str, Any]], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(conditions)
    if n == 0 or not any((row.get("scatter") or {}).get("d_sem") for row in conditions):
        return
    with plt.rc_context(bo_rc_params()):
        fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.8), squeeze=False)
        for ax, row in zip(axes[0], conditions):
            scatter = row.get("scatter") or {}
            xs = scatter.get("d_sem") or []
            ys = scatter.get("d_code") or []
            ax.scatter(
                xs,
                ys,
                s=6,
                alpha=0.25,
                c=CONDITION_COLORS.get(row["name"], "0.3"),
                linewidths=0,
            )
            rho = row.get("spearman_rho")
            ax.set_xlabel(r"$1-\cos\phi_i,\phi_j$")
            ax.set_ylabel(r"$\|\mu_i-\mu_j\|_2$")
            title = condition_label(row["name"])
            if rho is not None:
                title = rf"{title}  ($\rho={float(rho):.3f}$)"
            ax.set_title(title)
            apply_bo_axes_style(ax)
        fig.tight_layout()
        save_plot(fig, path, dpi=200)
        plt.close(fig)


def condition_for_json(row: dict[str, Any]) -> dict[str, Any]:
    """Drop bulky scatter from the on-disk JSON (plots already consumed it)."""
    out = dict(row)
    scatter = out.get("scatter")
    if isinstance(scatter, dict):
        out["scatter_n"] = int(scatter.get("n") or 0)
        del out["scatter"]
    return out


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
        help="name=output_dir (repeatable). Default: canonical, noenc.",
    )
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--scatter-pairs",
        type=int,
        default=DEFAULT_SCATTER_PAIRS,
        help="Random pairwise subsample for the scatter plot (full Spearman uses all pairs).",
    )
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument(
        "--from-json",
        default=None,
        help="Replot from an existing code_geometry.json (no checkpoint load).",
    )
    add_wandb_cli(p)
    return p.parse_args(argv)


def write_outputs(
    payload: dict[str, Any],
    conditions_with_scatter: Sequence[dict[str, Any]],
    out_dir: str,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, JSON_NAME)
    spectrum_path = os.path.join(out_dir, SPECTRUM_PNG)
    scatter_path = os.path.join(out_dir, SCATTER_PNG)
    plot_spectra(conditions_with_scatter, spectrum_path)
    plot_scatter(conditions_with_scatter, scatter_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"[code_geometry] wrote {json_path}", flush=True)
    print(f"[code_geometry] wrote {spectrum_path}", flush=True)
    print(f"[code_geometry] wrote {scatter_path}", flush=True)
    return json_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.from_json:
        with open(args.from_json, encoding="utf-8") as f:
            payload = json.load(f)
        conditions = list(payload.get("conditions") or [])
        write_outputs(payload, conditions, args.out_dir)
        maybe_log_named_analysis(
            args.out_dir,
            os.path.join(args.out_dir, JSON_NAME),
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            group=args.wandb_group,
            name=args.wandb_run_name,
            wandb_dir=args.wandb_dir,
            no_wandb=args.no_wandb,
        )
        return

    specs = args.condition or [dict(row) for row in DEFAULT_CONDITIONS]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        rows.append(
            run_condition(
                name=spec["name"],
                output_dir=spec["output_dir"],
                wandb_id=spec.get("wandb_id") or "",
                cache_dir=args.cache_dir,
                batch_size=args.batch_size,
                scatter_pairs=args.scatter_pairs,
                seed=args.seed,
            )
        )

    payload = {
        "task": TASK,
        "analysis": "code_geometry",
        "embed_model": task_embedding_model(TASK),
        "n_conditions": len(rows),
        "conditions": [condition_for_json(row) for row in rows],
    }
    json_path = write_outputs(payload, rows, args.out_dir)
    maybe_log_named_analysis(
        args.out_dir,
        json_path,
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        group=args.wandb_group,
        name=args.wandb_run_name,
        wandb_dir=args.wandb_dir,
        no_wandb=args.no_wandb,
    )


if __name__ == "__main__":
    main()
