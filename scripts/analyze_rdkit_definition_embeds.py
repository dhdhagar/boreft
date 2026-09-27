#!/usr/bin/env python
"""Does Qwen cosine over RDKit 2D-property text track ``rdkit_sim``?

``--use-definition-embeds`` with ``--append-rdkit-definitions`` feeds the encoder
a labeled suffix such as ``2D properties: average molecular weight 337.443; ...``.
This script asks two things about that string, using synthetic descriptor vectors
so the only tokens that change are the ten numbers:

1. Does embedding cosine fall as production RBF ``rdkit_sim`` falls?
2. Does a held-constant MolT5/ChEBI sentence in front of the suffix drown it?

Pairs are built in robust-normalized space so a target similarity *s* means
``mean((z − z')²) = −2 ln(s)``, then snapped to the same int/float types
``rdkit_descriptor_values`` writes. Integer snapping would otherwise pull
low-*s* pairs back toward 1, so a residual correction on unbounded ``clogp``
restores the target distance. The x-axis is the *actual* RBF after that
correction, computed by ``rdkit_similarity_from_values`` — the same kernel as
``rdkit_sim`` on real SMILES. Identity pairs (left = right) always have
cosine 1 under L2-normalization; they are a sanity check and a scatter
anchor, and are excluded from Pearson/Spearman.

Two text conditions share every pair:

``rdkit_only``
    ``2D properties: ...``  (what ``--omit-molt5-definitions`` embeds, minus the
    SMILES-leaking definition template)
``molt5_plus_rdkit``
    ``{fixed ChEBI sentence} 2D properties: ...``  (what ``--append-rdkit-definitions``
    embeds when the prose is held constant)

The encoder is ``Qwen/Qwen3-Embedding-0.6B`` through sentence-transformers, L2-
normalized, matching the training ``embed_cache`` path.

    python scripts/analyze_rdkit_definition_embeds.py
    python scripts/analyze_rdkit_definition_embeds.py --device cuda
    sbatch scripts/analyze_rdkit_definition_embeds.sh
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from boreft.bo.plotting import save_plot  # noqa: E402
from boreft.chem import (  # noqa: E402
    RDKIT_DESCRIPTOR_DIM,
    RDKIT_DESCRIPTOR_MAP,
    default_rdkit_map_path,
    load_rdkit_descriptor_map,
    normalize_rdkit_values,
    rdkit_similarity_from_values,
)
from boreft.search_wandb import add_wandb_cli, maybe_log_named_analysis  # noqa: E402
from boreft.task_config import task_embedding_model  # noqa: E402
from boreft.text_similarity import (  # noqa: E402
    append_rdkit_definition_values,
    default_definitions_path,
)

TASK = "molopt"
DEFAULT_MODEL = task_embedding_model(TASK)
DEFAULT_TARGET_SIMS = (1.0, 0.9, 0.7, 0.5, 0.3, 0.1)
RDKIT_ONLY = "rdkit_only"
MOLT5_PLUS_RDKIT = "molt5_plus_rdkit"
CONDITIONS = (RDKIT_ONLY, MOLT5_PLUS_RDKIT)
# Unbounded continuous dim used to absorb integer-snapping residual so a
# constructed pair can still hit its target RBF. Looked up by name so a schema
# reorder cannot silently point the correction at a clipped count.
_CLOGP_INDEX = next(
    i for i, d in enumerate(RDKIT_DESCRIPTOR_MAP) if d["name"] == "clogp"
)

EncodeFn = Callable[[list[str]], np.ndarray]


@dataclass(frozen=True)
class SyntheticPair:
    left: tuple[float | int, ...]
    right: tuple[float | int, ...]
    target_rdkit_sim: float
    rdkit_sim: float


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic descriptors
# ─────────────────────────────────────────────────────────────────────────────


def coerce_descriptor_values(
    values: Sequence[float | int],
) -> tuple[float | int, ...]:
    """Snap a raw vector to the int/float types ``rdkit_descriptor_values`` writes."""
    if len(values) != RDKIT_DESCRIPTOR_DIM:
        raise ValueError(
            f"RDKit descriptor must have {RDKIT_DESCRIPTOR_DIM} values, "
            f"got {len(values)}"
        )
    out: list[float | int] = []
    for descriptor, value in zip(RDKIT_DESCRIPTOR_MAP, values):
        unit = descriptor.get("unit")
        x = float(value)
        if unit == "count":
            out.append(max(0, int(round(x))))
        elif unit == "elementary_charge":
            out.append(int(round(x)))
        elif unit == "fraction":
            out.append(round(float(np.clip(x, 0.0, 1.0)), 6))
        else:
            if descriptor["name"] in ("molecular_weight", "tpsa"):
                x = max(0.0, x)
            out.append(round(x, 6))
    return tuple(out)


def _median_scale(map_path: str) -> tuple[np.ndarray, np.ndarray]:
    normalization = load_rdkit_descriptor_map(map_path)["normalization"]
    return (
        np.asarray(normalization["median"], dtype=np.float64),
        np.asarray(normalization["scale"], dtype=np.float64),
    )


def values_from_z(
    z: np.ndarray, median: np.ndarray, scale: np.ndarray
) -> tuple[float | int, ...]:
    return coerce_descriptor_values(median + np.asarray(z, dtype=np.float64) * scale)


def sample_base_values(
    rng: np.random.Generator, median: np.ndarray, scale: np.ndarray
) -> tuple[float | int, ...]:
    return values_from_z(rng.normal(size=RDKIT_DESCRIPTOR_DIM), median, scale)


def _correct_clogp_to_target(
    base: Sequence[float | int],
    partner: Sequence[float | int],
    mean_sq: float,
    median: np.ndarray,
    scale: np.ndarray,
    *,
    map_path: str,
) -> tuple[float | int, ...]:
    """Move unbounded ``clogp`` so ``mean((Δz)²)`` matches ``mean_sq`` after snapping.

    Count dimensions round to ints, which usually *shortens* the z-offset and
    leaves actual RBF above the target (pairs look more similar than intended).
    Putting the residual on ``clogp`` restores the distance without re-snapping
    the other nine values. If integer snapping already overshot, ``clogp`` is
    held at the base value so it does not add more distance.
    """
    z_base = normalize_rdkit_values(base, map_path=map_path)
    z_partner = normalize_rdkit_values(partner, map_path=map_path)
    delta = z_partner - z_base
    others = float(np.sum(np.square(delta)) - delta[_CLOGP_INDEX] ** 2)
    need = float(RDKIT_DESCRIPTOR_DIM) * mean_sq - others
    sign = float(np.sign(delta[_CLOGP_INDEX]) or 1.0)
    delta[_CLOGP_INDEX] = sign * math.sqrt(need) if need > 0 else 0.0
    return values_from_z(z_base + delta, median, scale)


def partner_at_target_sim(
    base: Sequence[float | int],
    target_sim: float,
    rng: np.random.Generator,
    median: np.ndarray,
    scale: np.ndarray,
    *,
    map_path: str,
) -> tuple[float | int, ...]:
    """Offset ``base`` so ``mean((Δz)²)`` matches ``−2 ln(target_sim)``, then snap.

    Identity (``target_sim == 1``) returns ``base`` unchanged. Integer snapping
    can move the actual RBF off the target; a ``clogp`` residual restores it.
    Callers still recompute actual RBF after this returns.
    """
    if target_sim >= 1.0 - 1e-12:
        return tuple(base)
    if not (0.0 < target_sim < 1.0):
        raise ValueError(f"target_sim must be in (0, 1], got {target_sim}")
    mean_sq = -2.0 * math.log(target_sim)
    delta = math.sqrt(mean_sq)
    signs = rng.choice(np.array([-1.0, 1.0]), size=RDKIT_DESCRIPTOR_DIM)
    z = normalize_rdkit_values(base, map_path=map_path)
    partner = values_from_z(z + delta * signs, median, scale)
    return _correct_clogp_to_target(
        base, partner, mean_sq, median, scale, map_path=map_path
    )


def build_pairs(
    n_bases: int,
    target_sims: Sequence[float],
    seed: int,
    *,
    map_path: str,
) -> list[SyntheticPair]:
    if n_bases <= 0:
        raise ValueError("--n-bases must be a positive integer")
    if not target_sims:
        raise ValueError("at least one target rdkit_sim is required")
    for s in target_sims:
        if not (0.0 < float(s) <= 1.0):
            raise ValueError(f"target rdkit_sim must be in (0, 1], got {s}")
    rng = np.random.default_rng(seed)
    median, scale = _median_scale(map_path)
    pairs: list[SyntheticPair] = []
    for _ in range(n_bases):
        left = sample_base_values(rng, median, scale)
        for target in target_sims:
            right = partner_at_target_sim(
                left, float(target), rng, median, scale, map_path=map_path
            )
            pairs.append(
                SyntheticPair(
                    left=left,
                    right=right,
                    target_rdkit_sim=float(target),
                    rdkit_sim=rdkit_similarity_from_values(
                        left, right, map_path=map_path
                    ),
                )
            )
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Text treatments
# ─────────────────────────────────────────────────────────────────────────────


def format_definition(
    values: Sequence[float | int],
    molt5_definition: str = "",
) -> str:
    """Labeled 2D-property string, optionally prepended with a ChEBI sentence.

    Delegates to :func:`append_rdkit_definition_values` so the suffix matches
    training byte-for-byte.
    """
    key = "_"
    return append_rdkit_definition_values(
        key,
        molt5_definition,
        rdkit_lookup={key: tuple(values)},
        require_lookup=True,
        omit_base_definition=not str(molt5_definition).strip(),
    )


def pair_texts(
    pair: SyntheticPair, condition: str, molt5_definition: str
) -> tuple[str, str]:
    if condition == RDKIT_ONLY:
        prose = ""
    elif condition == MOLT5_PLUS_RDKIT:
        prose = molt5_definition
    else:
        raise ValueError(f"unknown condition {condition!r}")
    return format_definition(pair.left, prose), format_definition(pair.right, prose)


def load_molt5_definition(path: str, seed: int) -> str:
    rows: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            definition = str(row.get("definition") or "").strip()
            if definition:
                rows.append(definition)
    if not rows:
        raise ValueError(f"{path}: no non-empty ChEBI definitions found")
    return random.Random(seed).choice(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


def _l2_normalize(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norms, 1e-12)


def summarize(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def correlations(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return out
    out["pearson_r"] = float(np.corrcoef(x, y)[0, 1])
    try:
        from scipy.stats import spearmanr

        out["spearman_rho"] = float(spearmanr(x, y).statistic)
    except ImportError:
        pass
    return out


def _bin_by_target(
    pairs: Sequence[SyntheticPair], cosines: np.ndarray
) -> dict[str, dict]:
    bins: dict[float, list[tuple[float, float]]] = {}
    for pair, cosine in zip(pairs, cosines):
        bins.setdefault(pair.target_rdkit_sim, []).append((pair.rdkit_sim, float(cosine)))
    report: dict[str, dict] = {}
    for target in sorted(bins):
        actual, cos = zip(*bins[target])
        actual_arr = np.asarray(actual, dtype=np.float64)
        cos_arr = np.asarray(cos, dtype=np.float64)
        report[f"{target:g}"] = {
            "target_rdkit_sim": float(target),
            "n": int(cos_arr.size),
            "rdkit_sim": summarize(actual_arr),
            "embed_cosine": summarize(cos_arr),
        }
    return report


def condition_metrics(
    pairs: Sequence[SyntheticPair], cosines: np.ndarray
) -> dict:
    if len(cosines) != len(pairs):
        raise ValueError(
            f"cosine count {len(cosines)} does not match pair count {len(pairs)}"
        )
    if not pairs:
        raise ValueError("pairs must be non-empty")
    rdkit = np.asarray([p.rdkit_sim for p in pairs], dtype=np.float64)
    identity_mask = np.asarray(
        [p.left == p.right for p in pairs], dtype=bool
    )
    identity = cosines[identity_mask]
    farthest = min(p.target_rdkit_sim for p in pairs)
    far = np.asarray(
        [
            c
            for p, c in zip(pairs, cosines)
            if abs(p.target_rdkit_sim - farthest) < 1e-12
        ],
        dtype=np.float64,
    )
    gap = (
        float(identity.mean() - far.mean())
        if identity.size and far.size
        else float("nan")
    )
    out: dict = {
        "n_pairs": int(cosines.size),
        "rdkit_sim": summarize(rdkit),
        "embed_cosine": summarize(cosines),
        "identity_cosine": summarize(identity),
        "farthest_target_rdkit_sim": float(farthest),
        "farthest_cosine": summarize(far),
        "cosine_gap_identity_minus_farthest": gap,
        "by_target_sim": _bin_by_target(pairs, cosines),
        "_rdkit_sim": rdkit,
        "_cosine": cosines,
    }
    # (1, 1) identity points are guaranteed by L2-normalization of a repeated
    # string; including them inflates r without testing the encoder.
    varied = ~identity_mask
    out["n_correlation_pairs"] = int(varied.sum())
    out.update(correlations(rdkit[varied], cosines[varied]))
    return out


def run_experiment(
    pairs: Sequence[SyntheticPair],
    molt5_definition: str,
    encode_fn: EncodeFn,
) -> dict:
    """Embed both conditions over the same synthetic pairs.

    Distinct strings are encoded once. ``encode_fn`` may return unnormalized
    rows; they are L2-normalized here so cosine is a dot product either way.
    """
    if not pairs:
        raise ValueError("pairs must be non-empty")
    if not str(molt5_definition).strip():
        raise ValueError("molt5_definition must be a non-empty ChEBI-like sentence")

    texts: list[str] = []
    index: dict[str, int] = {}

    def slot(text: str) -> int:
        if text not in index:
            index[text] = len(texts)
            texts.append(text)
        return index[text]

    pair_slots: dict[str, list[tuple[int, int]]] = {c: [] for c in CONDITIONS}
    for pair in pairs:
        for condition in CONDITIONS:
            left, right = pair_texts(pair, condition, molt5_definition)
            pair_slots[condition].append((slot(left), slot(right)))
    prose_only_i = slot(molt5_definition.strip())

    print(f"[analyze] encoding {len(texts)} unique definition strings...", flush=True)
    emb = np.asarray(encode_fn(texts), dtype=np.float64)
    if emb.ndim != 2 or emb.shape[0] != len(texts):
        raise ValueError(
            f"encode_fn returned shape {tuple(emb.shape)} for {len(texts)} texts"
        )
    emb = _l2_normalize(emb)

    conditions: dict[str, dict] = {}
    for condition in CONDITIONS:
        cosines = np.asarray(
            [float(emb[i] @ emb[j]) for i, j in pair_slots[condition]],
            dtype=np.float64,
        )
        conditions[condition] = condition_metrics(pairs, cosines)

    only_gap = conditions[RDKIT_ONLY]["cosine_gap_identity_minus_farthest"]
    plus_gap = conditions[MOLT5_PLUS_RDKIT]["cosine_gap_identity_minus_farthest"]
    drowning = {
        "cosine_gap_rdkit_only": float(only_gap),
        "cosine_gap_molt5_plus_rdkit": float(plus_gap),
        "gap_ratio_molt5_over_rdkit_only": (
            float(plus_gap / only_gap) if abs(only_gap) > 1e-12 else float("nan")
        ),
    }
    prose_only = float(emb[prose_only_i] @ emb[prose_only_i])
    return {
        "n_pairs": len(pairs),
        "n_unique_texts": len(texts),
        "molt5_definition": molt5_definition.strip(),
        "prose_only_self_cosine": prose_only,
        "drowning": drowning,
        "conditions": conditions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Encoder / report
# ─────────────────────────────────────────────────────────────────────────────


def _torch_device(device: str):
    import torch

    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_encode_fn(model_name: str, device: str, batch_size: int) -> EncodeFn:
    from sentence_transformers import SentenceTransformer

    dev = _torch_device(device)
    print(f"[analyze] loading {model_name} on {dev}...", flush=True)
    model = SentenceTransformer(model_name, device=str(dev))

    def encode(texts: list[str]) -> np.ndarray:
        return np.asarray(
            model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float64,
        )

    return encode


def strip_arrays(obj):
    if isinstance(obj, dict):
        return {k: strip_arrays(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [strip_arrays(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def print_report(result: dict) -> None:
    print("\n" + "=" * 88)
    print("Qwen cosine of RDKit definition text vs. RBF rdkit_sim (synthetic pairs)")
    print("=" * 88)
    prose = result["molt5_definition"]
    preview = prose if len(prose) <= 88 else prose[:85] + "..."
    print(f"held-constant ChEBI sentence: {preview}")
    print(f"prose-only self-cosine (sanity, always 1): {result['prose_only_self_cosine']:.6f}")
    print(
        f"{'condition':<22}{'r':>8}{'rho':>8}{'cos@1':>10}{'cos@far':>10}{'gap':>10}"
    )
    for name in CONDITIONS:
        c = result["conditions"][name]
        ident = c["identity_cosine"]["mean"]
        far = c["farthest_cosine"]["mean"]
        print(
            f"{name:<22}"
            f"{c.get('pearson_r', float('nan')):>8.3f}"
            f"{c.get('spearman_rho', float('nan')):>8.3f}"
            f"{ident:>10.4f}"
            f"{far:>10.4f}"
            f"{c['cosine_gap_identity_minus_farthest']:>10.4f}"
        )
    d = result["drowning"]
    ratio = d["gap_ratio_molt5_over_rdkit_only"]
    print(
        f"\ngap ratio (molt5_plus_rdkit / rdkit_only) = {ratio:.3f}"
        "  — 1 means the prose does not shrink the property signal; "
        "0 means it drowns it."
    )
    print("\nmean cosine by target rdkit_sim:")
    print(f"{'target s':>10}{'actual s':>12}{RDKIT_ONLY:>14}{MOLT5_PLUS_RDKIT:>20}")
    only_bins = result["conditions"][RDKIT_ONLY]["by_target_sim"]
    plus_bins = result["conditions"][MOLT5_PLUS_RDKIT]["by_target_sim"]
    for key in only_bins:
        o, p = only_bins[key], plus_bins[key]
        print(
            f"{o['target_rdkit_sim']:>10.2f}"
            f"{o['rdkit_sim']['mean']:>12.3f}"
            f"{o['embed_cosine']['mean']:>14.4f}"
            f"{p['embed_cosine']['mean']:>20.4f}"
        )
    print("=" * 88 + "\n")


def _style(ax) -> None:
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def plot_agreement(result: dict, save_path: str) -> Optional[str]:
    plt = _plt()
    if plt is None:
        return None

    colors = {RDKIT_ONLY: "#B2182B", MOLT5_PLUS_RDKIT: "#2166AC"}
    labels = {
        RDKIT_ONLY: "2D properties only",
        MOLT5_PLUS_RDKIT: "ChEBI sentence + 2D properties",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    ax = axes[0]
    for name in CONDITIONS:
        c = result["conditions"][name]
        corr = []
        if "pearson_r" in c:
            corr.append(f"r = {c['pearson_r']:.2f}")
        if "spearman_rho" in c:
            corr.append(f"ρ = {c['spearman_rho']:.2f}")
        legend = labels[name] + (f"  ({', '.join(corr)})" if corr else "")
        ax.scatter(
            c["_rdkit_sim"],
            c["_cosine"],
            s=10,
            alpha=0.45,
            color=colors[name],
            label=legend,
        )
    ax.set_xlabel("RDKit descriptor similarity (RBF)", fontsize=10)
    ax.set_ylabel("Embedding cosine", fontsize=10)
    ax.set_title("Pairwise cosine vs. rdkit_sim", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    _style(ax)

    ax = axes[1]
    for name in CONDITIONS:
        bins = result["conditions"][name]["by_target_sim"]
        xs = [b["rdkit_sim"]["mean"] for b in bins.values()]
        ys = [b["embed_cosine"]["mean"] for b in bins.values()]
        yerr = [b["embed_cosine"]["std"] for b in bins.values()]
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            fmt="o-",
            color=colors[name],
            label=labels[name],
            capsize=3,
        )
    ax.set_xlabel("Mean actual rdkit_sim in bin", fontsize=10)
    ax.set_ylabel("Mean embedding cosine", fontsize=10)
    ax.set_title("Dose–response by target RBF", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    _style(ax)

    ratio = result["drowning"]["gap_ratio_molt5_over_rdkit_only"]
    fig.suptitle(
        "Does Qwen cosine over RDKit 2D-property text track rdkit_sim?\n"
        f"gap ratio (with ChEBI prose / without) = {ratio:.2f}",
        fontsize=12,
        y=1.05,
    )
    fig.tight_layout()
    save_plot(fig, save_path, dpi=200)
    plt.close(fig)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--n-bases",
        type=int,
        default=200,
        help="Synthetic base descriptor vectors. Each is paired at every target sim.",
    )
    p.add_argument(
        "--target-sims",
        type=float,
        nargs="+",
        default=list(DEFAULT_TARGET_SIMS),
        help="Target RBF rdkit_sim values used to build partners (actual sim is recorded after snapping).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--definitions",
        default=default_definitions_path(TASK),
        help="definitions.jsonl; one ChEBI sentence is sampled and held constant.",
    )
    p.add_argument(
        "--molt5-definition",
        default="",
        help="Override the held-constant ChEBI sentence (skips --definitions).",
    )
    p.add_argument(
        "--rdkit-map",
        default=default_rdkit_map_path(),
        help="Committed descriptor map (normalization statistics).",
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--device",
        default="auto",
        help="torch device (auto/cpu/cuda/mps).",
    )
    p.add_argument(
        "--out-dir",
        default=os.path.join("data", "molopt", "analysis"),
        help="Where the JSON report and figure are written.",
    )
    add_wandb_cli(p)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    map_path = os.path.abspath(args.rdkit_map)
    molt5 = str(args.molt5_definition).strip() or load_molt5_definition(
        args.definitions, args.seed
    )
    pairs = build_pairs(
        args.n_bases, args.target_sims, args.seed, map_path=map_path
    )
    print(
        f"[analyze] {len(pairs)} synthetic pairs "
        f"({args.n_bases} bases × {len(args.target_sims)} target sims)\n"
        f"[analyze] rdkit map : {map_path}\n"
        f"[analyze] model     : {args.model}"
    )
    encode_fn = load_encode_fn(args.model, args.device, args.batch_size)
    result = run_experiment(pairs, molt5, encode_fn)
    print_report(result)

    os.makedirs(args.out_dir, exist_ok=True)
    report = {
        "n_bases": int(args.n_bases),
        "target_sims": [float(s) for s in args.target_sims],
        "seed": int(args.seed),
        "rdkit_map": map_path,
        "model": args.model,
        "embedding_prompt_defn_not_applied": True,
        "text_templates": {
            RDKIT_ONLY: "2D properties: <stringify_rdkit_definition>",
            MOLT5_PLUS_RDKIT: (
                "<held-constant ChEBI sentence> 2D properties: "
                "<stringify_rdkit_definition>"
            ),
        },
        **strip_arrays(result),
        "pairs": {
            "target_rdkit_sim": [p.target_rdkit_sim for p in pairs],
            "rdkit_sim": [p.rdkit_sim for p in pairs],
            "cosine": {
                name: result["conditions"][name]["_cosine"].tolist()
                for name in CONDITIONS
            },
        },
    }
    json_path = os.path.join(args.out_dir, "rdkit_definition_embeds.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[analyze] wrote {json_path}")

    fig_path = plot_agreement(
        result, os.path.join(args.out_dir, "rdkit_definition_embeds.png")
    )
    if fig_path:
        print(f"[analyze] wrote {fig_path}")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
