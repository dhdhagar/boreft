#!/usr/bin/env python
"""Which molopt text representation clusters by ChEBI category under Qwen?

The molopt encoder is a general text model (``Qwen/Qwen3-Embedding-0.6B``), so the
string we feed it *is* the representation. This script asks, for every eligible
ChEBI-20 molecule (or a seeded sample), which of seven strings lands
nearest-neighbours of the same ``category_normalized`` together:

    smiles          bare SMILES
    defn            bare ChEBI description
    smiles_defn     production ``embedding_prompt_defn`` (SMILES in the template)
    iupac           RDKit systematic name (InChI) from the parsed SMILES
    iupac_defn      same template as smiles_defn, InChI in the molecule slot
    rdkit           labeled ``2D properties: ...`` suffix (``--omit-molt5-definitions``)
    smiles_rdkit    ``{SMILES} 2D properties: ...``

Metrics, all on the same molecule set, using ``category_normalized`` as labels
(``unknown`` dropped from the sample):

    knn_purity@k    leave-one-out fraction of k nearest neighbours sharing the label
    kmeans_purity   k-means (k = #labels) vs. the category labels
    silhouette      cosine silhouette of the given labels

A 2-D PCA scatter per variant, coloured by category, is the visual counterpart.

    python scripts/analyze_molopt_text_variants.py
    python scripts/analyze_molopt_text_variants.py --n 1000
    python scripts/analyze_molopt_text_variants.py --device cuda
    sbatch scripts/analyze_molopt_text_variants.sh
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from boreft.bo.plotting import save_plot  # noqa: E402
from boreft.search_wandb import add_wandb_cli, maybe_log_named_analysis  # noqa: E402
from boreft.task_config import definition_embedding_text, task_embedding_model  # noqa: E402
from boreft.text_similarity import (  # noqa: E402
    append_rdkit_definition_values,
    default_definitions_path,
    default_rdkit_definitions_path,
    definitions_row_target,
    load_rdkit_definition_lookup,
)

TASK = "molopt"
DEFAULT_MODEL = task_embedding_model(TASK)
DEFAULT_N = 0  # 0 = every eligible molecule
KNN_KS = (5, 10)
KMEANS_CHANCE_TRIALS = 20

EncodeFn = Callable[[list[str]], np.ndarray]
IupacFn = Callable[[str], Optional[str]]
_META_CLUSTERS = frozenset({"multi-cluster", "unknown"})
_MARKERS = ("o", "s", "^", "D", "v", "P", "*", "X", "h", "<", ">", "p")
_EDGE_COLORS = ("black", "0.25", "0.55", "white")


# ─────────────────────────────────────────────────────────────────────────────
# Variants
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Molecule:
    smiles: str
    definition: str
    category: str
    category_normalized: str
    rdkit_values: Optional[tuple[float | int, ...]] = None
    iupac: Optional[str] = None


@dataclass(frozen=True)
class Variant:
    name: str
    question: str
    needs_iupac: bool = False


VARIANTS: tuple[Variant, ...] = (
    Variant("smiles", "Bare SMILES — what embed_sim encodes without the prompt wrap."),
    Variant("defn", "Bare ChEBI description — the prose the category labels came from."),
    Variant(
        "smiles_defn",
        "Production embedding_prompt_defn: The molecule '{SMILES}' is: {definition}.",
    ),
    Variant(
        "iupac",
        "RDKit systematic name: InChI of the parsed SMILES "
        "(RDKit has no IUPAC namer).",
        needs_iupac=True,
    ),
    Variant(
        "iupac_defn",
        "Same definition template as smiles_defn, with the RDKit InChI in the molecule slot.",
        needs_iupac=True,
    ),
    Variant(
        "rdkit",
        "Labeled 2D-property suffix only, matching --omit-molt5-definitions.",
    ),
    Variant(
        "smiles_rdkit",
        "Bare SMILES followed by the labeled 2D-property suffix.",
    ),
)
VARIANT_BY_NAME = {v.name: v for v in VARIANTS}


def variant_text(mol: Molecule, name: str) -> str:
    """Return the string this variant embeds for ``mol``."""
    if name == "smiles":
        return mol.smiles.strip()
    if name == "defn":
        return mol.definition.strip()
    if name == "smiles_defn":
        return definition_embedding_text(TASK, mol.smiles, mol.definition)
    if name == "iupac":
        if not mol.iupac:
            raise ValueError(f"IUPAC name missing for {mol.smiles!r}")
        return mol.iupac.strip()
    if name == "iupac_defn":
        if not mol.iupac:
            raise ValueError(f"IUPAC name missing for {mol.smiles!r}")
        return definition_embedding_text(TASK, mol.iupac, mol.definition)
    if name == "rdkit":
        return append_rdkit_definition_values(
            mol.smiles,
            mol.definition,
            rdkit_lookup=_rdkit_lookup_for(mol),
            require_lookup=True,
            omit_base_definition=True,
        )
    if name == "smiles_rdkit":
        return append_rdkit_definition_values(
            mol.smiles,
            mol.smiles.strip(),
            rdkit_lookup=_rdkit_lookup_for(mol),
            require_lookup=True,
            omit_base_definition=False,
        )
    raise ValueError(f"unknown variant {name!r}")


def _rdkit_lookup_for(mol: Molecule) -> dict[str, tuple[float | int, ...]]:
    if mol.rdkit_values is None:
        raise ValueError(f"RDKit descriptors missing for {mol.smiles!r}")
    return {mol.smiles: mol.rdkit_values}


# ─────────────────────────────────────────────────────────────────────────────
# Systematic names
#
# RDKit has no IUPAC namer. The identifier it *can* compute from a parsed
# molecule is InChI (``Chem.MolToInchi``). The corpus SMILES are valid, so
# parse-and-name is a pure function with no model, API, or cache.
# ─────────────────────────────────────────────────────────────────────────────


def rdkit_iupac(smiles: str) -> str:
    """InChI for ``smiles`` via RDKit. Corpus molecules are valid SMILES."""
    from rdkit import Chem

    from boreft.chem import unwrap_smiles_tags

    mol = Chem.MolFromSmiles(unwrap_smiles_tags(smiles).strip())
    if mol is None:
        raise ValueError(f"unparseable SMILES {smiles!r}")
    name = Chem.MolToInchi(mol)
    if not name:
        raise ValueError(f"RDKit produced empty InChI for {smiles!r}")
    return name


def assign_iupac_names(
    molecules: Sequence[Molecule], fn: IupacFn
) -> None:
    for mol in molecules:
        mol.iupac = fn(mol.smiles)
        if not mol.iupac:
            raise ValueError(f"empty systematic name for {mol.smiles!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────


def load_molecules(
    definitions_path: str, rdkit_path: str
) -> list[Molecule]:
    rdkit_lookup = load_rdkit_definition_lookup(rdkit_path)
    rows: list[Molecule] = []
    with open(definitions_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            smiles = definitions_row_target(row)
            definition = str(row.get("definition") or "").strip()
            if not smiles or not definition:
                continue
            values = rdkit_lookup.get(smiles)
            rows.append(
                Molecule(
                    smiles=smiles,
                    definition=definition,
                    category=str(row.get("category") or "unknown").strip() or "unknown",
                    category_normalized=str(
                        row.get("category_normalized") or row.get("category") or "unknown"
                    ).strip()
                    or "unknown",
                    rdkit_values=values,
                )
            )
    if not rows:
        raise ValueError(f"{definitions_path}: no (target, definition) pairs found")
    return rows


def label_of(mol: Molecule, field: str) -> str:
    if field == "category":
        return mol.category
    if field == "category_normalized":
        return mol.category_normalized
    raise ValueError(f"unknown label field {field!r}")


def eligible_molecules(
    rows: Sequence[Molecule],
    *,
    label_field: str,
    drop_unknown: bool = True,
) -> list[Molecule]:
    """Molecules that can form every variant string (have RDKit values)."""
    pool = [m for m in rows if m.rdkit_values is not None]
    if drop_unknown:
        pool = [m for m in pool if label_of(m, label_field) not in _META_CLUSTERS]
    if not pool:
        raise ValueError("no molecules left after dropping unknown / missing RDKit rows")
    return pool


def sample_molecules(
    rows: Sequence[Molecule],
    n: int,
    seed: int,
    *,
    label_field: str,
    drop_unknown: bool = True,
) -> list[Molecule]:
    """Seeded sample. Missing RDKit rows are always dropped; unknown labels optionally."""
    if n < 0:
        raise ValueError(f"--n must be >= 0, got {n}")
    pool = eligible_molecules(
        rows, label_field=label_field, drop_unknown=drop_unknown
    )
    rnd = random.Random(seed)
    rnd.shuffle(pool)
    if n and n < len(pool):
        return pool[:n]
    return pool


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


def cluster_purity(true_labels: Sequence[str], pred_labels: Sequence[str]) -> float:
    """Standard cluster purity: each predicted cluster contributes its majority class."""
    if len(true_labels) != len(pred_labels) or not true_labels:
        raise ValueError("cluster_purity needs equal-length non-empty label lists")
    groups: dict[object, list[str]] = defaultdict(list)
    for true, pred in zip(true_labels, pred_labels):
        groups[pred].append(true)
    majority = sum(Counter(members).most_common(1)[0][1] for members in groups.values())
    return float(majority) / float(len(true_labels))


def chance_knn_purity(labels: Sequence[str]) -> float:
    """Expected leave-one-out k-NN purity under random neighbours (any k)."""
    n = len(labels)
    if n < 2:
        return float("nan")
    counts = Counter(labels)
    return float(sum(c * (c - 1) for c in counts.values()) / (n * (n - 1)))


def knn_purity(emb: np.ndarray, labels: Sequence[str], k: int) -> float:
    """Leave-one-out fraction of k nearest neighbours (cosine) sharing the label."""
    n = emb.shape[0]
    if n < 2 or k < 1:
        return float("nan")
    k = min(k, n - 1)
    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)
    # argpartition is enough: purity does not care about neighbour order.
    neighbours = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    hits = 0
    for i, nbrs in enumerate(neighbours):
        hits += sum(1 for j in nbrs if labels[j] == labels[i])
    return float(hits) / float(n * k)


def kmeans_purity(
    emb: np.ndarray, labels: Sequence[str], seed: int
) -> tuple[float, np.ndarray]:
    from sklearn.cluster import KMeans

    n_clusters = len(set(labels))
    if n_clusters < 2 or emb.shape[0] < n_clusters:
        return float("nan"), np.full(emb.shape[0], -1)
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    pred = model.fit_predict(emb)
    return cluster_purity(labels, [str(p) for p in pred]), pred


def random_kmeans_purity(
    labels: Sequence[str], n_clusters: int, seed: int, n_trials: int = KMEANS_CHANCE_TRIALS
) -> float:
    """Purity of random assignments into ``n_clusters`` equal-opportunity bins."""
    if n_clusters < 1 or not labels:
        return float("nan")
    rng = np.random.default_rng(seed)
    n = len(labels)
    scores = [
        cluster_purity(labels, [str(x) for x in rng.integers(0, n_clusters, size=n)])
        for _ in range(n_trials)
    ]
    return float(np.mean(scores))


def cosine_silhouette(emb: np.ndarray, labels: Sequence[str]) -> dict[str, object]:
    from sklearn.metrics import silhouette_samples, silhouette_score

    y = np.asarray(labels)
    counts = Counter(y.tolist())
    keep = np.array([counts[label] >= 2 for label in y])
    X = emb[keep]
    y_kept = y[keep]
    n_labels = len(set(y_kept.tolist()))
    if n_labels < 2 or X.shape[0] < 2:
        return {
            "overall": float("nan"),
            "per_cluster": {},
            "n_used": int(X.shape[0]),
            "n_dropped_singleton": int((~keep).sum()),
        }
    sample = silhouette_samples(X, y_kept, metric="cosine")
    per_cluster = {
        c: float(sample[y_kept == c].mean()) for c in sorted(set(y_kept.tolist()))
    }
    return {
        "overall": float(silhouette_score(X, y_kept, metric="cosine")),
        "per_cluster": per_cluster,
        "n_used": int(X.shape[0]),
        "n_dropped_singleton": int((~keep).sum()),
    }


def majority_class_fraction(labels: Sequence[str]) -> float:
    if not labels:
        return float("nan")
    return float(Counter(labels).most_common(1)[0][1]) / float(len(labels))


def label_counts(labels: Sequence[str]) -> dict[str, int]:
    return dict(Counter(labels).most_common())


def score_variant(
    emb: np.ndarray, labels: Sequence[str], seed: int
) -> dict[str, object]:
    knn = {f"knn_purity_at_{k}": knn_purity(emb, labels, k) for k in KNN_KS}
    km_purity, pred = kmeans_purity(emb, labels, seed)
    n_clusters = len(set(labels))
    sil = cosine_silhouette(emb, labels)
    extras: dict[str, object] = {}
    if pred.size and not np.all(pred < 0):
        try:
            from sklearn.metrics import (
                adjusted_rand_score,
                normalized_mutual_info_score,
            )

            extras["kmeans_nmi"] = float(normalized_mutual_info_score(labels, pred))
            extras["kmeans_ari"] = float(adjusted_rand_score(labels, pred))
        except ImportError:
            pass
    return {
        "n": int(emb.shape[0]),
        "dim": int(emb.shape[1]),
        "n_labels": n_clusters,
        "chance_knn_purity": chance_knn_purity(labels),
        "chance_kmeans_purity": random_kmeans_purity(labels, n_clusters, seed),
        "majority_class_fraction": majority_class_fraction(labels),
        "kmeans_purity": km_purity,
        "silhouette_cosine": sil["overall"],
        "silhouette": sil,
        **knn,
        **extras,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Encoding
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


def _l2_normalize(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norms, 1e-12)


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


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────


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


def _pc_axis_label(pca, component: int) -> str:
    return f"PC{component + 1} ({100 * pca.explained_variance_ratio_[component]:.1f}% var)"


def _distinct_colors(n: int) -> list[str]:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    colors: list[str] = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(cmap_name)
        for i in range(cmap.N):
            colors.append(mcolors.to_hex(cmap(i)))
        if len(colors) >= n:
            break
    if len(colors) < n:
        raise ValueError(f"need {n} distinct colors but only built {len(colors)}")
    return colors[:n]


def build_cluster_style_map(clusters: Sequence[str]) -> dict[str, dict[str, str]]:
    real = [c for c in clusters if c not in _META_CLUSTERS]
    colors = _distinct_colors(len(real)) if real else []
    style_map: dict[str, dict[str, str]] = {}
    for i, cluster in enumerate(real):
        style_map[cluster] = {
            "color": colors[i],
            "marker": _MARKERS[i % len(_MARKERS)],
            "edgecolor": _EDGE_COLORS[(i // len(_MARKERS)) % len(_EDGE_COLORS)],
        }
    for cluster in clusters:
        if cluster == "unknown":
            style_map[cluster] = {
                "color": "#d9d9d9",
                "marker": "o",
                "edgecolor": "0.45",
            }
        elif cluster == "multi-cluster":
            style_map[cluster] = {
                "color": "#808080",
                "marker": "o",
                "edgecolor": "black",
            }
    return style_map


def _scatter_by_cluster(
    ax,
    xy: np.ndarray,
    labels: Sequence[str],
    unique_clusters: Sequence[str],
    style_map: dict[str, dict[str, str]],
    point_size: float,
) -> None:
    for cluster in unique_clusters:
        mask = [i for i, c in enumerate(labels) if c == cluster]
        if not mask:
            continue
        style = style_map[cluster]
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            c=style["color"],
            marker=style["marker"],
            s=point_size,
            alpha=0.85,
            label=cluster,
            edgecolors=style["edgecolor"],
            linewidths=0.7,
        )


def _legend_handles(unique_clusters: Sequence[str], style_map: dict[str, dict[str, str]]):
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0],
            [0],
            marker=style_map[cluster]["marker"],
            color="w",
            markerfacecolor=style_map[cluster]["color"],
            markeredgecolor=style_map[cluster]["edgecolor"],
            markeredgewidth=0.8,
            markersize=8,
            label=cluster,
        )
        for cluster in unique_clusters
        if cluster in style_map
    ]


def plot_variant_pca(
    emb: np.ndarray,
    labels: Sequence[str],
    title: str,
    save_path: str,
    style_map: dict,
    unique_clusters: Sequence[str],
) -> Optional[str]:
    plt = _plt()
    if plt is None:
        return None
    from sklearn.decomposition import PCA

    n_components = min(2, emb.shape[0], emb.shape[1])
    pca = PCA(n_components=n_components)
    xy = pca.fit_transform(emb)
    if xy.ndim == 1:
        xy = np.column_stack([xy, np.zeros_like(xy)])
    elif xy.shape[1] == 1:
        xy = np.column_stack([xy[:, 0], np.zeros(xy.shape[0])])

    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    _scatter_by_cluster(ax, xy, labels, unique_clusters, style_map, 28.0)
    ax.set_xlabel(_pc_axis_label(pca, 0), fontsize=10)
    ax.set_ylabel(_pc_axis_label(pca, 1) if n_components > 1 else "PC2", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(
        handles=_legend_handles(unique_clusters, style_map),
        title="Category",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=7,
        title_fontsize=8,
        framealpha=0.9,
        ncol=2 if len(unique_clusters) > 14 else 1,
    )
    _style(ax)
    fig.tight_layout()
    save_plot(fig, save_path, dpi=160)
    plt.close(fig)
    return save_path


def plot_pca_grid(
    embeddings: dict[str, np.ndarray],
    labels: Sequence[str],
    unique_clusters: Sequence[str],
    style_map: dict,
    save_path: str,
) -> Optional[str]:
    plt = _plt()
    if plt is None or not embeddings:
        return None
    from sklearn.decomposition import PCA

    names = list(embeddings)
    cols = 4
    rows = (len(names) + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.4 * cols, 3.8 * rows), squeeze=False
    )
    for ax, name in zip(axes.ravel(), names):
        emb = embeddings[name]
        n_components = min(2, emb.shape[0], emb.shape[1])
        pca = PCA(n_components=n_components)
        xy = pca.fit_transform(emb)
        if xy.ndim == 1:
            xy = np.column_stack([xy, np.zeros_like(xy)])
        elif xy.shape[1] == 1:
            xy = np.column_stack([xy[:, 0], np.zeros(xy.shape[0])])
        _scatter_by_cluster(ax, xy, labels, unique_clusters, style_map, 12.0)
        var = 100 * float(pca.explained_variance_ratio_[: min(2, n_components)].sum())
        ax.set_title(f"{name}  ({var:.0f}% var)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        _style(ax)
    for ax in axes.ravel()[len(names) :]:
        ax.set_visible(False)
    fig.legend(
        handles=_legend_handles(unique_clusters, style_map),
        title="Category",
        loc="lower center",
        ncol=min(8, max(1, len(unique_clusters))),
        fontsize=7,
        title_fontsize=8,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Qwen embeddings of molopt text variants, 2-D PCA, coloured by category",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    save_plot(fig, save_path, dpi=160)
    plt.close(fig)
    return save_path


def plot_purity(
    scores: dict[str, dict], save_path: str
) -> Optional[str]:
    plt = _plt()
    if plt is None or not scores:
        return None

    names = list(scores)
    metrics = [
        ("knn_purity_at_5", "k-NN purity @5"),
        ("knn_purity_at_10", "k-NN purity @10"),
        ("kmeans_purity", "k-means purity"),
        ("silhouette_cosine", "cosine silhouette"),
    ]
    chance_knn = next(iter(scores.values())).get("chance_knn_purity")
    chance_km = next(iter(scores.values())).get("chance_kmeans_purity")

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.6))
    xs = np.arange(len(names))
    for ax, (key, title) in zip(np.atleast_1d(axes), metrics):
        values = [float(scores[n].get(key, float("nan"))) for n in names]
        ax.bar(xs, values, color="#2166AC", width=0.72)
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0.0, 1.05 if key != "silhouette_cosine" else None)
        if key.startswith("knn") and chance_knn is not None:
            ax.axhline(
                float(chance_knn),
                color="0.2",
                ls="--",
                lw=1.2,
                label=f"chance = {float(chance_knn):.3f}",
            )
            ax.legend(fontsize=7, loc="upper right")
        if key == "kmeans_purity" and chance_km is not None:
            ax.axhline(
                float(chance_km),
                color="0.2",
                ls="--",
                lw=1.2,
                label=f"chance = {float(chance_km):.3f}",
            )
            ax.legend(fontsize=7, loc="upper right")
        if key == "silhouette_cosine":
            ax.axhline(0.0, color="0.5", ls=":", lw=1.0)
        _style(ax)
    fig.suptitle(
        "Does the text variant cluster by ChEBI category under Qwen?",
        fontsize=13,
        y=1.03,
    )
    fig.tight_layout()
    save_plot(fig, save_path, dpi=180)
    plt.close(fig)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


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


def print_report(
    scores: dict[str, dict],
    labels: Sequence[str],
    examples: dict[str, str],
) -> None:
    print("\n" + "=" * 96)
    print("Qwen text-variant category clustering on ChEBI-20")
    print("=" * 96)
    print(f"  n = {len(labels)}   labels = {len(set(labels))}")
    print(f"  label counts: {label_counts(labels)}")
    print("\nexample molecule, one string per variant:")
    for name, text in examples.items():
        preview = text if len(text) <= 110 else text[:107] + "..."
        print(f"  {name:<14} {preview}")

    header = (
        f"\n{'variant':<14}{'kNN@5':>8}{'kNN@10':>8}{'k-purity':>10}"
        f"{'NMI':>8}{'ARI':>8}{'sil':>8}"
    )
    print(header)
    print("-" * len(header.strip("\n")))
    for name, s in scores.items():
        print(
            f"{name:<14}"
            f"{s.get('knn_purity_at_5', float('nan')):>8.4f}"
            f"{s.get('knn_purity_at_10', float('nan')):>8.4f}"
            f"{s.get('kmeans_purity', float('nan')):>10.4f}"
            f"{s.get('kmeans_nmi', float('nan')):>8.4f}"
            f"{s.get('kmeans_ari', float('nan')):>8.4f}"
            f"{s.get('silhouette_cosine', float('nan')):>8.4f}"
        )
    chance_knn = next(iter(scores.values())).get("chance_knn_purity", float("nan"))
    chance_km = next(iter(scores.values())).get("chance_kmeans_purity", float("nan"))
    majority = next(iter(scores.values())).get("majority_class_fraction", float("nan"))
    print(
        f"\nchance k-NN purity = {chance_knn:.4f}    "
        f"chance k-means purity = {chance_km:.4f}    "
        f"majority class = {majority:.4f}"
    )
    print("=" * 96 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--definitions",
        default=default_definitions_path(TASK),
        help="definitions.jsonl with category / category_normalized fields.",
    )
    p.add_argument(
        "--rdkit-definitions",
        default=default_rdkit_definitions_path(),
        help="definitions_rdkit.jsonl (ten-d descriptor sidecar).",
    )
    p.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        help="Sample size. 0 uses every eligible molecule "
        "(unknown labels dropped; rows without RDKit values always dropped).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--label-field",
        default="category_normalized",
        choices=("category_normalized", "category"),
        help="Which definitions.jsonl field supplies the cluster labels.",
    )
    p.add_argument(
        "--keep-unknown",
        action="store_true",
        help="Keep molecules whose label is 'unknown' / 'multi-cluster'.",
    )
    p.add_argument(
        "--variants",
        nargs="*",
        default=[v.name for v in VARIANTS],
        choices=[v.name for v in VARIANTS],
        help="Subset of variants to run. Default: all seven.",
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--out-dir",
        default=os.path.join("data", "molopt", "analysis"),
    )
    add_wandb_cli(p)
    return p.parse_args(argv)


def selected_variants(names: Sequence[str]) -> list[Variant]:
    if not names:
        names = [v.name for v in VARIANTS]
    chosen = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        chosen.append(VARIANT_BY_NAME[name])
    return chosen


def run(
    args: argparse.Namespace,
    encode: Optional[EncodeFn] = None,
    iupac_fn: Optional[IupacFn] = None,
) -> dict:
    variants = selected_variants(args.variants)
    needs_iupac = any(v.needs_iupac for v in variants)
    if args.n < 0:
        raise ValueError(f"--n must be >= 0, got {args.n}")

    print(
        f"[analyze] loading molecules from {args.definitions}",
        flush=True,
    )
    rows = load_molecules(args.definitions, args.rdkit_definitions)
    pool = eligible_molecules(
        rows,
        label_field=args.label_field,
        drop_unknown=not args.keep_unknown,
    )
    rnd = random.Random(args.seed)
    rnd.shuffle(pool)
    molecules = pool[: args.n] if args.n and args.n < len(pool) else pool

    if needs_iupac:
        fn = iupac_fn or rdkit_iupac
        print(f"[analyze] computing RDKit InChI for {len(molecules)} molecules...", flush=True)
        assign_iupac_names(molecules, fn)

    print(
        f"[analyze] using {len(molecules)} molecules "
        f"(requested n={args.n}, seed={args.seed}, label={args.label_field})",
        flush=True,
    )

    labels = [label_of(m, args.label_field) for m in molecules]
    examples = {v.name: variant_text(molecules[0], v.name) for v in variants}
    print("[analyze] example strings (first molecule):", flush=True)
    for name, text in examples.items():
        preview = text if len(text) <= 140 else text[:137] + "..."
        print(f"    {name}: {preview}", flush=True)

    if encode is None:
        encode = load_encode_fn(args.model, args.device, args.batch_size)

    embeddings: dict[str, np.ndarray] = {}
    scores: dict[str, dict] = {}
    for variant in variants:
        texts = [variant_text(m, variant.name) for m in molecules]
        print(
            f"[analyze] encoding {variant.name} ({len(texts)} strings)...",
            flush=True,
        )
        emb = _l2_normalize(np.asarray(encode(texts), dtype=np.float64))
        embeddings[variant.name] = emb
        scores[variant.name] = score_variant(emb, labels, args.seed)
        scores[variant.name]["question"] = variant.question

    print_report(scores, labels, examples)

    os.makedirs(args.out_dir, exist_ok=True)
    report = {
        "definitions_path": os.path.abspath(args.definitions),
        "rdkit_definitions_path": os.path.abspath(args.rdkit_definitions),
        "n_requested": args.n,
        "n": len(molecules),
        "seed": args.seed,
        "model": args.model,
        "label_field": args.label_field,
        "label_counts": label_counts(labels),
        "variant_questions": {v.name: v.question for v in variants},
        "examples": examples,
        "scores": strip_arrays(scores),
        "smiles": [m.smiles for m in molecules],
        "iupac": [m.iupac for m in molecules] if needs_iupac else None,
        "labels": labels,
    }
    json_path = os.path.join(args.out_dir, "text_variants.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[analyze] wrote {json_path}")

    unique_clusters = sorted(set(labels))
    style_map = build_cluster_style_map(unique_clusters)
    written: list[str] = [json_path]
    for name, emb in embeddings.items():
        path = plot_variant_pca(
            emb,
            labels,
            title=f"{name}  —  Qwen 2-D PCA, coloured by {args.label_field}",
            save_path=os.path.join(args.out_dir, f"text_variants_pca_{name}.png"),
            style_map=style_map,
            unique_clusters=unique_clusters,
        )
        if path:
            written.append(path)
    grid = plot_pca_grid(
        embeddings,
        labels,
        unique_clusters,
        style_map,
        os.path.join(args.out_dir, "text_variants_pca.png"),
    )
    if grid:
        written.append(grid)
    bars = plot_purity(
        scores, os.path.join(args.out_dir, "text_variants_purity.png")
    )
    if bars:
        written.append(bars)
    for path in written[1:]:
        print(f"[analyze] wrote {path}")
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not os.path.isfile(args.definitions):
        print(
            f"ERROR: {args.definitions} not found — run "
            f"scripts/prepare_molopt_chebi20.py --download first",
            file=sys.stderr,
        )
        return 1
    if not os.path.isfile(args.rdkit_definitions):
        print(f"ERROR: {args.rdkit_definitions} not found", file=sys.stderr)
        return 1

    run(args)
    maybe_log_named_analysis(
        args.out_dir,
        os.path.join(args.out_dir, "text_variants.json"),
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
