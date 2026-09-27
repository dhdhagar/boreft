"""End-of-training eval suite: RECON / RECON_TEST / GENZ / LIPZ / DIST metrics.

Pure metric functions plus GENZ test-set construction. No model generation
happens here — callers pass already-decoded text and target sets:

  - ``eval.semantle.run_semantle_generation_eval`` computes RECON / RECON_TEST /
    GENZ / DIST from its train-target and test-target decodes and Sobol decodes.
  - ``eval.run_full_eval`` computes LIPZ by aggregating per-pair interpolation
    trajectory summaries produced by ``eval.interpolate.run_interpolation``.

Metric keys are emitted without the group prefix (e.g. ``embed_sim``,
``sobol_unseen_greedy``); the orchestrator namespaces them as ``recon/``,
``recon_test/``, ``genz/``, ``lipz/``, ``dist/`` for WandB.
"""

from __future__ import annotations

import csv
import os
import random
from collections import Counter
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from boreft.chem import DEFAULT_RDKIT_SIM_TAU
from boreft.task_config import (
    task_supports_fingerprints,
    task_supports_validity,
    task_target_kind,
)
from boreft.text_similarity import encode_texts_normalized

# Shared temperature settings for every temperature-sampling pass in the suite.
EVAL_TEMPERATURES: tuple[float, ...] = (1.0, 1.5)

# Flag defaults (mirrored by TrainConfig / FullEvalConfig).
DEFAULT_EMBED_SIM_TAU = 0.8
DEFAULT_TEST_N_SAMPLES = 512
DEFAULT_BBOX_PCA_VAR = 0.9

# "Structurally similar" cut-off for Morgan/Tanimoto, the TFS counterpart of
# DEFAULT_EMBED_SIM_TAU. 0.4 is the conventional ECFP4 threshold, and it is much
# lower than the embedding tau because Tanimoto occupies a far wider range: two
# unrelated drug-like molecules sit near 0.1, whereas encoder cosines between the
# same pair rarely drop below 0.7.
DEFAULT_TFS_TAU = 0.4


def _text_key(s: str) -> str:
    """Whitespace-collapsed, lower-cased text for exact-match comparisons."""
    return " ".join(str(s).strip().lower().split())


def target_normalizer(task: str = "semantle") -> Callable[[str], str]:
    """Exact-match key function for the task's target kind.

    Text targets compare case-insensitively, since capitalization carries no
    meaning for a word. SMILES cannot be folded that way (``C`` is an aliphatic
    carbon, ``c`` an aromatic one) and have many equivalent spellings, so they
    compare by RDKit canonical form (see :func:`boreft.chem.canonical_target_key`).
    Lower-casing is therefore reached only for text tasks.
    """
    if task_target_kind(task) == "smiles":
        from boreft.chem import canonical_target_key

        return canonical_target_key
    return _text_key


def normalize_text(s: str, *, task: str = "semantle") -> str:
    """Exact-match key for one target or decode (see :func:`target_normalizer`).

    Pass ``task`` from any call site that can run under molopt: the default
    lower-cases, which silently merges distinct molecules.
    """
    return target_normalizer(task)(s)


def validity_metrics(
    decodes: Sequence[str], *, task: str, suffix: str
) -> dict[str, float]:
    """``{f"validity_{suffix}": rate}`` for tasks whose targets can be invalid.

    Empty for tasks without a validity notion (e.g. Semantle words), so the key
    simply never appears on those runs.
    """
    if not task_supports_validity(task):
        return {}
    from boreft.chem import validity_rate

    return {f"validity_{suffix}": validity_rate(list(decodes))}


def temp_key(t: float) -> str:
    """WandB-friendly suffix for a temperature (1.0 -> 'temp1.0')."""
    return f"temp{t:.1f}"


# ─────────────────────────────────────────────────────────────────────────────
# TFS — Morgan/Tanimoto structural similarity (molecule tasks only)
# ─────────────────────────────────────────────────────────────────────────────
#
# Every TFS helper returns an empty dict for tasks without fingerprints, so the
# keys simply never appear on Semantle runs — the same convention as
# ``validity_metrics``. TFS is computed here rather than passed in (unlike
# ``embed_sim``, which needs a model): it is a pure function of the target/decode
# strings, and RDKit fingerprints are memoized in :mod:`boreft.chem`.


def tfs_per_target(
    targets: Sequence[str], decodes: Sequence[str], *, task: str
) -> Optional[np.ndarray]:
    """Per-target Morgan/Tanimoto similarity, or ``None`` for non-molecule tasks."""
    if not task_supports_fingerprints(task):
        return None
    from boreft.chem import tanimoto_sim_per_text

    return tanimoto_sim_per_text(list(targets), list(decodes))


def tfs_metrics(
    targets: Sequence[str],
    decodes: Sequence[str],
    *,
    task: str,
    tau: float = DEFAULT_TFS_TAU,
) -> dict[str, float]:
    """``{tfs, tfs_gte_tau, tfs_tau}`` — the structural mirror of the embed_sim keys."""
    sims = tfs_per_target(targets, decodes, task=task)
    if sims is None:
        return {}
    return {
        "tfs": float(sims.mean()) if sims.size else 0.0,
        "tfs_gte_tau": float(np.mean(sims >= tau)) if sims.size else 0.0,
        "tfs_tau": float(tau),
    }


def tfs_best_of_bag_metrics(
    targets: Sequence[str],
    sample_bags: Sequence[Sequence[str]],
    *,
    task: str,
    suffix: str,
) -> dict[str, float]:
    """``{f"tfs_best_at_n_{suffix}": rate}`` — best TFS per sample bag, averaged.

    The structural analogue of ``recall_at_n``: instead of asking whether the exact
    target appeared in ``n`` samples, it asks how structurally close the best of
    those ``n`` samples got.
    """
    if not task_supports_fingerprints(task):
        return {}
    from boreft.chem import tanimoto_similarity

    bests: list[float] = []
    for target, bag in zip(targets, sample_bags):
        if not bag:
            continue
        bests.append(max(tanimoto_similarity(target, s) for s in bag))
    return {f"tfs_best_at_n_{suffix}": float(np.mean(bests)) if bests else 0.0}


# Internal diversity is quadratic in the number of decodes, and GENZ hands it the
# flattened Sobol × samples list (tens of thousands of strings at default settings).
# Estimating it from a uniform subsample keeps the cost flat; sampling *with*
# multiplicity means a collapsed model still scores near 0.
TFS_INTDIV_MAX_N = 1000
TFS_INTDIV_SEED = 0


def tfs_diversity_metrics(
    decodes: Sequence[str], *, task: str, suffix: str
) -> dict[str, float]:
    """``{f"tfs_intdiv_{suffix}": 1 - mean pairwise Tanimoto}`` (GuacaMol IntDiv).

    Structural counterpart of :func:`semantic_dispersion`. Omitted (rather than
    zero) when fewer than two decodes parse, so a run that generated only junk does
    not report a diversity number. Estimated from at most :data:`TFS_INTDIV_MAX_N`
    decodes (see above).
    """
    if not task_supports_fingerprints(task):
        return {}
    from boreft.chem import tanimoto_internal_diversity

    pool = list(decodes)
    if len(pool) > TFS_INTDIV_MAX_N:
        pool = random.Random(TFS_INTDIV_SEED).sample(pool, TFS_INTDIV_MAX_N)
    intdiv = tanimoto_internal_diversity(pool)
    if intdiv is None:
        return {}
    return {f"tfs_intdiv_{suffix}": float(intdiv)}


# ─────────────────────────────────────────────────────────────────────────────
# RDKit descriptor-vector similarity (molecule tasks only)
# ─────────────────────────────────────────────────────────────────────────────


def rdkit_sim_per_target(
    targets: Sequence[str],
    decodes: Sequence[str],
    *,
    task: str,
    map_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Per-target robust-scaled descriptor RBF similarity."""
    if not task_supports_fingerprints(task):
        return None
    from boreft.chem import rdkit_sim_per_text

    return rdkit_sim_per_text(list(targets), list(decodes), map_path=map_path)


def rdkit_sim_metrics(
    targets: Sequence[str],
    decodes: Sequence[str],
    *,
    task: str,
    tau: float = DEFAULT_RDKIT_SIM_TAU,
    map_path: Optional[str] = None,
) -> dict[str, float]:
    sims = rdkit_sim_per_target(
        targets, decodes, task=task, map_path=map_path
    )
    if sims is None:
        return {}
    return {
        "rdkit_sim": float(sims.mean()) if sims.size else 0.0,
        "rdkit_sim_gte_tau": float(np.mean(sims >= tau)) if sims.size else 0.0,
        "rdkit_sim_tau": float(tau),
    }


def rdkit_sim_best_of_bag_metrics(
    targets: Sequence[str],
    sample_bags: Sequence[Sequence[str]],
    *,
    task: str,
    suffix: str,
    map_path: Optional[str] = None,
) -> dict[str, float]:
    if not task_supports_fingerprints(task):
        return {}
    from boreft.chem import rdkit_similarity

    bests: list[float] = []
    for target, bag in zip(targets, sample_bags):
        if not bag:
            continue
        bests.append(
            max(rdkit_similarity(target, sample, map_path=map_path) for sample in bag)
        )
    return {
        f"rdkit_sim_best_at_n_{suffix}": (
            float(np.mean(bests)) if bests else 0.0
        )
    }


def rdkit_sim_diversity_metrics(
    decodes: Sequence[str],
    *,
    task: str,
    suffix: str,
    map_path: Optional[str] = None,
) -> dict[str, float]:
    if not task_supports_fingerprints(task):
        return {}
    from boreft.chem import rdkit_internal_diversity

    pool = list(decodes)
    if len(pool) > TFS_INTDIV_MAX_N:
        pool = random.Random(TFS_INTDIV_SEED).sample(pool, TFS_INTDIV_MAX_N)
    intdiv = rdkit_internal_diversity(pool, map_path=map_path)
    if intdiv is None:
        return {}
    return {f"rdkit_sim_intdiv_{suffix}": float(intdiv)}


# ─────────────────────────────────────────────────────────────────────────────
# SMILES edit distance (molecule tasks only)
# ─────────────────────────────────────────────────────────────────────────────
#
# Character Levenshtein on canonical SMILES when both sides parse, else on the
# raw unwrapped strings. Lower is better. Emitted only for SMILES tasks, same
# convention as TFS / RDKit sim.


def edit_dist_per_target(
    targets: Sequence[str],
    decodes: Sequence[str],
    *,
    task: str,
) -> Optional[np.ndarray]:
    """Per-target SMILES edit distance, or ``None`` for non-molecule tasks."""
    if task_target_kind(task) != "smiles":
        return None
    from boreft.chem import smiles_edit_dist_per_text

    return smiles_edit_dist_per_text(list(targets), list(decodes))


def edit_dist_metrics(
    targets: Sequence[str],
    decodes: Sequence[str],
    *,
    task: str,
) -> dict[str, float]:
    dists = edit_dist_per_target(targets, decodes, task=task)
    if dists is None:
        return {}
    return {
        "edit_dist": float(dists.mean()) if dists.size else 0.0,
    }


def edit_dist_best_of_bag_metrics(
    targets: Sequence[str],
    sample_bags: Sequence[Sequence[str]],
    *,
    task: str,
    suffix: str,
) -> dict[str, float]:
    """``{f"edit_dist_best_at_n_{suffix}": mean}`` — closest sample in each bag.

    The edit-distance analogue of ``rdkit_sim_best_at_n_*``: min distance over
    the bag, then averaged over targets.
    """
    if task_target_kind(task) != "smiles":
        return {}
    from boreft.chem import smiles_edit_distance

    bests: list[float] = []
    for target, bag in zip(targets, sample_bags):
        if not bag:
            continue
        bests.append(min(smiles_edit_distance(target, sample) for sample in bag))
    return {
        f"edit_dist_best_at_n_{suffix}": (
            float(np.mean(bests)) if bests else 0.0
        )
    }


# ─────────────────────────────────────────────────────────────────────────────
# RECON — reconstruction of the training targets
# ─────────────────────────────────────────────────────────────────────────────


def recon_metrics(
    *,
    targets: Sequence[str],
    greedy_decodes: Sequence[str],
    greedy_embed_sims: np.ndarray,
    temp_results: dict[float, list[dict]],
    tau: float,
    task: str = "semantle",
    tfs_tau: float = DEFAULT_TFS_TAU,
    rdkit_sim_tau: float = DEFAULT_RDKIT_SIM_TAU,
    rdkit_map_path: Optional[str] = None,
) -> dict:
    """RECON metrics from greedy (b=μ) decodes and per-temperature sample bags.

    Args:
        targets:           train targets (eval subset).
        greedy_decodes:    greedy decode per target, parallel to ``targets``.
        greedy_embed_sims: per-target greedy embed_sim, parallel to ``targets``.
        temp_results:      ``{temperature: [engine_result_per_target]}`` where each
                           result dict carries the raw decode list under ``samples``.
        tau:               threshold for ``embed_sim_gte_tau``.
        task:              selects the exact-match key (see :func:`target_normalizer`)
                           and whether validity / TFS are reported.
        tfs_tau:           threshold for ``tfs_gte_tau`` (molecule tasks only).
    """
    n = len(targets)
    norm = target_normalizer(task)
    norm_targets = [norm(t) for t in targets]
    sims = np.asarray(greedy_embed_sims, dtype=np.float64)

    out: dict = {
        "embed_sim": float(sims.mean()) if n else 0.0,
        "embed_sim_gte_tau": float(np.mean(sims >= tau)) if n else 0.0,
        "embed_sim_tau": float(tau),
        "recall_greedy": (
            float(
                np.mean(
                    [norm(g) == nt for g, nt in zip(greedy_decodes, norm_targets)]
                )
            )
            if n
            else 0.0
        ),
    }
    out.update(tfs_metrics(targets, greedy_decodes, task=task, tau=tfs_tau))
    out.update(
        rdkit_sim_metrics(
            targets,
            greedy_decodes,
            task=task,
            tau=rdkit_sim_tau,
            map_path=rdkit_map_path,
        )
    )
    out.update(edit_dist_metrics(targets, greedy_decodes, task=task))
    out.update(validity_metrics(greedy_decodes, task=task, suffix="greedy"))

    for t, results in temp_results.items():
        hits = 0
        for nt, r in zip(norm_targets, results):
            if nt in {norm(x) for x in r["samples"]}:
                hits += 1
        out[f"recall_at_n_{temp_key(t)}"] = float(hits / n) if n else 0.0
        out.update(
            tfs_best_of_bag_metrics(
                targets,
                [r["samples"] for r in results],
                task=task,
                suffix=temp_key(t),
            )
        )
        out.update(
            rdkit_sim_best_of_bag_metrics(
                targets,
                [r["samples"] for r in results],
                task=task,
                suffix=temp_key(t),
                map_path=rdkit_map_path,
            )
        )
        out.update(
            edit_dist_best_of_bag_metrics(
                targets,
                [r["samples"] for r in results],
                task=task,
                suffix=temp_key(t),
            )
        )
        out.update(
            validity_metrics(
                [x for r in results for x in r["samples"]],
                task=task,
                suffix=temp_key(t),
            )
        )

    return out


def _subset_indices(
    full_targets: Sequence[str], subset: Sequence[str], *, task: str = "semantle"
) -> list[int]:
    """Indices into ``full_targets`` for each target in ``subset`` (normalized match)."""
    norm = target_normalizer(task)
    pos = {norm(w): i for i, w in enumerate(full_targets)}
    return [pos[norm(w)] for w in subset if norm(w) in pos]


def recon_test_metrics(
    *,
    full_targets: Sequence[str],
    interp_targets: Sequence[str],
    extrap_targets: Sequence[str],
    greedy_decodes: Sequence[str],
    greedy_embed_sims: np.ndarray,
    temp_results: dict[float, list[dict]],
    tau: float,
    task: str = "semantle",
    tfs_tau: float = DEFAULT_TFS_TAU,
    rdkit_sim_tau: float = DEFAULT_RDKIT_SIM_TAU,
    rdkit_map_path: Optional[str] = None,
) -> dict:
    """RECON metrics on held-out test targets plus interp/extrap slices.

    Full-test keys match :func:`recon_metrics`; interp/extrap keys are prefixed
    (e.g. ``interp_recall_greedy``). The threshold-metadata keys (``embed_sim_tau``,
    ``tfs_tau``) are emitted once.
    """
    out = recon_metrics(
        targets=full_targets,
        greedy_decodes=greedy_decodes,
        greedy_embed_sims=greedy_embed_sims,
        temp_results=temp_results,
        tau=tau,
        task=task,
        tfs_tau=tfs_tau,
        rdkit_sim_tau=rdkit_sim_tau,
        rdkit_map_path=rdkit_map_path,
    )

    for label, subset in (("interp", interp_targets), ("extrap", extrap_targets)):
        if not subset:
            continue
        idx = _subset_indices(full_targets, subset, task=task)
        if not idx:
            continue
        sub_targets = [full_targets[i] for i in idx]
        sub_greedy = [greedy_decodes[i] for i in idx]
        sub_sims = np.asarray(greedy_embed_sims, dtype=np.float64)[idx]
        sub_temp = {
            t: [results[i] for i in idx] for t, results in temp_results.items()
        }
        sub = recon_metrics(
            targets=sub_targets,
            greedy_decodes=sub_greedy,
            greedy_embed_sims=sub_sims,
            temp_results=sub_temp,
            tau=tau,
            task=task,
            tfs_tau=tfs_tau,
            rdkit_sim_tau=rdkit_sim_tau,
            rdkit_map_path=rdkit_map_path,
        )
        for k, v in sub.items():
            if k in ("embed_sim_tau", "tfs_tau", "rdkit_sim_tau"):
                continue
            out[f"{label}_{k}"] = v

    return out


# ─────────────────────────────────────────────────────────────────────────────
# DIST — output distribution per learned point
# ─────────────────────────────────────────────────────────────────────────────


def dist_metrics(
    *,
    targets: Sequence[str],
    temp_results: dict[float, list[dict]],
    task: str = "semantle",
    rdkit_map_path: Optional[str] = None,
) -> dict:
    """DIST metrics: modal-output match and sample similarity spread per temp.

    Reuses the raw-text sample→target similarities already computed by the
    sampling engine (``per_sample_sims`` aligned to ``unique_samples``); no
    re-encoding happens here. ``embed_sim_mean``/``embed_sim_std`` are computed
    over all samples (multiplicity-weighted) then averaged over targets. The
    ``notarget`` variants first remove exact target matches from each sample bag;
    target-only bags contribute 0 mean/std.

    On molecule tasks the same four aggregates are also reported over Morgan/Tanimoto
    similarity as ``tfs_*``. Those *are* computed here (fingerprints need no model,
    so the sampling engine has no reason to carry them).
    """
    out: dict = {}
    norm = target_normalizer(task)
    norm_targets = [norm(t) for t in targets]
    want_molecule_sims = task_supports_fingerprints(task)
    families = (
        ("embed_sim", "tfs", "rdkit_sim")
        if want_molecule_sims
        else ("embed_sim",)
    )
    if want_molecule_sims:
        from boreft.chem import rdkit_similarity, tanimoto_similarity

    for t, results in temp_results.items():
        mode_hits: list[float] = []
        agg: dict[str, list[float]] = {
            f"{fam}{part}": []
            for fam in families
            for part in ("_mean", "_std", "_notarget", "_notarget_std")
        }

        for raw_tgt, ntgt, r in zip(targets, norm_targets, results):
            samples = r["samples"]
            if not samples:
                continue
            counts = Counter(norm(x) for x in samples)
            mode_hits.append(1.0 if counts.most_common(1)[0][0] == ntgt else 0.0)

            notarget_samples = [x for x in samples if norm(x) != ntgt]
            sim_lookups = {
                "embed_sim": dict(zip(r["unique_samples"], r["per_sample_sims"]))
            }
            if want_molecule_sims:
                sim_lookups["tfs"] = {
                    x: tanimoto_similarity(raw_tgt, x) for x in r["unique_samples"]
                }
                sim_lookups["rdkit_sim"] = {
                    x: rdkit_similarity(raw_tgt, x, map_path=rdkit_map_path)
                    for x in r["unique_samples"]
                }

            for fam, sim_of in sim_lookups.items():
                all_sims = np.asarray([sim_of[x] for x in samples], dtype=np.float64)
                agg[f"{fam}_mean"].append(float(all_sims.mean()))
                agg[f"{fam}_std"].append(
                    float(all_sims.std()) if len(samples) > 1 else 0.0
                )
                if notarget_samples:
                    nt_sims = np.asarray(
                        [sim_of[x] for x in notarget_samples], dtype=np.float64
                    )
                    agg[f"{fam}_notarget"].append(float(nt_sims.mean()))
                    agg[f"{fam}_notarget_std"].append(
                        float(nt_sims.std()) if len(notarget_samples) > 1 else 0.0
                    )
                else:
                    agg[f"{fam}_notarget"].append(0.0)
                    agg[f"{fam}_notarget_std"].append(0.0)

        out[f"mode_is_target_{temp_key(t)}"] = (
            float(np.mean(mode_hits)) if mode_hits else 0.0
        )
        for name, vals in agg.items():
            out[f"{name}_{temp_key(t)}"] = float(np.mean(vals)) if vals else 0.0

    return out


# ─────────────────────────────────────────────────────────────────────────────
# GENZ — generalization (test-set construction + recovery from Sobol points)
# ─────────────────────────────────────────────────────────────────────────────


def _read_csv_words_sorted(path: str) -> list[str]:
    """Words from a Semantle CSV, ordered by descending similarity."""
    rows: list[tuple[str, float]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((row["Word"].strip(), float(row["Similarity"])))
    rows.sort(key=lambda x: -x[1])
    return [w for w, _ in rows]


#  Config key holding each task's data CSV path(s) (``--semantle-csv`` is a tuple,
#  ``--molopt-csv`` / ``--hypogen-csv`` a single path).
_TASK_CSV_CONFIG_KEY = {
    "semantle": "semantle_csv",
    "molopt": "molopt_csv",
    "hypogen": "hypogen_csv",
}


def task_csv_paths(
    saved_cfg: Mapping[str, object], *, task: Optional[str] = None
) -> list[str]:
    """Data CSV paths for the checkpoint's task, read from a saved run config.

    These are the files :func:`build_test_sets` reads in full to build the
    held-out pool, so they must be the same ones training loaded from.
    """
    resolved_task = str(task or saved_cfg.get("task", "semantle"))
    key = _TASK_CSV_CONFIG_KEY.get(resolved_task)
    raw = saved_cfg.get(key) if key else None
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(p) for p in raw]


def read_pool_csv_targets(path: str, *, task: str = "semantle") -> list[str]:
    """Every candidate target in one of the task's data CSVs.

    Semantle CSVs are ``Word,Similarity``; molopt CSVs carry a ``smiles`` column;
    hypogen CSVs carry a ``hypothesis`` column in file order.
    """
    if task_target_kind(task) == "smiles":
        from boreft.data.molopt import read_csv_smiles

        return read_csv_smiles(path)
    if task == "hypogen":
        from boreft.data.hypogen import read_csv_hypotheses

        return read_csv_hypotheses(path)
    return _read_csv_words_sorted(path)


def build_non_train_pool(
    csv_paths: Optional[Sequence[str]],
    train_targets: Sequence[str],
    *,
    task: str = "semantle",
) -> list[str]:
    """Targets in the training CSVs that are not in the trained vocabulary.

    Unions every target from each CSV (the full file, not a top-k head) and
    returns those absent from ``train_targets`` (the targets the checkpoint was
    actually trained on). Returns the original strings, de-duplicated by
    normalized form.
    """
    norm = target_normalizer(task)
    train_norm = {norm(t) for t in train_targets}
    pool: dict[str, str] = {}
    for path in csv_paths or []:
        if not path or not os.path.isfile(path):
            continue
        for target in read_pool_csv_targets(path, task=task):
            key = norm(target)
            if key in train_norm or key in pool:
                continue
            pool[key] = target
    return list(pool.values())


def split_interp_extrap(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    test_words: Sequence[str],
    *,
    pca_var: float,
) -> tuple[list[str], list[str]]:
    """Split test words by whether they fall inside the train PCA bounding box.

    PCA is fit on the train target embeddings (components retained to explain
    ``pca_var`` of variance). A test word is ``interp`` when its projection lies
    within the per-dimension min/max box of the projected train embeddings,
    otherwise ``extrap``.
    """
    if len(train_embeddings) < 2 or len(test_words) == 0:
        return [], list(test_words)

    n_train, n_dim = train_embeddings.shape
    max_components = min(n_train, n_dim)
    # sklearn interprets 0<n_components<1 as a variance ratio, but only when it
    # does not exceed the rank; clamp to be safe for small train sets.
    n_components: float | int = pca_var
    if not (0.0 < pca_var < 1.0):
        n_components = min(int(pca_var) if pca_var >= 1 else 1, max_components)

    pca = PCA(n_components=n_components, svd_solver="full")
    train_proj = pca.fit_transform(train_embeddings)
    test_proj = pca.transform(test_embeddings)

    lo = train_proj.min(axis=0)
    hi = train_proj.max(axis=0)
    inside = np.all((test_proj >= lo) & (test_proj <= hi), axis=1)

    interp = [w for w, ins in zip(test_words, inside) if ins]
    extrap = [w for w, ins in zip(test_words, inside) if not ins]
    return interp, extrap


def build_test_sets(
    *,
    train_targets: Sequence[str],
    csv_paths: Optional[Sequence[str]],
    test_n_samples: int,
    seed: int,
    pca_var: float,
    task: str = "semantle",
    train_embeddings: Optional[np.ndarray] = None,
    pool: Optional[Sequence[str]] = None,
    pool_source: Optional[str] = None,
) -> tuple[list[str], list[str], dict]:
    """Sample a non-train test set and split it into interp / extrap subsets.

    ``csv_paths`` are the task's data CSVs (``--semantle-csv`` / ``--molopt-csv`` /
    ``--hypogen-csv``), read in full so rows past the training head become the
    held-out pool. Pass ``pool`` to use an explicit candidate list (molopt
    high-tail test molecules) instead of the CSV complement.
    ``train_embeddings`` may supply the pre-encoded train-target embeddings (rows
    aligned to ``train_targets``) to avoid recomputing them; when ``None`` they
    are encoded here.
    """
    if pool is None:
        pool = build_non_train_pool(csv_paths, train_targets, task=task)
        source = "non_train_csv"
    else:
        pool = [str(s) for s in pool if str(s).strip()]
        source = pool_source or "explicit"
        norm = target_normalizer(task)
        train_norm = {norm(t) for t in train_targets}
        pool = [s for s in pool if norm(s) not in train_norm]
    meta: dict = {
        "test_n_samples_requested": test_n_samples,
        "n_pool": len(pool),
        "pca_var": pca_var,
        "seed": seed,
        "n_train_targets": len(train_targets),
        "pool_source": source,
    }
    if not pool:
        meta["note"] = (
            "empty non-train pool (no CSV targets outside the trained vocabulary)"
        )
        return [], [], meta

    n_draw = min(test_n_samples, len(pool))
    test_words = random.Random(seed).sample(sorted(pool), n_draw)

    train_emb = (
        train_embeddings
        if train_embeddings is not None
        else encode_texts_normalized(list(train_targets), task=task)
    )
    test_emb = encode_texts_normalized(test_words, task=task)
    interp, extrap = split_interp_extrap(
        train_emb, test_emb, test_words, pca_var=pca_var
    )

    meta.update(
        {
            "n_test": len(test_words),
            "n_interp": len(interp),
            "n_extrap": len(extrap),
            "interp_words": interp,
            "extrap_words": extrap,
        }
    )
    return interp, extrap, meta


def _unseen_rate(
    decodes: Sequence[str], train_norm: set[str], norm: Callable[[str], str]
) -> float:
    if not decodes:
        return 0.0
    unique_norm = {norm(x) for x in decodes}
    return float(np.mean([x not in train_norm for x in unique_norm]))


def _coverage(target_norm: set[str], decoded_norm: set[str]) -> Optional[float]:
    if not target_norm:
        return None
    return float(len(target_norm & decoded_norm) / len(target_norm))


def genz_metrics(
    *,
    train_targets: Sequence[str],
    interp_set: Sequence[str],
    extrap_set: Sequence[str],
    greedy_decodes: Sequence[str],
    temp_decodes: dict[float, list[str]],
    task: str = "semantle",
    rdkit_map_path: Optional[str] = None,
) -> dict:
    """GENZ metrics from Sobol-point decodes.

    ``greedy_decodes`` is one decode per Sobol point; ``temp_decodes`` maps each
    temperature to the flat list of all decodes across Sobol points × samples.
    Recall metrics are target-set coverage (distinct targets hit ÷ set size).
    ``validity_*`` is added for tasks whose decodes can be structurally invalid, and
    ``tfs_intdiv_*`` for those with fingerprints — there is no per-target similarity
    here (Sobol points have no target), so TFS shows up as structural diversity,
    the fingerprint analogue of the orchestrator's ``sobol_semdisp_*``.
    """
    norm = target_normalizer(task)
    train_norm = {norm(w) for w in train_targets}
    interp_norm = {norm(w) for w in interp_set}
    extrap_norm = {norm(w) for w in extrap_set}
    test_norm = interp_norm | extrap_norm

    out: dict = {
        "sobol_unseen_greedy": _unseen_rate(greedy_decodes, train_norm, norm)
    }
    out.update(validity_metrics(greedy_decodes, task=task, suffix="greedy"))
    out.update(tfs_diversity_metrics(greedy_decodes, task=task, suffix="greedy"))
    out.update(
        rdkit_sim_diversity_metrics(
            greedy_decodes,
            task=task,
            suffix="greedy",
            map_path=rdkit_map_path,
        )
    )
    for t, decs in temp_decodes.items():
        out[f"sobol_unseen_{temp_key(t)}"] = _unseen_rate(decs, train_norm, norm)
        out.update(validity_metrics(decs, task=task, suffix=temp_key(t)))
        out.update(tfs_diversity_metrics(decs, task=task, suffix=temp_key(t)))
        out.update(
            rdkit_sim_diversity_metrics(
                decs,
                task=task,
                suffix=temp_key(t),
                map_path=rdkit_map_path,
            )
        )

    greedy_norm = {norm(x) for x in greedy_decodes}
    for label, target_set in (
        ("train", train_norm),
        ("test", test_norm),
        ("test_interp", interp_norm),
        ("test_extrap", extrap_norm),
    ):
        cov = _coverage(target_set, greedy_norm)
        if cov is not None:
            out[f"sobol_recall_{label}_greedy"] = cov

    for t, decs in temp_decodes.items():
        decoded_norm = {norm(x) for x in decs}
        for label, target_set in (
            ("train", train_norm),
            ("test", test_norm),
            ("test_interp", interp_norm),
            ("test_extrap", extrap_norm),
        ):
            cov = _coverage(target_set, decoded_norm)
            if cov is not None:
                out[f"sobol_recall_{label}_{temp_key(t)}"] = cov

    return out


# Fraction of Sobol points used as each point's neighbourhood (k = p * N), and a
# floor on k so small runs still form a usable graph. Fixing the *fraction*
# (rather than an absolute k) pins the neighbourhood to a constant spatial scale
# of the bias-space box, making Geary's C comparable across runs with different
# ``n_uniform`` / rank / bias spread.
DEFAULT_SOBOL_GEARY_P = 0.05
DEFAULT_SOBOL_GEARY_KMIN = 5


def _pairwise_sq_dists(x: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between all rows of ``x`` ([N, d]) -> [N, N]."""
    sq = np.einsum("ij,ij->i", x, x)
    d = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    return np.maximum(d, 0.0)


def map_geary_c(
    bias_vecs: np.ndarray,
    centroids: np.ndarray,
    *,
    p: float = DEFAULT_SOBOL_GEARY_P,
    k_min: int = DEFAULT_SOBOL_GEARY_KMIN,
) -> Optional[float]:
    """Multivariate Geary's C of a bias->semantics map over a point set.

    ``bias_vecs`` are the ``[N, rank]`` bias vectors (Sobol draws over the μ box,
    or the trained per-word μ vectors); ``centroids`` are the ``[N, D]``
    L2-normalized decode-embedding centroids (one per point).

    Locality is a self-tuning kNN graph (Zelnik-Manor & Perona): each point
    connects to its ``k = max(k_min, round(p * N))`` nearest neighbours in bias
    space, with a per-point bandwidth ``sigma_i`` = distance to its k-th nearest
    neighbour, and edge weights ``w_ij = exp(-||b_i - b_j||^2 / (sigma_i sigma_j))``
    on the symmetric union of neighbourhoods. Fixing ``k`` as a *fraction* of
    ``N`` holds the neighbourhood at a constant spatial scale, so C is stable
    across runs with different point counts.

    Returns C (roughly in ``[0, 2]``): ``< 1`` means neighbouring bias points
    decode more similarly than random pairs (smooth manifold); ``~1`` means no
    local structure. Lower is smoother. Returns ``None`` when degenerate (fewer
    than 3 points, mismatched shapes, or zero embedding spread) so callers can
    skip logging rather than emit NaN.
    """
    b = np.asarray(bias_vecs, dtype=np.float64)
    f = np.asarray(centroids, dtype=np.float64)
    n = b.shape[0]
    if n < 3 or f.shape[0] != n:
        return None

    k = int(max(k_min, round(p * n)))
    k = min(k, n - 1)
    if k < 1:
        return None

    # kNN in bias space (first column is the point itself, distance 0).
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(b)
    dists, idx = nbrs.kneighbors(b)
    sigma = dists[:, k]  # distance to the k-th nearest neighbour (excludes self)
    sigma = np.where(sigma > 0.0, sigma, 1e-12)

    # Directed self-tuning weights on the k nearest edges, then symmetrize (union).
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].reshape(-1)
    edge_d = dists[:, 1:].reshape(-1)
    edge_w = np.exp(-(edge_d * edge_d) / (sigma[rows] * sigma[cols]))
    w = np.zeros((n, n), dtype=np.float64)
    w[rows, cols] = edge_w
    w = np.maximum(w, w.T)
    np.fill_diagonal(w, 0.0)  # defensive: exact-duplicate points can self-connect
    w_sum = float(w.sum())

    total_var = float(((f - f.mean(axis=0, keepdims=True)) ** 2).sum())
    if w_sum <= 0.0 or total_var <= 0.0:
        return None

    num = (n - 1) * float((w * _pairwise_sq_dists(f)).sum())
    return num / (2.0 * w_sum * total_var)


# Back-compat alias: the metric is now used on both Sobol draws and trained μ
# vectors, so the general name is ``map_geary_c``. Existing GENZ / Sobol callers
# and the Sobol backfill import ``sobol_geary_c``; keep it pointed at the same fn.
sobol_geary_c = map_geary_c


def semantic_dispersion(centroids: np.ndarray) -> Optional[float]:
    """Average pairwise ``1 - max(0, cosine_sim)`` over per-point sample centroids.

    ``centroids`` is ``[N, D]`` — the L2-normalized decode-embedding centroids,
    one per point (e.g. from ``_sobol_sample_centroids``). Since the rows are unit
    vectors, cosine similarity is the dot product; negative similarities are
    clamped to 0 so the per-pair term stays in ``[0, 1]``. Points with empty
    sample bags arrive as zero rows and are dropped.

    Returns the mean over all distinct pairs (higher = decoded semantics are more
    spread out), or ``None`` when fewer than 2 usable points exist so callers can
    skip logging rather than emit NaN.
    """
    f = np.asarray(centroids, dtype=np.float64)
    if f.ndim != 2 or f.shape[0] < 2:
        return None
    norms = np.linalg.norm(f, axis=1)
    f = f[norms > 0.0]
    n = f.shape[0]
    if n < 2:
        return None
    f = f / np.linalg.norm(f, axis=1, keepdims=True)  # defensive renorm
    sims = f @ f.T
    iu = np.triu_indices(n, k=1)
    d = 1.0 - np.maximum(0.0, sims[iu])
    return float(d.mean())


# ─────────────────────────────────────────────────────────────────────────────
# LIPZ — local continuity along interpolation paths
# ─────────────────────────────────────────────────────────────────────────────

_LIPZ_METRICS = (
    "mean_step",
    "max_step",
    "emp_l",
    "emp_l_p95",
    "peakiness",
    "peakiness_p95",
    "detour",
)

# Below this max per-step cosine distance a path has effectively no semantic
# movement (collapsed); ``peakiness`` / ``detour`` are then undefined (None).
_LIPZ_DEGENERATE_EPS = 1e-8


def trajectory_step_summary(
    step_centroids: Sequence[np.ndarray],
    bias_vectors: Sequence[np.ndarray],
) -> dict:
    """Per-pair continuity summary for one interpolation variant.

    ``step_centroids[i]`` is the (un-normalized) mean text-embedding of the
    decodes at interpolation step ``i``; ``bias_vectors[i]`` is the bias vector at that
    step. ``d_sem(i) = 1 - cos(c_i, c_{i+1})`` and ``d_bias(i) = ||b_{i+1}-b_i||``.

    Metrics (per interpolation pair):
      - ``mean_step`` / ``max_step``: mean / max per-step semantic distance
        ``d_sem`` (raw magnitude of semantic movement between adjacent steps).
      - ``emp_l`` / ``emp_l_p95``: empirical Lipschitz — max / p95 of ``d_sem`` per
        unit **normalized bias arc-length**. Each ``d_bias`` step is divided by the
        total bias path length (``sum(d_bias)``), so the denominator is unitless
        and independent of the pair's raw bias-space scale ``||v1 - v0||``. This
        makes ``emp_l`` **cross-run comparable** (a run whose μ are simply more
        spread out no longer gets an artificially lower Lipschitz) and stable to
        the number of interpolation steps for smooth paths.
      - ``peakiness`` = ``max_step / mean_step`` (dimensionless): concentration of
        semantic movement. ``~1`` = uniform, gradual traversal (smooth); ``>>1`` =
        movement concentrated in a few large jumps (discontinuous / memorized
        flat-then-flip). Invariant to overall semantic scale, so it separates
        *jerkiness* from *richness* (unlike raw magnitude metrics).
      - ``peakiness_p95`` = ``p95(d_sem) / mean_step``: the same concentration
        measure with the 95th-percentile step in the numerator instead of the max,
        so a single anomalous step (e.g. a lone tokenization glitch) does not
        dominate. Shares the ``mean_step`` denominator with ``peakiness`` (mirrors
        the ``emp_l`` / ``emp_l_p95`` pair).
      - ``detour`` = ``sum(theta_i) / theta(c_0, c_last)`` (``>= 1``): angular
        arc-length vs. geodesic endpoint distance, where ``theta = arccos(cos)`` is
        the (metric) angle between consecutive centroid directions. ``~1`` = direct
        / monotone traversal along the geodesic; ``>>1`` = wandering / backtracking.
        Scale-free. (Angular distance, unlike cosine distance, obeys the triangle
        inequality, so the ratio is genuinely ``>= 1``.)

    ``peakiness`` and ``detour`` are ``None`` (not ``0.0``) when the path is
    semantically degenerate (near-zero total or endpoint movement), so a collapsed
    space is not mistaken for a smooth one; read them alongside dispersion.
    """
    n = min(len(step_centroids), len(bias_vectors))
    if n < 2:
        return {m: (0.0 if m in ("mean_step", "max_step") else None) for m in _LIPZ_METRICS}

    eps = 1e-12
    cents = [np.asarray(step_centroids[i], dtype=np.float64) for i in range(n)]
    d_sem = np.empty(n - 1, dtype=np.float64)
    d_ang = np.empty(n - 1, dtype=np.float64)
    d_bias = np.empty(n - 1, dtype=np.float64)
    for i in range(n - 1):
        a = cents[i]
        b = cents[i + 1]
        cos_ab = float(np.dot(a, b)) / (
            float(np.linalg.norm(a) * np.linalg.norm(b)) + eps
        )
        d_sem[i] = 1.0 - cos_ab
        d_ang[i] = float(np.arccos(np.clip(cos_ab, -1.0, 1.0)))
        d_bias[i] = float(
            np.linalg.norm(
                np.asarray(bias_vectors[i + 1], dtype=np.float64)
                - np.asarray(bias_vectors[i], dtype=np.float64)
            )
        )

    # Arc-length-normalized bias steps: fraction of the total bias path length per
    # step. Removes the per-run bias-scale (``||v1 - v0||``) confound from emp_l.
    d_bias_frac = d_bias / (float(d_bias.sum()) + eps)
    ratio = d_sem / (d_bias_frac + eps)

    mean_step = float(d_sem.mean())
    max_step = float(d_sem.max())

    # A path with no meaningful semantic movement anywhere is collapsed, not
    # smooth: report peakiness/detour as None so it is not counted as a good path.
    moved = max_step > _LIPZ_DEGENERATE_EPS
    defined = moved and mean_step > 0.0
    peakiness = float(max_step / mean_step) if defined else None
    peakiness_p95 = (
        float(np.percentile(d_sem, 95) / mean_step) if defined else None
    )

    # Detour uses angular distance (a metric on the sphere) so the ratio is >= 1.
    a0 = cents[0]
    aN = cents[-1]
    cos_end = float(np.dot(a0, aN)) / (
        float(np.linalg.norm(a0) * np.linalg.norm(aN)) + eps
    )
    endpoint_ang = float(np.arccos(np.clip(cos_end, -1.0, 1.0)))
    detour = (
        float(d_ang.sum() / endpoint_ang)
        if (moved and endpoint_ang > _LIPZ_DEGENERATE_EPS)
        else None
    )

    return {
        "mean_step": mean_step,
        "max_step": max_step,
        "emp_l": float(ratio.max()),
        "emp_l_p95": float(np.percentile(ratio, 95)),
        "peakiness": peakiness,
        "peakiness_p95": peakiness_p95,
        "detour": detour,
    }


def lipz_aggregate(pair_summaries: Sequence[dict]) -> dict:
    """Average per-pair LIPZ summaries across interpolation pairs.

    Each entry maps a variant key (``greedy``, ``temp1.0``, …) to a dict of the
    per-variant step metrics (``_LIPZ_METRICS``). Metrics that are ``None`` for a
    pair (degenerate paths) are skipped. Returns
    ``{f"{metric}_{variant}": mean_over_pairs}``.
    """
    out: dict = {}
    variants: set[str] = set()
    for ps in pair_summaries:
        variants.update(ps.keys())
    for v in sorted(variants):
        for metric in _LIPZ_METRICS:
            vals = [
                ps[v][metric]
                for ps in pair_summaries
                if v in ps and ps[v].get(metric) is not None
            ]
            if vals:
                out[f"{metric}_{v}"] = float(np.mean(vals))
    return out
