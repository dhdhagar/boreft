#!/usr/bin/env python3
"""Coverage and box distance for baseline molecules that beat BOReFT.

For each property oracle, collect every unique molecule proposed by the
baselines, across seeds, whose score is strictly above the best score BOReFT
found on that oracle. Compare that set with the same number of highest-scoring
unique BOReFT molecules.

Coverage error is Euclidean distance in L2-normalized molopt φ to the nearest
training-set vertex (A = training vertices, the same protocol as
``scripts/molopt_coverage_error.py``). The bounding box is the axis-aligned
box of the checkpoint's training posterior means, with no extra padding: the
domain BOReFT searches. A catalog molecule uses its stored μ. Any other
molecule uses the bias network's unclipped prediction, with the same
definition embedding the network was trained on. A molecule with no definition
is left unplaced: this checkpoint does not embed raw SMILES into μ, so a
SMILES-prompt prediction is not a point in the learned box. Box distance is
Euclidean distance to that box (0 when the code lies inside).

Defaults are the p90 N=1024 checkpoint and the pretrained with-header
baselines (random, BOPRO, OPRO) against ``boreft_mu``.

  python scripts/analyze_molopt_baseline_coverage.py \\
    --reft-output-dir outputs/1789921464
  sbatch --export=ALL,REFT_OUTPUT_DIR=outputs/1789921464 \\
    scripts/analyze_molopt_baseline_coverage.sh
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

import numpy as np

from boreft.chem import canonical_target_key, is_valid_smiles, unwrap_smiles_tags

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SEARCH_ROOT = REPO / "experiments" / "outputs" / "molopt" / "search"
DEFAULT_BASELINES = ("random_sampling_mu", "bopro_mu", "opro_mu")
DEFAULT_BOREFT = "boreft_mu"
DEFAULT_ORACLES = ("DRD2", "GSK3B", "JNK3")
DEFAULT_SEEDS = (1, 2, 3, 4, 5)
BOX_ATOL = 1e-5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reft-output-dir", default="outputs/1789921464")
    parser.add_argument("--search-root", default=str(DEFAULT_SEARCH_ROOT))
    parser.add_argument("--boreft-dir", default=DEFAULT_BOREFT)
    parser.add_argument(
        "--baseline-dirs",
        nargs="+",
        default=list(DEFAULT_BASELINES),
        help="Method directories under the search root. A molecule from any "
        "of them counts once.",
    )
    parser.add_argument("--oracles", nargs="+", default=list(DEFAULT_ORACLES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--output",
        default=None,
        help="JSON report path. Default: experiments/outputs/molopt/analysis/"
        "baseline_box_coverage_<run>.json",
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--load-latest", action="store_true")
    parser.add_argument("--encode-batch-size", type=int, default=64)
    return parser.parse_args(argv)


def aabb_distance(point: np.ndarray, bounds: np.ndarray) -> float:
    """Euclidean distance from ``point`` to the axis-aligned box ``bounds`` [2, D]."""
    box = np.asarray(bounds, dtype=np.float64)
    if box.ndim != 2 or box.shape[0] != 2:
        raise ValueError("bounds must have shape [2, D]")
    x = np.asarray(point, dtype=np.float64).reshape(-1)
    if x.shape[0] != box.shape[1]:
        raise ValueError(
            f"point dimension {x.shape[0]} does not match box dimension {box.shape[1]}"
        )
    projected = np.minimum(np.maximum(x, box[0]), box[1])
    return float(np.linalg.norm(x - projected))


def outside_box(distance: float, *, atol: float = BOX_ATOL) -> bool:
    return float(distance) > float(atol)


def summarize(values: Sequence[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
        "min": float(array.min()),
    }


def _l2_normalize(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(array).all() or np.any(norms <= 0):
        raise ValueError("embeddings must be finite and nonzero")
    return array / norms


def nearest_vertex_distance(eval_phi: np.ndarray, train_phi: np.ndarray) -> np.ndarray:
    """Row-wise Euclidean distance to the nearest L2-normalized training vertex."""
    left = _l2_normalize(eval_phi)
    right = _l2_normalize(train_phi)
    if left.ndim != 2 or right.ndim != 2 or right.shape[0] == 0 or left.shape[0] == 0:
        raise ValueError("expected non-empty [N, D] and [M, D] embeddings")
    if left.shape[1] != right.shape[1]:
        raise ValueError("embedding dimensions differ")
    dots = left @ right.T
    return np.sqrt(np.clip(2.0 - 2.0 * dots, 0.0, None)).min(axis=1)


def definition_lookup_by_molecule(
    lookup: Mapping[str, str] | None,
) -> dict[str, str] | None:
    """Rekey a definition-embedding lookup by canonical SMILES.

    The checkpoint file is keyed by the stored target string. Search molecules
    are canonical, so also index each parseable target by that key. Values stay
    the decorated definition text the bias network was trained on.
    """
    if lookup is None:
        return None
    expanded = {str(key): str(value) for key, value in lookup.items()}
    for raw, text in lookup.items():
        key = _molecule_key(str(raw))
        if key and key not in expanded:
            expanded[key] = str(text)
    return expanded


def _molecule_key(text: str) -> str | None:
    key = canonical_target_key(unwrap_smiles_tags(str(text or "").strip()))
    if not key or not is_valid_smiles(key):
        return None
    return key


def molecules_from_record(record: Mapping) -> list[tuple[str, float]]:
    """Valid (canonical SMILES, score) pairs scored by one observation.

    Multi-sample rows contribute each sample at its own score. A single
    sample uses the oracle's canonical SMILES and ``oracle_score``.
    """
    samples = list(record.get("solution_samples") or record.get("decoded_samples") or [])
    scores = [float(value) for value in record.get("sample_scores") or []]
    if len(samples) > 1 and len(samples) == len(scores):
        found: list[tuple[str, float]] = []
        for sample, score in zip(samples, scores):
            key = _molecule_key(str(sample))
            if key is None or not np.isfinite(score):
                continue
            found.append((key, float(score)))
        return found
    components = record.get("components") or {}
    raw = components.get("canonical") or record.get("solution") or record.get("decoded") or ""
    key = _molecule_key(str(raw))
    if key is None:
        return []
    score = components.get("oracle_score", record.get("score"))
    if score is None or not np.isfinite(float(score)):
        return []
    return [(key, float(score))]


def load_method_molecules(
    search_root: Path,
    method_dir: str,
    oracle: str,
    seeds: Sequence[int],
) -> dict[str, dict]:
    """Best score of each canonical molecule, plus where it was proposed."""
    best: dict[str, dict] = {}
    for seed in seeds:
        path = search_root / method_dir / oracle / f"seed_{int(seed)}" / "observations.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing observations: {path}")
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                phase = str(record.get("phase") or record.get("source") or "")
                for key, score in molecules_from_record(record):
                    row = best.get(key)
                    source = {"method": method_dir, "seed": int(seed), "phase": phase}
                    if row is None:
                        best[key] = {
                            "smiles": key,
                            "score": float(score),
                            "sources": [source],
                        }
                        continue
                    if source not in row["sources"]:
                        row["sources"].append(source)
                    if score > row["score"]:
                        row["score"] = float(score)
    return best


def merge_molecules(groups: Iterable[Mapping[str, dict]]) -> dict[str, dict]:
    """Union molecule tables, keeping the highest score and every source."""
    merged: dict[str, dict] = {}
    for group in groups:
        for key, row in group.items():
            current = merged.get(key)
            if current is None:
                merged[key] = {
                    "smiles": key,
                    "score": float(row["score"]),
                    "sources": list(row["sources"]),
                }
                continue
            current["sources"].extend(row["sources"])
            if float(row["score"]) > current["score"]:
                current["score"] = float(row["score"])
    return merged


def molecules_above(molecules: Mapping[str, dict], threshold: float) -> list[dict]:
    """Unique molecules whose score is strictly above ``threshold``, best first."""
    chosen = [row for row in molecules.values() if float(row["score"]) > float(threshold)]
    chosen.sort(key=lambda row: (-float(row["score"]), row["smiles"]))
    return chosen


def top_molecules(molecules: Mapping[str, dict], count: int) -> list[dict]:
    """Highest-scoring unique molecules. Ties break by canonical SMILES."""
    if count < 0:
        raise ValueError("count must be nonnegative")
    ranked = sorted(
        molecules.values(),
        key=lambda row: (-float(row["score"]), row["smiles"]),
    )
    return ranked[:count]


def _source_methods(row: Mapping) -> list[str]:
    return sorted({str(source["method"]) for source in row.get("sources") or []})


def _source_seeds(row: Mapping) -> list[int]:
    return sorted({int(source["seed"]) for source in row.get("sources") or []})


def aggregate_rows(rows: Sequence[Mapping]) -> dict:
    coverage = [float(row["coverage_error"]) for row in rows]
    placed = [row for row in rows if row["box_distance"] is not None]
    outside = [row for row in placed if row["outside_box"]]
    return {
        "n": len(rows),
        "n_in_train_catalog": sum(1 for row in rows if row["in_train_catalog"]),
        "n_unplaced": len(rows) - len(placed),
        "n_outside_box": len(outside),
        "frac_outside_box": (len(outside) / len(placed)) if placed else None,
        "coverage_error": summarize(coverage),
        "box_distance": summarize([float(row["box_distance"]) for row in placed]),
        "box_distance_outside": summarize(
            [float(row["box_distance"]) for row in outside]
        ),
        "score": summarize([float(row["score"]) for row in rows]),
    }


def _fmt(stats: Mapping) -> str:
    if not stats or stats.get("n", 0) == 0:
        return "n=0"
    return (
        f"n={stats['n']} mean={stats['mean']:.4f} "
        f"median={stats['median']:.4f} max={stats['max']:.4f}"
    )


def format_report(payload: Mapping) -> str:
    lines = [
        f"checkpoint  {payload['reft_output_dir']}",
        f"boreft      {payload['boreft_dir']}",
        f"baselines   {', '.join(payload['baseline_dirs'])}",
        "coverage    nearest training vertex in molopt φ",
        "box         AABB of training posterior means; catalog μ or unclipped prediction",
        "",
    ]
    for oracle, block in payload["oracles"].items():
        lines.append(
            f"{oracle}  BOReFT best={block['boreft_best_score']:.6g}  "
            f"baseline molecules above={block['n_baseline_above']}"
        )
        for name in ("baselines_above_boreft", "boreft_top"):
            group = block[name]
            summary = group["summary"]
            outside = summary["frac_outside_box"]
            outside_text = "n/a" if outside is None else f"{outside:.3f}"
            lines.append(
                f"  {name}"
                f"  cov[{_fmt(summary['coverage_error'])}]"
                f"  outside={outside_text}"
                f"  unplaced={summary['n_unplaced']}"
                f"  dist[{_fmt(summary['box_distance'])}]"
                f"  dist_out[{_fmt(summary['box_distance_outside'])}]"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _train_key_index(words: Sequence[str]) -> tuple[dict[str, int], list[str]]:
    """Canonical key to train-row index, and the stored SMILES to embed for each key."""
    index: dict[str, int] = {}
    texts: dict[str, str] = {}
    for row_index, raw in enumerate(words):
        key = _molecule_key(str(raw))
        if key is None or key in index:
            continue
        index[key] = row_index
        texts[key] = unwrap_smiles_tags(str(raw).strip())
    if not index:
        raise ValueError("checkpoint train set has no valid SMILES")
    keys = list(index)
    return index, [texts[key] for key in keys]


def _encode(texts: Sequence[str], batch_size: int) -> np.ndarray:
    from boreft.text_similarity import encode_texts_normalized

    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        print(
            f"[coverage] encode {start + 1}-{start + len(batch)} / {len(texts)}",
            flush=True,
        )
        chunks.append(encode_texts_normalized(batch, task="molopt"))
    if not chunks:
        raise ValueError("no texts to encode")
    return np.concatenate(chunks, axis=0).astype(np.float64)


def _attach_geometry(
    rows: Sequence[dict],
    *,
    train_index: Mapping[str, int],
    train_mu: np.ndarray,
    bounds: np.ndarray,
    predicted: Mapping[str, np.ndarray],
    train_phi: np.ndarray,
    train_keys: Sequence[str],
    eval_phi: np.ndarray,
    eval_keys: Sequence[str],
) -> list[dict]:
    phi_index = {key: i for i, key in enumerate(eval_keys)}
    train_phi_index = {key: i for i, key in enumerate(train_keys)}
    attached = []
    for row in rows:
        key = row["smiles"]
        catalog_row = train_index.get(key)
        if catalog_row is not None:
            mu = np.asarray(train_mu[catalog_row], dtype=np.float64)
            source = "train"
            distance: float | None = aabb_distance(mu, bounds)
            is_outside: bool | None = outside_box(distance)
        elif key in predicted:
            mu = np.asarray(predicted[key], dtype=np.float64)
            source = "predicted"
            distance = aabb_distance(mu, bounds)
            is_outside = outside_box(distance)
        else:
            source = "no_definition"
            distance = None
            is_outside = None
        if key in train_phi_index:
            coverage = 0.0
        else:
            coverage = float(
                nearest_vertex_distance(eval_phi[phi_index[key] : phi_index[key] + 1], train_phi)[0]
            )
        attached.append(
            {
                "smiles": key,
                "score": float(row["score"]),
                "methods": _source_methods(row),
                "seeds": _source_seeds(row),
                "in_train_catalog": catalog_row is not None,
                "mu_source": source,
                "coverage_error": coverage,
                "outside_box": is_outside,
                "box_distance": distance,
            }
        )
    return attached


def select_oracle_sets(
    search_root: Path,
    *,
    boreft_dir: str,
    baseline_dirs: Sequence[str],
    oracle: str,
    seeds: Sequence[int],
) -> dict:
    boreft = load_method_molecules(search_root, boreft_dir, oracle, seeds)
    if not boreft:
        raise ValueError(f"{oracle}: BOReFT proposed no valid molecules")
    threshold = max(float(row["score"]) for row in boreft.values())
    baselines = merge_molecules(
        load_method_molecules(search_root, method, oracle, seeds)
        for method in baseline_dirs
    )
    above = molecules_above(baselines, threshold)
    matched = top_molecules(boreft, len(above))
    return {
        "boreft_best_score": threshold,
        "n_boreft_molecules": len(boreft),
        "n_baseline_molecules": len(baselines),
        "n_baseline_above": len(above),
        "baselines_above": above,
        "boreft_top": matched,
    }


def run(args: argparse.Namespace) -> dict:
    from boreft.bias_tables import (
        bias_predict_kwargs,
        predict_bias_vectors_for_words,
        stack_bias_vectors,
    )
    from boreft.bo.acquisition import latent_bounds
    from boreft.data_utils import load_merged_run_config
    from boreft.eval.semantle import load_eval_checkpoint, release_eval_checkpoint
    from boreft.text_similarity import definition_lookup_for_cfg

    reft_dir = Path(args.reft_output_dir).expanduser()
    search_root = Path(args.search_root).expanduser()
    oracles = tuple(dict.fromkeys(str(name).strip() for name in args.oracles if str(name).strip()))
    baseline_dirs = tuple(
        dict.fromkeys(str(name).strip() for name in args.baseline_dirs if str(name).strip())
    )
    seeds = tuple(int(seed) for seed in args.seeds)
    if not oracles or not baseline_dirs or not seeds:
        raise ValueError("oracles, baseline dirs, and seeds must be non-empty")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")

    selected = {
        oracle: select_oracle_sets(
            search_root,
            boreft_dir=args.boreft_dir,
            baseline_dirs=baseline_dirs,
            oracle=oracle,
            seeds=seeds,
        )
        for oracle in oracles
    }
    needed = []
    seen = set()
    for block in selected.values():
        for row in list(block["baselines_above"]) + list(block["boreft_top"]):
            if row["smiles"] not in seen:
                seen.add(row["smiles"])
                needed.append(row["smiles"])

    saved = load_merged_run_config(str(reft_dir))
    if saved.get("task", "semantle") != "molopt":
        raise ValueError(f"expected a molopt checkpoint, got task={saved.get('task')!r}")
    model_name = saved.get("model_name")
    if not model_name:
        raise ValueError("checkpoint config has no model_name")
    ckpt = load_eval_checkpoint(
        str(reft_dir),
        model_name,
        int(saved.get("layer", 15)),
        int(saved.get("low_rank_dim", 8)),
        args.cache_dir or saved.get("cache_dir"),
        load_latest=bool(args.load_latest),
        skip_embed_cache=True,
    )
    try:
        train_index, train_texts = _train_key_index(ckpt.words)
        train_keys = list(train_index)
        if hasattr(ckpt.reft_model, "eval"):
            ckpt.reft_model.eval()
        train_mu = stack_bias_vectors(
            ckpt.reft_model, list(range(len(ckpt.words)))
        ).astype(np.float64)
        bounds = latent_bounds(train_mu).astype(np.float64)
        definition_lookup = definition_lookup_by_molecule(
            definition_lookup_for_cfg(saved)
        )
        missing = [key for key in needed if key not in train_index]
        if definition_lookup is None:
            placeable = missing
        else:
            placeable = [key for key in missing if key in definition_lookup]
        print(
            f"[coverage] predict μ for {len(placeable)} / {len(missing)} "
            "molecules outside the train catalog"
            + (
                f" ({len(missing) - len(placeable)} have no definition)"
                if definition_lookup is not None
                else ""
            ),
            flush=True,
        )
        predicted: dict[str, np.ndarray] = {}
        if placeable:
            vectors = predict_bias_vectors_for_words(
                ckpt.reft_model,
                placeable,
                task="molopt",
                definition_lookup=definition_lookup,
                batch_size=int(args.encode_batch_size),
                **bias_predict_kwargs(saved, tokenizer=getattr(ckpt, "tokenizer", None)),
            )
            if len(vectors) != len(placeable):
                raise ValueError(
                    f"bias network returned {len(vectors)} μ for {len(placeable)} molecules"
                )
            rank = int(bounds.shape[1])
            for key, vector in zip(placeable, vectors):
                point = np.asarray(vector, dtype=np.float64).reshape(-1)
                if point.shape[0] != rank or not np.isfinite(point).all():
                    raise ValueError(
                        f"predicted μ for {key} must be a finite vector of rank {rank}"
                    )
                predicted[key] = point
        print(f"[coverage] encode {len(train_texts)} training vertices", flush=True)
        train_phi = _encode(train_texts, int(args.encode_batch_size))
        eval_keys = [key for key in needed if key not in train_index]
        eval_phi = (
            _encode(eval_keys, int(args.encode_batch_size))
            if eval_keys
            else np.zeros((0, train_phi.shape[1]), dtype=np.float64)
        )
    finally:
        release_eval_checkpoint(ckpt)

    oracle_payload = {}
    for oracle, block in selected.items():
        baseline_rows = _attach_geometry(
            block["baselines_above"],
            train_index=train_index,
            train_mu=train_mu,
            bounds=bounds,
            predicted=predicted,
            train_phi=train_phi,
            train_keys=train_keys,
            eval_phi=eval_phi,
            eval_keys=eval_keys,
        )
        boreft_rows = _attach_geometry(
            block["boreft_top"],
            train_index=train_index,
            train_mu=train_mu,
            bounds=bounds,
            predicted=predicted,
            train_phi=train_phi,
            train_keys=train_keys,
            eval_phi=eval_phi,
            eval_keys=eval_keys,
        )
        oracle_payload[oracle] = {
            "boreft_best_score": block["boreft_best_score"],
            "n_boreft_molecules": block["n_boreft_molecules"],
            "n_baseline_molecules": block["n_baseline_molecules"],
            "n_baseline_above": block["n_baseline_above"],
            "baselines_above_boreft": {
                "summary": aggregate_rows(baseline_rows),
                "molecules": baseline_rows,
            },
            "boreft_top": {
                "summary": aggregate_rows(boreft_rows),
                "molecules": boreft_rows,
            },
        }
    return {
        "reft_output_dir": str(reft_dir),
        "search_root": str(search_root),
        "boreft_dir": args.boreft_dir,
        "baseline_dirs": list(baseline_dirs),
        "seeds": list(seeds),
        "box": (
            "axis-aligned bounding box of training posterior means "
            "(search default, padding 0)"
        ),
        "coverage": (
            "Euclidean distance in L2-normalized molopt φ to the nearest "
            "training vertex; catalog molecules are 0"
        ),
        "mu": (
            "stored training μ when the molecule is in the checkpoint catalog; "
            "otherwise the bias network's unclipped prediction from the "
            "definition embedding it was trained on. Molecules with no "
            "definition are unplaced"
        ),
        "box_atol": BOX_ATOL,
        "oracles": oracle_payload,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run(args)
    dest = (
        Path(args.output).expanduser()
        if args.output
        else REPO
        / "experiments"
        / "outputs"
        / "molopt"
        / "analysis"
        / f"baseline_box_coverage_{Path(args.reft_output_dir).name}.json"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    sys.stdout.write(format_report(payload))
    print(f"[coverage] wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
