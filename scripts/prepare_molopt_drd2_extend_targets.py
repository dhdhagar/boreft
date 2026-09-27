#!/usr/bin/env python3
"""Build DRD2 absorption sets for continual training from a T=1 BOReFT search.

Two JSONL files, each a list of ``{"target", "definition"}`` rows for
``python -m boreft.extend_subspace --new-targets``:

* ``ge01`` — every unique molecule with DRD2 score at least 0.1.
* ``cov50`` — molecules with score above 0.03 and vertex coverage error at
  least 0.50, farthest first, capped at 32. The 0.03 floor drops the large
  tie at 0.029.

A molecule already in the checkpoint's training catalog is skipped. A molecule
present in the checkpoint definition file keeps that prose. Otherwise the
definition is the labeled RDKit sentence ``2D properties: ...``.

  python scripts/prepare_molopt_drd2_extend_targets.py \\
    --reft-output-dir outputs/1789921464
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from boreft.chem import is_valid_smiles, unwrap_smiles_tags
from boreft.data_utils import load_merged_run_config
from boreft.oracle_screen import smiles_from_items_json
from boreft.text_similarity import (
    append_rdkit_definition_values,
    load_raw_definitions,
)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import analyze_molopt_baseline_coverage as coverage  # noqa: E402

DEFAULT_SEARCH_ROOT = REPO / "experiments" / "outputs" / "molopt" / "search"
DEFAULT_OUT_DIR = REPO / "experiments" / "outputs" / "molopt" / "extend"
HIGH_SCORE = 0.1
SCORE_FLOOR = 0.03
COVERAGE_MIN = 0.50
COVERAGE_CAP = 32


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reft-output-dir", default="outputs/1789921464")
    parser.add_argument("--search-root", type=Path, default=DEFAULT_SEARCH_ROOT)
    parser.add_argument("--boreft-dir", default="boreft_mu")
    parser.add_argument("--oracle", default="DRD2")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--high-score", type=float, default=HIGH_SCORE)
    parser.add_argument("--score-floor", type=float, default=SCORE_FLOOR)
    parser.add_argument("--coverage-min", type=float, default=COVERAGE_MIN)
    parser.add_argument("--coverage-cap", type=int, default=COVERAGE_CAP)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def _raw_definitions_by_molecule(saved: dict) -> dict[str, str]:
    path = saved.get("definitions_path")
    if not path:
        return {}
    raw = load_raw_definitions(str(path))
    keyed: dict[str, str] = {}
    for target, text in raw.items():
        key = coverage._molecule_key(str(target))
        if key is None or key in keyed:
            continue
        keyed[key] = str(text).strip()
    return keyed


def _definition(smiles: str, prose: dict[str, str]) -> str:
    existing = prose.get(smiles)
    if existing:
        return existing
    if not is_valid_smiles(unwrap_smiles_tags(smiles)):
        raise ValueError(f"cannot build an RDKit definition for {smiles!r}")
    return append_rdkit_definition_values(smiles, "")


def candidate_rows(
    molecules: dict[str, dict],
    train_index: dict[str, int],
    *,
    high_score: float,
    score_floor: float,
) -> list[dict]:
    """Non-catalog molecules that can enter either absorption set.

    The score floor applies only to the coverage set. A molecule at or above
    ``high_score`` is kept even when it falls at or below that floor.
    """
    chosen = []
    for row in molecules.values():
        if row["smiles"] in train_index:
            continue
        score = float(row["score"])
        if score >= high_score or score > score_floor:
            chosen.append(row)
    chosen.sort(key=lambda row: (-float(row["score"]), row["smiles"]))
    return chosen


def split_absorption_sets(
    records: list[dict],
    *,
    high_score: float,
    score_floor: float,
    coverage_min: float,
    coverage_cap: int,
) -> tuple[list[dict], list[dict]]:
    """Return ``(score set, coverage set)``."""
    if coverage_cap < 0:
        raise ValueError("coverage cap must be nonnegative")
    by_score = [row for row in records if float(row["score"]) >= high_score]
    by_score.sort(key=lambda row: (-float(row["score"]), row["target"]))
    by_coverage = [
        row
        for row in records
        if float(row["score"]) > score_floor
        and float(row["coverage_error"]) >= coverage_min
    ]
    by_coverage.sort(
        key=lambda row: (
            -float(row["coverage_error"]),
            -float(row["score"]),
            row["target"],
        )
    )
    return by_score, by_coverage[:coverage_cap]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    reft_dir = Path(args.reft_output_dir)
    if not reft_dir.is_absolute():
        reft_dir = REPO / reft_dir
    items = reft_dir / "items.json"
    if not items.is_file():
        raise FileNotFoundError(f"missing training targets: {items}")

    saved = load_merged_run_config(str(reft_dir))
    prose = _raw_definitions_by_molecule(saved)
    molecules = coverage.load_method_molecules(
        args.search_root, args.boreft_dir, args.oracle, args.seeds
    )
    train_index, train_texts = coverage._train_key_index(smiles_from_items_json(str(items)))

    pool = candidate_rows(
        molecules,
        train_index,
        high_score=float(args.high_score),
        score_floor=float(args.score_floor),
    )
    if not pool:
        raise ValueError(
            f"no {args.oracle} molecules with score >= {args.high_score} "
            f"or score > {args.score_floor}"
        )

    print(f"[extend-targets] encode {len(train_texts)} train + {len(pool)} candidates", flush=True)
    train_phi = coverage._encode(train_texts, int(args.encode_batch_size))
    eval_phi = coverage._encode([row["smiles"] for row in pool], int(args.encode_batch_size))
    distances = coverage.nearest_vertex_distance(eval_phi, train_phi)

    records = []
    for row, distance in zip(pool, distances):
        smiles = row["smiles"]
        definition = _definition(smiles, prose)
        records.append(
            {
                "target": smiles,
                "definition": definition,
                "score": float(row["score"]),
                "coverage_error": float(distance),
                "definition_source": "molt5" if smiles in prose else "rdkit",
                "seeds": coverage._source_seeds(row),
            }
        )

    ge01, cov = split_absorption_sets(
        records,
        high_score=float(args.high_score),
        score_floor=float(args.score_floor),
        coverage_min=float(args.coverage_min),
        coverage_cap=int(args.coverage_cap),
    )
    if not ge01:
        raise ValueError(f"no molecules with score >= {args.high_score}")
    if not cov:
        raise ValueError(
            f"no molecules with score > {args.score_floor} and "
            f"coverage >= {args.coverage_min}"
        )

    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    ge01_path = out_dir / "drd2_t1_ge01.jsonl"
    cov_path = out_dir / "drd2_t1_cov50.jsonl"
    _write_jsonl(ge01_path, ge01)
    _write_jsonl(cov_path, cov)

    def _brief(name: str, rows: list[dict]) -> None:
        n_rdkit = sum(1 for row in rows if row["definition_source"] == "rdkit")
        scores = [float(row["score"]) for row in rows]
        dists = [float(row["coverage_error"]) for row in rows]
        print(
            f"[extend-targets] {name} n={len(rows)} rdkit_definitions={n_rdkit} "
            f"score {min(scores):.3f}..{max(scores):.3f} "
            f"coverage {min(dists):.3f}..{max(dists):.3f}"
        )

    _brief("ge01", ge01)
    _brief("cov50", cov)
    print(f"[extend-targets] wrote {ge01_path}")
    print(f"[extend-targets] wrote {cov_path}")


if __name__ == "__main__":
    main()
