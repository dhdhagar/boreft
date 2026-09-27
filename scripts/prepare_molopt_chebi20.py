#!/usr/bin/env python
"""Build the molopt training corpus from ChEBI-20 (the MolT5 SMILES↔text dataset).

Writes two files that must stay in lockstep, because training validates that every
target has a definition:

``<out>``
    ``inchikey,smiles,cid`` — read by ``MolOptItem.load_csv``. Row order matters:
    ``--train-top-k`` takes a prefix as the trained vocabulary and everything past
    the head becomes the held-out pool for RECON_TEST and GENZ. Rows are therefore
    shuffled with ``--seed`` rather than left in source order, so the trained head
    and the held-out tail are exchangeable — sorting by anything correlated with
    difficulty (SMILES length, say) would make the pool systematically harder than
    the vocabulary and quietly bias every generalization metric.

``<definitions>``
    ``{"target": <canonical smiles>, "definition": <ChEBI description>}`` per row,
    consumed by ``--use-definition-embeds`` and by SDPO as the teacher's privileged
    context. The committed file carries ontology-derived ``category_normalized``
    labels on top of these two fields; this script does not write them, so re-run
    ``label_molopt_chebi_ontology.py`` afterwards to restore them.

``<rdkit-definitions>``
    ``{"target": <canonical smiles>, "definition": [<scalar>, ...]}`` per row,
    containing a fixed-order vector of calculated 2D molecular descriptors.

``<rdkit-map>``
    JSON metadata defining the human-readable definition label, units, and RDKit
    calculation for every position in the descriptor vector.

Source: https://github.com/blender-nlp/MolT5/tree/main/ChEBI-20_data (33k molecules
with human-written ChEBI descriptions, tab-separated ``CID/SMILES/description``).

    python scripts/prepare_molopt_chebi20.py --download

Every description begins with the constant lead-in "The molecule is ", which is
stripped by default: a prefix shared by 100% of rows adds no discriminative signal
to the embedding and only eats into the encoder's token budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import urllib.request

from rdkit import Chem, RDLogger

from boreft.chem import (
    RDKIT_DESCRIPTOR_MAP,
    RDKIT_DESCRIPTOR_SCHEMA_VERSION,
    rdkit_descriptor_values,
    robust_descriptor_stats,
)

CHEBI_SPLITS = ("train", "validation", "test")
CHEBI_BASE_URL = (
    "https://raw.githubusercontent.com/blender-nlp/MolT5/main/ChEBI-20_data"
)

# Shared by every ChEBI-20 description (see module docstring).
DESCRIPTION_LEAD_IN = "The molecule is "

DEFAULT_CHEBI_DIR = os.path.join("data", "molopt", "raw", "ChEBI-20_data")
DEFAULT_OUT = os.path.join("data", "molopt", "train", "chebi20.csv")
DEFAULT_DEFINITIONS = os.path.join("data", "molopt", "train", "definitions.jsonl")
DEFAULT_RDKIT_DEFINITIONS = os.path.join(
    "data", "molopt", "train", "definitions_rdkit.jsonl"
)
DEFAULT_RDKIT_MAP = os.path.join(
    "data", "molopt", "train", "definitions_rdkit_map.json"
)

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--chebi-dir",
        default=DEFAULT_CHEBI_DIR,
        help="Directory holding train.txt / validation.txt / test.txt.",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="Fetch any missing split files into --chebi-dir first.",
    )
    p.add_argument(
        "--splits",
        nargs="*",
        default=list(CHEBI_SPLITS),
        choices=list(CHEBI_SPLITS),
        help=(
            "Which ChEBI-20 splits to pool. Their split is irrelevant here — the "
            "interp/extrap split is recomputed from bias-space geometry at eval "
            "time — so all three are pooled by default."
        ),
    )
    p.add_argument("--out", default=DEFAULT_OUT, help="Output CSV path.")
    p.add_argument(
        "--definitions",
        default=DEFAULT_DEFINITIONS,
        help="Output definitions JSONL path.",
    )
    p.add_argument(
        "--rdkit-definitions",
        default=DEFAULT_RDKIT_DEFINITIONS,
        help="Output JSONL path for fixed-order RDKit descriptor vectors.",
    )
    p.add_argument(
        "--rdkit-map",
        default=DEFAULT_RDKIT_MAP,
        help="Output JSON path describing each RDKit descriptor-vector position.",
    )
    p.add_argument(
        "--n",
        type=int,
        default=8000,
        help=(
            "Molecules to keep after filtering and shuffling (0 = all ~33k). The "
            "default leaves plenty of held-out pool past a few-thousand-target "
            "vocabulary while keeping the committed files small."
        ),
    )
    p.add_argument(
        "--seed", type=int, default=0, help="Shuffle seed (see module docstring)."
    )
    p.add_argument(
        "--max-smiles-length",
        type=int,
        default=100,
        help=(
            "Drop molecules with longer SMILES; long targets need more decode "
            "tokens, which slows every reconstruction pass."
        ),
    )
    p.add_argument(
        "--min-heavy-atoms",
        type=int,
        default=5,
        help=(
            "Drop tiny entries. ChEBI includes single-atom records such as "
            "'[125Te]', which are valid but not drug-like generation targets."
        ),
    )
    p.add_argument(
        "--keep-lead-in",
        action="store_true",
        help=f"Keep the constant {DESCRIPTION_LEAD_IN!r} description prefix.",
    )
    return p.parse_args(argv)


def download_splits(chebi_dir: str, splits: list[str]) -> None:
    os.makedirs(chebi_dir, exist_ok=True)
    for split in splits:
        dest = os.path.join(chebi_dir, f"{split}.txt")
        if os.path.isfile(dest):
            print(f"[chebi20] {dest} already present")
            continue
        url = f"{CHEBI_BASE_URL}/{split}.txt"
        print(f"[chebi20] downloading {url} → {dest}")
        urllib.request.urlretrieve(url, dest)


def clean_description(text: str, *, keep_lead_in: bool) -> str:
    description = " ".join(str(text).strip().split())
    if not keep_lead_in and description.startswith(DESCRIPTION_LEAD_IN):
        description = description[len(DESCRIPTION_LEAD_IN) :]
    return description.strip()


def read_split(path: str, *, keep_lead_in: bool) -> list[dict[str, str]]:
    """``{cid, smiles, description}`` rows from one ChEBI-20 TSV."""
    rows: list[dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        missing = {"CID", "SMILES", "description"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
        for row in reader:
            description = clean_description(
                row.get("description") or "", keep_lead_in=keep_lead_in
            )
            if not description:
                continue
            rows.append(
                {
                    "cid": (row.get("CID") or "").strip(),
                    "smiles": (row.get("SMILES") or "").strip(),
                    "description": description,
                }
            )
    return rows


def filter_and_canonicalize(
    rows: list[dict[str, str]], *, max_smiles_length: int, min_heavy_atoms: int
) -> list[dict[str, str]]:
    """Canonicalize SMILES and drop rows that cannot serve as training targets.

    De-duplicates by canonical SMILES: distinct spellings collapse to one string,
    and a repeated target would get two independent bias vectors, quietly skewing
    every RECON metric.
    """
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    counts = {"unparseable": 0, "too_long": 0, "too_small": 0, "duplicate": 0}

    for row in rows:
        mol = Chem.MolFromSmiles(row["smiles"]) if row["smiles"] else None
        if mol is None:
            counts["unparseable"] += 1
            continue
        if mol.GetNumHeavyAtoms() < min_heavy_atoms:
            counts["too_small"] += 1
            continue
        canonical = Chem.MolToSmiles(mol)
        if len(canonical) > max_smiles_length:
            counts["too_long"] += 1
            continue
        if canonical in seen:
            counts["duplicate"] += 1
            continue
        seen.add(canonical)
        out.append(
            {
                "inchikey": Chem.MolToInchiKey(mol),
                "smiles": canonical,
                "cid": row["cid"],
                "description": row["description"],
            }
        )

    print(
        f"[chebi20] kept {len(out)}/{len(rows)} rows  "
        + "  ".join(f"{k}={v}" for k, v in counts.items() if v)
    )
    return out


def write_outputs(
    rows: list[dict[str, str]],
    *,
    out: str,
    definitions: str,
    rdkit_definitions: str,
    rdkit_map: str,
) -> None:
    fieldnames = ["inchikey", "smiles", "cid"]
    for path in (out, definitions, rdkit_definitions, rdkit_map):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})

    with open(definitions, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    {"target": row["smiles"], "definition": row["description"]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    descriptor_vectors: list[list[float | int]] = []
    with open(rdkit_definitions, "w", encoding="utf-8") as f:
        for row in rows:
            values = rdkit_descriptor_values(row["smiles"])
            if values is None:  # Defensive: canonicalization validated every row.
                raise ValueError(f"could not parse canonical SMILES {row['smiles']!r}")
            descriptor_vectors.append(values)
            f.write(
                json.dumps(
                    {
                        "target": row["smiles"],
                        "definition": values,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with open(rdkit_map, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": RDKIT_DESCRIPTOR_SCHEMA_VERSION,
                "definition_length": len(RDKIT_DESCRIPTOR_MAP),
                "positions": list(RDKIT_DESCRIPTOR_MAP),
                "normalization": robust_descriptor_stats(descriptor_vectors),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")

    lengths = [len(r["description"]) for r in rows]
    print(
        f"[chebi20] wrote {len(rows)} molecules → {out} "
        f"(columns: {', '.join(fieldnames)})"
    )
    print(
        f"[chebi20] wrote {len(rows)} definitions → {definitions} "
        f"(description chars: min={min(lengths)} "
        f"median={sorted(lengths)[len(lengths) // 2]} max={max(lengths)})"
    )
    print(
        f"[chebi20] wrote {len(rows)} RDKit descriptor vectors "
        f"({len(RDKIT_DESCRIPTOR_MAP)} scalars each) → {rdkit_definitions}"
    )
    print(f"[chebi20] wrote RDKit descriptor map → {rdkit_map}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    RDLogger.DisableLog("rdApp.*")

    if args.download:
        download_splits(args.chebi_dir, args.splits)

    rows: list[dict[str, str]] = []
    for split in args.splits:
        path = os.path.join(args.chebi_dir, f"{split}.txt")
        if not os.path.isfile(path):
            print(
                f"ERROR: {path} not found — pass --download to fetch it, or point "
                f"--chebi-dir at a local ChEBI-20_data checkout",
                file=sys.stderr,
            )
            return 1
        split_rows = read_split(path, keep_lead_in=args.keep_lead_in)
        print(f"[chebi20] read {len(split_rows)} rows from {split}.txt")
        rows.extend(split_rows)

    rows = filter_and_canonicalize(
        rows,
        max_smiles_length=args.max_smiles_length,
        min_heavy_atoms=args.min_heavy_atoms,
    )
    if not rows:
        print("ERROR: no molecules survived filtering", file=sys.stderr)
        return 1

    random.Random(args.seed).shuffle(rows)
    if args.n:
        rows = rows[: args.n]

    write_outputs(
        rows,
        out=args.out,
        definitions=args.definitions,
        rdkit_definitions=args.rdkit_definitions,
        rdkit_map=args.rdkit_map,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
