#!/usr/bin/env python
"""Label molopt molecules with semantic categories taken from the ChEBI ontology.

Semantle's ``definitions.jsonl`` carries a ``category_normalized`` field that
``plot_cluster_pca`` and the subspace explorer colour points by. This script gives
molopt the same field, so a molopt run trained with ``--use-definition-embeds``
gets category-coloured geometry plots for free — nothing downstream needs to change,
because ``load_category_normalized_lookup`` already reads whatever task's
``definitions.jsonl`` the run config points at.

The labels are *derived*, not authored: ChEBI-20 molecules are ChEBI entries, so
each one can be looked up in ChEBI's own ontology and its class read off the
``is_a`` graph. See ``label_molopt_chebi_ontology.md`` for the full rationale.

    python scripts/label_molopt_chebi_ontology.py --download

Two independent axes are written per molecule:

``category`` / ``category_normalized``
    Structural class, from the ``is_a`` taxonomy ("steroid", "alkaloid", ...).

``role`` / ``role_normalized``
    What the molecule is *for*, from ChEBI's separate role ontology reached via
    ``has role`` ("antineoplastic", "agrochemical", ...). Roles are annotated far
    more sparsely than structure, so a third of the corpus is legitimately unknown
    on this axis.

Rewrites ``definitions.jsonl`` in place (``target`` and ``definition`` are copied
through untouched) and writes a ``definitions_chebi_map.json`` sidecar recording
the ontology release, the label vocabulary, and the realised counts.

Note that ``prepare_molopt_chebi20.py`` writes ``definitions.jsonl`` from scratch
and therefore drops these fields; re-run this script after it.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import os
import sys
import urllib.request
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional, Sequence

CHEBI_OBO_URL = "https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi_core.obo.gz"

DEFAULT_OBO = os.path.join(
    "data", "molopt", "raw", "chebi_ontology", "chebi_core.obo.gz"
)
DEFAULT_CORPUS = os.path.join("data", "molopt", "train", "chebi20.csv")
DEFAULT_DEFINITIONS = os.path.join("data", "molopt", "train", "definitions.jsonl")
DEFAULT_MAP = os.path.join("data", "molopt", "train", "definitions_chebi_map.json")

SCHEMA_VERSION = 1
UNKNOWN = "unknown"

# ChEBI relationship IDs (names are the ontology's own, see its [Typedef] stanzas).
HAS_ROLE = "RO:0000087"

# Relations naming a molecule that is, for classification purposes, the same
# compound in different clothes. ChEBI files a deprotonated acid under ion classes
# rather than under its own structural class, so without these ~22% of the corpus
# (every anion and zwitterion) lands in no structural class at all.
IDENTITY_BRIDGES: tuple[str, ...] = (
    "RO:0018033",  # is conjugate base of
    "RO:0018034",  # is conjugate acid of
    "RO:0018036",  # is tautomer of
)

# Structure additionally follows ``has part``, which recovers salts: the class of
# dextromethorphan hydrobromide is the class of dextromethorphan.
#
# ``has functional parent`` is pointedly *not* here even though it would cut the
# unlabelled share further. It relates a molecule to what it was derived from,
# which crosses class boundaries: it files butyl formate under "fatty acid" (via
# formic acid) and swallows most of the "lipid" bucket into "fatty acid" (via the
# acyl chain), both plainly wrong.
STRUCTURAL_BRIDGES: dict[str, str] = {
    "RO:0018033": "is conjugate base of",
    "RO:0018034": "is conjugate acid of",
    "RO:0018036": "is tautomer of",
    "BFO:0000051": "has part",
}

# Priority-ordered structural classes: the first entry with an ancestor in common
# with the molecule wins. Order is specific → generic, because ChEBI is a DAG and
# a molecule genuinely is_a several of these at once (every steroid is also a
# lipid and an organooxygen compound); "nearest ancestor by hop count" is not
# meaningful across branches, but a curated priority is.
STRUCTURAL_FRONTIER: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tetrapyrrole", ("CHEBI:26932",)),
    ("steroid", ("CHEBI:35341",)),
    ("terpenoid", ("CHEBI:26873", "CHEBI:24913", "CHEBI:23044")),
    ("alkaloid", ("CHEBI:22315",)),
    ("flavonoid", ("CHEBI:47916",)),
    ("polyketide", ("CHEBI:26188", "CHEBI:25106")),
    ("nucleoside/nucleotide", ("CHEBI:36976", "CHEBI:33838", "CHEBI:18282")),
    ("peptide/amino acid", ("CHEBI:16670", "CHEBI:33709", "CHEBI:36080")),
    # Ahead of "carbohydrate", whose ChEBI class spans carbohydrate *derivatives*
    # and would otherwise claim every glycosylated natural product.
    ("glycoside", ("CHEBI:24400",)),
    ("carbohydrate", ("CHEBI:78616", "CHEBI:16646")),
    ("fatty acid", ("CHEBI:35366", "CHEBI:26333")),
    ("lipid", ("CHEBI:18059",)),
    ("polyphenol", ("CHEBI:26195", "CHEBI:25036", "CHEBI:26776", "CHEBI:23403")),
    ("phenol", ("CHEBI:33853",)),
    ("benzenoid", ("CHEBI:33836",)),
    ("heterocycle", ("CHEBI:33833", "CHEBI:24532")),
    ("organohalogen", ("CHEBI:17792",)),
    ("organophosphorus", ("CHEBI:25710",)),
    ("organosulfur", ("CHEBI:33261",)),
    ("organic acid", ("CHEBI:64709",)),
    ("amine/amide", ("CHEBI:50047", "CHEBI:32988")),
    ("organooxygen", ("CHEBI:36963",)),
    ("hydrocarbon", ("CHEBI:24632",)),
    ("organometallic", ("CHEBI:25707",)),
    # ChEBI's top-level organic/inorganic split, as a backstop. "other organic"
    # must precede "inorganic": a salt of an organic drug reaches its inorganic
    # counterion through ``has part``, and calling it inorganic would be wrong.
    ("other organic", ("CHEBI:50860",)),
    ("inorganic", ("CHEBI:24835",)),
)

# Same priority idea on the role axis: specific applications first, with the
# near-universal "metabolite" last so it only claims what nothing else does.
ROLE_FRONTIER: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("antineoplastic", ("CHEBI:35610",)),
    (
        "antimicrobial",
        ("CHEBI:33281", "CHEBI:33282", "CHEBI:22587", "CHEBI:35718", "CHEBI:35442"),
    ),
    ("anti-inflammatory/analgesic", ("CHEBI:67079", "CHEBI:35480")),
    ("neuroactive drug", ("CHEBI:38867", "CHEBI:35469", "CHEBI:35623")),
    ("cardiovascular drug", ("CHEBI:35554",)),
    ("immunomodulator", ("CHEBI:50846",)),
    ("radiopharmaceutical", ("CHEBI:35232",)),
    ("drug", ("CHEBI:23888", "CHEBI:52217")),
    (
        "agrochemical",
        (
            "CHEBI:33286",
            "CHEBI:25944",
            "CHEBI:24527",
            "CHEBI:24852",
            "CHEBI:24127",
            "CHEBI:26155",
        ),
    ),
    ("dye/fluorochrome", ("CHEBI:37958", "CHEBI:51121", "CHEBI:51217")),
    ("toxin/contaminant", ("CHEBI:27026", "CHEBI:50904", "CHEBI:78298")),
    ("signalling molecule", ("CHEBI:24621", "CHEBI:25512", "CHEBI:62488", "CHEBI:26013")),
    ("nutrient/cofactor", ("CHEBI:33229", "CHEBI:23357", "CHEBI:33284")),
    ("food/flavour", ("CHEBI:78295", "CHEBI:35617", "CHEBI:48318")),
    ("antioxidant/protective", ("CHEBI:22586", "CHEBI:50267")),
    (
        "reagent/solvent",
        ("CHEBI:46787", "CHEBI:33893", "CHEBI:27780", "CHEBI:35195", "CHEBI:35225"),
    ),
    ("xenobiotic", ("CHEBI:35703",)),
    ("metabolite", ("CHEBI:25212",)),
)


@dataclass
class ChebiTerm:
    id: str
    name: str = ""
    is_a: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    bridges: dict[str, list[str]] = field(default_factory=dict)
    inchikey: Optional[str] = None
    obsolete: bool = False


class ChebiOntology:
    """The ``is_a`` graph, role annotations, and InChIKey index of one OBO release."""

    def __init__(
        self,
        terms: dict[str, ChebiTerm],
        *,
        version: str = "",
        date: str = "",
    ) -> None:
        self.terms = terms
        self.version = version
        self.date = date
        self._by_inchikey: dict[str, str] = {}
        self._by_skeleton: dict[str, list[str]] = collections.defaultdict(list)
        for term in terms.values():
            if term.obsolete or not term.inchikey:
                continue
            # Lowest ChEBI ID wins a collision: stable across releases, and ChEBI
            # hands out low IDs to the long-curated (so better annotated) entries.
            previous = self._by_inchikey.get(term.inchikey)
            if previous is None or _chebi_sort_key(term.id) < _chebi_sort_key(previous):
                self._by_inchikey[term.inchikey] = term.id
        for inchikey, term_id in self._by_inchikey.items():
            self._by_skeleton[_skeleton(inchikey)].append(term_id)

    def name_of(self, term_id: str) -> str:
        term = self.terms.get(term_id)
        return term.name if term else term_id

    def resolve_inchikey(
        self, inchikey: str, *, allow_skeleton: bool = True
    ) -> tuple[Optional[str], str]:
        """Map an InChIKey to a ChEBI ID, reporting how the match was made.

        The full key encodes connectivity, stereochemistry, and protonation. An
        exact hit is unambiguous. Failing that the 14-character skeleton (the
        connectivity layer alone) still identifies the compound up to stereo and
        charge, which is more than good enough for a class label — ChEBI-20
        canonicalization loses stereo on a couple of hundred molecules that would
        otherwise go unlabelled.
        """
        inchikey = (inchikey or "").strip()
        if not inchikey:
            return None, "none"
        exact = self._by_inchikey.get(inchikey)
        if exact is not None:
            return exact, "exact"
        if allow_skeleton:
            candidates = self._by_skeleton.get(_skeleton(inchikey))
            if candidates:
                return min(candidates, key=_chebi_sort_key), "skeleton"
        return None, "none"

    def structural_closure(self, term_id: str, *, max_bridge_hops: int) -> set[str]:
        """Every class the molecule belongs to, via ``is_a`` plus bounded bridges.

        ``is_a`` is followed without limit (it is pure subsumption and cannot
        drift). Bridge relations are capped because each one steps to a *different*
        molecule, and chaining them without bound wanders off into unrelated
        chemistry — two hops covers the common "anion of a derivative of X" case.
        """
        seen = {term_id}
        stack: list[tuple[str, int]] = [(term_id, 0)]
        while stack:
            current, hops = stack.pop()
            term = self.terms.get(current)
            if term is None:
                continue
            for parent in term.is_a:
                if parent not in seen:
                    seen.add(parent)
                    stack.append((parent, hops))
            if hops < max_bridge_hops:
                for targets in term.bridges.values():
                    for target in targets:
                        if target not in seen:
                            seen.add(target)
                            stack.append((target, hops + 1))
        return seen

    def role_closure(self, term_id: str) -> set[str]:
        """Every role the molecule has, including roles inherited from its classes.

        Only identity bridges are crossed here — never ``has part``, which structure
        does use. A part's purpose is not the whole's: every organic molecule
        ``has part`` a carbon atom, and ChEBI annotates carbon atom as an
        antineoplastic agent, so allowing it labels two thirds of the corpus
        "antineoplastic".
        """
        sources = {term_id}
        term = self.terms.get(term_id)
        if term is not None:
            for relation in IDENTITY_BRIDGES:
                sources.update(term.bridges.get(relation, ()))
        classes: set[str] = set()
        for source in sources:
            classes |= self._isa_ancestors(source)
        roles: set[str] = set()
        for klass in classes:
            klass_term = self.terms.get(klass)
            if klass_term is None:
                continue
            for role in klass_term.roles:
                roles |= self._isa_ancestors(role)
        return roles

    def _isa_ancestors(self, term_id: str) -> set[str]:
        seen = {term_id}
        stack = [term_id]
        while stack:
            term = self.terms.get(stack.pop())
            if term is None:
                continue
            for parent in term.is_a:
                if parent not in seen:
                    seen.add(parent)
                    stack.append(parent)
        return seen


def _skeleton(inchikey: str) -> str:
    return inchikey.split("-")[0]


def _chebi_sort_key(term_id: str) -> tuple[int, str]:
    _, _, digits = term_id.partition(":")
    return (int(digits), term_id) if digits.isdigit() else (sys.maxsize, term_id)


def _open_obo(path: str) -> Iterator[str]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:  # type: ignore[operator]
        yield from f


def parse_obo(path: str) -> ChebiOntology:
    """Read the ``[Term]`` stanzas of a ChEBI OBO file.

    Deliberately hand-rolled rather than pulling in an OBO library: the four line
    kinds below are all this script needs, and the file is a flat, line-oriented
    format.
    """
    terms: dict[str, ChebiTerm] = {}
    version = ""
    date = ""
    current: Optional[ChebiTerm] = None
    in_term = False

    for raw in _open_obo(path):
        line = raw.rstrip("\n")
        if line.startswith("["):
            if current is not None:
                terms[current.id] = current
            current = None
            in_term = line == "[Term]"
            continue
        if not line:
            continue
        key, _, value = line.partition(": ")
        value = value.strip()
        if not in_term:
            if key == "data-version":
                version = value
            elif key == "date":
                date = value
            continue
        if key == "id":
            if current is not None:
                terms[current.id] = current
            current = ChebiTerm(id=value)
        elif current is None:
            continue
        elif key == "name":
            current.name = value
        elif key == "is_a":
            current.is_a.append(_strip_comment(value))
        elif key == "is_obsolete":
            current.obsolete = value == "true"
        elif key == "relationship":
            parts = value.split()
            if len(parts) < 2:
                continue
            relation, target = parts[0], parts[1]
            if relation == HAS_ROLE:
                current.roles.append(target)
            elif relation in STRUCTURAL_BRIDGES:
                current.bridges.setdefault(relation, []).append(target)
        elif key == "property_value" and value.startswith("chemrof:inchi_key_string"):
            current.inchikey = _quoted(value)
    if current is not None:
        terms[current.id] = current

    return ChebiOntology(terms, version=version, date=date)


def _strip_comment(value: str) -> str:
    """``CHEBI:35341 ! steroid`` → ``CHEBI:35341``."""
    return value.split("!")[0].strip()


def _quoted(value: str) -> Optional[str]:
    start = value.find('"')
    end = value.find('"', start + 1)
    return value[start + 1 : end] if start >= 0 and end > start else None


def match_frontier(
    memberships: Iterable[str],
    frontier: Sequence[tuple[str, Sequence[str]]],
) -> tuple[str, str]:
    """First frontier entry the molecule belongs to → ``(label, matched ChEBI ID)``."""
    members = set(memberships)
    for label, term_ids in frontier:
        for term_id in term_ids:
            if term_id in members:
                return label, term_id
    return UNKNOWN, ""


def validate_frontier(
    ontology: ChebiOntology, frontier: Sequence[tuple[str, Sequence[str]]], kind: str
) -> None:
    """Fail loudly when a hard-coded ID has been retired or merged upstream.

    ChEBI obsoletes terms between releases. A frontier entry that no longer exists
    silently matches nothing, which looks like a corpus with no molecules of that
    class rather than like a bug.
    """
    problems = []
    for label, term_ids in frontier:
        for term_id in term_ids:
            term = ontology.terms.get(term_id)
            if term is None:
                problems.append(f"{kind} {label!r}: {term_id} not in ontology")
            elif term.obsolete:
                problems.append(f"{kind} {label!r}: {term_id} ({term.name}) obsolete")
    if problems:
        raise SystemExit(
            "ERROR: frontier is stale against this ChEBI release:\n  "
            + "\n  ".join(problems)
        )


def read_inchikeys(corpus_csv: str) -> dict[str, str]:
    """``smiles -> inchikey`` from the molopt corpus CSV."""
    with open(corpus_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"inchikey", "smiles"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{corpus_csv}: missing column(s) {sorted(missing)}")
        return {
            row["smiles"].strip(): row["inchikey"].strip()
            for row in reader
            if row.get("smiles")
        }


def read_definitions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def label_rows(
    rows: Sequence[dict],
    *,
    ontology: ChebiOntology,
    inchikeys: dict[str, str],
    max_bridge_hops: int,
    allow_skeleton: bool,
) -> list[dict]:
    """Rewrite each definitions row with its structural and role labels."""
    labelled: list[dict] = []
    for row in rows:
        target = str(row.get("target") or row.get("word") or "").strip()
        chebi_id, match = ontology.resolve_inchikey(
            inchikeys.get(target, ""), allow_skeleton=allow_skeleton
        )
        category = role = UNKNOWN
        category_id = role_id = ""
        if chebi_id is not None:
            category, category_id = match_frontier(
                ontology.structural_closure(chebi_id, max_bridge_hops=max_bridge_hops),
                STRUCTURAL_FRONTIER,
            )
            role, role_id = match_frontier(
                ontology.role_closure(chebi_id), ROLE_FRONTIER
            )
        labelled.append(
            {
                "target": target,
                "definition": str(row["definition"]),
                # ``category`` is ChEBI's own wording for the class that matched;
                # ``category_normalized`` is our short label, mirroring how the
                # semantle definitions pair a raw category with a collapsed one.
                "category": ontology.name_of(category_id) if category_id else UNKNOWN,
                "category_normalized": category,
                "role": ontology.name_of(role_id) if role_id else UNKNOWN,
                "role_normalized": role,
                "chebi_id": chebi_id or "",
                "chebi_match": match,
            }
        )
    return labelled


def write_definitions(path: str, rows: Sequence[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_map(
    path: str,
    rows: Sequence[dict],
    *,
    ontology: ChebiOntology,
    max_bridge_hops: int,
) -> dict:
    """Sidecar recording the ontology release, label vocabulary, and realised counts.

    The counts are what makes a re-run reviewable: ChEBI ships a new release every
    month, and a diff of this file shows whether an upgrade reshuffled the corpus.
    """
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "ontology": CHEBI_OBO_URL,
            "data_version": ontology.version,
            "date": ontology.date,
        },
        "max_bridge_hops": max_bridge_hops,
        "structural_bridges": STRUCTURAL_BRIDGES,
        "identity_bridges": list(IDENTITY_BRIDGES),
        "match": dict(collections.Counter(r["chebi_match"] for r in rows).most_common()),
        "categories": _axis_summary(
            rows, ontology, STRUCTURAL_FRONTIER, "category_normalized"
        ),
        "roles": _axis_summary(rows, ontology, ROLE_FRONTIER, "role_normalized"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return payload


def _axis_summary(
    rows: Sequence[dict],
    ontology: ChebiOntology,
    frontier: Sequence[tuple[str, Sequence[str]]],
    field_name: str,
) -> list[dict]:
    counts = collections.Counter(r[field_name] for r in rows)
    summary = [
        {
            "label": label,
            "priority": index,
            "chebi_terms": [
                {"id": term_id, "name": ontology.name_of(term_id)}
                for term_id in term_ids
            ],
            "count": counts.get(label, 0),
        }
        for index, (label, term_ids) in enumerate(frontier)
    ]
    summary.append(
        {
            "label": UNKNOWN,
            "priority": len(frontier),
            "chebi_terms": [],
            "count": counts.get(UNKNOWN, 0),
        }
    )
    return summary


def print_report(rows: Sequence[dict], payload: dict) -> None:
    total = len(rows)
    match = payload["match"]
    print(
        "[chebi-labels] InChIKey join: "
        + "  ".join(f"{k}={v} ({v / total:.1%})" for k, v in match.items())
    )
    for axis in ("categories", "roles"):
        print(f"[chebi-labels] {axis}:")
        for entry in sorted(payload[axis], key=lambda e: -e["count"]):
            if not entry["count"]:
                continue
            count = entry["count"]
            print(f"    {count:5d} {count / total:6.1%}  {entry['label']}")


def download_obo(path: str) -> None:
    if os.path.isfile(path):
        print(f"[chebi-labels] {path} already present")
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    print(f"[chebi-labels] downloading {CHEBI_OBO_URL} → {path}")
    urllib.request.urlretrieve(CHEBI_OBO_URL, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--chebi-obo", default=DEFAULT_OBO, help="ChEBI core OBO release.")
    p.add_argument(
        "--download",
        action="store_true",
        help="Fetch --chebi-obo first if it is not already on disk.",
    )
    p.add_argument(
        "--corpus", default=DEFAULT_CORPUS, help="molopt CSV supplying InChIKeys."
    )
    p.add_argument(
        "--definitions",
        default=DEFAULT_DEFINITIONS,
        help="Definitions JSONL to label (rewritten in place unless --out is given).",
    )
    p.add_argument("--out", default=None, help="Write labelled JSONL here instead.")
    p.add_argument("--map", default=DEFAULT_MAP, help="Output label-map JSON path.")
    p.add_argument(
        "--max-bridge-hops",
        type=int,
        default=2,
        help=(
            "How many non-is_a relations may be crossed when walking the "
            "structural taxonomy (0 disables ion/salt normalization)."
        ),
    )
    p.add_argument(
        "--no-skeleton-match",
        action="store_true",
        help="Require an exact InChIKey match, ignoring the connectivity fallback.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the label distribution without writing anything.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.download:
        download_obo(args.chebi_obo)
    if not os.path.isfile(args.chebi_obo):
        print(
            f"ERROR: {args.chebi_obo} not found — pass --download to fetch it",
            file=sys.stderr,
        )
        return 1

    print(f"[chebi-labels] parsing {args.chebi_obo}")
    ontology = parse_obo(args.chebi_obo)
    print(
        f"[chebi-labels] ChEBI {ontology.version or '?'} ({ontology.date or '?'}): "
        f"{len(ontology.terms)} terms"
    )
    validate_frontier(ontology, STRUCTURAL_FRONTIER, "category")
    validate_frontier(ontology, ROLE_FRONTIER, "role")

    inchikeys = read_inchikeys(args.corpus)
    rows = read_definitions(args.definitions)
    labelled = label_rows(
        rows,
        ontology=ontology,
        inchikeys=inchikeys,
        max_bridge_hops=args.max_bridge_hops,
        allow_skeleton=not args.no_skeleton_match,
    )

    out = args.out or args.definitions
    if args.dry_run:
        payload = {
            "match": dict(
                collections.Counter(r["chebi_match"] for r in labelled).most_common()
            ),
            "categories": _axis_summary(
                labelled, ontology, STRUCTURAL_FRONTIER, "category_normalized"
            ),
            "roles": _axis_summary(labelled, ontology, ROLE_FRONTIER, "role_normalized"),
        }
        print_report(labelled, payload)
        print("[chebi-labels] --dry-run: nothing written")
        return 0

    write_definitions(out, labelled)
    payload = write_map(
        args.map, labelled, ontology=ontology, max_bridge_hops=args.max_bridge_hops
    )
    print_report(labelled, payload)
    print(f"[chebi-labels] wrote {len(labelled)} labelled definitions → {out}")
    print(f"[chebi-labels] wrote label map → {args.map}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
