"""Tests for ``scripts/label_molopt_chebi_ontology.py``.

The graph-walking rules are pinned against a hand-built miniature ontology whose
answers are known by construction, because the behaviours that matter are the ones
that are wrong in an interesting way: an anion has to find its neutral parent, a
salt has to find its organic component, and a molecule must *not* inherit the role
of a fragment it happens to contain.

A second group asserts the committed ``definitions.jsonl`` still carries the labels
that ``plot_cluster_pca`` colours by, so a stray re-run of
``prepare_molopt_chebi20.py`` (which rewrites the file without them) is caught here
rather than as an all-grey PCA plot.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script():
    """Import the script by path; it lives in scripts/, not the package."""
    path = os.path.join(_REPO_ROOT, "scripts", "label_molopt_chebi_ontology.py")
    spec = importlib.util.spec_from_file_location("label_molopt_chebi_ontology", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lbl = _load_script()

DEFINITIONS = os.path.join(
    _REPO_ROOT, "data", "molopt", "train", "definitions.jsonl"
)
CHEBI_MAP = os.path.join(
    _REPO_ROOT, "data", "molopt", "train", "definitions_chebi_map.json"
)

# A miniature ChEBI: an acid and its anion, a salt of that acid with an inorganic
# counterion, and a role annotation parked on a fragment the salt contains.
MINI_OBO = """format-version: 1.2
data-version: chebi/999
date: 01:01:2026 00:00

[Term]
id: CHEBI:1
name: organic molecular entity

[Term]
id: CHEBI:2
name: steroid
is_a: CHEBI:1

[Term]
id: CHEBI:3
name: cholic acid
is_a: CHEBI:2
property_value: chemrof:inchi_key_string "AAAAAAAAAAAAAA-BBBBBBBBBB-C" xsd:string

[Term]
id: CHEBI:4
name: cholate
is_a: CHEBI:9 ! organic anion
relationship: RO:0018033 CHEBI:3 ! is conjugate base of cholic acid
property_value: chemrof:inchi_key_string "DDDDDDDDDDDDDD-EEEEEEEEEE-F" xsd:string

[Term]
id: CHEBI:5
name: sodium cholate
relationship: BFO:0000051 CHEBI:3 ! has part cholic acid
relationship: BFO:0000051 CHEBI:7 ! has part sodium atom
property_value: chemrof:inchi_key_string "GGGGGGGGGGGGGG-HHHHHHHHHH-I" xsd:string

[Term]
id: CHEBI:6
name: antineoplastic agent

[Term]
id: CHEBI:7
name: sodium atom
relationship: RO:0000087 CHEBI:6 ! has role antineoplastic agent

[Term]
id: CHEBI:8
name: retired steroid
is_a: CHEBI:2
is_obsolete: true

[Term]
id: CHEBI:9
name: organic anion
is_a: CHEBI:1
"""

STRUCTURAL = (("steroid", ("CHEBI:2",)), ("other organic", ("CHEBI:1",)))
ROLES = (("antineoplastic", ("CHEBI:6",)),)


def _mini_ontology():
    with tempfile.NamedTemporaryFile(
        "w", suffix=".obo", delete=False, encoding="utf-8"
    ) as f:
        f.write(MINI_OBO)
        path = f.name
    try:
        return lbl.parse_obo(path)
    finally:
        os.unlink(path)


class ParseObo(unittest.TestCase):
    def setUp(self):
        self.ontology = _mini_ontology()

    def test_reads_release_metadata(self):
        self.assertEqual(self.ontology.version, "chebi/999")
        self.assertEqual(self.ontology.date, "01:01:2026 00:00")

    def test_strips_trailing_name_comment_from_is_a(self):
        self.assertEqual(self.ontology.terms["CHEBI:4"].is_a, ["CHEBI:9"])

    def test_separates_roles_from_structural_bridges(self):
        self.assertEqual(self.ontology.terms["CHEBI:7"].roles, ["CHEBI:6"])
        self.assertEqual(
            self.ontology.terms["CHEBI:5"].bridges["BFO:0000051"],
            ["CHEBI:3", "CHEBI:7"],
        )

    def test_obsolete_terms_are_flagged(self):
        self.assertTrue(self.ontology.terms["CHEBI:8"].obsolete)


class ResolveInchikey(unittest.TestCase):
    def setUp(self):
        self.ontology = _mini_ontology()

    def test_exact_key_matches(self):
        self.assertEqual(
            self.ontology.resolve_inchikey("AAAAAAAAAAAAAA-BBBBBBBBBB-C"),
            ("CHEBI:3", "exact"),
        )

    def test_falls_back_to_connectivity_skeleton(self):
        """Differing stereo/charge layers still identify the same compound."""
        self.assertEqual(
            self.ontology.resolve_inchikey("AAAAAAAAAAAAAA-ZZZZZZZZZZ-N"),
            ("CHEBI:3", "skeleton"),
        )

    def test_skeleton_fallback_can_be_disabled(self):
        self.assertEqual(
            self.ontology.resolve_inchikey(
                "AAAAAAAAAAAAAA-ZZZZZZZZZZ-N", allow_skeleton=False
            ),
            (None, "none"),
        )

    def test_unknown_key_reports_no_match(self):
        self.assertEqual(
            self.ontology.resolve_inchikey("ZZZZZZZZZZZZZZ-ZZZZZZZZZZ-Z"),
            (None, "none"),
        )


class StructuralClosure(unittest.TestCase):
    def setUp(self):
        self.ontology = _mini_ontology()

    def _category(self, term_id, hops=2):
        return lbl.match_frontier(
            self.ontology.structural_closure(term_id, max_bridge_hops=hops),
            STRUCTURAL,
        )[0]

    def test_neutral_molecule_uses_plain_is_a(self):
        self.assertEqual(self._category("CHEBI:3"), "steroid")

    def test_anion_reaches_its_conjugate_acid(self):
        self.assertEqual(self._category("CHEBI:4"), "steroid")

    def test_salt_reaches_its_organic_component(self):
        self.assertEqual(self._category("CHEBI:5"), "steroid")

    def test_zero_hops_strands_the_anion(self):
        """Without bridges the anion only knows it is an ion."""
        self.assertEqual(self._category("CHEBI:4", hops=0), "other organic")

    def test_priority_order_decides_between_valid_ancestors(self):
        """Cholic acid is_a both frontier entries; the earlier one must win."""
        closure = self.ontology.structural_closure("CHEBI:3", max_bridge_hops=2)
        self.assertLessEqual({"CHEBI:1", "CHEBI:2"}, closure)
        self.assertEqual(
            lbl.match_frontier(closure, STRUCTURAL), ("steroid", "CHEBI:2")
        )
        reversed_frontier = tuple(reversed(STRUCTURAL))
        self.assertEqual(
            lbl.match_frontier(closure, reversed_frontier)[0], "other organic"
        )

    def test_unmatched_membership_is_unknown(self):
        self.assertEqual(lbl.match_frontier({"CHEBI:404"}, STRUCTURAL), ("unknown", ""))


class RoleClosure(unittest.TestCase):
    def setUp(self):
        self.ontology = _mini_ontology()

    def _role(self, term_id):
        return lbl.match_frontier(self.ontology.role_closure(term_id), ROLES)[0]

    def test_role_is_not_inherited_from_a_contained_part(self):
        """The whole point of keeping ``has part`` off the role axis."""
        self.assertEqual(self._role("CHEBI:5"), "unknown")

    def test_role_is_inherited_from_a_conjugate_acid(self):
        self.ontology.terms["CHEBI:3"].roles.append("CHEBI:6")
        self.assertEqual(self._role("CHEBI:4"), "antineoplastic")

    def test_role_is_inherited_from_a_superclass(self):
        self.ontology.terms["CHEBI:2"].roles.append("CHEBI:6")
        self.assertEqual(self._role("CHEBI:3"), "antineoplastic")


class ValidateFrontier(unittest.TestCase):
    def setUp(self):
        self.ontology = _mini_ontology()

    def test_accepts_live_terms(self):
        lbl.validate_frontier(self.ontology, STRUCTURAL, "category")

    def test_rejects_missing_term(self):
        with self.assertRaises(SystemExit):
            lbl.validate_frontier(self.ontology, (("bogus", ("CHEBI:404",)),), "category")

    def test_rejects_obsolete_term(self):
        """A retired ID silently matches nothing, which reads as an empty class."""
        with self.assertRaises(SystemExit):
            lbl.validate_frontier(self.ontology, (("gone", ("CHEBI:8",)),), "category")


class LabelRows(unittest.TestCase):
    def test_copies_target_and_definition_through_unchanged(self):
        ontology = _mini_ontology()
        rows = [{"target": "CCO", "definition": "  an alcohol  "}]
        labelled = lbl.label_rows(
            rows,
            ontology=ontology,
            inchikeys={"CCO": "AAAAAAAAAAAAAA-BBBBBBBBBB-C"},
            max_bridge_hops=2,
            allow_skeleton=True,
        )
        self.assertEqual(labelled[0]["target"], "CCO")
        self.assertEqual(labelled[0]["definition"], "  an alcohol  ")
        self.assertEqual(labelled[0]["chebi_id"], "CHEBI:3")
        self.assertEqual(labelled[0]["chebi_match"], "exact")

    def test_unjoinable_molecule_is_labelled_unknown(self):
        """``unknown`` is already a meta-cluster for plot_cluster_pca."""
        labelled = lbl.label_rows(
            [{"target": "CCO", "definition": "d"}],
            ontology=_mini_ontology(),
            inchikeys={},
            max_bridge_hops=2,
            allow_skeleton=True,
        )
        self.assertEqual(labelled[0]["category_normalized"], "unknown")
        self.assertEqual(labelled[0]["role_normalized"], "unknown")
        self.assertEqual(labelled[0]["chebi_match"], "none")


class CommittedLabels(unittest.TestCase):
    """The shipped molopt definitions must stay colourable by category."""

    @classmethod
    def setUpClass(cls):
        with open(DEFINITIONS, encoding="utf-8") as f:
            cls.rows = [json.loads(line) for line in f if line.strip()]

    def test_every_molecule_has_both_labels(self):
        for field in ("category", "category_normalized", "role", "role_normalized"):
            missing = [r["target"] for r in self.rows if not r.get(field)]
            self.assertEqual(missing[:5], [], f"{len(missing)} rows lack {field}")

    def test_labels_come_from_the_declared_vocabulary(self):
        for field, frontier in (
            ("category_normalized", lbl.STRUCTURAL_FRONTIER),
            ("role_normalized", lbl.ROLE_FRONTIER),
        ):
            allowed = {label for label, _ in frontier} | {lbl.UNKNOWN}
            self.assertEqual(
                {r[field] for r in self.rows} - allowed, set(), f"stray {field}"
            )

    def test_most_molecules_are_classified(self):
        """A join regression shows up as a sudden pile of unknowns, not a crash."""
        unknown = sum(1 for r in self.rows if r["category_normalized"] == lbl.UNKNOWN)
        self.assertLess(unknown / len(self.rows), 0.10)

    def test_map_sidecar_counts_match_the_definitions(self):
        with open(CHEBI_MAP, encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(payload["schema_version"], lbl.SCHEMA_VERSION)
        counts = {e["label"]: e["count"] for e in payload["categories"]}
        for label in {r["category_normalized"] for r in self.rows}:
            observed = sum(
                1 for r in self.rows if r["category_normalized"] == label
            )
            self.assertEqual(counts.get(label), observed, label)


if __name__ == "__main__":
    unittest.main()
