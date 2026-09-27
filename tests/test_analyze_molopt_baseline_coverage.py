from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_molopt_baseline_coverage import (  # noqa: E402
    aabb_distance,
    aggregate_rows,
    definition_lookup_by_molecule,
    load_method_molecules,
    merge_molecules,
    molecules_above,
    molecules_from_record,
    nearest_vertex_distance,
    outside_box,
    top_molecules,
)


class MoleculeExtractionTest(unittest.TestCase):
    def test_single_sample_uses_oracle_canonical_and_score(self):
        record = {
            "solution": "CCO",
            "score": 0.2,
            "sample_count": 1,
            "sample_scores": [0.2],
            "solution_samples": ["CCO"],
            "components": {"canonical": "CCO", "oracle_score": 0.25, "valid": True},
        }
        self.assertEqual(molecules_from_record(record), [("CCO", 0.25)])

    def test_multi_sample_keeps_each_sample_score(self):
        record = {
            "solution": "CCO",
            "score": 0.3,
            "sample_scores": [0.1, 0.9],
            "solution_samples": ["CCO", "CCN"],
            "components": {"canonical": "CCN", "oracle_score": 0.5},
        }
        self.assertEqual(
            molecules_from_record(record),
            [("CCO", 0.1), ("CCN", 0.9)],
        )

    def test_invalid_smiles_are_dropped(self):
        record = {
            "solution": "not a molecule",
            "score": 0.0,
            "components": {"canonical": "", "oracle_score": 0.0, "valid": False},
        }
        self.assertEqual(molecules_from_record(record), [])


class SelectionTest(unittest.TestCase):
    def test_baseline_set_is_strictly_above_boreft_best_and_unique(self):
        boreft = {
            "CCO": {"smiles": "CCO", "score": 0.7, "sources": []},
            "CCN": {"smiles": "CCN", "score": 0.4, "sources": []},
        }
        baselines = {
            "CCC": {"smiles": "CCC", "score": 0.9, "sources": [{"method": "opro"}]},
            "CCO": {"smiles": "CCO", "score": 0.7, "sources": []},
            "CCCl": {"smiles": "CCCl", "score": 0.71, "sources": []},
        }
        threshold = max(row["score"] for row in boreft.values())
        above = molecules_above(baselines, threshold)
        self.assertEqual([row["smiles"] for row in above], ["CCC", "CCCl"])
        matched = top_molecules(boreft, len(above))
        self.assertEqual([row["smiles"] for row in matched], ["CCO", "CCN"])

    def test_merge_keeps_the_higher_score_and_both_sources(self):
        merged = merge_molecules(
            [
                {"CCO": {"smiles": "CCO", "score": 0.2, "sources": [{"method": "a"}]}},
                {"CCO": {"smiles": "CCO", "score": 0.8, "sources": [{"method": "b"}]}},
            ]
        )
        self.assertEqual(merged["CCO"]["score"], 0.8)
        self.assertEqual(
            [source["method"] for source in merged["CCO"]["sources"]],
            ["a", "b"],
        )

    def test_load_observations_keeps_the_best_score_per_molecule(self):
        with self._dir() as raw:
            root = Path(raw)
            path = root / "random_sampling_mu" / "DRD2" / "seed_1"
            path.mkdir(parents=True)
            rows = [
                {"solution": "CCO", "score": 0.2, "components": {"canonical": "CCO", "oracle_score": 0.2}, "phase": "search", "seed": 1},
                {"solution": "CCO", "score": 0.5, "components": {"canonical": "CCO", "oracle_score": 0.5}, "phase": "search", "seed": 1},
                {"solution": "bad", "score": 0.0, "components": {"canonical": "", "oracle_score": 0.0}, "phase": "search", "seed": 1},
            ]
            (path / "observations.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            found = load_method_molecules(root, "random_sampling_mu", "DRD2", [1])
        self.assertEqual(set(found), {"CCO"})
        self.assertEqual(found["CCO"]["score"], 0.5)

    def _dir(self):
        import tempfile

        return tempfile.TemporaryDirectory()


class GeometryTest(unittest.TestCase):
    def test_inside_box_has_zero_distance(self):
        bounds = np.array([[0.0, 0.0], [1.0, 1.0]])
        self.assertEqual(aabb_distance(np.array([0.2, 0.8]), bounds), 0.0)
        self.assertFalse(outside_box(0.0))

    def test_outside_box_distance_is_euclidean(self):
        bounds = np.array([[0.0, 0.0], [1.0, 1.0]])
        distance = aabb_distance(np.array([1.0, 1.0 + 3.0]), bounds)
        self.assertAlmostEqual(distance, 3.0)
        self.assertTrue(outside_box(distance))

    def test_nearest_vertex_is_zero_for_a_training_embedding(self):
        train = np.array([[1.0, 0.0], [0.0, 1.0]])
        distance = nearest_vertex_distance(train[:1], train)
        self.assertAlmostEqual(float(distance[0]), 0.0, places=6)

    def test_nearest_vertex_renormalizes_before_the_distance(self):
        train = np.array([[2.0, 0.0]])
        distance = nearest_vertex_distance(np.array([[4.0, 0.0]]), train)
        self.assertAlmostEqual(float(distance[0]), 0.0, places=6)


class DefinitionPlacementTest(unittest.TestCase):
    def test_lookup_is_also_indexed_by_canonical_smiles(self):
        lookup = definition_lookup_by_molecule({"OCC": "ethanol definition"})
        self.assertEqual(lookup["CCO"], "ethanol definition")
        self.assertEqual(lookup["OCC"], "ethanol definition")

    def test_unplaced_molecules_are_excluded_from_box_stats(self):
        summary = aggregate_rows(
            [
                {
                    "coverage_error": 0.2,
                    "box_distance": None,
                    "outside_box": None,
                    "in_train_catalog": False,
                    "score": 0.9,
                },
                {
                    "coverage_error": 0.0,
                    "box_distance": 0.0,
                    "outside_box": False,
                    "in_train_catalog": True,
                    "score": 0.8,
                },
            ]
        )
        self.assertEqual(summary["n_unplaced"], 1)
        self.assertEqual(summary["n_outside_box"], 0)
        self.assertEqual(summary["frac_outside_box"], 0.0)
        self.assertEqual(summary["box_distance"]["n"], 1)
        self.assertEqual(summary["coverage_error"]["n"], 2)


if __name__ == "__main__":
    unittest.main()
