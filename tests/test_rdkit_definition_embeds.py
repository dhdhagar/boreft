"""Tests for ``scripts/analyze_rdkit_definition_embeds.py``.

Encoders are stubbed. The experiment's job is to load Qwen; the test suite's job
is to pin the pairing, the RBF x-axis, and the MolT5 overlay against inputs whose
answers are known by construction.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest

import numpy as np

from boreft.chem import (
    default_rdkit_map_path,
    normalize_rdkit_values,
    rdkit_similarity_from_values,
)
from boreft.text_similarity import stringify_rdkit_definition

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script():
    path = os.path.join(_REPO_ROOT, "scripts", "analyze_rdkit_definition_embeds.py")
    spec = importlib.util.spec_from_file_location(
        "analyze_rdkit_definition_embeds", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ard = _load_script()
MAP = default_rdkit_map_path()
PROSE = "a member of the class of biphenyls that is benzidine."


def _z_encode_fn(pairs, molt5_definition):
    """Embed each string as the L2-normalized descriptor z of its values.

    Cosine then tracks property-space geometry, so Spearman vs. rdkit_sim must
    be strongly positive in *both* conditions — the stub ignores the ChEBI
    sentence. That is the wiring check: if a pair's left/right embeddings were
    swapped, the correlation would collapse.
    """
    z_by_text = {}
    for pair in pairs:
        for values in (pair.left, pair.right):
            z = normalize_rdkit_values(values, map_path=MAP)
            z_by_text[ard.format_definition(values, "")] = z
            z_by_text[ard.format_definition(values, molt5_definition)] = z
    z_by_text[molt5_definition.strip()] = np.ones(z.shape, dtype=np.float64)

    def encode(texts):
        return np.asarray([z_by_text[t] for t in texts], dtype=np.float64)

    return encode


class CoerceAndFormatTests(unittest.TestCase):
    def test_counts_snap_to_nonnegative_ints(self):
        raw = [300.1234567, 1.9, 80.0, -0.4, 4.6, 3.2, -1.4, 1.2, 2.4, 0.6]
        out = ard.coerce_descriptor_values(raw)
        self.assertEqual(out[3], 0)
        self.assertEqual(out[4], 5)
        self.assertEqual(out[6], -1)
        self.assertEqual(out[7], 1.0)
        self.assertIsInstance(out[0], float)
        self.assertIsInstance(out[3], int)

    def test_fraction_is_clipped_to_the_unit_interval(self):
        raw = [300.0, 1.0, 80.0, 1, 1, 1, 0, 1.7, 0, 0]
        self.assertEqual(ard.coerce_descriptor_values(raw)[7], 1.0)
        raw = [300.0, 1.0, 80.0, 1, 1, 1, 0, -0.3, 0, 0]
        self.assertEqual(ard.coerce_descriptor_values(raw)[7], 0.0)

    def test_rdkit_only_is_the_training_suffix(self):
        values = ard.coerce_descriptor_values(
            [314.221, 1.9969, 80.26, 2, 4, 4, 0, 0.533333, 0, 1]
        )
        text = ard.format_definition(values, "")
        self.assertTrue(text.startswith("2D properties: "))
        self.assertEqual(text, f"2D properties: {stringify_rdkit_definition(values)}")
        self.assertNotIn(PROSE, text)

    def test_molt5_overlay_prepends_the_same_sentence(self):
        values = ard.coerce_descriptor_values(
            [314.221, 1.9969, 80.26, 2, 4, 4, 0, 0.533333, 0, 1]
        )
        text = ard.format_definition(values, PROSE)
        self.assertTrue(text.startswith(f"{PROSE} 2D properties: "))
        self.assertIn(stringify_rdkit_definition(values), text)

    def test_unknown_condition_raises(self):
        pair = ard.SyntheticPair((0,) * 10, (0,) * 10, 1.0, 1.0)
        with self.assertRaises(ValueError):
            ard.pair_texts(pair, "nope", PROSE)


class PairBuilderTests(unittest.TestCase):
    def test_identity_target_keeps_the_vector_and_scores_one(self):
        pairs = ard.build_pairs(3, [1.0], seed=0, map_path=MAP)
        self.assertEqual(len(pairs), 3)
        for pair in pairs:
            self.assertEqual(pair.left, pair.right)
            self.assertEqual(pair.rdkit_sim, 1.0)
            self.assertEqual(pair.target_rdkit_sim, 1.0)

    def test_actual_sim_stays_near_the_target_after_snapping(self):
        """clogp residual must undo the integer-snap bias, especially at low s."""
        pairs = ard.build_pairs(40, [1.0, 0.9, 0.5, 0.1], seed=0, map_path=MAP)
        by_target: dict[float, list[float]] = {}
        for pair in pairs:
            by_target.setdefault(pair.target_rdkit_sim, []).append(pair.rdkit_sim)
        for target, actuals in by_target.items():
            mean = float(np.mean(actuals))
            self.assertLess(
                abs(mean - target),
                0.03,
                f"target {target:g}: mean actual rdkit_sim {mean:.3f}",
            )

    def test_recorded_sim_is_production_rbf(self):
        pairs = ard.build_pairs(5, [1.0, 0.5], seed=1, map_path=MAP)
        for pair in pairs:
            self.assertAlmostEqual(
                pair.rdkit_sim,
                rdkit_similarity_from_values(pair.left, pair.right, map_path=MAP),
            )

    def test_pair_count_is_bases_times_targets(self):
        pairs = ard.build_pairs(4, [1.0, 0.7, 0.3], seed=2, map_path=MAP)
        self.assertEqual(len(pairs), 12)

    def test_pairs_are_seeded(self):
        a = ard.build_pairs(6, [1.0, 0.5], seed=3, map_path=MAP)
        b = ard.build_pairs(6, [1.0, 0.5], seed=3, map_path=MAP)
        c = ard.build_pairs(6, [1.0, 0.5], seed=4, map_path=MAP)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_nonunit_target_usually_moves_the_vector(self):
        pairs = [
            p
            for p in ard.build_pairs(20, [0.5], seed=0, map_path=MAP)
        ]
        moved = sum(p.left != p.right for p in pairs)
        self.assertGreater(moved, 15)

    def test_rejects_nonpositive_sim(self):
        with self.assertRaises(ValueError):
            ard.build_pairs(1, [0.0], seed=0, map_path=MAP)


class RunExperimentTests(unittest.TestCase):
    def setUp(self):
        self.pairs = ard.build_pairs(8, [1.0, 0.7, 0.3], seed=0, map_path=MAP)
        self.encode = _z_encode_fn(self.pairs, PROSE)
        self.result = ard.run_experiment(self.pairs, PROSE, self.encode)

    def test_identity_cosine_is_one_in_both_conditions(self):
        """Same string dotted with itself is 1 after L2-norm; not an encoder result."""
        for name in ard.CONDITIONS:
            ident = self.result["conditions"][name]["identity_cosine"]["mean"]
            self.assertAlmostEqual(ident, 1.0, places=6)

    def test_prose_only_self_cosine_is_one(self):
        self.assertAlmostEqual(self.result["prose_only_self_cosine"], 1.0)

    def test_z_stub_tracks_rdkit_sim_even_with_prose(self):
        """The stub ignores ChEBI text, so drowning must not shrink the gap."""
        only = self.result["conditions"][ard.RDKIT_ONLY]
        plus = self.result["conditions"][ard.MOLT5_PLUS_RDKIT]
        self.assertGreater(only.get("pearson_r", 0.0), 0.5)
        self.assertGreater(plus.get("pearson_r", 0.0), 0.5)
        if "spearman_rho" in only:
            self.assertGreater(only["spearman_rho"], 0.5)
            self.assertGreater(plus["spearman_rho"], 0.5)
        self.assertEqual(
            only["n_correlation_pairs"],
            sum(p.left != p.right for p in self.pairs),
        )
        np.testing.assert_allclose(only["_cosine"], plus["_cosine"], atol=1e-9)
        ratio = self.result["drowning"]["gap_ratio_molt5_over_rdkit_only"]
        self.assertAlmostEqual(ratio, 1.0, places=6)

    def test_cosine_gap_is_positive_when_embeddings_are_the_descriptors(self):
        gap = self.result["conditions"][ard.RDKIT_ONLY][
            "cosine_gap_identity_minus_farthest"
        ]
        self.assertGreater(gap, 0.0)

    def test_empty_pairs_are_rejected(self):
        with self.assertRaises(ValueError):
            ard.run_experiment([], PROSE, lambda texts: np.ones((len(texts), 4)))

    def test_empty_prose_is_rejected(self):
        with self.assertRaises(ValueError):
            ard.run_experiment(self.pairs, "  ", self.encode)

    def test_report_without_raw_arrays_is_json_serializable(self):
        stripped = ard.strip_arrays(self.result)
        for name in ard.CONDITIONS:
            self.assertNotIn("_cosine", stripped["conditions"][name])
            self.assertIn("pearson_r", stripped["conditions"][name])
        json.dumps(stripped)


class LoadMolt5Tests(unittest.TestCase):
    def test_samples_a_nonempty_definition(self):
        path = os.path.join(self._tmp(), "definitions.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"target": "CCO", "definition": "ethanol"}) + "\n")
            f.write(json.dumps({"target": "CCN", "definition": "ethylamine"}) + "\n")
        self.assertIn(
            ard.load_molt5_definition(path, seed=0),
            {"ethanol", "ethylamine"},
        )

    def test_empty_file_raises(self):
        path = os.path.join(self._tmp(), "empty.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"target": "CCO", "definition": "  "}) + "\n")
        with self.assertRaises(ValueError):
            ard.load_molt5_definition(path, seed=0)

    def _tmp(self):
        import tempfile

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return tmp.name


if __name__ == "__main__":
    unittest.main()
