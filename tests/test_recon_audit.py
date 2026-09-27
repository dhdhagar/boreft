from __future__ import annotations

import unittest

import numpy as np
import torch

from boreft.data_utils import IGNORE_INDEX
from boreft.recon_audit import (
    code_in_bounds,
    generation_word_idx,
    identity_match,
    interpret_inversion,
    inversion_inits,
    items_words,
    mean_finite,
    project_to_bounds,
    rate,
    select_audit_panel,
    shuffled_index,
    teacher_forced_from_logits,
)


class IdentityMatchTests(unittest.TestCase):
    def test_raw_vs_canonical_ethanol(self):
        row = identity_match("CCO", "OCC")
        self.assertFalse(row.raw_exact)
        self.assertTrue(row.canonical_exact)
        self.assertTrue(row.graph_exact)
        self.assertTrue(row.valid)
        self.assertAlmostEqual(row.tanimoto or 0.0, 1.0)

    def test_stereo_preserved_unless_graph(self):
        row = identity_match("F/C=C/F", r"F/C=C\F")
        self.assertTrue(row.valid)
        self.assertFalse(row.canonical_exact)
        self.assertTrue(row.graph_exact)

    def test_invalid_has_no_tanimoto(self):
        row = identity_match("CCO", "not-a-molecule")
        self.assertFalse(row.valid)
        self.assertFalse(row.raw_exact)
        self.assertFalse(row.canonical_exact)
        self.assertIsNone(row.tanimoto)

    def test_mist_tags_unwrap(self):
        row = identity_match("[START_SMILES]CCO[END_SMILES]", "CCO")
        self.assertTrue(row.raw_exact)
        self.assertTrue(row.canonical_exact)


class PanelTests(unittest.TestCase):
    def test_includes_catalog_and_is_unique(self):
        smiles = ["C" * (i + 1) for i in range(40)]
        panel = select_audit_panel(smiles, catalog_indices=[2, 39, 2], n=8, seed=0)
        self.assertEqual(panel[0], 2)
        self.assertEqual(panel[1], 39)
        self.assertEqual(len(panel), len(set(panel)))
        self.assertEqual(len(panel), 8)

    def test_catalog_survives_small_n(self):
        smiles = ["CCO", "CCC", "CC"]
        panel = select_audit_panel(smiles, catalog_indices=[0, 2, 1], n=1, seed=0)
        self.assertEqual(set(panel), {0, 1, 2})

    def test_length_stratified_and_deterministic(self):
        smiles = ["C" * (i + 1) for i in range(50)]
        a = select_audit_panel(smiles, catalog_indices=[0], n=10, seed=7)
        b = select_audit_panel(smiles, catalog_indices=[0], n=10, seed=7)
        self.assertEqual(a, b)
        extras = [i for i in a if i != 0]
        lengths = [len(smiles[i]) for i in extras]
        self.assertLess(min(lengths), max(lengths))

    def test_shuffled_index_skips_self(self):
        self.assertEqual(shuffled_index(3, [3, 8, 1], 10), 8)
        self.assertEqual(shuffled_index(8, [3, 8, 1], 10), 1)
        self.assertEqual(shuffled_index(1, [3, 8, 1], 10), 3)
        self.assertEqual(shuffled_index(0, [0], 4), 1)


class TeacherForcedStatsTests(unittest.TestCase):
    def test_all_greedy_tokens(self):
        labels = torch.tensor([[IGNORE_INDEX, 1, 2]])
        logits = torch.zeros(1, 3, 4)
        logits[0, 0, 1] = 5.0
        logits[0, 1, 2] = 5.0
        stats = teacher_forced_from_logits(logits, labels)
        self.assertEqual(stats["n_tokens"], 2)
        self.assertAlmostEqual(stats["token_acc"], 1.0)
        self.assertTrue(stats["greedy_prefix"])
        self.assertLess(stats["nll"], 0.2)

    def test_one_wrong_token_breaks_greedy_prefix(self):
        labels = torch.tensor([[IGNORE_INDEX, 1, 2]])
        logits = torch.zeros(1, 3, 4)
        logits[0, 0, 1] = 5.0
        logits[0, 1, 0] = 5.0
        stats = teacher_forced_from_logits(logits, labels)
        self.assertAlmostEqual(stats["token_acc"], 0.5)
        self.assertFalse(stats["greedy_prefix"])


class InversionHelperTests(unittest.TestCase):
    def test_inits_start_at_mu_and_stay_in_box(self):
        mu = np.array([0.0, 1.0], dtype=np.float32)
        std = np.array([0.1, 0.1], dtype=np.float32)
        bounds = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
        inits = inversion_inits(mu, std=std, bounds=bounds, n_restarts=4, seed=1)
        self.assertEqual(len(inits), 4)
        np.testing.assert_array_equal(inits[0], mu)
        for point in inits:
            self.assertTrue(code_in_bounds(point, bounds))

    def test_project_to_bounds(self):
        bounds = np.array([[0.0, -1.0], [1.0, 1.0]], dtype=np.float32)
        point = torch.tensor([1.5, -2.0])
        clipped = project_to_bounds(point, bounds)
        self.assertAlmostEqual(float(clipped[0]), 1.0)
        self.assertAlmostEqual(float(clipped[1]), -1.0)

    def test_interpret_table(self):
        self.assertEqual(
            interpret_inversion(
                recovered_in_bounds=True,
                recovered_unconstrained=True,
                nll_improved=True,
                inverted_in_bounds=True,
            ),
            "encoder",
        )
        self.assertEqual(
            interpret_inversion(
                recovered_in_bounds=False,
                recovered_unconstrained=True,
                nll_improved=True,
                inverted_in_bounds=False,
            ),
            "bounds",
        )
        self.assertEqual(
            interpret_inversion(
                recovered_in_bounds=False,
                recovered_unconstrained=False,
                nll_improved=True,
                inverted_in_bounds=True,
            ),
            "capacity_or_decoding",
        )
        self.assertEqual(
            interpret_inversion(
                recovered_in_bounds=False,
                recovered_unconstrained=False,
                nll_improved=False,
                inverted_in_bounds=True,
            ),
            "no_improvement",
        )


class GenerationWordIdxTests(unittest.TestCase):
    def test_ints_stay_ints_for_mu_lookup(self):
        self.assertEqual(generation_word_idx(626), 626)
        self.assertIsInstance(generation_word_idx(np.int64(3)), int)
        self.assertNotIsInstance(generation_word_idx(3), np.ndarray)

    def test_vectors_stay_raw_codes(self):
        code = np.array([0.1, -0.2], dtype=np.float32)
        out = generation_word_idx(code)
        self.assertIsInstance(out, np.ndarray)
        np.testing.assert_array_equal(out, code)


class MiscHelperTests(unittest.TestCase):
    def test_named_summary_flattens_recon_tree(self):
        from boreft.search_wandb import _named_summary

        out = _named_summary(
            {
                "summary": {
                    "nll_own": 1.25,
                    "greedy_mu": {"canonical_exact": 0.0, "n": 32},
                    "interpretation_counts": {"encoder": 2, "bounds": 1},
                }
            }
        )
        self.assertAlmostEqual(out["nll_own"], 1.25)
        self.assertAlmostEqual(out["greedy_mu/canonical_exact"], 0.0)
        self.assertAlmostEqual(out["greedy_mu/n"], 32.0)
        self.assertAlmostEqual(out["interpretation_counts/encoder"], 2.0)

    def test_items_words(self):
        self.assertEqual(
            items_words([{"word": "CCO"}, {"target": "CCC"}]),
            ["CCO", "CCC"],
        )

    def test_mean_and_rate(self):
        self.assertAlmostEqual(mean_finite([1.0, None, 3.0]) or 0.0, 2.0)
        self.assertIsNone(mean_finite([None]))
        self.assertAlmostEqual(rate([True, False, True]) or 0.0, 2 / 3)
        self.assertIsNone(rate([]))


if __name__ == "__main__":
    unittest.main()
