"""Tests for RECON_TEST metric helpers in eval_suite."""

from __future__ import annotations

import unittest

import numpy as np

from boreft.eval.eval_suite import recon_test_metrics


class ReconTestMetricsTest(unittest.TestCase):
    def test_full_interp_extrap_slices(self):
        full = ["apple", "banana", "cherry", "date"]
        interp = ["apple", "banana"]
        extrap = ["cherry", "date"]
        greedy = ["apple", "pear", "cherry", "fig"]
        sims = np.array([1.0, 0.5, 0.9, 0.2])
        temp_results = {
            1.0: [
                {"samples": ["apple", "fruit"]},
                {"samples": ["banana"]},
                {"samples": ["cherry"]},
                {"samples": ["fig"]},
            ],
        }

        metrics = recon_test_metrics(
            full_targets=full,
            interp_targets=interp,
            extrap_targets=extrap,
            greedy_decodes=greedy,
            greedy_embed_sims=sims,
            temp_results=temp_results,
            tau=0.8,
        )

        self.assertEqual(metrics["recall_greedy"], 0.5)
        self.assertEqual(metrics["interp_recall_greedy"], 0.5)
        self.assertEqual(metrics["extrap_recall_greedy"], 0.5)
        # temp hits: apple, banana, cherry (not date)
        self.assertEqual(metrics["recall_at_n_temp1.0"], 0.75)
        self.assertEqual(metrics["interp_recall_at_n_temp1.0"], 1.0)
        self.assertEqual(metrics["extrap_recall_at_n_temp1.0"], 0.5)
        self.assertAlmostEqual(metrics["embed_sim"], float(sims.mean()))
        self.assertAlmostEqual(metrics["interp_embed_sim"], 0.75)
        self.assertAlmostEqual(metrics["extrap_embed_sim"], 0.55)
        self.assertEqual(metrics["embed_sim_tau"], 0.8)
        self.assertNotIn("interp_embed_sim_tau", metrics)

    def test_skips_empty_subset(self):
        metrics = recon_test_metrics(
            full_targets=["apple"],
            interp_targets=[],
            extrap_targets=["apple"],
            greedy_decodes=["apple"],
            greedy_embed_sims=np.array([1.0]),
            temp_results={1.0: [{"samples": ["apple"]}]},
            tau=0.8,
        )
        self.assertNotIn("interp_recall_greedy", metrics)
        self.assertEqual(metrics["extrap_recall_greedy"], 1.0)


if __name__ == "__main__":
    unittest.main()
