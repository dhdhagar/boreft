"""RDKit descriptor similarity parity with TFS evaluation families."""

from __future__ import annotations

import unittest

import numpy as np

from boreft.eval.eval_suite import (
    dist_metrics,
    edit_dist_best_of_bag_metrics,
    edit_dist_metrics,
    genz_metrics,
    rdkit_sim_best_of_bag_metrics,
    rdkit_sim_diversity_metrics,
    rdkit_sim_metrics,
    recon_metrics,
    recon_test_metrics,
)
from boreft.eval.plot_recon_similarity import rdkit_pairs_from_per_target


ETHANOL = "CCO"
ETHYLAMINE = "CCN"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def _result(samples: list[str]) -> dict:
    unique = list(dict.fromkeys(samples))
    return {
        "samples": samples,
        "unique_samples": unique,
        "per_sample_sims": [0.5] * len(unique),
    }


class RdkitMetricPrimitiveTests(unittest.TestCase):
    def test_identity_mean_and_threshold(self):
        metrics = rdkit_sim_metrics(
            [ETHANOL], [ETHANOL], task="molopt"
        )
        self.assertEqual(metrics["rdkit_sim"], 1.0)
        self.assertEqual(metrics["rdkit_sim_gte_tau"], 1.0)
        self.assertEqual(metrics["rdkit_sim_tau"], 0.5)

    def test_invalid_decode_is_zero(self):
        metrics = rdkit_sim_metrics(
            [ETHANOL], ["not smiles"], task="molopt"
        )
        self.assertEqual(metrics["rdkit_sim"], 0.0)

    def test_best_of_bag_and_diversity(self):
        best = rdkit_sim_best_of_bag_metrics(
            [ETHANOL],
            [[ASPIRIN, ETHANOL]],
            task="molopt",
            suffix="temp1.0",
        )
        self.assertEqual(best["rdkit_sim_best_at_n_temp1.0"], 1.0)
        diversity = rdkit_sim_diversity_metrics(
            [ETHANOL, ASPIRIN], task="molopt", suffix="greedy"
        )
        self.assertGreater(diversity["rdkit_sim_intdiv_greedy"], 0.0)

    def test_semantle_emits_no_rdkit_keys(self):
        self.assertEqual(
            rdkit_sim_metrics(["cat"], ["dog"], task="semantle"), {}
        )


class EditDistMetricPrimitiveTests(unittest.TestCase):
    def test_identity_mean_is_zero(self):
        metrics = edit_dist_metrics([ETHANOL], [ETHANOL], task="molopt")
        self.assertEqual(metrics["edit_dist"], 0.0)

    def test_best_of_bag_picks_the_closest_sample(self):
        best = edit_dist_best_of_bag_metrics(
            [ETHANOL],
            [[ASPIRIN, ETHANOL]],
            task="molopt",
            suffix="temp1.0",
        )
        self.assertEqual(best["edit_dist_best_at_n_temp1.0"], 0.0)

    def test_semantle_emits_no_edit_dist_keys(self):
        self.assertEqual(
            edit_dist_metrics(["cat"], ["dog"], task="semantle"), {}
        )


class RdkitMetricIntegrationTests(unittest.TestCase):
    def test_recon_and_best_of_bag_keys(self):
        metrics = recon_metrics(
            targets=[ETHANOL],
            greedy_decodes=[ETHANOL],
            greedy_embed_sims=np.asarray([1.0]),
            temp_results={1.0: [_result([ETHYLAMINE, ETHANOL])]},
            tau=0.8,
            task="molopt",
        )
        self.assertEqual(metrics["rdkit_sim"], 1.0)
        self.assertEqual(metrics["rdkit_sim_best_at_n_temp1.0"], 1.0)
        self.assertEqual(metrics["edit_dist"], 0.0)
        self.assertEqual(metrics["edit_dist_best_at_n_temp1.0"], 0.0)

    def test_recon_test_prefixes_slices_but_not_tau(self):
        metrics = recon_test_metrics(
            full_targets=[ETHANOL, ASPIRIN],
            interp_targets=[ETHANOL],
            extrap_targets=[ASPIRIN],
            greedy_decodes=[ETHANOL, ASPIRIN],
            greedy_embed_sims=np.asarray([1.0, 1.0]),
            temp_results={
                1.0: [_result([ETHANOL]), _result([ASPIRIN])]
            },
            tau=0.8,
            task="molopt",
        )
        self.assertIn("interp_rdkit_sim", metrics)
        self.assertIn("extrap_rdkit_sim", metrics)
        self.assertNotIn("interp_rdkit_sim_tau", metrics)
        self.assertIn("interp_edit_dist", metrics)
        self.assertIn("extrap_edit_dist", metrics)

    def test_dist_reports_all_rdkit_spread_variants(self):
        metrics = dist_metrics(
            targets=[ETHANOL],
            temp_results={
                1.0: [_result([ETHANOL, ETHYLAMINE, ETHYLAMINE])]
            },
            task="molopt",
        )
        for stem in (
            "rdkit_sim_mean",
            "rdkit_sim_std",
            "rdkit_sim_notarget",
            "rdkit_sim_notarget_std",
        ):
            self.assertIn(f"{stem}_temp1.0", metrics)

    def test_genz_reports_descriptor_internal_diversity(self):
        metrics = genz_metrics(
            train_targets=[ETHANOL],
            interp_set=[ETHYLAMINE],
            extrap_set=[ASPIRIN],
            greedy_decodes=[ETHANOL, ASPIRIN],
            temp_decodes={1.0: [ETHANOL, ETHYLAMINE]},
            task="molopt",
        )
        self.assertIn("rdkit_sim_intdiv_greedy", metrics)
        self.assertIn("rdkit_sim_intdiv_temp1.0", metrics)

    def test_plot_rows_extract_rdkit_similarity(self):
        embed, rdkit = rdkit_pairs_from_per_target(
            [{"sim": 0.8, "rdkit_sim": 0.6}, {"sim": 0.2}]
        )
        np.testing.assert_allclose(embed, [0.8])
        np.testing.assert_allclose(rdkit, [0.6])


if __name__ == "__main__":
    unittest.main()
