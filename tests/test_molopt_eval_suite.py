"""Tests for molopt behavior in the eval suite.

Two things separate molopt from Semantle here: exact-match identity is RDKit
canonical SMILES rather than lower-cased text, and the held-out pool comes from
the ``smiles`` column of the molopt CSV. Validity and TFS metrics are molopt-only.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from boreft.eval.eval_suite import (
    DEFAULT_TFS_TAU,
    build_non_train_pool,
    build_test_sets,
    dist_metrics,
    genz_metrics,
    normalize_text,
    read_pool_csv_targets,
    recon_metrics,
    recon_test_metrics,
    target_normalizer,
    task_csv_paths,
    tfs_best_of_bag_metrics,
    tfs_diversity_metrics,
    tfs_metrics,
    tfs_per_target,
    validity_metrics,
)
from boreft.eval.run_full_eval import interp_pair_key, resolve_interp_pairs

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ASPIRIN_KEKULIZED = "CC(=O)OC1=CC=CC=C1C(=O)O"

MOLOPT_CSV = """inchikey,smiles,qed
K1,CCO,0.41
K2,c1ccccc1,0.35
K3,CC(=O)Oc1ccccc1C(=O)O,0.55
K4,CCN(CC)CC,0.29
K5,CCCC,0.22
"""


def _write(text: str, test: unittest.TestCase, suffix: str = ".csv") -> str:
    tmp = tempfile.NamedTemporaryFile(
        "w", suffix=suffix, delete=False, encoding="utf-8", newline=""
    )
    tmp.write(text)
    tmp.close()
    test.addCleanup(os.unlink, tmp.name)
    return tmp.name


def _sample_bag(samples: list[str]) -> dict:
    """Engine-shaped result dict, with a similarity per unique sample."""
    unique = list(dict.fromkeys(samples))
    return {
        "samples": samples,
        "unique_samples": unique,
        "per_sample_sims": [1.0] * len(unique),
    }


class TargetNormalizerTests(unittest.TestCase):
    def test_semantle_lowercases(self):
        self.assertEqual(normalize_text("  The   Cat ", task="semantle"), "the cat")

    def test_molopt_canonicalizes_instead_of_lowercasing(self):
        self.assertEqual(
            normalize_text("OCC", task="molopt"), normalize_text("CCO", task="molopt")
        )

    def test_molopt_keeps_case_significant(self):
        norm = target_normalizer("molopt")
        self.assertNotEqual(norm("C1CCCCC1"), norm("c1ccccc1"))

    def test_default_task_is_semantle(self):
        self.assertEqual(normalize_text("Cat"), "cat")


class ReconMetricsMolOptTests(unittest.TestCase):
    def test_equivalent_smiles_counts_as_a_greedy_hit(self):
        recon = recon_metrics(
            targets=["CCO"],
            greedy_decodes=["OCC"],
            greedy_embed_sims=np.array([0.99]),
            temp_results={},
            tau=0.8,
            task="molopt",
        )
        self.assertEqual(recon["recall_greedy"], 1.0)

    def test_case_changed_smiles_is_not_a_hit(self):
        """Under the Semantle normalizer this would have scored as a hit."""
        recon = recon_metrics(
            targets=["C1CCCCC1"],
            greedy_decodes=["c1ccccc1"],
            greedy_embed_sims=np.array([0.9]),
            temp_results={},
            tau=0.8,
            task="molopt",
        )
        self.assertEqual(recon["recall_greedy"], 0.0)

    def test_recall_at_n_uses_canonical_matching(self):
        recon = recon_metrics(
            targets=["CCO", "CCC"],
            greedy_decodes=["junk", "junk"],
            greedy_embed_sims=np.array([0.1, 0.1]),
            temp_results={1.0: [_sample_bag(["OCC", "x"]), _sample_bag(["y", "z"])]},
            tau=0.8,
            task="molopt",
        )
        self.assertEqual(recon["recall_at_n_temp1.0"], 0.5)

    def test_validity_reported_for_greedy_and_each_temperature(self):
        recon = recon_metrics(
            targets=["CCO", "CCC"],
            greedy_decodes=["CCO", "not a molecule"],
            greedy_embed_sims=np.array([1.0, 0.0]),
            temp_results={1.0: [_sample_bag(["CCO", "??"]), _sample_bag(["CCC", "CC"])]},
            tau=0.8,
            task="molopt",
        )
        self.assertAlmostEqual(recon["validity_greedy"], 0.5)
        self.assertAlmostEqual(recon["validity_temp1.0"], 0.75)

    def test_semantle_runs_get_no_validity_keys(self):
        recon = recon_metrics(
            targets=["cat"],
            greedy_decodes=["cat"],
            greedy_embed_sims=np.array([1.0]),
            temp_results={1.0: [_sample_bag(["cat"])]},
            tau=0.8,
        )
        self.assertNotIn("validity_greedy", recon)
        self.assertNotIn("validity_temp1.0", recon)


class DistMetricsMolOptTests(unittest.TestCase):
    def test_modal_sample_matches_the_target_by_canonical_form(self):
        dist = dist_metrics(
            targets=["CCO"],
            temp_results={1.0: [_sample_bag(["OCC", "OCC", "CCC"])]},
            task="molopt",
        )
        self.assertEqual(dist["mode_is_target_temp1.0"], 1.0)

    def test_case_changed_modal_sample_is_not_the_target(self):
        dist = dist_metrics(
            targets=["C1CCCCC1"],
            temp_results={1.0: [_sample_bag(["c1ccccc1", "c1ccccc1"])]},
            task="molopt",
        )
        self.assertEqual(dist["mode_is_target_temp1.0"], 0.0)

    def test_notarget_drops_equivalent_spellings_of_the_target(self):
        """A re-spelled target must not leak into the notarget similarity bag."""
        bag = {
            "samples": ["OCC", "CCC"],
            "unique_samples": ["OCC", "CCC"],
            "per_sample_sims": [1.0, 0.2],
        }
        dist = dist_metrics(
            targets=["CCO"], temp_results={1.0: [bag]}, task="molopt"
        )
        self.assertAlmostEqual(dist["embed_sim_notarget_temp1.0"], 0.2)


class GenzMetricsMolOptTests(unittest.TestCase):
    def test_recall_matches_canonical_test_targets(self):
        genz = genz_metrics(
            train_targets=["CCO"],
            interp_set=["c1ccccc1"],
            extrap_set=["CCCC"],
            greedy_decodes=["C1=CC=CC=C1"],  # kekulized benzene
            temp_decodes={},
            task="molopt",
        )
        self.assertEqual(genz["sobol_recall_test_interp_greedy"], 1.0)
        self.assertEqual(genz["sobol_recall_test_extrap_greedy"], 0.0)

    def test_equivalent_spelling_of_a_train_target_is_not_unseen(self):
        genz = genz_metrics(
            train_targets=["CCO"],
            interp_set=[],
            extrap_set=[],
            greedy_decodes=["OCC"],
            temp_decodes={},
            task="molopt",
        )
        self.assertEqual(genz["sobol_unseen_greedy"], 0.0)

    def test_validity_reported_per_decode_set(self):
        genz = genz_metrics(
            train_targets=["CCO"],
            interp_set=[],
            extrap_set=[],
            greedy_decodes=["CCO", "!!"],
            temp_decodes={1.5: ["CCC", "CCO", "??", "@@"]},
            task="molopt",
        )
        self.assertAlmostEqual(genz["validity_greedy"], 0.5)
        self.assertAlmostEqual(genz["validity_temp1.5"], 0.5)

    def test_semantle_runs_get_no_validity_keys(self):
        genz = genz_metrics(
            train_targets=["cat"],
            interp_set=[],
            extrap_set=[],
            greedy_decodes=["dog"],
            temp_decodes={1.0: ["dog"]},
        )
        self.assertNotIn("validity_greedy", genz)


class PoolBuildingTests(unittest.TestCase):
    def test_reads_the_smiles_column(self):
        path = _write(MOLOPT_CSV, self)
        self.assertEqual(len(read_pool_csv_targets(path, task="molopt")), 5)

    def test_pool_excludes_trained_targets(self):
        path = _write(MOLOPT_CSV, self)
        pool = build_non_train_pool([path], ["CCO", "c1ccccc1"], task="molopt")
        self.assertNotIn("CCO", pool)
        self.assertEqual(len(pool), 3)

    def test_pool_excludes_equivalent_spellings_of_trained_targets(self):
        """A trained molecule must not reappear in the held-out pool re-spelled."""
        path = _write(MOLOPT_CSV, self)
        pool = build_non_train_pool([path], ["OCC"], task="molopt")
        self.assertEqual(len(pool), 4)
        self.assertNotIn("CCO", pool)

    def test_missing_files_are_skipped(self):
        self.assertEqual(
            build_non_train_pool(["/nonexistent.csv"], ["CCO"], task="molopt"), []
        )

    def test_empty_pool_is_reported_in_meta(self):
        path = _write("inchikey,smiles\nK1,CCO\n", self)
        interp, extrap, meta = build_test_sets(
            train_targets=["CCO"],
            csv_paths=[path],
            test_n_samples=4,
            seed=0,
            pca_var=0.9,
            task="molopt",
        )
        self.assertEqual((interp, extrap), ([], []))
        self.assertEqual(meta["n_pool"], 0)
        self.assertIn("note", meta)

    def test_build_test_sets_encodes_with_the_molopt_backend(self):
        path = _write(MOLOPT_CSV, self)
        rng = np.random.default_rng(0)

        def fake_encode(texts, *, task):
            self.assertEqual(task, "molopt")
            return rng.normal(size=(len(texts), 8))

        with mock.patch(
            "boreft.eval.eval_suite.encode_texts_normalized", side_effect=fake_encode
        ) as encode:
            interp, extrap, meta = build_test_sets(
                train_targets=["CCO", "c1ccccc1"],
                csv_paths=[path],
                test_n_samples=3,
                seed=0,
                pca_var=0.9,
                task="molopt",
            )
        self.assertTrue(encode.called)
        self.assertEqual(meta["n_pool"], 3)
        self.assertEqual(len(interp) + len(extrap), 3)

    def test_explicit_high_tail_pool_is_used_and_train_filtered(self):
        rng = np.random.default_rng(1)

        def fake_encode(texts, *, task):
            return rng.normal(size=(len(texts), 8))

        with mock.patch(
            "boreft.eval.eval_suite.encode_texts_normalized", side_effect=fake_encode
        ):
            interp, extrap, meta = build_test_sets(
                train_targets=["CCO"],
                csv_paths=None,
                test_n_samples=8,
                seed=0,
                pca_var=0.9,
                task="molopt",
                pool=["OCC", "CCC", "CCCC"],
                pool_source="oracle_high_tail",
            )
        self.assertEqual(meta["pool_source"], "oracle_high_tail")
        self.assertEqual(meta["n_pool"], 2)
        self.assertEqual(sorted(interp + extrap), ["CCC", "CCCC"])


class TaskCsvPathsTests(unittest.TestCase):
    def test_semantle_reads_the_tuple_of_csvs(self):
        cfg = {"task": "semantle", "semantle_csv": ["a.csv", "b.csv"]}
        self.assertEqual(task_csv_paths(cfg), ["a.csv", "b.csv"])

    def test_molopt_reads_the_single_path(self):
        cfg = {"task": "molopt", "molopt_csv": "mols.csv"}
        self.assertEqual(task_csv_paths(cfg), ["mols.csv"])

    def test_hypogen_reads_the_single_path(self):
        cfg = {"task": "hypogen", "hypogen_csv": "hyps.csv"}
        self.assertEqual(task_csv_paths(cfg), ["hyps.csv"])

    def test_explicit_task_overrides_the_saved_one(self):
        cfg = {"task": "semantle", "molopt_csv": "mols.csv"}
        self.assertEqual(task_csv_paths(cfg, task="molopt"), ["mols.csv"])

    def test_missing_key_returns_empty(self):
        self.assertEqual(task_csv_paths({"task": "molopt"}), [])
        self.assertEqual(task_csv_paths({"task": "arc", "arc_dir": "d"}), [])


class ValidityMetricsTests(unittest.TestCase):
    def test_suffix_names_the_key(self):
        self.assertEqual(
            validity_metrics(["CCO"], task="molopt", suffix="greedy"),
            {"validity_greedy": 1.0},
        )

    def test_empty_for_text_tasks(self):
        self.assertEqual(validity_metrics(["cat"], task="semantle", suffix="greedy"), {})


class TfsMetricsTests(unittest.TestCase):
    """The Morgan/Tanimoto family reported next to embed_sim on molecule tasks."""

    def test_perfect_reconstruction_scores_one(self):
        out = tfs_metrics([ASPIRIN], [ASPIRIN_KEKULIZED], task="molopt")
        self.assertEqual(out["tfs"], 1.0)
        self.assertEqual(out["tfs_gte_tau"], 1.0)
        self.assertEqual(out["tfs_tau"], DEFAULT_TFS_TAU)

    def test_tau_rate_counts_targets_at_or_above_the_threshold(self):
        out = tfs_metrics(
            [ASPIRIN, ASPIRIN], [ASPIRIN, "CCO"], task="molopt", tau=0.9
        )
        self.assertEqual(out["tfs_gte_tau"], 0.5)

    def test_invalid_decode_scores_zero_rather_than_dropping_out(self):
        out = tfs_metrics([ASPIRIN, ASPIRIN], [ASPIRIN, "banana"], task="molopt")
        self.assertEqual(out["tfs"], 0.5)

    def test_empty_input_is_zero_not_an_error(self):
        out = tfs_metrics([], [], task="molopt")
        self.assertEqual(out["tfs"], 0.0)
        self.assertEqual(out["tfs_gte_tau"], 0.0)

    def test_text_tasks_get_no_tfs_keys(self):
        self.assertEqual(tfs_metrics(["cat"], ["cat"], task="semantle"), {})
        self.assertIsNone(tfs_per_target(["cat"], ["cat"], task="semantle"))

    def test_per_target_values_are_parallel_to_the_targets(self):
        sims = tfs_per_target([ASPIRIN, ASPIRIN], [ASPIRIN, "CCO"], task="molopt")
        self.assertEqual(sims.shape, (2,))
        self.assertEqual(sims[0], 1.0)
        self.assertLess(sims[1], 1.0)

    def test_best_of_bag_takes_the_closest_sample(self):
        out = tfs_best_of_bag_metrics(
            [ASPIRIN],
            [["CCO", ASPIRIN_KEKULIZED, "banana"]],
            task="molopt",
            suffix="temp1.0",
        )
        self.assertEqual(out, {"tfs_best_at_n_temp1.0": 1.0})

    def test_best_of_bag_skips_targets_with_no_samples(self):
        out = tfs_best_of_bag_metrics(
            [ASPIRIN, "CCO"], [[ASPIRIN], []], task="molopt", suffix="temp1.0"
        )
        self.assertEqual(out, {"tfs_best_at_n_temp1.0": 1.0})

    def test_diversity_is_zero_for_a_collapsed_decode_set(self):
        out = tfs_diversity_metrics(
            [ASPIRIN, ASPIRIN_KEKULIZED], task="molopt", suffix="greedy"
        )
        self.assertEqual(out, {"tfs_intdiv_greedy": 0.0})

    def test_diversity_is_omitted_when_too_few_decodes_parse(self):
        self.assertEqual(
            tfs_diversity_metrics(["banana"], task="molopt", suffix="greedy"), {}
        )

    def test_diversity_is_subsampled_for_large_decode_sets(self):
        """GENZ passes tens of thousands of decodes; the cost must stay flat."""
        with mock.patch("boreft.eval.eval_suite.TFS_INTDIV_MAX_N", 4):
            with mock.patch(
                "boreft.chem.tanimoto_internal_diversity", return_value=0.5
            ) as fake:
                tfs_diversity_metrics(
                    [ASPIRIN] * 50, task="molopt", suffix="temp1.0"
                )
        self.assertEqual(len(fake.call_args.args[0]), 4)


class ReconTfsIntegrationTests(unittest.TestCase):
    def test_recon_reports_tfs_next_to_embed_sim(self):
        out = recon_metrics(
            targets=[ASPIRIN],
            greedy_decodes=[ASPIRIN_KEKULIZED],
            greedy_embed_sims=np.array([0.9]),
            temp_results={1.0: [_sample_bag([ASPIRIN, "CCO"])]},
            tau=0.8,
            task="molopt",
        )
        self.assertEqual(out["tfs"], 1.0)
        self.assertEqual(out["tfs_best_at_n_temp1.0"], 1.0)
        self.assertIn("embed_sim", out)

    def test_recon_test_prefixes_tfs_per_slice_and_emits_tau_once(self):
        out = recon_test_metrics(
            full_targets=[ASPIRIN, "CCO"],
            interp_targets=[ASPIRIN],
            extrap_targets=["CCO"],
            greedy_decodes=[ASPIRIN_KEKULIZED, "banana"],
            greedy_embed_sims=np.array([0.9, 0.1]),
            temp_results={1.0: [_sample_bag([ASPIRIN]), _sample_bag(["CCO"])]},
            tau=0.8,
            task="molopt",
        )
        self.assertEqual(out["interp_tfs"], 1.0)
        self.assertEqual(out["extrap_tfs"], 0.0)
        self.assertNotIn("interp_tfs_tau", out)
        self.assertNotIn("extrap_tfs_tau", out)
        self.assertIn("tfs_tau", out)

    def test_dist_reports_tfs_spread_alongside_embed_sim_spread(self):
        out = dist_metrics(
            targets=[ASPIRIN],
            temp_results={1.0: [_sample_bag([ASPIRIN, ASPIRIN, "CCO"])]},
            task="molopt",
        )
        # Two of three samples are the target itself, the third is unrelated.
        self.assertGreater(out["tfs_mean_temp1.0"], out["tfs_notarget_temp1.0"])
        self.assertGreater(out["tfs_std_temp1.0"], 0.0)
        self.assertIn("embed_sim_mean_temp1.0", out)

    def test_dist_keeps_the_semantle_key_set_unchanged(self):
        out = dist_metrics(
            targets=["cat"],
            temp_results={1.0: [_sample_bag(["cat", "dog"])]},
            task="semantle",
        )
        self.assertEqual(
            sorted(out),
            [
                "embed_sim_mean_temp1.0",
                "embed_sim_notarget_std_temp1.0",
                "embed_sim_notarget_temp1.0",
                "embed_sim_std_temp1.0",
                "mode_is_target_temp1.0",
            ],
        )

    def test_genz_reports_structural_diversity_of_sobol_decodes(self):
        out = genz_metrics(
            train_targets=[ASPIRIN],
            interp_set=[],
            extrap_set=[],
            greedy_decodes=[ASPIRIN, "CCO", "c1ccccc1"],
            temp_decodes={1.0: [ASPIRIN, ASPIRIN_KEKULIZED]},
            task="molopt",
        )
        self.assertGreater(out["tfs_intdiv_greedy"], 0.0)
        self.assertEqual(out["tfs_intdiv_temp1.0"], 0.0)

    def test_genz_semantle_runs_get_no_tfs_keys(self):
        out = genz_metrics(
            train_targets=["cat"],
            interp_set=[],
            extrap_set=[],
            greedy_decodes=["cat", "dog"],
            temp_decodes={1.0: ["cat"]},
            task="semantle",
        )
        self.assertEqual([k for k in out if "tfs" in k], [])


class InterpPairKeyTests(unittest.TestCase):
    def test_plain_words_are_left_alone(self):
        self.assertEqual(interp_pair_key("cat", "dog"), "cat__dog")

    def test_smiles_key_is_path_and_namespace_safe(self):
        key = interp_pair_key("C/C=C/C", "CC(=O)O")
        for bad in ("/", "\\", "(", ")", "=", "#"):
            self.assertNotIn(bad, key)

    def test_distinct_smiles_pairs_get_distinct_keys(self):
        self.assertNotEqual(
            interp_pair_key("C/C=C/C", "CCO"), interp_pair_key("C\\C=C\\C", "CCO")
        )

    def test_key_is_deterministic(self):
        self.assertEqual(
            interp_pair_key("C/C=C/C", "CCO"), interp_pair_key("C/C=C/C", "CCO")
        )


class RunFullEvalGatingTests(unittest.TestCase):
    """``train_reft`` runs the suite for any task config validation accepts.

    It no longer re-checks the task, so these bounds are what keep an unsupported
    task out of the eval pipeline — and what keep molopt in it.
    """

    def _config(self, **kwargs):
        from boreft.train_args import TrainConfig

        return TrainConfig(run_full_eval=True, **kwargs)

    def test_accepted_for_molopt(self):
        cfg = self._config(task="molopt", molopt_csv="mols.csv")
        self.assertTrue(cfg.run_full_eval)

    def test_accepted_for_semantle(self):
        cfg = self._config(task="semantle", semantle_csv=("words.csv",))
        self.assertTrue(cfg.run_full_eval)

    def test_accepted_for_hypogen(self):
        cfg = self._config(task="hypogen", hypogen_csv="hyps.csv")
        self.assertTrue(cfg.run_full_eval)

    def test_rejected_for_tasks_without_reconstruction_eval(self):
        with self.assertRaisesRegex(ValueError, "--run-full-eval"):
            self._config(task="arc", arc_dir="d")

    def test_eval_steps_rejected_for_tasks_without_reconstruction_eval(self):
        from boreft.train_args import TrainConfig

        with self.assertRaisesRegex(ValueError, "--eval-steps"):
            TrainConfig(task="arc", arc_dir="d", eval_steps=10)


class ResolveInterpPairsTests(unittest.TestCase):
    def test_explicit_pair_resolves_to_the_trained_spelling(self):
        pairs, meta = resolve_interp_pairs(
            ["OCC", "CCC"], ["CCO", "CCC"], seed=0, task="molopt"
        )
        self.assertEqual(pairs, [("CCO", "CCC")])
        self.assertEqual(meta["mode"], "explicit")

    def test_untrained_endpoint_is_rejected_by_name(self):
        with self.assertRaisesRegex(ValueError, "CCCCCCC"):
            resolve_interp_pairs(
                ["CCO", "CCCCCCC"], ["CCO", "CCC"], seed=0, task="molopt"
            )

    def test_semantle_pairs_are_matched_case_insensitively(self):
        pairs, _ = resolve_interp_pairs(["Cat", "dog"], ["cat", "dog"], seed=0)
        self.assertEqual(pairs, [("cat", "dog")])

    def test_integer_argument_still_samples_random_pairs(self):
        pairs, meta = resolve_interp_pairs(
            ["2"], ["CCO", "CCC", "CCCC"], seed=0, task="molopt"
        )
        self.assertEqual(len(pairs), 2)
        self.assertEqual(meta["mode"], "random_sample")


if __name__ == "__main__":
    unittest.main()
