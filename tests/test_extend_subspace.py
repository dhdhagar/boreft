"""Unit tests for pure logic in boreft.extend_subspace.

These exercise unlearnt-target selection / capping and the combined + reindexed
items construction without a GPU or a real trained checkpoint.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

import boreft.extend_subspace as ext
from boreft.learn_bias import sorted_greedy_decode_rows
from boreft.extend_subspace import (
    EXTEND_QUESTION_GOAL,
    SCHEMA_VERSION,
    _define_extend_wandb_metrics,
    _log_extend_to_wandb,
    _make_live_wandb_progress_callback,
    _materialize_combined_bias_tables,
    _panel_from_slice,
    absorption_before_hit_maps,
    before_after_bar_rows,
    build_extend_summary,
    cell_from_panel,
    estimate_count_from_rate_delta,
    headline_console_line,
    hit_maps_from_slice,
    original_test_misses_result,
    original_train_result,
    original_test_hits_result,
    build_combined_items,
    resolve_previous_sample_count,
    select_previous_train_indices,
    select_previous_train_targets,
    select_unlearnt_targets,
    training_curve_logs,
    wandb_log_payloads,
    write_extend_sidecars,
)


def _fake_ckpt(*, fixed_logvar, learnable_logvar):
    """A minimal stand-in whose ``_get_intervention`` returns a fake intervention."""
    bias_network = types.SimpleNamespace(learnable_logvar=learnable_logvar, fc1=None)
    intervention = types.SimpleNamespace(
        fixed_logvar=fixed_logvar, bias_network=bias_network
    )
    reft_model = types.SimpleNamespace(interventions={"k": intervention})
    return types.SimpleNamespace(reft_model=reft_model)


class SelectUnlearntTargetsTest(unittest.TestCase):
    def test_concatenates_train_then_test(self):
        out = select_unlearnt_targets(["a", "b"], ["c", "d"], None)
        self.assertEqual(out, ["a", "b", "c", "d"])

    def test_dedup_preserves_first_occurrence(self):
        # "B" duplicates "b" (normalized) and is dropped; test-side dup also dropped.
        out = select_unlearnt_targets(["a", "b", "B"], ["b", "c"], None)
        self.assertEqual(out, ["a", "b", "c"])

    def test_cap_takes_first_n(self):
        out = select_unlearnt_targets(["a", "b"], ["c", "d"], 3)
        self.assertEqual(out, ["a", "b", "c"])

    def test_cap_zero_yields_empty(self):
        self.assertEqual(select_unlearnt_targets(["a"], ["b"], 0), [])

    def test_cap_larger_than_available(self):
        out = select_unlearnt_targets(["a"], ["b"], 10)
        self.assertEqual(out, ["a", "b"])

    def test_negative_cap_raises(self):
        with self.assertRaises(ValueError):
            select_unlearnt_targets(["a"], [], -1)

    def test_molopt_dedups_equivalent_smiles_spellings(self):
        out = select_unlearnt_targets(["CCO"], ["OCC"], None, task="molopt")
        self.assertEqual(out, ["CCO"])

    def test_molopt_keeps_case_distinct_smiles_apart(self):
        """Lower-casing SMILES would merge cyclohexane into benzene."""
        out = select_unlearnt_targets(
            ["C1CCCCC1"], ["c1ccccc1"], None, task="molopt"
        )
        self.assertEqual(out, ["C1CCCCC1", "c1ccccc1"])


class BuildCombinedItemsTest(unittest.TestCase):
    def _old_items(self):
        return [
            {"id": 0, "prompt": "P", "target": "apple<eos>", "word": "apple"},
            {"id": 1, "prompt": "P", "target": "banana<eos>", "word": "banana"},
        ]

    def test_reindex_and_origin_tags(self):
        items, old_idx, new_idx = build_combined_items(
            self._old_items(),
            ["cherry", "date"],
            prompt="P",
            new_original_splits={"cherry": "test", "date": "additional"},
        )
        self.assertEqual([it["id"] for it in items], [0, 1, 2, 3])
        self.assertEqual([it["word"] for it in items], ["apple", "banana", "cherry", "date"])
        self.assertEqual([it["origin"] for it in items], ["old", "old", "new", "new"])
        self.assertEqual(
            [it["original_split"] for it in items],
            ["train", "train", "test", "additional"],
        )
        self.assertTrue(all(it["seen"] for it in items))
        self.assertEqual(old_idx, [0, 1])
        self.assertEqual(new_idx, [2, 3])

    def test_ids_are_contiguous_from_zero(self):
        items, _, _ = build_combined_items(self._old_items(), ["cherry"], prompt="P")
        self.assertEqual([it["id"] for it in items], list(range(len(items))))

    def test_legacy_new_origin_defaults_to_additional(self):
        items, _, _ = build_combined_items(
            [{"id": 0, "word": "cherry", "origin": "new"}],
            [],
            prompt="P",
        )
        self.assertEqual(items[0]["original_split"], "additional")

    def test_new_items_use_given_prompt_and_word_target_by_default(self):
        items, _, new_idx = build_combined_items(
            self._old_items(), ["cherry"], prompt="NEW_PROMPT"
        )
        new_item = items[new_idx[0]]
        self.assertEqual(new_item["prompt"], "NEW_PROMPT")
        self.assertEqual(new_item["target"], "cherry")
        self.assertEqual(new_item["word"], "cherry")

    def test_new_targets_text_override(self):
        items, _, new_idx = build_combined_items(
            self._old_items(),
            ["cherry"],
            prompt="P",
            new_targets_text={"cherry": "cherry<eos>"},
        )
        self.assertEqual(items[new_idx[0]]["target"], "cherry<eos>")

    def test_empty_new_words(self):
        items, old_idx, new_idx = build_combined_items(self._old_items(), [], prompt="P")
        self.assertEqual(len(items), 2)
        self.assertEqual(new_idx, [])
        self.assertEqual(old_idx, [0, 1])

    def test_no_old_items(self):
        items, old_idx, new_idx = build_combined_items([], ["x", "y"], prompt="P")
        self.assertEqual([it["id"] for it in items], [0, 1])
        self.assertEqual(old_idx, [])
        self.assertEqual(new_idx, [0, 1])

    def test_repeated_extension_preserves_original_split(self):
        once, _, _ = build_combined_items(
            self._old_items(),
            ["cherry"],
            prompt="P",
            new_original_splits={"cherry": "test"},
        )
        twice, _, _ = build_combined_items(
            once,
            ["date"],
            prompt="P",
            new_original_splits={"date": "additional"},
        )
        self.assertEqual(
            [item["original_split"] for item in twice],
            ["train", "train", "test", "additional"],
        )
        self.assertEqual(
            [item["origin"] for item in twice],
            ["old", "old", "old", "new"],
        )


class SelectPreviousTrainTargetsTest(unittest.TestCase):
    def test_none_keeps_all(self):
        words = ["a", "b", "c", "d"]
        self.assertEqual(
            select_previous_train_targets(words, previous_n=None, seed=0),
            words,
        )

    def test_samples_preserving_order(self):
        words = ["a", "b", "c", "d", "e"]
        out = select_previous_train_targets(words, previous_n=3, seed=7)
        self.assertEqual(len(out), 3)
        self.assertEqual(out, [w for w in words if w in out])
        # Deterministic for a fixed seed.
        self.assertEqual(
            out,
            select_previous_train_targets(words, previous_n=3, seed=7),
        )

    def test_cap_and_zero(self):
        words = ["a", "b"]
        self.assertEqual(
            select_previous_train_targets(words, previous_n=10, seed=0),
            words,
        )
        self.assertEqual(
            select_previous_train_targets(words, previous_n=0, seed=0),
            [],
        )

    def test_previous_prop_rounds_to_count(self):
        words = ["a", "b", "c", "d"]
        self.assertEqual(
            resolve_previous_sample_count(4, previous_prop=0.5),
            2,
        )
        out = select_previous_train_targets(words, previous_prop=0.5, seed=3)
        self.assertEqual(len(out), 2)
        self.assertEqual(out, [w for w in words if w in out])

    def test_previous_prop_bounds_and_mutex(self):
        self.assertEqual(resolve_previous_sample_count(5, previous_prop=0.0), 0)
        self.assertEqual(resolve_previous_sample_count(5, previous_prop=1.0), 5)
        with self.assertRaises(ValueError):
            resolve_previous_sample_count(5, previous_prop=1.5)
        with self.assertRaises(ValueError):
            resolve_previous_sample_count(5, previous_n=2, previous_prop=0.5)

    def test_herding_strategy_uses_features_and_preserves_order(self):
        words = ["a", "b", "c", "d"]
        # Points on axes + diagonal; herding should prefer the diagonal first.
        features = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.5, 0.0],
            ],
            dtype=np.float64,
        )
        out = select_previous_train_targets(
            words,
            previous_n=2,
            strategy="herding",
            features=features,
            seed=0,
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(out, [w for w in words if w in out])
        self.assertEqual(
            out,
            select_previous_train_targets(
                words,
                previous_n=2,
                strategy="herding",
                features=features,
                seed=99,
            ),
        )

    def test_k_center_strategy_covers_clusters(self):
        words = ["e0", "e1", "w0", "w1", "n0", "n1", "s0", "s1"]
        features = np.array(
            [
                [10.0, 0.0],
                [10.1, 0.0],
                [-10.0, 0.0],
                [-10.1, 0.0],
                [0.0, 10.0],
                [0.0, 10.1],
                [0.0, -10.0],
                [0.0, -10.1],
            ],
            dtype=np.float64,
        )
        out = select_previous_train_targets(
            words,
            previous_n=4,
            strategy="k-center-greedy",
            features=features,
            seed=0,
        )
        self.assertEqual(len(out), 4)
        self.assertEqual(out, [w for w in words if w in out])
        prefixes = sorted(w[0] for w in out)
        self.assertEqual(prefixes, ["e", "n", "s", "w"])

    def test_geometry_strategy_requires_features(self):
        with self.assertRaises(ValueError):
            select_previous_train_targets(
                ["a", "b", "c"],
                previous_n=2,
                strategy="herding",
                features=None,
                seed=0,
            )

    def test_required_targets_fill_the_rest_of_the_budget(self):
        words = [f"m{i}" for i in range(10)]
        required = [1, 4, 7]
        chosen = select_previous_train_targets(
            words,
            previous_n=6,
            strategy="random",
            seed=42,
            required=required,
        )
        self.assertEqual(len(chosen), 6)
        for index in required:
            self.assertIn(words[index], chosen)
        again = select_previous_train_targets(
            words,
            previous_n=6,
            strategy="random",
            seed=42,
            required=required,
        )
        self.assertEqual(chosen, again)

    def test_required_targets_are_kept_when_they_exceed_the_budget(self):
        words = ["a", "b", "c", "d"]
        chosen = select_previous_train_indices(
            len(words),
            previous_n=2,
            strategy="random",
            seed=0,
            required=[0, 2, 3],
        )
        self.assertEqual(chosen, [0, 2, 3])

    def test_warmstart_file_lists_each_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "warm.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"seeds": {"1": ["CCO", "CCO"], "2": ["CCN"]}}, handle)
            self.assertEqual(ext.load_previous_include_targets(path), ["CCO", "CCN"])

    def test_indices_align_with_targets_and_features(self):
        words = ["a", "b", "c", "d"]
        features = np.eye(4, dtype=np.float64)
        idxs = select_previous_train_indices(
            len(words),
            previous_n=2,
            strategy="herding",
            features=features,
            seed=0,
        )
        self.assertEqual(idxs, sorted(idxs))
        targets = select_previous_train_targets(
            words,
            previous_n=2,
            strategy="herding",
            features=features,
            seed=0,
        )
        self.assertEqual(targets, [words[i] for i in idxs])


class MaterializeCombinedBiasTablesTest(unittest.TestCase):
    def test_include_prev_uses_learned_rows_directly(self):
        ckpt = _fake_ckpt(fixed_logvar=0.0, learnable_logvar=False)
        learned = types.SimpleNamespace(
            mu=np.arange(8, dtype=np.float32).reshape(4, 2),
            logvar=None,
            targets=["a", "b", "c", "d"],
        )
        mu_all, logvar_all = _materialize_combined_bias_tables(
            ckpt,
            old_words=["a", "b"],
            new_words=["c", "d"],
            old_mu_source=np.zeros((2, 2), dtype=np.float32),
            old_logvar_source=None,
            learned=learned,
            include_prev=True,
            learn_bias_network=False,
            raw_defs={},
            batch_size=8,
        )
        # include_prev: learned.mu already covers old+new in order.
        np.testing.assert_array_equal(mu_all, learned.mu)
        self.assertIsNone(logvar_all)

    def test_include_prev_subset_stitches_source_for_untrained_old(self):
        ckpt = _fake_ckpt(fixed_logvar=0.0, learnable_logvar=False)
        old_source = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
        learned = types.SimpleNamespace(
            mu=np.array([[20.0, 20.0], [9.0, 9.0]], dtype=np.float32),
            logvar=None,
            targets=["b", "c"],  # sampled previous=b, new=c
        )
        mu_all, logvar_all = _materialize_combined_bias_tables(
            ckpt,
            old_words=["a", "b", "d"],
            new_words=["c"],
            old_mu_source=old_source,
            old_logvar_source=None,
            learned=learned,
            include_prev=True,
            learn_bias_network=False,
            raw_defs={},
            batch_size=8,
        )
        expected = np.array(
            [[1.0, 1.0], [20.0, 20.0], [3.0, 3.0], [9.0, 9.0]], dtype=np.float32
        )
        np.testing.assert_array_equal(mu_all, expected)
        self.assertIsNone(logvar_all)

    def test_direct_mode_concats_source_old_and_learned_new(self):
        ckpt = _fake_ckpt(fixed_logvar=0.0, learnable_logvar=False)
        old_source = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        learned = types.SimpleNamespace(
            mu=np.array([[9.0, 9.0]], dtype=np.float32), logvar=None
        )
        mu_all, logvar_all = _materialize_combined_bias_tables(
            ckpt,
            old_words=["a", "b"],
            new_words=["c"],
            old_mu_source=old_source,
            old_logvar_source=None,
            learned=learned,
            include_prev=False,
            learn_bias_network=False,
            raw_defs={},
            batch_size=8,
        )
        expected = np.array([[1.0, 1.0], [2.0, 2.0], [9.0, 9.0]], dtype=np.float32)
        np.testing.assert_array_equal(mu_all, expected)
        self.assertIsNone(logvar_all)

    def test_direct_mode_learnable_logvar_concats_logvar(self):
        ckpt = _fake_ckpt(fixed_logvar=None, learnable_logvar=True)
        old_source = np.array([[1.0, 1.0]], dtype=np.float32)
        old_logvar = np.array([[0.1, 0.1]], dtype=np.float32)
        learned = types.SimpleNamespace(
            mu=np.array([[9.0, 9.0]], dtype=np.float32),
            logvar=np.array([[0.9, 0.9]], dtype=np.float32),
        )
        mu_all, logvar_all = _materialize_combined_bias_tables(
            ckpt,
            old_words=["a"],
            new_words=["c"],
            old_mu_source=old_source,
            old_logvar_source=old_logvar,
            learned=learned,
            include_prev=False,
            learn_bias_network=False,
            raw_defs={},
            batch_size=8,
        )
        np.testing.assert_array_equal(mu_all, np.array([[1.0, 1.0], [9.0, 9.0]], np.float32))
        np.testing.assert_array_equal(
            logvar_all, np.array([[0.1, 0.1], [0.9, 0.9]], np.float32)
        )

    def test_network_mode_refreshes_old_from_network(self):
        ckpt = _fake_ckpt(fixed_logvar=0.0, learnable_logvar=False)
        learned = types.SimpleNamespace(
            mu=np.array([[9.0, 9.0]], dtype=np.float32), logvar=None
        )
        refreshed_old = np.array([[5.0, 5.0], [6.0, 6.0]], dtype=np.float32)
        with mock.patch.object(
            ext, "predict_bias_network_rows", return_value=(refreshed_old, None)
        ) as pbnr:
            mu_all, logvar_all = _materialize_combined_bias_tables(
                ckpt,
                old_words=["a", "b"],
                new_words=["c"],
                old_mu_source=np.zeros((2, 2), dtype=np.float32),  # must be ignored
                old_logvar_source=None,
                learned=learned,
                include_prev=False,
                learn_bias_network=True,
                raw_defs={},
                batch_size=8,
            )
        pbnr.assert_called_once()
        expected = np.array([[5.0, 5.0], [6.0, 6.0], [9.0, 9.0]], dtype=np.float32)
        np.testing.assert_array_equal(mu_all, expected)
        self.assertIsNone(logvar_all)


class PanelFromSliceTest(unittest.TestCase):
    def test_slice_recall_and_embed_sim(self):
        words = ["apple", "banana", "cherry", "date"]
        greedy = ["apple", "pear", "cherry", "fig"]
        greedy_sims = np.array([1.0, 0.5, 0.9, 0.2])
        temp_results = {
            1.0: [
                {"samples": ["apple"]},
                {"samples": ["banana"]},
                {"samples": ["kiwi"]},
                {"samples": ["fig"]},
            ]
        }
        # New-target slice = last two words.
        panel = _panel_from_slice(words, greedy, greedy_sims, temp_results, [2, 3], tau=0.8)
        self.assertEqual(panel["recall_greedy"], 0.5)  # cherry hits, date (fig) misses
        self.assertEqual(panel["recall_at_n_temp1.0"], 0.0)  # neither cherry nor date recalled
        self.assertAlmostEqual(panel["embed_sim"], float(np.mean([0.9, 0.2])))

    def test_empty_slice(self):
        panel = _panel_from_slice(["a"], ["a"], np.array([1.0]), {1.0: [{"samples": ["a"]}]}, [], tau=0.8)
        self.assertEqual(panel["recall_greedy"], 0.0)


class HeadlineMetricsTest(unittest.TestCase):
    def test_absorption_before_reuses_detection_maps(self):
        recall, greedy = absorption_before_hit_maps(
            ["alpha", "beta", "gamma"],
            train_recall={"alpha": False, "beta": True},
            train_greedy={"alpha": False, "beta": True},
            test_recall={"beta": False, "gamma": False},
            test_greedy={"beta": False, "gamma": False},
            prefer_train_words=["alpha"],
        )
        # alpha is a preferred train miss; beta/gamma come from the test map.
        self.assertEqual(recall, {"alpha": False, "beta": False, "gamma": False})
        self.assertEqual(greedy, {"alpha": False, "beta": False, "gamma": False})
        block = original_test_misses_result(
            words=["alpha", "beta", "gamma"],
            before_recall=recall,
            before_greedy=greedy,
            after_recall={"alpha": True, "beta": False, "gamma": True},
        )
        self.assertEqual(block["before"]["recall_at_n"], 0.0)
        self.assertEqual(block["n_recovered"], 2)

    def test_original_train_forgotten_count(self):
        words = ["a", "b", "c", "d"]
        before = {"a": True, "b": True, "c": True, "d": False}
        after = {"a": True, "b": False, "c": True, "d": False}
        block = original_train_result(
            words=words,
            before_recall=before,
            before_greedy=None,
            after_recall=after,
        )
        self.assertEqual(block["n"], 4)
        self.assertEqual(block["n_forgotten"], 1)
        self.assertAlmostEqual(block["forget_rate"], 0.25)
        self.assertAlmostEqual(block["before"]["recall_at_n"], 0.75)
        self.assertAlmostEqual(block["after"]["recall_at_n"], 0.5)
        self.assertAlmostEqual(block["delta_recall_at_n"], -0.25)

    def test_forgotten_rate_fallback_without_before_hits(self):
        words = ["a", "b", "c", "d"]
        after = {"a": True, "b": False, "c": True, "d": False}
        block = original_train_result(
            words=words,
            before_recall=None,
            before_greedy=None,
            after_recall=after,
            before_recall_rate=0.75,
        )
        self.assertEqual(block["n_forgotten"], 1)
        self.assertEqual(block["before"]["recall_at_n"], 0.75)

    def test_estimate_count_from_rate_delta(self):
        self.assertEqual(estimate_count_from_rate_delta(100, 1.0, 0.9), 10)
        self.assertEqual(estimate_count_from_rate_delta(100, 0.5, 0.6), 0)

    def test_original_test_misses_recovered(self):
        words = ["x", "y", "z"]
        before = {"x": False, "y": False, "z": False}
        after = {"x": True, "y": False, "z": True}
        block = original_test_misses_result(
            words=words,
            before_recall=before,
            before_greedy=None,
            after_recall=after,
            after_embed_sim=0.7,
        )
        self.assertEqual(block["n_recovered"], 2)
        self.assertAlmostEqual(block["recover_rate"], 2 / 3)
        self.assertEqual(block["after"]["embed_sim"], 0.7)

    def test_remaining_still_misses(self):
        words = ["p", "q", "r"]
        before = {"p": True, "q": False, "r": True}
        after = {"p": True, "q": False, "r": False}
        block = original_test_hits_result(
            words=words,
            before_recall=before,
            before_greedy=None,
            after_recall=after,
            interp_after={"recall_at_n": 0.5, "recall_greedy": 0.0, "embed_sim": None, "embed_sim_gte_tau": None},
        )
        self.assertEqual(block["n_still_misses"], 2)
        self.assertAlmostEqual(block["still_miss_rate"], 2 / 3)
        self.assertIn("interp", block)
        self.assertEqual(block["interp"]["after"]["recall_at_n"], 0.5)

    def test_cell_from_panel_uses_fixed_keys(self):
        panel = {
            "recall_at_n_temp1.0": 0.4,
            "recall_greedy": 0.1,
            "embed_sim": 0.8,
            "embed_sim_gte_tau": 0.5,
        }
        cell = cell_from_panel(panel, 1.0)
        self.assertEqual(
            set(cell.keys()),
            {"recall_at_n", "recall_greedy", "embed_sim", "embed_sim_gte_tau"},
        )
        self.assertEqual(cell["recall_at_n"], 0.4)

    def test_origin_slices_do_not_overwrite_duplicate_words(self):
        words = ["same", "same"]
        greedy = ["same", "other"]
        temp_results = {
            1.0: [
                {"samples": ["same"]},
                {"samples": ["other"]},
            ]
        }
        old_recall, old_greedy = hit_maps_from_slice(
            words, greedy, temp_results, [0], 1.0
        )
        new_recall, new_greedy = hit_maps_from_slice(
            words, greedy, temp_results, [1], 1.0
        )
        self.assertEqual(old_recall, {"same": True})
        self.assertEqual(old_greedy, {"same": True})
        self.assertEqual(new_recall, {"same": False})
        self.assertEqual(new_greedy, {"same": False})


class SummarySchemaTest(unittest.TestCase):
    def test_build_extend_summary_shape(self):
        results = {
            "original_train": original_train_result(
                words=["a", "b"],
                before_recall={"a": True, "b": True},
                before_greedy=None,
                after_recall={"a": True, "b": True},
            ),
            "original_test_misses": original_test_misses_result(
                words=["x"],
                before_recall={"x": False},
                before_greedy=None,
                after_recall={"x": True},
            ),
            "original_test_hits": original_test_hits_result(
                words=["p", "q"],
                before_recall={"p": True, "q": False},
                before_greedy=None,
                after_recall={"p": True, "q": True},
            ),
        }
        summary = build_extend_summary(
            paths={"source_checkpoint": "/src", "extended_checkpoint": "/dst"},
            question={"miss_temp": 1.0, "n_samples": 25},
            setup={
                "task": "semantle",
                "source": "recall_at_n_misses",
                "n_original_train": 2,
                "n_original_test_misses": 1,
                "n_original_test_hits": 2,
            },
            training={"epochs": 10, "steps": 100, "mean_train_loss": 1.5},
            results=results,
            artifacts={
                "new_targets": "new_targets.jsonl",
                "eval_history": "eval_history.json",
                "learn_config": "learn_config.json",
                "post_eval": "eval/results.json",
            },
            created_at="2026-01-01T00:00:00Z",
        )
        self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
        self.assertEqual(summary["question"]["goal"], EXTEND_QUESTION_GOAL)
        self.assertEqual(summary["question"]["miss_temp"], 1.0)
        for bulky in ("new_targets", "eval_history", "learn_config"):
            self.assertNotIn(bulky, summary)
            self.assertIsInstance(summary["artifacts"][bulky], str)
        self.assertNotIn("recon_old", summary)
        self.assertIn("n_forgotten", summary["results"]["original_train"])
        self.assertIn("n_recovered", summary["results"]["original_test_misses"])
        self.assertIn("n_still_misses", summary["results"]["original_test_hits"])
        line = headline_console_line(summary["results"])
        self.assertIn("forgotten=", line)
        self.assertIn("recovered=", line)
        self.assertIn("still_test_misses=", line)

    def test_write_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = write_extend_sidecars(
                tmp,
                new_words=["alpha", "beta"],
                new_targets_defs={"alpha": "def a"},
                eval_history=[{"epoch": 1, "avg_embed_sim": 0.5}],
                learn_config={"epochs": 5, "steps": 10},
            )
            self.assertTrue(os.path.isfile(os.path.join(tmp, artifacts["new_targets"])))
            self.assertTrue(os.path.isfile(os.path.join(tmp, artifacts["eval_history"])))
            self.assertTrue(os.path.isfile(os.path.join(tmp, artifacts["learn_config"])))
            with open(os.path.join(tmp, artifacts["new_targets"]), encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            self.assertEqual(rows[0]["target"], "alpha")
            self.assertEqual(rows[0]["definition"], "def a")


class WandbPayloadTest(unittest.TestCase):
    def test_greedy_decode_rows_sort_descending(self):
        rows = sorted_greedy_decode_rows(
            ["alpha", "beta", "gamma"],
            ["other", "beta", "near"],
            [0.2, 1.0, 0.7],
        )
        self.assertEqual(
            [row["target"] for row in rows],
            ["beta", "gamma", "alpha"],
        )
        self.assertEqual([row["rank"] for row in rows], [1, 2, 3])
        self.assertTrue(rows[0]["exact_match"])
        self.assertFalse(rows[1]["exact_match"])

    def test_molopt_exact_match_compares_canonical_smiles(self):
        rows = sorted_greedy_decode_rows(
            ["CCO", "C1CCCCC1"], ["OCC", "c1ccccc1"], [0.9, 0.9], task="molopt"
        )
        by_target = {row["target"]: row["exact_match"] for row in rows}
        self.assertTrue(by_target["CCO"])
        self.assertFalse(by_target["C1CCCCC1"])

    def test_wandb_metrics_define_shared_train_global_step_axis(self):
        run = types.SimpleNamespace(define_metric=mock.Mock())
        _define_extend_wandb_metrics(run)
        run.define_metric.assert_any_call("train/global_step")
        run.define_metric.assert_any_call(
            "train/loss", step_metric="train/global_step"
        )
        run.define_metric.assert_any_call(
            "eval/epoch", step_metric="train/global_step"
        )
        run.define_metric.assert_any_call(
            "eval/avg_embed_sim", step_metric="train/global_step"
        )
        run.define_metric.assert_any_call(
            "eval/previous/avg_embed_sim", step_metric="train/global_step"
        )
        run.define_metric.assert_any_call(
            "eval/new/recover_rate", step_metric="train/global_step"
        )
        run.define_metric.assert_any_call(
            "eval/combined/min_embed_sim", step_metric="train/global_step"
        )

    def test_live_callback_streams_train_and_eval_panels(self):
        run = types.SimpleNamespace(log=mock.Mock())
        callback = _make_live_wandb_progress_callback(
            run, n_eval_targets=4
        )
        callback(
            {
                "event": "log",
                "step": 12,
                "epoch": 2.0,
                "loss": 1.5,
                "learning_rate": 1e-4,
                "grad_norm": 0.7,
            }
        )
        run.log.assert_called_once_with(
            {
                "train/loss": 1.5,
                "train/learning_rate": 1e-4,
                "train/grad_norm": 0.7,
                "train/global_step": 12.0,
                "train/epoch": 2.0,
            }
        )

        decode_rows = [
            {
                "rank": 1,
                "target": "alpha",
                "greedy_decode": "alpha",
                "embed_similarity": 1.0,
                "exact_match": True,
            }
        ]
        with mock.patch.object(
            ext, "_wandb_eval_decode_table", return_value="decode-table"
        ) as table_mock:
            callback(
                {
                    "event": "eval_result",
                    "epoch": 5.0,
                    "step": 48,
                    "avg_embed_sim": 0.6,
                    "min_embed_sim": 0.2,
                    "embed_sim_gte_tau": 0.5,
                    "embed_sim_tau": 0.8,
                    "n_recovered": 3,
                    "n_targets": 4,
                    "greedy_decodes": decode_rows,
                }
            )
        table_mock.assert_called_once_with(decode_rows)
        self.assertEqual(run.log.call_count, 2)
        run.log.assert_called_with(
            {
                "eval/avg_embed_sim": 0.6,
                "eval/min_embed_sim": 0.2,
                "eval/embed_sim_gte_tau": 0.5,
                "eval/embed_sim_tau": 0.8,
                "eval/n_recovered": 3.0,
                "eval/epoch": 5.0,
                "train/global_step": 48.0,
                "eval/n_targets": 4.0,
                "eval/recover_rate": 0.75,
                "eval/greedy_decodes": "decode-table",
            }
        )

    def test_live_callback_streams_grouped_eval_panels(self):
        run = types.SimpleNamespace(log=mock.Mock())
        callback = _make_live_wandb_progress_callback(run, n_eval_targets=2)
        with mock.patch.object(
            ext, "_wandb_eval_decode_table", side_effect=lambda rows: f"t-{len(rows)}"
        ):
            callback(
                {
                    "event": "eval_result",
                    "epoch": 2.0,
                    "step": 20,
                    "avg_embed_sim": 0.5,
                    "min_embed_sim": 0.4,
                    "embed_sim_gte_tau": 0.5,
                    "embed_sim_tau": 0.8,
                    "n_recovered": 1,
                    "n_targets": 2,
                    "greedy_decodes": [{"rank": 1}],
                    "groups": {
                        "previous": {
                            "avg_embed_sim": 0.9,
                            "min_embed_sim": 0.8,
                            "embed_sim_gte_tau": 1.0,
                            "embed_sim_tau": 0.8,
                            "n_recovered": 3,
                            "n_targets": 3,
                            "greedy_decodes": [{"rank": 1}, {"rank": 2}, {"rank": 3}],
                        },
                        "new": {
                            "avg_embed_sim": 0.5,
                            "min_embed_sim": 0.4,
                            "embed_sim_gte_tau": 0.5,
                            "embed_sim_tau": 0.8,
                            "n_recovered": 1,
                            "n_targets": 2,
                            "greedy_decodes": [{"rank": 1}, {"rank": 2}],
                        },
                        "combined": {
                            "avg_embed_sim": 0.7,
                            "min_embed_sim": 0.4,
                            "embed_sim_gte_tau": 0.8,
                            "embed_sim_tau": 0.8,
                            "n_recovered": 4,
                            "n_targets": 5,
                            "greedy_decodes": [{"rank": 1}],
                        },
                    },
                }
            )
        logged = run.log.call_args[0][0]
        self.assertEqual(logged["eval/previous/avg_embed_sim"], 0.9)
        self.assertEqual(logged["eval/previous/recover_rate"], 1.0)
        self.assertEqual(logged["eval/new/avg_embed_sim"], 0.5)
        self.assertEqual(logged["eval/new/recover_rate"], 0.5)
        self.assertEqual(logged["eval/combined/n_recovered"], 4.0)
        self.assertEqual(logged["eval/combined/recover_rate"], 0.8)
        self.assertEqual(logged["eval/previous/greedy_decodes"], "t-3")
        self.assertEqual(logged["eval/new/greedy_decodes"], "t-2")

    def test_training_curve_logs(self):
        rows = training_curve_logs(
            [
                {
                    "epoch": 5,
                    "avg_embed_sim": 0.4,
                    "min_embed_sim": 0.2,
                    "embed_sim_gte_tau": 0.3,
                    "embed_sim_tau": 0.8,
                    "n_recovered": 3,
                    "n_targets": 10,
                },
                {
                    "epoch": 10,
                    "step": 80,
                    "avg_embed_sim": 0.6,
                    "min_embed_sim": 0.3,
                    "embed_sim_gte_tau": 0.5,
                    "embed_sim_tau": 0.8,
                    "n_recovered": 7,
                    "n_targets": 10,
                },
            ]
        )
        self.assertEqual(rows[0]["_step"], 5)
        self.assertEqual(rows[0]["train/global_step"], 5.0)
        self.assertEqual(rows[0]["eval/epoch"], 5.0)
        self.assertEqual(rows[0]["eval/avg_embed_sim"], 0.4)
        self.assertEqual(rows[0]["eval/embed_sim_gte_tau"], 0.3)
        self.assertEqual(rows[0]["eval/embed_sim_tau"], 0.8)
        self.assertEqual(rows[1]["_step"], 80)
        self.assertEqual(rows[1]["train/global_step"], 80.0)
        self.assertEqual(rows[1]["eval/epoch"], 10.0)
        self.assertEqual(rows[1]["eval/n_recovered"], 7.0)
        self.assertEqual(rows[1]["eval/recover_rate"], 0.7)

    def test_wandb_log_payloads(self):
        results = {
            "original_train": {
                "n": 2,
                "before": {"recall_at_n": 1.0},
                "after": {"recall_at_n": 1.0},
                "n_forgotten": 0,
            },
            "original_test_misses": {
                "n": 4,
                "before": {"recall_at_n": 0.0},
                "after": {"recall_at_n": 0.5},
                "n_recovered": 2,
            },
            "original_test_hits": {
                "n": 10,
                "before": {"recall_at_n": 0.8},
                "after": {"recall_at_n": 0.7},
                "n_still_misses": 3,
            },
        }
        summary = build_extend_summary(
            paths={"source_checkpoint": "/s", "extended_checkpoint": "/e"},
            question={"miss_temp": 1.0, "n_samples": 25},
            setup={"task": "semantle", "source": "recall_at_n_misses"},
            training={"epochs": 2, "steps": 20, "mean_train_loss": 1.0},
            results=results,
            artifacts={},
        )
        payloads = wandb_log_payloads(
            summary,
            [{"epoch": 1, "avg_embed_sim": 0.5, "min_embed_sim": 0.1, "n_recovered": 1}],
            raw_panels={"recon_new": {"recall_at_n_temp1.0": 0.5}},
        )
        self.assertEqual(len(payloads["curves"]), 1)
        self.assertIn("results/original_train/n_forgotten", payloads["final"])
        self.assertIn("raw/recon_new/recall_at_n_temp1.0", payloads["final"])
        bar = before_after_bar_rows(results)
        self.assertEqual(len(bar), 6)  # 3 pops × before+after
        self.assertEqual(
            [row["population"] for row in bar],
            [
                "original_train",
                "original_train",
                "original_test_misses",
                "original_test_misses",
                "original_test_hits",
                "original_test_hits",
            ],
        )
        self.assertEqual(payloads["bar_rows"], bar)

    def test_logger_uses_step_after_last_curve_for_headlines(self):
        logged = []
        fake_wandb = types.SimpleNamespace(
            init=mock.Mock(return_value=object()),
            log=mock.Mock(side_effect=lambda payload, step: logged.append((payload, step))),
            finish=mock.Mock(),
            Table=mock.Mock(return_value="table"),
            plot=types.SimpleNamespace(bar=mock.Mock(return_value="chart")),
        )
        args = types.SimpleNamespace(
            wandb_project="project",
            wandb_entity=None,
            wandb_run_name=None,
            wandb_group=None,
            wandb_dir=None,
        )
        summary = build_extend_summary(
            paths={"source_checkpoint": "/s", "extended_checkpoint": "/e"},
            question={"miss_temp": 1.0, "n_samples": 25},
            setup={"task": "semantle"},
            training={"epochs": 5, "steps": 10},
            results={
                "original_train": {
                    "n": 1,
                    "before": {"recall_at_n": 1.0},
                    "after": {"recall_at_n": 1.0},
                    "n_forgotten": 0,
                }
            },
            artifacts={},
        )
        with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
            _log_extend_to_wandb(
                summary,
                [{"epoch": 5, "avg_embed_sim": 0.5}],
                args=args,
                output_dir="/tmp/extend",
                wandb_config={"result_dir": "/source"},
            )
        self.assertEqual([step for _, step in logged], [5, 6])
        fake_wandb.init.assert_called_once()
        fake_wandb.finish.assert_called_once()

    def test_logger_finishes_existing_live_run_without_replaying_curves(self):
        logged = []
        config = mock.Mock()
        run = types.SimpleNamespace(config=config)
        fake_wandb = types.SimpleNamespace(
            init=mock.Mock(),
            log=mock.Mock(
                side_effect=lambda payload, step=None: logged.append((payload, step))
            ),
            finish=mock.Mock(),
            Table=mock.Mock(return_value="table"),
            plot=types.SimpleNamespace(bar=mock.Mock(return_value="chart")),
        )
        args = types.SimpleNamespace(
            wandb_project="project",
            wandb_entity=None,
            wandb_run_name=None,
            wandb_group=None,
            wandb_dir=None,
        )
        summary = build_extend_summary(
            paths={"source_checkpoint": "/s", "extended_checkpoint": "/e"},
            question={"miss_temp": 1.0, "n_samples": 25},
            setup={"task": "semantle"},
            training={"epochs": 5, "steps": 10},
            results={},
            artifacts={},
        )
        with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
            _log_extend_to_wandb(
                summary,
                [{"epoch": 5, "avg_embed_sim": 0.5}],
                args=args,
                output_dir="/tmp/extend",
                wandb_config={"result_dir": "/source"},
                run=run,
                log_curves=False,
            )
        fake_wandb.init.assert_not_called()
        self.assertEqual([step for _, step in logged], [None])
        config.update.assert_called_once()
        fake_wandb.finish.assert_called_once()


if __name__ == "__main__":
    unittest.main()
