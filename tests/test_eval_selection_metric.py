"""--eval-selection-metric: which eval metric drives early stop / checkpoints."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from boreft.data.base import ReftItem
from boreft.data.eval_callback import TargetReconEval
from boreft.learn_bias import _LearnedSetEvalCallback, learned_set_slice_metrics
from boreft.train_args import TrainConfig


class _Control:
    def __init__(self):
        self.should_training_stop = False


class _State:
    global_step = 1
    epoch = 1


def make_callback(selection_metric: str, *, task: str = "molopt", **kwargs):
    items = [
        ReftItem(id=0, prompt="p", target="CCO"),
        ReftItem(id=1, prompt="p", target="CCN"),
    ]
    return TargetReconEval(
        None,
        None,
        items,
        eval_steps=1,
        task=task,
        selection_metric=selection_metric,
        **kwargs,
    )


def run_eval(callback, *, embed, rdkit_sim, tfs):
    """Drive one eval round with the generation and scoring calls stubbed out."""
    control = _Control()
    embed_patch = (
        {"side_effect": embed}
        if isinstance(embed, Exception)
        else {"return_value": np.asarray(embed, dtype=np.float64)}
    )
    with mock.patch(
        "boreft.eval.semantle.generate_text",
        side_effect=lambda *a, **k: "CCO",
    ), mock.patch(
        "boreft.data.eval_callback.embedding_sim_per_text", **embed_patch
    ), mock.patch(
        "boreft.chem.rdkit_sim_per_text",
        return_value=np.asarray(rdkit_sim, dtype=np.float64),
    ), mock.patch(
        "boreft.chem.tanimoto_sim_per_text",
        return_value=np.asarray(tfs, dtype=np.float64),
    ), mock.patch(
        "boreft.chem.validity_rate", return_value=1.0
    ):
        callback._run_periodic_eval(_State(), control, label="test")
    return control


class SelectionMetricConfigTests(unittest.TestCase):
    def test_defaults_to_embed_sim(self):
        cfg = TrainConfig(
            task="semantle", semantle_csv=("data.csv",), eval_epochs=5
        )
        self.assertEqual(cfg.eval_selection_metric, "embed_sim")

    def test_requires_eval(self):
        with self.assertRaisesRegex(ValueError, "--eval-selection-metric"):
            TrainConfig(
                task="molopt",
                molopt_csv=("mols.csv",),
                eval_selection_metric="rdkit_sim",
            )

    def test_rejects_molecular_metric_for_word_task(self):
        with self.assertRaisesRegex(ValueError, "--eval-selection-metric"):
            TrainConfig(
                task="semantle",
                semantle_csv=("data.csv",),
                eval_epochs=5,
                eval_selection_metric="rdkit_sim",
            )

    def test_accepts_molecular_metric_for_molopt(self):
        cfg = TrainConfig(
            task="molopt",
            molopt_csv=("mols.csv",),
            eval_epochs=5,
            eval_selection_metric="tfs",
        )
        self.assertEqual(cfg.eval_selection_metric, "tfs")


class SelectionMetricCallbackValidationTests(unittest.TestCase):
    def test_rejects_unknown_metric(self):
        with self.assertRaisesRegex(ValueError, "selection_metric"):
            make_callback("morgan")

    def test_rejects_molecular_metric_for_word_task(self):
        with self.assertRaisesRegex(ValueError, "rdkit_sim"):
            make_callback("rdkit_sim", task="semantle")

    def test_embed_sim_works_for_every_task(self):
        callback = make_callback("embed_sim", task="semantle")
        self.assertEqual(callback.selection_metric, "embed_sim")


class SelectionMetricStopTests(unittest.TestCase):
    """A high selected metric stops training even when the others are low."""

    def test_rdkit_sim_selection_drives_the_stop(self):
        callback = make_callback("rdkit_sim", stop_threshold=0.9)
        control = run_eval(
            callback, embed=[0.1, 0.2], rdkit_sim=[0.95, 0.97], tfs=[0.1, 0.1]
        )
        self.assertTrue(control.should_training_stop)

    def test_embed_sim_selection_ignores_a_high_rdkit_sim(self):
        callback = make_callback("embed_sim", stop_threshold=0.9)
        control = run_eval(
            callback, embed=[0.1, 0.2], rdkit_sim=[0.95, 0.97], tfs=[0.1, 0.1]
        )
        self.assertFalse(control.should_training_stop)

    def test_selected_min_is_honored(self):
        callback = make_callback("tfs", stop_threshold_min=0.6)
        control = run_eval(
            callback, embed=[0.99, 0.99], rdkit_sim=[0.99, 0.99], tfs=[0.9, 0.5]
        )
        self.assertFalse(control.should_training_stop)

    def test_a_broken_embed_model_does_not_block_a_molecular_stop(self):
        callback = make_callback("rdkit_sim", stop_threshold=0.9)
        control = run_eval(
            callback,
            embed=RuntimeError("no embedding model"),
            rdkit_sim=[0.95, 0.97],
            tfs=[0.1, 0.1],
        )
        self.assertTrue(control.should_training_stop)

    def test_a_broken_embed_model_blocks_an_embed_stop(self):
        callback = make_callback("embed_sim", stop_threshold=0.9)
        control = run_eval(
            callback,
            embed=RuntimeError("no embedding model"),
            rdkit_sim=[0.95, 0.97],
            tfs=[0.1, 0.1],
        )
        self.assertFalse(control.should_training_stop)


class LearnedSetSliceMetricTests(unittest.TestCase):
    """learn_bias reports embed_sim and the selected metric side by side."""

    def test_defaults_to_embed_sim(self):
        metrics = learned_set_slice_metrics(
            ["apple"], ["apple"], [0.5], embed_sim_tau=0.8
        )
        self.assertEqual(metrics["selection_metric"], "embed_sim")
        self.assertAlmostEqual(metrics["avg_selection_sim"], 0.5)
        self.assertAlmostEqual(metrics["min_selection_sim"], 0.5)

    def test_embed_sim_keys_survive_a_molecular_selection(self):
        metrics = learned_set_slice_metrics(
            ["CCO", "CCN"],
            ["CCO", "O"],
            [0.1, 0.2],
            embed_sim_tau=0.8,
            stop_threshold_min=0.6,
            task="molopt",
            selection_metric="rdkit_sim",
            selection_similarities=[0.9, 0.7],
        )
        self.assertAlmostEqual(metrics["avg_embed_sim"], 0.15)
        self.assertAlmostEqual(metrics["min_embed_sim"], 0.1)
        self.assertAlmostEqual(metrics["embed_sim_gte_stop_min"], 0.0)
        self.assertEqual(metrics["selection_metric"], "rdkit_sim")
        self.assertAlmostEqual(metrics["avg_selection_sim"], 0.8)
        self.assertAlmostEqual(metrics["min_selection_sim"], 0.7)
        self.assertAlmostEqual(metrics["selection_gte_stop_min"], 1.0)

    def test_rejects_a_misaligned_selection_array(self):
        with self.assertRaisesRegex(ValueError, "parallel"):
            learned_set_slice_metrics(
                ["apple", "pear"],
                ["apple", "pear"],
                [0.1, 0.2],
                embed_sim_tau=0.8,
                selection_similarities=[0.5],
            )


class LearnedSetEvalCallbackValidationTests(unittest.TestCase):
    """Validation runs before the callback loads any embedding model."""

    def make_callback(self, task: str, selection_metric: str):
        return _LearnedSetEvalCallback(
            SimpleNamespace(saved_cfg={"task": task}),
            ["apple"],
            eval_epochs=1,
            batch_size=1,
            max_new_tokens=8,
            stop_threshold=None,
            stop_threshold_min=None,
            selection_metric=selection_metric,
        )

    def test_rejects_molecular_metric_for_word_task(self):
        with self.assertRaisesRegex(ValueError, "rdkit_sim"):
            self.make_callback("semantle", "rdkit_sim")

    def test_rejects_unknown_metric(self):
        with self.assertRaisesRegex(ValueError, "selection_metric"):
            self.make_callback("molopt", "morgan")


if __name__ == "__main__":
    unittest.main()
