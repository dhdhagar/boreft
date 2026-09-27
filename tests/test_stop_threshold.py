import unittest

from boreft.data.eval_callback import (
    embed_sim_checkpoint_thresholds_to_save,
    embed_sim_frac_at_or_above,
    embed_sim_stop_triggered,
)
from boreft.learn_bias import BatchLearnControl, batch_learn_stopped_early
from boreft.train_args import TrainConfig


class TestEmbedSimStopTriggered(unittest.TestCase):
    def test_no_thresholds(self):
        self.assertFalse(
            embed_sim_stop_triggered(0.9, 0.8, stop_threshold=None, stop_threshold_min=None)
        )

    def test_perfect_mean_always_stops(self):
        self.assertTrue(
            embed_sim_stop_triggered(1.0, 1.0, stop_threshold=None, stop_threshold_min=None)
        )
        self.assertTrue(
            embed_sim_stop_triggered(1.0, 0.5, stop_threshold=None, stop_threshold_min=None)
        )

    def test_mean_only(self):
        self.assertTrue(
            embed_sim_stop_triggered(0.85, 0.5, stop_threshold=0.8, stop_threshold_min=None)
        )
        self.assertFalse(
            embed_sim_stop_triggered(0.75, 0.9, stop_threshold=0.8, stop_threshold_min=None)
        )

    def test_min_only(self):
        self.assertTrue(
            embed_sim_stop_triggered(0.5, 0.7, stop_threshold=None, stop_threshold_min=0.6)
        )
        self.assertFalse(
            embed_sim_stop_triggered(0.95, 0.55, stop_threshold=None, stop_threshold_min=0.6)
        )

    def test_both_must_pass(self):
        self.assertTrue(
            embed_sim_stop_triggered(
                0.85, 0.7, stop_threshold=0.8, stop_threshold_min=0.6
            )
        )
        self.assertFalse(
            embed_sim_stop_triggered(
                0.85, 0.55, stop_threshold=0.8, stop_threshold_min=0.6
            )
        )
        self.assertFalse(
            embed_sim_stop_triggered(
                0.75, 0.7, stop_threshold=0.8, stop_threshold_min=0.6
            )
        )

    def test_frac_partial_pass(self):
        sims = [0.95, 0.92, 0.4, 0.3, 0.2]  # 2/5 = 0.4 above 0.9
        self.assertAlmostEqual(embed_sim_frac_at_or_above(sims, 0.9), 0.4)
        self.assertTrue(
            embed_sim_stop_triggered(
                0.55,
                0.2,
                stop_threshold=None,
                stop_threshold_min=0.9,
                stop_threshold_frac=0.4,
                sims=sims,
            )
        )
        self.assertFalse(
            embed_sim_stop_triggered(
                0.55,
                0.2,
                stop_threshold=None,
                stop_threshold_min=0.9,
                stop_threshold_frac=0.5,
                sims=sims,
            )
        )

    def test_frac_one_matches_min(self):
        sims = [0.7, 0.65, 0.62]
        self.assertTrue(
            embed_sim_stop_triggered(
                0.66,
                0.62,
                stop_threshold_min=0.6,
                stop_threshold_frac=1.0,
                sims=sims,
            )
        )
        self.assertFalse(
            embed_sim_stop_triggered(
                0.66,
                0.55,
                stop_threshold_min=0.6,
                stop_threshold_frac=1.0,
                sims=[0.7, 0.65, 0.55],
            )
        )

    def test_frac_without_sims_requires_full_pass(self):
        with self.assertRaisesRegex(ValueError, "sims is required"):
            embed_sim_stop_triggered(
                0.9,
                0.5,
                stop_threshold_min=0.6,
                stop_threshold_frac=0.8,
            )


class TestStopThresholdMinValidation(unittest.TestCase):
    def test_requires_eval(self):
        with self.assertRaisesRegex(ValueError, "--stop-threshold-min"):
            TrainConfig(
                task="semantle",
                semantle_csv=("data.csv",),
                stop_threshold_min=0.6,
            )

    def test_rejects_out_of_range(self):
        with self.assertRaisesRegex(ValueError, "--stop-threshold-min"):
            TrainConfig(
                task="semantle",
                semantle_csv=("data.csv",),
                eval_epochs=5,
                stop_threshold_min=1.5,
            )

    def test_valid_config(self):
        cfg = TrainConfig(
            task="semantle",
            semantle_csv=("data.csv",),
            eval_epochs=5,
            stop_threshold=0.8,
            stop_threshold_min=0.6,
        )
        self.assertEqual(cfg.stop_threshold_min, 0.6)
        self.assertEqual(cfg.stop_threshold_frac, 1.0)

    def test_frac_requires_min(self):
        with self.assertRaisesRegex(ValueError, "--stop-threshold-frac"):
            TrainConfig(
                task="semantle",
                semantle_csv=("data.csv",),
                eval_epochs=5,
                stop_threshold_frac=0.8,
            )

    def test_frac_out_of_range(self):
        with self.assertRaisesRegex(ValueError, "--stop-threshold-frac"):
            TrainConfig(
                task="semantle",
                semantle_csv=("data.csv",),
                eval_epochs=5,
                stop_threshold_min=0.6,
                stop_threshold_frac=0.0,
            )

    def test_frac_valid(self):
        cfg = TrainConfig(
            task="semantle",
            semantle_csv=("data.csv",),
            eval_epochs=5,
            stop_threshold_min=0.9,
            stop_threshold_frac=0.8,
        )
        self.assertEqual(cfg.stop_threshold_frac, 0.8)


class TestEmbedSimCheckpointThresholdsToSave(unittest.TestCase):
    def test_returns_newly_crossed(self):
        done: set = set()
        crossed = embed_sim_checkpoint_thresholds_to_save(
            0.75, [0.6, 0.7, 0.8], done
        )
        self.assertEqual(crossed, [0.6, 0.7])

    def test_skips_already_done(self):
        done = {0.6}
        crossed = embed_sim_checkpoint_thresholds_to_save(
            0.75, [0.6, 0.7, 0.8], done
        )
        self.assertEqual(crossed, [0.7])

    def test_min_metric_independent_of_mean(self):
        """Min checkpoint can fire when mean would not."""
        done_mean: set = set()
        done_min: set = set()
        mean_sim, min_sim = 0.55, 0.65
        mean_crossed = embed_sim_checkpoint_thresholds_to_save(
            mean_sim, [0.6, 0.7], done_mean
        )
        min_crossed = embed_sim_checkpoint_thresholds_to_save(
            min_sim, [0.6], done_min
        )
        self.assertEqual(mean_crossed, [])
        self.assertEqual(min_crossed, [0.6])


class TestCheckpointEmbedSimThresholdsMinValidation(unittest.TestCase):
    def test_requires_eval(self):
        with self.assertRaisesRegex(ValueError, "--checkpoint-embed-sim-thresholds-min"):
            TrainConfig(
                task="semantle",
                semantle_csv=("data.csv",),
                checkpoint_embed_sim_thresholds_min=(0.6,),
            )

    def test_rejects_out_of_range(self):
        with self.assertRaisesRegex(
            ValueError, "--checkpoint-embed-sim-thresholds-min"
        ):
            TrainConfig(
                task="semantle",
                semantle_csv=("data.csv",),
                eval_epochs=5,
                checkpoint_embed_sim_thresholds_min=(0.6, 1.5),
            )

    def test_valid_config(self):
        cfg = TrainConfig(
            task="semantle",
            semantle_csv=("data.csv",),
            eval_epochs=5,
            checkpoint_embed_sim_thresholds=(0.7, 0.8),
            checkpoint_embed_sim_thresholds_min=(0.5, 0.6),
        )
        self.assertEqual(cfg.checkpoint_embed_sim_thresholds_min_list, [0.5, 0.6])


class TestBatchLearnStoppedEarly(unittest.TestCase):
    def test_user_stop(self):
        control = BatchLearnControl()
        control.stop()
        self.assertTrue(
            batch_learn_stopped_early(eval_history=[], training_control=control)
        )

    def test_threshold_stop(self):
        self.assertTrue(
            batch_learn_stopped_early(
                eval_history=[{"early_stopped": True, "avg_embed_sim": 0.9}],
                training_control=None,
            )
        )

    def test_no_stop(self):
        self.assertFalse(
            batch_learn_stopped_early(
                eval_history=[{"early_stopped": False}],
                training_control=BatchLearnControl(),
            )
        )


if __name__ == "__main__":
    unittest.main()
