"""Tests for DIST metric helpers in eval_suite."""

from __future__ import annotations

import unittest

import numpy as np

from boreft.eval.eval_suite import dist_metrics


class DistMetricsTest(unittest.TestCase):
    def test_notarget_embed_sim_filters_target_samples(self):
        temp_results = {
            1.0: [
                {
                    "samples": ["apple", "pear", "pear", "plum"],
                    "unique_samples": ["apple", "pear", "plum"],
                    "per_sample_sims": [1.0, 0.4, 0.8],
                },
                {
                    "samples": ["BANANA", " banana "],
                    "unique_samples": ["BANANA", " banana "],
                    "per_sample_sims": [1.0, 0.99],
                },
            ],
        }

        metrics = dist_metrics(targets=["apple", "banana"], temp_results=temp_results)

        # "apple" is dropped as the target; "pear"/"plum" remain. Both "BANANA" and
        # " banana " normalize to the target, so that bag contributes nothing.
        apple_notarget_sims = np.array([0.4, 0.4, 0.8])
        self.assertAlmostEqual(
            metrics["embed_sim_notarget_temp1.0"],
            float((apple_notarget_sims.mean() + 0.0) / 2),
        )
        self.assertAlmostEqual(
            metrics["embed_sim_notarget_std_temp1.0"],
            float((apple_notarget_sims.std() + 0.0) / 2),
        )


if __name__ == "__main__":
    unittest.main()
