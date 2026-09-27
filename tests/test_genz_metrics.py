"""Tests for GENZ metric helpers in eval_suite."""

from __future__ import annotations

import math
import unittest

import numpy as np

from boreft.eval.eval_suite import (
    genz_metrics,
    map_geary_c,
    semantic_dispersion,
    sobol_geary_c,
)


class GenzUnseenRateTest(unittest.TestCase):
    def test_unseen_temp_deduplicates_before_rate(self):
        train = ["apple", "banana"]
        metrics = genz_metrics(
            train_targets=train,
            interp_set=[],
            extrap_set=[],
            greedy_decodes=["apple", "cherry"],
            temp_decodes={
                1.0: ["fruit", "fruit", "fruit", "berry"],
            },
        )
        # 2 unique unseen texts out of 2 unique decodes
        self.assertEqual(metrics["sobol_unseen_temp1.0"], 1.0)

    def test_unseen_greedy_deduplicates_across_points(self):
        train = ["apple"]
        metrics = genz_metrics(
            train_targets=train,
            interp_set=[],
            extrap_set=[],
            greedy_decodes=["apple", "fruit", "fruit"],
            temp_decodes={},
        )
        # unique: apple (seen), fruit (unseen) -> 1/2
        self.assertEqual(metrics["sobol_unseen_greedy"], 0.5)

    def test_recall_test_covers_full_test_set(self):
        train = ["apple"]
        metrics = genz_metrics(
            train_targets=train,
            interp_set=["banana"],
            extrap_set=["cherry"],
            greedy_decodes=["banana"],
            temp_decodes={
                1.0: ["cherry", "date"],
            },
        )
        self.assertEqual(metrics["sobol_recall_test_greedy"], 0.5)
        self.assertEqual(metrics["sobol_recall_test_interp_greedy"], 1.0)
        self.assertEqual(metrics["sobol_recall_test_extrap_greedy"], 0.0)
        self.assertEqual(metrics["sobol_recall_test_temp1.0"], 0.5)


class SemanticDispersionTest(unittest.TestCase):
    def test_identical_centroids_zero_dispersion(self):
        centroids = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        self.assertAlmostEqual(semantic_dispersion(centroids), 0.0)

    def test_orthogonal_centroids_full_dispersion(self):
        # cos = 0 for every pair -> 1 - max(0, 0) = 1 each.
        centroids = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(semantic_dispersion(centroids), 1.0)

    def test_antipodal_clamped_to_one(self):
        # cos = -1, clamped by max(0, .) -> 1 - 0 = 1 (not 2).
        centroids = np.array([[1.0, 0.0], [-1.0, 0.0]])
        self.assertAlmostEqual(semantic_dispersion(centroids), 1.0)

    def test_averages_over_distinct_pairs(self):
        # Pairs: (0,1) cos=0 ->1, (0,2) cos=-1 clamp->1, (1,2) cos=0 ->1.
        centroids = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        expected = (1.0 + 1.0 + 1.0) / 3.0
        self.assertAlmostEqual(semantic_dispersion(centroids), expected)

    def test_known_angle(self):
        # 60 degrees apart -> cos = 0.5 -> dispersion 0.5.
        centroids = np.array([[1.0, 0.0], [0.5, math.sqrt(3) / 2.0]])
        self.assertAlmostEqual(semantic_dispersion(centroids), 0.5)

    def test_drops_empty_bag_rows(self):
        # A zero row (empty sample bag) is dropped, leaving one usable point.
        centroids = np.array([[1.0, 0.0], [0.0, 0.0]])
        self.assertIsNone(semantic_dispersion(centroids))

    def test_degenerate_returns_none(self):
        self.assertIsNone(semantic_dispersion(np.array([[1.0, 0.0]])))
        self.assertIsNone(semantic_dispersion(np.zeros((0, 0))))


class MapGearyCTest(unittest.TestCase):
    def test_sobol_alias_is_same_function(self):
        self.assertIs(sobol_geary_c, map_geary_c)

    def test_too_few_points_returns_none(self):
        bias = np.array([[0.0], [1.0]])
        centroids = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.assertIsNone(map_geary_c(bias, centroids))

    def test_shape_mismatch_returns_none(self):
        bias = np.array([[0.0], [1.0], [2.0]])
        centroids = np.array([[1.0, 0.0], [0.0, 1.0]])  # only 2 rows
        self.assertIsNone(map_geary_c(bias, centroids))

    def test_zero_embedding_variance_returns_none(self):
        # All centroids identical -> zero total variance -> None (not NaN).
        bias = np.array([[0.0], [1.0], [2.0], [3.0]])
        centroids = np.tile(np.array([1.0, 0.0]), (4, 1))
        self.assertIsNone(map_geary_c(bias, centroids))

    def test_smooth_map_is_more_ordered_than_shuffled(self):
        # A bias->semantics map that varies smoothly along a 1-D bias axis should
        # score a lower Geary's C than the same centroids randomly reassigned to
        # bias positions.
        n = 40
        bias = np.linspace(0.0, 1.0, n).reshape(-1, 1)
        angles = np.linspace(0.0, np.pi, n)
        centroids = np.stack([np.cos(angles), np.sin(angles)], axis=1)

        smooth_c = map_geary_c(bias, centroids)
        rng = np.random.default_rng(0)
        shuffled_c = map_geary_c(bias, centroids[rng.permutation(n)])

        self.assertIsNotNone(smooth_c)
        self.assertIsNotNone(shuffled_c)
        self.assertLess(smooth_c, 1.0)
        self.assertLess(smooth_c, shuffled_c)


if __name__ == "__main__":
    unittest.main()
