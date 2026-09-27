"""Unit tests for geometry-based coreset selectors."""

from __future__ import annotations

import unittest

import numpy as np

from boreft.coreset import (
    herding_indices,
    k_center_greedy_indices,
    select_subset_indices,
)


class HerdingIndicesTest(unittest.TestCase):
    def test_size_unique_deterministic(self):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(20, 3)).astype(np.float64)
        a = herding_indices(features, 5)
        b = herding_indices(features, 5)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 5)
        self.assertEqual(len(set(a)), 5)
        self.assertTrue(all(0 <= i < 20 for i in a))

    def test_k_bounds(self):
        features = np.eye(4, dtype=np.float64)
        self.assertEqual(herding_indices(features, 0), [])
        self.assertEqual(herding_indices(features, 4), [0, 1, 2, 3])
        self.assertEqual(herding_indices(features, 10), [0, 1, 2, 3])

    def test_first_pick_aligns_with_mean(self):
        # One point near the mean of three unit axes should be preferred first
        # after L2-normalization of a skewed cloud.
        features = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [10.0, 10.0],
            ],
            dtype=np.float64,
        )
        idxs = herding_indices(features, 1)
        self.assertEqual(len(idxs), 1)
        # After L2-norm, [10,10] and [1,1] coincide; lowest-index tie -> 2.
        self.assertEqual(idxs[0], 2)


class KCenterGreedyIndicesTest(unittest.TestCase):
    def test_size_unique_deterministic(self):
        rng = np.random.default_rng(1)
        features = rng.normal(size=(15, 2)).astype(np.float64)
        a = k_center_greedy_indices(features, 4)
        b = k_center_greedy_indices(features, 4)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 4)
        self.assertEqual(len(set(a)), 4)

    def test_covers_separated_clusters(self):
        # Four well-separated clusters; k=4 should pick one from each.
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
        idxs = k_center_greedy_indices(features, 4)
        clusters = []
        for i in idxs:
            x, y = features[i]
            if x > 5:
                clusters.append("e")
            elif x < -5:
                clusters.append("w")
            elif y > 5:
                clusters.append("n")
            else:
                clusters.append("s")
        self.assertEqual(sorted(clusters), ["e", "n", "s", "w"])

    def test_first_is_farthest_from_mean(self):
        features = np.array(
            [
                [0.1, 0.1],
                [1.0, 0.0],
                [0.0, 1.0],
                [5.0, 5.0],
            ],
            dtype=np.float64,
        )
        mean = features.mean(axis=0)
        expected_first = int(np.argmax(np.linalg.norm(features - mean, axis=1)))
        self.assertEqual(expected_first, 3)
        self.assertEqual(k_center_greedy_indices(features, 1), [expected_first])

    def test_rejects_nonfinite_features(self):
        features = np.array([[1.0, 0.0], [np.nan, 0.0]], dtype=np.float64)
        with self.assertRaises(ValueError):
            herding_indices(features, 1)
        with self.assertRaises(ValueError):
            k_center_greedy_indices(features, 1)


class SelectSubsetIndicesTest(unittest.TestCase):
    def test_all_keeps_every_row(self):
        self.assertEqual(select_subset_indices(4, strategy="all", seed=0), [0, 1, 2, 3])

    def test_all_rejects_n(self):
        with self.assertRaisesRegex(ValueError, "all"):
            select_subset_indices(4, strategy="all", n=2, seed=0)

    def test_unset_count_keeps_all_for_random(self):
        self.assertEqual(
            select_subset_indices(3, strategy="random", seed=0),
            [0, 1, 2],
        )

    def test_random_is_deterministic(self):
        a = select_subset_indices(10, n=3, strategy="random", seed=7)
        b = select_subset_indices(10, n=3, strategy="random", seed=7)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 3)
        self.assertEqual(a, sorted(a))


if __name__ == "__main__":
    unittest.main()
