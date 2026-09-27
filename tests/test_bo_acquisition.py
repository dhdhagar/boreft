from __future__ import annotations

from unittest.mock import patch
import math
import unittest

import numpy as np
import torch

from boreft.bo import (
    LatentEllipsoid,
    SurrogateFitConfig,
    box_log_volume,
    fit_surrogate,
    latent_bounds,
    latent_ellipsoid,
    propose_candidates,
    propose_expected_improvement,
    score_discrete_candidates,
    select_discrete_candidates,
    unit_ball_log_volume,
)
from boreft.bo.acquisition import union_latent_bounds


class LatentBoundsTest(unittest.TestCase):
    def test_padding_and_constant_dimension(self):
        points = np.array([[0.0, 2.0], [1.0, 2.0]])
        bounds = latent_bounds(points, padding=0.1)
        self.assertAlmostEqual(bounds[0, 0], -0.1)
        self.assertAlmostEqual(bounds[1, 0], 1.1)
        self.assertGreater(bounds[1, 1], bounds[0, 1])

    def test_rejects_negative_padding(self):
        with self.assertRaisesRegex(ValueError, "padding"):
            latent_bounds([[0.0], [1.0]], padding=-0.1)

    def test_std_k_zero_matches_mean_box(self):
        points = np.array([[0.0, 1.0], [2.0, -1.0]])
        std = np.array([[10.0, 10.0], [10.0, 10.0]])
        np.testing.assert_allclose(
            latent_bounds(points, std=std, std_k=0.0),
            latent_bounds(points),
        )

    def test_std_k_expands_per_row_std(self):
        mu = np.array([[0.0, 1.0], [2.0, 1.0]])
        std = np.array([[0.5, 0.1], [0.0, 3.0]])
        bounds = latent_bounds(mu, std=std, std_k=2.0)
        self.assertAlmostEqual(bounds[0, 0], -1.0)
        self.assertAlmostEqual(bounds[1, 0], 2.0)
        self.assertAlmostEqual(bounds[0, 1], -5.0)
        self.assertAlmostEqual(bounds[1, 1], 7.0)

    def test_std_k_then_padding(self):
        mu = np.array([[0.0], [1.0]])
        std = np.array([[1.0], [0.0]])
        bounds = latent_bounds(mu, padding=0.1, std=std, std_k=1.0)
        self.assertAlmostEqual(bounds[0, 0], -1.2)
        self.assertAlmostEqual(bounds[1, 0], 1.2)

    def test_std_k_requires_matching_nonnegative_std(self):
        mu = np.array([[0.0], [1.0]])
        with self.assertRaisesRegex(ValueError, "std is required"):
            latent_bounds(mu, std_k=1.0)
        with self.assertRaisesRegex(ValueError, "std_k"):
            latent_bounds(mu, std=np.array([[0.0], [0.0]]), std_k=-0.1)
        with self.assertRaisesRegex(ValueError, "std_k"):
            latent_bounds(mu, std=np.array([[0.0], [0.0]]), std_k=float("nan"))
        with self.assertRaisesRegex(ValueError, "std_k"):
            latent_bounds(mu, std=np.array([[0.0], [0.0]]), std_k=float("inf"))
        with self.assertRaisesRegex(ValueError, "shape"):
            latent_bounds(mu, std=np.array([[0.0]]), std_k=1.0)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            latent_bounds(mu, std=np.array([[-0.1], [0.0]]), std_k=1.0)

    def test_union_latent_bounds(self):
        left = np.array([[0.0, -1.0], [1.0, 0.0]])
        right = np.array([[-2.0, 0.0], [0.5, 3.0]])
        union = union_latent_bounds(left, right)
        np.testing.assert_allclose(union[0], [-2.0, -1.0])
        np.testing.assert_allclose(union[1], [1.0, 3.0])
        with self.assertRaisesRegex(ValueError, "shape"):
            union_latent_bounds(left, np.array([[0.0], [1.0]]))


class LatentEllipsoidTest(unittest.TestCase):
    def test_encloses_means_and_aabb_contains_points(self):
        mu = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        ell = latent_ellipsoid(mu)
        self.assertTrue(np.all(ell.contains(mu)))
        box = ell.aabb()
        self.assertTrue(np.all(mu.min(axis=0) >= box[0] - 1e-8))
        self.assertTrue(np.all(mu.max(axis=0) <= box[1] + 1e-8))
        mean_box = latent_bounds(mu)
        self.assertTrue(np.all(box[0] <= mean_box[0] + 1e-8))
        self.assertTrue(np.all(box[1] >= mean_box[1] - 1e-8))

    def test_std_k_expands_radius(self):
        mu = np.array([[0.0, 0.0], [1.0, 0.0]])
        std = np.array([[0.5, 2.0], [0.0, 0.0]])
        tight = latent_ellipsoid(mu)
        wide = latent_ellipsoid(mu, std=std, std_k=2.0)
        vertex = mu[0] + np.array([0.0, 4.0])
        self.assertFalse(bool(tight.contains(vertex).item()))
        self.assertTrue(bool(wide.contains(vertex).item()))
        self.assertGreater(wide.aabb()[1, 1] - wide.aabb()[0, 1], tight.aabb()[1, 1] - tight.aabb()[0, 1])

    def test_padding_inflates(self):
        mu = np.array([[0.0, 0.0], [1.0, 0.0]])
        base = latent_ellipsoid(mu)
        padded = latent_ellipsoid(mu, padding=0.1)
        self.assertGreater(np.linalg.det(padded.chol), np.linalg.det(base.chol))

    def test_project_is_identity_inside(self):
        mu = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        ell = latent_ellipsoid(mu)
        np.testing.assert_allclose(ell.project(mu[0]), mu[0], atol=1e-8)
        far = np.array([10.0, 10.0])
        projected = ell.project(far)
        self.assertTrue(bool(ell.contains(projected, atol=1e-6).item()))
        self.assertAlmostEqual(float(ell.mahalanobis(projected)[0]), 1.0, places=5)

    def test_padding_matches_aabb_in_1d(self):
        mu = np.array([[0.0], [1.0]])
        box = latent_bounds(mu, padding=0.1)
        ell = latent_ellipsoid(mu, padding=0.1)
        np.testing.assert_allclose(ell.aabb(), box, atol=1e-8)

    def test_minimum_span_floor_preserves_covering(self):
        mu = np.array([[0.0, 0.0], [1.0, 10.0]])
        ell = latent_ellipsoid(mu, minimum_span=20.0)
        self.assertTrue(np.all(ell.contains(mu, atol=1e-6)))
        self.assertTrue(np.all(np.linalg.norm(ell.chol, axis=1) >= 10.0 - 1e-8))

    def test_sample_stays_inside(self):
        mu = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        ell = latent_ellipsoid(mu)
        points = ell.sample(200, seed=0)
        self.assertTrue(np.all(ell.contains(points, atol=1e-6)))

    def test_rejects_bad_std_k(self):
        mu = np.array([[0.0], [1.0]])
        with self.assertRaisesRegex(ValueError, "std is required"):
            latent_ellipsoid(mu, std_k=1.0)
        with self.assertRaisesRegex(ValueError, "std_k"):
            latent_ellipsoid(mu, std=np.array([[0.0], [0.0]]), std_k=float("nan"))


class DomainVolumeTest(unittest.TestCase):
    def test_unit_ball_known_dimensions(self):
        self.assertAlmostEqual(unit_ball_log_volume(1), math.log(2.0))
        self.assertAlmostEqual(unit_ball_log_volume(2), math.log(math.pi))
        self.assertAlmostEqual(unit_ball_log_volume(3), math.log(4.0 * math.pi / 3.0))
        with self.assertRaisesRegex(ValueError, "positive"):
            unit_ball_log_volume(0)

    def test_box_log_volume_product_of_sides(self):
        bounds = np.array([[0.0, -1.0], [2.0, 3.0]])
        self.assertAlmostEqual(box_log_volume(bounds), math.log(8.0))
        with self.assertRaisesRegex(ValueError, "positive"):
            box_log_volume(np.array([[0.0], [0.0]]))

    def test_1d_ellipsoid_matches_mean_box(self):
        mu = np.array([[0.0], [2.0]])
        box = latent_bounds(mu)
        ell = latent_ellipsoid(mu)
        np.testing.assert_allclose(ell.aabb(), box, atol=1e-10)
        self.assertAlmostEqual(box_log_volume(box), math.log(2.0))
        self.assertAlmostEqual(ell.log_volume(), math.log(2.0), places=10)

    def test_identity_ellipsoid_is_unit_ball(self):
        ell = LatentEllipsoid(center=np.zeros(2), chol=np.eye(2))
        self.assertAlmostEqual(ell.log_volume(), math.log(math.pi))
        scaled = LatentEllipsoid(center=np.zeros(2), chol=2.0 * np.eye(2))
        self.assertAlmostEqual(scaled.log_volume(), math.log(math.pi) + math.log(4.0))

    def test_aabb_std_k_increases_volume(self):
        mu = np.array([[0.0, 0.0], [1.0, 2.0]])
        std = np.array([[0.5, 0.0], [0.0, 1.0]])
        v0 = box_log_volume(latent_bounds(mu, std=std, std_k=0.0))
        v1 = box_log_volume(latent_bounds(mu, std=std, std_k=0.1))
        v2 = box_log_volume(latent_bounds(mu, std=std, std_k=0.5))
        self.assertLess(v0, v1)
        self.assertLess(v1, v2)

    def test_enclosing_aabb_has_larger_volume_than_ellipsoid(self):
        mu = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        ell = latent_ellipsoid(mu)
        self.assertGreater(box_log_volume(ell.aabb()), ell.log_volume())
        self.assertTrue(np.all(ell.contains(mu)))


class ExpectedImprovementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = np.array(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
            dtype=np.float64,
        )
        cls.bounds = latent_bounds(cls.x)
        y = -np.square(cls.x - np.array([0.7, 0.7])).sum(axis=1)
        cls.surrogate = fit_surrogate(
            cls.x, y, cls.bounds, SurrogateFitConfig(kind="static")
        )

    def test_proposal_is_bounded_and_not_duplicate(self):
        point = propose_expected_improvement(
            self.surrogate,
            self.bounds,
            self.x,
            seed=7,
            num_restarts=2,
            raw_samples=32,
        )
        self.assertTrue(np.all(point >= self.bounds[0]))
        self.assertTrue(np.all(point <= self.bounds[1]))
        distances = np.linalg.norm(point[None, :] - self.x, axis=1)
        self.assertGreater(distances.min(), 1e-6)

    def test_seed_is_deterministic(self):
        kwargs = dict(seed=11, num_restarts=2, raw_samples=32)
        first = propose_expected_improvement(
            self.surrogate, self.bounds, self.x, **kwargs
        )
        second = propose_expected_improvement(
            self.surrogate, self.bounds, self.x, **kwargs
        )
        np.testing.assert_allclose(first, second, atol=1e-6)

    def test_ellipsoid_proposals_stay_inside(self):
        ell = latent_ellipsoid(self.x)
        for acquisition in ("log_ei", "thompson"):
            points = propose_candidates(
                self.surrogate,
                ell.aabb(),
                self.x,
                acquisition=acquisition,
                batch_size=1,
                seed=7,
                num_restarts=2,
                raw_samples=32,
                mc_samples=16,
                thompson_candidates=64,
                ellipsoid=ell,
            )
            self.assertEqual(points.shape, (1, 2))
            self.assertTrue(np.all(ell.contains(points, atol=1e-4)))

    def test_ellipsoid_joint_batch_stays_inside(self):
        ell = latent_ellipsoid(self.x)
        points = propose_candidates(
            self.surrogate,
            ell.aabb(),
            self.x,
            acquisition="log_ei",
            batch_size=2,
            seed=13,
            num_restarts=2,
            raw_samples=32,
            mc_samples=16,
            ellipsoid=ell,
        )
        self.assertEqual(points.shape, (2, 2))
        self.assertTrue(np.all(ell.contains(points, atol=1e-4)))
        self.assertGreater(np.linalg.norm(points[0] - points[1]), 1e-6)

    def test_joint_logei_and_ucb_batches(self):
        for acquisition in ("log_ei", "ucb"):
            points = propose_candidates(
                self.surrogate,
                self.bounds,
                self.x,
                acquisition=acquisition,
                batch_size=2,
                seed=13,
                num_restarts=2,
                raw_samples=32,
                mc_samples=32,
            )
            self.assertEqual(points.shape, (2, 2))
            self.assertGreater(np.linalg.norm(points[0] - points[1]), 1e-6)

    def test_batch_thompson_sampling(self):
        points = propose_candidates(
            self.surrogate,
            self.bounds,
            self.x,
            acquisition="thompson",
            batch_size=2,
            seed=19,
            num_restarts=2,
            raw_samples=32,
            thompson_candidates=128,
        )
        self.assertEqual(points.shape, (2, 2))
        self.assertGreater(np.linalg.norm(points[0] - points[1]), 1e-6)

    def test_duplicate_latent_allowed_with_repeated_observations(self):
        duplicate = torch.as_tensor(self.x[:1], dtype=torch.float64)
        with patch(
            "boreft.bo.acquisition.optimize_acqf",
            return_value=(duplicate, None),
        ):
            allowed = propose_candidates(
                self.surrogate,
                self.bounds,
                self.x,
                acquisition="log_ei",
                batch_size=1,
                seed=3,
                num_restarts=1,
                raw_samples=8,
                mc_samples=8,
                observation_samples=3,
            )
            rejected = propose_candidates(
                self.surrogate,
                self.bounds,
                self.x,
                acquisition="log_ei",
                batch_size=1,
                seed=3,
                num_restarts=1,
                raw_samples=8,
                mc_samples=8,
                observation_samples=1,
            )
        np.testing.assert_allclose(allowed[0], self.x[0], atol=1e-6)
        self.assertGreater(np.linalg.norm(rejected[0] - self.x[0]), 1e-6)

    def test_discrete_logei_ranks_the_interior_optimum_first(self):
        points = np.array([[0.0, 0.0], [0.7, 0.7], [1.0, 1.0]], dtype=np.float64)
        scores = score_discrete_candidates(
            self.surrogate,
            points,
            acquisition="log_ei",
            seed=5,
            mc_samples=32,
        )
        self.assertEqual(scores.shape, (3,))
        self.assertEqual(int(np.argmax(scores)), 1)
        chosen = select_discrete_candidates(
            self.surrogate,
            points,
            acquisition="log_ei",
            batch_size=2,
            seed=5,
            mc_samples=32,
        )
        self.assertEqual(int(chosen[0]), 1)
        self.assertEqual(len(set(chosen.tolist())), 2)

    def test_discrete_thompson_returns_in_set_indices(self):
        chosen = select_discrete_candidates(
            self.surrogate,
            self.x,
            acquisition="thompson",
            batch_size=2,
            seed=19,
        )
        self.assertEqual(len(chosen), 2)
        self.assertTrue(np.all(chosen >= 0) and np.all(chosen < len(self.x)))
        self.assertEqual(len(set(chosen.tolist())), 2)


if __name__ == "__main__":
    unittest.main()
