"""Tests for LIPZ trajectory metrics in eval_suite.

Covers peakiness, detour, and the arc-length normalization of emp_l (which makes
it invariant to the run's bias-space scale).
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from boreft.eval.eval_suite import trajectory_step_summary


def _unit(theta: float) -> list[float]:
    return [math.cos(theta), math.sin(theta)]


class LipzTrajectoryTest(unittest.TestCase):
    def test_uniform_arc_is_smooth(self):
        # Equal angular steps along the unit circle + equal bias steps => uniform
        # semantic movement: peakiness == 1 and detour == 1 (geodesic).
        d = 0.2
        cents = [_unit(i * d) for i in range(5)]
        bias = [[float(i)] for i in range(5)]
        out = trajectory_step_summary(cents, bias)
        self.assertAlmostEqual(out["peakiness"], 1.0, places=6)
        self.assertAlmostEqual(out["peakiness_p95"], 1.0, places=6)
        self.assertAlmostEqual(out["detour"], 1.0, places=6)

    def test_flat_then_jump_has_high_peakiness_but_direct(self):
        # Stay at A for several steps, then jump to B: movement is concentrated in
        # one step => high peakiness, but the path is still direct => detour == 1.
        A, B = [1.0, 0.0], [0.0, 1.0]
        cents = [A, A, A, B, B]
        bias = [[float(i)] for i in range(5)]
        out = trajectory_step_summary(cents, bias)
        # steps d_sem = [0, 0, 1, 0] -> max/mean = 1 / 0.25 = 4
        self.assertAlmostEqual(out["peakiness"], 4.0, places=4)
        # p95 of [0,0,0,1] ~= 0.85 -> 0.85 / 0.25 = 3.4 (robust: below the max-based 4)
        self.assertAlmostEqual(out["peakiness_p95"], 3.4, places=4)
        self.assertLess(out["peakiness_p95"], out["peakiness"])
        self.assertAlmostEqual(out["detour"], 1.0, places=4)

    def test_wandering_path_has_high_detour(self):
        # Oscillate A -> B -> A -> B: angular arc-length is 3x the geodesic.
        A, B = [1.0, 0.0], [0.0, 1.0]
        cents = [A, B, A, B]
        bias = [[float(i)] for i in range(4)]
        out = trajectory_step_summary(cents, bias)
        self.assertAlmostEqual(out["detour"], 3.0, places=4)

    def test_emp_l_is_invariant_to_bias_scale(self):
        # Same semantic centroids, bias scaled 10x: arc-length normalization makes
        # emp_l / emp_l_p95 identical (the old raw d_sem/d_bias would differ 10x).
        cents = [_unit(0.0), _unit(0.1), _unit(0.5), _unit(0.7)]
        bias_small = [[0.0], [1.0], [2.0], [3.0]]
        bias_large = [[0.0], [10.0], [20.0], [30.0]]
        out_s = trajectory_step_summary(cents, bias_small)
        out_l = trajectory_step_summary(cents, bias_large)
        self.assertAlmostEqual(out_s["emp_l"], out_l["emp_l"], places=6)
        self.assertAlmostEqual(out_s["emp_l_p95"], out_l["emp_l_p95"], places=6)

    def test_emp_l_uses_normalized_bias_arc_length(self):
        # Non-uniform bias steps: emp_l is d_sem per unit *normalized* bias arc, so
        # a step covering little bias arc but big semantics has a large ratio.
        cents = [_unit(0.0), _unit(0.05), _unit(1.25), _unit(1.30)]
        bias = [[0.0], [0.5], [2.5], [3.0]]  # d_bias = [0.5, 2.0, 0.5]
        out = trajectory_step_summary(cents, bias)
        self.assertGreaterEqual(out["emp_l"], out["emp_l_p95"])
        self.assertGreater(out["emp_l"], 0.0)
        # Still invariant to an overall bias rescale.
        out_scaled = trajectory_step_summary(cents, [[10.0 * b[0]] for b in bias])
        self.assertAlmostEqual(out["emp_l"], out_scaled["emp_l"], places=6)

    def test_collapsed_path_returns_none(self):
        # Identical centroids -> no semantic movement -> peakiness/detour undefined.
        A = [1.0, 0.0]
        cents = [A, A, A, A]
        bias = [[float(i)] for i in range(4)]
        out = trajectory_step_summary(cents, bias)
        self.assertIsNone(out["peakiness"])
        self.assertIsNone(out["peakiness_p95"])
        self.assertIsNone(out["detour"])

    def test_too_few_points(self):
        out = trajectory_step_summary([[1.0, 0.0]], [[0.0]])
        self.assertEqual(out["mean_step"], 0.0)
        self.assertIsNone(out["peakiness"])
        self.assertIsNone(out["peakiness_p95"])
        self.assertIsNone(out["detour"])
        self.assertIsNone(out["emp_l"])


if __name__ == "__main__":
    unittest.main()
