"""Tests for scripts/measure_semantle_cov_interp.py (no GPU, no checkpoints)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import unittest

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script():
    src = os.path.join(_REPO_ROOT, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    path = os.path.join(_REPO_ROOT, "scripts", "measure_semantle_cov_interp.py")
    spec = importlib.util.spec_from_file_location("measure_semantle_cov_interp", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ci = _load_script()


class SimplexProjectionTests(unittest.TestCase):
    def test_known_projection(self):
        np.testing.assert_allclose(
            ci.project_simplex(np.array([1.5, -0.5, 0.0])),
            np.array([1.0, 0.0, 0.0]),
            atol=1e-8,
        )

    def test_already_on_simplex_is_idempotent(self):
        v = np.array([0.2, 0.3, 0.5])
        w = ci.project_simplex(v)
        np.testing.assert_allclose(w, v, atol=1e-8)
        np.testing.assert_allclose(ci.project_simplex(w), w, atol=1e-8)

    def test_equal_vector_goes_to_barycenter(self):
        w = ci.project_simplex(np.array([0.4, 0.4, 0.4]))
        np.testing.assert_allclose(w, np.full(3, 1.0 / 3.0), atol=1e-8)
        self.assertAlmostEqual(float(w.sum()), 1.0)
        self.assertTrue(np.all(w >= -1e-12))


class HullProjectionTests(unittest.TestCase):
    def test_point_on_segment(self):
        points = np.array([[1.0, 0.0], [0.0, 1.0]])
        alpha, dist = ci.project_onto_hull(np.array([0.5, 0.5]), points)
        np.testing.assert_allclose(alpha, [0.5, 0.5], atol=1e-3)
        self.assertLess(dist, 1e-3)

    def test_point_outside_projects_to_vertex(self):
        points = np.array([[1.0, 0.0], [0.0, 1.0]])
        alpha, dist = ci.project_onto_hull(np.array([2.0, 0.0]), points)
        self.assertGreater(alpha[0], 0.95)
        self.assertAlmostEqual(dist, 1.0, places=2)


class EstimatorTests(unittest.TestCase):
    def test_floor_identity(self):
        x = np.array([0.6, 0.0])
        floor = ci.interpolation_floor(x)
        self.assertAlmostEqual(floor, 0.4)
        u = np.array([1.0, 0.0])
        self.assertAlmostEqual(float(np.linalg.norm(u - x)), floor)

    def test_split_half_recovers_known_offset(self):
        rng = np.random.default_rng(0)
        mean = np.array([1.0, 0.0, 0.0])
        x = np.zeros(3)
        ests = [
            ci.split_half_displacement(
                mean + rng.normal(scale=0.05, size=(64, 3)), x
            )
            for _ in range(80)
        ]
        self.assertTrue(all(v is not None for v in ests))
        self.assertAlmostEqual(float(np.mean(ests)), 1.0, delta=0.05)

    def test_plugin_is_nonnegative(self):
        u = np.array([[1.0, 0.0], [0.0, 1.0]])
        x = np.array([0.5, 0.5])
        self.assertGreaterEqual(ci.plugin_displacement(u, x), 0.0)


class WeightSetTests(unittest.TestCase):
    def test_weights_sum_to_one_and_support_size(self):
        rng = np.random.default_rng(0)
        phi = ci.l2_normalize(rng.normal(size=(12, 4)))
        weights = ci.build_weight_set(
            phi, rng, n_pairs=6, n_mix3=10, n_mix5=8, pair_grid=(0.3, 0.7)
        )
        families = {w["family"] for w in weights}
        self.assertEqual(families, {"vertices", "pairs", "mix3", "mix5"})
        for w in weights:
            self.assertAlmostEqual(sum(w["weights"]), 1.0, places=8)
            self.assertEqual(len(w["support"]), len(w["weights"]))
            if w["family"] == "mix3":
                self.assertEqual(len(w["support"]), 3)
            if w["family"] == "mix5":
                self.assertEqual(len(w["support"]), 5)
            if w["family"] == "pairs":
                self.assertEqual(len(w["support"]), 2)
                self.assertTrue(w["decode"])

    def test_seed_reproducible(self):
        phi = ci.l2_normalize(np.random.default_rng(1).normal(size=(10, 3)))
        a = ci.build_weight_set(phi, np.random.default_rng(7), n_pairs=5, n_mix3=4, n_mix5=4)
        b = ci.build_weight_set(phi, np.random.default_rng(7), n_pairs=5, n_mix3=4, n_mix5=4)
        self.assertEqual(a, b)

    def test_single_target_has_only_vertices(self):
        phi = np.array([[1.0, 0.0]])
        weights = ci.build_weight_set(
            phi, np.random.default_rng(0), n_pairs=40, n_mix3=100, n_mix5=100
        )
        self.assertTrue(all(w["family"] == "vertices" for w in weights))
        self.assertEqual(len(weights), 1)


class CoverageTests(unittest.TestCase):
    def test_min_matches_brute_force(self):
        interpolants = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        x = np.array([0.2, 0.8])
        dist, idx = ci.coverage_min(x, interpolants)
        brute = np.linalg.norm(interpolants - x, axis=1)
        self.assertEqual(idx, int(np.argmin(brute)))
        self.assertAlmostEqual(dist, float(brute.min()))

    def test_inner_product_would_pick_a_vertex(self):
        # Guardrail for the vertex collapse the notes warn about.
        points = np.array([[1.0, 0.0], [0.0, 1.0]])
        x = np.array([0.6, 0.8])
        alphas = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.5])]
        dots = [float(np.dot(x, a @ points)) for a in alphas]
        self.assertEqual(int(np.argmax(dots)), 1)
        dists = [float(np.linalg.norm(x - a @ points)) for a in alphas]
        self.assertLess(dists[2], min(dists[0], dists[1]))


class NestedAndJumpTests(unittest.TestCase):
    def test_nested_mask_is_cumulative(self):
        weights = [
            {"family": "vertices"},
            {"family": "pairs"},
            {"family": "mix3"},
            {"family": "mix5"},
        ]
        self.assertEqual(ci.nested_mask(weights, "vertices").tolist(), [True, False, False, False])
        self.assertEqual(ci.nested_mask(weights, "pairs").tolist(), [True, True, False, False])
        self.assertEqual(ci.nested_mask(weights, "mix5").tolist(), [True, True, True, True])

    def test_jump_strawman_peaks_at_half(self):
        phi = np.array([[1.0, 0.0], [-1.0, 0.0]])
        mid = {"family": "pairs", "support": [0, 1], "t": 0.5}
        early = {"family": "pairs", "support": [0, 1], "t": 0.25}
        self.assertAlmostEqual(ci.jump_strawman(phi, mid), 1.0)
        self.assertAlmostEqual(ci.jump_strawman(phi, early), 0.5)

    def test_midpath_stats_pairs_against_jump(self):
        records = [
            {
                "family": "pairs",
                "decode": True,
                "weights": [0.5, 0.5],
                "t": 0.5,
                "jump_strawman": 0.4,
                "variants": {
                    "temperature_1.0": {
                        "epsint": 0.5,
                        "excess": 0.4,
                        "displacement_split": 0.3,
                        "floor": 0.1,
                    }
                },
            },
            {
                "family": "pairs",
                "decode": True,
                "t": 0.3,
                "jump_strawman": 0.25,
                "variants": {
                    "temperature_1.0": {
                        "epsint": 0.4,
                        "excess": 0.3,
                        "displacement_split": 0.2,
                        "floor": 0.1,
                    }
                },
            },
            {
                "family": "pairs",
                "decode": True,
                "t": 0.2,
                "jump_strawman": 0.15,
                "variants": {
                    "temperature_1.0": {
                        "epsint": 0.2,
                        "excess": 0.1,
                        "displacement_split": 0.05,
                        "floor": 0.1,
                    }
                },
            },
            {
                "family": "pairs",
                "decode": True,
                "t": 0.1,
                "jump_strawman": 0.1,
                "variants": {
                    "temperature_1.0": {
                        "epsint": 0.2,
                        "excess": 0.1,
                        "displacement_split": 0.05,
                        "floor": 0.1,
                    }
                },
            },
            {
                "family": "vertices",
                "decode": True,
                "jump_strawman": None,
                "variants": {
                    "temperature_1.0": {
                        "epsint": 0.0,
                        "excess": 0.0,
                        "displacement_split": 0.0,
                        "floor": 0.0,
                    }
                },
            },
        ]
        rng = np.random.default_rng(0)
        stats = ci.midpath_stats(records, "temperature_1.0", rng)
        self.assertEqual(stats["n"], 1)
        self.assertEqual(stats["t_lo"], 0.4)
        self.assertEqual(stats["t_hi"], 0.6)
        self.assertAlmostEqual(stats["displacement_split"]["median"], 0.3)
        self.assertAlmostEqual(stats["disp_minus_jump"]["median"], -0.1)
        self.assertAlmostEqual(stats["beat_jump_disp"], 1.0)
        self.assertAlmostEqual(stats["beat_jump_epsint"], 0.0)

    def test_code_stays_in_box(self):
        mu = np.array([[0.0, 0.0], [1.0, 2.0], [0.5, 1.0]])
        b = 0.3 * mu[0] + 0.7 * mu[1]
        self.assertTrue(ci.in_bounding_box(b, mu))
        self.assertFalse(ci.in_bounding_box(np.array([1.1, 0.0]), mu))


class CliAndTableTests(unittest.TestCase):
    def test_parse_condition(self):
        parsed = ci.parse_condition("sdpo0=outputs/1788622157")
        self.assertEqual(parsed["name"], "sdpo0")
        self.assertEqual(parsed["wandb_id"], "yt8pnmyh")
        self.assertEqual(parsed["output_dir"], "outputs/1788622157")

    def test_ladder_conditions_skip_missing_dirs(self):
        path = os.path.join(
            _REPO_ROOT, "experiments", "semantle", "ladder_checkpoints.json"
        )
        rows = ci.conditions_from_ladder(path)
        names = [r["name"] for r in rows]
        self.assertIn("n1", names)
        self.assertIn("n3072", names)
        self.assertNotIn("n512", names)
        self.assertTrue(all(r.get("coverage_only") == "1" for r in rows))

    def test_latex_tables_include_n_ladder(self):
        payloads = [
            {
                "name": "n4",
                "coverage": {
                    "eval_test": {
                        "summary": {
                            "vertices": {"median": 0.4},
                            "pairs": {"median": 0.3},
                            "mix3": {"median": 0.25},
                            "mix5": {"median": 0.2},
                            "hull": {"median": 0.1},
                        }
                    }
                },
            }
        ]
        tex = ci.latex_tables(payloads)
        self.assertIn("Coverage vs training-set size", tex)
        self.assertIn("0.400", tex)
        mixed = payloads + [{"name": "novae", "coverage": payloads[0]["coverage"]}]
        tex_mixed = ci.latex_tables(mixed)
        self.assertNotIn("ovae", tex_mixed)

    def test_latex_tables_midpath_from_records(self):
        payloads = [
            {
                "name": "canonical",
                "interpolation": {
                    "records": [
                        {
                            "family": "pairs",
                            "decode": True,
                            "t": 0.5,
                            "jump_strawman": 0.4,
                            "variants": {
                                "temperature_1.0": {
                                    "epsint": 0.5,
                                    "excess": 0.4,
                                    "displacement_split": 0.3,
                                    "floor": 0.1,
                                }
                            },
                        }
                    ]
                },
            }
        ]
        tex = ci.latex_tables(payloads)
        self.assertIn("Distance to interpolant", tex)
        self.assertIn("0.300", tex)
        self.assertIn("No interpolation", tex)
        self.assertNotIn("closer than jump", tex)
        self.assertNotIn("Jump (reference)", tex)

    def test_word_set_key_ignores_order(self):
        self.assertEqual(
            ci.word_set_key(["Sugar", "trash"]),
            ci.word_set_key(["trash", "sugar"]),
        )

    def test_remap_weight_supports(self):
        weights = [{"family": "pairs", "support": [0, 1], "weights": [0.25, 0.75]}]
        remapped = ci.remap_weight_supports(
            weights, ["cat", "dog"], ["dog", "bird", "cat"]
        )
        self.assertEqual(remapped[0]["support"], [2, 0])

    def test_resolve_specs_from_ladder_keeps_ablations(self):
        args = argparse.Namespace(
            condition=None,
            from_ladder=True,
            coverage_only=False,
            ladder_json=os.path.join(
                _REPO_ROOT, "experiments", "semantle", "ladder_checkpoints.json"
            ),
        )
        specs = ci.resolve_specs(args)
        names = [row["name"] for row in specs]
        self.assertEqual(names[:4], ["canonical", "sdpo0", "novae", "joint"])
        self.assertIn("n8", names)
        self.assertEqual(next(r for r in specs if r["name"] == "n8")["coverage_only"], "1")
        self.assertNotIn("coverage_only", specs[0])

    def test_resolve_specs_coverage_only_ladder_skips_defaults(self):
        args = argparse.Namespace(
            condition=None,
            from_ladder=True,
            coverage_only=True,
            ladder_json=os.path.join(
                _REPO_ROOT, "experiments", "semantle", "ladder_checkpoints.json"
            ),
        )
        specs = ci.resolve_specs(args)
        names = [row["name"] for row in specs]
        self.assertTrue(all(name.startswith("n") for name in names))
        self.assertTrue(all(row.get("coverage_only") == "1" for row in specs))

    def test_dry_run_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            ci.main(
                [
                    "--dry-run",
                    "--no-wandb",
                    f"--condition=toy={os.path.join(tmp, 'missing')}",
                    f"--out-dir={tmp}",
                ]
            )
        with tempfile.TemporaryDirectory() as tmp:
            items = [{"id": 0, "word": "cat", "target": "cat"}]
            ckpt = os.path.join(tmp, "ckpt")
            os.makedirs(ckpt)
            with open(os.path.join(ckpt, "items.json"), "w", encoding="utf-8") as handle:
                json.dump(items, handle)
            ci.main(
                [
                    "--dry-run",
                    "--no-wandb",
                    f"--condition=toy={ckpt}",
                    f"--out-dir={tmp}",
                ]
            )


if __name__ == "__main__":
    unittest.main()
