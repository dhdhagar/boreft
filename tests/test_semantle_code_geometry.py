"""Tests for scripts/analyze_semantle_code_geometry.py (no GPU, no checkpoints)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script():
    path = os.path.join(_REPO_ROOT, "scripts", "analyze_semantle_code_geometry.py")
    spec = importlib.util.spec_from_file_location(
        "analyze_semantle_code_geometry", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cg = _load_script()


class CodeGeometryMathTests(unittest.TestCase):
    def test_semantic_distances_are_one_minus_cosine(self):
        phi = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        d = cg.pairwise_semantic_distances(phi)
        self.assertAlmostEqual(d[0, 1], 1.0)
        self.assertAlmostEqual(d[0, 2], 0.0)
        np.testing.assert_allclose(np.diag(d), 0.0)

    def test_code_distances_match_l2(self):
        mu = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float64)
        d = cg.pairwise_code_distances(mu)
        self.assertAlmostEqual(d[0, 1], 5.0)
        self.assertAlmostEqual(d[1, 0], 5.0)
        self.assertAlmostEqual(d[0, 0], 0.0)

    def test_spearman_of_matching_distance_vectors_is_one(self):
        mu = np.arange(6, dtype=np.float64).reshape(6, 1)
        d = cg.pairwise_code_distances(mu)
        from scipy.stats import spearmanr

        vec = cg.upper_tri(d)
        rho = float(spearmanr(vec, vec).statistic)
        self.assertAlmostEqual(rho, 1.0)
        geom = cg.spearman_geometry(np.hstack([mu, mu]), mu)
        self.assertTrue(np.isfinite(geom["spearman_rho"]))
        self.assertEqual(geom["n_pairs"], 15)

    def test_rank_one_spread_mu_has_unit_erank(self):
        mu = np.zeros((10, 64), dtype=np.float64)
        mu[:, 0] = np.linspace(0.0, 1.0, 10)
        util = cg.utilization_from_mu(mu)
        self.assertAlmostEqual(util["erank"], 1.0, places=5)
        self.assertAlmostEqual(util["utilization"], 1.0 / 9.0)
        self.assertEqual(util["utilization_denom"], 9)

    def test_orthonormal_mu_uses_available_rank(self):
        q, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(64, 64)))
        util = cg.utilization_from_mu(q)
        self.assertGreater(util["erank"], 50.0)
        self.assertGreater(util["utilization"], 0.8)
        self.assertEqual(util["rank"], 64)
        self.assertEqual(util["utilization_denom"], 63)

    def test_parse_condition_fills_known_wandb_id(self):
        parsed = cg.parse_condition("noenc=outputs/1788894671")
        self.assertEqual(parsed["name"], "noenc")
        self.assertEqual(parsed["wandb_id"], "6yv75wzj")
        self.assertEqual(parsed["output_dir"], "outputs/1788894671")

    def test_parse_condition_unknown_name_has_empty_wandb_id(self):
        parsed = cg.parse_condition("other=/tmp/run")
        self.assertEqual(parsed["wandb_id"], "")

    def test_condition_for_json_drops_scatter(self):
        row = {
            "name": "canonical",
            "spearman_rho": 0.5,
            "scatter": {"d_sem": [0.1], "d_code": [1.0], "n": 1},
        }
        out = cg.condition_for_json(row)
        self.assertNotIn("scatter", out)
        self.assertEqual(out["scatter_n"], 1)
        self.assertIn("scatter", row)

    def test_subsample_pairs_is_seed_stable(self):
        d = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
        a = cg.subsample_pairs(d, d, 2, 0)
        b = cg.subsample_pairs(d, d, 2, 0)
        self.assertEqual(a, b)
        self.assertEqual(a["n"], 2)


class CodeGeometryWandbTests(unittest.TestCase):
    def test_dry_run_discovers_code_geometry(self):
        from boreft.search_wandb import _named_summary, log_analysis_dir, named_analysis_reports

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data" / "semantle" / "analysis"
            root.mkdir(parents=True)
            payload = {
                "task": "semantle",
                "analysis": "code_geometry",
                "conditions": [
                    {
                        "name": "canonical",
                        "utilization": 0.4,
                        "spearman_rho": 0.55,
                        "erank": 25.0,
                    },
                    {
                        "name": "noenc",
                        "utilization": 0.2,
                        "spearman_rho": 0.1,
                        "erank": 12.0,
                    },
                ],
            }
            (root / "code_geometry.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            (root / "code_geometry_spectrum.png").write_bytes(b"png")
            (root / "code_geometry_scatter.png").write_bytes(b"png")
            with patch("wandb.init") as init:
                results = log_analysis_dir(root, project="boreft", dry_run=True)
            init.assert_not_called()
            names = {row["name"] for row in results}
            self.assertIn("semantle-code-geometry", names)
            self.assertEqual(len(named_analysis_reports(root)), 1)
            geo = next(row for row in results if "code-geometry" in row["name"])
            self.assertEqual(geo["n_images"], 2)
            summary = _named_summary(payload)
            self.assertAlmostEqual(summary["canonical/utilization"], 0.4)
            self.assertAlmostEqual(summary["noenc/spearman_rho"], 0.1)


if __name__ == "__main__":
    unittest.main()
