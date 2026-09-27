from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "compare_search_domain_volumes.py"
    spec = importlib.util.spec_from_file_location("compare_search_domain_volumes", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CompareSearchDomainVolumesScriptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_script()

    def test_compare_domains_nested_boxes_and_ellipsoid(self):
        mu = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float32)
        std = np.array([[0.5, 0.0], [0.0, 1.0]], dtype=np.float32)
        payload = self.mod.compare_domains(
            mu, std, aabb_std_k=(0.0, 0.1, 0.5), bounds_padding=0.0
        )
        names = [row["name"] for row in payload["domains"]]
        self.assertEqual(
            names,
            [
                "AABB k=0",
                "AABB k=0.1",
                "AABB k=0.5",
                "covering ellipsoid",
                "ellipsoid enclosing AABB",
            ],
        )
        logs = [row["log_volume"] for row in payload["domains"]]
        self.assertLess(logs[0], logs[1])
        self.assertLess(logs[1], logs[2])
        self.assertLess(logs[3], logs[4])
        self.assertEqual(payload["domains"][0]["volume_ratio_vs_k0"], 1.0)
        ell = payload["domains"][3]
        self.assertEqual(ell["n_means_outside"], 0)
        self.assertLessEqual(ell["max_mahalanobis"], 1.0 + 1e-6)
        self.assertAlmostEqual(math.exp(logs[0]), 2.0, places=6)

    def test_equivalent_cube_side_recovers_box_side(self):
        side = self.mod.equivalent_cube_side(64.0 * math.log(3.0), 64)
        self.assertAlmostEqual(side, 3.0)
