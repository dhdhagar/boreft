"""Unit tests for interpolation helpers."""

from __future__ import annotations

import unittest

import torch

from boreft.eval.interpolate import (
    _dominant_label_indices,
    _interp_subplot_grid,
    _sim_legend_label,
    interpolate_bias_torch,
    lerp_torch,
    normalize_interp_method,
    slerp_torch,
)
from boreft.text_display import plot_label


class TestInterpSubplotGrid(unittest.TestCase):
    def test_no_empty_panels_for_three_variants(self):
        nrows, ncols, figsize = _interp_subplot_grid(3)
        self.assertEqual((nrows, ncols), (1, 3))
        self.assertEqual(nrows * ncols, 3)

    def test_four_variants_use_two_by_two(self):
        nrows, ncols, _ = _interp_subplot_grid(4)
        self.assertEqual((nrows, ncols), (2, 2))

    def test_single_variant(self):
        nrows, ncols, _ = _interp_subplot_grid(1)
        self.assertEqual((nrows, ncols), (1, 1))


class TestInterpPlotLabels(unittest.TestCase):
    def test_plot_label_keeps_short_text(self):
        self.assertEqual(plot_label("benzene"), "benzene")

    def test_plot_label_truncates_smiles(self):
        smiles = (
            "C/C1=C\\[C@H]2[C@@H](O)[C@@H](C)C[C@]2(OC(=O)c2ccccc2)"
            "C(=O)/C(C)=C/[C@H]2[C@H](CC1)C2(C)C"
        )
        shown = plot_label(smiles)
        self.assertLess(len(shown), len(smiles))
        self.assertIn("...", shown)
        self.assertTrue(shown.startswith("C/C1=C"))
        self.assertTrue(shown.endswith("C2(C)C"))

    def test_legend_uses_truncated_endpoint(self):
        smiles = "[O]=[Mn](=[O])([O-])[O-]" + "C" * 80
        label = _sim_legend_label(smiles)
        self.assertTrue(label.startswith("sim(gen, "))
        self.assertNotIn("C" * 40, label)

    def test_dominant_indices_keep_all_when_sparse(self):
        texts = ["a"] * 5 + ["b"] * 5
        self.assertEqual(_dominant_label_indices(texts, max_labels=8), [0, 5])

    def test_dominant_indices_cap_and_keep_ends(self):
        texts = [f"mol{i}" for i in range(40)]
        idxs = _dominant_label_indices(texts, max_labels=8)
        self.assertEqual(len(idxs), 8)
        self.assertEqual(idxs[0], 0)
        self.assertEqual(idxs[-1], 39)


class TestBiasInterpolation(unittest.TestCase):
    def test_lerp_endpoints_and_midpoint(self):
        v0 = torch.tensor([1.0, 0.0])
        v1 = torch.tensor([0.0, 1.0])
        self.assertTrue(torch.allclose(lerp_torch(v0, v1, 0.0), v0))
        self.assertTrue(torch.allclose(lerp_torch(v0, v1, 1.0), v1))
        self.assertTrue(torch.allclose(lerp_torch(v0, v1, 0.5), torch.tensor([0.5, 0.5])))

    def test_slerp_matches_lerp_for_equal_norm_orthogonal_pair(self):
        v0 = torch.tensor([1.0, 0.0])
        v1 = torch.tensor([0.0, 1.0])
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            self.assertTrue(
                torch.allclose(
                    slerp_torch(v0, v1, t),
                    lerp_torch(v0, v1, t),
                    atol=1e-6,
                )
            )

    def test_interpolate_bias_torch_dispatches(self):
        v0 = torch.tensor([2.0, 0.0])
        v1 = torch.tensor([0.0, 2.0])
        t = 0.5
        self.assertTrue(
            torch.allclose(interpolate_bias_torch(v0, v1, t, "lerp"), lerp_torch(v0, v1, t))
        )
        self.assertTrue(
            torch.allclose(interpolate_bias_torch(v0, v1, t, "slerp"), slerp_torch(v0, v1, t))
        )

    def test_normalize_interp_method(self):
        self.assertEqual(normalize_interp_method(" LERP "), "lerp")
        with self.assertRaises(ValueError):
            normalize_interp_method("nlerp")


if __name__ == "__main__":
    unittest.main()
