"""Tests for scripts/analyze_semantle_local_erank.py (no GPU, no checkpoints)."""

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
    path = os.path.join(_REPO_ROOT, "scripts", "analyze_semantle_local_erank.py")
    spec = importlib.util.spec_from_file_location(
        "analyze_semantle_local_erank", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


er = _load_script()


class LocalErankMathTests(unittest.TestCase):
    def test_identical_rows_are_rank_one(self):
        row = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        emb = np.stack([row, row, row])
        self.assertEqual(er.local_effective_rank(emb), 1.0)

    def test_orthonormal_rows_near_full_rank(self):
        q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(8, 8)))
        rank = er.local_effective_rank(q)
        self.assertGreater(rank, 6.5)
        self.assertLessEqual(rank, 8.0)

    def test_normalized_erank_uses_min_n_minus_one(self):
        self.assertAlmostEqual(er.normalized_erank(4.0, n=5, dim=64), 1.0)

    def test_summarize_eranks(self):
        stats = er.summarize_eranks([1.0, 3.0, 5.0])
        self.assertEqual(stats["n"], 3)
        self.assertEqual(stats["mean"], 3.0)
        self.assertEqual(stats["median"], 3.0)

    def test_posterior_codes_without_variance_are_unit_gaussian_around_mu(self):
        mu = np.array([0.5, -1.0], dtype=np.float64)
        rng = np.random.default_rng(0)
        codes = er.posterior_codes(mu, None, 4000, rng)
        self.assertEqual(codes.shape, (4000, 2))
        np.testing.assert_allclose(codes.mean(axis=0), mu, atol=0.08)
        np.testing.assert_allclose(codes.std(axis=0, ddof=1), [1.0, 1.0], atol=0.08)

    def test_posterior_codes_spread_with_logvar(self):
        mu = np.zeros(3, dtype=np.float64)
        logvar = np.zeros(3, dtype=np.float64)
        rng = np.random.default_rng(1)
        codes = er.posterior_codes(mu, logvar, 200, rng)
        self.assertGreater(float(codes.std()), 0.5)

    def test_sample_items_is_seed_stable(self):
        items = [{"id": i, "word": f"w{i}"} for i in range(20)]
        a = [er.item_word(x) for x in er.sample_items(items, 5, 42)]
        b = [er.item_word(x) for x in er.sample_items(items, 5, 42)]
        self.assertEqual(a, b)
        self.assertEqual(len(a), 5)

    def test_parse_condition(self):
        parsed = er.parse_condition("sdpo0=outputs/1788622157")
        self.assertEqual(parsed["name"], "sdpo0")
        self.assertEqual(parsed["wandb_id"], "yt8pnmyh")
        self.assertEqual(parsed["output_dir"], "outputs/1788622157")
        joint = er.parse_condition("sdpo0_novae=outputs/1789021668")
        self.assertEqual(joint["wandb_id"], "gsuy3cnv")
        ce0 = er.parse_condition("ce0=outputs/1789136583")
        self.assertEqual(ce0["wandb_id"], "bh4ceow2")

    def test_item_word_id_requires_training_id(self):
        self.assertEqual(er.item_word_id({"id": 7, "word": "cat"}), 7)
        with self.assertRaises(ValueError):
            er.item_word_id({"word": "cat"})

    def test_saved_float_keeps_zero(self):
        self.assertEqual(er._saved_float({"sdpo_sample_temperature": 0.0}, "sdpo_sample_temperature", 1.0), 0.0)
        self.assertEqual(er._saved_float({}, "sdpo_sample_temperature", 1.0), 1.0)

    def test_token_ids_flattens_batched(self):
        self.assertEqual(er._token_ids([[1, 2, 3]]), [1, 2, 3])
        self.assertEqual(er._token_ids(np.array([4, 5])), [4, 5])

    def test_domain_replicate_eranks_are_per_sample_slot(self):
        rng = np.random.default_rng(0)
        points = rng.normal(size=(6, 4))
        # Two replicates: first is the 6 points, second is the same points rotated.
        first = points
        second = np.roll(points, 1, axis=0)
        stacked = np.empty((6, 2, 4), dtype=np.float64)
        stacked[:, 0, :] = first
        stacked[:, 1, :] = second
        flat = stacked.reshape(12, 4)
        ranks = er.domain_replicate_eranks(flat, n_points=6, n_samples=2)
        self.assertEqual(ranks.shape, (2,))
        self.assertAlmostEqual(ranks[0], er.local_effective_rank(first))
        self.assertAlmostEqual(ranks[1], er.local_effective_rank(second))

    def test_domain_erank_stats_use_replicate_mean_and_std(self):
        mean, std = er.domain_erank_stats(
            {"erank": 99.0, "erank_replicates": [2.0, 4.0, 6.0]}
        )
        self.assertAlmostEqual(mean, 4.0)
        self.assertAlmostEqual(std, 2.0)

    def test_parse_args_cache_dir_default(self):
        args = er.parse_args(["--no-wandb"])
        self.assertEqual(args.cache_dir, er.DEFAULT_CACHE_DIR)
        self.assertEqual(args.n_sobol, er.DEFAULT_N_SOBOL)
        self.assertEqual(args.n_sobol_samples, 8)
        self.assertEqual(args.n_targets, 1000)
        self.assertEqual(args.n_samples, 8)
        self.assertFalse(args.skip_local)
        self.assertFalse(args.skip_sobol)

    def test_set_order_includes_greedy_and_t1_posterior(self):
        self.assertEqual(
            er.SET_ORDER, ("teacher", "mu", "posterior", "posterior_t1")
        )
        self.assertEqual(er.SET_LABELS["posterior"], "N(μ, σ²) T=0")
        self.assertEqual(er.SET_LABELS["posterior_t1"], "N(μ, σ²) T=1")
        self.assertNotIn("\n", er.SET_LABELS["posterior_t1"])
        self.assertEqual(
            [label for _, label in er.LOCAL_BREADTH_GROUPS],
            ["Learned Target", "Learned Neighborhood"],
        )

    def test_condition_labels(self):
        self.assertEqual(er.CONDITION_LABELS["canonical"], "BOReFT")
        self.assertEqual(er.CONDITION_LABELS["sdpo0"], "w/o self-distillation")
        self.assertEqual(er.CONDITION_LABELS["novae"], "w/o variational training")
        self.assertEqual(er.CONDITION_LABELS["ce0"], "w/o reconstruction")
        self.assertEqual(er.condition_label("canonical"), "BOReFT")
        self.assertEqual(er.condition_label("Canonical"), "BOReFT")

    def test_teacher_alignment_matches_centroid_cosine(self):
        teacher = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        student = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        self.assertAlmostEqual(er.teacher_alignment(teacher, teacher), 1.0)
        self.assertAlmostEqual(er.teacher_alignment(student, teacher), 0.5)

    def test_orthogonal_student_has_zero_alignment(self):
        teacher = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        student = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        self.assertAlmostEqual(er.teacher_alignment(student, teacher), 0.0)

    def test_local_figure_omits_joint_ablation(self):
        rows = [
            {"name": "canonical"},
            {"name": "sdpo0"},
            {"name": "novae"},
            {"name": "sdpo0_novae"},
        ]
        keep = er.conditions_for_local_figure(rows)
        self.assertEqual(
            [row["name"] for row in keep],
            ["canonical", "sdpo0", "novae"],
        )


class LocalErankPlotTests(unittest.TestCase):
    def test_plot_writes_png(self):
        conditions = [
            {
                "name": "canonical",
                "per_target": [
                    {
                        "teacher": {"erank": 4.0, "teacher_align": 0.9},
                        "mu": {"erank": 3.0, "teacher_align": 0.7},
                        "posterior": {"erank": 2.5, "teacher_align": 0.75},
                        "posterior_t1": {"erank": 3.4, "teacher_align": 0.8},
                    },
                    {
                        "teacher": {"erank": 5.0, "teacher_align": 0.85},
                        "mu": {"erank": 2.0, "teacher_align": 0.4},
                        "posterior": {"erank": 1.8, "teacher_align": 0.5},
                        "posterior_t1": {"erank": 2.8, "teacher_align": 0.55},
                    },
                ],
            },
            {
                "name": "sdpo0",
                "per_target": [
                    {
                        "teacher": {"erank": 4.1, "teacher_align": 0.88},
                        "mu": {"erank": 1.2, "teacher_align": 0.2},
                        "posterior": {"erank": 1.1, "teacher_align": 0.25},
                        "posterior_t1": {"erank": 1.8, "teacher_align": 0.3},
                    }
                ],
            },
            {
                "name": "novae",
                "per_target": [
                    {
                        "teacher": {"erank": 9.0, "teacher_align": 0.1},
                        "mu": {"erank": 9.0, "teacher_align": 0.1},
                        "posterior": {"erank": 1.0, "teacher_align": 0.1},
                        "posterior_t1": {"erank": 9.0, "teacher_align": 0.1},
                    }
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "local_semantic_erank.png")
            er.plot_local_erank(conditions, path, n_targets=2, n_samples=32)
            self.assertGreater(os.path.getsize(path), 0)
            self.assertGreater(
                os.path.getsize(os.path.splitext(path)[0] + ".pdf"), 0
            )

    def test_plot_skips_empty_series(self):
        conditions = [
            {"name": "canonical", "per_target": []},
            {
                "name": "sdpo0",
                "per_target": [
                    {
                        "teacher": {"erank": 2.0},
                        "mu": {"erank": 1.0},
                        "posterior": {"erank": 1.0},
                        "posterior_t1": {"erank": 1.5},
                    }
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "local_semantic_erank.png")
            er.plot_local_erank(conditions, path, n_targets=1, n_samples=4)
            self.assertGreater(os.path.getsize(path), 0)

    def test_domain_plot_writes_png(self):
        conditions = [
            {
                "name": "canonical",
                "erank": 12.0,
                "erank_std": 0.8,
                "erank_replicates": [11.2, 12.0, 12.8],
                "n_unique": 80,
                "n": 200,
            },
            {
                "name": "sdpo0",
                "erank": 18.0,
                "erank_std": 1.1,
                "erank_replicates": [16.9, 18.0, 19.1],
                "n_unique": 120,
                "n": 200,
            },
            {
                "name": "novae",
                "erank": 3.0,
                "erank_std": 0.2,
                "erank_replicates": [2.8, 3.0, 3.2],
                "n_unique": 10,
                "n": 200,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "domain_semantic_erank.png")
            er.plot_domain_erank(
                conditions, path, n_sobol_points=200, n_samples=8
            )
            self.assertGreater(os.path.getsize(path), 0)

    def test_local_breadth_pdf_uses_canonical_teacher_only(self):
        conditions = [
            {
                "name": "canonical",
                "per_target": [
                    {
                        "teacher": {"erank": 4.0},
                        "mu": {"erank": 3.0},
                        "posterior_t1": {"erank": 3.4},
                    },
                    {
                        "teacher": {"erank": 5.0},
                        "mu": {"erank": 2.0},
                        "posterior_t1": {"erank": 2.8},
                    },
                ],
            },
            {
                "name": "sdpo0",
                "per_target": [
                    {
                        "teacher": {"erank": 99.0},
                        "mu": {"erank": 1.2},
                        "posterior_t1": {"erank": 1.8},
                    }
                ],
            },
            {
                "name": "novae",
                "per_target": [
                    {
                        "teacher": {"erank": 80.0},
                        "mu": {"erank": 9.0},
                        "posterior_t1": {"erank": 8.0},
                    }
                ],
            },
        ]
        self.assertEqual(
            er.teacher_eranks_for_training_box(conditions), [4.0, 5.0]
        )
        with tempfile.TemporaryDirectory() as tmp:
            pdf = os.path.join(tmp, "local_breadth.pdf")
            er.plot_local_breadth(conditions, pdf)
            self.assertGreater(os.path.getsize(pdf), 0)
            self.assertGreater(
                os.path.getsize(os.path.join(tmp, "local_breadth.png")), 0
            )


class LocalErankFromJsonTests(unittest.TestCase):
    def test_from_json_writes_plot(self):
        payload = {
            "task": "semantle",
            "n_targets": 1,
            "n_samples": 4,
            "sets": list(er.SET_ORDER),
            "conditions": [
                {
                    "name": "canonical",
                    "summary": {
                        set_name: {
                            "erank": {"mean": 2.0, "median": 2.0, "std": 0.0},
                            "teacher_align": {"mean": 0.5, "median": 0.5, "std": 0.0},
                        }
                        for set_name in er.SET_ORDER
                    },
                    "per_target": [
                        {
                            set_name: {"erank": 2.0, "teacher_align": 0.5}
                            for set_name in er.SET_ORDER
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "local_semantic_erank.json")
            out = os.path.join(tmp, "out")
            with open(src, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            rc = er.main(["--from-json", src, "--out-dir", out, "--no-wandb"])
            self.assertEqual(rc, 0)
            self.assertGreater(
                os.path.getsize(os.path.join(out, "local_semantic_erank.png")), 0
            )
            self.assertGreater(
                os.path.getsize(os.path.join(out, "local_semantic_erank.pdf")), 0
            )
            self.assertGreater(
                os.path.getsize(os.path.join(out, "local_breadth.pdf")), 0
            )

    def test_from_json_writes_domain_plot(self):
        payload = {
            "task": "semantle",
            "n_sobol_points": 200,
            "n_samples": 8,
            "conditions": [
                {
                    "name": "canonical",
                    "erank": 11.0,
                    "erank_std": 0.4,
                    "erank_replicates": [10.6, 11.0, 11.4],
                    "n_unique": 40,
                    "n": 200,
                },
                {
                    "name": "novae",
                    "erank": 2.0,
                    "erank_std": 0.1,
                    "erank_replicates": [1.9, 2.0, 2.1],
                    "n_unique": 8,
                    "n": 200,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "domain_semantic_erank.json")
            out = os.path.join(tmp, "out")
            with open(src, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            rc = er.main(["--from-json", src, "--out-dir", out, "--no-wandb"])
            self.assertEqual(rc, 0)
            self.assertGreater(
                os.path.getsize(os.path.join(out, "domain_semantic_erank.png")), 0
            )
            self.assertGreater(
                os.path.getsize(os.path.join(out, "domain_semantic_erank.pdf")), 0
            )


class LocalErankWandbTests(unittest.TestCase):
    def test_dry_run_discovers_local_erank(self):
        from boreft.search_wandb import (
            _local_erank_summary,
            log_analysis_dir,
            local_erank_jsons,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data" / "semantle" / "analysis"
            root.mkdir(parents=True)
            (root / "local_semantic_erank.json").write_text(
                json.dumps(
                    {
                        "task": "semantle",
                        "n_targets": 8,
                        "conditions": [
                            {
                                "name": "canonical",
                                "summary": {
                                    "teacher": {
                                        "erank": {"mean": 3.2},
                                        "teacher_align": {"mean": 0.81},
                                    },
                                    "mu": {"mean": 2.1},
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (root / "local_semantic_erank.png").write_bytes(b"png")
            (root / "domain_semantic_erank.json").write_text(
                json.dumps(
                    {
                        "task": "semantle",
                        "n_sobol_points": 200,
                        "conditions": [
                            {"name": "canonical", "erank": 12.5, "n_unique": 90}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (root / "domain_semantic_erank.png").write_bytes(b"png")
            with patch("wandb.init") as init:
                results = log_analysis_dir(root, project="boreft", dry_run=True)
            init.assert_not_called()
            names = {row["name"] for row in results}
            self.assertIn("semantle-local-semantic-erank", names)
            self.assertEqual(len(local_erank_jsons(root)), 2)
            erank = next(row for row in results if "local-semantic-erank" in row["name"])
            self.assertEqual(erank["n_images"], 2)
            self.assertEqual(erank["n_json"], 2)

            summary = {}
            for path in local_erank_jsons(root):
                summary.update(_local_erank_summary(json.loads(path.read_text())))
            self.assertAlmostEqual(summary["canonical/teacher/mean_erank"], 3.2)
            self.assertAlmostEqual(summary["canonical/teacher/mean_teacher_align"], 0.81)
            self.assertAlmostEqual(summary["canonical/domain/erank"], 12.5)


if __name__ == "__main__":
    unittest.main()
