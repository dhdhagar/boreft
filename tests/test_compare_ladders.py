"""Tests for experiments/semantle/compare_ladders.py."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script():
    path = os.path.join(_REPO_ROOT, "experiments", "semantle", "compare_ladders.py")
    spec = importlib.util.spec_from_file_location("compare_ladders", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cl = _load_script()


def _write_run(root: Path, method: str, split: str, target: str, seed: int) -> None:
    seed_dir = root / method / f"{split}-{target}" / f"seed_{seed}"
    seed_dir.mkdir(parents=True)
    rows = [
        {
            "source": "warmstart",
            "decoded": "near",
            "score": 0.6,
            "sample_count": 1,
            "sample_scores": [0.6],
            "decoded_samples": ["near"],
        },
        {
            "source": "acquisition",
            "decoded": target,
            "score": 1.0,
            "sample_count": 1,
            "sample_scores": [1.0],
            "decoded_samples": [target],
            "components": {"exact_match": True},
        },
    ]
    with (seed_dir / "observations.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class CompareLaddersTests(unittest.TestCase):
    def test_loads_n_and_rank_trees_and_canonical(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            search = root / "search"
            sweep = root / "sweep"
            _write_run(sweep, "s1_t0_ard_d64", "test", "pudding", 1)
            _write_run(sweep, "s1_t0_ard_d64_rerun", "train", "arms", 1)
            _write_run(search, "boreft_N256", "test", "bureau", 1)
            _write_run(search, "boreft_rank32", "train", "wizard", 1)
            runs, dirs, override = cl.load_ladder_runs(
                search_dir=search,
                sweep_dir=sweep,
                canonical_slug="s1_t0_ard_d64",
                train_override="s1_t0_ard_d64_rerun",
            )
            methods = {run["method"] for run in runs}
            self.assertEqual(methods, {"n256", "n3072", "r32", "r64"})
            self.assertEqual(override, "s1_t0_ard_d64_rerun")
            self.assertEqual(dirs["n256"], search / "boreft_N256")
            self.assertEqual(dirs["r32"], search / "boreft_rank32")
            self.assertTrue(
                any(run["method"] == "n3072" and run["split"] == "train" for run in runs)
            )
            self.assertTrue(
                any(run["method"] == "n3072" and run["split"] == "test" for run in runs)
            )

    def test_skip_prefixes_keep_space_ablations(self):
        self.assertTrue(cl.ar.skip_search_dir("boreft_N256"))
        self.assertTrue(cl.ar.skip_search_dir("boreft_rank32"))
        self.assertTrue(cl.ar.skip_search_dir("sdpo_ttt_lora_e9"))
        self.assertTrue(cl.ar.skip_search_dir("bopro_lora_e9"))
        self.assertFalse(cl.ar.skip_search_dir("bopro"))
        self.assertFalse(cl.ar.skip_search_dir("boreft_novae"))
        self.assertFalse(cl.ar.skip_search_dir("boreft_noenc"))
        self.assertEqual(cl.ar.label("n3072"), "N=3072 (main)")
        self.assertEqual(cl.ar.label("r64"), "rank=64 (main)")
        self.assertEqual(cl.ar.label("n16"), "N=16")

    def test_joint_search_series_follows_n_order(self):
        table = [
            {
                "method": "n1024",
                "all": {"mean_best": 0.8, "success_rate": 0.1},
            },
            {
                "method": "n1",
                "all": {"mean_best": 0.7, "success_rate": 0.0},
            },
            {
                "method": "n3072",
                "all": {"mean_best": 0.88, "success_rate": 0.47},
            },
        ]
        methods = cl.ordered_methods(table, cl.N_SIZES, cl.n_method)
        xs, cosine, exact = cl.joint_search_series(table, methods)
        self.assertEqual(methods, ["n1", "n1024", "n3072"])
        self.assertEqual(xs, [1, 1024, 3072])
        self.assertEqual(cosine, [0.7, 0.8, 0.88])
        self.assertEqual(exact, [0.0, 0.1, 0.47])

    def test_load_coverage_vs_n_reads_eval_medians(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "conditions": [
                            {
                                "name": "n4",
                                "coverage": {
                                    "eval_test": {
                                        "summary": {"vertices": {"median": 0.84}}
                                    }
                                },
                            },
                            {
                                "name": "novae",
                                "coverage": {
                                    "eval_test": {
                                        "summary": {"vertices": {"median": 0.5}}
                                    }
                                },
                            },
                            {
                                "name": "n3072",
                                "coverage": {
                                    "eval_test": {
                                        "summary": {"vertices": {"median": 0.588}}
                                    }
                                },
                            },
                        ]
                    }
                )
            )
            coverage = cl.load_coverage_vs_n(path)
            self.assertEqual(coverage, {4: 0.84, 3072: 0.588})
            xs, ys = cl.coverage_series([1, 4, 512, 3072], coverage)
            self.assertEqual(xs, [4, 3072])
            self.assertEqual(ys, [0.84, 0.588])

    def test_load_erank_results_keeps_ladder_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "erank.json"
            path.write_text(
                json.dumps(
                    {
                        "results": [
                            {"train_n_samples": 4000, "n_drawn": 4000, "effective_rank": 9.0},
                            {"train_n_samples": 4, "n_drawn": 4, "effective_rank": 3.0},
                            {"train_n_samples": 3072, "n_drawn": 3072, "effective_rank": 627.0},
                        ]
                    }
                )
            )
            rows = cl.load_erank_results(path)
            self.assertEqual([row["n_drawn"] for row in rows], [4, 3072])
            self.assertEqual([row["effective_rank"] for row in rows], [3.0, 627.0])

    def test_paper_plots_write_pdf_and_copy(self):
        table = [
            {
                "method": "n4",
                "all": {"mean_best": 0.72, "success_rate": 0.0},
            },
            {
                "method": "n3072",
                "all": {"mean_best": 0.88, "success_rate": 0.47},
            },
            {
                "method": "r16",
                "all": {"mean_best": 0.87, "success_rate": 0.33},
            },
            {
                "method": "r64",
                "all": {"mean_best": 0.88, "success_rate": 0.47},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            erank = root / "erank.json"
            erank.write_text(
                json.dumps(
                    {
                        "results": [
                            {"n_drawn": 4, "effective_rank": 3.0},
                            {"n_drawn": 3072, "effective_rank": 627.0},
                        ]
                    }
                )
            )
            coverage = root / "coverage.json"
            coverage.write_text(
                json.dumps(
                    {
                        "conditions": [
                            {
                                "name": "n4",
                                "coverage": {
                                    "eval_test": {
                                        "summary": {"vertices": {"median": 0.84}}
                                    }
                                },
                            },
                            {
                                "name": "n3072",
                                "coverage": {
                                    "eval_test": {
                                        "summary": {"vertices": {"median": 0.59}}
                                    }
                                },
                            },
                        ]
                    }
                )
            )
            plt = cl.ar._plt()
            written = cl.write_paper_ladder_plots(
                plt,
                table,
                root,
                erank_json=erank,
                n_only=False,
                rank_only=False,
                coverage_json=coverage,
            )
            self.assertEqual(set(written), {"n", "rank", "erank"})
            for path in written.values():
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)
                self.assertGreater(path.with_suffix(".png").stat().st_size, 0)
            paper = root / "paper"
            cl.copy_paper_ladders(
                written,
                n_dest=paper / "search_n_ladder.pdf",
                rank_dest=paper / "search_rank_ladder.pdf",
                erank_dest=paper / "train_erank.pdf",
            )
            self.assertGreater((paper / "search_n_ladder.pdf").stat().st_size, 0)
            self.assertGreater((paper / "search_n_ladder.png").stat().st_size, 0)
            self.assertGreater((paper / "train_erank.pdf").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
