"""Tests for experiments/semantle/compare_spaces.py."""

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
    path = os.path.join(_REPO_ROOT, "experiments", "semantle", "compare_spaces.py")
    spec = importlib.util.spec_from_file_location("compare_spaces", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cs = _load_script()


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


class CompareSpacesTests(unittest.TestCase):
    def test_paper_space_order_omits_joint_ablation(self):
        self.assertEqual(
            list(cs.PAPER_SPACE_ORDER),
            [
                "boreft",
                "boreft_sdpo0",
                "boreft_novae",
                "boreft_ce0",
                "boreft_noenc",
            ],
        )
        self.assertNotIn("boreft_sdpo0_novae", cs.PAPER_SPACE_ORDER)

    def test_load_sdpo0_and_novae_trees(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            search = root / "search"
            sweep = root / "sweep"
            _write_run(search, "boreft_sdpo0", "train", "apple", 1)
            _write_run(search, "boreft_novae", "test", "berry", 1)
            _write_run(search, "boreft_sdpo0_novae", "train", "cherry", 1)
            _write_run(search, "boreft_ce0", "test", "date", 1)
            _write_run(search, "boreft_noenc", "train", "fig", 1)
            runs, dirs, override = cs.load_space_runs(
                search_dir=search,
                sweep_dir=sweep,
                canonical_slug="s1_t0_ard_d64",
                train_override="",
                sdpo0_dir=None,
                novae_dir=None,
            )
            methods = {run["method"] for run in runs}
            self.assertEqual(
                methods,
                {
                    "boreft_sdpo0",
                    "boreft_novae",
                    "boreft_sdpo0_novae",
                    "boreft_ce0",
                    "boreft_noenc",
                },
            )
            self.assertIsNone(override)
            self.assertIn("boreft_sdpo0", dirs)
            self.assertIn("boreft_novae", dirs)
            self.assertIn("boreft_sdpo0_novae", dirs)
            self.assertIn("boreft_ce0", dirs)
            self.assertIn("boreft_noenc", dirs)

    def test_empty_method_dir_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            search = root / "search"
            (search / "boreft_sdpo0").mkdir(parents=True)
            _write_run(search, "boreft_novae", "test", "berry", 1)
            runs, dirs, _ = cs.load_space_runs(
                search_dir=search,
                sweep_dir=root / "sweep",
                canonical_slug="s1_t0_ard_d64",
                train_override="",
                sdpo0_dir=None,
                novae_dir=None,
            )
            methods = {run["method"] for run in runs}
            self.assertEqual(methods, {"boreft_novae"})
            self.assertNotIn("boreft_sdpo0", dirs)

    def test_remap_method(self):
        rows = cs.remap_method([{"method": "boreft", "target": "x"}], "boreft")
        self.assertEqual(rows[0]["method"], "boreft")
        self.assertEqual(rows[0]["target"], "x")
        self.assertEqual(cs.ar.label("boreft"), "BOReFT")
        self.assertEqual(cs.ar.label("canonical"), "BOReFT")
        self.assertEqual(cs.ar.label("boreft_sdpo0"), "w/o self-distillation")
        self.assertEqual(cs.ar.label("boreft_novae"), "w/o variational training")
        self.assertEqual(
            cs.ar.label("boreft_sdpo0_novae"),
            "w/o both",
        )
        self.assertEqual(cs.ar.label("boreft_ce0"), "w/o reconstruction")
        self.assertEqual(cs.ar.label("boreft_ce0_t1"), "w/o reconstruction (T=1)")
        self.assertEqual(cs.ar.label("boreft_noenc"), "w/o shared encoder")
        self.assertEqual(cs.ar.label("random_sampling"), "Random (base)")
        self.assertEqual(
            cs.ar.label("random_sampling_lora_e9"), "Random (post-SFT)"
        )

    def test_parse_wandb_display_name(self):
        self.assertEqual(
            cs.parse_wandb_display_name("boreft_novae_test-pudding_seed3"),
            ("test", "pudding", 3),
        )
        self.assertEqual(
            cs.parse_wandb_display_name("s1_t0_ard_d64_rerun_train-wizard_seed1"),
            ("train", "wizard", 1),
        )
        self.assertIsNone(cs.parse_wandb_display_name("s1_t0_ard_d64_ucb_train-wizard_seed1"))

    def test_random_post_sft_wandb_name_and_skip(self):
        self.assertEqual(
            cs.ar.parse_wandb_fallback_name(
                "random_sampling_lora_e9",
                "random_sampling_lora_e9_train-carnivore_seed1",
            ),
            ("train", "carnivore", 1),
        )
        self.assertTrue(
            cs.ar.skip_search_dir("random_sampling_lora_e1")
        )
        self.assertTrue(
            cs.ar.skip_search_dir("random_sampling_lora_e9")
        )

    def test_curves_from_history_pads_budget(self):
        rows = [
            {"search/verifications": 1, "search/best_so_far": 0.5, "search/found": 0, "search/n_unique": 1},
            {"search/verifications": 3, "search/best_so_far": 0.9, "search/found": 1, "search/n_unique": 2},
        ]
        stats = cs.curves_from_history(rows, budget=5)
        self.assertEqual(list(stats["best_curve"]), [0.5, 0.5, 0.9, 0.9, 0.9])
        self.assertEqual(list(stats["found_curve"]), [0.0, 0.0, 1.0, 1.0, 1.0])
        self.assertTrue(stats["found"])
        self.assertEqual(stats["found_at"], 3)

    def test_curves_from_history_uses_step_when_verifications_missing(self):
        rows = [
            {"_step": 1, "search/best_so_far": 0.4, "search/found": 0, "search/n_unique": 1},
            {"_step": 2, "search/best_so_far": 1.0, "search/found": 1, "search/n_unique": 2},
        ]
        stats = cs.curves_from_history(rows, budget=4)
        self.assertEqual(list(stats["best_curve"]), [0.4, 1.0, 1.0, 1.0])
        self.assertEqual(stats["found_at"], 2)


if __name__ == "__main__":
    unittest.main()
