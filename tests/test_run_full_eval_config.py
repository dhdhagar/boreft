"""Tests for full-eval config resolution from checkpoint metadata."""

from __future__ import annotations

import json
import tempfile
import unittest

from boreft.eval.run_full_eval import FullEvalConfig, _resolve_full_eval_config


def _write_checkpoint_config(
    output_dir: str,
    *,
    training: dict | None = None,
    intervention: dict | None = None,
) -> None:
    with open(f"{output_dir}/training_config.json", "w", encoding="utf-8") as f:
        json.dump(training or {}, f)
    with open(f"{output_dir}/intervention_config.json", "w", encoding="utf-8") as f:
        json.dump(intervention or {}, f)


class ResolveFullEvalConfigTests(unittest.TestCase):
    def test_fills_model_shape_from_saved_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_checkpoint_config(
                tmp,
                training={
                    "model_name": "meta-llama/Llama-3.2-1B-Instruct",
                    "low_rank_dim": 8,
                    "layer": 0,
                    "position": "f1",
                    "train_top_k": 4000,
                    "full_eval_n_samples": 512,
                    "cache_dir": None,
                    "semantle_dir": "data/semantle/train",
                },
            )
            resolved = _resolve_full_eval_config(FullEvalConfig(output_dir=tmp))
            self.assertEqual(resolved.model_name, "meta-llama/Llama-3.2-1B-Instruct")
            self.assertEqual(resolved.low_rank_dim, 8)
            self.assertEqual(resolved.layer, 0)
            self.assertEqual(resolved.position, "f1")
            self.assertEqual(resolved.top_k, 4000)
            self.assertEqual(resolved.full_eval_n_samples, 512)
            self.assertEqual(resolved.cache_dir, None)
            self.assertEqual(resolved.semantle_dir, "data/semantle/train")

    def test_cli_override_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_checkpoint_config(
                tmp,
                training={"low_rank_dim": 8, "layer": 0},
            )
            resolved = _resolve_full_eval_config(
                FullEvalConfig(output_dir=tmp, low_rank_dim=16, layer=3)
            )
            self.assertEqual(resolved.low_rank_dim, 16)
            self.assertEqual(resolved.layer, 3)

    def test_fallback_when_checkpoint_missing_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_checkpoint_config(tmp, training={})
            resolved = _resolve_full_eval_config(FullEvalConfig(output_dir=tmp))
            self.assertEqual(resolved.model_name, "meta-llama/Llama-3.2-1B")
            self.assertEqual(resolved.layer, 13)
            self.assertEqual(resolved.low_rank_dim, 64)
            self.assertEqual(resolved.position, "l1")

    def test_oracle_screen_defaults_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_checkpoint_config(tmp, training={})
            resolved = _resolve_full_eval_config(FullEvalConfig(output_dir=tmp))
            self.assertTrue(resolved.oracle_screen)
            self.assertEqual(resolved.oracle_screen_n_sobol, 2048)

        with tempfile.TemporaryDirectory() as tmp:
            _write_checkpoint_config(
                tmp,
                training={
                    "full_eval_oracle_screen": False,
                    "full_eval_oracle_screen_n_sobol": 128,
                },
            )
            resolved = _resolve_full_eval_config(FullEvalConfig(output_dir=tmp))
            self.assertFalse(resolved.oracle_screen)
            self.assertEqual(resolved.oracle_screen_n_sobol, 128)

        with tempfile.TemporaryDirectory() as tmp:
            _write_checkpoint_config(
                tmp,
                training={"full_eval_oracle_screen": True},
            )
            resolved = _resolve_full_eval_config(
                FullEvalConfig(output_dir=tmp, oracle_screen=False)
            )
            self.assertFalse(resolved.oracle_screen)

    def test_low_rank_dim_from_intervention_config_only(self):
        """Embed-sim sub-checkpoints copy intervention_config.json but not training_config.json."""
        with tempfile.TemporaryDirectory() as tmp:
            with open(
                f"{tmp}/intervention_config.json", "w", encoding="utf-8"
            ) as f:
                json.dump(
                    {
                        "low_rank_dim": 8,
                        "layer": 0,
                        "position": "marker",
                        "model_name": "meta-llama/Llama-3.2-1B-Instruct",
                    },
                    f,
                )
            resolved = _resolve_full_eval_config(FullEvalConfig(output_dir=tmp))
            self.assertEqual(resolved.low_rank_dim, 8)
            self.assertEqual(resolved.layer, 0)
            self.assertEqual(resolved.position, "marker")


if __name__ == "__main__":
    unittest.main()
