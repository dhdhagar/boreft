"""Tests for PCA/silhouette cluster label resolution."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from boreft.eval.plot_cluster_pca import (
    assign_category_labels,
    build_cluster_style_map,
    resolve_pca_checkpoint_args,
    resolve_word_cluster_labels,
)
from boreft.eval.run_full_eval import _geometry_eval_enabled


def _write_training_config(output_dir: str, training: dict) -> None:
    with open(f"{output_dir}/training_config.json", "w", encoding="utf-8") as f:
        json.dump(training, f)


class ClusterLabelTests(unittest.TestCase):
    def test_assign_category_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            defs_path = f"{tmp}/definitions.jsonl"
            rows = [
                {"target": "laptop", "category_normalized": "computing"},
                {"target": "apple", "category_normalized": "food drink"},
            ]
            with open(defs_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            labels = assign_category_labels(["laptop", "apple", "missing"], defs_path)
            self.assertEqual(labels["laptop"], "computing")
            self.assertEqual(labels["apple"], "food drink")
            self.assertEqual(labels["missing"], "unknown")

    def test_resolve_word_cluster_labels_uses_categories_when_definition_embeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            defs_path = f"{tmp}/definitions.jsonl"
            with open(defs_path, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"target": "laptop", "category_normalized": "computing"}
                    )
                    + "\n"
                )
            _write_training_config(
                tmp,
                {
                    "use_definition_embeds": True,
                    "definitions_path": defs_path,
                },
            )
            labels, source = resolve_word_cluster_labels(["laptop"], tmp)
            self.assertEqual(source, "category")
            self.assertEqual(labels["laptop"], "computing")

    def test_geometry_eval_enabled_with_definition_embeds(self) -> None:
        # Definition categories enable the cluster-PCA plot regardless of
        # whether semantle_dir is unset (None) or empty ("").
        with tempfile.TemporaryDirectory() as tmp:
            _write_training_config(tmp, {"use_definition_embeds": True})
            self.assertTrue(_geometry_eval_enabled(tmp, None))
            self.assertTrue(_geometry_eval_enabled(tmp, ""))

    def test_geometry_eval_enabled_requires_semantle_dir_without_categories(self) -> None:
        # Without definition categories, geometry eval needs a semantle_dir.
        with tempfile.TemporaryDirectory() as tmp:
            _write_training_config(tmp, {"use_definition_embeds": False})
            self.assertFalse(_geometry_eval_enabled(tmp, None))
            self.assertFalse(_geometry_eval_enabled(tmp, ""))
            self.assertTrue(_geometry_eval_enabled(tmp, "data/semantle/train"))

    def test_build_cluster_style_map_assigns_unique_styles(self) -> None:
        clusters = [f"category-{i:02d}" for i in range(28)]
        style_map = build_cluster_style_map(clusters)
        self.assertEqual(len(style_map), 28)
        signatures = {
            (style["color"], style["marker"], style["edgecolor"])
            for style in style_map.values()
        }
        self.assertEqual(len(signatures), 28)

    def test_build_cluster_style_map_meta_clusters(self) -> None:
        style_map = build_cluster_style_map(["animals", "unknown", "multi-cluster"])
        self.assertEqual(style_map["unknown"]["color"], "#d9d9d9")
        self.assertEqual(style_map["multi-cluster"]["color"], "#808080")

    def test_resolve_pca_checkpoint_args_from_training_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_training_config(
                tmp,
                {
                    "model_name": "meta-llama/Llama-3.2-1B-Instruct",
                    "layer": 15,
                    "low_rank_dim": 8,
                    "train_top_k": 50,
                    "cache_dir": "/tmp/hf-cache",
                    "semantle_dir": "data/semantle/train",
                },
            )
            resolved = resolve_pca_checkpoint_args(tmp)
            self.assertEqual(
                resolved["model_name"], "meta-llama/Llama-3.2-1B-Instruct"
            )
            self.assertEqual(resolved["layer"], 15)
            self.assertEqual(resolved["low_rank_dim"], 8)
            self.assertEqual(resolved["top_k"], 50)
            self.assertEqual(resolved["cache_dir"], "/tmp/hf-cache")
            self.assertEqual(resolved["semantle_dir"], "data/semantle/train")
            self.assertEqual(
                resolved["save_path"],
                os.path.join(tmp, "eval", "cluster_pca.png"),
            )

    def test_resolve_pca_checkpoint_args_cli_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_training_config(tmp, {"layer": 15, "low_rank_dim": 8})
            resolved = resolve_pca_checkpoint_args(
                tmp, layer=3, low_rank_dim=16, save_path=f"{tmp}/custom.png"
            )
            self.assertEqual(resolved["layer"], 3)
            self.assertEqual(resolved["low_rank_dim"], 16)
            self.assertEqual(resolved["save_path"], f"{tmp}/custom.png")


if __name__ == "__main__":
    unittest.main()
