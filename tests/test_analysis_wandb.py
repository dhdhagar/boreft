from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from boreft.search_wandb import (
    is_encoder_cluster_report,
    log_analysis_dir,
    maybe_log_encoder_cluster,
    train_embed_rank_jsons,
)


class AnalysisDiscoveryTests(unittest.TestCase):
    def test_skips_encoder_cache_sidecars(self):
        self.assertTrue(is_encoder_cluster_report(Path("encoder_cluster.json")))
        self.assertTrue(
            is_encoder_cluster_report(Path("encoder_cluster_maxlen512.json"))
        )
        self.assertFalse(
            is_encoder_cluster_report(Path("encoder_cluster_qwen_cache.json"))
        )
        self.assertFalse(
            is_encoder_cluster_report(
                Path("encoder_cluster_llama_cache_maxlen512.json")
            )
        )

    def _write(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_dry_run_finds_encoder_and_rank_bundles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data" / "semantle" / "analysis"
            root.mkdir(parents=True)
            self._write(
                root / "encoder_cluster.json",
                {
                    "task": "semantle",
                    "artifact_tag": "",
                    "scores": {"qwen": {"knn_purity_at_5": 0.5}},
                },
            )
            self._write(root / "encoder_cluster_qwen_cache.json", {"n": 1})
            (root / "encoder_cluster_purity.png").write_bytes(b"png")
            (root / "encoder_cluster_metrics.png").write_bytes(b"png")
            self._write(
                root / "qwen_train_embed_effective_rank.json",
                {
                    "task": "semantle",
                    "mode": "prompt",
                    "results": [
                        {
                            "train_n_samples": 8,
                            "effective_rank": 2.5,
                            "stable_rank": 1.1,
                            "participation_ratio": 1.2,
                            "dims_for_variance": {"0.9": 3},
                        }
                    ],
                },
            )
            (root / "qwen_train_embed_effective_rank.png").write_bytes(b"png")
            self._write(
                root / "qwen_train_embed_effective_rank_defn.json",
                {
                    "task": "semantle",
                    "mode": "defn",
                    "results": [
                        {
                            "train_n_samples": 8,
                            "effective_rank": 3.1,
                            "stable_rank": 1.4,
                            "participation_ratio": 1.5,
                            "dims_for_variance": {"0.9": 4},
                        }
                    ],
                },
            )
            (root / "qwen_train_embed_effective_rank_defn.png").write_bytes(b"png")
            with patch("wandb.init") as init:
                results = log_analysis_dir(root, project="boreft", dry_run=True)
            init.assert_not_called()
            names = {row["name"] for row in results}
            self.assertEqual(
                names, {"semantle-encoder-cluster", "semantle-train-embed-rank"}
            )
            encoder = next(row for row in results if row["name"].endswith("cluster"))
            rank = next(row for row in results if "embed-rank" in row["name"])
            self.assertEqual(encoder["n_images"], 2)
            self.assertEqual(rank["n_images"], 2)
            self.assertEqual(rank["n_history"], 1)
            self.assertEqual(len(train_embed_rank_jsons(root)), 2)

    def test_tagged_encoder_cluster_is_its_own_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data" / "molopt" / "analysis"
            root.mkdir(parents=True)
            self._write(
                root / "encoder_cluster.json",
                {"task": "molopt", "artifact_tag": "", "scores": {}},
            )
            (root / "encoder_cluster_purity.png").write_bytes(b"png")
            self._write(
                root / "encoder_cluster_maxlen512.json",
                {"task": "molopt", "artifact_tag": "_maxlen512", "scores": {}},
            )
            (root / "encoder_cluster_purity_maxlen512.png").write_bytes(b"png")
            results = log_analysis_dir(root, project="boreft", dry_run=True)
            by_name = {row["name"]: row for row in results}
            self.assertEqual(by_name["molopt-encoder-cluster"]["n_images"], 1)
            self.assertEqual(
                by_name["molopt-encoder-cluster_maxlen512"]["n_images"], 1
            )

    def test_named_molopt_reports_and_leftover_rank_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data" / "molopt" / "analysis"
            root.mkdir(parents=True)
            self._write(root / "embedding_models.json", {"n_pairs": 10, "seed": 0})
            (root / "embedding_models_retrieval.png").write_bytes(b"png")
            (root / "embedding_models_structure.png").write_bytes(b"png")
            self._write(
                root / "rdkit_definition_embeds.json",
                {"n_bases": 8, "drowning": {"cosine_gap_rdkit_only": 0.01}},
            )
            (root / "rdkit_definition_embeds.png").write_bytes(b"png")
            self._write(root / "rdkit_definition_embeds_rbf.json", {"n_bases": 8})
            (root / "rdkit_definition_embeds_rbf.png").write_bytes(b"png")
            self._write(
                root / "qwen_train_embed_effective_rank.json",
                {
                    "task": "molopt",
                    "mode": "prompt",
                    "results": [{"train_n_samples": 4, "effective_rank": 1.5}],
                },
            )
            (root / "qwen_train_embed_effective_rank.png").write_bytes(b"png")
            (root / "qwen_train_embed_effective_rank_defn.png").write_bytes(b"png")
            results = log_analysis_dir(root, project="boreft", dry_run=True)
            by_name = {row["name"]: row for row in results}
            self.assertEqual(by_name["molopt-embedding-models"]["n_images"], 2)
            self.assertEqual(by_name["molopt-rdkit-definition-embeds"]["n_images"], 1)
            self.assertEqual(
                by_name["molopt-rdkit-definition-embeds-rbf"]["n_images"], 1
            )
            self.assertEqual(by_name["molopt-train-embed-rank"]["n_images"], 2)

    def test_skip_when_meta_is_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data" / "semantle" / "analysis"
            root.mkdir(parents=True)
            self._write(
                root / "encoder_cluster.json",
                {"task": "semantle", "artifact_tag": "", "scores": {}},
            )
            png = root / "encoder_cluster_purity.png"
            png.write_bytes(b"png")
            (root / "wandb_meta.json").write_text(
                json.dumps(
                    {
                        "encoder_cluster": {
                            "run_id": "already",
                            "n_images": 1,
                            "source_mtime": png.stat().st_mtime + 10,
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch("wandb.init") as init:
                results = log_analysis_dir(root, project="boreft")
            init.assert_not_called()
            self.assertEqual(results[0]["status"], "skipped")
            self.assertEqual(results[0]["run_id"], "already")

    def test_maybe_log_respects_no_wandb(self):
        self.assertIsNone(
            maybe_log_encoder_cluster(
                "unused", {}, project="boreft", no_wandb=True
            )
        )


class AnalysisLogTests(unittest.TestCase):
    def test_wandb_error_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data" / "semantle" / "analysis"
            root.mkdir(parents=True)
            (root / "encoder_cluster.json").write_text(
                json.dumps({"task": "semantle", "artifact_tag": "", "scores": {}}),
                encoding="utf-8",
            )
            (root / "encoder_cluster_purity.png").write_bytes(b"png")
            run = MagicMock()
            run.id = "partial"
            with patch("wandb.init", return_value=run), patch(
                "wandb.log", side_effect=RuntimeError("network")
            ), patch("wandb.finish"), patch("wandb.Image", side_effect=lambda p: p):
                result = maybe_log_encoder_cluster(
                    root,
                    {"task": "semantle", "artifact_tag": ""},
                    project="boreft",
                )
            self.assertEqual(result["status"], "error")
            meta = root / "wandb_meta.json"
            self.assertFalse(meta.exists() or "encoder_cluster" in (
                json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
            ))


if __name__ == "__main__":
    unittest.main()
