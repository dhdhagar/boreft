"""Tests for ``scripts/analyze_semantle_encoder_cluster.py``.

Encoders are stubbed. The experiment's job is to load Qwen embedding,
Qwen3-1.7B last-token, and Llama; the test suite pins word+defn clustering
metrics on inputs whose answers are known by construction.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script():
    path = os.path.join(_REPO_ROOT, "scripts", "analyze_semantle_encoder_cluster.py")
    spec = importlib.util.spec_from_file_location(
        "analyze_semantle_encoder_cluster", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


enc = _load_script()
mtv = enc.mtv

LAPTOP = "laptop"
LAPTOP_DEFN = "A portable personal machine with a hinged screen and keys."
APPLE = "apple"
APPLE_DEFN = "The round fruit of a tree in the rose family."


def _write_fixture(tmp: str) -> str:
    defs_path = os.path.join(tmp, "definitions.jsonl")
    words = [
        "laptop",
        "desktop",
        "keyboard",
        "monitor",
        "server",
        "router",
        "apple",
        "banana",
        "orange",
        "grape",
        "pear",
        "peach",
    ]
    cats = ["computing"] * 6 + ["food drink"] * 6
    defns = [LAPTOP_DEFN if cat == "computing" else APPLE_DEFN for cat in cats]
    with open(defs_path, "w", encoding="utf-8") as handle:
        for word, cat, defn in zip(words, cats, defns):
            handle.write(
                json.dumps(
                    {
                        "target": word,
                        "definition": defn,
                        "category": cat,
                        "category_normalized": cat,
                    }
                )
                + "\n"
            )
    return defs_path


class SemantleEncoderClusterTests(unittest.TestCase):
    def test_variant_is_word_defn(self):
        self.assertEqual(enc.VARIANT, "word_defn")
        word = enc.Word(
            target=LAPTOP,
            definition=LAPTOP_DEFN,
            category="computing hardware",
            category_normalized="computing",
        )
        self.assertEqual(
            enc.variant_text(word),
            f"The meaning of '{LAPTOP}' is: {LAPTOP_DEFN}",
        )
        self.assertEqual(
            enc.mec.item_word_definition_pair(word), (LAPTOP, LAPTOP_DEFN)
        )

    def test_empty_encoders_means_all(self):
        self.assertEqual(
            enc.selected_encoders([]), ["qwen", "qwen_llm", "llama"]
        )

    def _one_hot_encode_fns(self, calls: list[str] | None = None):
        vec = {
            "computing": np.array([1.0, 0.0], dtype=np.float64),
            "food drink": np.array([0.0, 1.0], dtype=np.float64),
        }

        def make(name: str, scale: float):
            def encode(words, texts):
                if calls is not None:
                    calls.append(name)
                rows = np.stack(
                    [vec[mtv.label_of(w, "category_normalized")] for w in words]
                )
                return scale * rows

            return encode

        return {
            "qwen": make("qwen", 1.0),
            "qwen_llm": make("qwen_llm", 0.25),
            "llama": make("llama", 0.5),
        }

    def test_run_scores_label_one_hots_at_purity_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            defs_path = _write_fixture(tmp)
            args = enc.parse_args(
                [
                    "--definitions",
                    defs_path,
                    "--n",
                    "12",
                    "--seed",
                    "0",
                    "--out-dir",
                    tmp,
                ]
            )
            report = enc.run(args, encode_fns=self._one_hot_encode_fns())
            self.assertEqual(report["n"], 12)
            self.assertEqual(report["task"], "semantle")
            self.assertEqual(report["variant"], "word_defn")
            self.assertEqual(report["qwen_llm_model"], "Qwen/Qwen3-1.7B")
            self.assertEqual(set(report["scores"]), {"qwen", "qwen_llm", "llama"})
            for name, scores in report["scores"].items():
                self.assertEqual(scores["knn_purity_at_5"], 1.0, msg=name)
                self.assertEqual(scores["kmeans_purity"], 1.0, msg=name)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "encoder_cluster_n12.json")))
            self.assertTrue(
                os.path.isfile(os.path.join(tmp, "encoder_cluster_qwen_emb_n12.npy"))
            )
            self.assertEqual(report["artifact_tag"], "_n12")
            self.assertEqual(report["reused_encoders"], [])
            self.assertIn(LAPTOP, report["targets"])
            self.assertIn(LAPTOP_DEFN, report["example"])

    def test_missing_encode_fn_raises_instead_of_loading_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            defs_path = _write_fixture(tmp)

            def encode_qwen(words, texts):
                return np.ones((len(words), 2), dtype=np.float64)

            args = enc.parse_args(
                [
                    "--definitions",
                    defs_path,
                    "--n",
                    "12",
                    "--out-dir",
                    tmp,
                ]
            )
            with self.assertRaises(ValueError) as ctx:
                enc.run(args, encode_fns={"qwen": encode_qwen})
            self.assertIn("qwen_llm", str(ctx.exception))
            self.assertIn("llama", str(ctx.exception))

    def test_parse_args_accepts_sbatch_flags(self):
        args = enc.parse_args(
            [
                "--definitions",
                "defs.jsonl",
                "--n",
                "0",
                "--seed",
                "0",
                "--label-field",
                "category_normalized",
                "--qwen-model",
                "Qwen/Qwen3-Embedding-0.6B",
                "--qwen-llm-model",
                "Qwen/Qwen3-1.7B",
                "--llama-model",
                "meta-llama/Llama-3.2-1B-Instruct",
                "--cache-dir",
                None,
                "--pooling",
                "last_instruction",
                "--max-length",
                "128",
                "--qwen-batch-size",
                "64",
                "--qwen-llm-batch-size",
                "16",
                "--llama-batch-size",
                "16",
                "--device",
                "auto",
                "--out-dir",
                "/tmp/out",
            ]
        )
        self.assertEqual(args.pooling, "last_instruction")
        self.assertEqual(args.max_length, 128)
        self.assertEqual(args.qwen_llm_model, "Qwen/Qwen3-1.7B")
        self.assertFalse(args.reuse_previous)
        self.assertEqual(
            enc.selected_encoders(args.encoders),
            ["qwen", "qwen_llm", "llama"],
        )

    def test_artifact_tag_empty_for_defaults(self):
        self.assertEqual(enc.artifact_tag(enc.parse_args([])), "")
        self.assertEqual(enc.cache_tag(enc.parse_args([])), "")
        self.assertEqual(enc.artifact_tag(enc.parse_args(["--reuse-previous"])), "")

    def test_reuse_previous_skips_encoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            defs_path = _write_fixture(tmp)
            argv = [
                "--definitions",
                defs_path,
                "--n",
                "12",
                "--out-dir",
                tmp,
            ]
            first = enc.run(enc.parse_args(argv), encode_fns=self._one_hot_encode_fns())
            self.assertEqual(first["reused_encoders"], [])
            calls: list[str] = []
            second = enc.run(
                enc.parse_args(argv + ["--reuse-previous"]),
                encode_fns=self._one_hot_encode_fns(calls),
            )
            self.assertEqual(calls, [])
            self.assertEqual(second["reused_encoders"], ["qwen", "qwen_llm", "llama"])
            self.assertEqual(second["scores"]["qwen"]["kmeans_purity"], 1.0)


if __name__ == "__main__":
    unittest.main()
