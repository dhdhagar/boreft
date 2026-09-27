"""Tests for ``scripts/analyze_molopt_encoder_cluster.py``.

Encoders are stubbed. The experiment loads Qwen embedding, Qwen3-1.7B
last-token, Llama, and a local MiST checkpoint. These tests pin clustering
metrics for the SMILES+definition string and the SMILES-only prompt.
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
    path = os.path.join(_REPO_ROOT, "scripts", "analyze_molopt_encoder_cluster.py")
    spec = importlib.util.spec_from_file_location(
        "analyze_molopt_encoder_cluster", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


enc = _load_script()
mtv = enc.mtv

ETHANOL = "CCO"
ETHANOL_DEFN = "a primary alcohol that is ethane substituted by a hydroxy group."
ETHANOL_RDKIT = (46.069, -0.0014, 20.23, 1, 1, 0, 0, 1.0, 0, 0)
BENZENE = "c1ccccc1"
BENZENE_DEFN = "the simplest aromatic hydrocarbon, consisting of a six-membered ring."
BENZENE_RDKIT = (78.114, 1.6866, 0.0, 0, 0, 0, 0, 0.0, 1, 0)


def _write_fixture(tmp: str) -> tuple[str, str, list[str], list[str]]:
    defs_path = os.path.join(tmp, "definitions.jsonl")
    rdkit_path = os.path.join(tmp, "rdkit.jsonl")
    smiles = [
        "CCO",
        "CCOC",
        "CCOCC",
        "CCCO",
        "CCCCO",
        "CO",
        "c1ccccc1",
        "Cc1ccccc1",
        "CCc1ccccc1",
        "CCCc1ccccc1",
        "CCCCc1ccccc1",
        "c1ccc(O)cc1",
    ]
    cats = ["organooxygen"] * 6 + ["benzenoid"] * 6
    defns = [ETHANOL_DEFN if cat == "organooxygen" else BENZENE_DEFN for cat in cats]
    with open(defs_path, "w", encoding="utf-8") as f:
        for s, cat, defn in zip(smiles, cats, defns):
            f.write(
                json.dumps(
                    {
                        "target": s,
                        "definition": defn,
                        "category": cat,
                        "category_normalized": cat,
                    }
                )
                + "\n"
            )
    with open(rdkit_path, "w", encoding="utf-8") as f:
        for s, cat in zip(smiles, cats):
            values = list(ETHANOL_RDKIT if cat == "organooxygen" else BENZENE_RDKIT)
            f.write(json.dumps({"target": s, "definition": values}) + "\n")
    return defs_path, rdkit_path, smiles, cats


class EncoderClusterTests(unittest.TestCase):
    def test_variant_is_smiles_defn(self):
        self.assertEqual(enc.VARIANT, "smiles_defn")
        mol = mtv.Molecule(
            smiles=ETHANOL,
            definition=ETHANOL_DEFN,
            category="alcohol",
            category_normalized="organooxygen",
            rdkit_values=ETHANOL_RDKIT,
        )
        self.assertEqual(
            mtv.variant_text(mol, enc.VARIANT),
            f"The molecule '{ETHANOL}' is: {ETHANOL_DEFN}",
        )

    def test_empty_encoders_means_all(self):
        self.assertEqual(
            enc.selected_encoders([]), ["qwen", "qwen_llm", "llama"]
        )
        self.assertEqual(
            enc.selected_encoders([], enc.MOLOPT_ENCODER_NAMES),
            ["qwen", "qwen_llm", "llama", "mist"],
        )
        self.assertEqual(
            enc.selected_variants([]), ["smiles_defn", "smiles_prompt"]
        )

    def test_run_scores_label_one_hots_at_purity_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            defs_path, rdkit_path, smiles, cats = _write_fixture(tmp)
            vec = {
                "organooxygen": np.array([1.0, 0.0], dtype=np.float64),
                "benzenoid": np.array([0.0, 1.0], dtype=np.float64),
            }

            def encode_qwen(molecules, texts):
                return np.stack([vec[mtv.label_of(m, "category_normalized")] for m in molecules])

            def encode_qwen_llm(molecules, texts):
                return 0.25 * encode_qwen(molecules, texts)

            def encode_llama(molecules, texts):
                # Same geometry, slightly scaled so the encoders are distinct arrays.
                return 0.5 * encode_qwen(molecules, texts)

            def encode_mist(molecules, texts):
                return 0.75 * encode_qwen(molecules, texts)

            args = enc.parse_args(
                [
                    "--definitions",
                    defs_path,
                    "--rdkit-definitions",
                    rdkit_path,
                    "--n",
                    "12",
                    "--seed",
                    "0",
                    "--out-dir",
                    tmp,
                ]
            )
            reports = enc.run(
                args,
                encode_fns={
                    "qwen": encode_qwen,
                    "qwen_llm": encode_qwen_llm,
                    "llama": encode_llama,
                    "mist": encode_mist,
                },
            )
            self.assertEqual(
                [report["variant"] for report in reports],
                ["smiles_defn", "smiles_prompt"],
            )
            report = reports[0]
            prompt = reports[1]
            self.assertEqual(report["n"], 12)
            self.assertEqual(report["mist_model"], enc.DEFAULT_MIST_MODEL)
            self.assertIsNone(report["mist_checkpoint"])
            self.assertEqual(report["llama_pooling"], "last_instruction")
            self.assertEqual(prompt["llama_pooling"], "last_token")
            self.assertEqual(
                set(report["scores"]), {"qwen", "qwen_llm", "llama", "mist"}
            )
            self.assertEqual(set(prompt["scores"]), set(report["scores"]))
            for name, scores in report["scores"].items():
                self.assertEqual(scores["knn_purity_at_5"], 1.0, msg=name)
                self.assertEqual(scores["kmeans_purity"], 1.0, msg=name)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "encoder_cluster_n12.json")))
            self.assertTrue(
                os.path.isfile(os.path.join(tmp, "encoder_cluster_n12_prompt.json"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(tmp, "encoder_cluster_qwen_emb_n12.npy"))
            )
            self.assertTrue(
                os.path.isfile(
                    os.path.join(tmp, "encoder_cluster_mist_emb_n12_prompt.npy")
                )
            )
            self.assertEqual(report["artifact_tag"], "_n12")
            self.assertEqual(report["cache_tag"], "_n12")
            self.assertEqual(prompt["artifact_tag"], "_n12_prompt")
            self.assertEqual(prompt["cache_tag"], "_n12_prompt")
            self.assertEqual(report["reused_encoders"], [])
            self.assertIn(ETHANOL, report["example"])
            self.assertIn(ETHANOL_DEFN, report["example"])
            self.assertTrue(
                prompt["example"].startswith("The description for molecule '")
            )
            self.assertTrue(prompt["example"].endswith("'."))
            self.assertNotIn(ETHANOL_DEFN, prompt["example"])

    def test_missing_encode_fn_raises_instead_of_loading_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            defs_path, rdkit_path, _, _ = _write_fixture(tmp)

            def encode_qwen(molecules, texts):
                return np.ones((len(molecules), 2), dtype=np.float64)

            args = enc.parse_args(
                [
                    "--definitions",
                    defs_path,
                    "--rdkit-definitions",
                    rdkit_path,
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
            self.assertIn("mist", str(ctx.exception))

    def test_count_truncated_uses_untruncated_token_ids(self):
        seen = {}

        class _Tok:
            def __call__(self, texts, **kwargs):
                seen["truncation"] = kwargs.get("truncation")
                return {"input_ids": [[0] * (5 if t == "short" else 20) for t in texts]}

        self.assertEqual(enc._count_truncated(_Tok(), ["short", "long-string"], 10), 1)
        self.assertFalse(seen["truncation"])

    def test_parse_args_accepts_sbatch_flags(self):
        args = enc.parse_args(
            [
                "--definitions",
                "defs.jsonl",
                "--rdkit-definitions",
                "rdkit.jsonl",
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
                "--mist-model",
                "Qwen2.5-3B_pretrained-v4-cot",
                "--mist-batch-size",
                "4",
                "--device",
                "auto",
                "--out-dir",
                "/tmp/out",
            ]
        )
        self.assertEqual(args.pooling, "last_instruction")
        self.assertEqual(args.max_length, 128)
        self.assertEqual(args.qwen_llm_model, "Qwen/Qwen3-1.7B")
        self.assertEqual(args.mist_model, enc.DEFAULT_MIST_MODEL)
        self.assertEqual(args.mist_batch_size, 4)
        self.assertEqual(args.cache_dir, None)
        self.assertEqual(
            enc.selected_encoders(args.encoders, enc.MOLOPT_ENCODER_NAMES),
            ["qwen", "qwen_llm", "llama", "mist"],
        )
        self.assertEqual(enc.selected_variants(args.variants), list(enc.VARIANT_NAMES))
        self.assertFalse(args.reuse_previous)

    def test_artifact_tag_empty_for_defaults(self):
        self.assertEqual(enc.artifact_tag(enc.parse_args([])), "")
        self.assertEqual(enc.cache_tag(enc.parse_args([])), "")
        self.assertEqual(enc.artifact_tag(enc.parse_args(["--reuse-previous"])), "")

    def test_cache_tag_omits_encoders_and_models(self):
        args = enc.parse_args(
            ["--n", "12", "--encoders", "qwen", "--qwen-llm-model", "Qwen/Qwen3-4B"]
        )
        self.assertEqual(enc.artifact_tag(args), "_n12_encoders-qwen_qwenllm-Qwen3-4B")
        self.assertEqual(enc.cache_tag(args), "_n12")

    def test_artifact_tag_appends_custom_flags(self):
        args = enc.parse_args(["--max-length", "512", "--encoders", "qwen"])
        self.assertEqual(enc.artifact_tag(args), "_encoders-qwen_maxlen512")
        args = enc.parse_args(["--qwen-llm-model", "Qwen/Qwen3-4B"])
        self.assertEqual(enc.artifact_tag(args), "_qwenllm-Qwen3-4B")
        args = enc.parse_args(["--max-length", "128", "--seed", "0"])
        self.assertEqual(enc.artifact_tag(args), "")

    def _one_hot_encode_fns(self, calls: list[str] | None = None):
        vec = {
            "organooxygen": np.array([1.0, 0.0], dtype=np.float64),
            "benzenoid": np.array([0.0, 1.0], dtype=np.float64),
        }

        def make(name: str, scale: float):
            def encode(molecules, texts):
                if calls is not None:
                    calls.append(name)
                rows = np.stack(
                    [vec[mtv.label_of(m, "category_normalized")] for m in molecules]
                )
                return scale * rows

            return encode

        return {
            "qwen": make("qwen", 1.0),
            "qwen_llm": make("qwen_llm", 0.25),
            "llama": make("llama", 0.5),
            "mist": make("mist", 0.75),
        }

    def test_reuse_previous_skips_encoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            defs_path, rdkit_path, _, _ = _write_fixture(tmp)
            argv = [
                "--definitions",
                defs_path,
                "--rdkit-definitions",
                rdkit_path,
                "--n",
                "12",
                "--out-dir",
                tmp,
            ]
            first = enc.run(
                enc.parse_args(argv), encode_fns=self._one_hot_encode_fns()
            )
            self.assertEqual(first[0]["reused_encoders"], [])
            self.assertEqual(first[1]["variant"], "smiles_prompt")
            calls: list[str] = []
            second = enc.run(
                enc.parse_args(argv + ["--reuse-previous"]),
                encode_fns=self._one_hot_encode_fns(calls),
            )
            self.assertEqual(calls, [])
            self.assertEqual(
                second[0]["reused_encoders"], ["qwen", "qwen_llm", "llama", "mist"]
            )
            self.assertEqual(second[0]["scores"]["qwen"]["kmeans_purity"], 1.0)
            self.assertEqual(
                second[1]["reused_encoders"], ["qwen", "qwen_llm", "llama", "mist"]
            )

    def test_reuse_previous_fills_missing_encode_fns(self):
        with tempfile.TemporaryDirectory() as tmp:
            defs_path, rdkit_path, _, _ = _write_fixture(tmp)
            argv = [
                "--definitions",
                defs_path,
                "--rdkit-definitions",
                rdkit_path,
                "--n",
                "12",
                "--encoders",
                "qwen",
                "--out-dir",
                tmp,
            ]
            enc.run(enc.parse_args(argv), encode_fns=self._one_hot_encode_fns())
            calls: list[str] = []
            fns = self._one_hot_encode_fns(calls)
            reports = enc.run(
                enc.parse_args(
                    [
                        "--definitions",
                        defs_path,
                        "--rdkit-definitions",
                        rdkit_path,
                        "--n",
                        "12",
                        "--out-dir",
                        tmp,
                        "--reuse-previous",
                    ]
                ),
                encode_fns={
                    "qwen_llm": fns["qwen_llm"],
                    "llama": fns["llama"],
                    "mist": fns["mist"],
                },
            )
            self.assertEqual(
                calls,
                ["qwen_llm", "qwen_llm", "llama", "llama", "mist", "mist"],
            )
            self.assertEqual(reports[0]["reused_encoders"], ["qwen"])
            self.assertEqual(reports[1]["reused_encoders"], ["qwen"])
            self.assertEqual(
                set(reports[0]["scores"]), {"qwen", "qwen_llm", "llama", "mist"}
            )

    def test_reuse_previous_rejects_model_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            defs_path, rdkit_path, _, _ = _write_fixture(tmp)
            argv = [
                "--definitions",
                defs_path,
                "--rdkit-definitions",
                rdkit_path,
                "--n",
                "12",
                "--encoders",
                "qwen",
                "--out-dir",
                tmp,
            ]
            enc.run(enc.parse_args(argv), encode_fns=self._one_hot_encode_fns())
            calls: list[str] = []
            reports = enc.run(
                enc.parse_args(
                    argv + ["--reuse-previous", "--qwen-model", "Qwen/other"]
                ),
                encode_fns=self._one_hot_encode_fns(calls),
            )
            self.assertEqual(calls, ["qwen", "qwen"])
            self.assertEqual(reports[0]["reused_encoders"], [])
            self.assertEqual(reports[1]["reused_encoders"], [])

    def test_prompt_text_uses_embedding_prompt(self):
        mol = mtv.Molecule(
            smiles=ETHANOL,
            definition=ETHANOL_DEFN,
            category="alcohol",
            category_normalized="organooxygen",
            rdkit_values=ETHANOL_RDKIT,
        )
        self.assertEqual(
            enc.texts_for_variant([mol], "smiles_prompt"),
            [f"The description for molecule '{ETHANOL}'."],
        )

    def test_resolve_mist_model_uses_directory_name_under_hf_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = os.path.join(tmp, "mist", enc.DEFAULT_MIST_MODEL)
            os.makedirs(checkpoint)
            with open(os.path.join(checkpoint, "config.json"), "w", encoding="utf-8") as handle:
                handle.write("{}\n")
            previous = os.environ.get("HF_HOME")
            os.environ["HF_HOME"] = tmp
            try:
                self.assertEqual(
                    enc.resolve_mist_model(enc.DEFAULT_MIST_MODEL), checkpoint
                )
                self.assertEqual(enc.resolve_mist_model(checkpoint), checkpoint)
            finally:
                if previous is None:
                    os.environ.pop("HF_HOME", None)
                else:
                    os.environ["HF_HOME"] = previous

    def test_resolve_mist_model_falls_back_to_manifest_preferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = os.path.join(tmp, "mist", "models", "qwen_pretranined_v6")
            os.makedirs(checkpoint)
            with open(os.path.join(checkpoint, "config.json"), "w", encoding="utf-8") as handle:
                handle.write("{}\n")
            manifest = os.path.join(tmp, "mist", "mist_models.json")
            with open(manifest, "w", encoding="utf-8") as handle:
                json.dump({"preferred": checkpoint}, handle)
            previous = os.environ.get("HF_HOME")
            os.environ["HF_HOME"] = tmp
            try:
                self.assertEqual(
                    enc.resolve_mist_model(enc.DEFAULT_MIST_MODEL), checkpoint
                )
                bare = os.path.join(tmp, "mist", "not-a-checkpoint")
                os.makedirs(bare)
                with self.assertRaises(FileNotFoundError):
                    enc.resolve_mist_model(bare)
            finally:
                if previous is None:
                    os.environ.pop("HF_HOME", None)
                else:
                    os.environ["HF_HOME"] = previous

    def test_prompt_artifact_tag(self):
        args = enc.parse_args([])
        self.assertEqual(enc.artifact_tag(args, "smiles_prompt"), "_prompt")
        self.assertEqual(enc.cache_tag(args, "smiles_prompt"), "_prompt")
        self.assertEqual(enc.artifact_tag(args, "smiles_defn"), "")


if __name__ == "__main__":
    unittest.main()
