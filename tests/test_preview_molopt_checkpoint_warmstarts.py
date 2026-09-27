"""Tests for scripts/preview_molopt_checkpoint_warmstarts.py."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tests.test_molopt_oracle_screen import ETHANOL, ETHANOL_PERM, FakeOracle

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script():
    path = os.path.join(
        _REPO_ROOT, "scripts", "preview_molopt_checkpoint_warmstarts.py"
    )
    spec = importlib.util.spec_from_file_location(
        "preview_molopt_checkpoint_warmstarts", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preview = _load_script()


class PreviewCheckpointWarmstartsTests(unittest.TestCase):
    def test_methods_would_share_oracle_targets_and_scores(self):
        words = ["CCO", "CCC", "CCCC", "CC", "C", "c1ccccc1"]
        train_mu = np.arange(len(words), dtype=np.float32).reshape(-1, 1)
        ckpt = SimpleNamespace(words=words, saved_cfg={"task": "molopt"})

        def decode(point):
            index = int(round(float(np.asarray(point).reshape(-1)[0])))
            gold = words[index]
            if gold == "CCC":
                return "not a molecule"
            if gold == "CCO":
                return ETHANOL_PERM
            return gold

        payload = preview.preview_checkpoint_warmstarts(
            ckpt=ckpt,
            train_mu=train_mu,
            reft_output_dir="outputs/demo",
            seeds=(1, 2),
            warmstart_count=5,
            oracles={"DRD2": FakeOracle(), "GSK3B": FakeOracle(), "JNK3": FakeOracle()},
            decode=decode,
        )
        self.assertEqual(len(payload["seeds"]), 2)
        for block in payload["seeds"]:
            self.assertTrue(block["shared_across_oracles"])
            self.assertEqual(len(block["rows"]), 5)
            labels = [row["decoded"] for row in block["rows"]]
            self.assertNotIn("CCC", labels)
            self.assertIn("CCO", labels)
            ethanol = next(row for row in block["rows"] if row["decoded"] == "CCO")
            self.assertEqual(ethanol["greedy"], ETHANOL_PERM)
            self.assertEqual(ethanol["scores"]["DRD2"], 0.8)
            self.assertTrue(ethanol["reconstructed"])

        text = preview.format_preview(payload)
        self.assertIn("shared      every method", text)
        self.assertIn("DRD2", text)
        self.assertIn("CCO", text)

    def test_mismatched_oracle_targets_raise(self):
        words = ["CCO", "CCC", "CCCC", "CC"]
        train_mu = np.arange(len(words), dtype=np.float32).reshape(-1, 1)
        ckpt = SimpleNamespace(words=words, saved_cfg={"task": "molopt"})

        def fake_select(config, *_args, **_kwargs):
            label = "CCO" if config.target == "DRD2" else "CC"
            records = [
                {
                    "decoded": label,
                    "components": {
                        "warmstart_word": label,
                        "warmstart_reconstructed": True,
                    },
                }
            ] * 2
            return np.array([[0.0], [1.0]], dtype=np.float32), records

        with mock.patch.object(preview, "select_warmstarts", fake_select):
            with self.assertRaisesRegex(ValueError, "differ from DRD2"):
                preview.preview_checkpoint_warmstarts(
                    ckpt=ckpt,
                    train_mu=train_mu,
                    reft_output_dir="outputs/demo",
                    seeds=(1,),
                    warmstart_count=2,
                    oracles={"DRD2": FakeOracle(), "GSK3B": FakeOracle()},
                    decode=lambda _point: "CCO",
                )

    def test_pinned_file_is_looked_up_on_this_checkpoint(self):
        words = ["CCO", "CCC", "CCCC", "CC"]
        train_mu = np.array(
            [[1.0], [2.0], [3.0], [4.0]], dtype=np.float32
        )
        ckpt = SimpleNamespace(words=words, saved_cfg={"task": "molopt"})

        def decode(point):
            index = int(round(float(np.asarray(point).reshape(-1)[0]))) - 1
            return words[index]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pin.json"
            path.write_text(
                json.dumps({"seeds": {"1": ["CCCC", "CC"]}}),
                encoding="utf-8",
            )
            payload = preview.preview_checkpoint_warmstarts(
                ckpt=ckpt,
                train_mu=train_mu,
                reft_output_dir="outputs/demo",
                seeds=(1,),
                warmstart_count=2,
                oracles={"DRD2": FakeOracle(), "GSK3B": FakeOracle()},
                decode=decode,
                warmstart_file=str(path),
            )
        self.assertEqual(payload["warmstart_file"], str(path))
        self.assertEqual(payload["seeds"][0]["labels"], ["CCCC", "CC"])
        self.assertEqual(payload["seeds"][0]["rows"][0]["train_index"], 2)

    def test_pinned_canonical_smiles_still_report_train_index(self):
        words = ["CCO", "CCC", "CCCC"]
        train_mu = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        ckpt = SimpleNamespace(words=words, saved_cfg={"task": "molopt"})

        def decode(point):
            index = int(round(float(np.asarray(point).reshape(-1)[0]))) - 1
            return words[index]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pin.json"
            path.write_text(
                json.dumps({"seeds": {"1": ["OCC", "CCCC"]}}),
                encoding="utf-8",
            )
            payload = preview.preview_checkpoint_warmstarts(
                ckpt=ckpt,
                train_mu=train_mu,
                reft_output_dir="outputs/demo",
                seeds=(1,),
                warmstart_count=2,
                oracles={"DRD2": FakeOracle()},
                decode=decode,
                warmstart_file=str(path),
            )
        self.assertEqual(payload["seeds"][0]["rows"][0]["train_index"], 0)
        self.assertEqual(payload["seeds"][0]["rows"][0]["warmstart_word"], "CCO")


if __name__ == "__main__":
    unittest.main()
