"""Tests for experiments/semantle/dump_canonical_warmstarts.py."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script():
    path = os.path.join(
        _REPO_ROOT, "experiments", "semantle", "dump_canonical_warmstarts.py"
    )
    spec = importlib.util.spec_from_file_location("dump_canonical_warmstarts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dump = _load_script()


def _write_obs(root: Path, split: str, target: str, seed: int, words: list[str]) -> None:
    seed_dir = root / f"{split}-{target}" / f"seed_{seed}"
    seed_dir.mkdir(parents=True)
    rows = [
        {
            "source": "warmstart",
            "decoded": word.lower(),
            "components": {"warmstart_word": word},
        }
        for word in words
    ]
    rows.append({"source": "acquisition", "decoded": target, "score": 1.0})
    with (seed_dir / "observations.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class DumpCanonicalWarmstartsTests(unittest.TestCase):
    def test_dump_cell_reads_labeled_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            cell = Path(tmp) / "s1_t0_ard_d64"
            _write_obs(cell, "train", "arms", 1, ["ranger", "putty"])
            _write_obs(cell, "test", "pudding", 2, ["custard", "flan"])
            targets = dump.dump_cell(cell)
            self.assertEqual(targets["arms"]["1"], ["ranger", "putty"])
            self.assertEqual(targets["pudding"]["2"], ["custard", "flan"])

    def test_merge_prefers_rerun_train_lists(self):
        primary = {
            "arms": {"1": ["old-a", "old-b"]},
            "pudding": {"1": ["keep-a", "keep-b"]},
        }
        fallback = {
            "arms": {"1": ["new-a", "new-b"]},
            "pudding": {"1": ["other-a", "other-b"]},
        }
        merged = dump.merge_protocol_warmstarts(
            primary, fallback, train_targets={"arms"}
        )
        self.assertEqual(merged["arms"]["1"], ["new-a", "new-b"])
        self.assertEqual(merged["pudding"]["1"], ["keep-a", "keep-b"])

    def test_merge_keeps_primary_test_lists(self):
        merged = dump.merge_protocol_warmstarts(
            {"pudding": {"1": ["a", "b"]}},
            {"pudding": {"1": ["c", "d"], "2": ["e", "f"]}},
            train_targets=set(),
        )
        self.assertEqual(merged["pudding"]["1"], ["a", "b"])
        self.assertEqual(merged["pudding"]["2"], ["e", "f"])


if __name__ == "__main__":
    unittest.main()
