"""p90 oracle train/test split (no live TDC)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from boreft.chem import canonical_target_key
from boreft.data.molopt import MolOptItem
from boreft.molopt_split import (
    DEFAULT_ORACLE_CAP_PERCENTILE,
    apply_molopt_oracle_cap,
    load_oracle_split,
    save_oracle_score_cache,
    split_smiles_by_oracle_percentiles,
    write_oracle_split,
)
from boreft.oracles import TDC_ORACLE_NAMES
from boreft.train_args import TrainConfig

SMILES = ["CCO", "CC", "CCC", "CCCC", "c1ccccc1"]


def _csv(path: str, smiles=SMILES) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("inchikey,smiles\n")
        for i, text in enumerate(smiles):
            handle.write(f"K{i},{text}\n")
    return path


def _scores_for(smiles, drd2_values) -> dict[str, dict[str, float]]:
    out = {}
    for text, drd2 in zip(smiles, drd2_values):
        key = canonical_target_key(text)
        out[key] = {"DRD2": float(drd2), "GSK3B": 0.1, "JNK3": 0.1}
    return out


class SplitLogicTests(unittest.TestCase):
    def test_high_on_any_oracle_is_test_else_train(self):
        smiles = list(SMILES)
        scores = _scores_for(smiles, [0.0, 0.1, 0.2, 0.3, 0.9])
        train_idx, test_idx, meta = split_smiles_by_oracle_percentiles(
            smiles, scores, percentile=90.0
        )
        self.assertEqual(meta["n_test_eligible"], 1)
        self.assertEqual([smiles[i] for i in test_idx], ["c1ccccc1"])
        self.assertEqual(len(train_idx), 4)
        self.assertNotIn(4, train_idx)

    def test_union_high_tail_uses_any_oracle(self):
        smiles = ["CCO", "CC", "CCC"]
        scores = {
            canonical_target_key("CCO"): {"DRD2": 0.1, "GSK3B": 0.1, "JNK3": 0.1},
            canonical_target_key("CC"): {"DRD2": 0.1, "GSK3B": 0.99, "JNK3": 0.1},
            canonical_target_key("CCC"): {"DRD2": 0.1, "GSK3B": 0.1, "JNK3": 0.1},
        }
        _train, test_idx, _meta = split_smiles_by_oracle_percentiles(
            smiles, scores, percentile=90.0
        )
        self.assertEqual([smiles[i] for i in test_idx], ["CC"])

    def test_complete_cache_does_not_touch_tdc(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _csv(os.path.join(tmp, "mols.csv"))
            cache_path = os.path.join(tmp, "oracle_scores.json")
            smiles = list(SMILES)
            save_oracle_score_cache(cache_path, _scores_for(smiles, [0.1] * 5))
            items = MolOptItem.load_csv(csv_path)
            with mock.patch(
                "boreft.molopt_split.load_oracles",
                side_effect=AssertionError("TDC must not be called"),
            ):
                sampled, _idx, meta = apply_molopt_oracle_cap(
                    items,
                    percentile=90.0,
                    n_train=2,
                    seed=0,
                    cache_path=cache_path,
                )
            self.assertEqual(len(sampled), 2)
            self.assertEqual(meta["n_train"], 2)
            self.assertTrue(all(s in smiles for s in meta["test_smiles"]))

    def test_n_train_cannot_exceed_eligible_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _csv(os.path.join(tmp, "mols.csv"))
            cache_path = os.path.join(tmp, "oracle_scores.json")
            save_oracle_score_cache(
                cache_path, _scores_for(SMILES, [0.0, 0.1, 0.2, 0.3, 0.9])
            )
            items = MolOptItem.load_csv(csv_path)
            with self.assertRaisesRegex(ValueError, "exceeds the p90"):
                apply_molopt_oracle_cap(
                    items,
                    percentile=90.0,
                    n_train=5,
                    seed=0,
                    cache_path=cache_path,
                )


class LoadTrainingItemsTests(unittest.TestCase):
    def test_cap_zero_skips_oracle_split(self):
        from boreft.train import load_training_items

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _csv(os.path.join(tmp, "mols.csv"))
            out = os.path.join(tmp, "run")
            os.makedirs(out)
            config = TrainConfig(
                task="molopt",
                molopt_csv=csv_path,
                output_dir=out,
                molopt_oracle_cap_percentile=0,
                train_top_k=3,
                seed=0,
            )
            with mock.patch(
                "boreft.molopt_split.ensure_oracle_scores",
                side_effect=AssertionError("cap=0 must not score oracles"),
            ):
                items, _idx = load_training_items(config)
            self.assertEqual(len(items), 3)
            self.assertFalse(os.path.isfile(os.path.join(out, "oracle_split.json")))

    def test_capped_draw_uses_cache_and_writes_split(self):
        from boreft.train import load_training_items

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _csv(os.path.join(tmp, "mols.csv"))
            cache_path = os.path.join(tmp, "oracle_scores.json")
            save_oracle_score_cache(
                cache_path, _scores_for(SMILES, [0.0, 0.1, 0.2, 0.3, 0.9])
            )
            out = os.path.join(tmp, "run")
            os.makedirs(out)
            config = TrainConfig(
                task="molopt",
                molopt_csv=csv_path,
                output_dir=out,
                molopt_oracle_scores_path=cache_path,
                train_n_samples=2,
                seed=0,
            )
            self.assertEqual(config.molopt_oracle_cap_percentile, DEFAULT_ORACLE_CAP_PERCENTILE)
            with mock.patch(
                "boreft.molopt_split.load_oracles",
                side_effect=AssertionError("TDC must not be called"),
            ):
                items, _idx = load_training_items(config)
            self.assertEqual(len(items), 2)
            targets = {item.target for item in items}
            self.assertNotIn("c1ccccc1", targets)
            split = load_oracle_split(out)
            self.assertIsNotNone(split)
            self.assertEqual(split["test_smiles"], ["c1ccccc1"])

    def test_percentile_rejected_for_non_molopt(self):
        with self.assertRaisesRegex(ValueError, "only supported for task=molopt"):
            TrainConfig(
                task="semantle",
                semantle_csv=("words.csv",),
                molopt_oracle_cap_percentile=90,
            )


class OracleSplitIoTests(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "oracle_split.json")
            write_oracle_split(
                path,
                {
                    "percentile": 90.0,
                    "test_smiles": ["c1ccccc1"],
                    "thresholds": {"DRD2": 0.5},
                    "oracles": list(TDC_ORACLE_NAMES),
                },
            )
            loaded = load_oracle_split(tmp)
            self.assertEqual(loaded["test_smiles"], ["c1ccccc1"])
            with open(path, encoding="utf-8") as handle:
                json.load(handle)


if __name__ == "__main__":
    unittest.main()
