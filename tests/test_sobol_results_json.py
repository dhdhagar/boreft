"""Tests for sobol_results.json load/write helpers and per_sample row builders."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from boreft.eval.semantle import (
    SOBOL_SECTION_GREEDY,
    SOBOL_SECTION_TEMPERATURE_1_0,
    SOBOL_SECTION_TEMPERATURE_1_5,
    _build_sobol_per_sample_row,
    _nearest_mu_indices,
    _sample_sim_to_target_stats,
    load_sobol_results_json,
    write_sobol_results_json,
)


class NearestMuIndicesTest(unittest.TestCase):
    def test_picks_closest_mu_per_sample(self):
        mu_all = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 2.0],
            ]
        )
        sampled_bs = np.array(
            [
                [0.9, 0.1],
                [0.1, 0.9],
                [1.8, 2.1],
            ]
        )
        self.assertEqual(_nearest_mu_indices(mu_all, sampled_bs), [0, 1, 2])


class SampleSimToTargetStatsTest(unittest.TestCase):
    @patch("boreft.eval.semantle.encode_texts_normalized")
    def test_sorts_samples_by_descending_sim(self, mock_encode):
        mock_encode.side_effect = [
            np.array([[1.0, 0.0]]),
            np.array(
                [
                    [0.5, 0.5],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ),
        ]
        stats, sorted_samples = _sample_sim_to_target_stats(
            ["low", "high", "mid"], "target"
        )
        self.assertEqual(sorted_samples, ["high", "low", "mid"])
        self.assertEqual(stats["max"], 1.0)
        self.assertEqual(stats["min"], 0.0)
        self.assertAlmostEqual(stats["mean"], 0.5)


class BuildSobolPerSampleRowTest(unittest.TestCase):
    @patch("boreft.eval.semantle.encode_texts_normalized")
    def test_greedy_single_sample_list_format(self, mock_encode):
        mock_encode.side_effect = [
            np.array([[1.0, 0.0]]),
            np.array([[0.8, 0.2]]),
            np.array([[0.8, 0.2]]),
            np.array([[0.8, 0.2]]),
        ]
        mu_all = np.array([[1.0, 0.0], [0.0, 1.0]])
        row = _build_sobol_per_sample_row(
            idx=0,
            b=np.array([0.9, 0.1]),
            nearest_train_target="apple",
            nearest_idx=0,
            mu_all=mu_all,
            mu_norms=np.linalg.norm(mu_all, axis=1),
            sample_texts=["fruit"],
            train_vocab={"apple", "banana"},
        )
        self.assertEqual(row["nearest_train_target"], "apple")
        self.assertEqual(row["samples"], ["fruit"])
        self.assertEqual(row["n_unique"], 1)
        self.assertEqual(row["n_unseen"], 1)
        self.assertIn("bias_sim_to_nearest_train_target", row)
        self.assertIn("sample_sim_to_nearest_train_target", row)
        self.assertIn("sample_sim_to_mode", row)
        self.assertEqual(row["sample_mode"], "fruit")
        self.assertNotIn("unseen_rate", row)

    @patch("boreft.eval.semantle.encode_texts_normalized")
    def test_n_unseen_counts_unique_texts_only(self, mock_encode):
        mock_encode.side_effect = [
            np.array([[1.0, 0.0]]),
            np.array([[0.8, 0.2], [0.7, 0.3], [0.0, 1.0]]),
            np.array([[1.0, 0.0]]),
            np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        ]
        mu_all = np.array([[1.0, 0.0]])
        row = _build_sobol_per_sample_row(
            idx=0,
            b=np.array([0.9, 0.1]),
            nearest_train_target="apple",
            nearest_idx=0,
            mu_all=mu_all,
            mu_norms=np.linalg.norm(mu_all, axis=1),
            sample_texts=["fruit", "fruit", "berry"],
            train_vocab={"apple", "banana"},
        )
        self.assertEqual(row["n_unique"], 2)
        self.assertEqual(row["n_unseen"], 2)
        self.assertEqual(row["sample_mode"], "fruit")
        self.assertAlmostEqual(row["sample_sim_to_mode"]["mean"], 2 / 3)


class SobolResultsJsonTest(unittest.TestCase):
    def _per_sample_row(self) -> dict:
        return {
            "idx": 0,
            "b_norm": 1.0,
            "nearest_train_target": "apple",
            "n_unique": 1,
            "n_unseen": 0,
            "bias_sim_to_nearest_train_target": 0.9,
            "sample_sim_to_nearest_train_target": {
                "max": 0.8,
                "min": 0.8,
                "mean": 0.8,
                "std": 0.0,
            },
            "sample_sim_to_mode": {
                "max": 0.8,
                "min": 0.8,
                "mean": 0.8,
                "std": 0.0,
            },
            "samples": ["apple"],
        }

    def _greedy_section(self) -> dict:
        return {
            "n_sobol_points": 2,
            "unseen_rate": 0.5,
            "per_sample": [self._per_sample_row()],
        }

    def _temp_section(self, temperature: float) -> dict:
        return {
            "n_sobol_points": 2,
            "n_samples": 3,
            "temperature": temperature,
            "unique_frac": 0.8,
            "per_sample": [
                {
                    **self._per_sample_row(),
                    "n_unique": 2,
                    "samples": ["apple", "fruit"],
                }
            ],
        }

    def test_load_legacy_flat_greedy_only_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sobol_results.json"
            path.write_text(json.dumps(self._greedy_section()), encoding="utf-8")
            data = load_sobol_results_json(str(path))
            self.assertEqual(set(data.keys()), {SOBOL_SECTION_GREEDY})
            self.assertEqual(data[SOBOL_SECTION_GREEDY]["unseen_rate"], 0.5)

    def test_load_legacy_temperature_key_normalized(self):
        legacy = {"temperature": self._temp_section(1.0)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sobol_results.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            data = load_sobol_results_json(str(path))
            self.assertIn(SOBOL_SECTION_TEMPERATURE_1_0, data)
            self.assertEqual(
                data[SOBOL_SECTION_TEMPERATURE_1_0]["temperature"],
                1.0,
            )

    def test_write_round_trip_all_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "sobol_results.json")
            write_sobol_results_json(
                path,
                greedy=self._greedy_section(),
                temperature_1_0=self._temp_section(1.0),
                temperature_1_5=self._temp_section(1.5),
            )
            data = load_sobol_results_json(path)
            self.assertIn(SOBOL_SECTION_GREEDY, data)
            self.assertIn(SOBOL_SECTION_TEMPERATURE_1_0, data)
            self.assertIn(SOBOL_SECTION_TEMPERATURE_1_5, data)
            self.assertNotIn("temperature", json.loads(Path(path).read_text()))
            greedy_row = data[SOBOL_SECTION_GREEDY]["per_sample"][0]
            self.assertEqual(greedy_row["nearest_train_target"], "apple")
            self.assertEqual(greedy_row["samples"], ["apple"])
            self.assertEqual(data[SOBOL_SECTION_GREEDY]["mode"], SOBOL_SECTION_GREEDY)
            self.assertFalse(data[SOBOL_SECTION_GREEDY]["do_sample"])
            self.assertEqual(
                data[SOBOL_SECTION_TEMPERATURE_1_5]["mode"],
                SOBOL_SECTION_TEMPERATURE_1_5,
            )
            self.assertTrue(data[SOBOL_SECTION_TEMPERATURE_1_5]["do_sample"])

    def test_partial_write_preserves_existing_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "sobol_results.json")
            write_sobol_results_json(
                path,
                greedy=self._greedy_section(),
                temperature_1_0=self._temp_section(1.0),
            )
            write_sobol_results_json(
                path,
                temperature_1_5=self._temp_section(1.5),
            )
            data = load_sobol_results_json(path)
            self.assertEqual(data[SOBOL_SECTION_GREEDY]["unseen_rate"], 0.5)
            self.assertEqual(
                data[SOBOL_SECTION_TEMPERATURE_1_0]["temperature"],
                1.0,
            )
            self.assertEqual(
                data[SOBOL_SECTION_TEMPERATURE_1_5]["temperature"],
                1.5,
            )


if __name__ == "__main__":
    unittest.main()
