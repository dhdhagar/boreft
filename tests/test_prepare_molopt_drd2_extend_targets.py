from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_molopt_drd2_extend_targets import (  # noqa: E402
    candidate_rows,
    split_absorption_sets,
)


def _row(smiles: str, score: float) -> dict:
    return {"smiles": smiles, "score": score, "sources": []}


def _record(smiles: str, score: float, coverage: float) -> dict:
    return {"target": smiles, "score": score, "coverage_error": coverage}


class CandidateRowsTest(unittest.TestCase):
    def test_score_set_is_not_limited_by_the_coverage_floor(self):
        molecules = {
            "high": _row("HIGH", 0.2),
            "mid": _row("MID", 0.04),
            "tie": _row("TIE", 0.029),
            "catalog": _row("CAT", 0.9),
        }
        chosen = candidate_rows(
            molecules,
            {"CAT": 0},
            high_score=0.1,
            score_floor=0.5,
        )
        self.assertEqual([row["smiles"] for row in chosen], ["HIGH"])

    def test_coverage_pool_keeps_scores_above_the_floor(self):
        molecules = {
            "a": _row("A", 0.031),
            "b": _row("B", 0.029),
        }
        chosen = candidate_rows(
            molecules, {}, high_score=0.1, score_floor=0.03
        )
        self.assertEqual([row["smiles"] for row in chosen], ["A"])


class SplitAbsorptionSetsTest(unittest.TestCase):
    def test_coverage_set_is_the_farthest_above_the_floor(self):
        records = [
            _record("tie", 0.029, 0.90),
            _record("far-low", 0.04, 0.80),
            _record("far-high", 0.20, 0.70),
            _record("near", 0.50, 0.40),
            _record("best", 0.90, 0.20),
        ]
        by_score, by_coverage = split_absorption_sets(
            records,
            high_score=0.1,
            score_floor=0.03,
            coverage_min=0.50,
            coverage_cap=2,
        )
        self.assertEqual([row["target"] for row in by_score], ["best", "near", "far-high"])
        self.assertEqual(
            [row["target"] for row in by_coverage],
            ["far-low", "far-high"],
        )

    def test_coverage_cap_rejects_a_negative_limit(self):
        with self.assertRaises(ValueError):
            split_absorption_sets(
                [],
                high_score=0.1,
                score_floor=0.03,
                coverage_min=0.50,
                coverage_cap=-1,
            )


if __name__ == "__main__":
    unittest.main()
