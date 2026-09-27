from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from boreft.baselines.base import BaselineObservation, BaselineRunState
from boreft.bo.state import Observation, RunState
from boreft.chem import MIST_SMILES_CLOSE_TAG, MIST_SMILES_OPEN_TAG
from boreft.rescore_search import (
    SCORE_KIND,
    oracle_name_for_seed,
    rescore_search_tree,
    rescore_seed_directory,
)

ETHANOL = "CCO"
BENZENE = "c1ccccc1"


class FakeOracle:
    def __call__(self, smiles):
        out = []
        for text in smiles:
            if text in {ETHANOL, "OCC"}:
                out.append(0.8)
            elif text == BENZENE:
                out.append(0.2)
            else:
                out.append(0.1)
        return out


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _boreft_seed(root: Path, oracle: str = "GSK3B") -> Path:
    run_dir = root / "boreft_mu" / oracle
    seed_dir = run_dir / "seed_1"
    seed_dir.mkdir(parents=True)
    _write_json(run_dir / "config.json", {"oracle": oracle, "budget": 500, "target": oracle})
    state = RunState(seed_dir / "observations.jsonl")
    state.append(
        Observation(
            index=0,
            point=[0.0],
            decoded=ETHANOL,
            score=43.94,
            sample_count=1,
            sample_scores=[43.94],
            decoded_samples=[ETHANOL],
            components={"oracle": oracle, "oracle_score": 43.94, "valid": True},
            source="warmstart",
            seed=1,
        )
    )
    state.append(
        Observation(
            index=1,
            point=[1.0],
            decoded="not-a-molecule",
            score=0.0,
            sample_count=1,
            sample_scores=[0.0],
            decoded_samples=["not-a-molecule"],
            components={"oracle": oracle, "oracle_score": 0.0, "valid": False},
            source="acquisition",
            seed=1,
        )
    )
    _write_json(
        seed_dir / "summary.json",
        {
            **state.summary(),
            "seed": 1,
            "found_target": False,
            "elapsed_seconds": 12.0,
            "n_repeat_samples": 0,
            "n_repeat_proposals": 0,
            "n_expansion_rounds": 0,
        },
    )
    return seed_dir


def _baseline_seed(root: Path, oracle: str = "JNK3") -> Path:
    run_dir = root / "discrete_bo_mu" / oracle
    seed_dir = run_dir / "seed_2"
    seed_dir.mkdir(parents=True)
    _write_json(run_dir / "config.json", {"oracle": oracle, "budget": 500, "target": oracle})
    state = BaselineRunState(seed_dir / "observations.jsonl")
    state.append(
        BaselineObservation(
            index=0,
            solution=BENZENE,
            score=10.15,
            sample_count=1,
            sample_scores=[10.15],
            solution_samples=[BENZENE],
            components={"oracle": oracle, "oracle_score": 10.15, "valid": True},
            phase="warmstart",
            seed=2,
        )
    )
    state.append(
        BaselineObservation(
            index=1,
            solution=ETHANOL,
            score=4.0,
            sample_count=1,
            sample_scores=[4.0],
            solution_samples=[ETHANOL],
            components={"oracle": oracle, "oracle_score": 4.0, "valid": True},
            phase="search",
            seed=2,
        )
    )
    _write_json(
        seed_dir / "summary.json",
        {
            **state.summary(elapsed_seconds=9.0),
            "seed": 2,
            "found_target": False,
        },
    )
    return seed_dir


class OracleNameTests(unittest.TestCase):
    def test_reads_config_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed_dir = _boreft_seed(Path(tmp), "gsk3beta")
            self.assertEqual(oracle_name_for_seed(seed_dir), "GSK3B")

    def test_reads_dual_kinase_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed_dir = _boreft_seed(Path(tmp), "GSK3B_JNK3")
            self.assertEqual(oracle_name_for_seed(seed_dir), "GSK3B_JNK3")

    def test_skips_semantle_target_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed_dir = Path(tmp) / "search" / "boreft" / "train-computer" / "seed_1"
            seed_dir.mkdir(parents=True)
            _write_json(seed_dir.parent / "config.json", {"target": "computer"})
            self.assertIsNone(oracle_name_for_seed(seed_dir))


class RescoreSeedTests(unittest.TestCase):
    def test_boreft_jsonl_and_summary_use_probabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed_dir = _boreft_seed(Path(tmp))
            result = rescore_seed_directory(
                seed_dir,
                oracle=FakeOracle(),
                oracle_name="GSK3B",
                write_plots=False,
            )
            self.assertEqual(result["status"], "rescored")
            self.assertAlmostEqual(result["old_best"], 43.94)
            self.assertAlmostEqual(result["new_best"], 0.8)
            state = RunState.load(seed_dir / "observations.jsonl")
            self.assertAlmostEqual(state.observations[0].score, 0.8)
            self.assertAlmostEqual(state.observations[0].components["oracle_score"], 0.8)
            self.assertLessEqual(state.observations[0].score, 1.0)
            self.assertEqual(state.observations[1].score, 0.0)
            self.assertFalse(state.observations[1].components["valid"])
            self.assertAlmostEqual(state.observations[1].best_so_far, 0.8)
            summary = json.loads((seed_dir / "summary.json").read_text())
            self.assertEqual(summary["oracle_score_kind"], SCORE_KIND)
            self.assertAlmostEqual(summary["best_score"], 0.8)
            self.assertEqual(summary["best_decoded"], ETHANOL)
            self.assertTrue((seed_dir / "observations.jsonl.pre_proba").is_file())
            skipped = rescore_seed_directory(
                seed_dir,
                oracle=FakeOracle(),
                oracle_name="GSK3B",
                write_plots=False,
            )
            self.assertEqual(skipped["status"], "skipped")

    def test_baseline_uses_solution_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed_dir = _baseline_seed(Path(tmp))
            result = rescore_seed_directory(
                seed_dir,
                oracle=FakeOracle(),
                oracle_name="JNK3",
                write_plots=False,
            )
            self.assertEqual(result["status"], "rescored")
            state = BaselineRunState.load(seed_dir / "observations.jsonl")
            self.assertAlmostEqual(state.observations[0].score, 0.2)
            self.assertAlmostEqual(state.observations[1].score, 0.8)
            self.assertAlmostEqual(state.observations[1].best_so_far, 0.8)
            summary = json.loads((seed_dir / "summary.json").read_text())
            self.assertAlmostEqual(summary["best_score"], 0.8)
            self.assertEqual(summary["best_solution"], ETHANOL)

    def test_multi_sample_mean_and_representative(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed_dir = Path(tmp) / "boreft" / "DRD2" / "seed_1"
            seed_dir.mkdir(parents=True)
            _write_json(seed_dir.parent / "config.json", {"oracle": "DRD2"})
            state = RunState(seed_dir / "observations.jsonl")
            state.append(
                Observation(
                    index=0,
                    point=[0.0],
                    decoded=ETHANOL,
                    score=20.0,
                    sample_count=2,
                    sample_scores=[20.0, 5.0],
                    decoded_samples=[ETHANOL, BENZENE],
                    components={"oracle_score": 20.0},
                    source="acquisition",
                )
            )
            rescore_seed_directory(
                seed_dir,
                oracle=FakeOracle(),
                oracle_name="DRD2",
                write_plots=False,
            )
            updated = RunState.load(seed_dir / "observations.jsonl").observations[0]
            self.assertAlmostEqual(updated.score, 0.5)
            self.assertEqual(updated.decoded, ETHANOL)
            self.assertEqual(updated.sample_scores, [0.8, 0.2])

    def test_unwraps_mist_tags_without_changing_stored_string(self):
        tagged = f"{MIST_SMILES_OPEN_TAG}{ETHANOL}{MIST_SMILES_CLOSE_TAG}"
        with tempfile.TemporaryDirectory() as tmp:
            seed_dir = Path(tmp) / "boreft" / "GSK3B" / "seed_1"
            seed_dir.mkdir(parents=True)
            _write_json(seed_dir.parent / "config.json", {"oracle": "GSK3B"})
            state = RunState(seed_dir / "observations.jsonl")
            state.append(
                Observation(
                    index=0,
                    point=[0.0],
                    decoded=tagged,
                    score=43.94,
                    sample_count=1,
                    sample_scores=[43.94],
                    decoded_samples=[tagged],
                    components={"oracle_score": 43.94},
                    source="acquisition",
                )
            )
            rescore_seed_directory(
                seed_dir,
                oracle=FakeOracle(),
                oracle_name="GSK3B",
                write_plots=False,
            )
            updated = RunState.load(seed_dir / "observations.jsonl").observations[0]
            self.assertAlmostEqual(updated.score, 0.8)
            self.assertEqual(updated.decoded, tagged)


class RescoreTreeTests(unittest.TestCase):
    def test_walks_property_runs_and_rebuilds_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _boreft_seed(root, "GSK3B")
            _baseline_seed(root, "JNK3")
            semantle = root / "boreft" / "train-computer" / "seed_1"
            semantle.mkdir(parents=True)
            _write_json(semantle.parent / "config.json", {"target": "computer"})
            (semantle / "observations.jsonl").write_text(
                json.dumps(
                    {
                        "index": 0,
                        "point": [0.0],
                        "decoded": "hat",
                        "score": 0.4,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            results = rescore_search_tree(
                [root],
                oracle_factory=lambda _name: FakeOracle(),
                write_plots=False,
            )
            statuses = {row["status"] for row in results}
            self.assertEqual(statuses, {"rescored"})
            self.assertEqual(len(results), 2)
            parent = json.loads(
                (root / "discrete_bo_mu" / "JNK3" / "summary.json").read_text()
            )
            self.assertAlmostEqual(parent["best_score"], 0.8)
            self.assertEqual(parent["oracle_score_kind"], SCORE_KIND)

    def test_oracle_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _boreft_seed(root, "GSK3B")
            _baseline_seed(root, "JNK3")
            results = rescore_search_tree(
                [root],
                oracles=["GSK3B"],
                oracle_factory=lambda _name: FakeOracle(),
                write_plots=False,
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["oracle"], "GSK3B")

    @patch("boreft.rescore_search.log_seed_directory")
    def test_wandb_relog_keeps_recorded_group_on_skipped_seed(self, log_seed):
        log_seed.return_value = {"status": "logged"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed_dir = _boreft_seed(root, "GSK3B")
            rescore_seed_directory(
                seed_dir,
                oracle=FakeOracle(),
                oracle_name="GSK3B",
                write_plots=False,
            )
            _write_json(
                seed_dir / "wandb_meta.json",
                {
                    "run_id": "abc123",
                    "project": "boreft",
                    "entity": "example",
                    "group": "molopt_property",
                },
            )
            results = rescore_search_tree(
                [root],
                oracles=["GSK3B"],
                oracle_factory=lambda _name: FakeOracle(),
                write_plots=False,
                wandb_relog=True,
            )
            self.assertEqual(results[0]["status"], "skipped")
            log_seed.assert_called_once()
            kwargs = log_seed.call_args.kwargs
            self.assertEqual(kwargs["group"], "molopt_property")
            self.assertTrue(kwargs["overwrite_history"])
