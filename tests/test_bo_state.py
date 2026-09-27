from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from boreft.bo.plotting import aggregate_trajectories, plot_best_so_far
from boreft.bo.state import Observation, RunState


def _observation(
    index: int,
    score: float,
    seed: int = 1,
    warmstarts: int = 2,
) -> Observation:
    return Observation(
        index=index,
        point=[float(index), 0.0],
        decoded=f"item-{index}",
        score=score,
        source="warmstart" if index < warmstarts else "acquisition",
        seed=seed,
    )


class RunStateTest(unittest.TestCase):
    def test_append_round_trip_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observations.jsonl"
            state = RunState(path)
            state.append(_observation(0, 0.2))
            state.append(_observation(1, 0.1))
            state.append(_observation(2, 0.8))
            loaded = RunState.load(path)
            self.assertEqual(loaded.scores, [0.2, 0.1, 0.8])
            self.assertEqual(
                [item.best_so_far for item in loaded.observations],
                [0.2, 0.2, 0.8],
            )
            self.assertEqual(loaded.summary()["best_decoded"], "item-2")
            self.assertEqual(loaded.summary()["n_warmstart"], 2)

    def test_append_tracks_sample_peak_for_best_so_far(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            state.append(
                Observation(
                    index=0,
                    point=[0.0],
                    decoded="near",
                    score=0.5,
                    sample_count=2,
                    sample_scores=[0.4, 0.6],
                    decoded_samples=["miss", "near"],
                    source="warmstart",
                    seed=1,
                )
            )
            state.append(
                Observation(
                    index=1,
                    point=[1.0],
                    decoded="hit",
                    score=0.55,
                    sample_count=2,
                    sample_scores=[0.1, 1.0],
                    decoded_samples=["miss", "hit"],
                    source="acquisition",
                    seed=1,
                )
            )
            self.assertEqual(
                [item.best_so_far for item in state.observations],
                [0.6, 1.0],
            )
            self.assertAlmostEqual(state.best_score, 1.0)
            self.assertEqual(state.summary()["best_decoded"], "hit")
            self.assertEqual(state.scores, [0.5, 0.55])

    def test_to_dict_defaults_best_so_far_to_peak(self):
        observation = Observation(
            index=0,
            point=[0.0],
            decoded="near",
            score=0.5,
            sample_count=2,
            sample_scores=[0.4, 0.9],
            decoded_samples=["miss", "near"],
        )
        self.assertAlmostEqual(observation.to_dict()["best_so_far"], 0.9)

    def test_rejects_out_of_order_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            with self.assertRaisesRegex(ValueError, "index"):
                state.append(_observation(1, 0.2))

    def test_rejects_nonfinite_observation(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            _observation(0, float("nan"))

    def test_load_repairs_torn_final_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observations.jsonl"
            state = RunState(path)
            state.append(_observation(0, 0.2))
            with path.open("a", encoding="utf-8") as handle:
                handle.write('{"index":1,"point":')
            loaded = RunState.load(path)
            self.assertEqual(len(loaded.observations), 1)
            loaded.append(_observation(1, 0.4))
            self.assertEqual(len(RunState.load(path).observations), 2)

    def test_load_repairs_valid_final_record_without_newline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observations.jsonl"
            first = _observation(0, 0.2)
            path.write_text(json.dumps(first.to_dict()), encoding="utf-8")
            loaded = RunState.load(path)
            loaded.append(_observation(1, 0.4))
            self.assertEqual(len(RunState.load(path).observations), 2)


class BOPlotTest(unittest.TestCase):
    def test_aggregation_and_png(self):
        runs = [
            [_observation(0, 0.1, warmstarts=1), _observation(1, 0.4, warmstarts=1)],
            [
                _observation(0, 0.2, 2, warmstarts=1),
                _observation(1, 0.3, 2, warmstarts=1),
            ],
        ]
        x, mean, std = aggregate_trajectories(runs)
        np.testing.assert_array_equal(x, [1, 2])
        np.testing.assert_allclose(mean, [0.15, 0.35])
        np.testing.assert_allclose(std, [0.05, 0.05])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.png"
            plot_best_so_far(runs, path)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)
            pdf = path.with_suffix(".pdf")
            self.assertTrue(pdf.is_file())
            self.assertGreater(pdf.stat().st_size, 0)

    def test_trajectory_keeps_warmstart_iterations(self):
        from boreft.bo.plotting import bo_trajectory, iteration_ticks

        observations = [
            _observation(0, 0.1, warmstarts=2),
            _observation(1, 0.4, warmstarts=2),
            _observation(2, 0.3, warmstarts=2),
        ]
        x, values = bo_trajectory(observations)
        np.testing.assert_array_equal(x, [1, 2, 3])
        np.testing.assert_allclose(values, [0.1, 0.4, 0.4])
        ticks = iteration_ticks(50, [0, 10, 20, 30, 40, 60], warmstart_count=10)
        self.assertEqual(ticks, [1.0, 10.0, 20.0, 30.0, 40.0, 50.0])

    def test_short_completed_run_carries_best_forward(self):
        runs = [
            [_observation(0, 0.4, warmstarts=1)],
            [
                _observation(0, 0.2, 2, warmstarts=1),
                _observation(1, 0.6, 2, warmstarts=1),
            ],
        ]
        _x, mean, _std = aggregate_trajectories(runs)
        np.testing.assert_allclose(mean, [0.3, 0.5])


if __name__ == "__main__":
    unittest.main()
