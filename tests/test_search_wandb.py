from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from boreft.search_wandb import (
    WANDB_CONFIG_KEYS,
    SearchWandbSession,
    _run_config,
    identity_from_seed_dir,
    iter_seed_dirs,
    log_search_tree,
    log_seed_directory,
    make_expand_progress_callback,
    maybe_log_finished_seed,
    verification_history,
    wandb_enabled,
)


class VerificationHistoryTests(unittest.TestCase):
    def test_expands_multi_sample_observations_by_verification(self):
        rows = verification_history(
            [
                {
                    "source": "warmstart",
                    "decoded": "hat",
                    "score": 0.2,
                    "sample_count": 1,
                },
                {
                    "source": "acquisition",
                    "decoded_samples": ["dog", "cat"],
                    "sample_scores": [0.4, 1.0],
                    "sample_count": 2,
                    "components": {"exact_match": True},
                },
            ],
            "Cat",
            budget=10,
        )
        self.assertEqual([r["verifications"] for r in rows], [1, 2, 3])
        self.assertEqual(rows[0]["is_warmstart"], 1)
        self.assertEqual(rows[0]["found"], 0)
        self.assertEqual(rows[1]["found"], 0)
        self.assertEqual(rows[2]["found"], 1)
        self.assertEqual(rows[2]["best_so_far"], 1.0)
        self.assertEqual(rows[2]["n_unique"], 3)

    def test_budget_truncates_history(self):
        rows = verification_history(
            [
                {
                    "decoded_samples": ["a", "b", "c"],
                    "sample_scores": [0.1, 0.2, 0.3],
                    "sample_count": 3,
                }
            ],
            "z",
            budget=2,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]["verifications"], 2)

    def test_baseline_solution_keys(self):
        rows = verification_history(
            [
                {
                    "phase": "search",
                    "solution": "Apple",
                    "score": 1.0,
                    "components": {"exact_match": True},
                }
            ],
            "apple",
        )
        self.assertEqual(rows[0]["found"], 1)


class IdentityTests(unittest.TestCase):
    def test_search_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed = Path(tmp) / "search" / "discrete_bo" / "test-apple" / "seed_2"
            seed.mkdir(parents=True)
            ident = identity_from_seed_dir(seed)
            self.assertEqual(ident["method"], "discrete_bo")
            self.assertEqual(ident["protocol"], "search")
            self.assertEqual(ident["split"], "test")
            self.assertEqual(ident["target"], "apple")
            self.assertEqual(ident["seed"], 2)
            self.assertEqual(ident["group"], "semantle-search")
            self.assertIn("discrete_bo_test-apple_seed2", ident["name"])

    def test_sweep_layout_is_boreft(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed = Path(tmp) / "sweep" / "s1_t0_ard_d64" / "train-cat" / "seed_1"
            seed.mkdir(parents=True)
            ident = identity_from_seed_dir(seed)
            self.assertEqual(ident["method"], "boreft")
            self.assertEqual(ident["sweep_slug"], "s1_t0_ard_d64")
            self.assertEqual(ident["protocol"], "sweep")
            self.assertEqual(ident["group"], "semantle-sweep")

    def test_odd_seed_name_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed = Path(tmp) / "search" / "boreft" / "train-cat" / "seed_rerun"
            seed.mkdir(parents=True)
            ident = identity_from_seed_dir(seed)
            self.assertEqual(ident["seed"], 0)


class LogSeedDirectoryTests(unittest.TestCase):
    def _seed_dir(self, tmp: str) -> Path:
        seed = Path(tmp) / "search" / "boreft" / "train-cat" / "seed_1"
        seed.mkdir(parents=True)
        (seed / "observations.jsonl").write_text(
            json.dumps(
                {
                    "source": "warmstart",
                    "decoded": "cat",
                    "score": 1.0,
                    "sample_count": 1,
                    "components": {"exact_match": True},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (seed.parent / "config.json").write_text(
            json.dumps({"target": "cat", "budget": 10, "observation_samples": 1}),
            encoding="utf-8",
        )
        (seed / "summary.json").write_text(
            json.dumps({"found_target": True, "best_score": 1.0}),
            encoding="utf-8",
        )
        return seed

    def test_dry_run_does_not_call_wandb(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed = self._seed_dir(tmp)
            with patch("wandb.init") as init:
                result = log_seed_directory(
                    seed, project="boreft", dry_run=True
                )
            init.assert_not_called()
            self.assertEqual(result["status"], "dry_run")
            self.assertEqual(result["n_verifications"], 1)

    def test_skip_when_meta_is_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed = self._seed_dir(tmp)
            (seed / "wandb_meta.json").write_text(
                json.dumps({"run_id": "abc123", "n_verifications": 1}),
                encoding="utf-8",
            )
            with patch("wandb.init") as init:
                result = log_seed_directory(seed, project="boreft")
            init.assert_not_called()
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(result["run_id"], "abc123")

    def test_logs_and_writes_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed = self._seed_dir(tmp)
            run = MagicMock()
            run.id = "newrun"
            with patch("wandb.init", return_value=run) as init, patch(
                "wandb.finish"
            ) as finish:
                result = log_seed_directory(seed, project="boreft", entity="example")
            init.assert_called_once()
            self.assertEqual(init.call_args.kwargs["project"], "boreft")
            self.assertEqual(init.call_args.kwargs["job_type"], "search")
            run.log.assert_called()
            self.assertNotIn("step", run.log.call_args.kwargs)
            finish.assert_called_once()
            self.assertEqual(result["status"], "logged")
            meta = json.loads((seed / "wandb_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["run_id"], "newrun")
            self.assertEqual(meta["n_verifications"], 1)

    def test_config_target_beats_sanitized_dirname(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed = Path(tmp) / "search" / "boreft" / "train-self_esteem" / "seed_1"
            seed.mkdir(parents=True)
            (seed / "observations.jsonl").write_text(
                json.dumps(
                    {
                        "decoded": "self-esteem",
                        "score": 1.0,
                        "sample_count": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (seed.parent / "config.json").write_text(
                json.dumps({"target": "self-esteem", "budget": 10}),
                encoding="utf-8",
            )
            run = MagicMock()
            run.id = "hyphen"
            with patch("wandb.init", return_value=run), patch("wandb.finish"):
                result = log_seed_directory(seed, project="boreft")
            self.assertEqual(result["status"], "logged")
            self.assertEqual(result["target"], "self-esteem")
            payload = run.log.call_args_list[0].args[0]
            self.assertEqual(payload["search/found"], 1)
            self.assertEqual(payload["search/verifications"], 1)
            self.assertNotIn("step", run.log.call_args.kwargs)

    def test_extra_config_target_beats_custom_search_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed = Path(tmp) / "my_search" / "seed_1"
            seed.mkdir(parents=True)
            (seed / "observations.jsonl").write_text(
                json.dumps(
                    {
                        "decoded": "Apple",
                        "score": 1.0,
                        "sample_count": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            run = MagicMock()
            run.id = "custom"
            with patch("wandb.init", return_value=run), patch("wandb.finish"):
                result = log_seed_directory(
                    seed,
                    project="boreft",
                    extra_config={"target": "Apple"},
                )
            self.assertEqual(result["status"], "logged")
            self.assertEqual(result["target"], "Apple")
            payload = run.log.call_args_list[0].args[0]
            self.assertEqual(payload["search/found"], 1)

    def test_wandb_error_does_not_write_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed = self._seed_dir(tmp)
            run = MagicMock()
            run.id = "partial"
            run.log.side_effect = RuntimeError("network")
            with patch("wandb.init", return_value=run), patch("wandb.finish"):
                result = log_seed_directory(seed, project="boreft")
            self.assertEqual(result["status"], "error")
            self.assertFalse((seed / "wandb_meta.json").exists())

    def test_maybe_log_swallows_exceptions(self):
        with patch(
            "boreft.search_wandb.log_seed_directory",
            side_effect=RuntimeError("boom"),
        ):
            result = maybe_log_finished_seed("unused", project="boreft")
        self.assertEqual(result["status"], "error")
        self.assertIn("boom", result["error"])

    def test_log_search_tree_continues_after_bad_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = self._seed_dir(tmp)
            bad = Path(tmp) / "search" / "boreft" / "train-bad" / "seed_1"
            bad.mkdir(parents=True)
            (bad / "observations.jsonl").write_text("{not json\n", encoding="utf-8")
            results = log_search_tree(
                [Path(tmp) / "search"], project="boreft", dry_run=True
            )
            statuses = {row["status"] for row in results}
            self.assertIn("error", statuses)
            self.assertIn("dry_run", statuses)
            self.assertEqual(len(results), 2)
            self.assertEqual(good.name, "seed_1")

    def test_iter_seed_dirs_finds_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed = self._seed_dir(tmp)
            found = iter_seed_dirs(Path(tmp) / "search")
            self.assertEqual(found, [seed])

    def test_maybe_log_respects_no_wandb(self):
        self.assertFalse(wandb_enabled(project="boreft", no_wandb=True))
        self.assertIsNone(
            maybe_log_finished_seed("unused", project="boreft", no_wandb=True)
        )

    def test_nested_expand_config_is_flattened(self):
        cfg = _run_config(
            {"method": "boreft", "seed_dir": "x", "run_dir": "y"},
            {"expand": {"every": 5, "new_strategy": "all", "init": "predicted"}},
        )
        self.assertEqual(cfg["expand.every"], 5)
        self.assertEqual(cfg["expand.new_strategy"], "all")
        self.assertNotIn("expand", cfg)

    def test_wandb_config_keys_cover_cli_fields(self):
        self.assertIn("wandb_project", WANDB_CONFIG_KEYS)
        self.assertIn("no_wandb", WANDB_CONFIG_KEYS)


class SharedRunStepTests(unittest.TestCase):
    def test_live_search_logs_omit_global_step(self):
        run = MagicMock()
        session = SearchWandbSession(
            run, target="cat", budget=10, seed_dir="."
        )
        state = MagicMock()
        state.observations = [
            {
                "source": "warmstart",
                "decoded": "hat",
                "score": 0.2,
                "sample_count": 1,
            },
            {
                "source": "acquisition",
                "decoded": "cat",
                "score": 1.0,
                "sample_count": 1,
                "components": {"exact_match": True},
            },
        ]
        session.log_observations(state.observations[-1], state)
        self.assertEqual(run.log.call_count, 2)
        for call in run.log.call_args_list:
            self.assertNotIn("step", call.kwargs)
            self.assertIn("search/verifications", call.args[0])
        self.assertEqual(run.log.call_args_list[-1].args[0]["search/verifications"], 2)

    def test_search_can_log_after_expansion_without_step(self):
        run = MagicMock()
        session = SearchWandbSession(
            run, target="cat", budget=200, seed_dir="."
        )
        expand = session.expansion_callback(1)
        expand(
            {
                "event": "log",
                "step": 594,
                "epoch": 5.0,
                "loss": 0.1,
            }
        )
        self.assertNotIn("step", run.log.call_args.kwargs)
        self.assertIn("expand/r1/train/global_step", run.log.call_args.args[0])

        state = MagicMock()
        state.observations = [
            {
                "source": "acquisition",
                "decoded": f"w{i}",
                "score": 0.1,
                "sample_count": 1,
            }
            for i in range(110)
        ]
        session.log_observations(state.observations[-1], state)
        search_calls = [
            call
            for call in run.log.call_args_list
            if call.args and "search/verifications" in call.args[0]
        ]
        self.assertEqual(len(search_calls), 110)
        self.assertEqual(search_calls[-1].args[0]["search/verifications"], 110)
        for call in search_calls:
            self.assertNotIn("step", call.kwargs)

    def test_expand_progress_callback_omits_global_step(self):
        run = MagicMock()
        callback = make_expand_progress_callback(run, 2, phase="joint")
        callback(
            {
                "event": "eval_result",
                "epoch": 1.0,
                "step": 40,
                "avg_embed_sim": 0.5,
            }
        )
        self.assertEqual(run.log.call_count, 1)
        self.assertNotIn("step", run.log.call_args.kwargs)
        payload = run.log.call_args.args[0]
        self.assertEqual(payload["expand/r2/train/global_step"], 40.0)


if __name__ == "__main__":
    unittest.main()
