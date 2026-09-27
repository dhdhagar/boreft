from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from boreft.bo.state import Observation, RunState
from boreft.learn_bias import BatchLearnResult, GroupStopSpec, grouped_stop_triggered
from boreft.search_expand import (
    SearchExpandConfig,
    UniqueTarget,
    acquisition_batches_completed,
    expand_search_subspace_window,
    generate_target_definition,
    make_expansion_hook,
    partition_new_and_replay,
    remap_observation_points,
    replay_window_end,
    select_unique_subset,
    should_expand,
    strip_generated_definition,
    unique_targets_from_observations,
)
from boreft.task_config import (
    definition_generation_instruction,
    format_definition_generation_examples,
    task_supports_definition_generation,
)


def _obs(
    index: int,
    decoded: str,
    point: list[float] | None = None,
    *,
    source: str = "acquisition",
    bo_batch: int = 0,
) -> Observation:
    return Observation(
        index=index,
        point=point if point is not None else [float(index), 0.0],
        decoded=decoded,
        score=0.1 * (index + 1),
        source=source,
        components={"bo_batch": bo_batch} if source == "acquisition" else {},
    )


class SearchExpandConfigTest(unittest.TestCase):
    def test_disabled_by_default(self):
        cfg = SearchExpandConfig()
        self.assertFalse(cfg.enabled())
        self.assertTrue(cfg.learn_w)
        self.assertTrue(cfg.learn_r)
        self.assertTrue(cfg.learn_bias_network)
        cfg.validate()

    def test_rejects_all_plus_count(self):
        with self.assertRaisesRegex(ValueError, "all"):
            SearchExpandConfig(every=1, new_n=2).validate()

    def test_rejects_n_and_prop(self):
        with self.assertRaisesRegex(ValueError, "at most one"):
            SearchExpandConfig(
                every=1,
                new_strategy="random",
                new_n=2,
                new_prop=0.5,
            ).validate()

    def test_frac_requires_min(self):
        with self.assertRaisesRegex(ValueError, "stop_threshold_min"):
            SearchExpandConfig(every=1, stop_threshold_frac=0.5).validate()


class ShouldExpandTest(unittest.TestCase):
    def test_every_k_batches(self):
        self.assertFalse(should_expand(expand_every=0, n_batches=4, last_expanded_batches=0))
        self.assertFalse(should_expand(expand_every=3, n_batches=2, last_expanded_batches=0))
        self.assertTrue(should_expand(expand_every=3, n_batches=3, last_expanded_batches=0))
        self.assertFalse(should_expand(expand_every=3, n_batches=3, last_expanded_batches=3))
        self.assertTrue(should_expand(expand_every=3, n_batches=6, last_expanded_batches=3))

    def test_batch_count_from_observations(self):
        observations = [
            _obs(0, "a", source="warmstart"),
            _obs(1, "b", source="warmstart"),
            _obs(2, "c", bo_batch=0),
            _obs(3, "d", bo_batch=1),
        ]
        self.assertEqual(acquisition_batches_completed(observations), 2)
        self.assertEqual(acquisition_batches_completed(observations[:2]), 0)

    def test_first_round_replay_is_warmstart_prefix(self):
        observations = [
            _obs(0, "a", source="warmstart"),
            _obs(1, "b", source="warmstart"),
            _obs(2, "c", bo_batch=0),
        ]
        self.assertEqual(replay_window_end(observations, 0), 2)
        self.assertEqual(replay_window_end(observations, 3), 3)
        self.assertEqual(replay_window_end(observations, 99), 3)


class UniqueTargetTest(unittest.TestCase):
    def test_dedups_and_keeps_first_spelling(self):
        observations = [
            _obs(0, "Apple", source="warmstart"),
            _obs(1, "APPLE", bo_batch=0),
            _obs(2, "pear", bo_batch=0),
        ]
        unique = unique_targets_from_observations(observations, task="semantle")
        self.assertEqual([item.text for item in unique], ["Apple", "pear"])
        self.assertEqual(unique[0].observation_indices, (0, 1))

    def test_partition_treats_pre_window_as_replay(self):
        observations = [
            _obs(0, "hat", source="warmstart"),
            _obs(1, "dog", source="warmstart"),
            _obs(2, "cat", bo_batch=0),
            _obs(3, "hat", bo_batch=0),
        ]
        new, replay = partition_new_and_replay(
            observations, task="semantle", absorbed_through_index=2
        )
        self.assertEqual([item.text for item in replay], ["hat", "dog"])
        self.assertEqual([item.text for item in new], ["cat"])

    def test_partition_replays_checkpoint_train_set(self):
        observations = [
            _obs(0, "hat", point=[9.0, 9.0], source="warmstart"),
            _obs(1, "cat", bo_batch=0),
        ]
        train = [
            UniqueTarget(
                text="hat",
                key="hat",
                observation_indices=(),
                point=(0.0, 0.0),
                origin="train",
            ),
            UniqueTarget(
                text="dog",
                key="dog",
                observation_indices=(),
                point=(1.0, 0.0),
                origin="train",
            ),
        ]
        new, replay = partition_new_and_replay(
            observations,
            task="semantle",
            absorbed_through_index=1,
            train_targets=train,
        )
        self.assertEqual([item.text for item in replay], ["hat", "dog"])
        self.assertEqual([item.origin for item in replay], ["train", "train"])
        self.assertEqual(replay[0].point, (9.0, 9.0))
        self.assertEqual([item.text for item in new], ["cat"])

    def test_new_window_does_not_reabsorb_train_identities(self):
        observations = [
            _obs(0, "hat", source="warmstart"),
            _obs(1, "HAT", bo_batch=0),
        ]
        train = [
            UniqueTarget(
                text="hat",
                key="hat",
                observation_indices=(),
                point=(0.0, 0.0),
                origin="train",
            ),
        ]
        new, replay = partition_new_and_replay(
            observations,
            task="semantle",
            absorbed_through_index=1,
            train_targets=train,
        )
        self.assertEqual([item.text for item in new], [])
        self.assertEqual([item.text for item in replay], ["hat"])

    def test_select_all_keeps_order(self):
        items = unique_targets_from_observations(
            [_obs(0, "a"), _obs(1, "b"), _obs(2, "c")],
            task="semantle",
        )
        out = select_unique_subset(
            items,
            strategy="all",
            n=None,
            prop=None,
            seed=0,
            n_name="n",
            prop_name="prop",
            strategy_name="strategy",
        )
        self.assertEqual([item.text for item in out], ["a", "b", "c"])

    def test_select_random_subset(self):
        items = unique_targets_from_observations(
            [_obs(i, f"w{i}") for i in range(5)],
            task="semantle",
        )
        out = select_unique_subset(
            items,
            strategy="random",
            n=2,
            prop=None,
            seed=3,
            n_name="n",
            prop_name="prop",
            strategy_name="strategy",
        )
        self.assertEqual(len(out), 2)
        self.assertEqual([item.text for item in out], sorted(item.text for item in out))


class RemapPointsTest(unittest.TestCase):
    def test_rewrites_matching_identities(self):
        observations = [
            _obs(0, "hat", point=[0.0, 0.0], source="warmstart"),
            _obs(1, "HAT", point=[1.0, 1.0], bo_batch=0),
            _obs(2, "dog", point=[2.0, 2.0], bo_batch=0),
        ]
        updated = remap_observation_points(
            observations,
            task="semantle",
            key_to_mu={"hat": np.array([9.0, 8.0], dtype=np.float32)},
        )
        self.assertEqual(updated[0].point, [9.0, 8.0])
        self.assertEqual(updated[1].point, [9.0, 8.0])
        self.assertEqual(updated[2].point, [2.0, 2.0])
        self.assertEqual(updated[0].decoded, "hat")
        self.assertEqual(updated[0].score, observations[0].score)


class GroupedStopTest(unittest.TestCase):
    def test_and_combines_groups(self):
        replay = GroupStopSpec(stop_threshold=0.9)
        new = GroupStopSpec(stop_threshold=0.8)
        self.assertFalse(
            grouped_stop_triggered(
                global_mean=0.85,
                global_min=0.5,
                global_sims=[0.85],
                stop_threshold=None,
                stop_threshold_min=None,
                stop_threshold_frac=1.0,
                group_slices={
                    "replay": (0.95, 0.9, [0.95]),
                    "new": (0.7, 0.6, [0.7]),
                },
                stop_groups={"replay": replay, "new": new},
            )
        )
        self.assertTrue(
            grouped_stop_triggered(
                global_mean=0.85,
                global_min=0.5,
                global_sims=[0.85],
                stop_threshold=None,
                stop_threshold_min=None,
                stop_threshold_frac=1.0,
                group_slices={
                    "replay": (0.95, 0.9, [0.95]),
                    "new": (0.85, 0.8, [0.85]),
                },
                stop_groups={"replay": replay, "new": new},
            )
        )

    def test_global_and_groups(self):
        self.assertFalse(
            grouped_stop_triggered(
                global_mean=0.7,
                global_min=0.5,
                global_sims=[0.7],
                stop_threshold=0.9,
                stop_threshold_min=None,
                stop_threshold_frac=1.0,
                group_slices={"new": (0.95, 0.9, [0.95])},
                stop_groups={"new": GroupStopSpec(stop_threshold=0.8)},
            )
        )
        self.assertTrue(
            grouped_stop_triggered(
                global_mean=0.92,
                global_min=0.9,
                global_sims=[0.92],
                stop_threshold=0.9,
                stop_threshold_min=None,
                stop_threshold_frac=1.0,
                group_slices={"new": (0.95, 0.9, [0.95])},
                stop_groups={"new": GroupStopSpec(stop_threshold=0.8)},
            )
        )

    def test_perfect_global_always_stops(self):
        self.assertTrue(
            grouped_stop_triggered(
                global_mean=1.0,
                global_min=1.0,
                global_sims=[1.0],
                stop_threshold=None,
                stop_threshold_min=None,
                stop_threshold_frac=1.0,
                group_slices={"new": (0.1, 0.1, [0.1])},
                stop_groups={"new": GroupStopSpec(stop_threshold=0.9)},
            )
        )


class DefinitionPromptTest(unittest.TestCase):
    def test_semantle_icl_block(self):
        self.assertTrue(task_supports_definition_generation("semantle"))
        examples = format_definition_generation_examples(
            "semantle", [("apple", "a fruit"), ("dog", "a mammal")]
        )
        prompt = definition_generation_instruction(
            "semantle", "cat", examples=[("apple", "a fruit")]
        )
        self.assertIn("Word: apple", examples)
        self.assertIn("Definition: a fruit", examples)
        self.assertIn("Word: cat", prompt)
        self.assertIn("Word: apple", prompt)
        self.assertNotIn("{text}", prompt)
        self.assertNotIn("{examples}", prompt)

    def test_zero_shot_omits_empty_example_block(self):
        prompt = definition_generation_instruction("semantle", "cat")
        self.assertIn("Word: cat", prompt)
        self.assertNotIn("{examples}", prompt)

    def test_molopt_and_hypogen_templates_exist(self):
        self.assertTrue(task_supports_definition_generation("molopt"))
        self.assertTrue(task_supports_definition_generation("hypogen"))
        mol = definition_generation_instruction("molopt", "CCO")
        hypo = definition_generation_instruction(
            "hypogen", "Temperature predicts diversification."
        )
        self.assertIn("CCO", mol)
        self.assertIn("Temperature predicts diversification.", hypo)


class ExpansionHistoryLogsTest(unittest.TestCase):
    def test_flattens_eval_groups(self):
        from boreft.search_wandb import expansion_history_logs as wandb_logs

        record = {
            "round": 2,
            "phases": [
                {
                    "phase": "joint",
                    "eval_history": [
                        {
                            "epoch": 5,
                            "step": 20,
                            "avg_embed_sim": 0.8,
                            "min_embed_sim": 0.7,
                            "n_recovered": 2,
                            "n_targets": 4,
                            "groups": {
                                "new": {
                                    "avg_embed_sim": 0.9,
                                    "min_embed_sim": 0.85,
                                    "n_recovered": 1,
                                    "n_targets": 1,
                                }
                            },
                        }
                    ],
                }
            ],
        }
        rows = wandb_logs(record)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["expand/r2/train/global_step"], 20.0)
        self.assertAlmostEqual(rows[0]["expand/r2/eval/avg_embed_sim"], 0.8)
        self.assertAlmostEqual(rows[0]["expand/r2/eval/new/avg_embed_sim"], 0.9)
        self.assertAlmostEqual(rows[0]["expand/r2/eval/recover_rate"], 0.5)

    def test_new_only_phase_uses_nested_prefix(self):
        from boreft.search_wandb import expansion_history_logs as wandb_logs

        record = {
            "round": 1,
            "phases": [
                {
                    "phase": "new_only",
                    "eval_history": [
                        {
                            "epoch": 1,
                            "step": 3,
                            "avg_embed_sim": 0.4,
                            "n_recovered": 0,
                            "n_targets": 1,
                        }
                    ],
                },
                {
                    "phase": "joint",
                    "eval_history": [
                        {
                            "epoch": 2,
                            "step": 8,
                            "avg_embed_sim": 0.6,
                            "n_recovered": 1,
                            "n_targets": 2,
                        }
                    ],
                },
            ],
        }
        rows = wandb_logs(record)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["expand/r1/new_only/train/global_step"], 3.0)
        self.assertAlmostEqual(rows[0]["expand/r1/new_only/eval/avg_embed_sim"], 0.4)
        self.assertEqual(rows[1]["expand/r1/train/global_step"], 8.0)
        self.assertAlmostEqual(rows[1]["expand/r1/eval/avg_embed_sim"], 0.6)


class DefinitionGenerationTest(unittest.TestCase):
    def test_chat_checkpoints_wrap_user_content(self):
        ckpt = MagicMock()
        ckpt.from_chat_template = True
        ckpt.tokenizer = object()
        ckpt.assistant_suffix = None
        ckpt.reft_model.model = object()
        with (
            patch(
                "boreft.data_utils.chat_prompt", return_value="WRAPPED"
            ) as wrapped,
            patch(
                "boreft.search_expand.generate_base", return_value=" a feline "
            ) as gen,
        ):
            text = generate_target_definition(
                ckpt,
                "cat",
                task="semantle",
                examples=[],
                max_new_tokens=16,
            )
        wrapped.assert_called_once()
        self.assertEqual(gen.call_args.args[2], "WRAPPEDDefinition: ")
        self.assertTrue(gen.call_args.kwargs["from_chat_template"])
        self.assertEqual(text, "a feline")

    def test_strips_leading_definition_label(self):
        self.assertEqual(
            strip_generated_definition("Definition: a small feline", task="semantle"),
            "a small feline",
        )
        self.assertEqual(
            strip_generated_definition("**Description:** a solvent", task="molopt"),
            "a solvent",
        )
        self.assertEqual(
            strip_generated_definition(
                "Research objective: measure temperature",
                task="hypogen",
            ),
            "measure temperature",
        )
        self.assertEqual(
            strip_generated_definition(
                "Word: cat\nDefinition: a small feline",
                task="semantle",
            ),
            "a small feline",
        )
        self.assertEqual(
            strip_generated_definition("a small feline", task="semantle"),
            "a small feline",
        )
        self.assertEqual(
            strip_generated_definition(
                "A set of rotating blades that drive a craft through air or water.\n\nWord: lif",
                task="semantle",
            ),
            "A set of rotating blades that drive a craft through air or water.",
        )
        self.assertEqual(
            strip_generated_definition(
                "A young girl who attends school.\n\nWord",
                task="semantle",
            ),
            "A young girl who attends school.",
        )
        self.assertEqual(
            strip_generated_definition(
                "Definition: a tiny semiconductor component that switches current\n"
                "Word: foo\n"
                "Definition: a small enclosed area for",
                task="semantle",
            ),
            "a tiny semiconductor component that switches current",
        )

    def test_generate_strips_chat_definition_prefix(self):
        ckpt = MagicMock()
        ckpt.from_chat_template = True
        ckpt.tokenizer = object()
        ckpt.assistant_suffix = None
        ckpt.reft_model.model = object()
        with (
            patch("boreft.data_utils.chat_prompt", return_value="WRAPPED"),
            patch(
                "boreft.search_expand.generate_base",
                return_value="Definition: a feline",
            ),
        ):
            text = generate_target_definition(
                ckpt,
                "cat",
                task="semantle",
                examples=[],
                max_new_tokens=16,
            )
        self.assertEqual(text, "a feline")


class ExpansionWindowTest(unittest.TestCase):
    def test_predicted_init_collects_definitions_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            state.append(_obs(0, "hat", point=[0.0, 0.0], source="warmstart"))
            state.append(_obs(1, "cat", point=[1.0, 1.0], bo_batch=0))
            order: list[str] = []

            def collect(_ckpt, targets, **_kwargs):
                order.append("collect")
                return ({target: f"def-{target}" for target in targets}, {})

            def predict(_ckpt, targets, definitions=None):
                order.append("predict")
                self.assertEqual(list(targets), ["cat"])
                self.assertEqual(definitions["cat"], "def-cat")
                return np.ones((len(targets), 2), dtype=np.float32), None

            def learn(_ckpt, **kwargs):
                order.append("learn")
                targets = list(kwargs["targets"])
                mu = np.asarray(kwargs["warm_start_mu"], dtype=np.float32)
                return BatchLearnResult(
                    targets=targets,
                    mu=mu,
                    mu_pred=mu,
                    logvar=None,
                    bias_dim=2,
                    steps=1,
                    mean_train_loss=0.0,
                )

            ckpt = MagicMock()
            ckpt.saved_cfg = {
                "task": "semantle",
                "lambda_sdpo": 1.0,
                "bias_input_source": "embed_cache",
            }
            bounds = np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float32)
            with (
                patch("boreft.search_expand.collect_definitions", side_effect=collect),
                patch(
                    "boreft.search_expand.predict_bias_network_rows",
                    side_effect=predict,
                ),
                patch(
                    "boreft.search_expand.learn_biases_batched", side_effect=learn
                ),
                patch(
                    "boreft.search_expand.snapshot_search_intervention",
                    return_value={},
                ),
                patch("boreft.search_expand.restore_search_intervention"),
            ):
                expand_search_subspace_window(
                    ckpt=ckpt,
                    state=state,
                    config=SearchExpandConfig(every=1, init="predicted"),
                    bounds=bounds,
                    task="semantle",
                    seed=0,
                    round_index=1,
                    absorbed_through_index=1,
                )
            self.assertEqual(order[0], "collect")
            self.assertEqual(order[1], "predict")
            self.assertIn("learn", order)

    def test_hook_uses_warmstart_prefix_on_first_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            state.append(_obs(0, "a", source="warmstart"))
            state.append(_obs(1, "b", source="warmstart"))
            state.append(_obs(2, "c", bo_batch=0))
            captured: dict = {}

            def fake_window(**kwargs):
                captured["absorbed"] = kwargs["absorbed_through_index"]
                return (
                    state,
                    np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
                    {"skipped": True, "n_new": 0, "n_replay": 2},
                )

            hook = make_expansion_hook(
                ckpt=object(),
                config=SearchExpandConfig(every=1),
                task="semantle",
                seed=0,
                bounds=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
                bounds_padding=0.0,
                seed_dir=tmp,
            )
            with patch(
                "boreft.search_expand.expand_search_subspace_window",
                side_effect=lambda **kwargs: fake_window(**kwargs),
            ):
                hook(state)
            self.assertEqual(captured["absorbed"], 2)

    def test_hook_forwards_aabb_std_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            state.append(_obs(0, "a", source="warmstart"))
            state.append(_obs(1, "b", bo_batch=0))
            captured: dict = {}

            def fake_window(**kwargs):
                captured["aabb_std_k"] = kwargs["aabb_std_k"]
                return (
                    state,
                    np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
                    {"skipped": True, "n_new": 0, "n_replay": 1},
                )

            hook = make_expansion_hook(
                ckpt=object(),
                config=SearchExpandConfig(every=1),
                task="semantle",
                seed=0,
                bounds=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
                bounds_padding=0.0,
                seed_dir=tmp,
                aabb_std_k=1.5,
            )
            with patch(
                "boreft.search_expand.expand_search_subspace_window",
                side_effect=lambda **kwargs: fake_window(**kwargs),
            ):
                hook(state)
            self.assertEqual(captured["aabb_std_k"], 1.5)

    def test_hook_forwards_search_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            state.append(_obs(0, "a", source="warmstart"))
            state.append(_obs(1, "b", bo_batch=0))
            captured: dict = {}

            def fake_window(**kwargs):
                captured["search_domain"] = kwargs["search_domain"]
                return (
                    state,
                    np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
                    {"skipped": True, "n_new": 0, "n_replay": 1},
                )

            hook = make_expansion_hook(
                ckpt=object(),
                config=SearchExpandConfig(every=1),
                task="semantle",
                seed=0,
                bounds=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
                bounds_padding=0.0,
                seed_dir=tmp,
                search_domain="ellipsoid",
            )
            with patch(
                "boreft.search_expand.expand_search_subspace_window",
                side_effect=lambda **kwargs: fake_window(**kwargs),
            ):
                hook(state)
            self.assertEqual(captured["search_domain"], "ellipsoid")


class LiveWandbObservationHookTest(unittest.TestCase):
    def test_log_observations_accepts_runner_args(self):
        from boreft.search_wandb import SearchWandbSession

        session = SearchWandbSession(
            run=MagicMock(),
            target="cat",
            budget=10,
            seed_dir=".",
        )
        obs = _obs(0, "hat", source="warmstart")
        state = MagicMock()
        state.observations = [obs]
        captured: dict = {}

        def fake_history(observations, *_args, **_kwargs):
            captured["n"] = len(list(observations))
            return []

        with patch(
            "boreft.search_wandb.verification_history", side_effect=fake_history
        ):
            session.log_observations(obs, state)
        self.assertEqual(captured["n"], 1)


class RunStateRewriteTest(unittest.TestCase):
    def test_rewrite_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observations.jsonl"
            state = RunState(path)
            state.append(_obs(0, "a", source="warmstart"))
            state.append(_obs(1, "b", bo_batch=0))
            remapped = remap_observation_points(
                state.observations,
                task="semantle",
                key_to_mu={"a": np.array([3.0, 4.0], dtype=np.float32)},
            )
            state.rewrite(remapped)
            loaded = RunState.load(path)
            self.assertEqual(loaded.observations[0].point, [3.0, 4.0])
            self.assertEqual(loaded.observations[1].decoded, "b")

    def test_rewrite_rejects_length_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            state.append(_obs(0, "a", source="warmstart"))
            with self.assertRaisesRegex(ValueError, "number of observations"):
                state.rewrite([])


class OnAfterBatchHookTest(unittest.TestCase):
    def test_run_bo_applies_updated_bounds(self):
        from boreft.bo.runner import BOConfig, run_bo
        from boreft.search import Verification

        class Numeric:
            maximum = None

            def __call__(self, decoded):
                return Verification(score=float(decoded), components={})

        captured = []

        def on_after_batch(state):
            captured.append(len(state.observations))
            return np.array([[-1.0], [2.0]], dtype=np.float32)

        class FakeSurrogate:
            def metadata(self):
                return {"kind": "static"}

            def checkpoint(self):
                return {}

        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp) / "observations.jsonl")
            with (
                patch("boreft.bo.runner.fit_surrogate", return_value=FakeSurrogate()) as fit,
                patch(
                    "boreft.bo.runner.propose_candidates",
                    return_value=np.array([[0.6]]),
                ),
            ):
                run_bo(
                    seed=1,
                    bounds=np.array([[0.0], [1.0]]),
                    warmstart_points=[[0.1], [0.2]],
                    decode=lambda point: str(float(point[0])),
                    verifier=Numeric(),
                    state=state,
                    config=BOConfig(budget=4, observation_samples=1),
                    on_after_batch=on_after_batch,
                )
            self.assertGreaterEqual(len(captured), 1)
            # Second GP fit (if any) should see the updated bounds.
            if fit.call_count >= 2:
                np.testing.assert_allclose(
                    fit.call_args_list[1].args[2],
                    np.array([[-1.0], [2.0]], dtype=np.float32),
                )


if __name__ == "__main__":
    unittest.main()
