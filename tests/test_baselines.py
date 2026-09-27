from __future__ import annotations

import hashlib
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import tyro

from boreft.bo.runner import seed_everything
from boreft.search import SearchConfig, select_warmstarts

from boreft.baselines import (
    ArtifactLayout,
    AutoDiscoveryConfig,
    Baseline,
    BaselineLoopConfig,
    BaselineObservation,
    BaselineRunState,
    BOPROConfig,
    Candidate,
    DiscreteBOBaseline,
    DiscreteBOConfig,
    GenerationOptions,
    OPROConfig,
    SDPOTTTBaseline,
    SDPOTTTConfig,
    MiGrATeBaseline,
    MiGrATeConfig,
    RandomSamplingConfig,
    ScoreResult,
    SearchContext,
    WarmstartSeed,
    available_baselines,
    create_baseline,
    load_method_state,
    run_baseline,
    solution_key,
    timed_score,
)
from boreft.baselines.autodiscovery import (
    MCTSNode,
    build_branch_prompt,
    can_expand,
    path_from_root,
    pick_untried,
    select_expandable,
    ucb1,
)
from boreft.baselines.opro import _build_prompt
from boreft.baselines.llm_sample import (
    build_candidate_prompt,
    build_completion_fewshot,
    parse_generated_candidates,
    recent_incontext,
    sample_llm_candidates,
)
from boreft.chem import wrap_mist_smiles_tags
from boreft.task_config import task_instruction
from boreft.baselines.discrete_bo import _train_item_solution
from boreft.baselines.sdpo_ttt import (
    LoRALinear,
    LoRAStudentRuntime,
    _normalize_generated_text,
    build_teacher_prompt,
    clear_checkpoint_lora,
)
from boreft.baselines.migrate import (
    _grpo_loss,
    build_neighborhood_prompt,
    select_greedy,
)
from boreft.baselines.random_sampling import _BLANK_SAMPLE_RETRIES
from boreft.baselines.search import (
    BaselineSearchConfig,
    SearchBackend,
    _base_generate,
    _normalize_generated_text as _search_normalize_generated_text,
    _previous_elapsed,
    _resolved_seeds,
    _warmstart_seeds,
    aggregate_summaries,
    artifact_manifest,
    found_target,
    found_target_stats,
    generation_options,
    method_config,
    prepare_warmstarts,
    resolved_config,
    run_search,
    _print_finish_summary,
    _validate_resume_config,
)


EXPECTED_BASELINES = (
    "random_sampling",
    "discrete_bo",
    "opro",
    "sdpo_ttt",
    "bopro",
    "migrate",
    "autodiscovery",
)
IMPLEMENTED_BASELINES = (
    "random_sampling",
    "discrete_bo",
    "opro",
    "sdpo_ttt",
    "bopro",
    "migrate",
    "autodiscovery",
)
STUB_BASELINES = tuple(
    name for name in EXPECTED_BASELINES if name not in IMPLEMENTED_BASELINES
)


def _context(generate, score, **overrides) -> SearchContext:
    fields = {
        "task_description": "Generate a word.",
        "target": "target",
        "objective": "embed_sim",
        "seed": 1,
        "generate": generate,
        "score": score,
    }
    return SearchContext(**{**fields, **overrides})


def _mist(smiles: str) -> str:
    return wrap_mist_smiles_tags(smiles, include_open=True)


class _RecordingBaseline:
    """Minimal baseline used to exercise the shared loop without a model."""

    name = "recording"

    def __init__(self, solutions):
        self.solutions = list(solutions)
        self.requested: list[int] = []
        self.observed: list[str] = []
        self.restored: dict | None = None

    def propose(self, context, history, count):
        del context, history
        self.requested.append(count)
        batch = self.solutions[:count]
        del self.solutions[:count]
        return [Candidate(solution=solution) for solution in batch]

    def observe(self, observations):
        self.observed.extend(item.solution for item in observations)

    def state_dict(self):
        return {"observed": list(self.observed)}

    def load_state_dict(self, state):
        self.restored = dict(state)
        self.observed = list(state.get("observed", []))


def _fake_score(solution: str) -> ScoreResult:
    """Only the literal target reaches the maximum, so runs spend their budget."""
    if solution == "exact":
        return ScoreResult(score=1.0, components={"valid": True})
    return ScoreResult(
        score=0.5 if solution.startswith("g") else 0.05,
        components={"valid": True},
    )


def _fake_embed(texts):
    rows = []
    for text in texts:
        digest = hashlib.md5(str(text).encode("utf-8")).digest()
        vec = np.frombuffer(digest, dtype=np.uint8).astype(np.float64)
        rows.append((vec / np.linalg.norm(vec)).tolist())
    return rows


def _fake_backend(**overrides) -> SearchBackend:
    generated = iter(f"g{index}" for index in range(500))

    def generate(_prompt, count, _options):
        return [next(generated) for _ in range(count)]

    def decode_point(point):
        value = float(np.asarray(point).reshape(-1)[0])
        return f"w{int(round(value * 1000))}"

    fields = {
        "task": "semantle",
        "model_name": "fake/model",
        "checkpoint": None,
        "generate": generate,
        "score": _fake_score,
        "decode_point": decode_point,
        "train_mu": np.zeros((8, 1), dtype=np.float32),
        "bounds": np.array([[0.0], [1.0]], dtype=np.float32),
        "maximum": 1.0,
        "embed": _fake_embed,
    }
    return SearchBackend(**{**fields, **overrides})


def _run_config(tmp: str, **overrides) -> BaselineSearchConfig:
    fields = {
        "task_description": "Generate a word.",
        "target": "exact",
        "reft_output_dir": str(Path(tmp) / "checkpoint"),
        "search_dir": str(Path(tmp) / "search"),
        "warmstart_source": "sobol",
        "warmstart_count": 2,
        "budget": 5,
    }
    return BaselineSearchConfig(**{**fields, **overrides})


class BaselineRegistryTest(unittest.TestCase):
    def test_generation_options_disable_nucleus_sampling(self):
        self.assertEqual(GenerationOptions(temperature=0.5).top_p, 1.0)

    def test_all_baselines_are_registered_and_importable(self):
        self.assertEqual(available_baselines(), EXPECTED_BASELINES)
        for name in EXPECTED_BASELINES:
            baseline = create_baseline(name)
            self.assertIsInstance(baseline, Baseline)
            self.assertEqual(baseline.name, name)

    def test_unknown_baseline_lists_choices(self):
        with self.assertRaisesRegex(ValueError, "random_sampling"):
            create_baseline("unknown")

    def test_registry_accepts_method_specific_config(self):
        config = RandomSamplingConfig()
        baseline = create_baseline("random_sampling", config)
        self.assertIs(baseline.config, config)
        with self.assertRaisesRegex(TypeError, "RandomSamplingConfig"):
            create_baseline("random_sampling", object())

    def test_each_stub_has_a_method_specific_failure(self):
        context = _context(
            lambda _prompt, _count, _options: [],
            lambda _solution: ScoreResult(0.0),
        )
        for name in STUB_BASELINES:
            with self.subTest(name=name):
                baseline = create_baseline(name)
                with self.assertRaisesRegex(NotImplementedError, name):
                    baseline.propose(context, [], 1)


class BaselineStateTest(unittest.TestCase):
    def test_observations_round_trip_and_summarize(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observations.jsonl"
            state = BaselineRunState(path)
            first = state.append(
                BaselineObservation(
                    index=0,
                    solution="first",
                    score=0.2,
                    candidate_metadata={
                        "prompt": "task-only",
                        "neighbor_indices": [1, 2],
                    },
                    elapsed_seconds=0.4,
                    bbox_eval_seconds=0.1,
                    phase="warmstart",
                    seed=3,
                )
            )
            second = state.append(
                BaselineObservation(
                    index=1,
                    solution="second",
                    score=0.7,
                    components={"valid": True},
                    elapsed_seconds=0.6,
                    bbox_eval_seconds=0.2,
                    is_repeat_sample=True,
                    is_repeat_proposal=True,
                    seed=3,
                )
            )
            self.assertEqual(first.best_so_far, 0.2)
            self.assertEqual(second.best_so_far, 0.7)

            loaded = BaselineRunState.load(path)
            self.assertEqual(
                [item.solution for item in loaded.observations],
                ["first", "second"],
            )
            self.assertEqual(loaded.summary()["best_solution"], "second")
            summary = loaded.summary(elapsed_seconds=1.5)
            self.assertEqual(summary["elapsed_seconds"], 1.5)
            self.assertAlmostEqual(summary["bbox_eval_seconds"], 0.3)
            self.assertEqual(summary["n_repeat_samples"], 1)
            self.assertEqual(summary["n_repeat_proposals"], 1)
            self.assertEqual(summary["n_warmstart"], 1)
            self.assertEqual(summary["n_acquired"], 1)
            self.assertEqual(
                summary["n_observations"],
                summary["n_warmstart"] + summary["n_acquired"],
            )

    def test_observation_rejects_unknown_phase(self):
        with self.assertRaisesRegex(ValueError, "phase"):
            BaselineObservation(
                index=0,
                solution="candidate",
                score=0.1,
                phase="unknown",
            )

    def test_timed_score_returns_bbox_duration(self):
        with patch(
            "boreft.baselines.base.time.monotonic",
            side_effect=[2.0, 2.25],
        ):
            result, elapsed = timed_score(lambda _: ScoreResult(0.4), "candidate")
        self.assertEqual(result.score, 0.4)
        self.assertEqual(elapsed, 0.25)

    def test_load_repairs_torn_final_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observations.jsonl"
            state = BaselineRunState(path)
            state.append(
                BaselineObservation(index=0, solution="first", score=0.2)
            )
            with path.open("a", encoding="utf-8") as handle:
                handle.write('{"index": 1, "solution":')
            loaded = BaselineRunState.load(path)
            self.assertEqual(len(loaded.observations), 1)
            loaded.append(
                BaselineObservation(index=1, solution="second", score=0.3)
            )
            self.assertEqual(len(BaselineRunState.load(path).observations), 2)


class BaselineSearchTest(unittest.TestCase):
    def test_config_validation_and_seed_resolution(self):
        config = BaselineSearchConfig(
            task_description="Find a word.",
            target="apple",
            reft_output_dir="checkpoint",
            seeds=(4,),
            repeats=3,
        )
        config.validate()
        self.assertEqual(_resolved_seeds(config), (4, 5, 6))

        both = BaselineSearchConfig(
            task_description="Find a word.",
            target="apple",
            reft_output_dir="checkpoint",
            resume=True,
            overwrite=True,
        )
        both.validate()
        self.assertTrue(both.overwrite)
        self.assertFalse(both.resume)

        with self.assertRaisesRegex(ValueError, "batch_size"):
            BaselineSearchConfig(
                task_description="Find a word.",
                target="apple",
                reft_output_dir="checkpoint",
                budget=2,
                warmstart_count=2,
                batch_size=3,
            ).validate()
        BaselineSearchConfig(
            task_description="Find a word.",
            target="apple",
            reft_output_dir="checkpoint",
            budget=2,
            warmstart_count=2,
            observation_samples=5,
        ).validate()
        with self.assertRaisesRegex(ValueError, "warmstart verification cost"):
            BaselineSearchConfig(
                task_description="Find a word.",
                target="apple",
                reft_output_dir="checkpoint",
                budget=1,
                warmstart_count=2,
            ).validate()
        with self.assertRaisesRegex(ValueError, "warmstart verification cost"):
            BaselineSearchConfig(
                task_description="Find a word.",
                target="apple",
                reft_output_dir="checkpoint",
                budget=5,
                warmstart_count=2,
                observation_samples=3,
                warmstart_source="sobol",
            ).validate()

    def test_property_oracle_fills_target_and_rejects_semantle(self):
        config = BaselineSearchConfig(
            task_description="Generate a SMILES string.",
            target="",
            reft_output_dir="checkpoint",
            task="molopt",
            oracle="jnk3",
            warmstart_source="sobol",
            warmstart_count=2,
            budget=4,
        )
        config.validate()
        self.assertEqual(config.oracle, "JNK3")
        self.assertEqual(config.target, "JNK3")
        dual = BaselineSearchConfig(
            task_description="Generate a SMILES string.",
            target="",
            reft_output_dir="checkpoint",
            task="molopt",
            oracle="gsk3b*jnk3",
            warmstart_source="sobol",
            warmstart_count=2,
            budget=4,
        )
        dual.validate()
        self.assertEqual(dual.oracle, "GSK3B_JNK3")
        self.assertEqual(dual.target, "GSK3B_JNK3")
        with self.assertRaisesRegex(ValueError, "task=molopt"):
            BaselineSearchConfig(
                task_description="Find a word.",
                target="apple",
                reft_output_dir="checkpoint",
                oracle="DRD2",
            ).validate()

    def test_artifact_manifest_matches_main_search_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            search_root = Path(tmp) / "search"
            config = BaselineSearchConfig(
                task_description="Find a word.",
                target="apple",
                reft_output_dir=str(Path(tmp) / "checkpoint"),
                search_dir=str(search_root),
                seeds=(2,),
            )
            manifest = artifact_manifest(config)
            root = search_root.resolve()
            self.assertEqual(manifest["config"], root / "config.json")
            self.assertEqual(
                manifest["seeds"][2]["observations"],
                root / "seed_2" / "observations.jsonl",
            )
            self.assertEqual(
                manifest["seeds"][2]["method_state"],
                root / "seed_2" / "method_state",
            )

    def test_resolved_config_includes_method_defaults(self):
        config = BaselineSearchConfig(
            task_description="Find a word.",
            target="apple",
            reft_output_dir="checkpoint",
            baseline="bopro",
        )
        resolved = resolved_config(config)
        self.assertEqual(resolved["method_config"]["neighbors"], 5)
        self.assertEqual(resolved["candidate_top_p"], 1.0)
        self.assertNotIn("random_sampling", resolved)

    def test_selected_method_config_is_used(self):
        custom = BOPROConfig(neighbors=9)
        config = BaselineSearchConfig(
            task_description="Find a word.",
            target="apple",
            reft_output_dir="checkpoint",
            baseline="bopro",
            bopro=custom,
        )
        self.assertIs(method_config(config), custom)
        self.assertEqual(resolved_config(config)["method_config"]["neighbors"], 9)
        self.assertEqual(generation_options(config).top_p, 1.0)

    def test_method_config_is_exposed_through_cli(self):
        config = tyro.cli(
            BaselineSearchConfig,
            args=[
                "--task-description",
                "Find a word.",
                "--target",
                "apple",
                "--reft-output-dir",
                "checkpoint",
                "--baseline",
                "bopro",
                "--bopro.neighbors",
                "9",
                "--bopro.surrogate",
                "projected",
                "--bopro.use-ard",
                "--bopro.projection-layers",
                "2",
            ],
        )
        self.assertEqual(config.bopro.neighbors, 9)
        self.assertEqual(config.bopro.surrogate, "projected")
        self.assertTrue(config.bopro.use_ard)
        self.assertEqual(config.bopro.projection_layers, 2)

    def test_lora_sft_adapter_is_exposed_through_cli(self):
        config = tyro.cli(
            BaselineSearchConfig,
            args=[
                "--task-description",
                "Find a word.",
                "--target",
                "apple",
                "--reft-output-dir",
                "checkpoint",
                "--lora-sft-adapter",
                "adapters/epoch009.pt",
            ],
        )
        self.assertEqual(config.lora_sft_adapter, "adapters/epoch009.pt")

    def test_discrete_bo_config_is_exposed_through_cli(self):
        config = tyro.cli(
            BaselineSearchConfig,
            args=[
                "--task-description",
                "Find a word.",
                "--target",
                "apple",
                "--reft-output-dir",
                "checkpoint",
                "--baseline",
                "discrete_bo",
                "--discrete-bo.candidate-count",
                "8",
                "--discrete-bo.include-target",
                "--discrete-bo.acquisition",
                "ucb",
                "--discrete-bo.candidate-source",
                "train",
            ],
        )
        self.assertEqual(config.discrete_bo.candidate_count, 8)
        self.assertTrue(config.discrete_bo.include_target)
        self.assertEqual(config.discrete_bo.acquisition, "ucb")
        self.assertEqual(config.discrete_bo.candidate_source, "train")
        self.assertEqual(config.discrete_bo.last_k_incontext, 0)
        self.assertEqual(config.discrete_bo.candidates_per_call, 1)
        random_pool = tyro.cli(
            BaselineSearchConfig,
            args=[
                "--task-description",
                "Find a word.",
                "--target",
                "apple",
                "--reft-output-dir",
                "checkpoint",
                "--baseline",
                "discrete_bo",
                "--discrete-bo.acquisition",
                "random",
            ],
        )
        self.assertEqual(random_pool.discrete_bo.acquisition, "random")

    def test_llm_sample_knobs_are_exposed_through_cli(self):
        random_config = tyro.cli(
            BaselineSearchConfig,
            args=[
                "--task-description",
                "Find a word.",
                "--target",
                "apple",
                "--reft-output-dir",
                "checkpoint",
                "--baseline",
                "random_sampling",
                "--random-sampling.last-k-incontext",
                "6",
                "--random-sampling.candidates-per-call",
                "5",
            ],
        )
        self.assertEqual(random_config.random_sampling.last_k_incontext, 6)
        self.assertEqual(random_config.random_sampling.candidates_per_call, 5)
        default_random = tyro.cli(
            BaselineSearchConfig,
            args=[
                "--task-description",
                "Find a word.",
                "--target",
                "apple",
                "--reft-output-dir",
                "checkpoint",
                "--baseline",
                "random_sampling",
            ],
        )
        self.assertEqual(default_random.random_sampling.last_k_incontext, 20)
        self.assertEqual(default_random.random_sampling.candidates_per_call, 1)
        discrete_config = tyro.cli(
            BaselineSearchConfig,
            args=[
                "--task-description",
                "Find a word.",
                "--target",
                "apple",
                "--reft-output-dir",
                "checkpoint",
                "--baseline",
                "discrete_bo",
                "--discrete-bo.last-k-incontext",
                "4",
                "--discrete-bo.candidates-per-call",
                "8",
            ],
        )
        self.assertEqual(discrete_config.discrete_bo.last_k_incontext, 4)
        self.assertEqual(discrete_config.discrete_bo.candidates_per_call, 8)

    def test_sdpo_ttt_knobs_are_exposed_through_cli(self):
        config = tyro.cli(
            BaselineSearchConfig,
            args=[
                "--task-description",
                "Find a word.",
                "--target",
                "apple",
                "--reft-output-dir",
                "checkpoint",
                "--baseline",
                "sdpo_ttt",
                "--sdpo-ttt.n-onpolicy",
                "6",
                "--sdpo-ttt.teacher-policy",
                "ema",
                "--sdpo-ttt.lora-rank",
                "8",
                "--sdpo-ttt.last-k-incontext",
                "5",
                "--sdpo-ttt.last-k-strategy",
                "best",
            ],
        )
        self.assertEqual(config.sdpo_ttt.n_onpolicy, 6)
        self.assertEqual(config.sdpo_ttt.teacher_policy, "ema")
        self.assertEqual(config.sdpo_ttt.lora_rank, 8)
        self.assertEqual(config.sdpo_ttt.divergence, "reverse_kl")
        self.assertEqual(config.sdpo_ttt.last_k_incontext, 5)
        self.assertEqual(config.sdpo_ttt.last_k_strategy, "best")

    def test_migrate_knobs_are_exposed_through_cli(self):
        config = tyro.cli(
            BaselineSearchConfig,
            args=[
                "--task-description",
                "Find a word.",
                "--target",
                "apple",
                "--reft-output-dir",
                "checkpoint",
                "--baseline",
                "migrate",
                "--migrate.on-policy-count",
                "3",
                "--migrate.greedy-count",
                "2",
                "--migrate.neighborhood-count",
                "1",
                "--migrate.greedy-topk",
                "4",
                "--migrate.lora-rank",
                "8",
            ],
        )
        self.assertEqual(config.migrate.on_policy_count, 3)
        self.assertEqual(config.migrate.greedy_count, 2)
        self.assertEqual(config.migrate.neighborhood_count, 1)
        self.assertEqual(config.migrate.greedy_topk, 4)
        self.assertEqual(config.migrate.lora_rank, 8)
        self.assertEqual(config.migrate.clip_epsilon_high, 0.28)

    def test_discrete_bo_train_accepts_minus_one_candidate_count(self):
        config = tyro.cli(
            BaselineSearchConfig,
            args=[
                "--task-description",
                "Find a word.",
                "--target",
                "apple",
                "--reft-output-dir",
                "checkpoint",
                "--baseline",
                "discrete_bo",
                "--discrete-bo.candidate-source",
                "train",
                "--discrete-bo.candidate-count",
                "-1",
            ],
        )
        self.assertEqual(config.discrete_bo.candidate_count, -1)

    def test_autodiscovery_knobs_are_exposed_through_cli(self):
        config = tyro.cli(
            BaselineSearchConfig,
            args=[
                "--task-description",
                "Find a word.",
                "--target",
                "apple",
                "--reft-output-dir",
                "checkpoint",
                "--baseline",
                "autodiscovery",
                "--autodiscovery.exploration-constant",
                "0.5",
                "--autodiscovery.k-experiments",
                "4",
                "--autodiscovery.max-depth",
                "4",
                "--autodiscovery.parent-context",
                "2",
            ],
        )
        self.assertEqual(config.autodiscovery.exploration_constant, 0.5)
        self.assertEqual(config.autodiscovery.k_experiments, 4)
        self.assertEqual(config.autodiscovery.max_depth, 4)
        self.assertEqual(config.autodiscovery.parent_context, 2)

    def test_warmstarts_delegate_to_main_search(self):
        config = BaselineSearchConfig(
            task_description="Find a word.",
            target="apple",
            reft_output_dir="checkpoint",
            warmstart_source="sobol",
            warmstart_count=3,
        )
        with patch(
            "boreft.baselines.search.main_warmstarts",
            return_value=("points", "records"),
        ) as warmstarts:
            result = prepare_warmstarts(
                config,
                checkpoint=object(),
                train_mu=object(),
                bounds=object(),
                seed=7,
            )
        self.assertEqual(result, ("points", "records"))
        translated = warmstarts.call_args.args[0]
        self.assertEqual(translated.output_dir, "checkpoint")
        self.assertEqual(translated.warmstart_source, "sobol")
        self.assertEqual(translated.warmstart_count, 3)

    def test_file_warmstarts_keep_stored_decoded_text(self):
        config = BaselineSearchConfig(
            task_description="Find a word.",
            target="apple",
            reft_output_dir="checkpoint",
            warmstart_source="file",
            warmstart_file="warm.jsonl",
            warmstart_count=2,
        )
        with patch(
            "boreft.baselines.search.prepare_warmstarts",
            return_value=(
                np.array([[0.1], [0.2]]),
                [{"decoded": "stored-a", "score": 0.3}, {}],
            ),
        ):
            seeds = _warmstart_seeds(config, _fake_backend(), 1)
        self.assertEqual(seeds[0].solution, "stored-a")
        self.assertEqual(seeds[0].score, 0.3)
        self.assertIsNone(seeds[1].solution)

    def test_blank_decoded_warmstart_falls_back_to_point_decode(self):
        config = BaselineSearchConfig(
            task_description="Find a word.",
            target="apple",
            reft_output_dir="checkpoint",
            warmstart_source="file",
            warmstart_file="warm.jsonl",
            warmstart_count=2,
        )
        with patch(
            "boreft.baselines.search.prepare_warmstarts",
            return_value=(
                np.array([[0.1], [0.2]]),
                [{"decoded": "  "}, {"decoded": ""}],
            ),
        ):
            seeds = _warmstart_seeds(config, _fake_backend(), 1)
        self.assertIsNone(seeds[0].solution)
        self.assertIsNone(seeds[1].solution)
        self.assertEqual(seeds[0].decode(), "w100")

    def test_checkpoint_warmstart_seeds_match_main_search_labels(self):
        words = ["alpha", "target", "beta", "gamma", "delta"]
        train_mu = np.arange(len(words), dtype=np.float32).reshape(-1, 1)
        ckpt = type("Checkpoint", (), {"words": words, "saved_cfg": {"task": "semantle"}})()
        bounds = np.array([[0.0], [5.0]], dtype=np.float32)

        def boom(_point):
            raise AssertionError("checkpoint warm starts must not decode the vector")

        backend = _fake_backend(
            checkpoint=ckpt,
            train_mu=train_mu,
            bounds=bounds,
            decode_point=boom,
        )
        config = BaselineSearchConfig(
            task_description="Find a word.",
            target="Target",
            reft_output_dir="checkpoint",
            warmstart_source="checkpoint",
            warmstart_count=3,
        )
        seeds = _warmstart_seeds(config, backend, 7)
        _, records = select_warmstarts(
            SearchConfig(
                output_dir="checkpoint",
                target="Target",
                warmstart_source="checkpoint",
                warmstart_count=3,
            ),
            ckpt,
            train_mu,
            bounds,
            7,
        )
        self.assertEqual(
            [seed.solution for seed in seeds],
            [record["decoded"] for record in records],
        )
        self.assertNotIn("target", [seed.solution for seed in seeds])
        self.assertEqual(seeds[0].metadata["warmstart_word"], records[0]["components"]["warmstart_word"])

    def test_pinned_checkpoint_warmstart_file_is_shared_with_main_search(self):
        words = ["alpha", "target", "beta", "gamma", "delta"]
        train_mu = np.arange(len(words), dtype=np.float32).reshape(-1, 1)
        ckpt = type("Checkpoint", (), {"words": words, "saved_cfg": {"task": "semantle"}})()
        bounds = np.array([[0.0], [5.0]], dtype=np.float32)
        backend = _fake_backend(
            checkpoint=ckpt,
            train_mu=train_mu,
            bounds=bounds,
            decode_point=lambda _point: "unused",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pin.json"
            path.write_text(
                json.dumps({"seeds": {"7": ["delta", "beta"]}}),
                encoding="utf-8",
            )
            config = BaselineSearchConfig(
                task_description="Find a word.",
                target="Target",
                reft_output_dir="checkpoint",
                warmstart_source="checkpoint",
                warmstart_file=str(path),
                warmstart_count=2,
            )
            seeds = _warmstart_seeds(config, backend, 7)
            _, records = select_warmstarts(
                SearchConfig(
                    output_dir="checkpoint",
                    target="Target",
                    warmstart_source="checkpoint",
                    warmstart_file=str(path),
                    warmstart_count=2,
                ),
                ckpt,
                train_mu,
                bounds,
                7,
            )
        self.assertEqual(
            [seed.solution for seed in seeds],
            [record["decoded"] for record in records],
        )
        self.assertEqual([seed.solution for seed in seeds], ["delta", "beta"])
        self.assertTrue(records[0]["components"]["warmstart_pinned"])

    def test_checkpoint_warmstart_seeds_pass_reconstruct_decode(self):
        words = ["alpha", "target", "beta", "gamma", "delta"]
        train_mu = np.arange(len(words), dtype=np.float32).reshape(-1, 1)
        ckpt = SimpleNamespace(
            words=words,
            reft_model=object(),
            tokenizer=object(),
            saved_cfg={"task": "semantle"},
        )
        backend = _fake_backend(
            checkpoint=ckpt,
            train_mu=train_mu,
            bounds=np.array([[0.0], [5.0]], dtype=np.float32),
        )
        config = BaselineSearchConfig(
            task_description="Find a word.",
            target="Target",
            reft_output_dir="checkpoint",
            warmstart_source="checkpoint",
            warmstart_count=3,
        )
        records = [
            {
                "decoded": "alpha",
                "components": {"warmstart_word": "alpha", "warmstart_labeled": True},
            },
            {
                "decoded": "beta",
                "components": {"warmstart_word": "beta", "warmstart_labeled": True},
            },
            {
                "decoded": "gamma",
                "components": {"warmstart_word": "gamma", "warmstart_labeled": True},
            },
        ]
        sentinel = object()
        with patch(
            "boreft.baselines.search.checkpoint_reconstruct_decode",
            return_value=sentinel,
        ) as reconstruct:
            with patch(
                "boreft.baselines.search.prepare_warmstarts",
                return_value=(train_mu[:3], records),
            ) as prepare:
                seeds = _warmstart_seeds(config, backend, 7)
        reconstruct.assert_called_once()
        self.assertIs(prepare.call_args.kwargs["decode"], sentinel)
        self.assertEqual([seed.solution for seed in seeds], ["alpha", "beta", "gamma"])

    def test_resume_and_overwrite_match_main_search_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "search"
            with patch(
                "boreft.baselines.search.load_backend",
                side_effect=lambda _config: _fake_backend(),
            ):
                run_search(_run_config(tmp))
                self.assertTrue((root / "config.json").is_file())
                with self.assertRaisesRegex(FileExistsError, "not empty"):
                    run_search(_run_config(tmp))
                with self.assertRaisesRegex(ValueError, "target"):
                    run_search(_run_config(tmp, target="pear", resume=True))
                marker = root / "stale.txt"
                marker.write_text("stale", encoding="utf-8")
                run_search(_run_config(tmp, overwrite=True))
                self.assertFalse(marker.exists())

    def test_resume_allows_a_longer_decode_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "search"
            root.mkdir()
            original = _run_config(tmp, max_new_tokens=128)
            (root / "config.json").write_text(
                json.dumps(resolved_config(original)), encoding="utf-8"
            )
            _validate_resume_config(
                root / "config.json",
                _run_config(tmp, max_new_tokens=256, resume=True),
            )
            with self.assertRaisesRegex(ValueError, "target"):
                _validate_resume_config(
                    root / "config.json",
                    _run_config(tmp, target="pear", max_new_tokens=256, resume=True),
                )

    def test_aggregate_summary_totals_timings_and_repeats(self):
        summaries = {
            "1": {
                "elapsed_seconds": 2.0,
                "bbox_eval_seconds": 0.5,
                "n_repeat_samples": 1,
                "n_repeat_proposals": 2,
            },
            "2": {
                "elapsed_seconds": 3.0,
                "bbox_eval_seconds": 0.75,
                "n_repeat_samples": 4,
                "n_repeat_proposals": 1,
            },
        }
        aggregate = aggregate_summaries(summaries, (1, 2))
        self.assertEqual(aggregate["elapsed_seconds"], 5.0)
        self.assertEqual(aggregate["bbox_eval_seconds"], 1.25)
        self.assertEqual(aggregate["n_repeat_samples"], 5)
        self.assertEqual(aggregate["n_repeat_proposals"], 3)
        self.assertEqual(aggregate["n_found_target"], 0)
        self.assertEqual(aggregate["found_target_seeds"], [])

    def test_found_target_matches_semantle_words_case_insensitively(self):
        observations = [
            BaselineObservation(index=0, solution="Laptop", score=0.4),
            BaselineObservation(index=1, solution="Computer", score=0.9),
        ]
        self.assertTrue(found_target(observations, "computer", "semantle"))
        self.assertFalse(found_target(observations, "banana", "semantle"))
        hit, index, used = found_target_stats(observations, "computer", "semantle")
        self.assertTrue(hit)
        self.assertEqual(index, 1)
        self.assertEqual(used, 2)

    def test_found_target_uses_exact_match_component(self):
        observations = [
            BaselineObservation(
                index=0,
                solution="decoded",
                score=0.7,
                components={"exact_match": True},
            )
        ]
        self.assertTrue(found_target(observations, "gold", "semantle"))

    def test_finish_summary_prints_per_seed_and_overall(self):
        summaries = {
            "1": {
                "seed": 1,
                "best_score": 0.4,
                "best_solution": "laptop",
                "found_target": False,
            },
            "2": {
                "seed": 2,
                "best_score": 1.0,
                "best_solution": "computer",
                "found_target": True,
                "found_at_index": 4,
                "found_at_verifications": 12,
            },
        }
        with patch("builtins.print") as printed:
            _print_finish_summary(summaries, (1, 2))
        lines = [" ".join(str(arg) for arg in call.args) for call in printed.call_args_list]
        joined = "\n".join(lines)
        self.assertIn("Finished 2 seed(s).", joined)
        self.assertIn("seed 1  best=0.4 ('laptop')  target not found", joined)
        self.assertIn(
            "seed 2  best=1 ('computer')  target found at 12 verifications",
            joined,
        )
        self.assertIn("overall  best=1 from seed 2", joined)
        self.assertIn("mean best=0.7", joined)
        self.assertNotIn("overall  best=1 ('computer')", joined)
        self.assertIn("target found in 1/2 seed(s)", joined)
        self.assertIn("fastest at 12 verifications (seed 2)", joined)


class BaselineLoopTest(unittest.TestCase):
    def _run(self, tmp, baseline, *, config, warmstarts, score=_fake_score):
        state = BaselineRunState(Path(tmp) / "observations.jsonl")
        return run_baseline(
            seed=1,
            baseline=baseline,
            context=_context(
                lambda _prompt, _count, _options: [], score, seed=1
            ),
            state=state,
            config=config,
            warmstarts=warmstarts,
            method_state_dir=Path(tmp) / "method_state",
        )

    def test_budget_counts_warmstarts_and_searched_solutions(self):
        baseline = _RecordingBaseline(["g1", "g2", "g3"])
        with tempfile.TemporaryDirectory() as tmp:
            state = self._run(
                tmp,
                baseline,
                config=BaselineLoopConfig(budget=5, batch_size=2),
                warmstarts=[
                    WarmstartSeed(decode=lambda: "w1"),
                    WarmstartSeed(decode=lambda: "w2"),
                ],
            )
        summary = state.summary()
        self.assertEqual(summary["n_warmstart"], 2)
        self.assertEqual(summary["n_acquired"], 3)
        self.assertEqual(summary["n_observations"], 5)
        self.assertEqual(
            [item.phase for item in state.observations],
            ["warmstart", "warmstart", "search", "search", "search"],
        )
        self.assertEqual(baseline.requested, [2, 1])
        self.assertEqual(baseline.observed, ["g1", "g2", "g3"])

    def test_budget_counts_verifications_not_solutions(self):
        baseline = _RecordingBaseline(["g1", "g2"])
        with tempfile.TemporaryDirectory() as tmp:
            state = self._run(
                tmp,
                baseline,
                config=BaselineLoopConfig(
                    budget=9, batch_size=2, observation_samples=3
                ),
                warmstarts=[
                    WarmstartSeed(decode=lambda: "w1"),
                    WarmstartSeed(decode=lambda: "w2"),
                ],
            )
        summary = state.summary()
        self.assertEqual(summary["n_warmstart"], 2)
        self.assertEqual(summary["n_acquired"], 1)
        self.assertEqual(summary["n_observations"], 3)
        self.assertEqual(summary["n_verifications"], 9)
        self.assertEqual(baseline.requested, [1])
        self.assertEqual(baseline.observed, ["g1"])

    def test_stores_repaired_decoded_from_verifier_components(self):
        def score(solution: str) -> ScoreResult:
            repaired = "CCO" if solution == "not a molecule" else solution
            return ScoreResult(
                score=0.5,
                components={
                    "decoded": repaired,
                    "raw_decoded": solution,
                    "repaired": solution == "not a molecule",
                },
            )

        baseline = _RecordingBaseline(["not a molecule"])
        with tempfile.TemporaryDirectory() as tmp:
            state = self._run(
                tmp,
                baseline,
                score=score,
                config=BaselineLoopConfig(budget=3, batch_size=1),
                warmstarts=[
                    WarmstartSeed(decode=lambda: "c1ccccc1"),
                    WarmstartSeed(decode=lambda: "not a molecule"),
                ],
            )
        self.assertEqual(
            [item.solution for item in state.observations],
            ["c1ccccc1", "CCO", "CCO"],
        )
        self.assertFalse(state.observations[0].components["repaired"])
        self.assertTrue(state.observations[1].components["repaired"])
        self.assertTrue(state.observations[2].components["repaired"])
        self.assertEqual(baseline.observed, ["CCO"])

    def test_leftover_verifications_do_not_start_a_partial_observation(self):
        baseline = _RecordingBaseline(["g1"])
        with tempfile.TemporaryDirectory() as tmp:
            state = self._run(
                tmp,
                baseline,
                config=BaselineLoopConfig(budget=7, observation_samples=3),
                warmstarts=[
                    WarmstartSeed(decode=lambda: "w1"),
                    WarmstartSeed(decode=lambda: "w2"),
                ],
            )
        summary = state.summary()
        self.assertEqual(summary["n_warmstart"], 2)
        self.assertEqual(summary["n_acquired"], 0)
        self.assertEqual(summary["n_verifications"], 6)
        self.assertEqual(baseline.requested, [])

    def test_warmstart_does_not_exceed_verification_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = BaselineRunState(Path(tmp) / "observations.jsonl")
            state.append(
                BaselineObservation(
                    index=0,
                    solution="w1",
                    score=0.05,
                    sample_count=4,
                    phase="warmstart",
                    seed=1,
                )
            )
            with self.assertRaisesRegex(ValueError, "warmstart verification cost"):
                run_baseline(
                    seed=1,
                    baseline=_RecordingBaseline(["g1"]),
                    context=_context(
                        lambda _prompt, _count, _options: [],
                        _fake_score,
                        seed=1,
                    ),
                    state=state,
                    config=BaselineLoopConfig(budget=6, observation_samples=3),
                    warmstarts=[
                        WarmstartSeed(decode=lambda: "w1", solution="w1"),
                        WarmstartSeed(decode=lambda: "w2"),
                    ],
                    method_state_dir=Path(tmp) / "method_state",
                )
            self.assertEqual(len(state.observations), 1)
            self.assertEqual(state.summary()["n_verifications"], 4)

    def test_warmstart_metadata_and_precomputed_scores_are_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self._run(
                tmp,
                _RecordingBaseline([]),
                config=BaselineLoopConfig(budget=2),
                warmstarts=[
                    WarmstartSeed(
                        decode=lambda: "w1",
                        components={"llm_seed_text": "seed"},
                        metadata={"warmstart_point": [0.25]},
                    ),
                    WarmstartSeed(decode=lambda: "w2", score=0.42),
                ],
            )
        first, second = state.observations
        self.assertEqual(first.components["llm_seed_text"], "seed")
        self.assertEqual(first.candidate_metadata["warmstart_point"], [0.25])
        self.assertEqual(second.score, 0.42)
        self.assertEqual(second.bbox_eval_seconds, 0.0)

    def test_precomputed_decoded_text_skips_point_decode(self):
        def boom():
            raise AssertionError("warmstart decode should not run")

        with tempfile.TemporaryDirectory() as tmp:
            state = self._run(
                tmp,
                _RecordingBaseline([]),
                config=BaselineLoopConfig(budget=1),
                warmstarts=[
                    WarmstartSeed(decode=boom, solution="from-file", score=0.3),
                ],
            )
        self.assertEqual(state.observations[0].solution, "from-file")
        self.assertEqual(state.observations[0].score, 0.3)

    def test_labeled_warmstart_counts_one_verification(self):
        def boom():
            raise AssertionError("labeled warm starts must not decode the vector")

        with tempfile.TemporaryDirectory() as tmp:
            state = self._run(
                tmp,
                _RecordingBaseline([]),
                config=BaselineLoopConfig(budget=4, observation_samples=3),
                warmstarts=[
                    WarmstartSeed(decode=boom, solution="labeled-word"),
                    WarmstartSeed(decode=lambda: "w2"),
                ],
            )
        self.assertEqual(state.observations[0].solution, "labeled-word")
        self.assertEqual(state.observations[0].sample_count, 1)
        self.assertEqual(state.observations[1].sample_count, 3)
        self.assertEqual(state.summary()["n_verifications"], 4)
        self.assertEqual(state.summary()["n_acquired"], 0)

    def test_score_above_known_maximum_is_success(self):
        logs: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "boreft.baselines.runner._log",
                side_effect=lambda message="": logs.append(message),
            ):
                state = self._run(
                    tmp,
                    _RecordingBaseline([]),
                    config=BaselineLoopConfig(budget=4, maximum=1.0),
                    warmstarts=[WarmstartSeed(decode=lambda: "w1", score=1.1)],
                )
        self.assertEqual(len(state.observations), 1)
        self.assertEqual(state.best_score, 1.1)
        joined = "\n".join(logs)
        self.assertIn("[above known maximum]", joined)
        self.assertIn("1/4 verifications", joined)

    def test_repeated_scoring_records_uncertainty(self):
        scores = iter([0.1, 0.3, 0.5])

        with tempfile.TemporaryDirectory() as tmp:
            state = self._run(
                tmp,
                _RecordingBaseline([]),
                config=BaselineLoopConfig(budget=3, observation_samples=3),
                warmstarts=[WarmstartSeed(decode=lambda: "w1")],
                score=lambda _solution: ScoreResult(next(scores)),
            )
        observation = state.observations[0]
        self.assertAlmostEqual(observation.score, 0.3)
        self.assertAlmostEqual(observation.score_std, 0.2)
        self.assertEqual(observation.sample_scores, [0.1, 0.3, 0.5])
        self.assertEqual(state.summary()["n_verifications"], 3)

    def test_known_maximum_stops_before_spending_budget(self):
        baseline = _RecordingBaseline(["exact", "g2"])
        with tempfile.TemporaryDirectory() as tmp:
            state = self._run(
                tmp,
                baseline,
                config=BaselineLoopConfig(budget=6, batch_size=2, maximum=1.0),
                warmstarts=[
                    WarmstartSeed(decode=lambda: "w1"),
                    WarmstartSeed(decode=lambda: "w2"),
                ],
            )
        self.assertEqual(len(state.observations), 3)
        self.assertEqual(state.best_score, 1.0)
        self.assertEqual(baseline.observed, ["exact"])

    def test_resume_restores_method_state_and_skips_completed_work(self):
        decodes = 0

        def decode():
            nonlocal decodes
            decodes += 1
            return f"w{decodes}"

        with tempfile.TemporaryDirectory() as tmp:
            warmstarts = [WarmstartSeed(decode=decode) for _ in range(2)]
            self._run(
                tmp,
                _RecordingBaseline(["g1"]),
                config=BaselineLoopConfig(budget=3),
                warmstarts=warmstarts,
            )
            self.assertEqual(decodes, 2)
            self.assertEqual(
                load_method_state(Path(tmp) / "method_state"),
                {"observed": ["g1"]},
            )

            resumed = _RecordingBaseline(["g2"])
            state = BaselineRunState.load(Path(tmp) / "observations.jsonl")
            state = run_baseline(
                seed=1,
                baseline=resumed,
                context=_context(
                    lambda _prompt, _count, _options: [], _fake_score
                ),
                state=state,
                config=BaselineLoopConfig(budget=4),
                warmstarts=warmstarts,
                method_state_dir=Path(tmp) / "method_state",
            )
        self.assertEqual(decodes, 2)
        self.assertEqual(resumed.restored, {"observed": ["g1"]})
        self.assertEqual(
            [item.solution for item in state.observations],
            ["w1", "w2", "g1", "g2"],
        )

    def test_resume_rejects_changed_warmstart_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(
                tmp,
                _RecordingBaseline([]),
                config=BaselineLoopConfig(budget=2),
                warmstarts=[
                    WarmstartSeed(
                        decode=lambda: "w1",
                        metadata={"warmstart_point": [0.1]},
                    ),
                    WarmstartSeed(
                        decode=lambda: "w2",
                        metadata={"warmstart_point": [0.2]},
                    ),
                ],
            )
            state = BaselineRunState.load(Path(tmp) / "observations.jsonl")
            with self.assertRaisesRegex(ValueError, "warm starts"):
                run_baseline(
                    seed=1,
                    baseline=_RecordingBaseline(["g1"]),
                    context=_context(
                        lambda _prompt, _count, _options: [], _fake_score
                    ),
                    state=state,
                    config=BaselineLoopConfig(budget=3),
                    warmstarts=[
                        WarmstartSeed(
                            decode=lambda: "w1",
                            metadata={"warmstart_point": [0.9]},
                        ),
                        WarmstartSeed(
                            decode=lambda: "w2",
                            metadata={"warmstart_point": [0.2]},
                        ),
                    ],
                    method_state_dir=Path(tmp) / "method_state",
                )

    def test_rejects_methods_that_ignore_the_open_slot_count(self):
        overshooting = _RecordingBaseline(["g1", "g2"])
        overshooting.propose = lambda *_args: [
            Candidate(solution="g1"),
            Candidate(solution="g2"),
        ]
        for baseline, message in (
            (_RecordingBaseline([]), "proposed no candidates"),
            (overshooting, "proposed 2 candidates"),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, message):
                    self._run(
                        tmp,
                        baseline,
                        config=BaselineLoopConfig(budget=2),
                        warmstarts=[WarmstartSeed(decode=lambda: "w1")],
                    )

    def test_repeat_sample_is_flagged_against_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self._run(
                tmp,
                _RecordingBaseline(["W1", "g2"]),
                config=BaselineLoopConfig(budget=3),
                warmstarts=[WarmstartSeed(decode=lambda: "w1")],
            )
        self.assertTrue(state.observations[1].is_repeat_sample)
        self.assertFalse(state.observations[2].is_repeat_sample)
        self.assertEqual(state.summary()["n_repeat_samples"], 1)

    def test_warmstart_repeats_are_flagged_but_not_summarized(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self._run(
                tmp,
                _RecordingBaseline([]),
                config=BaselineLoopConfig(budget=2),
                warmstarts=[
                    WarmstartSeed(decode=lambda: "same"),
                    WarmstartSeed(decode=lambda: "SAME"),
                ],
            )
        self.assertTrue(state.observations[1].is_repeat_sample)
        self.assertEqual(state.summary()["n_repeat_samples"], 0)

    def test_random_sampling_scores_a_full_candidates_per_call_batch(self):
        calls: list[tuple[str, int]] = []

        def generate(prompt, count, _options):
            calls.append((prompt, count))
            return ["1. a\n2. b\n3. c" for _ in range(count)]

        with tempfile.TemporaryDirectory() as tmp:
            state = BaselineRunState(Path(tmp) / "observations.jsonl")
            run_baseline(
                seed=1,
                baseline=create_baseline(
                    "random_sampling",
                    RandomSamplingConfig(last_k_incontext=0, candidates_per_call=3),
                ),
                context=_context(generate, _fake_score),
                state=state,
                config=BaselineLoopConfig(budget=5, batch_size=1),
                warmstarts=[],
                method_state_dir=Path(tmp) / "method_state",
            )
        self.assertEqual(
            [item.solution for item in state.observations],
            ["a", "b", "c", "a", "b"],
        )
        self.assertEqual([count for _prompt, count in calls], [1, 1])
        self.assertIn("Generate 3 candidates", calls[0][0])
        self.assertIn("Generate 3 candidates", calls[1][0])

    def test_migrate_scores_on_policy_and_neighborhood_batch(self):
        calls: list[tuple[str, int]] = []

        def generate(prompt, count, _options):
            calls.append((prompt, count))
            return [f"{prompt[:1]}-{index}" for index in range(count)]

        with tempfile.TemporaryDirectory() as tmp:
            state = BaselineRunState(Path(tmp) / "observations.jsonl")
            run_baseline(
                seed=1,
                baseline=create_baseline(
                    "migrate",
                    MiGrATeConfig(
                        on_policy_count=2,
                        greedy_count=1,
                        neighborhood_count=1,
                        greedy_topk=1,
                    ),
                ),
                context=_context(generate, _fake_score),
                state=state,
                config=BaselineLoopConfig(budget=5, batch_size=1),
                warmstarts=[
                    WarmstartSeed(decode=lambda: "warm-a", score=0.2),
                    WarmstartSeed(decode=lambda: "warm-b", score=0.8),
                ],
                method_state_dir=Path(tmp) / "method_state",
            )
        search = [item for item in state.observations if item.phase == "search"]
        self.assertEqual(len(search), 3)
        self.assertEqual(
            [item.candidate_metadata["provenance"] for item in search],
            ["online", "online", "neighborhood"],
        )
        self.assertEqual([count for _prompt, count in calls], [2, 1])
        self.assertEqual(calls[0][0], "Generate a word.")
        self.assertIn("High-scoring solutions", calls[1][0])
        self.assertIn("solution: warm-b", calls[1][0])

    def test_autodiscovery_reuses_untried_pool(self):
        calls: list[tuple[str, int]] = []

        def generate(_prompt, count, _options):
            calls.append((_prompt, count))
            return [f"child-{letter}" for letter in "abc"][:count]

        with tempfile.TemporaryDirectory() as tmp:
            state = BaselineRunState(Path(tmp) / "observations.jsonl")
            run_baseline(
                seed=1,
                baseline=create_baseline(
                    "autodiscovery",
                    AutoDiscoveryConfig(
                        k_experiments=3,
                        exploration_constant=0.0,
                    ),
                ),
                context=_context(generate, _fake_score),
                state=state,
                config=BaselineLoopConfig(budget=4, batch_size=1),
                warmstarts=[
                    WarmstartSeed(decode=lambda: "warm-a", score=0.2),
                    WarmstartSeed(decode=lambda: "warm-b", score=0.8),
                ],
                method_state_dir=Path(tmp) / "method_state",
            )
        search = [item for item in state.observations if item.phase == "search"]
        self.assertEqual(len(search), 2)
        self.assertEqual([count for _prompt, count in calls], [3])
        self.assertEqual(
            [item.candidate_metadata["parent_id"] for item in search],
            [search[0].candidate_metadata["parent_id"]] * 2,
        )
        self.assertFalse(search[0].candidate_metadata["reused_untried"])
        self.assertTrue(search[1].candidate_metadata["reused_untried"])
        self.assertIn(
            "Previous solutions and scores, ordered from lowest score to highest:",
            calls[0][0],
        )
        self.assertIn("Propose a new solution that scores higher than those above.", calls[0][0])
        self.assertIn("Reply with only the solution.", calls[0][0])
        self.assertIn("solution: warm-b", calls[0][0])
        self.assertNotIn("solution: warm-a", calls[0][0])
        self.assertNotIn("build on this branch", calls[0][0])


class RandomSamplingTest(unittest.TestCase):
    def test_default_last_k_incontext_is_twenty(self):
        self.assertEqual(RandomSamplingConfig().last_k_incontext, 20)

    def test_prompt_is_task_only_when_last_k_incontext_is_zero(self):
        prompts: list[str] = []

        def generate(prompt, count, _options):
            prompts.append(prompt)
            return [f"word-{index}" for index in range(count)]

        baseline = create_baseline(
            "random_sampling", RandomSamplingConfig(last_k_incontext=0)
        )
        context = _context(generate, _fake_score)
        history = [
            BaselineObservation(index=0, solution="prior", score=0.9),
        ]
        candidates = baseline.propose(context, history, 2)
        self.assertEqual(prompts, ["Generate a word."])
        self.assertNotIn("prior", prompts[0])
        self.assertEqual(
            [candidate.solution for candidate in candidates],
            ["word-0", "word-1"],
        )
        self.assertEqual(candidates[0].metadata["prompt"], "Generate a word.")

    def test_default_prompt_includes_recent_history(self):
        prompts: list[str] = []

        def generate(prompt, count, _options):
            prompts.append(prompt)
            return [f"word-{len(prompts)}"]

        history = [
            BaselineObservation(index=0, solution="prior", score=0.9),
        ]
        candidates = create_baseline("random_sampling").propose(
            _context(generate, _fake_score), history, 1
        )
        self.assertEqual([item.solution for item in candidates], ["word-1"])
        self.assertIn("Previous candidates:", prompts[0])
        self.assertIn("prior", prompts[0])
        self.assertIn("Generate a new candidate different from those above.", prompts[0])

    def test_molopt_history_repeats_the_completion_prefix(self):
        prefix = task_instruction("molopt", use_chat_template=False)
        prompts: list[str] = []

        def generate(prompt, count, _options):
            prompts.append(prompt)
            return ["CCN"]

        candidates = create_baseline("random_sampling").propose(
            _context(
                generate,
                _fake_score,
                task="molopt",
                task_description=prefix,
            ),
            [BaselineObservation(index=0, solution="CCO", score=0.2)],
            1,
        )
        self.assertEqual(prompts, [f"{prefix} {_mist('CCO')}\n\n{prefix}"])
        self.assertNotIn("Previous candidates:", prompts[0])
        self.assertNotIn("Generate a new candidate", prompts[0])
        self.assertEqual(candidates[0].solution, "CCN")

    def test_repeat_solutions_are_kept_and_flagged(self):
        baseline = create_baseline("random_sampling")
        history = [BaselineObservation(index=0, solution="prior", score=0.9)]
        candidates = baseline.propose(
            _context(
                lambda _prompt, _count, _options: ["prior", "fresh"],
                _fake_score,
            ),
            history,
            2,
        )
        self.assertEqual(
            [item.solution for item in candidates],
            ["prior", "fresh"],
        )
        self.assertTrue(candidates[0].metadata["is_repeat_proposal"])
        self.assertFalse(candidates[1].metadata["is_repeat_proposal"])

    def test_in_batch_repeats_are_kept_and_flagged(self):
        candidates = create_baseline("random_sampling").propose(
            _context(
                lambda _prompt, _count, _options: ["same", "same"],
                _fake_score,
            ),
            [],
            2,
        )
        self.assertEqual([item.solution for item in candidates], ["same", "same"])
        self.assertFalse(candidates[0].metadata["is_repeat_proposal"])
        self.assertTrue(candidates[1].metadata["is_repeat_proposal"])

    def test_blank_samples_are_rejected(self):
        baseline = create_baseline(
            "random_sampling", RandomSamplingConfig(last_k_incontext=0)
        )
        calls: list[int] = []

        def generate(_prompt, count, _options):
            calls.append(count)
            return ["  " for _ in range(count)]

        with self.assertRaisesRegex(ValueError, "blank samples"):
            baseline.propose(
                _context(generate, _fake_score),
                [],
                1,
            )
        self.assertEqual(len(calls), _BLANK_SAMPLE_RETRIES)

    def test_blank_samples_are_retried_until_a_candidate_appears(self):
        calls: list[int] = []

        def generate(_prompt, count, _options):
            calls.append(count)
            if len(calls) < 3:
                return ["  " for _ in range(count)]
            return ["fresh" for _ in range(count)]

        candidates = create_baseline(
            "random_sampling", RandomSamplingConfig(last_k_incontext=0)
        ).propose(
            _context(generate, _fake_score),
            [],
            1,
        )
        self.assertEqual([item.solution for item in candidates], ["fresh"])
        self.assertEqual(len(calls), 3)

    def test_rejects_invalid_llm_sample_knobs(self):
        with self.assertRaisesRegex(ValueError, "last_k_incontext"):
            RandomSamplingConfig(last_k_incontext=-1)
        with self.assertRaisesRegex(ValueError, "candidates_per_call"):
            RandomSamplingConfig(candidates_per_call=0)

    def test_last_k_incontext_conditions_on_history_and_prior_samples(self):
        calls: list[tuple[str, int]] = []

        def generate(prompt, count, _options):
            calls.append((prompt, count))
            return [f"new-{len(calls)}"]

        history = [
            BaselineObservation(index=0, solution="old-a", score=0.1),
            BaselineObservation(index=1, solution="old-b", score=0.2),
            BaselineObservation(index=2, solution="old-c", score=0.3),
        ]
        candidates = create_baseline(
            "random_sampling", RandomSamplingConfig(last_k_incontext=2)
        ).propose(_context(generate, _fake_score), history, 2)
        self.assertEqual([item.solution for item in candidates], ["new-1", "new-2"])
        self.assertEqual([count for _prompt, count in calls], [1, 1])
        self.assertIn("old-b", calls[0][0])
        self.assertIn("old-c", calls[0][0])
        self.assertNotIn("old-a", calls[0][0])
        self.assertIn("old-c", calls[1][0])
        self.assertIn("new-1", calls[1][0])
        self.assertNotIn("old-b", calls[1][0])

    def test_candidates_per_call_parses_newline_separated_completions(self):
        calls: list[tuple[str, int]] = []

        def generate(prompt, count, _options):
            calls.append((prompt, count))
            return ["1. alpha\n2. beta\n3. gamma\n4. delta" for _ in range(count)]

        candidates = create_baseline(
            "random_sampling",
            RandomSamplingConfig(last_k_incontext=0, candidates_per_call=5),
        ).propose(_context(generate, _fake_score), [], 3)
        self.assertEqual(
            [item.solution for item in candidates],
            ["alpha", "beta", "gamma"],
        )
        self.assertEqual(
            calls,
            [
                (
                    "Generate a word.\n\nGenerate 5 candidates, one numbered new line each "
                    "(e.g., 1. word1\n2. word2\n...). "
                    "Reply with only the numbered candidates.",
                    1,
                )
            ],
        )


class LLMSampleTest(unittest.TestCase):
    def test_parse_takes_nonempty_lines_up_to_the_limit(self):
        self.assertEqual(
            parse_generated_candidates("\nalpha\n\nbeta\ngamma\n", 2),
            ["alpha", "beta"],
        )
        self.assertEqual(parse_generated_candidates("only", 5), ["only"])
        self.assertEqual(
            parse_generated_candidates("1. alpha\n2) beta\n3: gamma\n4. \n5. delta", 4),
            ["alpha", "beta", "gamma", "delta"],
        )
        self.assertEqual(
            parse_generated_candidates("line1\n\nline2", 1),
            ["line1\n\nline2"],
        )

    def test_recent_incontext_skips_older_duplicates(self):
        self.assertEqual(
            recent_incontext(["cat", "dog", "cat", "bird"], 2),
            ["cat", "bird"],
        )
        self.assertEqual(recent_incontext(["a", "b"], 0), [])

    def test_prompt_omits_history_when_last_k_incontext_is_zero_and_one_candidate(self):
        self.assertEqual(
            build_candidate_prompt(
                "Generate a word.",
                previous=["prior"],
                last_k_incontext=0,
                n_candidates=1,
            ),
            "Generate a word.",
        )

    def test_molopt_prompt_is_a_completion_fewshot(self):
        prefix = task_instruction("molopt", use_chat_template=False)
        self.assertEqual(
            build_candidate_prompt(
                prefix,
                previous=["CCO", "c1ccccc1"],
                last_k_incontext=2,
                n_candidates=1,
                task="molopt",
            ),
            f"{prefix} {_mist('CCO')}\n\n{prefix} {_mist('c1ccccc1')}\n\n{prefix}",
        )
        self.assertEqual(
            build_completion_fewshot(prefix, [("CCO", 0.25)]),
            f"{prefix} {_mist('CCO')}\nscore: 0.25\n\n{prefix}",
        )
        header = "The task is to optimize for DRD2 binding."
        tagged = build_completion_fewshot(
            f"{header}\n{prefix}",
            [("CCO", 0.1), ("c1ccccc1", None)],
        )
        self.assertEqual(tagged.count(header), 1)
        self.assertTrue(tagged.startswith(header + "\n"))
        self.assertIn(_mist("CCO"), tagged)
        self.assertTrue(tagged.endswith(prefix))

    def test_sample_unique_candidates_from_multiline_completions(self):
        calls: list[tuple[str, int]] = []

        def generate(prompt, count, _options):
            calls.append((prompt, count))
            return ["one\ntwo\nthree"]

        sampled = sample_llm_candidates(
            generate,
            GenerationOptions(),
            "Generate a word.",
            count=3,
            candidates_per_call=3,
            unique=True,
        )
        self.assertEqual([item.solution for item in sampled], ["one", "two", "three"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], 1)
        self.assertIn("one numbered new line each", calls[0][0])

    def test_unique_sampling_skips_previous_solutions(self):
        def generate(_prompt, _count, _options):
            return ["alpha\nbeta\ngamma"]

        sampled = sample_llm_candidates(
            generate,
            GenerationOptions(),
            "Generate a word.",
            count=2,
            candidates_per_call=3,
            previous=["Beta"],
            unique=True,
        )
        self.assertEqual([item.solution for item in sampled], ["alpha", "gamma"])


class OPROTest(unittest.TestCase):
    def _history(self) -> list[BaselineObservation]:
        return [
            BaselineObservation(index=0, solution="low", score=0.1),
            BaselineObservation(index=1, solution="high", score=0.9),
            BaselineObservation(index=2, solution="mid", score=0.5),
        ]

    def _propose(self, replies, history=None, **config):
        prompts: list[str] = []

        def generate(prompt, _count, _options):
            prompts.append(prompt)
            return list(replies)

        candidates = create_baseline("opro", OPROConfig(**config)).propose(
            _context(generate, _fake_score),
            self._history() if history is None else history,
            len(replies),
        )
        return prompts, candidates

    def test_rejects_nonpositive_history_count(self):
        with self.assertRaisesRegex(ValueError, "history_count"):
            OPROConfig(history_count=0)

    def test_molopt_prompt_repeats_completion_prefix_with_scores(self):
        prefix = task_instruction("molopt", use_chat_template=False)
        prompt = _build_prompt(
            prefix,
            [
                BaselineObservation(index=0, solution="CCO", score=0.1),
                BaselineObservation(index=1, solution="c1ccccc1", score=0.9),
            ],
            task="molopt",
        )
        self.assertEqual(
            prompt,
            f"{prefix} {_mist('CCO')}\nscore: 0.1\n\n{prefix} {_mist('c1ccccc1')}\nscore: 0.9\n\n{prefix}",
        )
        self.assertNotIn("Propose a new solution", prompt)
        self.assertNotIn("Reply with only the solution.", prompt)
        self.assertEqual(_build_prompt(prefix, [], task="molopt"), prefix)

    def test_history_count_is_exposed_through_cli(self):
        config = tyro.cli(
            BaselineSearchConfig,
            args=[
                "--task-description",
                "Find a word.",
                "--target",
                "apple",
                "--reft-output-dir",
                "checkpoint",
                "--baseline",
                "opro",
                "--opro.history-count",
                "7",
                "--opro.history-strategy",
                "recent",
            ],
        )
        self.assertEqual(config.opro.history_count, 7)
        self.assertEqual(config.opro.history_strategy, "recent")

    def test_top_k_prompt_orders_scores_ascending(self):
        prompts, candidates = self._propose(
            ["word-0"], history_strategy="top_k", history_count=2
        )
        prompt = prompts[0]
        self.assertTrue(prompt.startswith("Generate a word.\n\n"))
        self.assertIn("solution: mid", prompt)
        self.assertIn("solution: high", prompt)
        self.assertNotIn("solution: low", prompt)
        self.assertLess(prompt.index("solution: mid"), prompt.index("solution: high"))
        self.assertEqual(candidates[0].metadata["history_indices"], [2, 1])
        self.assertEqual(candidates[0].metadata["prompt"], prompt)
        self.assertEqual(candidates[0].solution, "word-0")
        self.assertNotIn("history_strategy", candidates[0].metadata)
        self.assertNotIn("target", prompt.casefold())

    def test_prompt_collapses_multiline_solutions(self):
        history = [
            BaselineObservation(index=0, solution="hello\nworld", score=0.4),
        ]
        prompts, _ = self._propose(["fresh"], history=history)
        self.assertIn("solution: hello world", prompts[0])
        self.assertNotIn("solution: hello\nworld", prompts[0])

    def test_recent_history_keeps_the_latest_observations(self):
        prompts, _ = self._propose(
            ["fresh"], history_strategy="recent", history_count=2
        )
        self.assertIn("solution: high", prompts[0])
        self.assertIn("solution: mid", prompts[0])
        self.assertNotIn("solution: low", prompts[0])

    def test_all_history_includes_every_observation(self):
        prompts, _ = self._propose(["fresh"], history_strategy="all")
        for solution in ("low", "mid", "high"):
            self.assertIn(f"solution: {solution}", prompts[0])

    def test_parses_solution_line_and_skips_blank_or_batch_duplicates(self):
        _, candidates = self._propose(
            [
                "solution: Cloud\nbecause it is related",
                "  ",
                "solution:\nlake",
                "CLOUD",
                "river",
            ]
        )
        self.assertEqual(
            [item.solution for item in candidates],
            ["Cloud", "lake", "river"],
        )

    def test_repeat_solutions_are_kept_and_flagged(self):
        baseline = create_baseline("opro")
        candidates = baseline.propose(
            _context(
                lambda _prompt, _count, _options: ["high", "fresh"],
                _fake_score,
            ),
            self._history(),
            2,
        )
        self.assertEqual(
            [item.solution for item in candidates],
            ["high", "fresh"],
        )
        self.assertTrue(candidates[0].metadata["is_repeat_proposal"])
        self.assertFalse(candidates[1].metadata["is_repeat_proposal"])

    def test_blank_samples_are_rejected(self):
        baseline = create_baseline("opro")
        with self.assertRaisesRegex(ValueError, "blank samples"):
            baseline.propose(
                _context(
                    lambda _prompt, _count, _options: ["solution:"],
                    _fake_score,
                ),
                [],
                1,
            )


class _RecordingStudent:
    def __init__(self, n_onpolicy=4):
        self.n_onpolicy = n_onpolicy
        self.calls: list[tuple[str, int]] = []
        self.teacher_prompts: list[str] = []
        self.student_prompts: list[str] = []
        self.loaded: dict | None = None

    def generate(self, prompt, count, _options):
        self.calls.append((prompt, count))
        return [f"word-{len(self.calls)}-{index}" for index in range(count)]

    def distill(self, student_prompt, teacher_prompt, options):
        self.student_prompts.append(student_prompt)
        self.teacher_prompts.append(teacher_prompt)
        self.generate(student_prompt, self.n_onpolicy, options)
        return None

    def state_dict(self):
        return {"marker": True}

    def load_state_dict(self, state):
        self.loaded = dict(state)


class SDPOTTTTest(unittest.TestCase):
    def _history(self) -> list[BaselineObservation]:
        return [
            BaselineObservation(index=0, solution="warm-a", score=0.1),
            BaselineObservation(index=1, solution="warm-b", score=0.4),
        ]

    def test_defaults_match_sdpo_ttt_knobs(self):
        config = SDPOTTTConfig()
        self.assertEqual(config.n_onpolicy, 16)
        self.assertEqual(config.teacher_policy, "ema")
        self.assertEqual(config.learning_rate, 1e-6)
        self.assertEqual(config.ema_decay, 0.99)
        self.assertEqual(config.distillation_topk, 20)
        self.assertTrue(config.distillation_add_tail)
        self.assertEqual(config.last_k_incontext, 0)
        self.assertEqual(config.last_k_strategy, "recent")

    def test_normalize_generated_text_unwraps_molopt_tags(self):
        self.assertEqual(
            _normalize_generated_text("CCO [END_SMILES]", "molopt"),
            "CCO",
        )
        self.assertEqual(_normalize_generated_text("Apple", "hypogen"), "Apple")

    def test_normalize_generated_text_repairs_invalid_molopt(self):
        with patch("boreft.chem.repair_smiles", return_value="CCO"):
            self.assertEqual(
                _normalize_generated_text("not a molecule [END_SMILES]", "molopt"),
                "CCO",
            )
            self.assertEqual(
                _search_normalize_generated_text(
                    "not a molecule [END_SMILES]", "molopt"
                ),
                "CCO",
            )

    def test_lora_completion_text_matches_mist_gold(self):
        runtime = SimpleNamespace(
            checkpoint=SimpleNamespace(
                saved_cfg={"mist_smiles_tags": True, "use_chat_template": False}
            )
        )
        self.assertEqual(
            LoRAStudentRuntime._completion_text(runtime, "CCO"),
            "CCO [END_SMILES]",
        )
        runtime.checkpoint.saved_cfg["use_chat_template"] = True
        self.assertEqual(
            LoRAStudentRuntime._completion_text(runtime, "CCO"),
            "[START_SMILES] CCO [END_SMILES]",
        )

    def test_rejects_invalid_knobs(self):
        with self.assertRaisesRegex(ValueError, "n_onpolicy"):
            SDPOTTTConfig(n_onpolicy=0)
        with self.assertRaisesRegex(ValueError, "lora_rank"):
            SDPOTTTConfig(lora_rank=0)
        with self.assertRaisesRegex(ValueError, "teacher policy"):
            SDPOTTTConfig(teacher_policy="lagged")
        with self.assertRaisesRegex(ValueError, "ema_decay"):
            SDPOTTTConfig(ema_decay=0.0)
        with self.assertRaisesRegex(ValueError, "distillation_topk"):
            SDPOTTTConfig(distillation_topk=-1)
        with self.assertRaisesRegex(ValueError, "last_k_incontext"):
            SDPOTTTConfig(last_k_incontext=-1)
        with self.assertRaisesRegex(ValueError, "last-k strategy"):
            SDPOTTTConfig(last_k_strategy="worst")

    def test_teacher_prompt_is_latest_batch_only_and_sorted(self):
        prompt = build_teacher_prompt(
            "Generate a word.",
            [
                BaselineObservation(index=2, solution="high", score=0.9),
                BaselineObservation(index=3, solution="low", score=0.2),
            ],
        )
        self.assertTrue(prompt.startswith("Generate a word.\n\n"))
        self.assertIn("Feedback from the latest evaluated batch", prompt)
        self.assertIn("solution: low", prompt)
        self.assertIn("solution: high", prompt)
        self.assertLess(prompt.index("solution: low"), prompt.index("solution: high"))
        self.assertNotIn("warm-a", prompt)
        self.assertIn("Propose a new solution that scores higher than those above.", prompt)

    def test_teacher_prompt_last_k_keeps_recent_history_and_batch(self):
        history = self._history()
        batch = [
            BaselineObservation(index=2, solution="search-x", score=0.7),
            BaselineObservation(index=3, solution="search-y", score=0.3),
        ]
        prompt = build_teacher_prompt(
            "Generate a word.",
            batch,
            history=history,
            last_k_incontext=3,
        )
        self.assertIn("Feedback from the last 3 observation(s)", prompt)
        self.assertIn("solution: warm-b", prompt)
        self.assertIn("solution: search-x", prompt)
        self.assertIn("solution: search-y", prompt)
        self.assertNotIn("warm-a", prompt)
        self.assertLess(prompt.index("solution: search-y"), prompt.index("solution: warm-b"))
        whole_batch = build_teacher_prompt(
            "Generate a word.", batch, history=history, last_k_incontext=0
        )
        self.assertNotIn("warm-b", whole_batch)

    def test_teacher_prompt_best_keeps_highest_scoring_uniques(self):
        history = [
            BaselineObservation(index=0, solution="warm-a", score=0.9),
            BaselineObservation(index=1, solution="warm-b", score=0.1),
        ]
        batch = [
            BaselineObservation(index=2, solution="search-x", score=0.2),
            BaselineObservation(index=3, solution="search-y", score=0.3),
        ]
        recent = build_teacher_prompt(
            "Generate a word.",
            batch,
            history=history,
            last_k_incontext=3,
            last_k_strategy="recent",
        )
        best = build_teacher_prompt(
            "Generate a word.",
            batch,
            history=history,
            last_k_incontext=3,
            last_k_strategy="best",
        )
        self.assertIn("Feedback from the last 3 observation(s)", recent)
        self.assertIn("solution: warm-b", recent)
        self.assertNotIn("warm-a", recent)
        self.assertIn("Feedback from the best 3 observation(s)", best)
        self.assertIn("solution: warm-a", best)
        self.assertNotIn("warm-b", best)
        self.assertIn("solution: search-x", best)
        self.assertIn("solution: search-y", best)

    def test_teacher_prompt_deduplicates_repeat_solutions(self):
        history = self._history() + [
            BaselineObservation(index=2, solution="WARM-B", score=0.5),
        ]
        batch = [
            BaselineObservation(index=3, solution="search-x", score=0.7),
            BaselineObservation(index=4, solution="SEARCH-X", score=0.2),
        ]
        prompt = build_teacher_prompt(
            "Generate a word.",
            batch,
            history=history,
            last_k_incontext=3,
        )
        self.assertEqual(prompt.count("solution:"), 3)
        self.assertIn("solution: WARM-B", prompt)
        self.assertIn("solution: SEARCH-X", prompt)
        self.assertIn("solution: warm-a", prompt)
        self.assertNotIn("solution: warm-b\n", prompt)
        self.assertNotIn("solution: search-x\n", prompt)
        batch_only = build_teacher_prompt("Generate a word.", batch)
        self.assertEqual(batch_only.count("solution:"), 1)
        self.assertIn("solution: SEARCH-X", batch_only)

    def test_propose_is_task_only_and_distills_warmstarts_first(self):
        runtime = _RecordingStudent(n_onpolicy=3)
        baseline = SDPOTTTBaseline(
            SDPOTTTConfig(n_onpolicy=3), runtime=runtime
        )
        candidates = baseline.propose(
            _context(lambda *_args: ["unused"], _fake_score),
            self._history(),
            2,
        )
        self.assertEqual(
            [item.solution for item in candidates],
            ["word-2-0", "word-2-1"],
        )
        self.assertEqual(runtime.student_prompts, ["Generate a word."])
        self.assertEqual(len(runtime.teacher_prompts), 1)
        self.assertIn("solution: warm-a", runtime.teacher_prompts[0])
        self.assertIn("solution: warm-b", runtime.teacher_prompts[0])
        self.assertEqual(
            runtime.calls,
            [("Generate a word.", 3), ("Generate a word.", 2)],
        )
        self.assertEqual(candidates[0].metadata["prompt"], "Generate a word.")
        self.assertEqual(baseline.state_dict()["consumed_through"], 1)
        self.assertNotIn("target", runtime.teacher_prompts[0].casefold())

    def test_observe_distills_only_the_latest_search_batch(self):
        runtime = _RecordingStudent(n_onpolicy=2)
        baseline = SDPOTTTBaseline(
            SDPOTTTConfig(n_onpolicy=2), runtime=runtime
        )
        context = _context(lambda *_args: ["unused"], _fake_score)
        baseline.propose(context, self._history(), 1)
        baseline.observe(
            [
                BaselineObservation(index=2, solution="search-x", score=0.7),
                BaselineObservation(index=3, solution="search-y", score=0.3),
            ]
        )
        self.assertEqual(len(runtime.teacher_prompts), 2)
        self.assertIn("solution: search-x", runtime.teacher_prompts[1])
        self.assertIn("solution: search-y", runtime.teacher_prompts[1])
        self.assertNotIn("warm-a", runtime.teacher_prompts[1])
        self.assertLess(
            runtime.teacher_prompts[1].index("solution: search-y"),
            runtime.teacher_prompts[1].index("solution: search-x"),
        )
        self.assertEqual(baseline.state_dict()["consumed_through"], 3)

    def test_observe_last_k_incontext_includes_prior_observations(self):
        runtime = _RecordingStudent(n_onpolicy=2)
        baseline = SDPOTTTBaseline(
            SDPOTTTConfig(n_onpolicy=2, last_k_incontext=3), runtime=runtime
        )
        context = _context(lambda *_args: ["unused"], _fake_score)
        baseline.propose(context, self._history(), 1)
        baseline.observe(
            [
                BaselineObservation(index=2, solution="search-x", score=0.7),
                BaselineObservation(index=3, solution="search-y", score=0.3),
            ]
        )
        teacher = runtime.teacher_prompts[1]
        self.assertIn("solution: warm-b", teacher)
        self.assertIn("solution: search-x", teacher)
        self.assertIn("solution: search-y", teacher)
        self.assertNotIn("warm-a", teacher)
        first = runtime.teacher_prompts[0]
        self.assertIn("solution: warm-a", first)
        self.assertIn("solution: warm-b", first)

    def test_repeat_proposals_are_kept_and_flagged(self):
        class _FixedStudent(_RecordingStudent):
            def generate(self, prompt, count, options):
                self.calls.append((prompt, count))
                return ["warm-a", "fresh"][:count]

        runtime = _FixedStudent(n_onpolicy=1)
        candidates = SDPOTTTBaseline(
            SDPOTTTConfig(n_onpolicy=1), runtime=runtime
        ).propose(
            _context(lambda *_args: ["unused"], _fake_score),
            self._history(),
            2,
        )
        self.assertEqual(
            [item.solution for item in candidates],
            ["warm-a", "fresh"],
        )
        self.assertTrue(candidates[0].metadata["is_repeat_proposal"])
        self.assertFalse(candidates[1].metadata["is_repeat_proposal"])

    def test_blank_samples_are_rejected(self):
        class _BlankStudent(_RecordingStudent):
            def generate(self, _prompt, _count, _options):
                return ["  "]

        with self.assertRaisesRegex(ValueError, "blank samples"):
            SDPOTTTBaseline(runtime=_BlankStudent()).propose(
                _context(lambda *_args: ["unused"], _fake_score),
                [],
                1,
            )

    def test_state_round_trips_consumed_cursor_and_runtime(self):
        runtime = _RecordingStudent(n_onpolicy=1)
        baseline = SDPOTTTBaseline(
            SDPOTTTConfig(n_onpolicy=1), runtime=runtime
        )
        baseline.propose(
            _context(lambda *_args: ["unused"], _fake_score),
            self._history(),
            1,
        )
        restored_runtime = _RecordingStudent(n_onpolicy=1)
        restored = SDPOTTTBaseline(
            SDPOTTTConfig(n_onpolicy=1), runtime=restored_runtime
        )
        restored.load_state_dict(baseline.state_dict())
        self.assertEqual(restored.state_dict()["consumed_through"], 1)
        restored.propose(
            _context(lambda *_args: ["unused"], _fake_score),
            self._history(),
            1,
        )
        self.assertEqual(restored_runtime.teacher_prompts, [])

    def test_lora_student_distills_on_a_tiny_causal_lm(self):
        import torch

        class _Tokenizer:
            pad_token_id = 0
            eos_token_id = 1

            def __call__(self, text, add_special_tokens=True, return_tensors=None, **_kwargs):
                ids = [2 + (ord(char) % 29) for char in str(text)[:24]] or [2]
                if add_special_tokens:
                    ids = [2] + ids
                tokens = torch.tensor([ids], dtype=torch.long)
                mask = torch.ones_like(tokens)
                if return_tensors == "pt":
                    return {"input_ids": tokens, "attention_mask": mask}
                return {"input_ids": ids, "attention_mask": [1] * len(ids)}

            def decode(self, ids, skip_special_tokens=True):
                del ids, skip_special_tokens
                return "tiny"

        class _TinyLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(32, 8)
                self.q_proj = torch.nn.Linear(8, 8)
                self.lm_head = torch.nn.Linear(8, 32)
                self.seen_training: list[bool] = []
                self.generate_calls = 0

            def forward(self, input_ids, attention_mask=None, **_kwargs):
                del attention_mask
                self.seen_training.append(self.training)
                hidden = self.q_proj(self.embed(input_ids))
                hidden = hidden + hidden.cumsum(dim=1)
                return type("Out", (), {"logits": self.lm_head(hidden)})()

            def generate(self, input_ids, max_new_tokens=1, **_kwargs):
                self.generate_calls += 1
                extra = torch.full(
                    (input_ids.shape[0], max_new_tokens),
                    5,
                    dtype=torch.long,
                    device=input_ids.device,
                )
                return torch.cat([input_ids, extra], dim=1)

            @property
            def device(self):
                return next(self.parameters()).device

        model = _TinyLM()
        checkpoint = type(
            "Checkpoint",
            (),
            {
                "reft_model": type("ReFT", (), {"model": model})(),
                "tokenizer": _Tokenizer(),
                "from_chat_template": False,
                "assistant_suffix": None,
                "saved_cfg": {},
                "prompt": "unused",
                "content_span": None,
            },
        )()
        context = _context(
            lambda *_args: ["unused"],
            _fake_score,
            checkpoint=checkpoint,
            model_name="tiny/model",
            task="semantle",
            generation_options=GenerationOptions(max_new_tokens=2, temperature=0.0),
        )
        baseline = SDPOTTTBaseline(
            SDPOTTTConfig(
                n_onpolicy=1,
                lora_rank=2,
                update_steps=2,
                learning_rate=0.05,
            )
        )
        baseline.propose(context, [], 1)
        self.assertIsInstance(baseline._runtime.optimizer, torch.optim.AdamW)
        before = baseline.state_dict()["adapters"]
        baseline.observe(
            [BaselineObservation(index=0, solution="scored", score=0.8)]
        )
        after = baseline.state_dict()["adapters"]
        self.assertNotEqual(before, after)
        self.assertEqual(baseline.state_dict()["consumed_through"], 0)
        self.assertTrue(after)
        self.assertTrue(model.seen_training)
        self.assertFalse(any(model.seen_training))
        # propose samples once; each update step draws a fresh on-policy rollout.
        self.assertEqual(model.generate_calls, 3)

    def test_clear_checkpoint_lora_makes_forward_match_base(self):
        import torch
        import torch.nn as nn

        base = nn.Linear(4, 4)
        adapter = LoRALinear(base, rank=2, alpha=4.0)
        adapter.lora_A.data.fill_(0.25)
        adapter.lora_B.data.fill_(0.5)
        adapter.ema_A.fill_(0.25)
        adapter.ema_B.fill_(0.5)
        inputs = torch.randn(3, 4)
        self.assertFalse(torch.allclose(adapter(inputs), base(inputs)))
        checkpoint = type(
            "Checkpoint",
            (),
            {"reft_model": type("ReFT", (), {"model": adapter})()},
        )()
        clear_checkpoint_lora(checkpoint)
        self.assertTrue(torch.allclose(adapter(inputs), base(inputs)))
        self.assertTrue(torch.equal(adapter.lora_A, torch.zeros_like(adapter.lora_A)))
        self.assertTrue(torch.equal(adapter.lora_B, torch.zeros_like(adapter.lora_B)))
        self.assertTrue(torch.equal(adapter.ema_A, torch.zeros_like(adapter.ema_A)))
        self.assertTrue(torch.equal(adapter.ema_B, torch.zeros_like(adapter.ema_B)))

    def test_clear_checkpoint_lora_finds_adapters_on_reft_wrapper(self):
        import torch
        import torch.nn as nn

        base = nn.Linear(4, 4)
        adapter = LoRALinear(base, rank=2, alpha=4.0)
        adapter.lora_B.data.fill_(0.5)
        checkpoint = type(
            "Checkpoint",
            (),
            {"reft_model": nn.Sequential(adapter)},
        )()
        clear_checkpoint_lora(checkpoint)
        self.assertTrue(torch.equal(adapter.lora_B, torch.zeros_like(adapter.lora_B)))

    def test_clear_checkpoint_lora_is_a_noop_without_adapters(self):
        clear_checkpoint_lora(None)
        clear_checkpoint_lora(object())
        checkpoint = type(
            "Checkpoint",
            (),
            {"reft_model": type("ReFT", (), {"model": object()})()},
        )()
        clear_checkpoint_lora(checkpoint)


class _RecordingPolicy:
    def __init__(self):
        self.calls: list[tuple[str, int]] = []
        self.groups: list[tuple[str, list[tuple[str, float]]]] = []
        self.loaded: dict | None = None

    def generate(self, prompt, count, _options):
        self.calls.append((prompt, count))
        return [f"word-{len(self.calls)}-{index}" for index in range(count)]

    def update_grpo(self, task_prompt, group):
        self.groups.append((task_prompt, list(group)))
        return None

    def state_dict(self):
        return {"marker": True}

    def load_state_dict(self, state):
        self.loaded = dict(state)


class AutoDiscoveryTest(unittest.TestCase):
    def _history(self) -> list[BaselineObservation]:
        return [
            BaselineObservation(
                index=0, solution="warm-a", score=0.1, phase="warmstart"
            ),
            BaselineObservation(
                index=1, solution="warm-b", score=0.9, phase="warmstart"
            ),
        ]

    def _propose(self, replies, history=None, count=1, **config):
        prompts: list[str] = []

        def generate(prompt, n, _options):
            del n
            prompts.append(prompt)
            return list(replies)

        baseline = create_baseline("autodiscovery", AutoDiscoveryConfig(**config))
        candidates = baseline.propose(
            _context(generate, _fake_score),
            self._history() if history is None else history,
            count,
        )
        return baseline, prompts, candidates

    def test_defaults_match_autodiscovery_repo_knobs(self):
        config = AutoDiscoveryConfig()
        self.assertEqual(config.exploration_constant, 2.0)
        self.assertEqual(config.k_experiments, 8)
        self.assertIsNone(config.max_depth)
        self.assertEqual(config.parent_context, 3)

    def test_rejects_invalid_knobs(self):
        with self.assertRaisesRegex(ValueError, "exploration_constant"):
            AutoDiscoveryConfig(exploration_constant=-0.1)
        with self.assertRaisesRegex(ValueError, "k_experiments"):
            AutoDiscoveryConfig(k_experiments=0)
        with self.assertRaisesRegex(ValueError, "max_depth"):
            AutoDiscoveryConfig(max_depth=0)
        with self.assertRaisesRegex(ValueError, "parent_context"):
            AutoDiscoveryConfig(parent_context=0)

    def test_ucb1_matches_paper_formula(self):
        parent = MCTSNode(node_id=0, parent_id=None, depth=0, visits=10, value=4.0)
        node = MCTSNode(node_id=1, parent_id=0, depth=1, visits=4, value=2.0)
        expected = 2.0 / 4.0 + 2.0 * math.sqrt(2.0 * math.log(10) / 4.0)
        self.assertAlmostEqual(ucb1(node, parent, 2.0), expected)
        self.assertEqual(
            ucb1(MCTSNode(node_id=2, parent_id=0, depth=1), parent, 2.0),
            math.inf,
        )
        self.assertEqual(ucb1(parent, None, 2.0), 0.4)

    def test_max_depth_gates_expandability(self):
        config = AutoDiscoveryConfig()
        leaf = MCTSNode(node_id=1, parent_id=0, depth=1, visits=1)
        self.assertTrue(can_expand(leaf, config))
        filled = MCTSNode(
            node_id=1,
            parent_id=0,
            depth=1,
            visits=1,
            children_ids=[2],
        )
        self.assertTrue(can_expand(filled, config))
        capped = MCTSNode(
            node_id=3, parent_id=1, depth=2, visits=1, children_ids=[]
        )
        self.assertFalse(can_expand(capped, AutoDiscoveryConfig(max_depth=2)))
        self.assertTrue(can_expand(capped, AutoDiscoveryConfig(max_depth=3)))

    def test_recursive_ucb_breaks_ties_by_node_id(self):
        config = AutoDiscoveryConfig(exploration_constant=0.0)
        nodes = {
            0: MCTSNode(
                node_id=0,
                parent_id=None,
                depth=0,
                visits=2,
                value=0.6,
                children_ids=[1, 2],
            ),
            1: MCTSNode(
                node_id=1,
                parent_id=0,
                depth=1,
                visits=1,
                value=0.5,
                solution="a",
            ),
            2: MCTSNode(
                node_id=2,
                parent_id=0,
                depth=1,
                visits=1,
                value=0.5,
                solution="b",
            ),
        }
        chosen = select_expandable(nodes, config)
        self.assertEqual(chosen.node_id, 1)

    def test_max_depth_prefers_widening_an_ancestor(self):
        config = AutoDiscoveryConfig(exploration_constant=0.0, max_depth=1)
        nodes = {
            0: MCTSNode(
                node_id=0,
                parent_id=None,
                depth=0,
                visits=2,
                value=1.0,
                children_ids=[1, 2],
            ),
            1: MCTSNode(
                node_id=1,
                parent_id=0,
                depth=1,
                visits=1,
                value=0.4,
                solution="a",
            ),
            2: MCTSNode(
                node_id=2,
                parent_id=0,
                depth=1,
                visits=1,
                value=0.6,
                solution="b",
            ),
        }
        chosen = select_expandable(nodes, config)
        self.assertEqual(chosen.node_id, 0)
        self.assertEqual(chosen.depth, 0)

    def test_greedy_constant_selects_the_higher_scoring_child(self):
        config = AutoDiscoveryConfig(exploration_constant=0.0)
        nodes = {
            0: MCTSNode(
                node_id=0,
                parent_id=None,
                depth=0,
                visits=2,
                value=1.0,
                children_ids=[1, 2],
            ),
            1: MCTSNode(
                node_id=1,
                parent_id=0,
                depth=1,
                visits=1,
                value=0.1,
                solution="low",
            ),
            2: MCTSNode(
                node_id=2,
                parent_id=0,
                depth=1,
                visits=1,
                value=0.9,
                solution="high",
            ),
        }
        self.assertEqual(select_expandable(nodes, config).node_id, 2)

    def test_branch_prompt_omits_siblings_and_the_target(self):
        _, prompts, candidates = self._propose(["fresh"])
        prompt = prompts[0]
        self.assertTrue(prompt.startswith("Generate a word.\n\n"))
        self.assertIn(
            "Previous solutions and scores, ordered from lowest score to highest:",
            prompt,
        )
        self.assertIn("solution: warm-b", prompt)
        self.assertIn("Propose a new solution that scores higher than those above.", prompt)
        self.assertIn("Reply with only the solution.", prompt)
        self.assertNotIn("build on this branch", prompt)
        self.assertNotIn("numbered", prompt.casefold())
        self.assertNotIn("solution: warm-a", prompt)
        self.assertNotIn("target", prompt.casefold())
        self.assertEqual(candidates[0].solution, "fresh")
        self.assertEqual(candidates[0].metadata["parent_id"], 2)
        self.assertEqual(candidates[0].metadata["history_indices"], [1])
        self.assertEqual(candidates[0].metadata["prompt"], prompt)

    def test_branch_prompt_matches_opro_for_the_same_history(self):
        history = [
            BaselineObservation(index=0, solution="warm-a", score=0.1),
            BaselineObservation(index=1, solution="warm-b", score=0.9),
        ]
        path = [
            MCTSNode(
                node_id=2,
                parent_id=0,
                depth=1,
                solution="warm-b",
                score=0.9,
                observation_index=1,
            ),
            MCTSNode(
                node_id=1,
                parent_id=0,
                depth=1,
                solution="warm-a",
                score=0.1,
                observation_index=0,
            ),
        ]
        self.assertEqual(
            build_branch_prompt("Generate a word.", path),
            _build_prompt("Generate a word.", history),
        )

    def test_parent_context_keeps_the_nearest_ancestors(self):
        history = self._history() + [
            BaselineObservation(
                index=2,
                solution="mid",
                score=0.92,
                candidate_metadata={"node_id": 3, "parent_id": 2, "depth": 2},
            ),
            BaselineObservation(
                index=3,
                solution="leaf",
                score=0.95,
                candidate_metadata={"node_id": 4, "parent_id": 3, "depth": 3},
            ),
        ]
        _, prompts, _ = self._propose(
            ["fresh"],
            history=history,
            exploration_constant=0.0,
            parent_context=2,
        )
        prompt = prompts[0]
        self.assertIn("solution: mid", prompt)
        self.assertIn("solution: leaf", prompt)
        self.assertNotIn("solution: warm-b", prompt)
        self.assertNotIn("solution: warm-a", prompt)

    def test_parses_solution_line_and_skips_blank_or_batch_duplicates(self):
        baseline, _, candidates = self._propose(
            ["solution: Cloud", "  ", "CLOUD", "river"],
            k_experiments=4,
        )
        self.assertEqual(len(candidates), 1)
        parent = baseline._nodes[int(candidates[0].metadata["parent_id"])]
        self.assertEqual(
            sorted([candidates[0].solution, *parent.untried_solutions]),
            ["Cloud", "river"],
        )

    def test_repeat_solutions_are_kept_and_flagged(self):
        baseline, _, candidates = self._propose(
            ["warm-b", "fresh"],
            k_experiments=2,
        )
        self.assertEqual(len(candidates), 1)
        leftover = baseline._nodes[
            int(candidates[0].metadata["parent_id"])
        ].untried_solutions
        self.assertEqual(len(leftover), 1)
        if candidates[0].solution == "warm-b":
            self.assertTrue(candidates[0].metadata["is_repeat_proposal"])
            self.assertEqual(leftover[0], "fresh")
        else:
            self.assertEqual(candidates[0].solution, "fresh")
            self.assertFalse(candidates[0].metadata["is_repeat_proposal"])
            self.assertEqual(leftover[0], "warm-b")

    def test_blank_samples_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "blank samples"):
            self._propose(["solution:"], count=1)

    def test_observe_backpropagates_along_the_branch(self):
        baseline, _, candidates = self._propose(["kid"])
        child_id = int(candidates[0].metadata["node_id"])
        parent_id = int(candidates[0].metadata["parent_id"])
        baseline.observe(
            [
                BaselineObservation(
                    index=2,
                    solution="kid",
                    score=0.5,
                    candidate_metadata=dict(candidates[0].metadata),
                )
            ]
        )
        root = baseline._nodes[0]
        parent = baseline._nodes[parent_id]
        child = baseline._nodes[child_id]
        self.assertEqual(child.visits, 1)
        self.assertEqual(child.value, 0.5)
        self.assertEqual(parent.visits, 2)
        self.assertAlmostEqual(parent.value, 1.4)
        self.assertEqual(root.visits, 3)
        self.assertAlmostEqual(root.value, 1.5)
        self.assertEqual(
            [node.solution for node in path_from_root(baseline._nodes, child_id)],
            ["warm-b", "kid"],
        )

    def test_state_round_trip_restores_the_tree(self):
        baseline, _, candidates = self._propose(["kid"])
        baseline.observe(
            [
                BaselineObservation(
                    index=2,
                    solution="kid",
                    score=0.5,
                    candidate_metadata=dict(candidates[0].metadata),
                )
            ]
        )
        payload = baseline.state_dict()
        restored = create_baseline("autodiscovery")
        restored.load_state_dict(payload)
        self.assertEqual(restored._next_id, baseline._next_id)
        self.assertEqual(
            {node_id: node.to_dict() for node_id, node in restored._nodes.items()},
            {node_id: node.to_dict() for node_id, node in baseline._nodes.items()},
        )
        json.dumps(payload, sort_keys=True)

    def test_resume_rebuilds_from_history_when_state_is_empty(self):
        history = self._history() + [
            BaselineObservation(
                index=2,
                solution="kid",
                score=0.95,
                candidate_metadata={"node_id": 3, "parent_id": 2, "depth": 2},
            )
        ]
        _, prompts, candidates = self._propose(
            ["next"],
            history=history,
            exploration_constant=0.0,
        )
        self.assertEqual(candidates[0].metadata["parent_id"], 3)
        self.assertIn("solution: kid", prompts[0])
        self.assertIn("solution: warm-b", prompts[0])

    def test_generates_a_pool_and_scores_one(self):
        calls: list[int] = []

        def generate(_prompt, n, _options):
            calls.append(n)
            return [f"pool-{index}" for index in range(n)]

        baseline = create_baseline(
            "autodiscovery",
            AutoDiscoveryConfig(k_experiments=4, exploration_constant=0.0),
        )
        candidates = baseline.propose(
            _context(generate, _fake_score), self._history(), 1
        )
        self.assertEqual(calls, [4])
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0].solution.startswith("pool-"))
        self.assertFalse(candidates[0].metadata["reused_untried"])
        parent = baseline._nodes[int(candidates[0].metadata["parent_id"])]
        self.assertEqual(len(parent.untried_solutions), 3)
        self.assertNotIn(candidates[0].solution, parent.untried_solutions)
        self.assertEqual(parent.node_id, 2)

    def test_reuses_untried_before_generating_again(self):
        calls: list[int] = []

        def generate(_prompt, n, _options):
            calls.append(n)
            return [f"pool-{index}" for index in range(n)]

        baseline = create_baseline(
            "autodiscovery",
            AutoDiscoveryConfig(k_experiments=3, exploration_constant=0.0),
        )
        context = _context(generate, _fake_score)
        first = baseline.propose(context, self._history(), 1)
        parent_id = int(first[0].metadata["parent_id"])
        observed = BaselineObservation(
            index=2,
            solution=first[0].solution,
            score=0.05,
            candidate_metadata=dict(first[0].metadata),
        )
        baseline.observe([observed])
        second = baseline.propose(context, self._history() + [observed], 1)
        self.assertEqual(calls, [3])
        self.assertTrue(second[0].metadata["reused_untried"])
        self.assertEqual(second[0].metadata["parent_id"], parent_id)
        self.assertNotEqual(second[0].solution, first[0].solution)
        parent = baseline._nodes[parent_id]
        self.assertEqual(len(parent.untried_solutions), 1)
        restored = create_baseline("autodiscovery")
        restored.load_state_dict(baseline.state_dict())
        self.assertEqual(
            restored._nodes[parent_id].untried_solutions,
            parent.untried_solutions,
        )

    def test_pick_untried_is_uniform_over_the_pool(self):
        node = MCTSNode(node_id=1, parent_id=0, depth=1)
        node.untried_solutions = ["a", "b", "c"]
        chosen = pick_untried(node, random.Random(0))
        self.assertIn(chosen, ["a", "b", "c"])
        self.assertEqual(len(node.untried_solutions), 2)
        self.assertNotIn(chosen, node.untried_solutions)


class MiGrATeTest(unittest.TestCase):
    def _history(self) -> list[BaselineObservation]:
        return [
            BaselineObservation(index=0, solution="warm-a", score=0.1),
            BaselineObservation(index=1, solution="warm-b", score=0.9),
            BaselineObservation(index=2, solution="warm-c", score=0.4),
        ]

    def test_defaults_match_mixed_policy_knobs(self):
        config = MiGrATeConfig()
        self.assertEqual(config.on_policy_count, 2)
        self.assertEqual(config.greedy_count, 1)
        self.assertEqual(config.neighborhood_count, 2)
        self.assertEqual(config.greedy_topk, 3)
        self.assertEqual(config.learning_rate, 1e-5)
        self.assertEqual(config.clip_epsilon, 0.2)
        self.assertEqual(config.clip_epsilon_high, 0.28)
        self.assertEqual(config.group_size, 5)
        self.assertEqual(config.new_sample_count, 4)

    def test_rejects_invalid_knobs(self):
        with self.assertRaisesRegex(ValueError, "on_policy_count"):
            MiGrATeConfig(on_policy_count=-1)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            MiGrATeConfig(on_policy_count=0, neighborhood_count=0)
        with self.assertRaisesRegex(ValueError, "greedy_topk"):
            MiGrATeConfig(greedy_topk=0)
        with self.assertRaisesRegex(ValueError, "lora_rank"):
            MiGrATeConfig(lora_rank=0)

    def test_select_greedy_samples_from_unique_topk(self):
        history = self._history() + [
            BaselineObservation(index=3, solution="WARM-A", score=0.2),
        ]
        chosen = select_greedy(history, count=1, topk=1, rng=random.Random(0))
        self.assertEqual([item.solution for item in chosen], ["warm-b"])
        chosen = select_greedy(history, count=2, topk=2, rng=random.Random(0))
        self.assertEqual({item.solution for item in chosen}, {"warm-b", "warm-c"})
        chosen = select_greedy(history, count=10, topk=2, rng=random.Random(0))
        self.assertEqual(len(chosen), 2)
        self.assertEqual({item.solution for item in chosen}, {"warm-b", "warm-c"})
        self.assertEqual(select_greedy([], count=1, topk=3, rng=random.Random(0)), [])

    def test_neighborhood_prompt_lists_greedy_solutions(self):
        prompt = build_neighborhood_prompt("Generate a word.", self._history()[:1])
        self.assertTrue(prompt.startswith("Generate a word.\n\n"))
        self.assertIn("High-scoring solutions", prompt)
        self.assertIn("solution: warm-a", prompt)
        self.assertIn("Generate a new solution related to those above.", prompt)
        self.assertEqual(
            build_neighborhood_prompt("Generate a word.", []),
            "Generate a word.",
        )

    def test_molopt_neighborhood_repeats_the_completion_prefix(self):
        prefix = task_instruction("molopt", use_chat_template=False)
        prompt = build_neighborhood_prompt(
            prefix, self._history()[:1], task="molopt"
        )
        self.assertEqual(prompt, f"{prefix} {_mist('warm-a')}\n\n{prefix}")
        self.assertNotIn("Generate a new solution", prompt)
        self.assertEqual(
            build_neighborhood_prompt(prefix, [], task="molopt"),
            prefix,
        )

    def test_propose_tags_online_and_neighborhood_provenance(self):
        runtime = _RecordingPolicy()
        baseline = MiGrATeBaseline(
            MiGrATeConfig(
                on_policy_count=2, greedy_count=1, neighborhood_count=1, greedy_topk=1
            ),
            runtime=runtime,
        )
        candidates = baseline.propose(
            _context(lambda *_args: ["unused"], _fake_score),
            self._history(),
            3,
        )
        self.assertEqual(
            [item.solution for item in candidates],
            ["word-1-0", "word-1-1", "word-2-0"],
        )
        self.assertEqual(
            [item.metadata["provenance"] for item in candidates],
            ["online", "online", "neighborhood"],
        )
        self.assertEqual(runtime.calls[0], ("Generate a word.", 2))
        self.assertIn("solution: warm-b", runtime.calls[1][0])
        self.assertNotIn("solution: warm-a", runtime.calls[1][0])
        self.assertEqual(candidates[0].metadata["prompt"], "Generate a word.")
        self.assertIn("High-scoring solutions", candidates[2].metadata["prompt"])

    def test_observe_updates_on_mixed_group_wrt_task_prompt(self):
        runtime = _RecordingPolicy()
        baseline = MiGrATeBaseline(
            MiGrATeConfig(
                on_policy_count=1, greedy_count=1, neighborhood_count=1, greedy_topk=1
            ),
            runtime=runtime,
        )
        context = _context(lambda *_args: ["unused"], _fake_score)
        baseline.propose(context, self._history(), 2)
        baseline.observe(
            [
                BaselineObservation(index=3, solution="online-x", score=0.5),
                BaselineObservation(index=4, solution="ns-y", score=0.2),
            ]
        )
        self.assertEqual(len(runtime.groups), 1)
        prompt, group = runtime.groups[0]
        self.assertEqual(prompt, "Generate a word.")
        self.assertEqual(group[0], ("warm-b", 0.9))
        self.assertEqual(group[1:], [("online-x", 0.5), ("ns-y", 0.2)])

    def test_grpo_loss_is_mean_of_sequence_means_and_masks_zero_reward(self):
        import torch

        short = torch.zeros(1)
        long = torch.zeros(4)
        old_short = torch.zeros(1)
        old_long = torch.zeros(4)
        loss = _grpo_loss(
            [short, long],
            [old_short, old_long],
            [1.0, 0.0],
            [1.0, 1.0],
            0.2,
            0.28,
        )
        # Sequence means -1 and 0, not token-mean -1/5.
        self.assertAlmostEqual(float(loss), -0.5)
        masked = _grpo_loss(
            [short, long],
            [old_short, old_long],
            [1.0, 1.0],
            [1.0, 0.0],
            0.2,
            0.28,
        )
        self.assertAlmostEqual(float(masked), -0.5)

    def test_neighborhood_uses_topk_even_without_greedy_insert(self):
        runtime = _RecordingPolicy()
        baseline = MiGrATeBaseline(
            MiGrATeConfig(
                on_policy_count=0,
                greedy_count=0,
                neighborhood_count=1,
                greedy_topk=1,
            ),
            runtime=runtime,
        )
        context = _context(lambda *_args: ["unused"], _fake_score)
        candidates = baseline.propose(context, self._history(), 1)
        self.assertEqual(
            [item.metadata["provenance"] for item in candidates],
            ["neighborhood"],
        )
        self.assertIn("solution: warm-b", runtime.calls[0][0])
        baseline.observe(
            [BaselineObservation(index=3, solution="ns-only", score=0.4)]
        )
        self.assertEqual(runtime.groups[0][1], [("ns-only", 0.4)])

    def test_empty_history_falls_back_to_task_prompt_for_neighborhood(self):
        runtime = _RecordingPolicy()
        candidates = MiGrATeBaseline(
            MiGrATeConfig(on_policy_count=1, greedy_count=1, neighborhood_count=1),
            runtime=runtime,
        ).propose(
            _context(lambda *_args: ["unused"], _fake_score),
            [],
            2,
        )
        self.assertEqual(
            [item.metadata["provenance"] for item in candidates],
            ["online", "neighborhood"],
        )
        self.assertEqual(
            runtime.calls,
            [("Generate a word.", 1), ("Generate a word.", 1)],
        )

    def test_repeat_proposals_are_kept_and_flagged(self):
        class _FixedPolicy(_RecordingPolicy):
            def generate(self, prompt, count, options):
                self.calls.append((prompt, count))
                return ["warm-a", "fresh"][:count]

        runtime = _FixedPolicy()
        candidates = MiGrATeBaseline(
            MiGrATeConfig(on_policy_count=2, greedy_count=0, neighborhood_count=0),
            runtime=runtime,
        ).propose(
            _context(lambda *_args: ["unused"], _fake_score),
            self._history(),
            2,
        )
        self.assertEqual(
            [item.solution for item in candidates],
            ["warm-a", "fresh"],
        )
        self.assertTrue(candidates[0].metadata["is_repeat_proposal"])
        self.assertFalse(candidates[1].metadata["is_repeat_proposal"])

    def test_blank_samples_are_rejected(self):
        class _BlankPolicy(_RecordingPolicy):
            def generate(self, _prompt, _count, _options):
                return ["  "]

        with self.assertRaisesRegex(ValueError, "blank samples"):
            MiGrATeBaseline(runtime=_BlankPolicy()).propose(
                _context(lambda *_args: ["unused"], _fake_score),
                [],
                1,
            )

    def test_state_round_trips_runtime(self):
        runtime = _RecordingPolicy()
        baseline = MiGrATeBaseline(
            MiGrATeConfig(on_policy_count=1, neighborhood_count=0),
            runtime=runtime,
        )
        baseline.propose(
            _context(lambda *_args: ["unused"], _fake_score),
            self._history(),
            1,
        )
        restored_runtime = _RecordingPolicy()
        restored = MiGrATeBaseline(
            MiGrATeConfig(on_policy_count=1, neighborhood_count=0),
            runtime=restored_runtime,
        )
        restored.load_state_dict(baseline.state_dict())
        self.assertEqual(restored_runtime.loaded, {"marker": True})

    def test_lora_policy_updates_on_a_tiny_causal_lm(self):
        import torch

        class _Tokenizer:
            pad_token_id = 0
            eos_token_id = 1

            def __call__(self, text, add_special_tokens=True, return_tensors=None, **_kwargs):
                ids = [2 + (ord(char) % 29) for char in str(text)[:24]] or [2]
                if add_special_tokens:
                    ids = [2] + ids
                tokens = torch.tensor([ids], dtype=torch.long)
                mask = torch.ones_like(tokens)
                if return_tensors == "pt":
                    return {"input_ids": tokens, "attention_mask": mask}
                return {"input_ids": ids, "attention_mask": [1] * len(ids)}

            def decode(self, ids, skip_special_tokens=True):
                del ids, skip_special_tokens
                return "tiny"

        class _TinyLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(32, 8)
                self.q_proj = torch.nn.Linear(8, 8)
                self.lm_head = torch.nn.Linear(8, 32)
                self.seen_training: list[bool] = []

            def forward(self, input_ids, attention_mask=None, **_kwargs):
                del attention_mask
                self.seen_training.append(self.training)
                hidden = self.q_proj(self.embed(input_ids))
                hidden = hidden + hidden.cumsum(dim=1)
                return type("Out", (), {"logits": self.lm_head(hidden)})()

            def generate(self, input_ids, max_new_tokens=1, **_kwargs):
                extra = torch.full(
                    (input_ids.shape[0], max_new_tokens),
                    5,
                    dtype=torch.long,
                    device=input_ids.device,
                )
                return torch.cat([input_ids, extra], dim=1)

            @property
            def device(self):
                return next(self.parameters()).device

        model = _TinyLM()
        checkpoint = type(
            "Checkpoint",
            (),
            {
                "reft_model": type("ReFT", (), {"model": model})(),
                "tokenizer": _Tokenizer(),
                "from_chat_template": False,
                "assistant_suffix": None,
                "saved_cfg": {},
                "prompt": "unused",
                "content_span": None,
            },
        )()
        context = _context(
            lambda *_args: ["unused"],
            _fake_score,
            checkpoint=checkpoint,
            model_name="tiny/model",
            task="semantle",
            generation_options=GenerationOptions(max_new_tokens=2, temperature=0.0),
        )
        baseline = MiGrATeBaseline(
            MiGrATeConfig(
                on_policy_count=1,
                greedy_count=1,
                neighborhood_count=1,
                lora_rank=2,
                update_steps=1,
                learning_rate=0.05,
            )
        )
        baseline.propose(context, self._history(), 2)
        self.assertIsInstance(baseline._runtime.optimizer, torch.optim.AdamW)
        before = baseline.state_dict()["adapters"]
        baseline.observe(
            [
                BaselineObservation(index=3, solution="low", score=0.1),
                BaselineObservation(index=4, solution="high", score=0.9),
            ]
        )
        after = baseline.state_dict()["adapters"]
        self.assertNotEqual(before, after)
        self.assertTrue(after)
        self.assertTrue(model.seen_training)
        self.assertFalse(any(model.seen_training))


class DiscreteBOTest(unittest.TestCase):
    _pool = ("alpha", "beta", "gamma", "delta")

    def _history(self) -> list[BaselineObservation]:
        return [
            BaselineObservation(index=0, solution="warm-a", score=0.1),
            BaselineObservation(index=1, solution="warm-b", score=0.4),
        ]

    def _config(self, **kwargs) -> DiscreteBOConfig:
        fields = {"candidate_count": 4}
        return DiscreteBOConfig(**{**fields, **kwargs})

    def _propose(self, scores, history=None, replies=None, batch=1, **config):
        replies = self._pool if replies is None else replies
        calls: list[tuple[str, int]] = []

        def generate(prompt, count, _options):
            calls.append((prompt, count))
            return list(replies[:count])

        with patch(
            "boreft.baselines.discrete_bo.fit_surrogate", return_value=object()
        ), patch(
            "boreft.baselines.discrete_bo.score_discrete_candidates",
            return_value=np.asarray(scores, dtype=np.float64),
        ):
            candidates = create_baseline("discrete_bo", self._config(**config)).propose(
                _context(generate, _fake_score, embed=_fake_embed),
                self._history() if history is None else history,
                batch,
            )
        return calls, candidates

    def test_rejects_file_source_without_path(self):
        with self.assertRaisesRegex(ValueError, "candidate_file"):
            DiscreteBOConfig(candidate_source="file")

    def test_rejects_unknown_acquisition(self):
        with self.assertRaisesRegex(ValueError, "acquisition"):
            DiscreteBOConfig(acquisition="ei")
        self.assertEqual(
            DiscreteBOConfig(acquisition="random", candidate_count=4).acquisition,
            "random",
        )

    def test_rejects_nonpositive_candidate_count(self):
        with self.assertRaisesRegex(ValueError, "candidate_count"):
            DiscreteBOConfig(candidate_count=0)

    def test_rejects_minus_one_candidate_count_unless_train(self):
        with self.assertRaisesRegex(ValueError, "candidate_source=train"):
            DiscreteBOConfig(candidate_count=-1)

    def test_rejects_llm_sample_knobs_unless_llm_source(self):
        with self.assertRaisesRegex(ValueError, "last_k_incontext only applies"):
            DiscreteBOConfig(candidate_source="train", last_k_incontext=3)
        with self.assertRaisesRegex(ValueError, "candidates_per_call only applies"):
            DiscreteBOConfig(
                candidate_source="file",
                candidate_file="pool.jsonl",
                candidates_per_call=4,
            )

    def test_llm_pool_parses_newline_separated_candidates(self):
        calls: list[tuple[str, int]] = []

        def generate(prompt, count, _options):
            calls.append((prompt, count))
            return ["1. alpha\n2. beta\n3. gamma\n4. delta"]

        with patch(
            "boreft.baselines.discrete_bo.fit_surrogate", return_value=object()
        ), patch(
            "boreft.baselines.discrete_bo.score_discrete_candidates",
            return_value=np.asarray([0.1, 0.4, 0.2, 0.9], dtype=np.float64),
        ):
            candidates = create_baseline(
                "discrete_bo",
                self._config(candidates_per_call=4),
            ).propose(
                _context(generate, _fake_score, embed=_fake_embed),
                self._history(),
                1,
            )
        self.assertEqual(calls, [
            (
                "Generate a word.\n\nGenerate 4 candidates, one numbered new line each "
                "(e.g., 1. word1\n2. word2\n...). "
                "Reply with only the numbered candidates.",
                1,
            )
        ])
        self.assertEqual(candidates[0].solution, "delta")

    def test_llm_pool_last_k_incontext_conditions_on_previous_unique_samples(self):
        calls: list[tuple[str, int]] = []

        def generate(prompt, count, _options):
            calls.append((prompt, count))
            return [f"w{len(calls)}"]

        with patch(
            "boreft.baselines.discrete_bo.fit_surrogate", return_value=object()
        ), patch(
            "boreft.baselines.discrete_bo.score_discrete_candidates",
            return_value=np.asarray([0.1, 0.2, 0.3, 0.9], dtype=np.float64),
        ):
            candidates = create_baseline(
                "discrete_bo",
                self._config(last_k_incontext=2),
            ).propose(
                _context(generate, _fake_score, embed=_fake_embed),
                self._history(),
                1,
            )
        self.assertEqual([count for _prompt, count in calls], [1, 1, 1, 1])
        self.assertNotIn("Previous candidates:", calls[0][0])
        self.assertIn("w1", calls[1][0])
        self.assertNotIn("w2", calls[1][0])
        self.assertIn("w1", calls[2][0])
        self.assertIn("w2", calls[2][0])
        self.assertNotIn("w3", calls[2][0])
        self.assertIn("w2", calls[3][0])
        self.assertIn("w3", calls[3][0])
        self.assertNotIn("w1", calls[3][0])
        self.assertEqual(candidates[0].solution, "w4")

    def test_requires_embed(self):
        calls: list[int] = []

        def generate(_prompt, count, _options):
            calls.append(count)
            return list(self._pool[:count])

        with self.assertRaisesRegex(ValueError, "embed"):
            create_baseline("discrete_bo", self._config()).propose(
                _context(generate, _fake_score),
                self._history(),
                1,
            )
        self.assertEqual(calls, [])

    def test_requires_two_unique_observations(self):
        calls: list[int] = []

        def generate(_prompt, count, _options):
            calls.append(count)
            return list(self._pool[:count])

        with self.assertRaisesRegex(ValueError, "at least two unique"):
            create_baseline("discrete_bo", self._config()).propose(
                _context(generate, _fake_score, embed=_fake_embed),
                self._history()[:1],
                1,
            )
        self.assertEqual(calls, [])

    def test_random_samples_unobserved_pool_members_without_a_surrogate(self):
        def generate(_prompt, count, _options):
            return list(self._pool[:count])

        def expected(seed: int, history, count: int) -> list[str]:
            observed = {solution_key(item.solution) for item in history}
            remaining = [
                solution
                for solution in self._pool
                if solution_key(solution) not in observed
            ]
            scores = np.random.default_rng(seed * 1_000_003 + len(history)).random(
                len(remaining)
            )
            chosen = np.argsort(-scores, kind="stable")[:count]
            return [remaining[int(index)] for index in chosen]

        history = self._history()
        skipped_history = history + [
            BaselineObservation(index=2, solution="Alpha", score=0.2),
        ]
        with patch(
            "boreft.baselines.discrete_bo.fit_surrogate",
            side_effect=AssertionError("random acquisition must not fit a GP"),
        ), patch(
            "boreft.baselines.discrete_bo.score_discrete_candidates",
            side_effect=AssertionError("random acquisition must not score a GP"),
        ):
            baseline = create_baseline(
                "discrete_bo", self._config(acquisition="random")
            )
            context = _context(generate, _fake_score)
            first = baseline.propose(context, history, 2)
            again = baseline.propose(context, history, 2)
            different_seed = baseline.propose(
                _context(generate, _fake_score, seed=2),
                history,
                2,
            )
            skipped = baseline.propose(context, skipped_history, 3)
        self.assertEqual(
            [item.solution for item in first], expected(1, history, 2)
        )
        self.assertEqual(
            [item.solution for item in again], [item.solution for item in first]
        )
        self.assertEqual(
            [item.solution for item in different_seed], expected(2, history, 2)
        )
        self.assertEqual(
            [item.solution for item in skipped],
            expected(1, skipped_history, 3),
        )
        self.assertNotIn("alpha", {item.solution.casefold() for item in skipped})
        self.assertFalse(first[0].metadata["is_repeat_proposal"])
        self.assertEqual(first[0].metadata["rank"], 0)
        self.assertIn("pool_index", first[0].metadata)

    def test_samples_a_frozen_pool_from_the_task_prompt(self):
        calls, candidates = self._propose([0.1, 0.4, 0.2, 0.9])
        self.assertEqual(calls, [("Generate a word.", 4)])
        self.assertEqual(candidates[0].solution, "delta")
        self.assertEqual(candidates[0].metadata["pool_index"], 3)
        self.assertEqual(candidates[0].metadata["rank"], 0)
        self.assertFalse(candidates[0].metadata["is_repeat_proposal"])

    def test_skips_pool_members_already_observed(self):
        history = self._history() + [
            BaselineObservation(index=2, solution="Alpha", score=0.2),
        ]
        _, candidates = self._propose([0.2, 0.1, 0.8], history=history)
        self.assertEqual(candidates[0].solution, "delta")
        self.assertEqual(candidates[0].metadata["pool_index"], 3)

    def test_appends_the_gold_target_when_requested(self):
        _, candidates = self._propose(
            [0.1, 0.1, 0.1, 0.1, 0.9],
            include_target=True,
        )
        self.assertEqual(candidates[0].solution, "target")
        self.assertEqual(candidates[0].metadata["pool_index"], 4)

    def _semantle_file_pool_logs(self, *, include_target: bool, target: str):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool.jsonl"
            path.write_text(
                '{"solution": "one"}\n{"solution": "two"}\n'
                '{"solution": "three"}\n{"solution": "four"}\n',
                encoding="utf-8",
            )
            n_scores = 5 if include_target and target.casefold() not in {
                "one",
                "two",
                "three",
                "four",
            } else 4
            with patch(
                "boreft.baselines.discrete_bo.fit_surrogate", return_value=object()
            ), patch(
                "boreft.baselines.discrete_bo.score_discrete_candidates",
                return_value=np.ones(n_scores, dtype=np.float64),
            ), patch("builtins.print") as printed:
                create_baseline(
                    "discrete_bo",
                    self._config(
                        candidate_source="file",
                        candidate_file=str(path),
                        include_target=include_target,
                    ),
                ).propose(
                    _context(
                        lambda *_args: [],
                        _fake_score,
                        embed=_fake_embed,
                        target=target,
                        task="semantle",
                    ),
                    self._history(),
                    1,
                )
        messages = [
            " ".join(str(arg) for arg in call.args)
            for call in printed.call_args_list
            if call.args and str(call.args[0]).startswith("[discrete_bo]")
        ]
        self.assertEqual(len(messages), 1)
        self.assertIn("pool:", messages[0])
        self.assertIn("source=file", messages[0])
        return messages[0]

    def test_semantle_logs_when_gold_target_is_already_in_the_pool(self):
        message = self._semantle_file_pool_logs(include_target=True, target="Two")
        self.assertIn("already in the pool", message)
        self.assertIn("did not add it", message)
        self.assertIn("'Two'", message)

    def test_semantle_logs_when_gold_target_is_added(self):
        message = self._semantle_file_pool_logs(include_target=True, target="gold")
        self.assertIn("not in the pool", message)
        self.assertIn("added it", message)
        self.assertIn("'gold'", message)

    def test_semantle_logs_when_gold_target_is_missing_and_not_added(self):
        message = self._semantle_file_pool_logs(include_target=False, target="gold")
        self.assertIn("not in the pool", message)
        self.assertIn("did not add it", message)

    def test_non_semantle_tasks_log_pool_metrics_without_gold_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool.jsonl"
            path.write_text(
                '{"solution": "one"}\n{"solution": "two"}\n'
                '{"solution": "three"}\n{"solution": "four"}\n',
                encoding="utf-8",
            )
            with patch(
                "boreft.baselines.discrete_bo.fit_surrogate", return_value=object()
            ), patch(
                "boreft.baselines.discrete_bo.score_discrete_candidates",
                return_value=np.ones(4, dtype=np.float64),
            ), patch("builtins.print") as printed:
                create_baseline(
                    "discrete_bo",
                    self._config(candidate_source="file", candidate_file=str(path)),
                ).propose(
                    _context(
                        lambda *_args: [],
                        _fake_score,
                        embed=_fake_embed,
                        task="molopt",
                    ),
                    self._history(),
                    1,
                )
        messages = [
            " ".join(str(arg) for arg in call.args)
            for call in printed.call_args_list
            if call.args and str(call.args[0]).startswith("[discrete_bo]")
        ]
        self.assertEqual(len(messages), 1)
        self.assertIn("pool: 4 unique", messages[0])
        self.assertNotIn("gold target", messages[0])

    def test_file_pool_does_not_call_generate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool.jsonl"
            path.write_text(
                '{"solution": "one"}\n{"text": "two"}\n"three"\n{"target": "four"}\n',
                encoding="utf-8",
            )
            calls: list[int] = []

            def generate(_prompt, count, _options):
                calls.append(count)
                return ["should-not-sample"]

            with patch(
                "boreft.baselines.discrete_bo.fit_surrogate", return_value=object()
            ), patch(
                "boreft.baselines.discrete_bo.score_discrete_candidates",
                return_value=np.asarray([0.1, 0.9, 0.2, 0.3], dtype=np.float64),
            ):
                candidates = create_baseline(
                    "discrete_bo",
                    self._config(candidate_source="file", candidate_file=str(path)),
                ).propose(
                    _context(generate, _fake_score, embed=_fake_embed),
                    self._history(),
                    1,
                )
        self.assertEqual(calls, [])
        self.assertEqual(candidates[0].solution, "two")

    def test_train_pool_reads_unique_words_from_items_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "items.json").write_text(
                json.dumps(
                    [
                        {
                            "id": 0,
                            "prompt": "p",
                            "target": "alpha </s>",
                            "word": "alpha",
                        },
                        {"id": 1, "prompt": "p", "target": "beta"},
                        {"id": 2, "prompt": "p", "target": "Alpha", "word": "Alpha"},
                        {"id": 3, "prompt": "p", "target": "gamma"},
                        {"id": 4, "prompt": "p", "target": "delta"},
                    ]
                ),
                encoding="utf-8",
            )
            calls: list[int] = []

            def generate(_prompt, count, _options):
                calls.append(count)
                return ["should-not-sample"]

            with patch(
                "boreft.baselines.discrete_bo.fit_surrogate", return_value=object()
            ), patch(
                "boreft.baselines.discrete_bo.score_discrete_candidates",
                return_value=np.asarray([0.1, 0.2, 0.9, 0.3], dtype=np.float64),
            ):
                candidates = create_baseline(
                    "discrete_bo",
                    self._config(candidate_source="train"),
                ).propose(
                    _context(
                        generate,
                        _fake_score,
                        embed=_fake_embed,
                        checkpoint_dir=tmp,
                    ),
                    self._history(),
                    1,
                )
        self.assertEqual(calls, [])
        self.assertEqual(candidates[0].solution, "gamma")
        self.assertEqual(candidates[0].metadata["pool_index"], 2)

    def test_train_item_solution_unwraps_mist_tags(self):
        tagged = {"target": "[START_SMILES] CCO [END_SMILES]"}
        self.assertEqual(_train_item_solution(tagged, Path("items.json"), 0), "CCO")
        self.assertEqual(
            _train_item_solution({"word": "CCO [END_SMILES]"}, Path("items.json"), 1),
            "CCO",
        )

    def test_train_pool_uses_all_unique_items_when_fewer_than_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "items.json").write_text(
                json.dumps(
                    [
                        {"word": "one"},
                        {"word": "two"},
                        {"word": "three"},
                    ]
                ),
                encoding="utf-8",
            )
            with patch(
                "boreft.baselines.discrete_bo.fit_surrogate", return_value=object()
            ), patch(
                "boreft.baselines.discrete_bo.score_discrete_candidates",
                return_value=np.asarray([0.1, 0.9, 0.2], dtype=np.float64),
            ):
                candidates = create_baseline(
                    "discrete_bo",
                    self._config(candidate_source="train", candidate_count=8),
                ).propose(
                    _context(
                        lambda *_args: [],
                        _fake_score,
                        embed=_fake_embed,
                        checkpoint_dir=tmp,
                    ),
                    self._history(),
                    1,
                )
        self.assertEqual(candidates[0].solution, "two")

    def test_train_pool_samples_a_seeded_random_subset(self):
        vocab = [f"word-{index}" for index in range(8)]
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "items.json").write_text(
                json.dumps([{"word": word} for word in vocab]),
                encoding="utf-8",
            )

            def pool_for(seed: int) -> list[str]:
                baseline = create_baseline(
                    "discrete_bo",
                    self._config(candidate_source="train", candidate_count=3),
                )
                with patch(
                    "boreft.baselines.discrete_bo.fit_surrogate",
                    return_value=object(),
                ), patch(
                    "boreft.baselines.discrete_bo.score_discrete_candidates",
                    return_value=np.asarray([0.3, 0.1, 0.2], dtype=np.float64),
                ):
                    baseline.propose(
                        _context(
                            lambda *_args: [],
                            _fake_score,
                            embed=_fake_embed,
                            checkpoint_dir=tmp,
                            seed=seed,
                        ),
                        self._history(),
                        1,
                    )
                return baseline.state_dict()["pool"]

            first = pool_for(1)
            again = pool_for(1)
            second = pool_for(2)
        self.assertEqual(first, again)
        self.assertEqual(first, random.Random(1).sample(vocab, 3))
        self.assertEqual(second, random.Random(2).sample(vocab, 3))
        self.assertNotEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertTrue(set(first).issubset(vocab))

    def test_train_pool_minus_one_uses_all_unique_items(self):
        vocab = ["one", "two", "three", "four", "five"]
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "items.json").write_text(
                json.dumps([{"word": word} for word in vocab]),
                encoding="utf-8",
            )
            baseline = create_baseline(
                "discrete_bo",
                self._config(candidate_source="train", candidate_count=-1),
            )
            with patch(
                "boreft.baselines.discrete_bo.fit_surrogate", return_value=object()
            ), patch(
                "boreft.baselines.discrete_bo.score_discrete_candidates",
                return_value=np.asarray([0.1, 0.2, 0.9, 0.3, 0.4], dtype=np.float64),
            ):
                candidates = baseline.propose(
                    _context(
                        lambda *_args: [],
                        _fake_score,
                        embed=_fake_embed,
                        checkpoint_dir=tmp,
                    ),
                    self._history(),
                    1,
                )
        self.assertEqual(baseline.state_dict()["pool"], vocab)
        self.assertEqual(candidates[0].solution, "three")

    def test_train_pool_requires_checkpoint_dir(self):
        with self.assertRaisesRegex(ValueError, "checkpoint_dir"):
            create_baseline(
                "discrete_bo", self._config(candidate_source="train")
            ).propose(
                _context(lambda *_args: [], _fake_score, embed=_fake_embed),
                self._history(),
                1,
            )

    def test_pool_and_embeddings_round_trip_through_state(self):
        embed_calls: list[list[str]] = []

        def embed(texts):
            embed_calls.append(list(texts))
            return _fake_embed(texts)

        with patch(
            "boreft.baselines.discrete_bo.fit_surrogate", return_value=object()
        ), patch(
            "boreft.baselines.discrete_bo.score_discrete_candidates",
            return_value=np.asarray([0.2, 0.1, 0.4, 0.3], dtype=np.float64),
        ):
            baseline = create_baseline("discrete_bo", self._config())
            context = _context(
                lambda _prompt, count, _options: list(self._pool[:count]),
                _fake_score,
                embed=embed,
            )
            first = baseline.propose(context, self._history(), 1)
            second = baseline.propose(context, self._history(), 1)
            restored = create_baseline("discrete_bo", self._config())
            restored.load_state_dict(baseline.state_dict())
            third = restored.propose(context, self._history(), 1)
        self.assertEqual([item.solution for item in first], ["gamma"])
        self.assertEqual([item.solution for item in second], ["gamma"])
        self.assertEqual([item.solution for item in third], ["gamma"])
        self.assertEqual(embed_calls[0], ["alpha", "beta", "gamma", "delta"])
        self.assertEqual(embed_calls[1], ["warm-a", "warm-b"])
        self.assertEqual(len(embed_calls), 2)
        self.assertEqual(baseline.state_dict()["pool"], list(self._pool))

    def test_injected_pool_survives_empty_or_conflicting_load_state(self):
        injected = ["one", "two", "three", "four"]
        vectors = {
            "one": [1.0, 0.0],
            "two": [0.0, 1.0],
            "three": [0.5, 0.5],
            "four": [0.25, 0.75],
        }
        baseline = DiscreteBOBaseline(
            self._config(), pool=injected, embeddings=vectors
        )
        baseline.load_state_dict({})
        self.assertEqual(baseline.state_dict()["pool"], injected)
        self.assertEqual(baseline.state_dict()["embeddings"]["one"], [1.0, 0.0])
        baseline.load_state_dict({"pool": ["other", "words", "here", "also"]})
        self.assertEqual(baseline.state_dict()["pool"], injected)
        baseline.load_state_dict(
            {"pool": ["other", "words", "here", "also"], "embeddings": {}}
        )
        self.assertEqual(baseline.state_dict()["pool"], injected)
        self.assertEqual(baseline.state_dict()["embeddings"]["one"], [1.0, 0.0])
        baseline.load_state_dict({"embeddings": {"warm-a": [0.5, 0.5]}})
        self.assertEqual(baseline.state_dict()["embeddings"]["one"], [1.0, 0.0])
        self.assertEqual(baseline.state_dict()["embeddings"]["warm-a"], [0.5, 0.5])

    def test_injected_embeddings_are_not_recomputed(self):
        pool = list(self._pool)
        embeddings = {name: _fake_embed([name])[0] for name in pool}
        embed_calls: list[list[str]] = []

        def embed(texts):
            embed_calls.append(list(texts))
            return _fake_embed(texts)

        with patch(
            "boreft.baselines.discrete_bo.fit_surrogate", return_value=object()
        ), patch(
            "boreft.baselines.discrete_bo.score_discrete_candidates",
            return_value=np.asarray([0.2, 0.1, 0.4, 0.3], dtype=np.float64),
        ):
            DiscreteBOBaseline(
                self._config(), pool=pool, embeddings=embeddings
            ).propose(
                _context(
                    lambda *_args: [],
                    _fake_score,
                    embed=embed,
                ),
                self._history(),
                1,
            )
        self.assertEqual(embed_calls, [["warm-a", "warm-b"]])

    def test_exhausted_pool_raises(self):
        history = [
            BaselineObservation(index=index, solution=name, score=0.1 * index)
            for index, name in enumerate(("warm-a", "warm-b") + self._pool)
        ]
        with self.assertRaisesRegex(ValueError, "exhausted"):
            self._propose([0.1], history=history)

    def test_surrogate_fit_config_is_forwarded(self):
        config = self._config(
            surrogate="projected",
            kernel="rbf",
            use_ard=True,
            projection_dim=8,
            projection_layers=2,
            projection_steps=12,
            gp_lr=0.05,
            projection_lr=0.001,
        )
        with patch(
            "boreft.baselines.discrete_bo.fit_surrogate", return_value=object()
        ) as fit, patch(
            "boreft.baselines.discrete_bo.score_discrete_candidates",
            return_value=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        ):
            create_baseline("discrete_bo", config).propose(
                _context(
                    lambda _prompt, count, _options: list(self._pool[:count]),
                    _fake_score,
                    embed=_fake_embed,
                    observation_samples=4,
                ),
                [
                    BaselineObservation(
                        index=0,
                        solution="warm-a",
                        score=0.1,
                        score_sem=0.2,
                        sample_count=4,
                    ),
                    BaselineObservation(
                        index=1,
                        solution="warm-b",
                        score=0.4,
                        score_sem=0.1,
                        sample_count=4,
                    ),
                ],
                1,
            )
        fit_config = fit.call_args.args[3]
        self.assertEqual(fit_config.kind, "projected")
        self.assertEqual(fit_config.kernel, "rbf")
        self.assertTrue(fit_config.use_ard)
        self.assertEqual(fit_config.projection_dim, 8)
        self.assertEqual(fit_config.projection_layers, 2)
        self.assertEqual(fit_config.steps, 12)
        np.testing.assert_allclose(
            fit.call_args.kwargs["observation_variances"], [0.04, 0.01]
        )


class BOPROTest(unittest.TestCase):
    _vectors = {
        "low": [1.0, 0.0, 0.0],
        "mid": [0.1, 0.0, 0.9],
        "high": [0.0, 0.0, 1.0],
    }

    def _history(self) -> list[BaselineObservation]:
        return [
            BaselineObservation(index=0, solution="low", score=0.1),
            BaselineObservation(index=1, solution="high", score=0.9),
            BaselineObservation(index=2, solution="mid", score=0.5),
        ]

    def _embed(self, texts):
        return [list(self._vectors[text]) for text in texts]

    def _propose(self, replies, proposal, history=None, **config):
        prompts: list[str] = []

        def generate(prompt, _count, _options):
            prompts.append(prompt)
            return list(replies)

        with patch(
            "boreft.baselines.bopro.fit_surrogate", return_value=object()
        ), patch(
            "boreft.baselines.bopro.propose_candidates",
            return_value=np.asarray([proposal], dtype=np.float32),
        ):
            candidates = create_baseline("bopro", BOPROConfig(**config)).propose(
                _context(generate, _fake_score, embed=self._embed),
                self._history() if history is None else history,
                1,
            )
        return prompts, candidates

    def test_rejects_nonpositive_neighbors(self):
        with self.assertRaisesRegex(ValueError, "neighbors"):
            BOPROConfig(neighbors=0)

    def test_rejects_nonpositive_projection_layers(self):
        with self.assertRaisesRegex(ValueError, "projection_layers"):
            BOPROConfig(projection_layers=0)

    def test_requires_embed(self):
        with self.assertRaisesRegex(ValueError, "embed"):
            create_baseline("bopro").propose(
                _context(lambda *_args: ["fresh"], _fake_score),
                self._history(),
                1,
            )

    def test_prompt_uses_nearest_neighbors_sorted_by_score(self):
        prompts, candidates = self._propose(
            ["fresh"], [0.0, 0.0, 1.0], neighbors=2
        )
        prompt = prompts[0]
        self.assertTrue(prompt.startswith("Generate a word.\n\n"))
        self.assertIn("solution: mid", prompt)
        self.assertIn("solution: high", prompt)
        self.assertNotIn("solution: low", prompt)
        self.assertLess(prompt.index("solution: mid"), prompt.index("solution: high"))
        self.assertEqual(candidates[0].metadata["history_indices"], [2, 1])
        self.assertEqual(candidates[0].metadata["proposal"], [0.0, 0.0, 1.0])
        self.assertTrue(candidates[0].metadata["is_repeat_proposal"])
        self.assertEqual(candidates[0].solution, "fresh")

    def test_novel_proposal_is_not_a_repeat(self):
        _, candidates = self._propose(
            ["fresh"], [0.0, 1.0, 0.0], neighbors=1
        )
        self.assertFalse(candidates[0].metadata["is_repeat_proposal"])

    def test_caches_embeddings_across_propose_and_state(self):
        calls: list[list[str]] = []

        def embed(texts):
            calls.append(list(texts))
            return self._embed(texts)

        with patch(
            "boreft.baselines.bopro.fit_surrogate", return_value=object()
        ), patch(
            "boreft.baselines.bopro.propose_candidates",
            return_value=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        ):
            baseline = create_baseline("bopro", BOPROConfig(neighbors=1))
            context = _context(
                lambda _prompt, _count, _options: ["fresh"],
                _fake_score,
                embed=embed,
            )
            baseline.propose(context, self._history(), 1)
            baseline.propose(context, self._history(), 1)
            restored = create_baseline("bopro", BOPROConfig(neighbors=1))
            restored.load_state_dict(baseline.state_dict())
            restored.propose(context, self._history(), 1)
        self.assertEqual(calls, [["low", "high", "mid"]])
        self.assertEqual(set(baseline.state_dict()["embeddings"]), {"low", "high", "mid"})

    def test_blank_samples_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "blank samples"):
            self._propose(["solution:"], [0.0, 0.0, 1.0])

    def test_surrogate_uses_mean_variances_when_scores_are_repeated(self):
        history = [
            BaselineObservation(
                index=0,
                solution="low",
                score=0.1,
                score_std=0.4,
                score_sem=0.2,
                sample_count=4,
                sample_scores=[0.1, 0.1, 0.1, 0.1],
            ),
            BaselineObservation(
                index=1,
                solution="high",
                score=0.9,
                score_std=0.2,
                score_sem=0.1,
                sample_count=4,
                sample_scores=[0.9, 0.9, 0.9, 0.9],
            ),
        ]
        with patch(
            "boreft.baselines.bopro.fit_surrogate", return_value=object()
        ) as fit, patch(
            "boreft.baselines.bopro.propose_candidates",
            return_value=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        ) as propose:
            create_baseline("bopro", BOPROConfig(neighbors=1)).propose(
                _context(
                    lambda _prompt, _count, _options: ["fresh"],
                    _fake_score,
                    embed=self._embed,
                    observation_samples=4,
                ),
                history,
                1,
            )
        np.testing.assert_allclose(
            fit.call_args.kwargs["observation_variances"], [0.04, 0.01]
        )
        self.assertEqual(propose.call_args.kwargs["observation_samples"], 4)

    def test_surrogate_fit_config_is_forwarded(self):
        history = self._history()
        config = BOPROConfig(
            neighbors=1,
            surrogate="projected",
            kernel="rbf",
            use_ard=True,
            projection_dim=8,
            projection_layers=2,
            projection_steps=12,
            gp_lr=0.05,
            projection_lr=0.001,
        )
        with patch(
            "boreft.baselines.bopro.fit_surrogate", return_value=object()
        ) as fit, patch(
            "boreft.baselines.bopro.propose_candidates",
            return_value=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        ):
            create_baseline("bopro", config).propose(
                _context(
                    lambda _prompt, _count, _options: ["fresh"],
                    _fake_score,
                    embed=self._embed,
                ),
                history,
                1,
            )
        fit_config = fit.call_args.args[3]
        self.assertEqual(fit_config.kind, "projected")
        self.assertEqual(fit_config.kernel, "rbf")
        self.assertTrue(fit_config.use_ard)
        self.assertEqual(fit_config.projection_dim, 8)
        self.assertEqual(fit_config.projection_layers, 2)
        self.assertEqual(fit_config.steps, 12)
        self.assertEqual(fit_config.gp_lr, 0.05)
        self.assertEqual(fit_config.projection_lr, 0.001)


class BaselineEndToEndTest(unittest.TestCase):
    def test_random_sampling_run_writes_main_search_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _run_config(tmp, seeds=(1, 2), budget=4, warmstart_count=2)
            with patch(
                "boreft.baselines.search.load_backend",
                side_effect=lambda _config: _fake_backend(),
            ):
                aggregate = run_search(config)
            manifest = artifact_manifest(config)
            for path in (manifest["config"], manifest["summary"], manifest["plot"]):
                self.assertTrue(path.is_file(), path)
            for seed in (1, 2):
                paths = manifest["seeds"][seed]
                self.assertTrue(paths["observations"].is_file())
                self.assertTrue(paths["summary"].is_file())
                self.assertTrue(paths["plot"].is_file())
                summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
                self.assertEqual(summary["n_warmstart"], 2)
                self.assertEqual(summary["n_acquired"], 2)
                self.assertEqual(summary["n_observations"], 4)
                self.assertGreater(summary["elapsed_seconds"], 0.0)
                self.assertGreaterEqual(summary["bbox_eval_seconds"], 0.0)
        self.assertEqual(aggregate["seeds"], [1, 2])
        self.assertEqual(aggregate["runs"]["1"]["seed"], 1)
        self.assertGreater(aggregate["elapsed_seconds"], 0.0)
        self.assertIn("found_target", aggregate["runs"]["1"])
        self.assertFalse(aggregate["runs"]["1"]["found_target"])
        self.assertEqual(aggregate["n_found_target"], 0)

    def test_lora_sft_adapter_must_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "missing.pt")
            with self.assertRaisesRegex(ValueError, "lora_sft_adapter"):
                _run_config(tmp, lora_sft_adapter=missing).validate()

    def test_run_search_reapplies_lora_sft_adapter_after_each_seed_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp) / "epoch009.pt"
            adapter.write_bytes(b"x")
            config = _run_config(
                tmp,
                seeds=(1, 2),
                budget=4,
                warmstart_count=2,
                lora_sft_adapter=str(adapter),
            )
            with patch(
                "boreft.baselines.search.load_backend",
                side_effect=lambda _config: _fake_backend(),
            ), patch("boreft.baselines.search.clear_checkpoint_lora") as clear, patch(
                "boreft.baselines.search._apply_lora_sft_adapter"
            ) as apply:
                run_search(config)
            self.assertEqual(clear.call_count, 2)
            self.assertEqual(apply.call_count, 2)

    def test_opro_run_writes_main_search_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _run_config(
                tmp, baseline="opro", seeds=(1,), budget=4, warmstart_count=2
            )
            with patch(
                "boreft.baselines.search.load_backend",
                side_effect=lambda _config: _fake_backend(),
            ):
                run_search(config)
            summary = json.loads(
                artifact_manifest(config)["seeds"][1]["summary"].read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["n_warmstart"], 2)
            self.assertEqual(summary["n_acquired"], 2)
            observations = BaselineRunState.load(
                artifact_manifest(config)["seeds"][1]["observations"]
            ).observations
            search_obs = [item for item in observations if item.phase == "search"]
            self.assertTrue(search_obs)
            self.assertIn(
                "Propose a new solution that scores higher than those above.",
                search_obs[0].candidate_metadata["prompt"],
            )
            self.assertIn("solution: w", search_obs[0].candidate_metadata["prompt"])

    def test_sdpo_ttt_run_writes_main_search_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _run_config(
                tmp, baseline="sdpo_ttt", seeds=(1,), budget=4, warmstart_count=2
            )
            with patch(
                "boreft.baselines.search.load_backend",
                side_effect=lambda _config: _fake_backend(),
            ):
                run_search(config)
            summary = json.loads(
                artifact_manifest(config)["seeds"][1]["summary"].read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["n_warmstart"], 2)
            self.assertEqual(summary["n_acquired"], 2)
            observations = BaselineRunState.load(
                artifact_manifest(config)["seeds"][1]["observations"]
            ).observations
            search_obs = [item for item in observations if item.phase == "search"]
            self.assertTrue(search_obs)
            self.assertEqual(
                search_obs[0].candidate_metadata["prompt"],
                "Generate a word.",
            )
            self.assertNotIn("Feedback from the latest evaluated batch", search_obs[0].candidate_metadata["prompt"])
            state = load_method_state(
                artifact_manifest(config)["seeds"][1]["method_state"]
            )
            self.assertGreaterEqual(state["consumed_through"], 1)

    def test_run_search_clears_leftover_lora_before_warmstarts(self):
        import torch.nn as nn

        base = nn.Linear(4, 4, bias=False)
        adapter = LoRALinear(base, rank=2, alpha=2.0)
        adapter.lora_B.data.fill_(0.5)
        self.assertGreater(float(adapter.lora_B.detach().abs().sum()), 0)
        residual_at_decode: list[float] = []

        def decode_point(_point):
            residual_at_decode.append(float(adapter.lora_B.detach().abs().sum()))
            if len(residual_at_decode) == 2:
                # First seed's search would leave trained LoRA on the shared
                # backbone; the next seed must clear it before decoding.
                adapter.lora_B.data.fill_(0.5)
            return "w0"

        model = nn.Sequential(adapter)
        checkpoint = type(
            "Checkpoint",
            (),
            {"reft_model": type("ReFT", (), {"model": model})()},
        )()
        with tempfile.TemporaryDirectory() as tmp:
            config = _run_config(
                tmp, seeds=(1, 2), budget=4, warmstart_count=2
            )
            with patch(
                "boreft.baselines.search.load_backend",
                side_effect=lambda _config: _fake_backend(
                    checkpoint=checkpoint, decode_point=decode_point
                ),
            ), patch(
                "boreft.baselines.search.clear_checkpoint_lora",
                wraps=clear_checkpoint_lora,
            ) as cleared:
                run_search(config)
        self.assertEqual(cleared.call_count, 2)
        self.assertEqual(residual_at_decode, [0.0, 0.0, 0.0, 0.0])

    def test_migrate_run_writes_main_search_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _run_config(
                tmp, baseline="migrate", seeds=(1,), budget=6, warmstart_count=2
            )
            with patch(
                "boreft.baselines.search.load_backend",
                side_effect=lambda _config: _fake_backend(),
            ):
                run_search(config)
            summary = json.loads(
                artifact_manifest(config)["seeds"][1]["summary"].read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["n_warmstart"], 2)
            self.assertEqual(summary["n_acquired"], 4)
            observations = BaselineRunState.load(
                artifact_manifest(config)["seeds"][1]["observations"]
            ).observations
            search_obs = [item for item in observations if item.phase == "search"]
            self.assertEqual(len(search_obs), 4)
            self.assertEqual(
                [item.candidate_metadata["provenance"] for item in search_obs],
                ["online", "online", "neighborhood", "neighborhood"],
            )
            self.assertEqual(
                search_obs[0].candidate_metadata["prompt"],
                "Generate a word.",
            )
            self.assertIn(
                "High-scoring solutions",
                search_obs[2].candidate_metadata["prompt"],
            )
            state = load_method_state(
                artifact_manifest(config)["seeds"][1]["method_state"]
            )
            self.assertEqual(state, {})

    def test_autodiscovery_run_writes_main_search_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _run_config(
                tmp, baseline="autodiscovery", seeds=(1,), budget=4, warmstart_count=2
            )
            with patch(
                "boreft.baselines.search.load_backend",
                side_effect=lambda _config: _fake_backend(
                    generate=lambda _prompt, count, _options: [
                        f"g{index}" for index in range(count)
                    ]
                ),
            ):
                run_search(config)
            manifest = artifact_manifest(config)
            summary = json.loads(
                manifest["seeds"][1]["summary"].read_text(encoding="utf-8")
            )
            self.assertEqual(summary["n_warmstart"], 2)
            self.assertEqual(summary["n_acquired"], 2)
            observations = BaselineRunState.load(
                manifest["seeds"][1]["observations"]
            ).observations
            search_obs = [item for item in observations if item.phase == "search"]
            self.assertTrue(search_obs)
            self.assertIn("node_id", search_obs[0].candidate_metadata)
            self.assertIn("parent_id", search_obs[0].candidate_metadata)
            prompt = search_obs[0].candidate_metadata["prompt"]
            self.assertIn(
                "Previous solutions and scores, ordered from lowest score to highest:",
                prompt,
            )
            self.assertIn(
                "Propose a new solution that scores higher than those above.",
                prompt,
            )
            self.assertIn("Reply with only the solution.", prompt)
            self.assertNotIn("build on this branch", prompt)
            state = load_method_state(manifest["seeds"][1]["method_state"])
            self.assertGreaterEqual(len(state["nodes"]), 3)
            self.assertEqual(state["nodes"][0]["node_id"], 0)
            self.assertTrue(
                any(node.get("untried_solutions") for node in state["nodes"])
            )

    def test_bopro_run_writes_main_search_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _run_config(
                tmp, baseline="bopro", seeds=(1,), budget=4, warmstart_count=2
            )
            with patch(
                "boreft.baselines.search.load_backend",
                side_effect=lambda _config: _fake_backend(),
            ):
                run_search(config)
            manifest = artifact_manifest(config)
            summary = json.loads(
                manifest["seeds"][1]["summary"].read_text(encoding="utf-8")
            )
            self.assertEqual(summary["n_warmstart"], 2)
            self.assertEqual(summary["n_acquired"], 2)
            observations = BaselineRunState.load(
                manifest["seeds"][1]["observations"]
            ).observations
            search_obs = [item for item in observations if item.phase == "search"]
            self.assertTrue(search_obs)
            self.assertIn("proposal", search_obs[0].candidate_metadata)
            self.assertIn(
                "Propose a new solution that scores higher than those above.",
                search_obs[0].candidate_metadata["prompt"],
            )
            state = load_method_state(manifest["seeds"][1]["method_state"])
            self.assertTrue(state["embeddings"])

    def test_discrete_bo_run_writes_main_search_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _run_config(
                tmp,
                baseline="discrete_bo",
                seeds=(1,),
                budget=4,
                warmstart_count=2,
                discrete_bo=DiscreteBOConfig(candidate_count=8),
            )
            with patch(
                "boreft.baselines.search.load_backend",
                side_effect=lambda _config: _fake_backend(),
            ):
                run_search(config)
            manifest = artifact_manifest(config)
            summary = json.loads(
                manifest["seeds"][1]["summary"].read_text(encoding="utf-8")
            )
            self.assertEqual(summary["n_warmstart"], 2)
            self.assertEqual(summary["n_acquired"], 2)
            observations = BaselineRunState.load(
                manifest["seeds"][1]["observations"]
            ).observations
            search_obs = [item for item in observations if item.phase == "search"]
            self.assertTrue(search_obs)
            self.assertIn("pool_index", search_obs[0].candidate_metadata)
            state = load_method_state(manifest["seeds"][1]["method_state"])
            self.assertEqual(len(state["pool"]), 8)
            self.assertTrue(state["embeddings"])
            pool_path = manifest["pool"]
            self.assertTrue(pool_path.is_file())
            saved = [
                json.loads(line)["solution"]
                for line in pool_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(saved, state["pool"])

    def test_discrete_bo_random_acquisition_skips_pool_embeddings(self):
        embed_calls: list[list[str]] = []

        def counting_embed(texts):
            embed_calls.append(list(texts))
            return _fake_embed(texts)

        with tempfile.TemporaryDirectory() as tmp:
            config = _run_config(
                tmp,
                baseline="discrete_bo",
                seeds=(1,),
                budget=4,
                warmstart_count=2,
                discrete_bo=DiscreteBOConfig(
                    candidate_count=8, acquisition="random"
                ),
            )
            with patch(
                "boreft.baselines.search.load_backend",
                side_effect=lambda _config: _fake_backend(embed=counting_embed),
            ):
                run_search(config)
            manifest = artifact_manifest(config)
            state = load_method_state(manifest["seeds"][1]["method_state"])
            observations = BaselineRunState.load(
                manifest["seeds"][1]["observations"]
            ).observations
            search_obs = [item for item in observations if item.phase == "search"]
        self.assertEqual(embed_calls, [])
        self.assertEqual(len(state["pool"]), 8)
        self.assertEqual(state["embeddings"], {})
        self.assertEqual(len(search_obs), 2)
        self.assertTrue({item.solution for item in search_obs}.issubset(state["pool"]))
        self.assertIn("pool_index", search_obs[0].candidate_metadata)

    def test_discrete_bo_reuses_one_pool_across_seeds(self):
        embed_calls: list[list[str]] = []

        def counting_embed(texts):
            embed_calls.append(list(texts))
            return _fake_embed(texts)

        with tempfile.TemporaryDirectory() as tmp:
            config = _run_config(
                tmp,
                baseline="discrete_bo",
                seeds=(1, 2),
                budget=4,
                warmstart_count=2,
                discrete_bo=DiscreteBOConfig(candidate_count=8),
            )
            with patch(
                "boreft.baselines.search.load_backend",
                side_effect=lambda _config: _fake_backend(embed=counting_embed),
            ):
                run_search(config)
            manifest = artifact_manifest(config)
            state_1 = load_method_state(manifest["seeds"][1]["method_state"])
            state_2 = load_method_state(manifest["seeds"][2]["method_state"])
            pool_1 = state_1["pool"]
            pool_2 = state_2["pool"]
            saved = [
                json.loads(line)["solution"]
                for line in manifest["pool"].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(saved, pool_1)
        self.assertEqual(pool_1, pool_2)
        self.assertEqual(len(pool_1), 8)
        self.assertEqual(embed_calls[0], pool_1)
        for solution in pool_1:
            self.assertEqual(
                sum(batch.count(solution) for batch in embed_calls),
                1,
            )
        for solution in pool_1:
            self.assertIn(solution, state_1["embeddings"])
            self.assertIn(solution, state_2["embeddings"])

    def test_warmstart_solutions_match_decoded_main_search_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = _fake_backend()
            config = _run_config(tmp, budget=3, warmstart_count=2)
            with patch(
                "boreft.baselines.search.load_backend", return_value=backend
            ):
                run_search(config)
            layout = artifact_manifest(config)["seeds"][1]
            observations = [
                json.loads(line)
                for line in layout["observations"]
                .read_text(encoding="utf-8")
                .splitlines()
            ]
        seed_everything(1)
        points, _ = prepare_warmstarts(
            config,
            checkpoint=backend.checkpoint,
            train_mu=backend.train_mu,
            bounds=backend.bounds,
            seed=1,
        )
        expected = [backend.decode_point(point) for point in points]
        self.assertEqual([item["solution"] for item in observations[:2]], expected)
        self.assertEqual(
            observations[0]["candidate_metadata"]["warmstart_point"],
            [float(points[0][0])],
        )

    def test_resume_continues_a_larger_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "boreft.baselines.search.load_backend",
                side_effect=lambda _config: _fake_backend(),
            ):
                run_search(_run_config(tmp, budget=3, warmstart_count=2))
                first = json.loads(
                    (Path(tmp) / "search" / "seed_1" / "summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                aggregate = run_search(
                    _run_config(tmp, budget=5, warmstart_count=2, resume=True)
                )
            summary = aggregate["runs"]["1"]
        self.assertEqual(first["n_observations"], 3)
        self.assertEqual(summary["n_observations"], 5)
        self.assertEqual(summary["n_warmstart"], 2)
        self.assertGreater(summary["elapsed_seconds"], first["elapsed_seconds"])

    def test_resume_adds_to_previous_summary_wall_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            layout = ArtifactLayout(Path(tmp), 1)
            layout.seed_dir.mkdir(parents=True)
            layout.summary.write_text(
                json.dumps({"elapsed_seconds": 12.5}), encoding="utf-8"
            )
            state = BaselineRunState(layout.observations)
            self.assertEqual(_previous_elapsed(layout, state, True), 12.5)
            self.assertEqual(_previous_elapsed(layout, state, False), 0.0)

    def test_base_generate_renders_chat_prompts(self):
        checkpoint = type(
            "Checkpoint",
            (),
            {
                "from_chat_template": True,
                "assistant_suffix": None,
                "prompt": "unused",
                "content_span": None,
                "saved_cfg": {},
                "tokenizer": object(),
                "reft_model": type("Model", (), {"model": object()})(),
            },
        )()
        generate = _base_generate(checkpoint, "semantle", "fake/model")
        with (
            patch(
                "boreft.baselines.search.build_checkpoint_prompt",
                return_value=("<chat>word</chat>", True, None),
            ) as build,
            patch(
                "boreft.baselines.search.generate_base",
                return_value=" Barn ",
            ) as sample,
        ):
            texts = generate("Generate a word.", 1, GenerationOptions())
        self.assertEqual(texts, ["barn"])
        self.assertEqual(build.call_args.kwargs["user_text"], "Generate a word.")
        self.assertFalse(build.call_args.kwargs["use_checkpoint_prompt"])
        self.assertEqual(sample.call_args.kwargs["from_chat_template"], True)
        self.assertEqual(sample.call_args.args[2], "<chat>word</chat>")

        with (
            patch(
                "boreft.baselines.search.build_checkpoint_prompt",
                return_value=("<chat>word</chat>", True, None),
            ),
            patch(
                "boreft.baselines.search.generate_base",
                return_value=" Cat \nDog\n",
            ),
        ):
            self.assertEqual(
                generate("Generate a word.", 1, GenerationOptions()),
                ["cat\ndog"],
            )

        molopt = _base_generate(checkpoint, "molopt", "fake/model")
        with (
            patch(
                "boreft.baselines.search.build_checkpoint_prompt",
                return_value=("<chat>mol</chat>", True, None),
            ),
            patch(
                "boreft.baselines.search.generate_base",
                return_value=" Barn ",
            ),
        ):
            self.assertEqual(molopt("unused", 1, GenerationOptions()), [" Barn "])


if __name__ == "__main__":
    unittest.main()
