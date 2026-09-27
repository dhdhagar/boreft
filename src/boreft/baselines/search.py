"""Central CLI and artifact contract for standardized baseline searches.

Run with ``python -m boreft.baselines.search``. All methods share this module's
validation, checkpoint/verifier wiring, budget accounting, and file layout so
their artifacts are directly comparable with ``boreft.search`` runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Callable, Literal, Sequence

import numpy as np
import tyro

from boreft.bias_tables import stack_bias_vectors
from boreft.bo import latent_bounds
from boreft.bo.plotting import plot_best_so_far
from boreft.bo.runner import Verifier, _labeled_decoded, seed_everything
from boreft.data_utils import load_merged_run_config, system_prompt_from_cfg
from boreft.eval.semantle import load_eval_checkpoint
from boreft.interactive import (
    build_checkpoint_prompt,
    generate_base,
    maybe_append_decode_smiles_open_tag,
)
from boreft.chem import canonical_target_key, maybe_repair_invalid_smiles, unwrap_smiles_tags
from boreft.search_wandb import WANDB_CONFIG_KEYS, maybe_log_finished_seed
from boreft.search import SearchConfig as MainSearchConfig
from boreft.search import TextSimilarityVerifier
from boreft.search import (
    checkpoint_decode,
    checkpoint_reconstruct_decode,
    memoize_point_decode,
    molopt_search_verifier,
    normalize_search_text,
    resolve_checkpoint_pin_file,
    warmstart_result_fields,
)
from boreft.search import select_warmstarts as main_warmstarts
from boreft.search import warmstart_verification_cost
from boreft.oracles import normalize_property_oracle_name
from boreft.text_similarity import (
    encode_texts_as_is,
    format_text_for_embedding,
    rdkit_map_path_for_cfg,
)

from .autodiscovery import AutoDiscoveryConfig
from .base import (
    ArtifactLayout,
    BaselineName,
    BaselineObservation,
    BaselineRunState,
    Embed,
    Generate,
    GenerationOptions,
    Score,
    ScoreResult,
    SearchContext,
    solution_key,
)
from .bopro import BOPROConfig
from .discrete_bo import (
    DiscreteBOBaseline,
    DiscreteBOConfig,
    DiscreteBOPoolReport,
    coerce_embeddings,
    coerce_pool,
    fill_embeddings,
    log_discrete_bo_pool,
    select_discrete_bo_pool,
    write_discrete_bo_pool,
)
from .migrate import MiGrATeConfig
from .opro import OPROConfig
from .sdpo_ttt import SDPOTTTConfig, clear_checkpoint_lora
from .lora_sft import attach_lora_sft_adapter
from .random_sampling import RandomSamplingConfig
from .registry import available_baselines, create_baseline
from .runner import BaselineLoopConfig, WarmstartSeed, load_method_state, run_baseline

WarmstartSource = Literal["checkpoint", "sobol", "file", "llm", "words"]


@dataclass
class BaselineSearchConfig:
    """Shared CLI fields; algorithm-specific options remain in method modules."""

    task_description: str
    target: str
    reft_output_dir: str
    baseline: BaselineName = "random_sampling"
    task: Literal["semantle", "molopt", "hypogen"] = "semantle"
    oracle: str | None = None
    objective: Literal["embed_sim", "tfs", "rdkit_sim"] = "embed_sim"
    search_dir: str = "baseline_search"
    budget: int = 60
    warmstart_count: int = 10
    warmstart_source: WarmstartSource = "checkpoint"
    warmstart_file: str | None = None
    bounds_padding: float = 0.0
    batch_size: int = 1
    observation_samples: int = 1
    seeds: tuple[int, ...] = (1,)
    repeats: int = 1
    model_name: str | None = None
    layer: int | None = None
    low_rank_dim: int | None = None
    cache_dir: str | None = None
    torch_dtype: Literal["bfloat16", "float16", "float32"] | None = None
    max_new_tokens: int = 128
    sampling_temperature: float = 1.0
    warmstart_temperature: float = 1.0
    warmstart_top_p: float = 0.9
    random_sampling: RandomSamplingConfig = field(
        default_factory=RandomSamplingConfig
    )
    discrete_bo: DiscreteBOConfig = field(default_factory=DiscreteBOConfig)
    opro: OPROConfig = field(default_factory=OPROConfig)
    sdpo_ttt: SDPOTTTConfig = field(default_factory=SDPOTTTConfig)
    bopro: BOPROConfig = field(default_factory=BOPROConfig)
    migrate: MiGrATeConfig = field(default_factory=MiGrATeConfig)
    autodiscovery: AutoDiscoveryConfig = field(default_factory=AutoDiscoveryConfig)
    resume: bool = False
    overwrite: bool = False
    load_latest: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_group: str | None = None
    wandb_dir: str | None = None
    no_wandb: bool = False
    lora_sft_adapter: str | None = None

    def validate(self) -> None:
        if self.oracle:
            self.oracle = normalize_property_oracle_name(self.oracle)
            if self.task != "molopt":
                raise ValueError("property oracles are only valid for task=molopt")
            if not self.target.strip():
                self.target = self.oracle
        if not self.task_description.strip():
            raise ValueError("task_description must not be blank")
        if not self.target.strip():
            raise ValueError("target must not be blank")
        if self.budget < 1:
            raise ValueError("budget must be positive")
        if self.warmstart_count < 2:
            raise ValueError("warmstart_count must be at least 2")
        if self.observation_samples < 1:
            raise ValueError("observation_samples must be positive")
        if (
            warmstart_verification_cost(
                self.warmstart_source,
                self.warmstart_count,
                self.observation_samples,
            )
            > self.budget
        ):
            raise ValueError("warmstart verification cost cannot exceed budget")
        if self.warmstart_source in ("file", "words") and not self.warmstart_file:
            raise ValueError(
                f"warmstart_source={self.warmstart_source} requires warmstart_file"
            )
        if self.bounds_padding < 0:
            raise ValueError("bounds_padding must be nonnegative")
        if self.batch_size < 1 or self.batch_size > self.budget:
            raise ValueError("batch_size must be in [1, budget]")
        if self.max_new_tokens < 1:
            raise ValueError("generation length must be positive")
        if self.sampling_temperature < 0:
            raise ValueError("sampling_temperature must be nonnegative")
        if self.warmstart_temperature <= 0:
            raise ValueError("warmstart_temperature must be positive")
        if not 0 < self.warmstart_top_p <= 1:
            raise ValueError("warmstart_top_p must be in (0, 1]")
        if self.task != "molopt" and self.objective != "embed_sim":
            raise ValueError(f"objective={self.objective!r} is only valid for molopt")
        if self.overwrite:
            # Launcher defaults to --resume; `--overwrite` after `--` must win.
            self.resume = False
        if self.lora_sft_adapter:
            adapter = Path(self.lora_sft_adapter).expanduser()
            if not adapter.is_file():
                raise ValueError(f"lora_sft_adapter does not exist: {adapter}")
            self.lora_sft_adapter = str(adapter.resolve())
        _resolved_seeds(self)
        self.warmstart_file = resolve_checkpoint_pin_file(
            self.reft_output_dir,
            warmstart_source=self.warmstart_source,
            warmstart_file=self.warmstart_file,
        )


def _resolved_seeds(config: BaselineSearchConfig) -> tuple[int, ...]:
    if config.repeats < 1:
        raise ValueError("repeats must be positive")
    if config.repeats == 1:
        seeds = config.seeds
    elif len(config.seeds) != 1:
        raise ValueError("use either multiple --seeds or --repeats, not both")
    else:
        seeds = tuple(config.seeds[0] + offset for offset in range(config.repeats))
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    return seeds


def method_config(config: BaselineSearchConfig) -> object:
    """Return the CLI-populated config belonging to the selected baseline."""
    return getattr(config, config.baseline)


def _checkpoint_lm(checkpoint: object):
    model = getattr(getattr(checkpoint, "reft_model", None), "model", None)
    if model is None:
        raise ValueError("checkpoint has no reft_model.model to attach a LoRA adapter")
    return model


def _apply_lora_sft_adapter(config: BaselineSearchConfig, checkpoint: object) -> None:
    if not config.lora_sft_adapter:
        return
    payload = attach_lora_sft_adapter(_checkpoint_lm(checkpoint), config.lora_sft_adapter)
    epoch = payload.get("epoch")
    print(
        f"[search] loaded LoRA-SFT adapter {config.lora_sft_adapter}"
        + (f" (epoch {epoch})" if epoch is not None else ""),
        flush=True,
    )


def generation_options(config: BaselineSearchConfig) -> GenerationOptions:
    """Build candidate decoding options with main-search top-p semantics."""
    return GenerationOptions(
        temperature=config.sampling_temperature,
        max_new_tokens=config.max_new_tokens,
    )


def _main_warmstart_config(config: BaselineSearchConfig) -> MainSearchConfig:
    """Translate shared fields without reimplementing main-search warmstarts."""
    return MainSearchConfig(
        output_dir=config.reft_output_dir,
        target=config.target,
        objective=config.objective,
        budget=config.budget,
        warmstart_count=config.warmstart_count,
        warmstart_source=config.warmstart_source,
        warmstart_file=config.warmstart_file,
        seeds=config.seeds,
        repeats=config.repeats,
        max_new_tokens=config.max_new_tokens,
        warmstart_temperature=config.warmstart_temperature,
        warmstart_top_p=config.warmstart_top_p,
        model_name=config.model_name,
        layer=config.layer,
        low_rank_dim=config.low_rank_dim,
        cache_dir=config.cache_dir,
        torch_dtype=config.torch_dtype,
    )


def prepare_warmstarts(
    config: BaselineSearchConfig,
    *,
    checkpoint,
    train_mu: np.ndarray,
    bounds: np.ndarray,
    seed: int,
    decode=None,
) -> tuple[np.ndarray, list[dict] | None]:
    """Use the exact main-search warmstart selector for baseline initialization."""
    return main_warmstarts(
        _main_warmstart_config(config),
        checkpoint,
        train_mu,
        bounds,
        seed,
        decode=decode,
    )


def _safe_search_root(search_dir: str, reft_output_dir: str) -> Path:
    root = Path(search_dir).expanduser().resolve()
    checkpoint = Path(reft_output_dir).expanduser().resolve()
    if root == checkpoint or root in checkpoint.parents:
        raise ValueError(
            "search_dir must not equal or contain the ReFT output directory"
        )
    return root


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _validate_resume_config(
    path: Path,
    config: BaselineSearchConfig,
) -> None:
    if not path.exists():
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    current = resolved_config(config)
    mutable = {
        "budget",
        "resume",
        "overwrite",
        "seeds",
        "repeats",
        "max_new_tokens",
        *WANDB_CONFIG_KEYS,
    }
    changed = sorted(
        key
        for key in set(previous) | set(current)
        if key not in mutable and previous.get(key) != current.get(key)
    )
    if changed:
        raise ValueError(
            "resume configuration differs in trajectory-defining fields: "
            + ", ".join(changed)
        )


def artifact_manifest(config: BaselineSearchConfig) -> dict:
    """Return the paths every completed baseline run must produce."""
    root = _safe_search_root(config.search_dir, config.reft_output_dir)
    seeds = _resolved_seeds(config)

    def seed_artifacts(seed: int) -> dict[str, Path]:
        layout = ArtifactLayout(root, seed)
        return {
            "observations": layout.observations,
            "summary": layout.summary,
            "plot": layout.plot,
            "method_state": layout.method_state,
        }

    data = {
        "config": root / "config.json",
        "summary": root / "summary.json",
        "plot": root / "best_so_far.png",
        "seeds": {seed: seed_artifacts(seed) for seed in seeds},
    }
    if (
        config.baseline == "discrete_bo"
        and config.discrete_bo.candidate_source == "llm"
    ):
        data["pool"] = root / "pool.jsonl"
    return data


@dataclass(frozen=True)
class SearchBackend:
    """Model and task services shared by every baseline in one run."""

    task: str
    model_name: str
    checkpoint: object
    generate: Generate
    score: Score
    decode_point: Callable[[np.ndarray], str]
    train_mu: np.ndarray
    bounds: np.ndarray
    maximum: float | None
    embed: Embed | None = None


def _normalize_generated_text(text: str, task: str) -> str:
    """Keep newline-separated candidates intact while folding Semantle case."""
    if task == "molopt":
        smiles, _repaired = maybe_repair_invalid_smiles(text)
        return smiles
    if task != "semantle":
        return text
    return "\n".join(
        normalize_search_text(line, task=task)
        for line in str(text).splitlines()
        if str(line).strip()
    )


def _base_generate(checkpoint, task: str, model_name: str) -> Generate:
    """Sample the pretrained model without interventions, one call per sample."""

    def generate(prompt: str, count: int, options: GenerationOptions) -> list[str]:
        prompt = maybe_append_decode_smiles_open_tag(checkpoint, prompt)
        rendered, from_chat, _ = build_checkpoint_prompt(
            checkpoint,
            user_text=prompt,
            history=[],
            accumulate_history=False,
            use_checkpoint_prompt=False,
            generation_mode="base",
            model_name=model_name,
            system_prompt=system_prompt_from_cfg(checkpoint.saved_cfg),
        )
        texts = []
        for _ in range(count):
            text = generate_base(
                checkpoint.reft_model.model,
                checkpoint.tokenizer,
                rendered,
                max_new_tokens=options.max_new_tokens,
                do_sample=options.temperature > 0,
                temperature=max(options.temperature, 1e-8),
                top_p=options.top_p,
                from_chat_template=from_chat,
                assistant_suffix=checkpoint.assistant_suffix,
            )
            texts.append(_normalize_generated_text(text, task))
        return texts

    return generate


def _backend_embed(task: str, model_name: str | None) -> Embed:
    """Embed solutions in the task's text space, optionally with another encoder."""

    def embed(texts: Sequence[str]) -> list[list[float]]:
        payload = []
        for text in texts:
            if task == "molopt":
                text = unwrap_smiles_tags(text)
            payload.append(format_text_for_embedding(text, task=task))
        rows = encode_texts_as_is(payload, task=task, model_name=model_name)
        return np.asarray(rows, dtype=float).tolist()

    return embed


def _verifier_score(verifier: Verifier) -> Score:
    def score(solution: str) -> ScoreResult:
        result = verifier(solution)
        return ScoreResult(score=result.score, components=dict(result.components))

    return score


def load_backend(config: BaselineSearchConfig) -> SearchBackend:
    """Load the ReFT checkpoint and task verifier used by ``boreft.search``."""
    saved = load_merged_run_config(config.reft_output_dir)
    model_name = config.model_name or saved.get("model_name")
    if not model_name:
        raise ValueError("model name is absent from both CLI and checkpoint config")
    layer = config.layer if config.layer is not None else int(saved.get("layer", 15))
    rank = (
        config.low_rank_dim
        if config.low_rank_dim is not None
        else int(saved.get("low_rank_dim", 8))
    )
    checkpoint = load_eval_checkpoint(
        config.reft_output_dir,
        model_name,
        layer,
        rank,
        config.cache_dir or saved.get("cache_dir"),
        torch_dtype=config.torch_dtype,
        load_latest=bool(config.load_latest),
    )
    task = saved.get("task", "semantle")
    if task != config.task:
        raise ValueError(
            f"checkpoint task {task!r} differs from requested task {config.task!r}"
        )
    if task == "molopt":
        verifier = molopt_search_verifier(
            target=config.target,
            objective=config.objective,
            rdkit_map_path=rdkit_map_path_for_cfg(saved),
            oracle=config.oracle,
        )
    else:
        verifier = TextSimilarityVerifier(config.target, task=task)
    train_mu = stack_bias_vectors(
        checkpoint.reft_model, list(range(len(checkpoint.words)))
    ).astype(np.float32)
    _apply_lora_sft_adapter(config, checkpoint)
    return SearchBackend(
        task=task,
        model_name=model_name,
        checkpoint=checkpoint,
        generate=_base_generate(checkpoint, task, model_name),
        score=_verifier_score(verifier),
        decode_point=checkpoint_decode(
            checkpoint,
            config.max_new_tokens,
            sampling_temperature=config.sampling_temperature,
            task=task,
        ),
        train_mu=train_mu,
        bounds=latent_bounds(train_mu, padding=config.bounds_padding).astype(
            np.float32
        ),
        maximum=verifier.maximum,
        embed=_backend_embed(
            task, getattr(method_config(config), "embedding_model", None)
        ),
    )


def _prepare_run_directory(config: BaselineSearchConfig, root: Path) -> None:
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(f"search_dir is not a directory: {root}")
    if (
        root.exists()
        and any(root.iterdir())
        and not (config.resume or config.overwrite)
    ):
        raise FileExistsError(f"{root} is not empty; pass --resume or --overwrite")
    if config.overwrite and not config.resume:
        import shutil

        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    if config.resume:
        _validate_resume_config(root / "config.json", config)
    _write_json(root / "config.json", resolved_config(config))


def _warmstart_seeds(
    config: BaselineSearchConfig,
    backend: SearchBackend,
    seed: int,
    decode=None,
) -> list[WarmstartSeed]:
    """Decode the main-search warmstart points into scored baseline solutions.

    Checkpoint warm starts already carry the labeled train word in ``decoded``,
    so those seeds score that word instead of decoding the bias vector.
    Reconstruction of train μ uses the ReFT decoder, not a search-time LoRA.
    """
    reconstruct = decode
    if (
        reconstruct is None
        and config.warmstart_source == "checkpoint"
        and getattr(backend.checkpoint, "reft_model", None) is not None
    ):
        reconstruct = checkpoint_reconstruct_decode(
            backend.checkpoint,
            config.max_new_tokens,
            task=backend.task,
        )
    points, records = prepare_warmstarts(
        config,
        checkpoint=backend.checkpoint,
        train_mu=backend.train_mu,
        bounds=backend.bounds,
        seed=seed,
        decode=reconstruct,
    )
    rows = list(records or [{} for _ in points])
    if len(rows) != len(points):
        raise ValueError("warmstart records must be parallel to warmstart points")

    def bound_decoder(point: np.ndarray):
        array = np.asarray(point, dtype=np.float32)
        return lambda: backend.decode_point(array)

    return [
        WarmstartSeed(
            decode=bound_decoder(point),
            solution=_labeled_decoded(record),
            components=dict(record.get("components", {})),
            metadata={
                "warmstart_source": config.warmstart_source,
                "warmstart_point": np.asarray(point, dtype=float).tolist(),
                **(
                    {"warmstart_word": record["components"]["warmstart_word"]}
                    if isinstance(record.get("components"), dict)
                    and record["components"].get("warmstart_word") is not None
                    else {}
                ),
            },
            score=None if record.get("score") is None else float(record["score"]),
        )
        for point, record in zip(points, rows)
    ]


def _previous_elapsed(
    layout: ArtifactLayout,
    state: BaselineRunState,
    resume: bool,
) -> float:
    """Keep resume wall-clock cumulative, matching ``boreft.search``."""
    if not resume:
        return 0.0
    if layout.summary.exists():
        previous = json.loads(layout.summary.read_text(encoding="utf-8"))
        return float(previous.get("elapsed_seconds", 0.0))
    return float(state.summary()["elapsed_seconds"])


def _print_run_banner(
    config: BaselineSearchConfig,
    backend: SearchBackend,
    root: Path,
    seeds: tuple[int, ...],
) -> None:
    print("=" * 64)
    print(f"Baseline search: {config.baseline}")
    print(f"  target      {config.target}")
    if config.oracle:
        print(f"  oracle      {config.oracle}")
    print(f"  task        {backend.task}")
    print(f"  checkpoint  {config.reft_output_dir}")
    if config.lora_sft_adapter:
        print(f"  lora-sft    {config.lora_sft_adapter}")
    print(f"  output      {root}")
    print(f"  model       {backend.model_name}")
    samples = config.observation_samples
    warm_each = (
        1 if config.warmstart_source in ("checkpoint", "words") else samples
    )
    warm_calls = config.warmstart_count * warm_each
    search_calls = config.budget - warm_calls
    print(
        f"  budget      {config.budget} verifications  "
        f"(warmstart={config.warmstart_count}×{warm_each}, "
        f"search={search_calls // samples}×{samples})"
    )
    print(
        f"  decode      samples={config.observation_samples}  "
        f"T={config.sampling_temperature}  ·  batch={config.batch_size}"
    )
    print(f"  seeds       {', '.join(str(seed) for seed in seeds)}")
    print("=" * 64, flush=True)


def _matches_gold_target(solution: str, target: str, task: str) -> bool:
    """True when a scored solution is the known gold target for this task."""
    if task == "molopt":
        return canonical_target_key(solution) == canonical_target_key(target)
    return normalize_search_text(solution, task=task) == normalize_search_text(
        target, task=task
    )


def _observation_found_target(
    observation: BaselineObservation, target: str, task: str
) -> bool:
    match = observation.components.get("exact_match")
    if match is True or match == 1 or match == 1.0:
        return True
    if isinstance(match, (int, float)) and match > 0:
        return True
    return _matches_gold_target(observation.solution, target, task)


def found_target(
    observations: Sequence[BaselineObservation], target: str, task: str
) -> bool:
    """Whether any observation is the known gold target."""
    return found_target_stats(observations, target, task)[0]


def found_target_stats(
    observations: Sequence[BaselineObservation],
    target: str,
    task: str,
    *,
    property_search: bool = False,
) -> tuple[bool, int | None, int | None]:
    """Return ``(found, index, verifications)`` for the first gold-target hit."""
    if property_search:
        return False, None, None
    used = 0
    for item in observations:
        used += item.sample_count
        if _observation_found_target(item, target, task):
            return True, item.index, used
    return False, None, None


def _best_run(summaries: dict[str, dict]) -> dict:
    return max(
        summaries.values(),
        key=lambda item: (
            item.get("best_score") is not None,
            item.get("best_score") or 0.0,
        ),
    )


def _fastest_found(summaries: dict[str, dict], seeds: tuple[int, ...]) -> tuple[int, int] | None:
    hits = [
        (
            int(summaries[str(seed)]["found_at_verifications"]),
            int(summaries[str(seed)].get("seed", seed)),
        )
        for seed in seeds
        if summaries[str(seed)].get("found_target")
        and summaries[str(seed)].get("found_at_verifications") is not None
    ]
    if not hits:
        return None
    return min(hits)


def _overall_found_label(
    summaries: dict[str, dict], seeds: tuple[int, ...], found_count: int
) -> str:
    label = f"target found in {found_count}/{len(seeds)} seed(s)"
    fastest = _fastest_found(summaries, seeds)
    if fastest is None:
        return label
    verifications, seed = fastest
    return f"{label}  ·  fastest at {verifications} verifications (seed {seed})"


def _mean_best_score(summaries: dict[str, dict], seeds: tuple[int, ...]) -> float | None:
    scores = [
        float(summaries[str(seed)]["best_score"])
        for seed in seeds
        if summaries[str(seed)].get("best_score") is not None
    ]
    if not scores:
        return None
    return sum(scores) / len(scores)


def _print_finish_summary(summaries: dict[str, dict], seeds: tuple[int, ...]) -> None:
    print()
    print("=" * 64)
    print(f"Finished {len(seeds)} seed(s).")
    found_count = 0
    for seed in seeds:
        summary = summaries[str(seed)]
        found = bool(summary.get("found_target"))
        found_count += int(found)
        best_score = summary.get("best_score")
        if best_score is None:
            print(f"  seed {seed}  no observations")
            continue
        found_label = "target not found"
        if found:
            at = summary.get("found_at_verifications")
            found_label = (
                f"target found at {at} verifications"
                if at is not None
                else "target found"
            )
        print(
            f"  seed {seed}  best={best_score:.6g} "
            f"({summary.get('best_solution')!r})  {found_label}"
        )
    best_run = _best_run(summaries)
    if best_run.get("best_score") is None:
        print("  overall  no observations")
    else:
        print(
            f"  overall  best={best_run['best_score']:.6g} from seed {best_run.get('seed')}  "
            f"·  mean best={_mean_best_score(summaries, seeds):.6g}  "
            f"·  {_overall_found_label(summaries, seeds, found_count)}"
        )
    print("=" * 64, flush=True)


def _search_context(
    config: BaselineSearchConfig,
    backend: SearchBackend,
    seed: int,
) -> SearchContext:
    return SearchContext(
        task_description=config.task_description,
        target=config.target,
        objective=config.objective,
        seed=seed,
        generate=backend.generate,
        score=backend.score,
        embed=backend.embed,
        generation_options=generation_options(config),
        observation_samples=config.observation_samples,
        checkpoint_dir=config.reft_output_dir,
        task=config.task,
        checkpoint=backend.checkpoint,
        model_name=backend.model_name,
    )


def _restored_discrete_bo_pool(
    root: Path, seeds: tuple[int, ...]
) -> tuple[list[str], dict[str, list[float]]] | None:
    for seed in seeds:
        state = load_method_state(ArtifactLayout(root, seed).method_state)
        raw = state.get("pool") if state else None
        if not raw:
            continue
        pool = coerce_pool(raw)
        keys = {solution_key(item) for item in pool}
        embeddings = {
            key: vector
            for key, vector in coerce_embeddings(state.get("embeddings", {})).items()
            if key in keys
        }
        return pool, embeddings
    return None


def _embed_discrete_bo_pool(
    pool: list[str],
    embed: Embed | None,
    embeddings: dict[str, list[float]],
) -> dict[str, list[float]]:
    if embed is None:
        raise ValueError("discrete_bo requires context.embed")
    fill_embeddings(pool, embed, embeddings)
    return embeddings


def _discrete_bo_embedding_dim(embeddings: dict[str, list[float]]) -> int | None:
    if not embeddings:
        return None
    return len(next(iter(embeddings.values())))


def _shared_discrete_bo_pool(
    config: BaselineSearchConfig,
    backend: SearchBackend,
    seeds: tuple[int, ...],
    root: Path,
) -> tuple[list[str], dict[str, list[float]]]:
    """Build or restore one candidate pool; embed it unless acquisition is random."""
    method = config.discrete_bo
    pool_seed = seeds[0]
    restored = _restored_discrete_bo_pool(root, seeds) if config.resume else None
    if restored is not None:
        pool, embeddings = restored
        present = solution_key(config.target) in {
            solution_key(item) for item in pool
        }
        report = DiscreteBOPoolReport(
            size=len(pool),
            source=method.candidate_source,
            pool_seed=pool_seed,
            target=config.target.strip(),
            target_present=present,
            target_added=False,
        )
    else:
        seed_everything(pool_seed)
        pool, report = select_discrete_bo_pool(
            method, _search_context(config, backend, pool_seed), log=False
        )
        embeddings = {}
    if method.acquisition != "random":
        _embed_discrete_bo_pool(pool, backend.embed, embeddings)
    saved_path = None
    if restored is None and method.candidate_source == "llm":
        saved_path = write_discrete_bo_pool(root / "pool.jsonl", pool)
    log_discrete_bo_pool(
        report,
        task=config.task,
        embedding_dim=_discrete_bo_embedding_dim(embeddings),
        saved_path=saved_path,
    )
    return pool, embeddings


def _make_baseline(
    config: BaselineSearchConfig,
    shared_pool: list[str] | None,
    shared_embeddings: dict[str, list[float]] | None,
):
    method = method_config(config)
    if config.baseline == "discrete_bo":
        return DiscreteBOBaseline(
            method,
            pool=list(shared_pool),
            embeddings=dict(shared_embeddings or {}),
        )
    return create_baseline(config.baseline, method)


def run_search(config: BaselineSearchConfig) -> dict:
    """Run the selected baseline across every requested seed."""
    config.validate()
    seeds = _resolved_seeds(config)
    root = _safe_search_root(config.search_dir, config.reft_output_dir)
    backend = load_backend(config)
    loop_config = BaselineLoopConfig(
        budget=config.budget,
        batch_size=config.batch_size,
        observation_samples=config.observation_samples,
        maximum=backend.maximum,
    )
    loop_config.validate()
    _prepare_run_directory(config, root)
    _print_run_banner(config, backend, root, seeds)
    reconstruct_decode = None
    if config.warmstart_source == "checkpoint" and getattr(
        backend.checkpoint, "reft_model", None
    ) is not None:
        reconstruct_decode = memoize_point_decode(
            checkpoint_reconstruct_decode(
                backend.checkpoint,
                config.max_new_tokens,
                task=backend.task,
            )
        )
    shared_pool: list[str] | None = None
    shared_embeddings: dict[str, list[float]] | None = None
    if config.baseline == "discrete_bo":
        shared_pool, shared_embeddings = _shared_discrete_bo_pool(
            config, backend, seeds, root
        )

    states: list[BaselineRunState] = []
    summaries: dict[str, dict] = {}
    for seed in seeds:
        layout = ArtifactLayout(root, seed)
        started = time.monotonic()
        state = BaselineRunState.load(layout.observations)
        previous_elapsed = _previous_elapsed(layout, state, config.resume)
        seed_everything(seed)
        clear_checkpoint_lora(backend.checkpoint)
        warmstarts = _warmstart_seeds(
            config, backend, seed, decode=reconstruct_decode
        )
        _apply_lora_sft_adapter(config, backend.checkpoint)
        state = run_baseline(
            seed=seed,
            baseline=_make_baseline(config, shared_pool, shared_embeddings),
            context=_search_context(config, backend, seed),
            state=state,
            config=loop_config,
            warmstarts=warmstarts,
            method_state_dir=layout.method_state,
        )
        hit, found_index, found_at = found_target_stats(
            state.observations,
            config.target,
            config.task,
            property_search=bool(config.oracle),
        )
        summary = {
            **state.summary(
                elapsed_seconds=previous_elapsed + (time.monotonic() - started)
            ),
            "seed": seed,
            "found_target": hit,
            "found_at_index": found_index,
            "found_at_verifications": found_at,
            **warmstart_result_fields(
                [
                    {
                        "decoded": seed_row.solution,
                        "components": seed_row.components,
                    }
                    for seed_row in warmstarts
                ]
            ),
        }
        _write_json(layout.summary, summary)
        plot_best_so_far(
            [state.observations],
            layout.plot,
            title=f"{config.baseline} seed {seed}",
        )
        maybe_log_finished_seed(
            layout.seed_dir,
            project=config.wandb_project,
            entity=config.wandb_entity,
            group=config.wandb_group,
            name=(
                None
                if not config.wandb_run_name
                else f"{config.wandb_run_name}_seed{seed}"
            ),
            wandb_dir=config.wandb_dir,
            extra_config=resolved_config(config),
            no_wandb=config.no_wandb,
        )
        states.append(state)
        summaries[str(seed)] = summary

    aggregate = aggregate_summaries(summaries, seeds)
    _write_json(root / "summary.json", aggregate)
    plot_best_so_far(
        [state.observations for state in states],
        root / "best_so_far.png",
        title=f"{config.baseline} across seeds",
    )
    _print_finish_summary(summaries, seeds)
    return aggregate


def resolved_config(config: BaselineSearchConfig) -> dict:
    """JSON-compatible shared config snapshot for future run artifacts."""
    data = asdict(config)
    selected = data[config.baseline]
    for name in available_baselines():
        data.pop(name)
    data["method_config"] = selected
    data["candidate_top_p"] = generation_options(config).top_p
    return data


def aggregate_summaries(summaries: dict[str, dict], seeds: tuple[int, ...]) -> dict:
    """Build the shared overall timing and repetition summary."""
    found_seeds = [
        int(summaries[str(seed)].get("seed", seed))
        for seed in seeds
        if summaries[str(seed)].get("found_target")
    ]
    best_run = _best_run(summaries)
    fastest = _fastest_found(summaries, seeds)
    return {
        "runs": summaries,
        "seeds": list(seeds),
        "elapsed_seconds": sum(
            summary["elapsed_seconds"] for summary in summaries.values()
        ),
        "bbox_eval_seconds": sum(
            summary["bbox_eval_seconds"] for summary in summaries.values()
        ),
        "n_repeat_samples": sum(
            summary["n_repeat_samples"] for summary in summaries.values()
        ),
        "n_repeat_proposals": sum(
            summary["n_repeat_proposals"] for summary in summaries.values()
        ),
        "found_target_seeds": found_seeds,
        "n_found_target": len(found_seeds),
        "best_score": best_run.get("best_score"),
        "best_solution": best_run.get("best_solution"),
        "best_seed": best_run.get("seed"),
        "mean_best_score": _mean_best_score(summaries, seeds),
        "fastest_found_at_verifications": (
            None if fastest is None else fastest[0]
        ),
        "fastest_found_seed": None if fastest is None else fastest[1],
        "n_warmstart_reconstruct_misses": sum(
            int(summaries[str(seed)].get("n_warmstart_reconstruct_misses") or 0)
            for seed in seeds
        ),
        "n_warmstart_noisy": sum(
            int(summaries[str(seed)].get("n_warmstart_noisy") or 0)
            for seed in seeds
        ),
        "n_warmstart_predicted": sum(
            int(summaries[str(seed)].get("n_warmstart_predicted") or 0)
            for seed in seeds
        ),
    }


def main() -> None:
    run_search(tyro.cli(BaselineSearchConfig))


if __name__ == "__main__":
    main()
