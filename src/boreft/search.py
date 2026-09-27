"""GOLLuM-style Bayesian optimization over BOReFT bias vectors.

The BO core is model-agnostic. This module connects it to a BOReFT checkpoint:
raw latent point -> model decode -> task verifier -> scalar observation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import time
from typing import Callable, Literal, Mapping, Sequence
import warnings

import numpy as np
import torch
import tyro

from boreft.bias_tables import (
    bias_predict_kwargs,
    predict_bias_vectors_for_words,
    predict_bias_vectors_from_raw_texts,
    stack_bias_mu_std,
    training_search_region,
)
from boreft.bo import Observation, RunState
from boreft.bo.runner import (
    BOConfig,
    Verification,
    Verifier,
    recorded_repeat_counts,
    run_bo,
    seed_everything as _seed_everything,
)
from boreft.bo.surrogate import KernelKind
from boreft.bo.plotting import plot_best_so_far
from boreft.chem import (
    canonical_target_key,
    is_valid_smiles,
    maybe_repair_invalid_smiles,
    rdkit_similarity,
    tanimoto_similarity,
    unwrap_smiles_tags,
)
from boreft.data_utils import load_merged_run_config, system_prompt_from_cfg
from boreft.eval.eval_suite import normalize_text
from boreft.eval.semantle import generate_text, load_eval_checkpoint
from boreft.interactive import build_checkpoint_prompt, generate_base
from boreft.oracles import (
    SupportsOracle,
    load_property_oracle,
    normalize_property_oracle_name,
    property_oracle_factors,
    property_search_task_description,
    score_molecules,
)
from boreft.search_expand import (
    SearchExpandConfig,
    load_expansion_state,
    load_expansion_weights,
    load_run_raw_definitions,
    make_expansion_hook,
    restore_search_intervention,
    snapshot_search_intervention,
)
from boreft.search_wandb import (
    WANDB_CONFIG_KEYS,
    maybe_log_finished_seed,
    start_search_wandb_session,
)
from boreft.task_config import task_target_kind
from boreft.text_similarity import (
    bias_network_embed_model_from_cfg,
    embedding_sim_per_text,
    rdkit_map_path_for_cfg,
)

WarmstartSource = Literal["checkpoint", "sobol", "file", "llm", "words"]
SearchPrompt = Literal["checkpoint", "task"]
ObjectiveName = Literal["embed_sim", "tfs", "rdkit_sim"]


def warmstart_verification_cost(
    source: WarmstartSource, count: int, observation_samples: int
) -> int:
    """Verifications spent on warm starts before acquisition begins.

    Checkpoint and ``words`` starts are labeled strings and cost one
    verification each. Sobol, LLM, and file starts decode (or may decode) and
    use ``observation_samples`` as the per-point cost.
    """
    if count < 1:
        raise ValueError("warmstart_count must be positive")
    if observation_samples < 1:
        raise ValueError("observation_samples must be positive")
    if source in ("checkpoint", "words"):
        return count
    return count * observation_samples


def normalize_search_text(text: str, *, task: str) -> str:
    """Lower-case Semantle words; unwrap molopt SMILES tags; keep Hypogen case."""
    if task == "semantle":
        return normalize_text(text, task=task)
    if task == "molopt":
        return unwrap_smiles_tags(str(text).strip())
    return str(text).strip()


def _matches_search_target(text: str, target: str, task: str) -> bool:
    if task == "molopt":
        return canonical_target_key(text) == canonical_target_key(target)
    return normalize_search_text(text, task=task) == normalize_search_text(
        target, task=task
    )


def found_target_stats(
    observations: Sequence[Observation],
    target: str,
    task: str,
    *,
    tolerance: float = 1e-6,
    property_search: bool = False,
) -> tuple[bool, int | None, int | None]:
    """Return ``(found, observation_index, verifications)`` for the first gold hit.

    Each decoded sample is one attempt. A hit is an exact target match or a
    sample score of 1 (the known Semantle/molopt maximum). ``tolerance`` matches
    ``BOConfig.maximum_tolerance`` so summaries agree with early stopping.
    Property-oracle search has no hidden molecule, so it never reports a found
    target even if a score reaches 1.
    """
    if property_search:
        return False, None, None
    used = 0
    for item in observations:
        texts = list(item.decoded_samples) if item.decoded_samples else [item.decoded]
        scores = list(item.sample_scores) if item.sample_scores else [item.score]
        n = max(item.sample_count, len(texts), 1)
        for offset in range(n):
            used += 1
            text = texts[offset] if offset < len(texts) else item.decoded
            score = scores[offset] if offset < len(scores) else item.score
            if _matches_search_target(text, target, task):
                return True, item.index, used
            if isinstance(score, (int, float)) and score >= 1.0 - tolerance:
                return True, item.index, used
        exact = item.components.get("exact_match")
        if exact is True or exact == 1 or exact == 1.0:
            return True, item.index, used
        if isinstance(exact, (int, float)) and exact > 0:
            return True, item.index, used
    return False, None, None


def _prepared_molopt_decode(decoded: str) -> tuple[str, dict[str, float | int | bool | str]]:
    """Unwrap, optionally SmiSelf-repair, and return scoring text plus log fields."""
    smiles, repaired = maybe_repair_invalid_smiles(decoded)
    raw = unwrap_smiles_tags(decoded)
    return smiles, {
        "decoded": smiles,
        "raw_decoded": raw,
        "repaired": repaired,
    }


class TextSimilarityVerifier:
    maximum = 1.0

    def __init__(self, target: str, *, task: str = "semantle") -> None:
        if task_target_kind(task) != "text":
            raise ValueError(
                f"TextSimilarityVerifier requires a text task, got {task!r}"
            )
        self.task = task
        self.target = normalize_search_text(target, task=task)

    def __call__(self, decoded: str) -> Verification:
        decoded = normalize_search_text(decoded, task=self.task)
        similarity = float(
            embedding_sim_per_text([self.target], [decoded], task=self.task)[0]
        )
        exact = decoded == self.target
        return Verification(
            score=similarity,
            components={"embed_sim": similarity, "exact_match": exact},
        )


class MoleculeVerifier:
    maximum = 1.0

    def __init__(
        self,
        target: str,
        *,
        objective: ObjectiveName = "tfs",
        rdkit_map_path: str | None = None,
    ) -> None:
        if objective not in ("embed_sim", "tfs", "rdkit_sim"):
            raise ValueError(f"unknown molecular objective: {objective!r}")
        if not is_valid_smiles(target):
            raise ValueError(f"molecular target is not valid SMILES: {target!r}")
        self.target = target
        self.objective = objective
        self.rdkit_map_path = rdkit_map_path

    def __call__(self, decoded: str) -> Verification:
        decoded, extras = _prepared_molopt_decode(decoded)
        embed_sim = float(
            embedding_sim_per_text([self.target], [decoded], task="molopt")[0]
        )
        tfs = tanimoto_similarity(self.target, decoded)
        rdkit_sim = rdkit_similarity(
            self.target, decoded, map_path=self.rdkit_map_path
        )
        components: dict[str, float | bool | str] = {
            "embed_sim": embed_sim,
            "tfs": tfs,
            "rdkit_sim": rdkit_sim,
            "valid": is_valid_smiles(decoded),
            "exact_match": canonical_target_key(self.target)
            == canonical_target_key(decoded),
            **extras,
        }
        return Verification(score=float(components[self.objective]), components=components)


class PropertyOracleVerifier:
    """Score decoded SMILES with a molopt property oracle.

    TDC fingerprint tasks (DRD2, GSK3B, JNK3) are class-1 probabilities in
    ``[0, 1]``. ``GSK3B_JNK3`` is their product. Invalid molecules score 0.
    A perfect 1.0 is not a hidden-target hit, so ``maximum`` stays unset.
    """

    maximum = None

    def __init__(
        self,
        oracle: str,
        *,
        score_fn: SupportsOracle | None = None,
        factor_oracles: Mapping[str, SupportsOracle] | None = None,
    ) -> None:
        self.oracle_name = normalize_property_oracle_name(oracle)
        self._factors = property_oracle_factors(self.oracle_name)
        self._oracle: SupportsOracle | None
        self._factor_oracles: dict[str, SupportsOracle] | None
        if score_fn is not None:
            self._oracle = score_fn
            self._factor_oracles = None
        elif factor_oracles is not None:
            missing = [name for name in self._factors if name not in factor_oracles]
            if missing:
                raise ValueError(
                    f"{self.oracle_name} needs factor oracles {self._factors}; "
                    f"missing {tuple(missing)}"
                )
            if len(self._factors) == 1:
                self._oracle = factor_oracles[self._factors[0]]
                self._factor_oracles = None
            else:
                self._oracle = None
                self._factor_oracles = {
                    name: factor_oracles[name] for name in self._factors
                }
        elif len(self._factors) == 1:
            self._oracle = load_property_oracle(self.oracle_name)
            self._factor_oracles = None
        else:
            self._oracle = None
            self._factor_oracles = {
                name: load_property_oracle(name) for name in self._factors
            }

    def __call__(self, decoded: str) -> Verification:
        smiles, extras = _prepared_molopt_decode(decoded)
        factor_scores: dict[str, float] = {}
        if self._factor_oracles:
            row = None
            for name, fn in self._factor_oracles.items():
                scored = score_molecules(
                    [smiles], fn, include_qed=row is None
                )[0]
                factor_scores[name] = float(scored.score)
                if row is None:
                    row = scored
            assert row is not None
            score = float(math.prod(factor_scores.values()))
        else:
            assert self._oracle is not None
            row = score_molecules([smiles], self._oracle, include_qed=True)[0]
            score = float(row.score)
        components: dict[str, float | int | bool | str] = {
            "oracle": self.oracle_name,
            "oracle_score": score,
            "valid": bool(row.valid),
            "canonical": row.canonical or "",
            "exact_match": False,
            **{f"oracle_{name}": value for name, value in factor_scores.items()},
            **extras,
        }
        if row.qed is not None:
            components["qed"] = float(row.qed)
        return Verification(score=score, components=components)


def molopt_search_verifier(
    *,
    target: str,
    objective: ObjectiveName,
    rdkit_map_path: str | None,
    oracle: str | None,
) -> Verifier:
    """Hidden-target similarity, or a named molopt property oracle."""
    if oracle:
        return PropertyOracleVerifier(oracle)
    return MoleculeVerifier(
        target, objective=objective, rdkit_map_path=rdkit_map_path
    )


@dataclass
class SearchConfig:
    """CLI configuration for ``python -m boreft.search``.

    Subspace expansion (``--expand.every K``) continue-trains the loaded
    checkpoint on decoded observations every K acquisition batches, remaps
    latent points, and refits the GP. Recipe knobs inherit from the checkpoint
    unless overridden under ``--expand.*``. On ``--resume``, expansion timing,
    subset selection, init, and ``learn_*`` stay frozen; recipe, eval cadence,
    and stop bars may change and apply to later rounds.

    ``--aabb-std-k`` (default 0) expands the mean bounding box by ``k`` times
    the learned posterior standard deviation per training target.
    ``--search-domain ellipsoid`` replaces that box with the covering
    Mahalanobis ellipsoid of the same generators (sample covariance, not
    MVEE); acquisition is constrained to the ellipsoid while the GP still
    scales inputs using its bounding box.
    """

    output_dir: str
    target: str
    oracle: str | None = None
    search_dir: str = "search"
    objective: ObjectiveName = "embed_sim"
    budget: int = 60
    warmstart_count: int = 10
    warmstart_source: WarmstartSource = "checkpoint"
    warmstart_file: str | None = None
    seeds: tuple[int, ...] = (1,)
    repeats: int = 1
    surrogate: Literal["static", "projected"] = "static"
    acquisition: Literal["log_ei", "ucb", "thompson"] = "log_ei"
    batch_size: int = 1
    observation_samples: int = 1
    kernel: KernelKind = "matern-2.5"
    use_ard: bool = False
    projection_dim: int = 64
    projection_layers: int = 1
    projection_steps: int = 300
    gp_lr: float = 0.2
    projection_lr: float = 0.002
    bounds_padding: float = 0.0
    aabb_std_k: float = 0.0
    search_domain: Literal["aabb", "ellipsoid"] = "aabb"
    acquisition_restarts: int = 10
    acquisition_raw_samples: int = 512
    acquisition_mc_samples: int = 128
    ucb_beta: float = 0.2
    thompson_candidates: int = 4096
    duplicate_tolerance: float = 1e-6
    max_new_tokens: int = 128
    sampling_temperature: float = 1.0
    search_prompt: SearchPrompt = "checkpoint"
    task_description: str | None = None
    warmstart_temperature: float = 1.0
    warmstart_top_p: float = 0.9
    log_gp_surrogate: bool = False
    resume: bool = False
    overwrite: bool = False
    load_latest: bool = False
    model_name: str | None = None
    layer: int | None = None
    low_rank_dim: int | None = None
    cache_dir: str | None = None
    torch_dtype: Literal["bfloat16", "float16", "float32"] | None = None
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_group: str | None = None
    wandb_dir: str | None = None
    no_wandb: bool = False
    expand: SearchExpandConfig = field(default_factory=SearchExpandConfig)

    def validate(self) -> None:
        if self.oracle:
            self.oracle = normalize_property_oracle_name(self.oracle)
            if not self.target.strip():
                self.target = self.oracle
        if not self.target.strip():
            raise ValueError("target must not be blank")
        if self.warmstart_count < 2:
            raise ValueError("warmstart_count must be at least 2")
        if self.observation_samples < 1:
            raise ValueError("observation_samples must be positive")
        if self.projection_layers < 1:
            raise ValueError("projection_layers must be positive")
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
                f"warmstart_source={self.warmstart_source} requires --warmstart-file"
            )
        if self.bounds_padding < 0:
            raise ValueError("bounds_padding must be nonnegative")
        if not np.isfinite(self.aabb_std_k) or self.aabb_std_k < 0:
            raise ValueError("aabb_std_k must be finite and nonnegative")
        if self.search_domain not in ("aabb", "ellipsoid"):
            raise ValueError("search_domain must be 'aabb' or 'ellipsoid'")
        if self.max_new_tokens < 1:
            raise ValueError("generation length must be positive")
        if self.sampling_temperature < 0:
            raise ValueError("sampling_temperature must be nonnegative")
        if self.search_prompt not in ("checkpoint", "task"):
            raise ValueError("search_prompt must be 'checkpoint' or 'task'")
        if self.search_prompt == "task" and not (
            (self.task_description or "").strip() or self.oracle
        ):
            raise ValueError(
                "search_prompt=task requires --oracle or --task-description"
            )
        if self.warmstart_temperature <= 0:
            raise ValueError("warmstart_temperature must be positive")
        if not 0 < self.warmstart_top_p <= 1:
            raise ValueError("warmstart_top_p must be in (0, 1]")
        if self.overwrite:
            # Launcher defaults to --resume; `--overwrite` after `--` must win.
            self.resume = False
        self.expand.validate()
        self.warmstart_file = resolve_checkpoint_pin_file(
            self.output_dir,
            warmstart_source=self.warmstart_source,
            warmstart_file=self.warmstart_file,
        )


def resolve_search_instruction(config: SearchConfig, *, task: str) -> str | None:
    """User instruction for decode, or ``None`` to keep ``ckpt.prompt``.

    ``search_prompt=task`` is the OPRO/BOPRO property-search header plus the
    MiST completion prefix, with no scored history. ``--task-description``
    overrides that text when set.
    """
    custom = (config.task_description or "").strip()
    if custom:
        return custom
    if config.search_prompt == "checkpoint":
        return None
    if task != "molopt":
        raise ValueError("search_prompt=task is only valid for molopt checkpoints")
    if not config.oracle:
        raise ValueError("search_prompt=task requires --oracle or --task-description")
    return property_search_task_description(config.oracle)


def resolve_search_decode_prompt(
    ckpt,
    config: SearchConfig,
    *,
    task: str,
    model_name: str,
) -> tuple[str, bool, tuple[int, int] | None]:
    """Intervention prompt used by :func:`checkpoint_decode`."""
    instruction = resolve_search_instruction(config, task=task)
    if instruction is None:
        return ckpt.prompt, ckpt.from_chat_template, ckpt.content_span
    return build_checkpoint_prompt(
        ckpt,
        user_text=instruction,
        history=[],
        accumulate_history=False,
        use_checkpoint_prompt=False,
        generation_mode="intervention",
        model_name=model_name,
        system_prompt=system_prompt_from_cfg(getattr(ckpt, "saved_cfg", None)),
    )


def checkpoint_decode(
    ckpt,
    max_new_tokens: int,
    *,
    sampling_temperature: float = 1.0,
    task: str = "semantle",
    prompt: str | None = None,
    from_chat_template: bool | None = None,
    content_span: tuple[int, int] | None = None,
    repair_invalid_smiles: bool = True,
) -> Callable[[np.ndarray], str]:
    saved = ckpt.saved_cfg or {}
    use_sample = sampling_temperature > 0
    normalize = task == "semantle"
    unwrap_molopt = task == "molopt"
    decode_prompt = ckpt.prompt if prompt is None else prompt
    decode_from_chat = (
        ckpt.from_chat_template if from_chat_template is None else from_chat_template
    )
    decode_span = ckpt.content_span if content_span is None else content_span

    def decode(point: np.ndarray) -> str:
        text = generate_text(
            ckpt.reft_model,
            ckpt.tokenizer,
            decode_prompt,
            point,
            max_new_tokens=max_new_tokens,
            use_sample=use_sample,
            temperature=max(sampling_temperature, 1e-8),
            top_p=1.0,
            position=saved.get("position", "l1"),
            assistant_suffix=ckpt.assistant_suffix,
            from_chat_template=decode_from_chat,
            intervention_token_id=ckpt.intervention_token_id,
            content_span=decode_span,
        )
        if unwrap_molopt:
            if repair_invalid_smiles:
                smiles, _repaired = maybe_repair_invalid_smiles(text)
                return smiles
            return unwrap_smiles_tags(text)
        return normalize_search_text(text, task=task) if normalize else text

    return decode


def checkpoint_reconstruct_decode(
    ckpt,
    max_new_tokens: int,
    *,
    task: str,
) -> Callable[[np.ndarray], str]:
    """Greedy decode of a train μ under the checkpoint prompt.

    SmiSelf is off: a repaired different molecule is not a reconstruction hit.
    """
    return checkpoint_decode(
        ckpt,
        max_new_tokens,
        sampling_temperature=0.0,
        task=task,
        repair_invalid_smiles=False,
    )


def memoize_point_decode(
    decode: Callable[[np.ndarray], str],
) -> Callable[[np.ndarray], str]:
    """Reuse greedy reconstructions of the same μ across warmstart draws."""
    cache: dict[bytes, str] = {}

    def wrapped(point: np.ndarray) -> str:
        key = np.asarray(point, dtype=np.float32).tobytes()
        cached = cache.get(key)
        if cached is None:
            cached = decode(point)
            cache[key] = cached
        return cached

    return wrapped


def _warmstart_identity(text: str, *, task: str) -> str:
    """Identity used to de-duplicate train words and exclude the search target."""
    if task == "molopt":
        return canonical_target_key(text)
    return normalize_search_text(text, task=task)


def _checkpoint_eligible_indices(
    words: Sequence[str],
    *,
    target: str,
    task: str,
) -> list[int]:
    target_key = _warmstart_identity(target, task=task)
    seen: set[str] = set()
    eligible: list[int] = []
    for index, raw in enumerate(words):
        text = str(raw).strip()
        if not text:
            continue
        key = _warmstart_identity(text, task=task)
        if not key or key in seen or key == target_key:
            continue
        seen.add(key)
        eligible.append(index)
    return eligible


def checkpoint_warmstart_indices(
    words: Sequence[str],
    *,
    target: str,
    count: int,
    seed: int,
    task: str = "semantle",
    accept: Callable[[int], bool] | None = None,
) -> list[int]:
    """Pick labeled train-word rows, excluding the target, with a shared RNG.

    First occurrence of each identity in ``words`` is kept (aligned with
    ``train_mu`` rows). Without ``accept``, the same
    ``(words, target, count, seed, task)`` always yields the same indices.
    With ``accept``, the draw is still seed-deterministic but also depends on
    which rows pass; every search method that shares that predicate still
    warms up on the same labels.

    When ``accept`` is set, candidates that return false are discarded and the
    next unused eligible train row is drawn until ``count`` rows pass or the
    pool is exhausted. A short list means the caller should fill the remainder
    with noisy posterior samples.
    """
    if count < 1:
        raise ValueError("warmstart_count must be positive")
    eligible = _checkpoint_eligible_indices(words, target=target, task=task)
    if count > len(eligible):
        raise ValueError(
            f"checkpoint has only {len(eligible)} labeled train words after "
            f"excluding the target for {count} warm starts"
        )
    rng = np.random.default_rng(seed)
    if accept is None:
        chosen = rng.choice(len(eligible), size=count, replace=False)
        return [eligible[int(position)] for position in chosen]
    accepted: list[int] = []
    for position in rng.permutation(len(eligible)):
        index = eligible[int(position)]
        if accept(index):
            accepted.append(index)
            if len(accepted) == count:
                return accepted
    return accepted


def _train_posterior_std(ckpt, train_mu: np.ndarray) -> np.ndarray:
    """Per-row posterior std, or a column-wise train-μ spread fallback."""
    array = np.asarray(train_mu, dtype=np.float32)
    reft = getattr(ckpt, "reft_model", None)
    if reft is not None:
        try:
            _mu, std = stack_bias_mu_std(reft, list(range(len(array))))
            std = np.asarray(std, dtype=np.float32)
            if std.shape == array.shape and np.isfinite(std).all():
                return np.maximum(std, 0.0)
        except (TypeError, ValueError, AttributeError):
            pass
    spread = np.std(array, axis=0, keepdims=True).astype(np.float32)
    spread = np.maximum(spread, 1e-3)
    return np.broadcast_to(spread, array.shape).copy()


def _noisy_checkpoint_points(
    train_mu: np.ndarray,
    source_indices: Sequence[int],
    *,
    count: int,
    seed: int,
    ckpt,
    bounds: np.ndarray,
    stream: int = 0,
) -> tuple[np.ndarray, list[int]]:
    """Sample ``μ + ε ⊙ σ`` from unused (or accepted) train rows."""
    if count < 1:
        return np.zeros((0, train_mu.shape[1]), dtype=np.float32), []
    sources = [int(index) for index in source_indices]
    if not sources:
        sources = list(range(len(train_mu)))
    rng = np.random.default_rng([int(seed), 0x4E01, int(stream)])
    picked = [
        int(sources[int(position)])
        for position in rng.choice(
            len(sources), size=count, replace=len(sources) < count
        )
    ]
    std = _train_posterior_std(ckpt, train_mu)
    noise = rng.standard_normal((count, train_mu.shape[1])).astype(np.float32)
    points = np.asarray(train_mu[np.asarray(picked, dtype=np.intp)], dtype=np.float32)
    points = points + std[np.asarray(picked, dtype=np.intp)] * noise
    box = np.asarray(bounds, dtype=np.float32)
    if box.shape == (2, train_mu.shape[1]) and np.isfinite(box).all():
        points = np.clip(points, box[0], box[1])
    return points, picked


def warmstart_result_fields(records: Sequence[dict] | None) -> dict:
    """Summary keys for reconstruction misses and noisy fill-ins."""
    if not records:
        return {}
    misses: list[dict] = []
    n_noisy = 0
    n_predicted = 0
    n_checked = 0
    n_hits = 0
    for rank, record in enumerate(records):
        components = record.get("components") or {}
        if components.get("warmstart_predicted"):
            n_predicted += 1
        if components.get("warmstart_noisy"):
            n_noisy += 1
            continue
        if "warmstart_reconstructed" not in components:
            continue
        n_checked += 1
        if components.get("warmstart_reconstructed"):
            n_hits += 1
            continue
        misses.append(
            {
                "rank": rank,
                "train_index": components.get("warmstart_train_index"),
                "label": record.get("decoded"),
                "greedy": components.get("warmstart_greedy"),
            }
        )
    payload: dict = {
        "n_warmstart_noisy": n_noisy,
        "n_warmstart_predicted": n_predicted,
        "n_warmstart_reconstruct_checked": n_checked,
        "n_warmstart_reconstruct_hits": n_hits,
        "n_warmstart_reconstruct_misses": len(misses),
    }
    if misses:
        payload["warmstart_reconstruct_misses"] = misses
    return payload


def _warmstart_target_key(target: str, *, task: str) -> str:
    return _warmstart_identity(target, task=task)


def load_word_warmstarts(
    path: str,
    *,
    target: str,
    seed: int,
    count: int,
    task: str = "semantle",
) -> list[str]:
    """Load ``count`` labeled warmstart words for ``(target, seed)``.

    Accepts either ``{"targets": {target: {seed: [word, ...]}}}`` or a flat
    ``{target: {seed: [word, ...]}}`` map. Target keys are matched after
    :func:`_warmstart_identity`.
    """
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    table = payload.get("targets")
    if not isinstance(table, dict):
        table = payload
    key = _warmstart_target_key(target, task=task)
    block = None
    for raw_key, value in table.items():
        if str(raw_key) in ("source", "warmstart_count"):
            continue
        if _warmstart_target_key(str(raw_key), task=task) == key:
            block = value
            break
    if not isinstance(block, dict):
        raise ValueError(f"{path}: no warmstart words for target {target!r}")
    words = block.get(str(seed), block.get(seed))
    if not isinstance(words, list):
        raise ValueError(f"{path}: no warmstart words for {target!r} seed {seed}")
    cleaned = [str(word).strip() for word in words if str(word).strip()]
    if len(cleaned) < count:
        raise ValueError(
            f"{path}: {target!r} seed {seed} has {len(cleaned)} words; "
            f"need {count}"
        )
    return cleaned[:count]


def load_pinned_checkpoint_labels(
    path: str,
    *,
    seed: int,
    count: int,
    target: str,
    task: str = "semantle",
) -> list[str]:
    """Load a frozen per-seed label list for checkpoint-μ warm starts.

    Accepts the p90 protocol file ``{"seeds": {"1": [smiles, ...]}}``, a
    preview dump ``{"seeds": [{"seed": 1, "labels": [...], "rows": [...]}]}``,
    or the target-keyed word-warmstart JSON.
    """
    source = Path(path).expanduser()
    if source.is_dir():
        raise ValueError(
            f"{path}: checkpoint pin file must be JSON, not a warmstart directory"
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{path}: pinned checkpoint labels file does not exist") from exc
    except OSError as exc:
        raise ValueError(f"{path}: could not read pinned checkpoint labels") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    if "seeds" not in payload:
        return load_word_warmstarts(
            path, target=target, seed=seed, count=count, task=task
        )
    block = payload.get("seeds")
    words: list[str] | None = None
    if isinstance(block, dict):
        raw = block.get(str(seed), block.get(seed))
        if isinstance(raw, list):
            words = [str(word).strip() for word in raw if str(word).strip()]
    elif isinstance(block, list):
        for row in block:
            if not isinstance(row, dict):
                continue
            try:
                row_seed = int(row.get("seed", -1))
            except (TypeError, ValueError):
                continue
            if row_seed != int(seed):
                continue
            raw = row.get("labels")
            if not isinstance(raw, list) and isinstance(row.get("rows"), list):
                raw = [
                    item.get("warmstart_word") or item.get("decoded")
                    for item in row["rows"]
                    if isinstance(item, dict)
                ]
            if isinstance(raw, list):
                words = [str(word).strip() for word in raw if str(word).strip()]
            break
    if words is None:
        raise ValueError(f"{path}: no pinned labels for seed {seed}")
    if len(words) < count:
        raise ValueError(
            f"{path}: seed {seed} has {len(words)} pinned labels; need {count}"
        )
    return words[:count]


def checkpoint_indices_for_labels(
    words: Sequence[str],
    labels: Sequence[str],
    *,
    task: str,
    target: str,
    allow_missing: bool = False,
) -> list[int | None]:
    """Map pinned labels onto first-occurrence train rows.

    ``allow_missing=True`` returns ``None`` for labels that are not in the
    train table so the caller can predict μ from the bias network.
    """
    target_key = _warmstart_identity(target, task=task)
    index_by_key: dict[str, int] = {}
    for index, raw in enumerate(words):
        text = str(raw).strip()
        if not text:
            continue
        key = _warmstart_identity(text, task=task)
        if not key or key in index_by_key:
            continue
        index_by_key[key] = index
    indices: list[int | None] = []
    seen: set[str] = set()
    for label in labels:
        text = str(label).strip()
        key = _warmstart_identity(text, task=task)
        if not key:
            raise ValueError(f"pinned warmstart {label!r} has no usable identity")
        if key == target_key:
            raise ValueError(
                f"pinned warmstart {label!r} is the search target and cannot be used"
            )
        if key in seen:
            raise ValueError(f"pinned warmstart {label!r} is duplicated")
        index = index_by_key.get(key)
        if index is None and not allow_missing:
            raise ValueError(
                f"pinned warmstart {label!r} is not in this checkpoint train set"
            )
        seen.add(key)
        indices.append(index)
    return indices


PINNED_MOLOPT_P90_SIZES = (1024, 2048, 3072)
PINNED_MOLOPT_P90_DIR_RELATIVE = "experiments/molopt/warmstarts"
PINNED_MOLOPT_P90_1024_RELATIVE = f"{PINNED_MOLOPT_P90_DIR_RELATIVE}/p90_1024.json"
PINNED_MOLOPT_P90_1024 = (
    Path(__file__).resolve().parents[2] / PINNED_MOLOPT_P90_1024_RELATIVE
)


def pinned_molopt_p90_relative(n: int) -> str:
    return f"{PINNED_MOLOPT_P90_DIR_RELATIVE}/p90_{int(n)}.json"


def _catalog_train_size(cfg: Mapping, split: Mapping | None, output_dir: str | None):
    n = cfg.get("num_training_examples")
    if n is None and isinstance(split, dict):
        n = split.get("n_train")
    if n is None and output_dir:
        items_path = Path(output_dir).expanduser() / "items.json"
        if items_path.is_file():
            payload = json.loads(items_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                n = len(payload)
    if n is None:
        n = cfg.get("train_n_samples")
    return n


def molopt_p90_catalog_size(
    output_dir: str | None = None,
    saved_cfg: Mapping | None = None,
) -> int | None:
    """Train-set size when this run is a p90 molopt catalog we pin, else None."""
    cfg: dict = dict(saved_cfg or {})
    if output_dir and not cfg:
        cfg = dict(load_merged_run_config(output_dir))
    task = str(cfg.get("task") or "")
    if task and task != "molopt":
        return None
    cap = cfg.get("molopt_oracle_cap_percentile")
    split = cfg.get("oracle_split")
    if not isinstance(split, dict) and output_dir:
        split_path = Path(output_dir).expanduser() / "oracle_split.json"
        if split_path.is_file():
            loaded = json.loads(split_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                split = loaded
    if cap is None and isinstance(split, dict):
        cap = split.get("percentile")
    n = _catalog_train_size(cfg, split if isinstance(split, dict) else None, output_dir)
    try:
        size = int(n)
        if abs(float(cap) - 90.0) < 1e-6 and size in PINNED_MOLOPT_P90_SIZES:
            return size
    except (TypeError, ValueError):
        return None
    return None


def checkpoint_is_molopt_p90_1024(
    output_dir: str | None = None,
    saved_cfg: Mapping | None = None,
) -> bool:
    """True when this run trained on the 1024-molecule p90 molopt catalog."""
    return molopt_p90_catalog_size(output_dir, saved_cfg=saved_cfg) == 1024


def pinned_checkpoint_labels_file(
    output_dir: str | None = None,
    saved_cfg: Mapping | None = None,
) -> str | None:
    """Protocol JSON of shared labels for a pinned p90 molopt checkpoint.

    Prefers the repo-relative path so search ``config.json`` is portable
    across machines that share the same working directory (the cluster
    launchers ``cd`` to the repo root).
    """
    n = molopt_p90_catalog_size(output_dir, saved_cfg=saved_cfg)
    if n is None:
        return None
    relative = pinned_molopt_p90_relative(n)
    if Path(relative).is_file():
        return relative
    absolute = Path(__file__).resolve().parents[2] / relative
    if absolute.is_file():
        return str(absolute)
    return None


def resolve_checkpoint_pin_file(
    output_dir: str,
    *,
    warmstart_source: str,
    warmstart_file: str | None,
    saved_cfg: Mapping | None = None,
) -> str | None:
    """Keep an explicit ``warmstart_file``; otherwise attach the matching p90 pin."""
    if warmstart_source != "checkpoint":
        return warmstart_file
    if warmstart_file:
        return warmstart_file
    return pinned_checkpoint_labels_file(output_dir, saved_cfg=saved_cfg)


def resolve_file_warmstart_path(path: str, seed: int) -> str:
    """``path`` may be a jsonl file or a directory of ``seed_<n>.jsonl`` dumps."""
    root = Path(path).expanduser()
    if root.is_dir():
        candidate = root / f"seed_{seed}.jsonl"
        if not candidate.is_file():
            raise ValueError(f"{path}: missing seed_{seed}.jsonl")
        return str(candidate.resolve())
    return str(root)


def _load_file_warmstarts(
    path: str, count: int, dimension: int
) -> tuple[np.ndarray, list[dict]]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            point = record.get("point", record.get("bias"))
            if point is None or len(point) != dimension:
                raise ValueError(
                    f"{path}:{line_number}: expected point/bias with dimension {dimension}"
                )
            record = {**record, "point": [float(v) for v in point]}
            decoded = record.get("decoded")
            if decoded is not None:
                record["decoded"] = unwrap_smiles_tags(str(decoded))
            records.append(record)
            if len(records) == count:
                break
    if len(records) < count:
        raise ValueError(f"{path}: requested {count} warm starts, found {len(records)}")
    return np.asarray([record["point"] for record in records]), records


def _clip_bias_to_bounds(point, bounds) -> np.ndarray:
    lo = np.asarray(bounds[0], dtype=np.float32)
    hi = np.asarray(bounds[1], dtype=np.float32)
    vec = np.asarray(point, dtype=np.float32).reshape(-1)
    return np.clip(vec, lo, hi)


def _predict_pinned_missing_mu(
    ckpt,
    labels: Sequence[str],
    *,
    task: str,
    saved: Mapping,
    bounds: np.ndarray,
    rank: int,
) -> dict[str, np.ndarray]:
    """Bias-network μ for pinned labels that are not train rows, clipped to the box."""
    if getattr(ckpt, "reft_model", None) is None:
        raise ValueError(
            "pinned warmstarts not in this train set need a bias-network "
            f"checkpoint to predict μ ({len(labels)} missing)"
        )
    from boreft.text_similarity import definition_lookup_for_cfg

    kwargs = bias_predict_kwargs(saved, tokenizer=getattr(ckpt, "tokenizer", None))
    preds = predict_bias_vectors_for_words(
        ckpt.reft_model,
        list(labels),
        task=task,
        definition_lookup=definition_lookup_for_cfg(saved),
        **kwargs,
    )
    if len(preds) != len(labels):
        raise ValueError(
            f"bias network returned {len(preds)} μ for {len(labels)} "
            "pinned labels not in the train set"
        )
    predicted: dict[str, np.ndarray] = {}
    for label, vec in zip(labels, preds):
        point = _clip_bias_to_bounds(vec, bounds)
        if point.shape[0] != rank:
            raise ValueError(
                f"predicted μ for {label!r} has dimension {point.shape[0]}; "
                f"expected {rank}"
            )
        predicted[str(label)] = point
    return predicted


def select_warmstarts(
    config: SearchConfig,
    ckpt,
    train_mu: np.ndarray,
    bounds: np.ndarray,
    seed: int,
    ellipsoid=None,
    decode: Callable[[np.ndarray], str] | None = None,
) -> tuple[np.ndarray, list[dict] | None]:
    """Return warmstart latent points and optional per-point records.

    ``checkpoint`` samples labeled train words (excluding the search target)
    so every method with the same checkpoint, target, count, and seed warms
    up on the same solutions. Those records set ``decoded`` to the train
    label; the BO / baseline loops score that word instead of decoding the
    bias vector. The corresponding train bias row is still the latent point
    used to fit BOReFT's GP. When ``decode`` is provided, a candidate is kept
    only if greedy decode of that μ matches the train label; failures are
    replaced by another unused train row, and leftover slots are filled with
    noisy posterior samples (``μ + ε ⊙ σ``) whose greedy decodes are frozen
    so every method scores the same molecules.
    ``warmstart_file`` pins the labels so every method shares the same
    molecules. Labels in this checkpoint's train table use that row's μ.
    Labels that are not in the table are mapped through the live bias
    network (clipped to the search box) and tagged ``warmstart_predicted``.
    Reconstruction misses warn and are logged, but the labeled molecules
    and those means are still used.

    ``ellipsoid`` only changes Sobol warm starts: they are drawn uniformly
    in the Mahalanobis ball instead of in its axis-aligned bounding box.
    Labeled checkpoint / word warm starts stay on the train ``μ`` rows.
    """
    count = config.warmstart_count
    if count < 2:
        raise ValueError("warmstart_count must be at least 2 for exact GP fitting")
    if config.warmstart_source == "checkpoint":
        words = [str(word) for word in getattr(ckpt, "words", []) or []]
        if len(words) != len(train_mu):
            raise ValueError(
                "checkpoint warm starts need train words aligned with bias rows "
                f"(got {len(words)} words and {len(train_mu)} rows)"
            )
        saved = getattr(ckpt, "saved_cfg", None) or {}
        task = str(saved.get("task") or "semantle")
        skipped: list[tuple[int, str, str]] = []
        greedy_by_index: dict[int, str] = {}
        reconstructed_by_index: dict[int, bool] = {}
        keep: Callable[[int], bool] | None = None
        if decode is not None:

            def keep(index: int) -> bool:
                gold = str(words[index]).strip()
                decoded = str(
                    decode(np.asarray(train_mu[index], dtype=np.float32))
                )
                greedy_by_index[index] = decoded
                ok = _matches_search_target(decoded, gold, task)
                reconstructed_by_index[index] = ok
                if ok:
                    return True
                skipped.append((index, gold, decoded))
                return False

        pinned_file = resolve_checkpoint_pin_file(
            config.output_dir,
            warmstart_source=config.warmstart_source,
            warmstart_file=config.warmstart_file,
            saved_cfg=saved,
        )
        pinned = bool(pinned_file)
        noisy_points = np.zeros((0, train_mu.shape[1]), dtype=np.float32)
        noisy_records: list[dict] = []
        if pinned:
            labels = load_pinned_checkpoint_labels(
                pinned_file,
                seed=seed,
                count=count,
                target=config.target,
                task=task,
            )
            indices = checkpoint_indices_for_labels(
                words,
                labels,
                task=task,
                target=config.target,
                allow_missing=True,
            )
            predicted_by_label: dict[str, np.ndarray] = {}
            predicted_greedy: dict[str, str] = {}
            predicted_reconstructed: dict[str, bool] = {}
            missing_labels = [
                label for label, index in zip(labels, indices) if index is None
            ]
            if missing_labels:
                predicted_by_label = _predict_pinned_missing_mu(
                    ckpt,
                    missing_labels,
                    task=task,
                    saved=saved,
                    bounds=bounds,
                    rank=int(train_mu.shape[1]),
                )
                preview = ", ".join(repr(label) for label in missing_labels[:5])
                extra = (
                    ""
                    if len(missing_labels) <= 5
                    else f" (+{len(missing_labels) - 5} more)"
                )
                print(
                    f"[warmstart] {len(missing_labels)} pinned labels are not in "
                    f"this train set; predicting μ via the bias network: "
                    f"{preview}{extra}",
                    flush=True,
                )
            if keep is not None:
                for index in indices:
                    if index is not None:
                        keep(index)
                for label, point in predicted_by_label.items():
                    greedy = str(decode(np.asarray(point, dtype=np.float32)))
                    predicted_greedy[label] = greedy
                    ok = _matches_search_target(greedy, label, task)
                    predicted_reconstructed[label] = ok
                    if not ok:
                        skipped.append((-1, label, greedy))
            n_lookup = sum(index is not None for index in indices)
            print(
                f"[warmstart] pinned {n_lookup} train μ"
                f"{f' + {len(missing_labels)} predicted' if missing_labels else ''} "
                f"from {pinned_file} seed {seed}",
                flush=True,
            )
        else:
            indices = checkpoint_warmstart_indices(
                words,
                target=config.target,
                count=count,
                seed=seed,
                task=task,
                accept=keep,
            )
            missing = count - len(indices)
            if missing:
                if decode is None:
                    raise ValueError(
                        "noisy warmstart fill-ins need a greedy decode to freeze "
                        "shared molecules"
                    )
                eligible = _checkpoint_eligible_indices(
                    words, target=config.target, task=task
                )
                used = set(indices)
                sources = [index for index in eligible if index not in used] or indices
                noisy_points_list: list[np.ndarray] = []
                stream = 0
                while len(noisy_points_list) < missing:
                    batch, batch_sources = _noisy_checkpoint_points(
                        train_mu,
                        sources,
                        count=missing - len(noisy_points_list),
                        seed=seed,
                        ckpt=ckpt,
                        bounds=bounds,
                        stream=stream,
                    )
                    stream += 1
                    if stream > 16:
                        raise ValueError(
                            "could not freeze greedy decodes for "
                            f"{missing} noisy warmstart fill-ins"
                        )
                    for source, point in zip(batch_sources, batch):
                        greedy = str(decode(np.asarray(point, dtype=np.float32)))
                        labeled = normalize_search_text(greedy, task=task)
                        if not labeled:
                            continue
                        noisy_points_list.append(np.asarray(point, dtype=np.float32))
                        noisy_records.append(
                            {
                                "decoded": labeled,
                                "sample_count": 1,
                                "components": {
                                    "warmstart_labeled": False,
                                    "warmstart_noisy": True,
                                    "warmstart_train_index": int(source),
                                    "warmstart_greedy": greedy,
                                },
                            }
                        )
                        if len(noisy_points_list) == missing:
                            break
                noisy_points = np.stack(noisy_points_list, axis=0)
                print(
                    f"[warmstart] reconstructing pool exhausted at "
                    f"{len(indices)}/{count}; filling {missing} noisy "
                    f"posterior samples with frozen greedy decodes",
                    flush=True,
                )
        if skipped:
            preview = "; ".join(
                f"[{index}] {gold!r} → {decoded!r}"
                for index, gold, decoded in skipped[:5]
            )
            extra = "" if len(skipped) <= 5 else f" (+{len(skipped) - 5} more)"
            if pinned:
                message = (
                    f"{len(skipped)} pinned μ did not greedily "
                    f"reconstruct; using the labeled molecules anyway: "
                    f"{preview}{extra}"
                )
                warnings.warn(message, stacklevel=2)
                print(f"[warmstart] WARNING: {message}", flush=True)
            else:
                print(
                    f"[warmstart] discarded {len(skipped)} train μ that did not "
                    f"greedily reconstruct: {preview}{extra}",
                    flush=True,
                )
        records = []
        labeled_points = []
        if pinned:
            for label, index in zip(labels, indices):
                if index is None:
                    word = str(label).strip()
                    labeled = normalize_search_text(word, task=task)
                    if not labeled:
                        raise ValueError(
                            f"predicted pinned warm start {label!r} has no usable label"
                        )
                    point = predicted_by_label.get(label)
                    if point is None:
                        raise ValueError(
                            f"pinned warmstart {label!r} is not in this checkpoint "
                            "train set and has no predicted μ"
                        )
                    reconstructed = predicted_reconstructed.get(label)
                    components = {
                        "warmstart_word": word,
                        "warmstart_labeled": True,
                        "warmstart_pinned": True,
                        "warmstart_predicted": True,
                    }
                    if reconstructed is not None:
                        components["warmstart_reconstructed"] = bool(reconstructed)
                        greedy = predicted_greedy.get(label)
                        if greedy is not None:
                            components["warmstart_greedy"] = str(greedy)
                    records.append(
                        {
                            "decoded": labeled,
                            "sample_count": 1,
                            "components": components,
                        }
                    )
                    labeled_points.append(np.asarray(point, dtype=np.float32))
                    continue
                word = str(words[index]).strip()
                labeled = normalize_search_text(word, task=task)
                if not labeled:
                    raise ValueError(
                        f"checkpoint warm start {index} has no usable train label"
                    )
                reconstructed = reconstructed_by_index.get(index)
                components = {
                    "warmstart_word": word,
                    "warmstart_labeled": True,
                    "warmstart_train_index": int(index),
                    "warmstart_pinned": True,
                }
                if reconstructed is not None:
                    components["warmstart_reconstructed"] = bool(reconstructed)
                    greedy = greedy_by_index.get(index)
                    if greedy is not None:
                        components["warmstart_greedy"] = str(greedy)
                records.append(
                    {
                        "decoded": labeled,
                        "sample_count": 1,
                        "components": components,
                    }
                )
                labeled_points.append(np.asarray(train_mu[index], dtype=np.float32))
        else:
            for index in indices:
                word = str(words[index]).strip()
                labeled = normalize_search_text(word, task=task)
                if not labeled:
                    raise ValueError(
                        f"checkpoint warm start {index} has no usable train label"
                    )
                reconstructed = reconstructed_by_index.get(index)
                components = {
                    "warmstart_word": word,
                    "warmstart_labeled": True,
                    "warmstart_train_index": int(index),
                }
                if reconstructed is not None:
                    components["warmstart_reconstructed"] = bool(reconstructed)
                    greedy = greedy_by_index.get(index)
                    if greedy is not None:
                        components["warmstart_greedy"] = str(greedy)
                records.append(
                    {
                        "decoded": labeled,
                        "sample_count": 1,
                        "components": components,
                    }
                )
                labeled_points.append(np.asarray(train_mu[index], dtype=np.float32))
        records.extend(noisy_records)
        if labeled_points:
            points = np.stack(labeled_points, axis=0)
            if len(noisy_points):
                points = np.concatenate([points, noisy_points], axis=0)
        else:
            points = noisy_points
        if len(points) != count:
            raise ValueError(
                f"checkpoint warm starts produced {len(points)} points; need {count}"
            )
        return np.asarray(points, dtype=np.float32), records
    if config.warmstart_source == "words":
        if not config.warmstart_file:
            raise ValueError("warmstart_source=words requires --warmstart-file")
        saved = getattr(ckpt, "saved_cfg", None) or {}
        task = str(saved.get("task") or "semantle")
        words = load_word_warmstarts(
            config.warmstart_file,
            target=config.target,
            seed=seed,
            count=count,
            task=task,
        )
        kwargs = bias_predict_kwargs(saved, tokenizer=getattr(ckpt, "tokenizer", None))
        points = predict_bias_vectors_for_words(
            ckpt.reft_model,
            words,
            task=task,
            **kwargs,
        )
        records = []
        for word in words:
            labeled = normalize_search_text(word, task=task)
            if not labeled:
                raise ValueError(f"word warm start {word!r} has no usable label")
            records.append(
                {
                    "decoded": labeled,
                    "sample_count": 1,
                    "components": {
                        "warmstart_word": word,
                        "warmstart_labeled": True,
                    },
                }
            )
        return np.asarray(points, dtype=np.float32), records
    if config.warmstart_source == "sobol":
        if ellipsoid is not None:
            return ellipsoid.sample(count, seed=seed).astype(np.float32), None
        engine = torch.quasirandom.SobolEngine(
            dimension=train_mu.shape[1], scramble=True, seed=seed
        )
        unit = engine.draw(count).numpy()
        return bounds[0] + unit * (bounds[1] - bounds[0]), None
    if config.warmstart_source == "file":
        if not config.warmstart_file:
            raise ValueError("warmstart_source=file requires --warmstart-file")
        return _load_file_warmstarts(
            resolve_file_warmstart_path(config.warmstart_file, seed),
            count,
            train_mu.shape[1],
        )
    if config.warmstart_source == "llm":
        intervention = list(ckpt.reft_model.interventions.values())[0]
        intervention = (
            intervention[0] if isinstance(intervention, (list, tuple)) else intervention
        )
        if getattr(intervention, "bias_network", None) is None:
            raise ValueError(
                "warmstart_source=llm requires a checkpoint with add_bias_network=True"
            )
        texts = [
            generate_base(
                ckpt.reft_model.model,
                ckpt.tokenizer,
                ckpt.prompt,
                max_new_tokens=config.max_new_tokens,
                do_sample=True,
                temperature=config.warmstart_temperature,
                top_p=config.warmstart_top_p,
                from_chat_template=ckpt.from_chat_template,
                assistant_suffix=ckpt.assistant_suffix,
            )
            for _ in range(count)
        ]
        saved = ckpt.saved_cfg or {}
        if saved.get("use_definition_embeds") and getattr(
            intervention, "bias_input_source", "embed_cache"
        ) != "llm_encoder":
            raise ValueError(
                "LLM warm starts cannot supply the definitions required by this "
                "definition-embedding bias network"
            )
        if getattr(intervention, "bias_input_source", "embed_cache") == "llm_encoder":
            points = predict_bias_vectors_from_raw_texts(
                ckpt.reft_model,
                texts,
                task=saved.get("task", "semantle"),
                tokenizer=ckpt.tokenizer,
                encoder_max_length=int(saved.get("bias_encoder_max_length", 64)),
                encoder_layer_index=saved.get("bias_encoder_layer_index"),
            )
        else:
            points = predict_bias_vectors_for_words(
                ckpt.reft_model,
                texts,
                task=saved.get("task", "semantle"),
                embed_model=bias_network_embed_model_from_cfg(saved),
            )
        return np.asarray(points), [
            {"components": {"llm_seed_text": text}} for text in texts
        ]
    raise ValueError(f"unknown warmstart source: {config.warmstart_source!r}")


def _resolved_seeds(config: SearchConfig) -> tuple[int, ...]:
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


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _config_json(config: SearchConfig) -> dict:
    return json.loads(json.dumps(asdict(config)))


def _search_config_defaults() -> dict:
    return _config_json(SearchConfig(output_dir="", target="_"))


def _resume_config_changed(
    previous: dict,
    current: dict,
    defaults: dict,
    mutable: set[str],
    prefix: str = "",
) -> list[str]:
    changed: list[str] = []
    keys = set(previous) | set(current) | set(defaults)
    for key in sorted(keys):
        dotted = f"{prefix}.{key}" if prefix else key
        if dotted in mutable:
            continue
        default = defaults.get(key)
        prev_val = previous.get(key, default)
        curr_val = current.get(key, default)
        if isinstance(prev_val, dict) or isinstance(curr_val, dict):
            nested_default = default if isinstance(default, dict) else {}
            prev_nested = prev_val if isinstance(prev_val, dict) else nested_default
            curr_nested = curr_val if isinstance(curr_val, dict) else nested_default
            changed.extend(
                _resume_config_changed(
                    prev_nested,
                    curr_nested,
                    nested_default,
                    mutable,
                    dotted,
                )
            )
            continue
        if prev_val != curr_val:
            changed.append(dotted)
    return changed


def _validate_resume_config(path: Path, config: SearchConfig) -> None:
    if not path.exists():
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    current = _config_json(config)
    defaults = _search_config_defaults()
    mutable = {
        "budget",
        "resume",
        "overwrite",
        "seeds",
        "repeats",
        "max_new_tokens",
        "log_gp_surrogate",
        *WANDB_CONFIG_KEYS,
        *(
            f"expand.{name}"
            for name in SearchExpandConfig.RESUME_MUTABLE_FIELDS
        ),
    }
    changed = _resume_config_changed(previous, current, defaults, mutable)
    if changed:
        raise ValueError(
            "resume configuration differs in trajectory-defining fields: "
            + ", ".join(changed)
        )


def _safe_search_root(search_dir: str, output_dir: str) -> Path:
    root = Path(search_dir).expanduser().resolve()
    checkpoint = Path(output_dir).expanduser().resolve()
    if root == checkpoint or root in checkpoint.parents:
        raise ValueError(
            "search_dir must not equal or contain the checkpoint output_dir"
        )
    return root


def run_search(config: SearchConfig) -> dict:
    """Load one checkpoint and execute all requested seeds."""
    config.validate()
    seeds = _resolved_seeds(config)
    bo_config = BOConfig(
        budget=config.budget,
        surrogate=config.surrogate,
        acquisition=config.acquisition,
        batch_size=config.batch_size,
        observation_samples=config.observation_samples,
        kernel=config.kernel,
        use_ard=config.use_ard,
        projection_dim=config.projection_dim,
        projection_layers=config.projection_layers,
        projection_steps=config.projection_steps,
        gp_lr=config.gp_lr,
        projection_lr=config.projection_lr,
        acquisition_restarts=config.acquisition_restarts,
        acquisition_raw_samples=config.acquisition_raw_samples,
        acquisition_mc_samples=config.acquisition_mc_samples,
        ucb_beta=config.ucb_beta,
        thompson_candidates=config.thompson_candidates,
        duplicate_tolerance=config.duplicate_tolerance,
        log_gp_surrogate=config.log_gp_surrogate,
    )
    bo_config.validate()
    root = _safe_search_root(config.search_dir, config.output_dir)
    saved = load_merged_run_config(config.output_dir)
    model_name = config.model_name or saved.get("model_name")
    if not model_name:
        raise ValueError("model name is absent from both CLI and checkpoint config")
    layer = config.layer if config.layer is not None else int(saved.get("layer", 15))
    rank = (
        config.low_rank_dim
        if config.low_rank_dim is not None
        else int(saved.get("low_rank_dim", 8))
    )
    ckpt = load_eval_checkpoint(
        config.output_dir,
        model_name,
        layer,
        rank,
        config.cache_dir or saved.get("cache_dir"),
        torch_dtype=config.torch_dtype,
        load_latest=bool(config.load_latest),
    )
    task = saved.get("task", "semantle")
    if task not in ("semantle", "molopt", "hypogen"):
        raise ValueError(f"search does not support checkpoint task {task!r}")
    if config.oracle and task != "molopt":
        raise ValueError("property oracles are only valid for molopt checkpoints")
    if task == "molopt":
        verifier = molopt_search_verifier(
            target=config.target,
            objective=config.objective,
            rdkit_map_path=rdkit_map_path_for_cfg(saved),
            oracle=config.oracle,
        )
    else:
        if config.objective != "embed_sim":
            raise ValueError(f"objective={config.objective!r} is only valid for molopt")
        verifier = TextSimilarityVerifier(config.target, task=task)

    if config.expand.enabled():
        intervention = list(ckpt.reft_model.interventions.values())[0]
        intervention = (
            intervention[0] if isinstance(intervention, (list, tuple)) else intervention
        )
        if getattr(intervention, "bias_network", None) is None:
            raise ValueError(
                "search expansion requires a checkpoint trained with "
                "--add-bias-network"
            )

    train_mu, bounds, ellipsoid = training_search_region(
        ckpt.reft_model,
        list(range(len(ckpt.words))),
        padding=config.bounds_padding,
        aabb_std_k=config.aabb_std_k,
        search_domain=config.search_domain,
    )
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
    _write_json(root / "config.json", asdict(config))

    decode_prompt, decode_from_chat, decode_span = resolve_search_decode_prompt(
        ckpt, config, task=task, model_name=model_name
    )
    decode = checkpoint_decode(
        ckpt,
        config.max_new_tokens,
        sampling_temperature=config.sampling_temperature,
        task=task,
        prompt=decode_prompt,
        from_chat_template=decode_from_chat,
        content_span=decode_span,
    )
    reconstruct_decode = None
    if config.warmstart_source == "checkpoint":
        reconstruct_decode = memoize_point_decode(
            checkpoint_reconstruct_decode(
                ckpt,
                config.max_new_tokens,
                task=task,
            )
        )
    print("=" * 64)
    print("Bayesian optimization")
    print(f"  target      {config.target}")
    if config.oracle:
        print(f"  oracle      {config.oracle}")
    print(f"  task        {task}")
    print(f"  checkpoint  {config.output_dir}")
    print(f"  output      {root}")
    print(f"  model       {model_name}")
    samples = config.observation_samples
    warm_each = (
        1 if config.warmstart_source in ("checkpoint", "words") else samples
    )
    warm_calls = config.warmstart_count * warm_each
    search_calls = config.budget - warm_calls
    print(
        f"  budget      {config.budget} verifications  "
        f"(warmstart={config.warmstart_count}×{warm_each}, "
        f"acquisition={search_calls // samples}×{samples})"
    )
    print(
        f"  surrogate   {config.surrogate}  ·  kernel={config.kernel}  "
        f"·  acquisition={config.acquisition}  "
        f"·  batch={config.batch_size}"
        f"{'  ·  ARD' if config.use_ard else ''}"
        f"{f'  ·  layers={config.projection_layers}' if config.surrogate == 'projected' else ''}"
    )
    prompt_label = (
        "custom" if (config.task_description or "").strip() else config.search_prompt
    )
    print(
        f"  decode      samples={config.observation_samples}  "
        f"T={config.sampling_temperature}  "
        f"prompt={prompt_label}"
    )
    print(
        f"  bounds      domain={config.search_domain}  "
        f"aabb_std_k={config.aabb_std_k}  "
        f"padding={config.bounds_padding}"
    )
    print(f"  seeds       {', '.join(str(seed) for seed in seeds)}")
    if config.expand.enabled():
        print(
            f"  expand      every={config.expand.every} batches  "
            f"new={config.expand.new_strategy}  "
            f"replay={config.expand.replay_strategy}  "
            f"init={config.expand.init}  "
            f"learn_w={config.expand.learn_w}  "
            f"learn_r={config.expand.learn_r}  "
            f"learn_bias_network={config.expand.learn_bias_network}"
            f"{f'  new_only_epochs={config.expand.new_only_epochs}' if config.expand.new_only_epochs else ''}"
        )
    print("=" * 64, flush=True)

    states: list[RunState] = []
    summaries: dict[str, dict] = {}
    for seed in seeds:
        run_dir = root / f"seed_{seed}"
        seed_started = time.monotonic()
        previous_elapsed = 0.0
        previous_summary_exists = (
            config.resume and (run_dir / "summary.json").exists()
        )
        if previous_summary_exists:
            previous_summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
            previous_elapsed = float(previous_summary.get("elapsed_seconds", 0.0))
        state = RunState.load(run_dir / "observations.jsonl")
        if config.resume and not previous_summary_exists:
            previous_elapsed = float(state.summary()["elapsed_seconds"])
        _seed_everything(seed)
        warm_points, records = select_warmstarts(
            config,
            ckpt,
            train_mu,
            bounds,
            seed,
            ellipsoid=ellipsoid,
            decode=reconstruct_decode,
        )
        warm_log = warmstart_result_fields(records)
        wandb_session = None
        if config.expand.enabled():
            wandb_session = start_search_wandb_session(
                run_dir,
                project=config.wandb_project,
                entity=config.wandb_entity,
                group=config.wandb_group,
                name=(
                    None
                    if not config.wandb_run_name
                    else f"{config.wandb_run_name}_seed{seed}"
                ),
                wandb_dir=config.wandb_dir,
                extra_config=asdict(config),
                no_wandb=config.no_wandb,
                budget=config.budget,
                target=config.target,
            )
        on_observation = (
            wandb_session.log_observations if wandb_session is not None else None
        )
        on_after_batch = None
        expand_snapshot = None
        if config.expand.enabled():
            expand_snapshot = snapshot_search_intervention(ckpt)
            load_expansion_weights(run_dir, ckpt)
            on_after_batch = make_expansion_hook(
                ckpt=ckpt,
                config=config.expand,
                task=task,
                seed=seed,
                bounds=bounds,
                bounds_padding=config.bounds_padding,
                aabb_std_k=config.aabb_std_k,
                search_domain=config.search_domain,
                seed_dir=run_dir,
                raw_defs=load_run_raw_definitions(saved),
                progress_factory=(
                    wandb_session.expansion_callback
                    if wandb_session is not None
                    else None
                ),
            )
        try:
            state = run_bo(
                seed=seed,
                bounds=bounds,
                warmstart_points=warm_points,
                warmstart_records=records,
                decode=decode,
                verifier=verifier,
                state=state,
                config=bo_config,
                ellipsoid=ellipsoid,
                on_observation=on_observation,
                on_after_batch=on_after_batch,
            )
        except Exception:
            if wandb_session is not None:
                try:
                    wandb_session.finish()
                except Exception as exc:
                    print(f"[wandb] failed to finish live run: {exc}", flush=True)
            raise
        finally:
            if expand_snapshot is not None:
                restore_search_intervention(ckpt, expand_snapshot)
        n_repeat_samples, n_repeat_proposals = recorded_repeat_counts(state)
        found, found_index, found_at = found_target_stats(
            state.observations,
            config.target,
            task,
            tolerance=bo_config.maximum_tolerance,
            property_search=bool(config.oracle),
        )
        expand_state = load_expansion_state(run_dir)
        summary = {
            **state.summary(),
            "seed": seed,
            "elapsed_seconds": previous_elapsed
            + (time.monotonic() - seed_started),
            "n_repeat_samples": n_repeat_samples,
            "n_repeat_proposals": n_repeat_proposals,
            "found_target": found,
            "found_at_index": found_index,
            "found_at_verifications": found_at,
            "n_expansion_rounds": expand_state.n_rounds,
            **warm_log,
        }
        _write_json(run_dir / "summary.json", summary)
        plot_best_so_far(
            [state.observations],
            run_dir / "best_so_far.png",
            title=f"Seed {seed}",
        )
        if wandb_session is not None:
            wandb_session.finish(summary)
        else:
            maybe_log_finished_seed(
                run_dir,
                project=config.wandb_project,
                entity=config.wandb_entity,
                group=config.wandb_group,
                name=(
                    None
                    if not config.wandb_run_name
                    else f"{config.wandb_run_name}_seed{seed}"
                ),
                wandb_dir=config.wandb_dir,
                extra_config=asdict(config),
                no_wandb=config.no_wandb,
            )
        states.append(state)
        summaries[str(seed)] = summary
    aggregate = {
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
        "n_found_target": sum(
            1 for summary in summaries.values() if summary.get("found_target")
        ),
        "found_target_seeds": [
            int(summary["seed"])
            for summary in summaries.values()
            if summary.get("found_target")
        ],
        "n_warmstart_reconstruct_misses": sum(
            int(summary.get("n_warmstart_reconstruct_misses") or 0)
            for summary in summaries.values()
        ),
        "n_warmstart_noisy": sum(
            int(summary.get("n_warmstart_noisy") or 0)
            for summary in summaries.values()
        ),
        "n_warmstart_predicted": sum(
            int(summary.get("n_warmstart_predicted") or 0)
            for summary in summaries.values()
        ),
        "n_expansion_rounds": sum(
            int(summary.get("n_expansion_rounds") or 0)
            for summary in summaries.values()
        ),
    }
    _write_json(root / "summary.json", aggregate)
    plot_best_so_far(
        [state.observations for state in states],
        root / "best_so_far.png",
        title="Across Seeds",
    )
    best_run = max(
        summaries.values(),
        key=lambda item: (item["best_score"] is not None, item["best_score"] or 0.0),
    )
    print()
    print("=" * 64)
    if best_run.get("best_score") is None:
        print(f"Finished {len(seeds)} seed(s).  no observations")
    else:
        print(
            f"Finished {len(seeds)} seed(s).  "
            f"overall best={best_run['best_score']:.6g} "
            f"({best_run['best_decoded']!r})  from seed {best_run['seed']}"
        )
    print("=" * 64, flush=True)
    return aggregate


def main() -> None:
    run_search(tyro.cli(SearchConfig))


if __name__ == "__main__":
    main()
