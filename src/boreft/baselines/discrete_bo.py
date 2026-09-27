"""Bayesian optimization over a finite solution candidate pool.

The pool is loaded from JSONL, from unique training targets in the ReFT
``items.json`` (random subset of ``candidate_count``, or all if that flag is
``-1``), or sampled once from the base model, then frozen for every search
seed. An LLM pool is also written to ``pool.jsonl`` in ``--search-dir``.
LLM pool sampling accepts ``--discrete-bo.last-k-incontext`` (condition on the last
*k* unique pool members) and ``--discrete-bo.candidates-per-call`` numbered
candidates per completion (``1. ...`` per line; the index is stripped).
Unless acquisition is ``random``, pool embeddings are computed once at that
construction step and reused across seeds; warm-start solutions not in the
pool are still encoded per seed.
``--discrete-bo.include-target`` appends ``--target`` at that shared
construction step if it is missing.
Each step ranks unobserved pool members by acquisition. ``random`` samples
uniformly from that remaining pool (no GP or embeddings). Otherwise the
shared GP is fit to embeddings of unique observed solutions; a batch is
that ranking's top-k (independent scores, not fantasized qEI). Thompson
uses one posterior draw.

Launch via the shared baseline CLI, for example::

    python -m boreft.baselines.search \
        --baseline discrete_bo \
        --task semantle \
        --task-description "Generate an English word as a guess to find the hidden word (only the word, without any decoration or formatting)." \
        --target computer \
        --reft-output-dir outputs/1784053292 \
        --search-dir outputs/1784053292/search/discreteBO+projected64-computer \
        --budget 500 \
        --warmstart-count 10 \
        --warmstart-source checkpoint \
        --batch-size 1 \
        --seeds 1 2 3 \
        --discrete-bo.include-target \
        --discrete-bo.candidate-source train \
        --discrete-bo.candidate-count -1 \
        --discrete-bo.surrogate projected \
        --discrete-bo.projection-dim 64 \
        --overwrite
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
from typing import Literal, Mapping, Sequence

import numpy as np

from boreft.bo import fit_surrogate, latent_bounds, score_discrete_candidates
from boreft.bo.runner import seed_everything
from boreft.bo.surrogate import KERNEL_CHOICES, KernelKind, SurrogateFitConfig
from boreft.chem import unwrap_smiles_tags

from .base import (
    BaselineObservation,
    Candidate,
    Embed,
    SearchContext,
    as_embedding,
    solution_key,
    unique_observations,
)
from .llm_sample import sample_llm_candidates, validate_llm_sample_knobs

AcquisitionKind = Literal["log_ei", "ucb", "thompson", "random"]
CandidateSource = Literal["file", "llm", "train"]
SurrogateKind = Literal["static", "projected"]
_POOL_LINE_KEYS = ("solution", "text", "target")
_TRAIN_ITEM_KEYS = ("word", "target")


@dataclass(frozen=True)
class DiscreteBOConfig:
    candidate_source: CandidateSource = "llm"
    candidate_file: str | None = None
    candidate_count: int = 1024
    last_k_incontext: int = 0
    candidates_per_call: int = 1
    include_target: bool = False
    acquisition: AcquisitionKind = "log_ei"
    surrogate: SurrogateKind = "static"
    kernel: KernelKind = "matern-2.5"
    use_ard: bool = False
    projection_dim: int = 64
    projection_layers: int = 1
    projection_steps: int = 300
    gp_lr: float = 0.2
    projection_lr: float = 0.002
    bounds_padding: float = 0.0
    embedding_model: str | None = None

    def __post_init__(self) -> None:
        if self.candidate_source not in ("file", "llm", "train"):
            raise ValueError(
                f"unknown discrete_bo candidate source: {self.candidate_source!r}"
            )
        if self.candidate_source == "file" and not self.candidate_file:
            raise ValueError("candidate_source=file requires candidate_file")
        if self.candidate_file is not None and not self.candidate_file.strip():
            raise ValueError("candidate_file must be a non-empty path")
        if self.candidate_count == -1:
            if self.candidate_source != "train":
                raise ValueError(
                    "candidate_count=-1 is only valid with candidate_source=train"
                )
        elif self.candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        validate_llm_sample_knobs(
            last_k_incontext=self.last_k_incontext,
            candidates_per_call=self.candidates_per_call,
        )
        if self.candidate_source != "llm":
            if self.last_k_incontext != 0:
                raise ValueError("last_k_incontext only applies to candidate_source=llm")
            if self.candidates_per_call != 1:
                raise ValueError(
                    "candidates_per_call only applies to candidate_source=llm"
                )
        if self.acquisition not in ("log_ei", "ucb", "thompson", "random"):
            raise ValueError(f"unknown discrete_bo acquisition: {self.acquisition!r}")
        if self.surrogate not in ("static", "projected"):
            raise ValueError(f"unknown discrete_bo surrogate: {self.surrogate!r}")
        if self.kernel not in KERNEL_CHOICES:
            raise ValueError(f"unknown discrete_bo kernel: {self.kernel!r}")
        if self.projection_dim < 1 or self.projection_steps < 1:
            raise ValueError("projection_dim and projection_steps must be positive")
        if self.projection_layers < 1:
            raise ValueError("projection_layers must be positive")
        if self.gp_lr <= 0 or self.projection_lr <= 0:
            raise ValueError("GP and projection learning rates must be positive")
        if self.bounds_padding < 0:
            raise ValueError("bounds_padding must be nonnegative")
        if self.embedding_model is not None and not self.embedding_model.strip():
            raise ValueError("embedding_model must be a non-empty model name")

    def surrogate_fit_config(self) -> SurrogateFitConfig:
        return SurrogateFitConfig(
            kind=self.surrogate,
            kernel=self.kernel,
            use_ard=self.use_ard,
            projection_dim=self.projection_dim,
            projection_layers=self.projection_layers,
            steps=self.projection_steps,
            gp_lr=self.gp_lr,
            projection_lr=self.projection_lr,
        )


def _pool_line_solution(payload: object, path: Path, line_number: int) -> str:
    if isinstance(payload, str):
        solution = payload.strip()
    elif isinstance(payload, dict):
        solution = ""
        for key in _POOL_LINE_KEYS:
            value = payload.get(key)
            if value is not None:
                solution = str(value).strip()
                break
        if not solution:
            raise ValueError(
                f"{path}:{line_number}: JSON object must contain "
                "solution, text, or target"
            )
    else:
        raise ValueError(f"{path}:{line_number}: expected a JSON object or string")
    if not solution:
        raise ValueError(f"{path}:{line_number}: candidate solution must not be blank")
    return unwrap_smiles_tags(solution)


def _load_file_pool(path: str, count: int) -> list[str]:
    pool_path = Path(path).expanduser()
    if not pool_path.is_file():
        raise FileNotFoundError(f"discrete_bo candidate file not found: {pool_path}")
    unique: dict[str, str] = {}
    for line_number, line in enumerate(
        pool_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{pool_path}:{line_number}: invalid JSON") from exc
        solution = _pool_line_solution(payload, pool_path, line_number)
        key = solution_key(solution)
        if key not in unique:
            unique[key] = solution
        if len(unique) >= count:
            break
    if len(unique) < count:
        raise ValueError(
            f"discrete_bo candidate file has {len(unique)} unique solutions; "
            f"need {count}"
        )
    return list(unique.values())


def _train_item_solution(payload: object, path: Path, index: int) -> str:
    if not isinstance(payload, dict):
        raise ValueError(f"{path}:{index}: expected a JSON object")
    for key in _TRAIN_ITEM_KEYS:
        value = payload.get(key)
        if value is not None and str(value).strip():
            return unwrap_smiles_tags(str(value).strip())
    raise ValueError(f"{path}:{index}: JSON object must contain word or target")


def _load_train_pool(output_dir: str, count: int, seed: int) -> list[str]:
    items_path = Path(output_dir).expanduser() / "items.json"
    if not items_path.is_file():
        raise FileNotFoundError(
            f"discrete_bo train pool needs items.json in {items_path.parent}"
        )
    try:
        payload = json.loads(items_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{items_path}: invalid JSON") from exc
    if not isinstance(payload, list):
        raise ValueError(f"{items_path}: expected a JSON list of training items")
    unique: dict[str, str] = {}
    for index, row in enumerate(payload):
        solution = _train_item_solution(row, items_path, index)
        key = solution_key(solution)
        if key not in unique:
            unique[key] = solution
    solutions = list(unique.values())
    if not solutions:
        raise ValueError(f"{items_path}: no usable training solutions")
    if count == -1 or count >= len(solutions):
        return solutions
    return random.Random(seed).sample(solutions, count)


def _sample_llm_pool(context: SearchContext, config: DiscreteBOConfig) -> list[str]:
    sampled = sample_llm_candidates(
        context.generate,
        context.generation_options,
        context.task_description,
        count=config.candidate_count,
        last_k_incontext=config.last_k_incontext,
        candidates_per_call=config.candidates_per_call,
        unique=True,
        task=context.task,
    )
    if len(sampled) < config.candidate_count:
        raise ValueError(
            f"discrete_bo sampled {len(sampled)} unique candidates from the base "
            f"model; need {config.candidate_count}"
        )
    return [item.solution for item in sampled]


def write_discrete_bo_pool(path: str | Path, pool: Sequence[str]) -> Path:
    """Write the frozen pool as JSONL ``{"solution": ...}`` lines."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps({"solution": solution}, ensure_ascii=False) + "\n"
        for solution in pool
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, destination)
    return destination


@dataclass(frozen=True)
class DiscreteBOPoolReport:
    size: int
    source: str
    pool_seed: int
    target: str
    target_present: bool
    target_added: bool


def coerce_pool(raw_pool: object) -> list[str]:
    if not isinstance(raw_pool, list):
        raise ValueError("discrete_bo pool must be a list")
    pool: list[str] = []
    seen: set[str] = set()
    for item in raw_pool:
        solution = unwrap_smiles_tags(str(item).strip())
        if not solution:
            raise ValueError("discrete_bo pool entries must not be blank")
        key = solution_key(solution)
        if key in seen:
            raise ValueError("discrete_bo pool must contain unique solutions")
        seen.add(key)
        pool.append(solution)
    return pool


def coerce_embeddings(raw: object) -> dict[str, list[float]]:
    if not isinstance(raw, dict):
        raise ValueError("discrete_bo embeddings must be a mapping")
    return {str(key): as_embedding(vector) for key, vector in raw.items()}


def fill_embeddings(
    solutions: Sequence[str],
    embed: Embed,
    cache: dict[str, list[float]],
) -> None:
    missing = [
        solution
        for solution in solutions
        if solution_key(solution) not in cache
    ]
    if not missing:
        return
    vectors = embed(missing)
    if len(vectors) != len(missing):
        raise ValueError("embed must return one vector per solution")
    for solution, vector in zip(missing, vectors):
        cache[solution_key(solution)] = as_embedding(vector)


def _source_pool(config: DiscreteBOConfig, context: SearchContext) -> list[str]:
    if config.candidate_source == "file":
        path = config.candidate_file
        if not path:
            raise ValueError("candidate_source=file requires candidate_file")
        return _load_file_pool(path, config.candidate_count)
    if config.candidate_source == "train":
        if not context.checkpoint_dir:
            raise ValueError(
                "candidate_source=train requires SearchContext.checkpoint_dir"
            )
        return _load_train_pool(
            context.checkpoint_dir, config.candidate_count, context.seed
        )
    seed_everything(context.seed)
    return _sample_llm_pool(context, config)


def _append_gold_target(
    pool: Sequence[str], config: DiscreteBOConfig, target: str
) -> tuple[list[str], bool, bool]:
    selected = list(pool)
    gold = unwrap_smiles_tags(target.strip())
    present = solution_key(gold) in {solution_key(item) for item in selected}
    added = False
    if config.include_target and not present:
        selected.append(gold)
        added = True
    return selected, present, added


def log_discrete_bo_pool(
    report: DiscreteBOPoolReport,
    *,
    task: str | None,
    embedding_dim: int | None = None,
    saved_path: Path | None = None,
) -> None:
    line = (
        f"[discrete_bo] pool: {report.size} unique "
        f"(source={report.source}, pool_seed={report.pool_seed})"
    )
    if embedding_dim is not None:
        line += f"; embedded {report.size}×{embedding_dim}"
    if saved_path is not None:
        line += f"; saved {saved_path}"
    if task == "semantle":
        location = "already in the pool" if report.target_present else "not in the pool"
        action = "added it" if report.target_added else "did not add it"
        line += f"; gold target {report.target!r} was {location}; {action}"
    print(line, flush=True)


def select_discrete_bo_pool(
    config: DiscreteBOConfig,
    context: SearchContext,
    *,
    log: bool = True,
) -> tuple[list[str], DiscreteBOPoolReport]:
    """Build the frozen candidate pool and optionally insert the gold target."""
    pool, present, added = _append_gold_target(
        _source_pool(config, context), config, context.target
    )
    report = DiscreteBOPoolReport(
        size=len(pool),
        source=config.candidate_source,
        pool_seed=context.seed,
        target=context.target.strip(),
        target_present=present,
        target_added=added,
    )
    if log:
        log_discrete_bo_pool(report, task=context.task)
    return pool, report


class DiscreteBOBaseline:
    name = "discrete_bo"

    def __init__(
        self,
        config: DiscreteBOConfig | None = None,
        pool: Sequence[str] | None = None,
        embeddings: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        self.config = config or DiscreteBOConfig()
        self._pool: list[str] = list(pool) if pool is not None else []
        self._embeddings: dict[str, list[float]] = (
            coerce_embeddings(dict(embeddings)) if embeddings is not None else {}
        )

    def propose(
        self,
        context: SearchContext,
        history: Sequence[BaselineObservation],
        count: int,
    ) -> Sequence[Candidate]:
        unique = unique_observations(history)
        acquisition = self.config.acquisition
        embed = context.embed
        if acquisition != "random":
            if embed is None:
                raise ValueError("discrete_bo requires context.embed")
            if len(unique) < 2:
                raise ValueError(
                    "discrete_bo needs at least two unique observed solutions"
                )
        pool = self._ensure_pool(context)
        observed = {solution_key(item.solution) for item in unique}
        remaining = [
            (index, solution)
            for index, solution in enumerate(pool)
            if solution_key(solution) not in observed
        ]
        if not remaining:
            raise ValueError("discrete_bo pool is exhausted")
        batch = min(count, len(remaining))
        propose_seed = context.seed * 1_000_003 + len(history)
        if acquisition == "random":
            scores = np.random.default_rng(propose_seed).random(len(remaining))
        else:
            assert embed is not None
            remaining_embeddings = self._embeddings_for(
                [solution for _index, solution in remaining],
                embed,
            )
            train_embeddings = self._embeddings_for(
                [item.solution for item in unique],
                embed,
            )
            bounds = latent_bounds(
                np.concatenate([train_embeddings, remaining_embeddings], axis=0),
                padding=self.config.bounds_padding,
            )
            seed_everything(propose_seed)
            noisy = any(item.sample_count > 1 for item in unique)
            surrogate = fit_surrogate(
                train_embeddings,
                [item.score for item in unique],
                bounds,
                self.config.surrogate_fit_config(),
                observation_variances=(
                    [item.score_sem**2 for item in unique] if noisy else None
                ),
            )
            scores = score_discrete_candidates(
                surrogate,
                remaining_embeddings,
                acquisition=acquisition,
                seed=propose_seed,
            )
        chosen = np.argsort(-scores, kind="stable")[:batch]
        candidates: list[Candidate] = []
        for rank, local_index in enumerate(chosen):
            pool_index, solution = remaining[int(local_index)]
            candidates.append(
                Candidate(
                    solution=solution,
                    metadata={
                        "pool_index": pool_index,
                        "acquisition": float(scores[int(local_index)]),
                        "rank": rank,
                        "is_repeat_proposal": False,
                    },
                )
            )
        return candidates

    def observe(self, observations: Sequence[BaselineObservation]) -> None:
        del observations

    def state_dict(self) -> dict:
        return {
            "pool": list(self._pool),
            "embeddings": {
                key: list(vector) for key, vector in self._embeddings.items()
            },
        }

    def load_state_dict(self, state: Mapping) -> None:
        if not state:
            return
        raw_pool = state.get("pool", [])
        if not self._pool:
            self._pool = coerce_pool(raw_pool)
        incoming = coerce_embeddings(state.get("embeddings", {}))
        self._embeddings = {**self._embeddings, **incoming}

    def _ensure_pool(self, context: SearchContext) -> list[str]:
        if self._pool:
            return self._pool
        pool, _report = select_discrete_bo_pool(self.config, context)
        self._pool = pool
        return pool

    def _embeddings_for(self, solutions: Sequence[str], embed: Embed) -> np.ndarray:
        fill_embeddings(solutions, embed, self._embeddings)
        matrix = np.asarray(
            [self._embeddings[solution_key(solution)] for solution in solutions],
            dtype=np.float64,
        )
        if matrix.ndim != 2 or matrix.shape[0] != len(solutions):
            raise ValueError("cached embeddings must form an [N, D] matrix")
        return matrix
