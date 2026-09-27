"""Bayesian Optimization via Prompting (BOPRO).

Reference: Agarwal et al., "Searching for Optimal Solutions with LLMs via
Bayesian Optimization" (ICLR 2025)
https://openreview.net/forum?id=aVfDrl7xDV

Each step fits the shared GP to embeddings of unique observed solutions,
optimizes acquisition in that box, retrieves the k nearest neighbors of each
proposed vector, and decodes with the OPRO prompt over those scored neighbors.
The GP is updated on actual solution embeddings, not on the unrealized
proposal vectors.

Launch via the shared baseline CLI, for example::

    python -m boreft.baselines.search \
      --baseline bopro \
      --task semantle \
      --task-description "Generate an English word as a guess to find the hidden word (only the word, without any decoration or formatting)." \
      --target computer \
      --reft-output-dir outputs/1784053292 \
      --search-dir outputs/1784053292/search/bopro-computer \
      --budget 500 \
      --warmstart-count 10 \
      --warmstart-source checkpoint \
      --batch-size 1 \
      --seeds 1 2 3 \
      --overwrite
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np

from boreft.bo import fit_surrogate, latent_bounds, propose_candidates
from boreft.bo.runner import seed_everything
from boreft.bo.surrogate import KERNEL_CHOICES, KernelKind, SurrogateFitConfig

from .base import (
    BaselineObservation,
    Candidate,
    Embed,
    SearchContext,
    as_embedding,
    solution_key,
    unique_observations,
)
from .opro import _build_prompt, _parse_solution

AcquisitionKind = Literal["log_ei", "ucb", "thompson"]
SurrogateKind = Literal["static", "projected"]
_DUPLICATE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class BOPROConfig:
    neighbors: int = 5
    acquisition: AcquisitionKind = "log_ei"
    surrogate: SurrogateKind = "static"
    kernel: KernelKind = "matern-2.5"
    use_ard: bool = False
    projection_dim: int = 64
    projection_layers: int = 1
    projection_steps: int = 300
    gp_lr: float = 0.2
    projection_lr: float = 0.002
    bounds_padding: float = 0.25
    embedding_model: str | None = None

    def __post_init__(self) -> None:
        if self.neighbors < 1:
            raise ValueError("neighbors must be positive")
        if self.acquisition not in ("log_ei", "ucb", "thompson"):
            raise ValueError(f"unknown BOPRO acquisition: {self.acquisition!r}")
        if self.surrogate not in ("static", "projected"):
            raise ValueError(f"unknown BOPRO surrogate: {self.surrogate!r}")
        if self.kernel not in KERNEL_CHOICES:
            raise ValueError(f"unknown BOPRO kernel: {self.kernel!r}")
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


def _cosine_similarities(proposal: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    proposal = np.asarray(proposal, dtype=np.float64).reshape(-1)
    matrix = np.asarray(matrix, dtype=np.float64)
    denom = np.linalg.norm(proposal) * np.linalg.norm(matrix, axis=1)
    dots = matrix @ proposal
    return np.divide(
        dots, denom, out=np.zeros(len(matrix), dtype=np.float64), where=denom > 0
    )


def _nearest_neighbors(
    history: Sequence[BaselineObservation],
    embeddings: np.ndarray,
    proposal: np.ndarray,
    k: int,
) -> list[BaselineObservation]:
    """Select by cosine to the proposal, then sort low-to-high score for the prompt."""
    similarities = _cosine_similarities(proposal, embeddings)
    order = np.argsort(-similarities)[: min(k, len(history))]
    selected = [history[int(index)] for index in order]
    selected.sort(key=lambda item: (item.score, item.index))
    return selected


def _is_embedding_duplicate(
    point: np.ndarray,
    observed: np.ndarray,
    bounds: np.ndarray,
    tolerance: float,
) -> bool:
    if len(observed) == 0:
        return False
    span = np.maximum(bounds[1] - bounds[0], np.finfo(np.float64).eps)
    distances = np.linalg.norm((observed - point) / span, axis=-1)
    return bool(np.any(distances <= tolerance))


def _first_solution(texts: Sequence[str]) -> str | None:
    for text in texts:
        solution = _parse_solution(text)
        if solution is not None:
            return solution
    return None


class BOPROBaseline:
    name = "bopro"

    def __init__(self, config: BOPROConfig | None = None) -> None:
        self.config = config or BOPROConfig()
        self._embeddings: dict[str, list[float]] = {}

    def propose(
        self,
        context: SearchContext,
        history: Sequence[BaselineObservation],
        count: int,
    ) -> Sequence[Candidate]:
        if context.embed is None:
            raise ValueError("bopro requires context.embed")
        unique = unique_observations(history)
        if len(unique) < 2:
            raise ValueError("bopro needs at least two unique observed solutions")
        embeddings = self._embeddings_for(unique, context.embed)
        bounds = latent_bounds(embeddings, padding=self.config.bounds_padding)
        fit_seed = context.seed * 1_000_003 + len(history)
        seed_everything(fit_seed)
        noisy = any(item.sample_count > 1 for item in unique)
        surrogate = fit_surrogate(
            embeddings,
            [item.score for item in unique],
            bounds,
            self.config.surrogate_fit_config(),
            observation_variances=(
                [item.score_sem**2 for item in unique] if noisy else None
            ),
        )
        proposals = propose_candidates(
            surrogate,
            bounds,
            embeddings,
            acquisition=self.config.acquisition,
            batch_size=count,
            seed=fit_seed,
            duplicate_tolerance=_DUPLICATE_TOLERANCE,
            observation_samples=context.observation_samples,
        )
        seen_batch: set[str] = set()
        candidates: list[Candidate] = []
        for proposal in np.asarray(proposals, dtype=np.float64):
            neighbors = _nearest_neighbors(
                unique, embeddings, proposal, self.config.neighbors
            )
            prompt = _build_prompt(
                context.task_description, neighbors, task=context.task
            )
            solution = _first_solution(
                context.generate(prompt, 1, context.generation_options)
            )
            if solution is None:
                continue
            key = solution_key(solution)
            if key in seen_batch:
                continue
            seen_batch.add(key)
            candidates.append(
                Candidate(
                    solution=solution,
                    metadata={
                        "prompt": prompt,
                        "history_indices": [item.index for item in neighbors],
                        "proposal": [float(value) for value in proposal],
                        "is_repeat_proposal": _is_embedding_duplicate(
                            proposal, embeddings, bounds, _DUPLICATE_TOLERANCE
                        ),
                    },
                )
            )
        if not candidates:
            raise ValueError(
                f"{self.name} received only blank samples from the base model"
            )
        return candidates

    def observe(self, observations: Sequence[BaselineObservation]) -> None:
        del observations

    def state_dict(self) -> dict:
        return {
            "embeddings": {
                key: list(vector) for key, vector in self._embeddings.items()
            }
        }

    def load_state_dict(self, state: Mapping) -> None:
        raw = state.get("embeddings", {}) if state else {}
        if not isinstance(raw, dict):
            raise ValueError("bopro embeddings must be a mapping")
        self._embeddings = {
            str(key): as_embedding(vector) for key, vector in raw.items()
        }

    def _embeddings_for(
        self,
        history: Sequence[BaselineObservation],
        embed: Embed,
    ) -> np.ndarray:
        missing = [
            item
            for item in history
            if solution_key(item.solution) not in self._embeddings
        ]
        if missing:
            vectors = embed([item.solution for item in missing])
            if len(vectors) != len(missing):
                raise ValueError("embed must return one vector per solution")
            for item, vector in zip(missing, vectors):
                self._embeddings[solution_key(item.solution)] = as_embedding(vector)
        matrix = np.asarray(
            [self._embeddings[solution_key(item.solution)] for item in history],
            dtype=np.float64,
        )
        if matrix.ndim != 2 or matrix.shape[0] != len(history):
            raise ValueError("cached embeddings must form an [N, D] matrix")
        return matrix
