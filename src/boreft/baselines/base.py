"""Shared contracts and artifacts for non-BOReFT search baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import time
from typing import (
    Callable,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TypeAlias,
    runtime_checkable,
)


BaselineName = Literal[
    "random_sampling",
    "discrete_bo",
    "opro",
    "sdpo_ttt",
    "bopro",
    "migrate",
    "autodiscovery",
]
ObservationPhase = Literal["warmstart", "search"]
JSONScalar = float | int | bool | str | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True)
class Candidate:
    """One solution proposed by a baseline before black-box evaluation."""

    solution: str
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.solution.strip():
            raise ValueError("candidate solution must not be blank")


@dataclass(frozen=True)
class ScoreResult:
    """Task-agnostic result returned by a black-box scorer."""

    score: float
    components: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")


@dataclass(frozen=True)
class GenerationOptions:
    """Decoding controls supplied with each baseline generation request."""

    temperature: float = 1.0
    max_new_tokens: int = 128
    top_p: float = field(default=1.0, init=False)

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be nonnegative")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")


Generate = Callable[[str, int, GenerationOptions], Sequence[str]]
Score = Callable[[str], ScoreResult]
Embed = Callable[[Sequence[str]], Sequence[Sequence[float]]]


def solution_key(solution: str) -> str:
    """Normalized identity used to detect a solution the search already tried."""
    return solution.strip().casefold()


def as_embedding(vector: Sequence[float]) -> list[float]:
    values = [float(component) for component in vector]
    if not values or not all(math.isfinite(component) for component in values):
        raise ValueError("embedding vectors must be finite and non-empty")
    return values


def timed_score(score: Score, solution: str) -> tuple[ScoreResult, float]:
    """Evaluate one black box and return its wall-clock duration."""
    started = time.monotonic()
    result = score(solution)
    return result, time.monotonic() - started


@dataclass(frozen=True)
class SearchContext:
    """Services and immutable task information supplied by the central runner."""

    task_description: str
    target: str
    objective: str
    seed: int
    generate: Generate
    score: Score
    embed: Embed | None = None
    generation_options: GenerationOptions = field(default_factory=GenerationOptions)
    observation_samples: int = 1
    checkpoint_dir: str | None = None
    task: str | None = None
    checkpoint: object | None = None
    model_name: str | None = None

    def __post_init__(self) -> None:
        if not self.task_description.strip():
            raise ValueError("task_description must not be blank")
        if not self.target.strip():
            raise ValueError("target must not be blank")
        if self.observation_samples < 1:
            raise ValueError("observation_samples must be positive")
        if self.checkpoint_dir is not None and not self.checkpoint_dir.strip():
            raise ValueError("checkpoint_dir must be a non-empty path")
        if self.task is not None and not self.task.strip():
            raise ValueError("task must be a non-empty name")
        if self.model_name is not None and not self.model_name.strip():
            raise ValueError("model_name must be a non-empty name")


@dataclass(frozen=True)
class BaselineObservation:
    """Serializable scored solution shared by every baseline."""

    index: int
    solution: str
    score: float
    score_std: float = 0.0
    score_sem: float = 0.0
    sample_count: int = 1
    sample_scores: list[float] = field(default_factory=list)
    solution_samples: list[str] = field(default_factory=list)
    components: dict[str, JSONValue] = field(default_factory=dict)
    candidate_metadata: dict[str, JSONValue] = field(default_factory=dict)
    phase: ObservationPhase = "search"
    seed: int = 0
    elapsed_seconds: float = 0.0
    bbox_eval_seconds: float = 0.0
    is_repeat_sample: bool = False
    is_repeat_proposal: bool = False
    best_so_far: float | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("observation index must be nonnegative")
        if not self.solution.strip():
            raise ValueError("observation solution must not be blank")
        if not math.isfinite(self.score):
            raise ValueError("observation score must be finite")
        if (
            not math.isfinite(self.score_std)
            or not math.isfinite(self.score_sem)
            or self.score_std < 0
            or self.score_sem < 0
        ):
            raise ValueError("observation uncertainty must be finite and nonnegative")
        if self.sample_count < 1:
            raise ValueError("sample_count must be positive")
        if self.sample_scores and (
            len(self.sample_scores) != self.sample_count
            or not all(math.isfinite(value) for value in self.sample_scores)
        ):
            raise ValueError("sample_scores must contain sample_count finite values")
        if self.solution_samples and len(self.solution_samples) != self.sample_count:
            raise ValueError("solution_samples must contain sample_count strings")
        if self.phase not in ("warmstart", "search"):
            raise ValueError(f"unknown observation phase: {self.phase!r}")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and nonnegative")
        if not math.isfinite(self.bbox_eval_seconds) or self.bbox_eval_seconds < 0:
            raise ValueError("bbox_eval_seconds must be finite and nonnegative")
        if self.best_so_far is not None and not math.isfinite(self.best_so_far):
            raise ValueError("best_so_far must be finite")

    def peak_score(self) -> float:
        """Best verified sample at this point; the logged score may be the mean."""
        if self.sample_scores:
            return max(self.sample_scores)
        return self.score

    def to_dict(self) -> dict:
        data = asdict(self)
        data["best_so_far"] = (
            self.score if self.best_so_far is None else self.best_so_far
        )
        return data

    @classmethod
    def from_dict(cls, data: Mapping) -> "BaselineObservation":
        return cls(
            index=int(data["index"]),
            solution=str(data["solution"]),
            score=float(data["score"]),
            score_std=float(data.get("score_std", 0.0)),
            score_sem=float(data.get("score_sem", 0.0)),
            sample_count=int(data.get("sample_count", 1)),
            sample_scores=[float(v) for v in data.get("sample_scores", [])],
            solution_samples=[str(v) for v in data.get("solution_samples", [])],
            components=dict(data.get("components", {})),
            candidate_metadata=dict(data.get("candidate_metadata", {})),
            phase=data.get("phase", "search"),
            seed=int(data.get("seed", 0)),
            elapsed_seconds=float(data.get("elapsed_seconds", 0.0)),
            bbox_eval_seconds=float(data.get("bbox_eval_seconds", 0.0)),
            is_repeat_sample=bool(data.get("is_repeat_sample", False)),
            is_repeat_proposal=bool(data.get("is_repeat_proposal", False)),
            best_so_far=float(data.get("best_so_far", data["score"])),
        )


def unique_observations(
    history: Sequence[BaselineObservation],
) -> list[BaselineObservation]:
    """Keep the latest observation for each normalized solution."""
    latest: dict[str, BaselineObservation] = {}
    for item in history:
        latest[solution_key(item.solution)] = item
    return list(latest.values())


class BaselineRunState:
    """Append-only baseline history using the main search artifact convention."""

    def __init__(
        self,
        path: str | Path,
        observations: Sequence[BaselineObservation] | None = None,
    ) -> None:
        self.path = Path(path)
        self.observations = list(observations or [])

    @classmethod
    def load(cls, path: str | Path) -> "BaselineRunState":
        target = Path(path)
        if not target.exists():
            return cls(target)
        observations: list[BaselineObservation] = []
        valid_lines: list[str] = []
        repaired_tail = False
        lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                valid_lines.append(line)
                continue
            try:
                observation = BaselineObservation.from_dict(json.loads(line))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                is_torn_tail = line_number == len(lines) and not line.endswith("\n")
                if is_torn_tail:
                    repaired_tail = True
                    break
                raise ValueError(
                    f"{target}:{line_number}: invalid baseline observation"
                ) from exc
            if observation.index != len(observations):
                raise ValueError(
                    f"{target}:{line_number}: expected index {len(observations)}, "
                    f"got {observation.index}"
                )
            observations.append(observation)
            if line_number == len(lines) and not line.endswith("\n"):
                valid_lines.append(line + "\n")
                repaired_tail = True
            else:
                valid_lines.append(line)
        if repaired_tail:
            temporary = target.with_suffix(target.suffix + ".repair")
            temporary.write_text("".join(valid_lines), encoding="utf-8")
            os.replace(temporary, target)
        return cls(target, observations)

    def append(self, observation: BaselineObservation) -> BaselineObservation:
        if observation.index != len(self.observations):
            raise ValueError(
                f"observation index must be {len(self.observations)}, "
                f"got {observation.index}"
            )
        previous = self.best_score
        stored = BaselineObservation(
            **{
                **asdict(observation),
                "best_so_far": (
                    observation.score
                    if previous is None
                    else max(previous, observation.score)
                ),
            }
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(stored.to_dict(), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.observations.append(stored)
        return stored

    def rewrite(self, observations: list[BaselineObservation]) -> None:
        """Atomically replace the JSONL (used when rescoring stored SMILES)."""
        rewritten = list(observations)
        if [item.index for item in rewritten] != list(range(len(rewritten))):
            raise ValueError("rewritten observations must have contiguous indices")
        if self.observations and len(rewritten) != len(self.observations):
            raise ValueError(
                "rewrite cannot change the number of observations "
                f"({len(self.observations)} -> {len(rewritten)})"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for item in rewritten:
                handle.write(json.dumps(item.to_dict(), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
        self.observations = rewritten

    @property
    def best_score(self) -> float | None:
        if not self.observations:
            return None
        return max(observation.score for observation in self.observations)

    def summary(self, *, elapsed_seconds: float | None = None) -> dict:
        if elapsed_seconds is not None and (
            not math.isfinite(elapsed_seconds) or elapsed_seconds < 0
        ):
            raise ValueError("elapsed_seconds must be finite and nonnegative")
        best = (
            max(self.observations, key=lambda observation: observation.score)
            if self.observations
            else None
        )
        return {
            "n_observations": len(self.observations),
            "n_warmstart": sum(
                observation.phase == "warmstart"
                for observation in self.observations
            ),
            "n_acquired": sum(
                observation.phase == "search" for observation in self.observations
            ),
            "n_verifications": sum(
                observation.sample_count for observation in self.observations
            ),
            "elapsed_seconds": sum(
                observation.elapsed_seconds for observation in self.observations
            )
            if elapsed_seconds is None
            else elapsed_seconds,
            "bbox_eval_seconds": sum(
                observation.bbox_eval_seconds for observation in self.observations
            ),
            "n_repeat_samples": sum(
                observation.phase == "search" and observation.is_repeat_sample
                for observation in self.observations
            ),
            "n_repeat_proposals": sum(
                observation.phase == "search" and observation.is_repeat_proposal
                for observation in self.observations
            ),
            "best_score": None if best is None else best.score,
            "best_solution": None if best is None else best.solution,
            "best_index": None if best is None else best.index,
        }


@dataclass(frozen=True)
class ArtifactLayout:
    """Canonical paths shared by baseline and main-search runs."""

    root: Path
    seed: int

    @property
    def seed_dir(self) -> Path:
        return self.root / f"seed_{self.seed}"

    @property
    def observations(self) -> Path:
        return self.seed_dir / "observations.jsonl"

    @property
    def summary(self) -> Path:
        return self.seed_dir / "summary.json"

    @property
    def plot(self) -> Path:
        return self.seed_dir / "best_so_far.png"

    @property
    def method_state(self) -> Path:
        return self.seed_dir / "method_state"


@runtime_checkable
class Baseline(Protocol):
    """Ask/observe interface implemented independently by each search method."""

    name: BaselineName

    def propose(
        self,
        context: SearchContext,
        history: Sequence[BaselineObservation],
        count: int,
    ) -> Sequence[Candidate]: ...

    def observe(self, observations: Sequence[BaselineObservation]) -> None: ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: Mapping) -> None: ...


class StatelessBaseline:
    """Defaults for methods whose only state is the shared observation history."""

    name: BaselineName

    def observe(self, observations: Sequence[BaselineObservation]) -> None:
        del observations

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: Mapping) -> None:
        if state:
            raise ValueError(f"{self.name} baseline does not persist method state")


class StubBaseline(StatelessBaseline):
    """Placeholder behavior while an algorithm implementation is pending."""

    def _not_implemented(self) -> NotImplementedError:
        return NotImplementedError(
            f"{self.name} baseline is scaffolding; implementation is pending"
        )
