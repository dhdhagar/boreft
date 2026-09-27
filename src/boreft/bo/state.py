"""Serializable observations and append-only BO run state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Literal

ObservationSource = Literal["warmstart", "acquisition"]


@dataclass(frozen=True)
class Observation:
    index: int
    point: list[float]
    decoded: str
    score: float
    score_std: float = 0.0
    score_sem: float = 0.0
    sample_count: int = 1
    sample_scores: list[float] = field(default_factory=list)
    decoded_samples: list[str] = field(default_factory=list)
    components: dict[str, float | int | bool | str] = field(default_factory=dict)
    source: ObservationSource = "acquisition"
    seed: int = 0
    elapsed_seconds: float = 0.0
    bbox_eval_seconds: float = 0.0
    is_repeat_sample: bool = False
    is_repeat_proposal: bool = False
    best_so_far: float | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("observation index must be nonnegative")
        if not self.point or not all(math.isfinite(value) for value in self.point):
            raise ValueError("observation point must be non-empty and finite")
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
        if self.decoded_samples and len(self.decoded_samples) != self.sample_count:
            raise ValueError("decoded_samples must contain sample_count strings")
        if self.source not in ("warmstart", "acquisition"):
            raise ValueError(f"unknown observation source: {self.source!r}")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and nonnegative")
        if not math.isfinite(self.bbox_eval_seconds) or self.bbox_eval_seconds < 0:
            raise ValueError("bbox_eval_seconds must be finite and nonnegative")
        if self.best_so_far is not None and not math.isfinite(self.best_so_far):
            raise ValueError("best_so_far must be finite")

    def peak_score(self) -> float:
        """Best verified sample at this point; GP still trains on ``score`` (mean)."""
        if self.sample_scores:
            return max(self.sample_scores)
        return self.score

    def to_dict(self) -> dict:
        data = asdict(self)
        data["best_so_far"] = (
            self.peak_score() if self.best_so_far is None else self.best_so_far
        )
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Observation":
        return cls(
            index=int(data["index"]),
            point=[float(v) for v in data["point"]],
            decoded=str(data.get("decoded", "")),
            score=float(data["score"]),
            score_std=float(data.get("score_std", 0.0)),
            score_sem=float(data.get("score_sem", 0.0)),
            sample_count=int(data.get("sample_count", 1)),
            sample_scores=[float(v) for v in data.get("sample_scores", [])],
            decoded_samples=[str(v) for v in data.get("decoded_samples", [])],
            components=dict(data.get("components", {})),
            source=data.get("source", "acquisition"),
            seed=int(data.get("seed", 0)),
            elapsed_seconds=float(data.get("elapsed_seconds", 0.0)),
            bbox_eval_seconds=float(data.get("bbox_eval_seconds", 0.0)),
            is_repeat_sample=bool(data.get("is_repeat_sample", False)),
            is_repeat_proposal=bool(data.get("is_repeat_proposal", False)),
            best_so_far=float(
                data["best_so_far"]
                if "best_so_far" in data
                else max(
                    [float(v) for v in data.get("sample_scores", [])]
                    or [data["score"]]
                )
            ),
        )


class RunState:
    """In-memory view backed by an append-only JSONL file."""

    def __init__(self, path: str | Path, observations: list[Observation] | None = None):
        self.path = Path(path)
        self.observations = list(observations or [])

    @classmethod
    def load(cls, path: str | Path) -> "RunState":
        target = Path(path)
        if not target.exists():
            return cls(target)
        observations: list[Observation] = []
        valid_lines: list[str] = []
        repaired_tail = False
        lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                valid_lines.append(line)
                continue
            try:
                observation = Observation.from_dict(json.loads(line))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                is_torn_tail = line_number == len(lines) and not line.endswith("\n")
                if is_torn_tail:
                    repaired_tail = True
                    break
                raise ValueError(
                    f"{target}:{line_number}: invalid BO observation"
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

    def append(self, observation: Observation) -> Observation:
        if observation.index != len(self.observations):
            raise ValueError(
                f"observation index must be {len(self.observations)}, "
                f"got {observation.index}"
            )
        previous = self.best_score
        peak = observation.peak_score()
        best = peak if previous is None else max(previous, peak)
        stored = Observation(
            **{
                **asdict(observation),
                "best_so_far": best,
            }
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(stored.to_dict(), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.observations.append(stored)
        return stored

    def rewrite(self, observations: list[Observation]) -> None:
        """Atomically replace the JSONL (used when expansion remaps points)."""
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
        return max(observation.peak_score() for observation in self.observations)

    @property
    def points(self) -> list[list[float]]:
        return [observation.point for observation in self.observations]

    @property
    def scores(self) -> list[float]:
        return [observation.score for observation in self.observations]

    @property
    def score_variances(self) -> list[float]:
        return [observation.score_sem**2 for observation in self.observations]

    def summary(self) -> dict:
        best = None
        if self.observations:
            best = max(self.observations, key=lambda observation: observation.peak_score())
        return {
            "n_observations": len(self.observations),
            "n_warmstart": sum(o.source == "warmstart" for o in self.observations),
            "n_acquired": sum(o.source == "acquisition" for o in self.observations),
            "n_verifications": sum(o.sample_count for o in self.observations),
            "bbox_eval_seconds": sum(o.bbox_eval_seconds for o in self.observations),
            "n_repeat_samples": sum(o.is_repeat_sample for o in self.observations),
            "n_repeat_proposals": sum(
                o.is_repeat_proposal for o in self.observations
            ),
            "elapsed_seconds": sum(o.elapsed_seconds for o in self.observations),
            "best_score": None if best is None else best.peak_score(),
            "best_decoded": None if best is None else best.decoded,
            "best_index": None if best is None else best.index,
        }
