"""Shared ask/observe loop executed identically by every baseline method.

The loop is model-agnostic: generation, scoring, and warmstart decoding arrive
as callables so a baseline can be exercised without loading a checkpoint. The
budget counts verifier calls, matching ``boreft.search``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import time
from typing import Callable, Sequence

from .base import (
    Baseline,
    BaselineObservation,
    BaselineRunState,
    Candidate,
    JSONValue,
    ObservationPhase,
    Score,
    SearchContext,
    solution_key,
    timed_score,
)

_RULE = "-" * 64
_STATE_FILE = "state.json"


def _log(message: str = "") -> None:
    print(message, flush=True)


@dataclass(frozen=True)
class WarmstartSeed:
    """One shared warm start, decoded lazily so resume never repeats work."""

    decode: Callable[[], str]
    components: dict[str, JSONValue] = field(default_factory=dict)
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    score: float | None = None
    solution: str | None = None

    def __post_init__(self) -> None:
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("warmstart score must be finite")
        if self.solution is not None and not self.solution.strip():
            raise ValueError("warmstart solution must not be blank")


@dataclass
class BaselineLoopConfig:
    """Budget and observation controls shared by all baselines."""

    budget: int = 60
    batch_size: int = 1
    observation_samples: int = 1
    maximum: float | None = None
    maximum_tolerance: float = 1e-6

    def validate(self) -> None:
        if self.observation_samples < 1:
            raise ValueError("observation_samples must be positive")
        if self.budget < 1:
            raise ValueError("budget must be positive")
        if self.budget < self.observation_samples:
            raise ValueError("budget must cover at least one observation")
        if self.batch_size < 1 or self.batch_size > self.budget:
            raise ValueError("batch_size must be in [1, budget]")
        if self.maximum is not None and not math.isfinite(self.maximum):
            raise ValueError("maximum must be finite or None")
        if self.maximum_tolerance < 0:
            raise ValueError("maximum_tolerance must be nonnegative")


def _n_verifications(observations: Sequence[BaselineObservation]) -> int:
    return sum(item.sample_count for item in observations)


def _remaining_verifications(state: BaselineRunState, budget: int) -> int:
    return budget - _n_verifications(state.observations)


def _affordable_points(remaining: int, batch_size: int, samples: int) -> int:
    if samples < 1 or remaining < samples:
        return 0
    return min(batch_size, remaining // samples)


def _propose_batch_size(baseline: Baseline, config: BaselineLoopConfig) -> int:
    """How many candidates to request from ``propose`` this step.

    Random sampling packs ``candidates_per_call`` solutions into one completion.
    Scoring only ``batch_size`` (default 1) of them would discard the rest, so
    the propose count is at least that packed size. Discrete BO uses the same
    knob only while building an LLM pool, not at propose time. MiGrATe scores
    the new on-policy plus neighborhood samples each step (greedy members are
    reused from history), so     the propose count is that mixed-group size rather
    than ``batch_size``. AutoDiscovery expands one MCTS node per step: it
    samples ``k_experiments`` OPRO-style one-solution completions when that
    node has no untried leftovers (default 8), then scores one of them.
    """
    batch = config.batch_size
    name = getattr(baseline, "name", None)
    cfg = getattr(baseline, "config", None)
    if name == "random_sampling":
        per_call = getattr(cfg, "candidates_per_call", 1)
        if not isinstance(per_call, int) or per_call < 1:
            return batch
        return max(batch, per_call)
    if name == "migrate":
        new = getattr(cfg, "new_sample_count", 0)
        if isinstance(new, int) and new >= 1:
            return new
        return batch
    if name == "autodiscovery":
        return 1
    return batch


def _ensure_verification_budget(
    state: BaselineRunState, budget: int, cost: int, *, what: str
) -> None:
    if cost > _remaining_verifications(state, budget):
        raise ValueError(f"{what} verification cost cannot exceed budget")


def _warmstart_seed_cost(warmstart: WarmstartSeed, observation_samples: int) -> int:
    if warmstart.score is not None or warmstart.solution is not None:
        return 1
    return observation_samples


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def _aggregate_components(
    results: Sequence[tuple[float, dict[str, JSONValue]]],
) -> dict[str, JSONValue]:
    """Average numeric components and keep components shared by every sample."""
    if len(results) == 1:
        return dict(results[0][1])
    components: dict[str, JSONValue] = {}
    shared = set.intersection(*(set(item[1]) for item in results))
    for key in shared:
        values = [item[1][key] for item in results]
        if all(isinstance(value, (int, float, bool)) for value in values):
            mean, std = _mean_std([float(value) for value in values])
            components[key] = mean
            components[f"{key}_std"] = std
        elif all(value == values[0] for value in values):
            components[key] = values[0]
    return components


@dataclass(frozen=True)
class _Evaluation:
    score: float
    score_std: float
    score_sem: float
    sample_scores: list[float]
    components: dict[str, JSONValue]
    bbox_eval_seconds: float


def _evaluate(score: Score, solution: str, samples: int) -> _Evaluation:
    """Score one solution ``samples`` times to expose verifier stochasticity."""
    results: list[tuple[float, dict[str, JSONValue]]] = []
    bbox_eval_seconds = 0.0
    for _ in range(samples):
        result, elapsed = timed_score(score, solution)
        bbox_eval_seconds += elapsed
        results.append((result.score, dict(result.components)))
    scores = [item[0] for item in results]
    mean, std = _mean_std(scores)
    return _Evaluation(
        score=mean,
        score_std=std,
        score_sem=std / math.sqrt(samples),
        sample_scores=scores,
        components=_aggregate_components(results),
        bbox_eval_seconds=bbox_eval_seconds,
    )


def _above_maximum(score: float, config: BaselineLoopConfig) -> bool:
    return (
        config.maximum is not None
        and score > config.maximum + config.maximum_tolerance
    )


def method_state_path(directory: str | Path) -> Path:
    return Path(directory) / _STATE_FILE


def save_method_state(baseline: Baseline, directory: str | Path) -> None:
    """Persist JSON-serializable method state next to the seed's observations."""
    path = method_state_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(baseline.state_dict(), sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def load_method_state(directory: str | Path) -> dict:
    path = method_state_path(directory)
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid baseline method state: {path}") from exc
    if not isinstance(state, dict):
        raise ValueError(f"baseline method state must be an object: {path}")
    return state


def _reached_maximum(state: BaselineRunState, config: BaselineLoopConfig) -> bool:
    return (
        config.maximum is not None
        and state.best_score is not None
        and state.best_score >= config.maximum - config.maximum_tolerance
    )


def _best(state: BaselineRunState) -> BaselineObservation | None:
    if not state.observations:
        return None
    return max(state.observations, key=lambda item: item.score)


def _format(observation: BaselineObservation) -> str:
    return f"{observation.score:.6g} ({observation.solution!r})"


def _print_banner(
    seed: int,
    config: BaselineLoopConfig,
    name: str,
    resumed: bool,
    *,
    propose_batch: int | None = None,
) -> None:
    _log()
    _log(_RULE)
    _log(f"seed {seed}{'  (resume)' if resumed else ''}")
    extra = ""
    if propose_batch is not None and propose_batch != config.batch_size:
        extra = f"  ·  propose={propose_batch}"
    _log(
        f"  {name}  ·  batch={config.batch_size}{extra}  ·  "
        f"samples={config.observation_samples}"
    )
    _log(_RULE)


def _print_warmstart_table(
    state: BaselineRunState,
    budget: int,
    config: BaselineLoopConfig,
) -> None:
    warmstarts = [item for item in state.observations if item.phase == "warmstart"]
    _log(f"Warmstart ({len(warmstarts)})")
    if not warmstarts:
        _log("  (none)")
        return
    width = max(len(str(budget)), 2)
    used = 0
    for item in warmstarts:
        used += item.sample_count
        above = (
            "  [above known maximum]"
            if _above_maximum(item.score, config)
            else ""
        )
        _log(
            f"  {used:{width}d}/{budget}  "
            f"{item.score:8.4f}  {item.solution}{above}"
        )
    best = _best(state)
    if best is not None:
        _log(f"  best after warmstart: {_format(best)}")
    _log()


def _print_step(
    *,
    seed: int,
    observation: BaselineObservation,
    best: BaselineObservation,
    budget: int,
    used: int,
    config: BaselineLoopConfig,
) -> None:
    tags = "".join(
        flag
        for flag, enabled in (
            (" [repeat sample]", observation.is_repeat_sample),
            (" [repeat proposal]", observation.is_repeat_proposal),
            (" [above known maximum]", _above_maximum(observation.score, config)),
        )
        if enabled
    )
    _log(
        f"  [seed={seed}; {used}/{budget}] "
        f"best={_format(best)}, solution={_format(observation)}{tags}"
    )


def _print_summary(
    seed: int,
    state: BaselineRunState,
    config: BaselineLoopConfig,
    *,
    stopped_at_maximum: bool = False,
) -> None:
    best = _best(state)
    if best is None:
        _log(f"Done seed {seed}: no observations")
        return
    summary = state.summary()
    _log()
    if stopped_at_maximum and _above_maximum(best.score, config):
        extra = "  [above known maximum]"
    elif stopped_at_maximum:
        extra = "  [known maximum]"
    else:
        extra = ""
    _log(
        f"Done seed {seed}: best={_format(best)}  "
        f"({summary['n_verifications']}/{config.budget} verifications){extra}"
    )
    _log(
        f"  repeat samples={summary['n_repeat_samples']}, "
        f"repeat proposals={summary['n_repeat_proposals']}"
    )


def _validated_candidates(
    candidates: Sequence[Candidate],
    count: int,
    name: str,
) -> Sequence[Candidate]:
    if not candidates:
        raise ValueError(f"{name} proposed no candidates for {count} open slot(s)")
    if len(candidates) > count:
        raise ValueError(
            f"{name} proposed {len(candidates)} candidates for {count} open slot(s)"
        )
    return candidates


def _as_floats(values: object) -> list[float] | None:
    if not isinstance(values, list) or not values:
        return None
    try:
        return [float(value) for value in values]
    except (TypeError, ValueError):
        return None


def _points_close(left: object, right: object) -> bool:
    first, second = _as_floats(left), _as_floats(right)
    return bool(
        first
        and second
        and len(first) == len(second)
        and all(abs(a - b) <= 1e-6 for a, b in zip(first, second))
    )


def _validate_resumed_warmstarts(
    state: BaselineRunState,
    warmstarts: Sequence[WarmstartSeed],
) -> int:
    """Return how many configured warm starts are already persisted."""
    prefix = 0
    for item in state.observations:
        if item.phase != "warmstart":
            break
        prefix += 1
    stored = [item for item in state.observations if item.phase == "warmstart"]
    if len(stored) != prefix:
        raise ValueError("warmstart observations must form a prefix of the run")
    searched = any(item.phase == "search" for item in state.observations)
    if searched and prefix != len(warmstarts):
        raise ValueError("cannot change warmstart_count after searching begins")
    if prefix > len(warmstarts):
        raise ValueError(
            "resumed warm starts do not match the configured source and seed"
        )
    for observation, warmstart in zip(stored, warmstarts):
        expected = warmstart.metadata.get("warmstart_point")
        actual = observation.candidate_metadata.get("warmstart_point")
        if (
            expected is not None
            and actual is not None
            and not _points_close(expected, actual)
        ) or (
            warmstart.solution is not None
            and solution_key(observation.solution)
            != solution_key(warmstart.solution)
        ):
            raise ValueError(
                "resumed warm starts do not match the configured source and seed"
            )
    return prefix


def _scored_solution(candidate: Candidate, evaluation: _Evaluation) -> str:
    scored = evaluation.components.get("decoded")
    if isinstance(scored, str) and scored.strip():
        return scored
    return candidate.solution


def _observation(
    *,
    index: int,
    candidate: Candidate,
    evaluation: _Evaluation,
    phase: ObservationPhase,
    seed: int,
    elapsed_seconds: float,
    is_repeat_sample: bool,
    extra_components: dict[str, JSONValue] | None = None,
) -> BaselineObservation:
    return BaselineObservation(
        index=index,
        solution=_scored_solution(candidate, evaluation),
        score=evaluation.score,
        score_std=evaluation.score_std,
        score_sem=evaluation.score_sem,
        sample_count=len(evaluation.sample_scores),
        sample_scores=evaluation.sample_scores,
        components={**evaluation.components, **(extra_components or {})},
        candidate_metadata=dict(candidate.metadata),
        phase=phase,
        seed=seed,
        elapsed_seconds=elapsed_seconds,
        bbox_eval_seconds=evaluation.bbox_eval_seconds,
        is_repeat_sample=is_repeat_sample,
        is_repeat_proposal=bool(candidate.metadata.get("is_repeat_proposal", False)),
    )


def _append_observation(
    *,
    state: BaselineRunState,
    candidate: Candidate,
    evaluation: _Evaluation,
    phase: ObservationPhase,
    seed: int,
    started: float,
    extra_components: dict[str, JSONValue] | None = None,
) -> BaselineObservation:
    seen = {solution_key(item.solution) for item in state.observations}
    return state.append(
        _observation(
            index=len(state.observations),
            candidate=candidate,
            evaluation=evaluation,
            phase=phase,
            seed=seed,
            elapsed_seconds=time.monotonic() - started,
            is_repeat_sample=solution_key(candidate.solution) in seen,
            extra_components=extra_components,
        )
    )


def _run_warmstarts(
    *,
    seed: int,
    state: BaselineRunState,
    config: BaselineLoopConfig,
    context: SearchContext,
    warmstarts: Sequence[WarmstartSeed],
) -> bool:
    """Materialize any warm starts this seed has not observed yet."""
    completed = _validate_resumed_warmstarts(state, warmstarts)
    remaining = warmstarts[completed:]
    _ensure_verification_budget(
        state,
        config.budget,
        sum(
            _warmstart_seed_cost(warmstart, config.observation_samples)
            for warmstart in remaining
        ),
        what="warmstart",
    )
    for warmstart in remaining:
        _ensure_verification_budget(
            state,
            config.budget,
            _warmstart_seed_cost(warmstart, config.observation_samples),
            what="warmstart",
        )
        started = time.monotonic()
        solution = (
            warmstart.solution
            if warmstart.solution is not None
            else warmstart.decode()
        )
        candidate = Candidate(solution=solution, metadata=dict(warmstart.metadata))
        if warmstart.score is None:
            samples = (
                1
                if warmstart.solution is not None
                else config.observation_samples
            )
            evaluation = _evaluate(
                context.score, candidate.solution, samples
            )
        else:
            evaluation = _Evaluation(
                score=warmstart.score,
                score_std=0.0,
                score_sem=0.0,
                sample_scores=[warmstart.score],
                components={},
                bbox_eval_seconds=0.0,
            )
        _append_observation(
            state=state,
            candidate=candidate,
            evaluation=evaluation,
            phase="warmstart",
            seed=seed,
            started=started,
            extra_components=dict(warmstart.components),
        )
        if _reached_maximum(state, config):
            return True
    return False


def run_baseline(
    *,
    seed: int,
    baseline: Baseline,
    context: SearchContext,
    state: BaselineRunState,
    config: BaselineLoopConfig,
    warmstarts: Sequence[WarmstartSeed],
    method_state_dir: str | Path,
) -> BaselineRunState:
    """Run or resume one seed until the budget is spent or the maximum is hit."""
    config.validate()
    if _n_verifications(state.observations) > config.budget:
        raise ValueError("resumed run already exceeds the configured budget")

    resumed = bool(state.observations)
    if resumed:
        baseline.load_state_dict(load_method_state(method_state_dir))
    propose_batch = _propose_batch_size(baseline, config)
    _print_banner(
        seed, config, baseline.name, resumed, propose_batch=propose_batch
    )

    if _reached_maximum(state, config):
        _print_warmstart_table(state, config.budget, config)
        _print_summary(seed, state, config, stopped_at_maximum=True)
        return state

    if _run_warmstarts(
        seed=seed,
        state=state,
        config=config,
        context=context,
        warmstarts=warmstarts,
    ):
        save_method_state(baseline, method_state_dir)
        _print_warmstart_table(state, config.budget, config)
        _print_summary(seed, state, config, stopped_at_maximum=True)
        return state

    _print_warmstart_table(state, config.budget, config)
    remaining_points = _affordable_points(
        _remaining_verifications(state, config.budget),
        config.budget,
        config.observation_samples,
    )
    if remaining_points > 0:
        _log(f"Search ({remaining_points})")

    while True:
        remaining = _remaining_verifications(state, config.budget)
        count = _affordable_points(
            remaining, propose_batch, config.observation_samples
        )
        if count < 1:
            break
        candidates = _validated_candidates(
            baseline.propose(context, tuple(state.observations), count),
            count,
            baseline.name,
        )
        observed: list[BaselineObservation] = []
        stopped = False
        for candidate in candidates:
            if (
                _affordable_points(
                    _remaining_verifications(state, config.budget),
                    1,
                    config.observation_samples,
                )
                < 1
            ):
                break
            started = time.monotonic()
            observation = _append_observation(
                state=state,
                candidate=candidate,
                evaluation=_evaluate(
                    context.score,
                    candidate.solution,
                    config.observation_samples,
                ),
                phase="search",
                seed=seed,
                started=started,
            )
            observed.append(observation)
            best = _best(state)
            assert best is not None
            _print_step(
                seed=seed,
                observation=observation,
                best=best,
                used=_n_verifications(state.observations),
                budget=config.budget,
                config=config,
            )
            if _reached_maximum(state, config):
                stopped = True
                break
        baseline.observe(observed)
        save_method_state(baseline, method_state_dir)
        if stopped:
            _print_summary(seed, state, config, stopped_at_maximum=True)
            return state

    _print_summary(seed, state, config)
    return state
