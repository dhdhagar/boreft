"""Reusable sequential Bayesian-optimization loop."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import time
from typing import Callable, Literal, Optional, Protocol, Sequence

import numpy as np
import torch

from .acquisition import AcquisitionKind, LatentEllipsoid, propose_candidates
from .state import Observation, RunState
from .surrogate import KERNEL_CHOICES, KernelKind, SurrogateFitConfig, fit_surrogate

_RULE = "-" * 64


def _log(message: str = "") -> None:
    print(message, flush=True)


def _n_verifications(observations: Sequence[Observation]) -> int:
    return sum(item.sample_count for item in observations)


def _remaining_verifications(state: RunState, budget: int) -> int:
    return budget - _n_verifications(state.observations)


def _affordable_points(remaining: int, batch_size: int, samples: int) -> int:
    if samples < 1 or remaining < samples:
        return 0
    return min(batch_size, remaining // samples)


def _ensure_verification_budget(
    state: RunState, budget: int, cost: int, *, what: str
) -> None:
    if cost > _remaining_verifications(state, budget):
        raise ValueError(f"{what} verification cost cannot exceed budget")


def _labeled_decoded(record: dict) -> str | None:
    decoded = record.get("decoded")
    if decoded is None:
        return None
    text = str(decoded).strip()
    return text or None


def _warmstart_record_cost(record: dict, observation_samples: int) -> int:
    if "score" in record or _labeled_decoded(record) is not None:
        sample_count = int(record.get("sample_count", 1))
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        return sample_count
    return observation_samples


def _decoded_key(text: str) -> str:
    return text.strip().casefold()


def _observation_decoded_keys(observation: Observation) -> list[str]:
    texts = observation.decoded_samples or [observation.decoded]
    return [_decoded_key(text) for text in texts]


def _best_observation(state: RunState) -> Observation | None:
    if not state.observations:
        return None
    return max(state.observations, key=lambda item: item.peak_score())


def _is_latent_duplicate(
    point: np.ndarray,
    observed: Sequence[Sequence[float]],
    bounds: np.ndarray,
    tolerance: float,
) -> bool:
    if not len(observed):
        return False
    previous = np.asarray(observed, dtype=np.float64)
    candidate = np.asarray(point, dtype=np.float64)
    span = np.maximum(bounds[1] - bounds[0], np.finfo(np.float64).eps)
    distances = np.linalg.norm((previous - candidate) / span, axis=-1)
    return bool(np.any(distances <= tolerance))


def _format_observation(observation: Observation) -> str:
    return f"{observation.peak_score():.6g} ({observation.decoded!r})"


def _print_seed_banner(seed: int, config: BOConfig, resumed: bool) -> None:
    _log()
    _log(_RULE)
    suffix = "  (resume)" if resumed else ""
    _log(f"seed {seed}{suffix}")
    _log(
        f"  {config.surrogate} GP  ·  {config.acquisition}  ·  "
        f"kernel={config.kernel}  ·  "
        f"batch={config.batch_size}  ·  samples={config.observation_samples}"
        f"{'  ·  ARD' if config.use_ard else ''}"
        f"{f'  ·  layers={config.projection_layers}' if config.surrogate == 'projected' else ''}"
    )
    _log(_RULE)


def _print_warmstart_table(state: RunState, budget: int) -> None:
    warmstarts = [item for item in state.observations if item.source == "warmstart"]
    _log(f"Warmstart ({len(warmstarts)})")
    if not warmstarts:
        _log("  (none)")
        return
    width = max(len(str(budget)), 2)
    used = 0
    for item in warmstarts:
        used += item.sample_count
        _log(
            f"  {used:{width}d}/{budget}  "
            f"{item.peak_score():8.4f}  {item.decoded}"
        )
    best = _best_observation(state)
    if best is not None:
        _log(f"  best after warmstart: {_format_observation(best)}")
    _log()


def _print_acquisition_step(
    *,
    seed: int,
    observation: Observation,
    best: Observation,
    budget: int,
    used: int,
    repeat: bool,
    duplicate: bool,
) -> None:
    tags = "".join(
        flag
        for flag, enabled in (
            (" [repeat sample]", repeat),
            (" [repeat proposal]", duplicate),
        )
        if enabled
    )
    _log(
        f"  [seed={seed}; {used}/{budget}] "
        f"best={_format_observation(best)}, "
        f"decoded={_format_observation(observation)}"
        f"{tags}"
    )


def repeat_counts(
    state: RunState,
    bounds: np.ndarray,
    tolerance: float,
) -> tuple[int, int]:
    """Count acquisition steps tagged as a repeat sample or repeat proposal."""
    n_repeat_samples = 0
    n_repeat_proposals = 0
    seen_decoded: set[str] = set()
    seen_points: list[Sequence[float]] = []
    for item in state.observations:
        is_repeat_sample = _decoded_key(item.decoded) in seen_decoded
        is_repeat_proposal = _is_latent_duplicate(
            item.point, seen_points, bounds, tolerance
        )
        if item.source == "acquisition":
            n_repeat_samples += int(is_repeat_sample)
            n_repeat_proposals += int(is_repeat_proposal)
        seen_decoded.update(_observation_decoded_keys(item))
        seen_points.append(item.point)
    return n_repeat_samples, n_repeat_proposals


def recorded_repeat_counts(state: RunState) -> tuple[int, int]:
    """Repeat flags stored at observation time (stable after point remaps)."""
    n_repeat_samples = 0
    n_repeat_proposals = 0
    for item in state.observations:
        if item.source != "acquisition":
            continue
        n_repeat_samples += int(item.is_repeat_sample)
        n_repeat_proposals += int(item.is_repeat_proposal)
    return n_repeat_samples, n_repeat_proposals


_repeat_counts = repeat_counts


def _print_seed_summary(
    seed: int,
    state: RunState,
    config: BOConfig,
    bounds: np.ndarray,
    *,
    stopped_at_maximum: bool = False,
) -> None:
    del bounds  # historical repeat flags live on observations; remaps must not recount
    best = _best_observation(state)
    if best is None:
        _log(f"Done seed {seed}: no observations")
        return
    n_repeat_samples, n_repeat_proposals = recorded_repeat_counts(state)
    _log()
    extra = "  [known maximum]" if stopped_at_maximum else ""
    _log(
        f"Done seed {seed}: best={_format_observation(best)}  "
        f"({_n_verifications(state.observations)}/{config.budget} verifications){extra}"
    )
    _log(
        f"  repeat samples={n_repeat_samples}, "
        f"repeat proposals={n_repeat_proposals}"
    )


@dataclass(frozen=True)
class Verification:
    score: float
    components: dict[str, float | int | bool | str]


class Verifier(Protocol):
    maximum: float | None

    def __call__(self, decoded: str) -> Verification: ...


@dataclass
class BOConfig:
    """Programmatic controls independent of checkpoint loading and decoding."""

    budget: int = 60
    surrogate: Literal["static", "projected"] = "static"
    acquisition: AcquisitionKind = "log_ei"
    batch_size: int = 1
    observation_samples: int = 1
    kernel: KernelKind = "matern-2.5"
    use_ard: bool = False
    projection_dim: int = 64
    projection_layers: int = 1
    projection_steps: int = 300
    gp_lr: float = 0.2
    projection_lr: float = 0.002
    acquisition_restarts: int = 10
    acquisition_raw_samples: int = 256
    acquisition_mc_samples: int = 128
    ucb_beta: float = 0.2
    thompson_candidates: int = 4096
    duplicate_tolerance: float = 1e-6
    maximum_tolerance: float = 1e-6
    log_gp_surrogate: bool = False

    def validate(self) -> None:
        if self.observation_samples < 1:
            raise ValueError("observation_samples must be positive")
        if self.budget < 2 * self.observation_samples:
            raise ValueError("budget must cover at least two observations")
        if self.acquisition not in ("log_ei", "ucb", "thompson"):
            raise ValueError(f"unknown acquisition function: {self.acquisition!r}")
        if self.kernel not in KERNEL_CHOICES:
            raise ValueError(
                f"unknown GP kernel: {self.kernel!r} (expected one of {KERNEL_CHOICES})"
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.projection_dim < 1 or self.projection_steps < 1:
            raise ValueError("projection_dim and projection_steps must be positive")
        if self.projection_layers < 1:
            raise ValueError("projection_layers must be positive")
        if self.gp_lr <= 0 or self.projection_lr <= 0:
            raise ValueError("GP and projection learning rates must be positive")
        if self.acquisition_restarts < 1 or self.acquisition_raw_samples < 1:
            raise ValueError("acquisition restarts and raw samples must be positive")
        if self.acquisition_mc_samples < 1:
            raise ValueError("acquisition_mc_samples must be positive")
        if self.ucb_beta < 0:
            raise ValueError("ucb_beta must be nonnegative")
        if self.thompson_candidates < self.batch_size:
            raise ValueError("thompson_candidates must be at least batch_size")
        if self.duplicate_tolerance < 0 or self.maximum_tolerance < 0:
            raise ValueError("tolerances must be nonnegative")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _observation_found_target(
    observation: Observation, verifier: Verifier, tolerance: float
) -> bool:
    exact = observation.components.get("exact_match")
    if exact is True or exact == 1 or exact == 1.0:
        return True
    if isinstance(exact, (int, float)) and exact > 0:
        return True
    if verifier.maximum is None:
        return False
    return observation.peak_score() >= verifier.maximum - tolerance


def _reached_maximum(state: RunState, verifier: Verifier, tolerance: float) -> bool:
    return any(
        _observation_found_target(item, verifier, tolerance)
        for item in state.observations
    )


def _sample_hits_known_maximum(
    result: Verification, verifier: Verifier, tolerance: float
) -> bool:
    exact = result.components.get("exact_match")
    if exact is True or exact == 1 or exact == 1.0:
        return True
    return (
        verifier.maximum is not None
        and np.isfinite(verifier.maximum)
        and result.score >= verifier.maximum - tolerance
    )


def _validated_result(
    result: Verification,
    verifier: Verifier,
    tolerance: float,
) -> Verification:
    if not np.isfinite(result.score):
        raise ValueError(f"verifier returned a non-finite score: {result.score!r}")
    if (
        verifier.maximum is not None
        and result.score > verifier.maximum + tolerance
    ):
        raise ValueError(
            f"verifier score {result.score!r} exceeds maximum {verifier.maximum!r}"
        )
    return result


def _observe_point(
    point: np.ndarray,
    *,
    decode: Callable[[np.ndarray], str],
    verifier: Verifier,
    samples: int,
    maximum_tolerance: float,
) -> tuple[Verification, str, list[float], list[str], float, float, float]:
    decoded_samples: list[str] = []
    results: list[Verification] = []
    bbox_eval_seconds = 0.0
    for _ in range(samples):
        decoded = decode(point)
        bbox_started = time.monotonic()
        raw_result = verifier(decoded)
        bbox_eval_seconds += time.monotonic() - bbox_started
        result = _validated_result(
            raw_result,
            verifier,
            maximum_tolerance,
        )
        scored = result.components.get("decoded")
        if isinstance(scored, str) and scored.strip():
            decoded = scored
        decoded_samples.append(decoded)
        results.append(result)
        if _sample_hits_known_maximum(result, verifier, maximum_tolerance):
            break

    n = len(results)
    scores = np.asarray([result.score for result in results], dtype=np.float64)
    score_std = float(scores.std(ddof=1)) if n > 1 else 0.0
    score_sem = score_std / float(np.sqrt(n))
    components: dict[str, float | int | bool | str] = {}
    keys = set.intersection(*(set(result.components) for result in results))
    for key in keys:
        values = [result.components[key] for result in results]
        if n == 1:
            components[key] = values[0]
        elif all(isinstance(value, (int, float, bool)) for value in values):
            numeric = np.asarray(values, dtype=np.float64)
            components[key] = float(numeric.mean())
            components[f"{key}_std"] = float(numeric.std(ddof=1))
        elif all(value == values[0] for value in values):
            components[key] = values[0]
    representative = int(np.argmax(scores))
    return (
        Verification(score=float(scores.mean()), components=components),
        decoded_samples[representative],
        scores.tolist(),
        decoded_samples,
        score_std,
        score_sem,
        bbox_eval_seconds,
    )


def _save_surrogate(surrogate, state: RunState) -> None:
    if not hasattr(surrogate, "checkpoint"):
        return
    path = state.path.parent / f"surrogate_{len(state.observations):04d}.pt"
    temporary = Path(f"{path}.tmp")
    torch.save(surrogate.checkpoint(), temporary)
    os.replace(temporary, path)


def _pending_path(state: RunState) -> Path:
    return state.path.parent / "pending_batch.json"


def _write_pending_batch(
    state: RunState,
    *,
    points: np.ndarray,
    metadata: dict,
    batch_index: int,
    acquisition: AcquisitionKind,
) -> dict:
    payload = {
        "start_index": len(state.observations),
        "batch_index": batch_index,
        "acquisition": acquisition,
        "points": np.asarray(points, dtype=float).tolist(),
        "surrogate_metadata": metadata,
    }
    path = _pending_path(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
    return payload


def _load_pending_batch(
    state: RunState,
    budget: int,
    acquisition: AcquisitionKind,
    observation_samples: int,
) -> dict | None:
    path = _pending_path(state)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        start = int(payload["start_index"])
        points = payload["points"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid pending BO batch: {path}") from exc
    completed = len(state.observations) - start
    if start < 0 or completed < 0 or completed > len(points):
        raise ValueError(f"pending BO batch does not match run state: {path}")
    remaining_points = len(points) - completed
    if remaining_points < 1:
        path.unlink()
        return None
    if payload.get("acquisition") != acquisition:
        raise ValueError("acquisition function differs from the pending BO batch")
    if (
        _affordable_points(
            budget - _n_verifications(state.observations),
            remaining_points,
            observation_samples,
        )
        < 1
    ):
        path.unlink()
        return None
    return payload


def _clear_pending_batch(state: RunState) -> None:
    _pending_path(state).unlink(missing_ok=True)


def _apply_bounds_update(bound_array: np.ndarray, updated: np.ndarray) -> np.ndarray:
    new_bounds = np.asarray(updated, dtype=np.float32)
    if new_bounds.shape != bound_array.shape:
        raise ValueError(
            "on_after_batch bounds must match the current latent dimension "
            f"(got {new_bounds.shape}, expected {bound_array.shape})"
        )
    if not np.isfinite(new_bounds).all() or np.any(new_bounds[1] <= new_bounds[0]):
        raise ValueError("updated bounds must be finite and strictly increasing")
    return new_bounds


def run_bo(
    *,
    seed: int,
    bounds: np.ndarray,
    warmstart_points: Sequence[Sequence[float]],
    decode: Callable[[np.ndarray], str],
    verifier: Verifier,
    state: RunState,
    config: BOConfig,
    warmstart_records: Sequence[dict] | None = None,
    ellipsoid: LatentEllipsoid | None = None,
    on_observation: Callable[[Observation, RunState], None] | None = None,
    on_after_batch: Callable[[RunState], Optional[np.ndarray | tuple]] | None = None,
) -> RunState:
    """Run or resume one seed; budget counts verifier calls.

    ``on_observation`` fires after each appended observation (warmstart or
    acquisition). ``on_after_batch`` fires after each finished acquisition
    batch and may return replacement latent bounds (e.g. after subspace
    expansion remaps observation points). Pass a ``(bounds, ellipsoid)``
    tuple to update an ellipsoid domain as well.
    """
    config.validate()
    if verifier.maximum is not None and not np.isfinite(verifier.maximum):
        raise ValueError("verifier maximum must be finite or None")
    points = np.asarray(warmstart_points, dtype=np.float32)
    bound_array = np.asarray(bounds, dtype=np.float32)
    if points.ndim != 2 or len(points) < 2:
        raise ValueError("warmstart_points must contain at least two [D] points")
    if not np.isfinite(points).all():
        raise ValueError("warmstart points must be finite")
    # Warm starts are prior observations and may intentionally lie outside the
    # acquisition box; only newly proposed points are constrained to ``bounds``.
    if bound_array.shape != (2, points.shape[1]):
        raise ValueError("bounds and warmstart point dimensions do not match")
    if not np.isfinite(bound_array).all() or np.any(bound_array[1] <= bound_array[0]):
        raise ValueError("bounds must be finite and strictly increasing")
    if _n_verifications(state.observations) > config.budget:
        raise ValueError("resumed run already exceeds the configured budget")
    if any(len(observation.point) != points.shape[1] for observation in state.observations):
        raise ValueError("resumed observations have a different latent dimension")

    seed_everything(seed)
    resumed = bool(state.observations)
    _print_seed_banner(seed, config, resumed)
    if _reached_maximum(state, verifier, config.maximum_tolerance):
        _clear_pending_batch(state)
        _print_warmstart_table(state, config.budget)
        _print_seed_summary(
            seed, state, config, bound_array, stopped_at_maximum=True
        )
        return state

    records = list(warmstart_records or [{} for _ in points])
    current_ellipsoid = ellipsoid
    if len(records) != len(points):
        raise ValueError("warmstart_records must be parallel to warmstart_points")
    completed_warmstarts = sum(
        observation.source == "warmstart" for observation in state.observations
    )
    for index in range(min(completed_warmstarts, len(points))):
        if not np.allclose(
            state.observations[index].point,
            points[index],
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                "resumed warm starts do not match the configured source and seed"
            )
    if (
        any(observation.source == "acquisition" for observation in state.observations)
        and completed_warmstarts != len(points)
    ):
        raise ValueError("cannot change warmstart_count after acquisitions exist")
    remaining_records = records[completed_warmstarts:]
    _ensure_verification_budget(
        state,
        config.budget,
        sum(
            _warmstart_record_cost(record, config.observation_samples)
            for record in remaining_records
        ),
        what="warmstart",
    )

    for point, record in zip(
        points[completed_warmstarts:], records[completed_warmstarts:]
    ):
        sample_count = _warmstart_record_cost(record, config.observation_samples)
        _ensure_verification_budget(
            state, config.budget, sample_count, what="warmstart"
        )
        started = time.monotonic()
        labeled = _labeled_decoded(record)
        if "score" in record:
            decoded = str(record.get("decoded") or decode(point))
            result = Verification(
                score=float(record["score"]),
                components=dict(record.get("components", {})),
            )
            sample_scores = [
                float(value) for value in record.get("sample_scores", [])
            ]
            decoded_samples = [
                str(value) for value in record.get("decoded_samples", [])
            ]
            score_std = float(record.get("score_std", 0.0))
            score_sem = float(record.get("score_sem", 0.0))
            bbox_eval_seconds = float(record.get("bbox_eval_seconds", 0.0))
        else:
            observe_decode = (
                (lambda _point, text=labeled: text)
                if labeled is not None
                else decode
            )
            (
                result,
                decoded,
                sample_scores,
                decoded_samples,
                score_std,
                score_sem,
                bbox_eval_seconds,
            ) = _observe_point(
                point,
                decode=observe_decode,
                verifier=verifier,
                samples=sample_count,
                maximum_tolerance=config.maximum_tolerance,
            )
            result = Verification(
                score=result.score,
                components={
                    **result.components,
                    **dict(record.get("components", {})),
                },
            )
        result = _validated_result(result, verifier, config.maximum_tolerance)
        observed_count = len(sample_scores) if sample_scores else sample_count
        observation = state.append(
            Observation(
                index=len(state.observations),
                point=point.astype(float).tolist(),
                decoded=decoded,
                score=result.score,
                score_std=score_std,
                score_sem=score_sem,
                sample_count=observed_count,
                sample_scores=sample_scores,
                decoded_samples=decoded_samples,
                components=result.components,
                source="warmstart",
                seed=seed,
                elapsed_seconds=time.monotonic() - started,
                bbox_eval_seconds=bbox_eval_seconds,
            )
        )
        if on_observation is not None:
            on_observation(observation, state)
        if _reached_maximum(state, verifier, config.maximum_tolerance):
            _clear_pending_batch(state)
            _print_warmstart_table(state, config.budget)
            _print_seed_summary(
                seed, state, config, bound_array, stopped_at_maximum=True
            )
            return state

    _print_warmstart_table(state, config.budget)
    remaining_points = _affordable_points(
        _remaining_verifications(state, config.budget),
        config.budget,
        config.observation_samples,
    )
    if remaining_points > 0:
        _log(f"Acquisition ({remaining_points})")

    while True:
        remaining = _remaining_verifications(state, config.budget)
        if _affordable_points(remaining, 1, config.observation_samples) < 1:
            _clear_pending_batch(state)
            break
        pending = _load_pending_batch(
            state,
            config.budget,
            config.acquisition,
            config.observation_samples,
        )
        if pending is None:
            start_index = len(state.observations)
            fit_seed = seed * 1_000_003 + start_index
            seed_everything(fit_seed)
            surrogate = fit_surrogate(
                state.points,
                state.scores,
                bound_array,
                SurrogateFitConfig(
                    kind=config.surrogate,
                    kernel=config.kernel,
                    use_ard=config.use_ard,
                    projection_dim=config.projection_dim,
                    projection_layers=config.projection_layers,
                    steps=config.projection_steps,
                    gp_lr=config.gp_lr,
                    projection_lr=config.projection_lr,
                ),
                observation_variances=(
                    state.score_variances
                    if any(o.sample_count > 1 for o in state.observations)
                    else None
                ),
            )
            if config.log_gp_surrogate:
                _save_surrogate(surrogate, state)
            batch_size = _affordable_points(
                _remaining_verifications(state, config.budget),
                config.batch_size,
                config.observation_samples,
            )
            if batch_size < 1:
                break
            points = propose_candidates(
                surrogate,
                bound_array,
                state.points,
                acquisition=config.acquisition,
                batch_size=batch_size,
                seed=fit_seed,
                num_restarts=config.acquisition_restarts,
                raw_samples=config.acquisition_raw_samples,
                mc_samples=config.acquisition_mc_samples,
                ucb_beta=config.ucb_beta,
                thompson_candidates=config.thompson_candidates,
                duplicate_tolerance=config.duplicate_tolerance,
                observation_samples=config.observation_samples,
                ellipsoid=current_ellipsoid,
            )
            batch_index = (
                sum(o.source == "acquisition" for o in state.observations)
                // config.batch_size
            )
            pending = _write_pending_batch(
                state,
                points=points,
                metadata=surrogate.metadata(),
                batch_index=batch_index,
                acquisition=config.acquisition,
            )

        start_index = int(pending["start_index"])
        points = np.asarray(pending["points"], dtype=np.float32)
        metadata = dict(pending["surrogate_metadata"])
        batch_index = int(pending["batch_index"])
        completed = len(state.observations) - start_index
        for position, point in enumerate(points[completed:], start=completed):
            if (
                _affordable_points(
                    _remaining_verifications(state, config.budget),
                    1,
                    config.observation_samples,
                )
                < 1
            ):
                _clear_pending_batch(state)
                break
            started = time.monotonic()
            (
                result,
                decoded,
                sample_scores,
                decoded_samples,
                score_std,
                score_sem,
                bbox_eval_seconds,
            ) = _observe_point(
                point,
                decode=decode,
                verifier=verifier,
                samples=config.observation_samples,
                maximum_tolerance=config.maximum_tolerance,
            )
            seen = {
                key
                for item in state.observations
                for key in _observation_decoded_keys(item)
            }
            duplicate = _is_latent_duplicate(
                point,
                [item.point for item in state.observations],
                bound_array,
                config.duplicate_tolerance,
            )
            repeat = _decoded_key(decoded) in seen
            observation = state.append(
                Observation(
                    index=len(state.observations),
                    point=point.astype(float).tolist(),
                    decoded=decoded,
                    score=result.score,
                    score_std=score_std,
                    score_sem=score_sem,
                    sample_count=len(sample_scores) if sample_scores else 1,
                    sample_scores=sample_scores,
                    decoded_samples=decoded_samples,
                    components={
                        **result.components,
                        **{f"gp_{key}": value for key, value in metadata.items()},
                        "bo_acquisition": config.acquisition,
                        "bo_batch": batch_index,
                        "bo_batch_position": position,
                        "bo_batch_size": len(points),
                    },
                    source="acquisition",
                    seed=seed,
                    elapsed_seconds=time.monotonic() - started,
                    bbox_eval_seconds=bbox_eval_seconds,
                    is_repeat_sample=repeat,
                    is_repeat_proposal=duplicate,
                )
            )
            if on_observation is not None:
                on_observation(observation, state)
            best = _best_observation(state)
            assert best is not None
            _print_acquisition_step(
                seed=seed,
                observation=observation,
                best=best,
                used=_n_verifications(state.observations),
                budget=config.budget,
                repeat=repeat,
                duplicate=duplicate,
            )
            if _reached_maximum(state, verifier, config.maximum_tolerance):
                _clear_pending_batch(state)
                _print_seed_summary(
                    seed, state, config, bound_array, stopped_at_maximum=True
                )
                return state
        _clear_pending_batch(state)
        if on_after_batch is not None:
            updated_bounds = on_after_batch(state)
            if updated_bounds is not None:
                if isinstance(updated_bounds, tuple):
                    new_bounds, current_ellipsoid = updated_bounds
                    bound_array = _apply_bounds_update(bound_array, new_bounds)
                else:
                    bound_array = _apply_bounds_update(bound_array, updated_bounds)
    _print_seed_summary(seed, state, config, bound_array)
    return state
