"""Continuous acquisition optimization in the learned bias-vector box."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.exceptions import CandidateGenerationError
from botorch.generation.sampling import MaxPosteriorSampling
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler

from .surrogate import BiasGPSurrogate

AcquisitionKind = Literal["log_ei", "ucb", "thompson"]


def latent_bounds(
    bias_vectors,
    *,
    padding: float = 0.0,
    minimum_span: float = 1e-6,
    std=None,
    std_k: float = 0.0,
) -> np.ndarray:
    """Axis-aligned bounds from trained μ rows, optionally padded by span.

    With ``std_k == 0`` this is the coordinate-wise min/max of the means.
    With ``std_k > 0`` the box expands by the learned posterior std:

        B_k[j] = [min_i (μ_ij − k σ_ij), max_i (μ_ij + k σ_ij)]

    ``padding`` then expands that box by a fraction of its span.
    """
    points = np.asarray(bias_vectors, dtype=np.float64)
    if points.ndim != 2 or not len(points) or points.shape[1] == 0:
        raise ValueError("bias_vectors must be a non-empty [N,D] array")
    if not np.isfinite(points).all():
        raise ValueError("bias_vectors must be finite")
    if padding < 0 or minimum_span <= 0:
        raise ValueError("padding must be nonnegative and minimum_span positive")
    if not np.isfinite(std_k) or std_k < 0:
        raise ValueError("std_k must be finite and nonnegative")
    if std_k > 0:
        if std is None:
            raise ValueError("std is required when std_k > 0")
        sigma = np.asarray(std, dtype=np.float64)
        if sigma.shape != points.shape:
            raise ValueError("std must match bias_vectors shape")
        if not np.isfinite(sigma).all() or np.any(sigma < 0):
            raise ValueError("std must be finite and nonnegative")
        lo = (points - std_k * sigma).min(axis=0)
        hi = (points + std_k * sigma).max(axis=0)
    else:
        lo = points.min(axis=0)
        hi = points.max(axis=0)
    span = hi - lo
    floor = np.maximum(minimum_span - span, 0.0) / 2.0
    lo = lo - padding * np.maximum(span, minimum_span) - floor
    hi = hi + padding * np.maximum(span, minimum_span) + floor
    return np.stack([lo, hi])


def union_latent_bounds(*boxes) -> np.ndarray:
    """Coordinate-wise union of ``[2, D]`` axis-aligned boxes."""
    if not boxes:
        raise ValueError("at least one box is required")
    arrays = [np.asarray(box, dtype=np.float64) for box in boxes]
    shape = arrays[0].shape
    if shape[0] != 2 or arrays[0].ndim != 2 or shape[1] == 0:
        raise ValueError("bounds must be a non-empty [2, D] array")
    for array in arrays:
        if array.shape != shape:
            raise ValueError("bounds to union must share shape [2, D]")
        if not np.isfinite(array).all() or np.any(array[1] <= array[0]):
            raise ValueError("bounds must be finite and strictly increasing")
    lo = np.min([array[0] for array in arrays], axis=0)
    hi = np.max([array[1] for array in arrays], axis=0)
    return np.stack([lo, hi])


def unit_ball_log_volume(dim: int) -> float:
    """Log Lebesgue volume of the Euclidean unit ball in ``dim`` dimensions."""
    if dim < 1:
        raise ValueError("dim must be positive")
    return 0.5 * dim * math.log(math.pi) - math.lgamma(0.5 * dim + 1.0)


def box_log_volume(bounds) -> float:
    """Log Lebesgue volume of an axis-aligned box given as ``[2, D]`` (lo, hi)."""
    array = np.asarray(bounds, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != 2 or array.shape[1] == 0:
        raise ValueError("bounds must be a non-empty [2, D] array")
    sides = array[1] - array[0]
    if not np.isfinite(sides).all() or np.any(sides <= 0):
        raise ValueError("box sides must be positive and finite")
    return float(np.log(sides).sum())


def _cholesky_psd(matrix: np.ndarray, jitter: float) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    try:
        return np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError:
        dim = symmetric.shape[0]
        return np.linalg.cholesky(symmetric + jitter * np.eye(dim))


@dataclass(frozen=True)
class LatentEllipsoid:
    """Mahalanobis ball ``{c + L u : ||u||_2 ≤ 1}``.

    ``chol`` is the lower-triangular factor ``L`` in ``x = c + L u``.
    BoTorch still optimizes inside the axis-aligned bounding box of this
    ellipsoid; a quadratic inequality ``1 − ||L^{-1}(x−c)||^2 ≥ 0`` keeps
    proposals inside the ball (BoTorch 0.16 ``optimize_acqf`` / SLSQP).
    """

    center: np.ndarray
    chol: np.ndarray

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64).reshape(-1)
        chol = np.asarray(self.chol, dtype=np.float64)
        if center.ndim != 1 or center.size == 0:
            raise ValueError("center must be a non-empty vector")
        if chol.shape != (center.size, center.size):
            raise ValueError("chol must be square with center.size columns")
        if not np.isfinite(center).all() or not np.isfinite(chol).all():
            raise ValueError("ellipsoid parameters must be finite")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "chol", chol)

    @property
    def dim(self) -> int:
        return int(self.center.size)

    def aabb(self) -> np.ndarray:
        """Tight axis-aligned box containing the ellipsoid."""
        radii = np.linalg.norm(self.chol, axis=1)
        return np.stack([self.center - radii, self.center + radii])

    def log_volume(self) -> float:
        """Log Lebesgue volume of ``{c + L u : ||u||_2 ≤ 1}``."""
        sign, logdet = np.linalg.slogdet(self.chol)
        if sign <= 0 or not np.isfinite(logdet):
            raise ValueError("ellipsoid factor must have positive finite determinant")
        return unit_ball_log_volume(self.dim) + float(logdet)

    def mahalanobis(self, points) -> np.ndarray:
        array = np.asarray(points, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[-1] != self.dim:
            raise ValueError("points must have the ellipsoid dimension")
        whitened = np.linalg.solve(self.chol, (array - self.center).T).T
        return np.linalg.norm(whitened, axis=-1)

    def contains(self, points, *, atol: float = 1e-6) -> np.ndarray:
        return self.mahalanobis(points) <= 1.0 + atol

    def project(self, points, *, atol: float = 1e-12) -> np.ndarray:
        """Radial projection onto the ellipsoid (identity inside)."""
        array = np.asarray(points, dtype=np.float64)
        squeeze = array.ndim == 1
        if squeeze:
            array = array.reshape(1, -1)
        radius = np.maximum(self.mahalanobis(array), atol)
        scale = np.minimum(1.0 / radius, 1.0)
        projected = self.center + (array - self.center) * scale[:, None]
        return projected[0] if squeeze else projected

    def sample(self, n: int, *, seed: int) -> np.ndarray:
        """Draw ``n`` points uniformly in the ellipsoid (Gaussian direction, ``U^{1/d}``)."""
        if n < 1:
            raise ValueError("n must be positive")
        rng = np.random.default_rng(seed)
        direction = rng.standard_normal((n, self.dim))
        norms = np.linalg.norm(direction, axis=1, keepdims=True)
        direction = direction / np.clip(norms, 1e-12, None)
        radius = rng.random((n, 1)) ** (1.0 / self.dim)
        points = self.center + (direction * radius) @ self.chol.T
        return self.project(points)


def latent_ellipsoid(
    bias_vectors,
    *,
    padding: float = 0.0,
    minimum_span: float = 1e-6,
    std=None,
    std_k: float = 0.0,
    ridge: float = 1e-6,
) -> LatentEllipsoid:
    """Enclosing Mahalanobis ellipsoid of the training posterior means.

    This is the covering ellipsoid of the sample covariance, not a
    minimum-volume enclosing ellipsoid. Points are ``x = c + L u`` with
    ``||u||_2 ≤ 1`` and ``L = chol(Σ)`` scaled so every generator has
    Mahalanobis radius ≤ 1: the means, and when ``std_k > 0`` the per-axis
    vertices ``μ_i ± k σ_{ij} e_j`` (the ℓ₂ analog of the AABB expansion,
    not the ℓ∞ box corners).

    ``padding`` then multiplies the covering radius by ``(1 + 2 padding)``,
    matching AABB's per-side expansion of ``padding * span``. Collapsed
    axes are floored by a Loewner update
    ``Σ ← Σ + diag(max(0, (s/2)² − Σ_ii))`` before that padding, so the
    set remains a superset of the unfloored ellipsoid.
    """
    points = np.asarray(bias_vectors, dtype=np.float64)
    if points.ndim != 2 or not len(points) or points.shape[1] == 0:
        raise ValueError("bias_vectors must be a non-empty [N,D] array")
    if not np.isfinite(points).all():
        raise ValueError("bias_vectors must be finite")
    if padding < 0 or minimum_span <= 0 or ridge <= 0:
        raise ValueError("padding must be nonnegative; minimum_span and ridge positive")
    if not np.isfinite(std_k) or std_k < 0:
        raise ValueError("std_k must be finite and nonnegative")
    sigma = None
    if std_k > 0:
        if std is None:
            raise ValueError("std is required when std_k > 0")
        sigma = np.asarray(std, dtype=np.float64)
        if sigma.shape != points.shape:
            raise ValueError("std must match bias_vectors shape")
        if not np.isfinite(sigma).all() or np.any(sigma < 0):
            raise ValueError("std must be finite and nonnegative")

    n_points, dim = points.shape
    center = points.mean(axis=0)
    centered = points - center
    denom = max(n_points - 1, 1)
    cov = (centered.T @ centered) / denom
    jitter = ridge * max(float(np.trace(cov) / dim), 1.0)
    chol = _cholesky_psd(cov + jitter * np.eye(dim), jitter)

    def _max_radius_sq(chol_l: np.ndarray) -> float:
        whitened = np.linalg.solve(chol_l, centered.T).T
        radius_sq = float((whitened**2).sum(axis=1).max())
        if sigma is None:
            return radius_sq
        chol_inv = np.linalg.inv(chol_l)
        # u_i = L^{-1}(μ_i − c); vertex along axis j shifts u by ± a_ij L^{-1}[:, j]
        u_dot_col = whitened @ chol_inv
        col_norm_sq = (chol_inv**2).sum(axis=0)
        offset = std_k * sigma
        base = (whitened**2).sum(axis=1, keepdims=True)
        high = base + 2.0 * offset * u_dot_col + (offset**2) * col_norm_sq
        low = base - 2.0 * offset * u_dot_col + (offset**2) * col_norm_sq
        return float(max(radius_sq, high.max(), low.max()))

    radius_sq = max(_max_radius_sq(chol), 1e-12)
    chol = chol * np.sqrt(radius_sq)
    # Axis extent of {c + L u : ||u||≤1} is sqrt(cov_ii). Floor in the
    # Loewner order so previously covered generators stay covered.
    cov = chol @ chol.T
    half = 0.5 * minimum_span
    cov = cov + np.diag(np.maximum(half**2 - np.diag(cov), 0.0))
    chol = _cholesky_psd(cov, jitter)
    chol = chol * (1.0 + 2.0 * padding)
    return LatentEllipsoid(center=center, chol=chol)


def _project_torch(points: torch.Tensor, ellipsoid: LatentEllipsoid) -> torch.Tensor:
    center = torch.as_tensor(
        ellipsoid.center, device=points.device, dtype=points.dtype
    )
    chol = torch.as_tensor(ellipsoid.chol, device=points.device, dtype=points.dtype)
    original = points.shape
    flat = (points - center).reshape(-1, original[-1])
    whitened = torch.linalg.solve_triangular(
        chol, flat.transpose(0, 1), upper=False
    ).transpose(0, 1)
    radius = torch.linalg.vector_norm(whitened, dim=-1).clamp_min(
        torch.finfo(points.dtype).eps
    )
    scale = torch.clamp(1.0 / radius, max=1.0)
    return center + (flat * scale.unsqueeze(-1)).reshape(original)


def _ellipsoid_constraint(ellipsoid: LatentEllipsoid):
    """Intra-point BoTorch constraint ``1 − ||L^{-1}(x−c)||^2 ≥ 0``."""

    def _fn(x: torch.Tensor) -> torch.Tensor:
        center = torch.as_tensor(ellipsoid.center, device=x.device, dtype=x.dtype)
        chol = torch.as_tensor(ellipsoid.chol, device=x.device, dtype=x.dtype)
        whitened = torch.linalg.solve_triangular(
            chol, (x - center).unsqueeze(-1), upper=False
        ).squeeze(-1)
        return 1.0 - whitened.square().sum()

    return _fn


def _sample_unit_ball(
    n: int,
    dim: int,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if n < 1:
        raise ValueError("n must be positive")
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        direction = torch.randn(n, dim, device=device, dtype=dtype)
        direction = direction / torch.linalg.vector_norm(
            direction, dim=-1, keepdim=True
        ).clamp_min(torch.finfo(dtype).eps)
        radius = torch.rand(n, 1, device=device, dtype=dtype).pow(1.0 / dim)
        return direction * radius


def _ellipsoid_candidates(
    ellipsoid: LatentEllipsoid,
    *,
    seed: int,
    samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ball = _sample_unit_ball(
        samples,
        ellipsoid.dim,
        seed=seed,
        device=device,
        dtype=dtype,
    )
    center = torch.as_tensor(ellipsoid.center, device=device, dtype=dtype)
    chol = torch.as_tensor(ellipsoid.chol, device=device, dtype=dtype)
    # Project so SLSQP starts stay feasible under BoTorch's 1e-8 double tolerance.
    return _project_torch(center + ball @ chol.transpose(0, 1), ellipsoid)


def _is_duplicate(
    candidate: torch.Tensor,
    observed: torch.Tensor,
    bounds: torch.Tensor,
    tolerance: float,
) -> bool:
    if observed.numel() == 0:
        return False
    span = (bounds[1] - bounds[0]).clamp_min(torch.finfo(bounds.dtype).eps)
    distances = torch.linalg.vector_norm((observed - candidate) / span, dim=-1)
    return bool(torch.any(distances <= tolerance).item())


def _nonduplicate_mask(
    candidates: torch.Tensor,
    observed: torch.Tensor,
    bounds: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    if observed.numel() == 0:
        return torch.ones(len(candidates), dtype=torch.bool, device=candidates.device)
    span = (bounds[1] - bounds[0]).clamp_min(torch.finfo(bounds.dtype).eps)
    distances = torch.linalg.vector_norm(
        (candidates[:, None, :] - observed[None, :, :]) / span,
        dim=-1,
    )
    return torch.all(distances > tolerance, dim=1)


def _sobol_candidates(
    bounds: torch.Tensor,
    *,
    seed: int,
    samples: int,
) -> torch.Tensor:
    engine = torch.quasirandom.SobolEngine(
        dimension=bounds.shape[1], scramble=True, seed=seed
    )
    unit = engine.draw(samples).to(device=bounds.device, dtype=bounds.dtype)
    return bounds[0] + unit * (bounds[1] - bounds[0])


def _domain_candidates(
    bounds: torch.Tensor,
    *,
    seed: int,
    samples: int,
    ellipsoid: LatentEllipsoid | None = None,
) -> torch.Tensor:
    if ellipsoid is None:
        return _sobol_candidates(bounds, seed=seed, samples=samples)
    return _ellipsoid_candidates(
        ellipsoid,
        seed=seed,
        samples=samples,
        device=bounds.device,
        dtype=bounds.dtype,
    )


def _sobol_fallback(
    acquisition,
    bounds: torch.Tensor,
    observed: torch.Tensor,
    *,
    batch_size: int,
    seed: int,
    samples: int,
    duplicate_tolerance: float,
    allow_duplicates: bool = False,
    ellipsoid: LatentEllipsoid | None = None,
) -> torch.Tensor:
    candidates = _domain_candidates(
        bounds, seed=seed, samples=samples, ellipsoid=ellipsoid
    )
    if not allow_duplicates:
        candidates = candidates[
            _nonduplicate_mask(candidates, observed, bounds, duplicate_tolerance)
        ]
    if len(candidates) < batch_size:
        raise RuntimeError("acquisition exhausted: insufficient candidate points")
    with torch.no_grad():
        values = acquisition(candidates.unsqueeze(-2)).reshape(-1)
    ranked = candidates[torch.argsort(values, descending=True)[:batch_size]]
    if ellipsoid is not None:
        ranked = _project_torch(ranked, ellipsoid)
    return ranked


def _proposal_has_duplicates(
    points: torch.Tensor,
    observed: torch.Tensor,
    bounds: torch.Tensor,
    tolerance: float,
) -> bool:
    if not bool(
        torch.all(_nonduplicate_mask(points, observed, bounds, tolerance)).item()
    ):
        return True
    for index in range(len(points)):
        if _is_duplicate(points[index], points[:index], bounds, tolerance):
            return True
    return False


def _validate_inputs(
    surrogate: BiasGPSurrogate,
    bounds,
    observed_x,
    *,
    batch_size: int,
    num_restarts: int,
    raw_samples: int,
    duplicate_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_restarts < 1 or raw_samples < 1:
        raise ValueError("num_restarts and raw_samples must be positive")
    if duplicate_tolerance < 0:
        raise ValueError("duplicate_tolerance must be nonnegative")
    model = surrogate.model
    bound_t = torch.as_tensor(
        bounds, device=model.train_inputs[0].device, dtype=model.train_inputs[0].dtype
    )
    if (
        bound_t.shape != (2, model.train_inputs[0].shape[-1])
        or not torch.isfinite(bound_t).all()
        or torch.any(bound_t[1] <= bound_t[0])
    ):
        raise ValueError("bounds must be finite, [2,D], and strictly increasing")
    observed = torch.as_tensor(
        observed_x, device=bound_t.device, dtype=bound_t.dtype
    ).reshape(-1, bound_t.shape[1])
    if not torch.isfinite(observed).all():
        raise ValueError("observed points must be finite")
    return bound_t, observed


def _mc_acquisition(
    surrogate: BiasGPSurrogate,
    kind: Literal["log_ei", "ucb"],
    *,
    seed: int,
    mc_samples: int,
    ucb_beta: float,
):
    sampler = SobolQMCNormalSampler(
        sample_shape=torch.Size([mc_samples]),
        seed=seed,
    )
    if kind == "log_ei":
        return qLogExpectedImprovement(
            model=surrogate.model,
            best_f=surrogate.best_f,
            sampler=sampler,
        )
    return qUpperConfidenceBound(
        model=surrogate.model,
        beta=ucb_beta,
        sampler=sampler,
    )


def _thompson_candidates(
    surrogate: BiasGPSurrogate,
    bounds: torch.Tensor,
    observed: torch.Tensor,
    *,
    batch_size: int,
    seed: int,
    candidate_samples: int,
    duplicate_tolerance: float,
    allow_duplicates: bool,
    ellipsoid: LatentEllipsoid | None = None,
) -> torch.Tensor:
    candidates = _domain_candidates(
        bounds,
        seed=seed,
        samples=candidate_samples,
        ellipsoid=ellipsoid,
    )
    if not allow_duplicates:
        candidates = candidates[
            _nonduplicate_mask(candidates, observed, bounds, duplicate_tolerance)
        ]
    if len(candidates) < batch_size:
        raise RuntimeError("Thompson sampling candidate set is too small")
    sampler = MaxPosteriorSampling(model=surrogate.model, replacement=False)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        points = sampler(candidates, num_samples=batch_size)
    if ellipsoid is not None:
        points = _project_torch(points, ellipsoid)
    return points


def propose_candidates(
    surrogate: BiasGPSurrogate,
    bounds,
    observed_x,
    *,
    acquisition: AcquisitionKind = "log_ei",
    batch_size: int = 1,
    seed: int = 0,
    num_restarts: int = 10,
    raw_samples: int = 256,
    mc_samples: int = 128,
    ucb_beta: float = 0.2,
    thompson_candidates: int = 4096,
    duplicate_tolerance: float = 1e-6,
    observation_samples: int = 1,
    ellipsoid: LatentEllipsoid | None = None,
) -> np.ndarray:
    """Propose a joint batch from LogEI, UCB, or approximate Thompson sampling.

    Latent near-duplicates of observed points are allowed when
    ``observation_samples > 1`` (re-querying the same location is useful under
    noisy verification). With ``observation_samples == 1``, duplicate proposals
    fall back to a Sobol-ranked non-duplicate set.

    ``ellipsoid`` restricts proposals to a Mahalanobis ball. BoTorch still
    uses the ellipsoid's axis-aligned bounding box as ``bounds``; LogEI/UCB
    add BoTorch's intra-point nonlinear inequality and feasible ball starts
    so SLSQP does not initialize from empty high-D box samples. Nonlinear
    constraints force ``batch_limit=1``. Thompson and the discrete fallback
    draw uniformly inside the ellipsoid.
    """
    if acquisition not in ("log_ei", "ucb", "thompson"):
        raise ValueError(f"unknown acquisition function: {acquisition!r}")
    if mc_samples < 1 or thompson_candidates < batch_size:
        raise ValueError("MC samples and Thompson candidate count are too small")
    if ucb_beta < 0:
        raise ValueError("ucb_beta must be nonnegative")
    if observation_samples < 1:
        raise ValueError("observation_samples must be positive")
    if ellipsoid is not None:
        bounds = ellipsoid.aabb()
    bound_t, observed = _validate_inputs(
        surrogate,
        bounds,
        observed_x,
        batch_size=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        duplicate_tolerance=duplicate_tolerance,
    )
    allow_duplicates = observation_samples > 1

    if acquisition == "thompson":
        points = _thompson_candidates(
            surrogate,
            bound_t,
            observed,
            batch_size=batch_size,
            seed=seed,
            candidate_samples=thompson_candidates,
            duplicate_tolerance=duplicate_tolerance,
            allow_duplicates=allow_duplicates,
            ellipsoid=ellipsoid,
        )
        return points.detach().cpu().numpy().astype(np.float32)

    acq = _mc_acquisition(
        surrogate,
        acquisition,
        seed=seed,
        mc_samples=mc_samples,
        ucb_beta=ucb_beta,
    )
    points: torch.Tensor | None = None
    try:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            opt_kwargs: dict = {
                "acq_function": acq,
                "bounds": bound_t,
                "q": batch_size,
                "num_restarts": num_restarts,
                "raw_samples": raw_samples,
                "options": {
                    "batch_limit": 1 if ellipsoid is not None else 8,
                    "maxiter": 200,
                },
            }
            if ellipsoid is not None:
                starters = _ellipsoid_initial_conditions(
                    acq,
                    ellipsoid,
                    bound_t,
                    batch_size=batch_size,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    seed=seed,
                )
                opt_kwargs["num_restarts"] = int(starters.shape[0])
                opt_kwargs["raw_samples"] = None
                opt_kwargs["batch_initial_conditions"] = starters
                opt_kwargs["nonlinear_inequality_constraints"] = [
                    (_ellipsoid_constraint(ellipsoid), True)
                ]
                opt_kwargs["post_processing_func"] = (
                    lambda x, ell=ellipsoid: _project_torch(x, ell)
                )
            points, _ = optimize_acqf(**opt_kwargs)
    except (RuntimeError, ValueError, CandidateGenerationError):
        points = None

    needs_fallback = points is None or (
        not allow_duplicates
        and _proposal_has_duplicates(
            points, observed, bound_t, duplicate_tolerance
        )
    )
    if needs_fallback:
        points = _sobol_fallback(
            acq,
            bound_t,
            observed,
            batch_size=batch_size,
            seed=seed,
            samples=max(raw_samples, 256),
            duplicate_tolerance=duplicate_tolerance,
            allow_duplicates=allow_duplicates,
            ellipsoid=ellipsoid,
        )
    elif ellipsoid is not None:
        points = _project_torch(points, ellipsoid)
    return points.detach().cpu().numpy().astype(np.float32)


def _ellipsoid_initial_conditions(
    acquisition,
    ellipsoid: LatentEllipsoid,
    bounds: torch.Tensor,
    *,
    batch_size: int,
    num_restarts: int,
    raw_samples: int,
    seed: int,
) -> torch.Tensor:
    """Top-acquisition feasible starts, required by BoTorch for nonlinear constraints."""
    n_draw = max(raw_samples, num_restarts * batch_size, batch_size)
    pool = _ellipsoid_candidates(
        ellipsoid,
        seed=seed,
        samples=n_draw,
        device=bounds.device,
        dtype=bounds.dtype,
    )
    if batch_size == 1:
        with torch.no_grad():
            values = acquisition(pool.unsqueeze(-2)).reshape(-1)
        keep = min(num_restarts, len(pool))
        chosen = pool[torch.argsort(values, descending=True)[:keep]]
        return chosen.unsqueeze(-2)
    # Joint q-batches: consecutive groups ranked by mean acquisition.
    usable = (len(pool) // batch_size) * batch_size
    grouped = pool[:usable].reshape(-1, batch_size, pool.shape[-1])
    with torch.no_grad():
        values = acquisition(grouped).reshape(-1)
    keep = min(num_restarts, len(grouped))
    return grouped[torch.argsort(values, descending=True)[:keep]]


def score_discrete_candidates(
    surrogate: BiasGPSurrogate,
    points,
    *,
    acquisition: AcquisitionKind = "log_ei",
    seed: int = 0,
    mc_samples: int = 128,
    ucb_beta: float = 0.2,
) -> np.ndarray:
    """Evaluate LogEI, UCB, or one Thompson draw on a finite candidate set.

    LogEI and UCB are independent q=1 scores. Thompson is a single posterior
    sample; callers that take top-k of that sample are not sequential TS.
    """
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or not len(array) or array.shape[1] == 0:
        raise ValueError("points must be a non-empty [N, D] array")
    if not np.isfinite(array).all():
        raise ValueError("points must be finite")
    if acquisition not in ("log_ei", "ucb", "thompson"):
        raise ValueError(f"unknown acquisition function: {acquisition!r}")
    if mc_samples < 1:
        raise ValueError("mc_samples must be positive")
    if ucb_beta < 0:
        raise ValueError("ucb_beta must be nonnegative")
    model = surrogate.model
    tensor = torch.as_tensor(
        array,
        device=model.train_inputs[0].device,
        dtype=model.train_inputs[0].dtype,
    )
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(seed)
        if acquisition == "thompson":
            sample = surrogate.model.posterior(tensor).rsample(torch.Size([1]))
            values = sample.reshape(-1)
        else:
            acq = _mc_acquisition(
                surrogate,
                acquisition,
                seed=seed,
                mc_samples=mc_samples,
                ucb_beta=ucb_beta,
            )
            values = acq(tensor.unsqueeze(-2)).reshape(-1)
    if values.numel() != len(array):
        raise RuntimeError("discrete acquisition returned the wrong number of scores")
    scores = values.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(scores).all():
        raise RuntimeError("discrete acquisition produced non-finite scores")
    return scores


def select_discrete_candidates(
    surrogate: BiasGPSurrogate,
    points,
    *,
    acquisition: AcquisitionKind = "log_ei",
    batch_size: int = 1,
    seed: int = 0,
    mc_samples: int = 128,
    ucb_beta: float = 0.2,
) -> np.ndarray:
    """Return indices of the highest-acquisition members of a finite set.

    Ranking is independent top-k of :func:`score_discrete_candidates`.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    scores = score_discrete_candidates(
        surrogate,
        points,
        acquisition=acquisition,
        seed=seed,
        mc_samples=mc_samples,
        ucb_beta=ucb_beta,
    )
    if batch_size > len(scores):
        raise ValueError("batch_size exceeds the discrete candidate set")
    return np.argsort(-scores, kind="stable")[:batch_size]


def propose_expected_improvement(
    surrogate: BiasGPSurrogate,
    bounds,
    observed_x,
    *,
    seed: int = 0,
    num_restarts: int = 10,
    raw_samples: int = 256,
    duplicate_tolerance: float = 1e-6,
) -> np.ndarray:
    """Backward-compatible single-point LogEI proposal."""
    return propose_candidates(
        surrogate,
        bounds,
        observed_x,
        acquisition="log_ei",
        batch_size=1,
        seed=seed,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        duplicate_tolerance=duplicate_tolerance,
    )[0]
