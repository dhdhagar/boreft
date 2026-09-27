"""Bayesian optimization over BOReFT latent bias vectors."""

from .acquisition import (
    AcquisitionKind,
    LatentEllipsoid,
    box_log_volume,
    latent_bounds,
    latent_ellipsoid,
    propose_candidates,
    propose_expected_improvement,
    score_discrete_candidates,
    select_discrete_candidates,
    unit_ball_log_volume,
)
from .runner import BOConfig, Verification, Verifier, run_bo
from .state import Observation, RunState
from .surrogate import (
    BiasGPSurrogate,
    KernelKind,
    SurrogateFitConfig,
    fit_surrogate,
)

__all__ = [
    "BiasGPSurrogate",
    "BOConfig",
    "AcquisitionKind",
    "KernelKind",
    "Observation",
    "RunState",
    "SurrogateFitConfig",
    "Verification",
    "Verifier",
    "fit_surrogate",
    "LatentEllipsoid",
    "latent_bounds",
    "latent_ellipsoid",
    "box_log_volume",
    "unit_ball_log_volume",
    "propose_candidates",
    "propose_expected_improvement",
    "run_bo",
    "score_discrete_candidates",
    "select_discrete_candidates",
]
