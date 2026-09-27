"""Standardized search-baseline interfaces and implementation scaffolding."""

from .autodiscovery import AutoDiscoveryBaseline, AutoDiscoveryConfig
from .base import (
    ArtifactLayout,
    Baseline,
    BaselineName,
    BaselineObservation,
    BaselineRunState,
    Candidate,
    GenerationOptions,
    ObservationPhase,
    ScoreResult,
    SearchContext,
    StatelessBaseline,
    solution_key,
    timed_score,
)
from .bopro import BOPROBaseline, BOPROConfig
from .discrete_bo import DiscreteBOBaseline, DiscreteBOConfig
from .migrate import MiGrATeBaseline, MiGrATeConfig
from .opro import OPROBaseline, OPROConfig
from .sdpo_ttt import SDPOTTTBaseline, SDPOTTTConfig
from .random_sampling import RandomSamplingBaseline, RandomSamplingConfig
from .registry import (
    BASELINE_REGISTRY,
    BaselineSpec,
    available_baselines,
    create_baseline,
    default_baseline_config,
)
from .runner import (
    BaselineLoopConfig,
    WarmstartSeed,
    load_method_state,
    run_baseline,
    save_method_state,
)

__all__ = [
    "ArtifactLayout",
    "AutoDiscoveryBaseline",
    "AutoDiscoveryConfig",
    "BASELINE_REGISTRY",
    "BOPROBaseline",
    "BOPROConfig",
    "Baseline",
    "BaselineLoopConfig",
    "BaselineName",
    "BaselineObservation",
    "BaselineRunState",
    "BaselineSpec",
    "Candidate",
    "DiscreteBOBaseline",
    "DiscreteBOConfig",
    "GenerationOptions",
    "MiGrATeBaseline",
    "MiGrATeConfig",
    "OPROBaseline",
    "OPROConfig",
    "SDPOTTTBaseline",
    "SDPOTTTConfig",
    "ObservationPhase",
    "RandomSamplingBaseline",
    "RandomSamplingConfig",
    "ScoreResult",
    "SearchContext",
    "StatelessBaseline",
    "WarmstartSeed",
    "available_baselines",
    "create_baseline",
    "default_baseline_config",
    "load_method_state",
    "run_baseline",
    "save_method_state",
    "solution_key",
    "timed_score",
]
