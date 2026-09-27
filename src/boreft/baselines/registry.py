"""Central registry for baseline search implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .autodiscovery import AutoDiscoveryBaseline, AutoDiscoveryConfig
from .base import Baseline, BaselineName
from .bopro import BOPROBaseline, BOPROConfig
from .discrete_bo import DiscreteBOBaseline, DiscreteBOConfig
from .migrate import MiGrATeBaseline, MiGrATeConfig
from .opro import OPROBaseline, OPROConfig
from .sdpo_ttt import SDPOTTTBaseline, SDPOTTTConfig
from .random_sampling import RandomSamplingBaseline, RandomSamplingConfig


BaselineFactory = Callable[[Any], Baseline]


@dataclass(frozen=True)
class BaselineSpec:
    factory: BaselineFactory
    config_type: type

    def create(self, config: object | None = None) -> Baseline:
        if config is not None and not isinstance(config, self.config_type):
            raise TypeError(
                f"expected {self.config_type.__name__}, got {type(config).__name__}"
            )
        return self.factory(self.config_type() if config is None else config)


BASELINE_REGISTRY: dict[BaselineName, BaselineSpec] = {
    "random_sampling": BaselineSpec(RandomSamplingBaseline, RandomSamplingConfig),
    "discrete_bo": BaselineSpec(DiscreteBOBaseline, DiscreteBOConfig),
    "opro": BaselineSpec(OPROBaseline, OPROConfig),
    "sdpo_ttt": BaselineSpec(SDPOTTTBaseline, SDPOTTTConfig),
    "bopro": BaselineSpec(BOPROBaseline, BOPROConfig),
    "migrate": BaselineSpec(MiGrATeBaseline, MiGrATeConfig),
    "autodiscovery": BaselineSpec(AutoDiscoveryBaseline, AutoDiscoveryConfig),
}


def available_baselines() -> tuple[BaselineName, ...]:
    return tuple(BASELINE_REGISTRY)


def _spec(name: str) -> BaselineSpec:
    try:
        return BASELINE_REGISTRY[name]
    except KeyError as exc:
        choices = ", ".join(available_baselines())
        raise ValueError(f"unknown baseline {name!r}; choose one of: {choices}") from exc


def create_baseline(name: str, config: object | None = None) -> Baseline:
    return _spec(name).create(config)


def default_baseline_config(name: str) -> object:
    return _spec(name).config_type()
