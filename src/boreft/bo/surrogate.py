"""Exact-GP surrogates for Bayesian optimization over bias vectors."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP

SurrogateKind = Literal["static", "projected"]
KernelKind = Literal["matern-0.5", "matern-1.5", "matern-2.5", "rbf"]
KERNEL_CHOICES: tuple[KernelKind, ...] = ("matern-0.5", "matern-1.5", "matern-2.5", "rbf")
DEFAULT_KERNEL: KernelKind = "matern-2.5"

_MATERN_NU = {
    "matern-0.5": 0.5,
    "matern-1.5": 1.5,
    "matern-2.5": 2.5,
}


def covariance_kernel(
    kind: KernelKind,
    feature_dim: int,
    use_ard: bool,
) -> ScaleKernel:
    """Scale-wrapped RBF or Matern kernel with optional ARD lengthscales."""
    if kind not in KERNEL_CHOICES:
        raise ValueError(
            f"unknown GP kernel: {kind!r} (expected one of {KERNEL_CHOICES})"
        )
    ard_num_dims = feature_dim if use_ard else None
    if kind == "rbf":
        base = RBFKernel(ard_num_dims=ard_num_dims)
    else:
        base = MaternKernel(nu=_MATERN_NU[kind], ard_num_dims=ard_num_dims)
    return ScaleKernel(base)


class Projection(torch.nn.Module):
    """GOLLuM-style feature map: ``num_layers`` stacked Linear -> ELU blocks."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 64,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("projection num_layers must be positive")
        layers = [torch.nn.Linear(input_dim, output_dim)]
        layers.extend(
            torch.nn.Linear(output_dim, output_dim) for _ in range(num_layers - 1)
        )
        for layer in layers:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.constant_(layer.bias, 0.01)
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.nn.functional.elu(layer(x))
        return x


class BiasExactGP(ExactGP, GPyTorchModel):
    """Exact GP whose inputs are normalized raw bias vectors."""

    _num_outputs = 1

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        bounds: torch.Tensor,
        *,
        projection_dim: int | None = None,
        projection_layers: int = 1,
        use_ard: bool = False,
        kernel: KernelKind = DEFAULT_KERNEL,
        train_yvar: torch.Tensor | None = None,
    ) -> None:
        if train_yvar is None:
            likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-6))
        else:
            likelihood = FixedNoiseGaussianLikelihood(
                noise=train_yvar.clamp_min(1e-8),
                learn_additional_noise=True,
                noise_constraint=GreaterThan(1e-6),
            )
        super().__init__(train_x, train_y, likelihood)
        self.register_buffer("lower", bounds[0].clone())
        span = (bounds[1] - bounds[0]).clone()
        self.register_buffer("span", torch.where(span > 0, span, torch.ones_like(span)))
        self.projection_layers = (
            int(projection_layers) if projection_dim is not None else 0
        )
        self.projection = (
            Projection(
                train_x.shape[-1],
                projection_dim,
                num_layers=self.projection_layers,
            )
            if projection_dim is not None
            else torch.nn.Identity()
        )
        self.use_ard = bool(use_ard)
        self.kernel = kernel
        feature_dim = (
            int(projection_dim)
            if projection_dim is not None
            else int(train_x.shape[-1])
        )
        self.mean_module = ConstantMean()
        self.covar_module = covariance_kernel(
            self.kernel, feature_dim, self.use_ard
        )
        if isinstance(self.likelihood, FixedNoiseGaussianLikelihood):
            self.likelihood.second_noise = 1e-4
        else:
            self.likelihood.noise = 1e-4
        self.covar_module.outputscale = 1.0
        self.covar_module.base_kernel.lengthscale = 1.0

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection((x - self.lower) / self.span)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        z = self.features(x)
        return MultivariateNormal(self.mean_module(z), self.covar_module(z))


@dataclass
class SurrogateFitConfig:
    kind: SurrogateKind = "static"
    kernel: KernelKind = DEFAULT_KERNEL
    use_ard: bool = False
    projection_dim: int = 64
    projection_layers: int = 1
    steps: int = 300
    gp_lr: float = 0.2
    projection_lr: float = 0.002
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    tolerance: float = 1e-7
    patience: int = 25
    device: str = "cpu"
    dtype: torch.dtype = torch.double

    def validate(self) -> None:
        if self.kind not in ("static", "projected"):
            raise ValueError(f"unknown surrogate kind: {self.kind!r}")
        if self.kernel not in KERNEL_CHOICES:
            raise ValueError(
                f"unknown GP kernel: {self.kernel!r} (expected one of {KERNEL_CHOICES})"
            )
        if self.projection_dim < 1 or self.steps < 1 or self.patience < 1:
            raise ValueError("projection_dim, steps, and patience must be positive")
        if self.projection_layers < 1:
            raise ValueError("projection_layers must be positive")
        if self.gp_lr <= 0 or self.projection_lr <= 0:
            raise ValueError("GP and projection learning rates must be positive")
        if self.weight_decay < 0 or self.grad_clip <= 0 or self.tolerance < 0:
            raise ValueError(
                "weight_decay/tolerance must be nonnegative and grad_clip positive"
            )


class BiasGPSurrogate:
    """Fitted BO surrogate with stable target standardization."""

    def __init__(
        self,
        model: BiasExactGP,
        *,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
        kind: SurrogateKind,
        fit_loss: float,
    ) -> None:
        self.model = model
        self.y_mean = y_mean
        self.y_std = y_std
        self.kind = kind
        self.fit_loss = fit_loss

    @property
    def best_f(self) -> float:
        return float(self.model.train_targets.max().item())

    def unstandardize(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.y_std + self.y_mean

    def metadata(self) -> dict[str, float | str | int | bool]:
        return {
            "kind": self.kind,
            "kernel": self.model.kernel,
            "use_ard": self.model.use_ard,
            "projection_layers": self.model.projection_layers,
            "fit_loss": self.fit_loss,
            "lengthscale": float(
                self.model.covar_module.base_kernel.lengthscale.mean().item()
            ),
            "outputscale": float(self.model.covar_module.outputscale.item()),
            "noise": float(self.model.likelihood.noise.mean().item()),
            "feature_dim": int(
                self.model.features(self.model.train_inputs[0][:1]).shape[-1]
            ),
        }

    def checkpoint(self) -> dict:
        """Portable fitted state plus the data needed to reconstruct the model."""
        return {
            "kind": self.kind,
            "kernel": self.model.kernel,
            "use_ard": self.model.use_ard,
            "projection_layers": self.model.projection_layers,
            "model_state_dict": self.model.state_dict(),
            "train_x": self.model.train_inputs[0].detach().cpu(),
            "train_y_standardized": self.model.train_targets.detach().cpu(),
            "train_yvar_standardized": (
                self.model.likelihood.noise_covar.noise.detach().cpu()
                if isinstance(
                    self.model.likelihood,
                    FixedNoiseGaussianLikelihood,
                )
                else None
            ),
            "bounds": torch.stack([self.model.lower, self.model.lower + self.model.span])
            .detach()
            .cpu(),
            "y_mean": self.y_mean.detach().cpu(),
            "y_std": self.y_std.detach().cpu(),
            "metadata": self.metadata(),
        }


def _as_training_tensors(
    x,
    y,
    bounds,
    config: SurrogateFitConfig,
    observation_variances=None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    train_x = torch.as_tensor(x, device=config.device, dtype=config.dtype)
    train_y_raw = torch.as_tensor(y, device=config.device, dtype=config.dtype).reshape(-1)
    bound_t = torch.as_tensor(bounds, device=config.device, dtype=config.dtype)
    if train_x.ndim != 2 or train_x.shape[1] == 0 or len(train_x) != len(train_y_raw):
        raise ValueError("x must be [N,D] and y must contain N scalar observations")
    if len(train_x) < 2:
        raise ValueError("an exact GP requires at least two observations")
    if bound_t.shape != (2, train_x.shape[1]):
        raise ValueError(f"bounds must have shape [2,{train_x.shape[1]}]")
    if (
        not torch.isfinite(train_x).all()
        or not torch.isfinite(train_y_raw).all()
        or not torch.isfinite(bound_t).all()
    ):
        raise ValueError("GP observations and bounds must be finite")
    if torch.any(bound_t[1] <= bound_t[0]):
        raise ValueError("GP bounds must be strictly increasing")
    y_mean = train_y_raw.mean()
    y_std = train_y_raw.std(unbiased=False).clamp_min(1e-8)
    train_y = (train_y_raw - y_mean) / y_std
    train_yvar = None
    if observation_variances is not None:
        raw_yvar = torch.as_tensor(
            observation_variances,
            device=config.device,
            dtype=config.dtype,
        ).reshape(-1)
        if (
            len(raw_yvar) != len(train_y)
            or not torch.isfinite(raw_yvar).all()
            or torch.any(raw_yvar < 0)
        ):
            raise ValueError(
                "observation_variances must contain N finite, nonnegative values"
            )
        train_yvar = raw_yvar / y_std.square()
    return train_x, train_y, bound_t, y_mean, y_std, train_yvar


def fit_surrogate(
    x,
    y,
    bounds,
    config: SurrogateFitConfig | None = None,
    *,
    observation_variances=None,
) -> BiasGPSurrogate:
    """Fit a fresh static or projected exact GP to all observations."""
    config = config or SurrogateFitConfig()
    config.validate()
    train_x, train_y, bound_t, y_mean, y_std, train_yvar = _as_training_tensors(
        x,
        y,
        bounds,
        config,
        observation_variances,
    )
    model = BiasExactGP(
        train_x,
        train_y,
        bound_t,
        projection_dim=config.projection_dim if config.kind == "projected" else None,
        projection_layers=config.projection_layers,
        use_ard=config.use_ard,
        kernel=config.kernel,
        train_yvar=train_yvar,
    ).to(device=config.device, dtype=config.dtype)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    if config.kind == "static":
        fit_gpytorch_mll(mll)
        model.train()
        model.likelihood.train()
        with torch.no_grad():
            fit_loss = float(-mll(model(train_x), train_y).item())
        if not math.isfinite(fit_loss):
            raise RuntimeError("static GP marginal likelihood became non-finite")
        model.eval()
        model.likelihood.eval()
    elif config.kind == "projected":
        projection_params = list(model.projection.parameters())
        projection_ids = {id(param) for param in projection_params}
        gp_params = [p for p in model.parameters() if id(p) not in projection_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": projection_params, "lr": config.projection_lr},
                {"params": gp_params, "lr": config.gp_lr},
            ],
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95)
        best = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0
        fit_loss = best
        model.train()
        model.likelihood.train()
        for _ in range(config.steps):
            optimizer.zero_grad(set_to_none=True)
            loss = -mll(model(train_x), train_y)
            if not torch.isfinite(loss):
                raise RuntimeError("projected GP marginal likelihood became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                fit_loss = float((-mll(model(train_x), train_y)).item())
            if not math.isfinite(fit_loss):
                raise RuntimeError("projected GP marginal likelihood became non-finite")
            if best - fit_loss > config.tolerance:
                best, stale = fit_loss, 0
                best_state = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                }
            else:
                stale += 1
                if stale >= config.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        with torch.no_grad():
            fit_loss = float((-mll(model(train_x), train_y)).item())
        model.eval()
        model.likelihood.eval()
    return BiasGPSurrogate(
        model,
        y_mean=y_mean,
        y_std=y_std,
        kind=config.kind,
        fit_loss=fit_loss,
    )
