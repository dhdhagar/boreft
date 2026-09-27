"""MLP that maps frozen target embeddings to low-rank bias vectors."""

from __future__ import annotations

import torch
import torch.nn as nn


class TargetBiasNetwork(nn.Module):
    """Map target embedding e -> mu (and optionally logvar for VAE mode).

    Architecture:
        Linear(d, 256) -> ReLU -> LayerNorm -> Linear(256, 128) -> ReLU
        mu head: Linear(128, b_dim)
        logvar head (optional): Linear(128, b_dim)

    Optional residual on mu: mu = Linear_skip(d, b_dim)(e) + mu_head(trunk(e))
    """

    def __init__(
        self,
        embed_dim: int,
        bias_dim: int,
        *,
        learnable_logvar: bool = True,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.learnable_logvar = learnable_logvar
        self.residual = residual

        self.fc1 = nn.Linear(embed_dim, 256)
        self.ln = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.mu_head = nn.Linear(128, bias_dim)
        self.logvar_head = nn.Linear(128, bias_dim) if learnable_logvar else None
        self.mu_skip = (
            nn.Linear(embed_dim, bias_dim, bias=False) if residual else None
        )

    def _trunk(self, e: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(e.float()))
        h = self.ln(h)
        h = torch.relu(self.fc2(h))
        return h

    def _mu_from_hidden(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        mu = self.mu_head(h)
        if self.mu_skip is not None:
            mu = mu + self.mu_skip(e.float())
        return mu

    def mu(self, e: torch.Tensor) -> torch.Tensor:
        """Return mu [B, bias_dim] with one trunk pass (logvar head not used)."""
        h = self._trunk(e)
        return self._mu_from_hidden(h, e)
    
    def forward(
        self, e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Single trunk pass; returns (mu, logvar) or (mu, None) if logvar is fixed."""
        h = self._trunk(e)
        mu = self._mu_from_hidden(h, e)
        logvar = self.logvar_head(h) if self.logvar_head is not None else None
        return mu, logvar
