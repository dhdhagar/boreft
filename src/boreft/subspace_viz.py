"""Projection and serialization helpers for the interactive bias-space map."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class Projection2D:
    """A two-axis PCA plane embedded in a higher-dimensional bias space."""

    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray

    @property
    def bias_dim(self) -> int:
        return int(self.mean.shape[0])

    @property
    def projected_dim(self) -> int:
        return int(self.components.shape[0])

    def project(self, vectors: np.ndarray | Sequence[float]) -> np.ndarray:
        array = np.asarray(vectors, dtype=np.float64)
        one_vector = array.ndim == 1
        if one_vector:
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != self.bias_dim:
            raise ValueError(
                f"expected vectors with bias dimension {self.bias_dim}, got {array.shape}"
            )
        coordinates = (array - self.mean) @ self.components.T
        padded = np.zeros((coordinates.shape[0], 2), dtype=np.float64)
        padded[:, : self.projected_dim] = coordinates
        return padded[0] if one_vector else padded

    def inverse(self, x: float, y: float) -> np.ndarray:
        """Lift a 2D click onto the PCA plane, with discarded PCs set to zero."""
        coordinates = np.asarray([x, y], dtype=np.float64)[: self.projected_dim]
        return self.mean + coordinates @ self.components


def fit_projection(vectors: np.ndarray | Sequence[Sequence[float]]) -> Projection2D:
    """Fit a deterministic PCA plane without requiring plotting dependencies."""
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("bias vectors must be a non-empty [points, bias_dim] matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("bias vectors contain non-finite values")

    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    n_components = min(2, matrix.shape[0], matrix.shape[1])
    components = vh[:n_components]
    variances = singular_values[:n_components] ** 2
    total_variance = float(np.square(singular_values).sum())
    ratios = (
        variances / total_variance
        if total_variance > np.finfo(np.float64).eps
        else np.zeros(n_components, dtype=np.float64)
    )
    return Projection2D(
        mean=mean,
        components=components,
        explained_variance_ratio=ratios,
    )


def bias_l2_norm(vector: np.ndarray | Sequence[float]) -> float:
    """L2 norm of a bias vector."""
    array = np.asarray(vector, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(array))


def mean_bias_l2_norm(vectors: np.ndarray | Sequence[Sequence[float]]) -> float:
    """Mean L2 norm across rows of a bias matrix."""
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(matrix, axis=1).mean())


def projected_point(
    *,
    point_id: str,
    kind: str,
    label: str,
    coordinates: Sequence[float],
    cluster: str,
    metadata: dict[str, Any] | None = None,
    bias_norm: float | None = None,
) -> dict[str, Any]:
    point: dict[str, Any] = {
        "id": point_id,
        "kind": kind,
        "label": label,
        "x": float(coordinates[0]),
        "y": float(coordinates[1]),
        "cluster": cluster,
        "metadata": metadata or {},
    }
    if bias_norm is not None:
        point["bias_norm"] = float(bias_norm)
    return point


def plot_bounds(points: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Return padded finite bounds suitable for initializing the browser canvas."""
    if not points:
        return {"min_x": -1.0, "max_x": 1.0, "min_y": -1.0, "max_y": 1.0}
    xs = np.asarray([point["x"] for point in points], dtype=np.float64)
    ys = np.asarray([point["y"] for point in points], dtype=np.float64)

    def _axis_bounds(values: np.ndarray) -> tuple[float, float]:
        lo, hi = float(values.min()), float(values.max())
        span = hi - lo
        padding = 0.08 * span if span > 1e-12 else max(abs(lo) * 0.08, 1.0)
        return lo - padding, hi + padding

    min_x, max_x = _axis_bounds(xs)
    min_y, max_y = _axis_bounds(ys)
    return {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y}
