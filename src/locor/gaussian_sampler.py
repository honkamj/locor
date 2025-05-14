"""Gaussian smoothing sampler."""

from typing import Sequence

import jax.numpy as jnp
import numpy as np
from jaxmorph import PiecewiseKernelDefinition, SeparableSampler


class GaussianKernel(PiecewiseKernelDefinition):
    """Gaussian kernel."""

    _mean: Sequence[int | float] | jnp.ndarray | int | float
    _std: Sequence[int | float] | jnp.ndarray | int | float
    _truncate_at: (
        Sequence[tuple[int | float, int | float]] | tuple[int | float, int | float] | float | int
    )

    def __init__(
        self,
        truncate_at: (
            Sequence[tuple[int | float, int | float]]
            | tuple[int | float, int | float]
            | float
            | int
        ),
        mean: Sequence[int | float] | jnp.ndarray | None = None,
        std: Sequence[int | float] | jnp.ndarray | None = None,
    ) -> None:
        self._truncate_at = truncate_at
        self._mean = 0.0 if mean is None else mean
        self._std = 0.0 if std is None else std

    def is_interpolating_kernel(self, spatial_dim: int) -> bool:
        return False

    def edge_continuity_schedule(self, spatial_dim: int) -> np.ndarray:
        return np.zeros((1, 2), dtype=np.dtype("bool"))

    def piece_edges(self, spatial_dim: int, dtype: jnp.dtype | np.dtype) -> np.ndarray:
        truncate_at = np.array(self._truncate_at)
        if truncate_at.ndim == 0:
            truncate_at_min = -truncate_at
            truncate_at_max = truncate_at
        elif truncate_at.ndim == 1:
            truncate_at_min, truncate_at_max = truncate_at
        elif truncate_at.ndim == 2:
            truncate_at_min, truncate_at_max = truncate_at[spatial_dim]
        else:
            raise ValueError("Invalid truncation specification")
        return np.linspace(
            truncate_at_min,
            truncate_at_max,
            2,
            dtype=dtype,
        )

    def evaluate(self, spatial_dim: int, coordinate: jnp.ndarray) -> jnp.ndarray:
        mean = jnp.asarray(self._mean, dtype=coordinate.dtype).reshape(-1)
        std = jnp.asarray(self._std, dtype=coordinate.dtype).reshape(-1)
        values = jnp.exp(-((coordinate - mean[spatial_dim]) ** 2) / (2 * std[spatial_dim] ** 2))
        return values

    def derivative(self, spatial_dim: int) -> PiecewiseKernelDefinition:
        raise NotImplementedError(
            "Derivative of Gaussian kernel is not implemented "
            "(normalization brakes the generic implementation)."
        )

    def __call__(self, spatial_dim: int, coordinates: jnp.ndarray):
        kernel = super().__call__(spatial_dim, coordinates)
        kernel = kernel / kernel.sum(axis=1, keepdims=True)
        return kernel


class GaussianSampler(SeparableSampler):
    """Smoothing with Gaussian kernel."""

    def __init__(
        self,
        truncate_at: (
            Sequence[tuple[int | float, int | float]]
            | tuple[int | float, int | float]
            | float
            | int
        ),
        mean: Sequence[int | float] | jnp.ndarray | None = None,
        std: Sequence[int | float] | jnp.ndarray | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            kernel=GaussianKernel(truncate_at=truncate_at, mean=mean, std=std), **kwargs
        )
