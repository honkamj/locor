"""Gaussian smoothing sampler."""

from typing import Sequence

import jax.numpy as jnp
import numpy as np
from jaxmorph import LimitDirection, PiecewiseKernelDefinition, SeparableSampler


class GaussianKernel(PiecewiseKernelDefinition):
    """Gaussian kernel."""

    def __init__(
        self,
        truncate_at: Sequence[int | float],
        mean: jnp.ndarray | None = None,
        std: jnp.ndarray | None = None,
    ) -> None:
        self._truncate_at = truncate_at
        self._mean = mean
        self._std = std

    def is_interpolating_kernel(self, spatial_dim: int) -> bool:
        return False

    def edge_continuity_schedule(self, spatial_dim: int) -> np.ndarray:
        return np.zeros((1, 2), dtype=np.dtype("bool"))

    def piece_edges(self, spatial_dim: int, dtype: jnp.dtype | np.dtype) -> np.ndarray:
        return np.linspace(
            -self._truncate_at[spatial_dim],
            self._truncate_at[spatial_dim],
            2,
            dtype=dtype,
        )

    def evaluate(self, spatial_dim: int, coordinate: jnp.ndarray) -> jnp.ndarray:
        if self._mean is None:
            mean: float | jnp.ndarray = 0.0
        else:
            mean = self._mean[spatial_dim]
        if self._std is None:
            std: float | jnp.ndarray = 1.0
        else:
            std = self._std[spatial_dim]
        values = jnp.exp(-((coordinate - mean) ** 2) / (2 * std**2))
        values = values / values.sum()
        return values


class GaussianSampler(SeparableSampler):
    """Interpolation with gaussian kernel."""

    def __init__(
        self,
        truncate_at: Sequence[int | float],
        mean: jnp.ndarray | None = None,
        std: jnp.ndarray | None = None,
    ) -> None:
        self._mean = mean
        self._std = std
        self._truncate_at = truncate_at
        super().__init__(
            kernel=GaussianKernel(truncate_at=truncate_at, mean=mean, std=std),
            extrapolation_mode="constant",
            mask_extrapolated_regions=False,
            limit_direction=LimitDirection.average(),
        )
