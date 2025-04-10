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
        if mean is None:
            self._mean: Sequence[float] | jnp.ndarray = [0.0] * len(truncate_at)
        else:
            self._mean = mean
        if std is None:
            self._std: Sequence[float] | jnp.ndarray = [1.0] * len(truncate_at)
        else:
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
        values = jnp.exp(
            -((coordinate - self._mean[spatial_dim]) ** 2) / (2 * self._std[spatial_dim] ** 2)
        )
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
