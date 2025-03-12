"""Gaussian smoothing sampler."""

from typing import Sequence, Union

from composable_mapping import (
    LimitDirection,
    PiecewiseKernelDefinition,
    SeparableSampler,
)
from torch import Tensor
from torch import bool as torch_bool
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import exp, linspace, zeros


class GaussianKernel(PiecewiseKernelDefinition):
    """Gaussian kernel."""

    def __init__(
        self,
        truncate_at: Sequence[int | float],
        mean: Sequence[int | float] | None = None,
        std: Sequence[int | float] | None = None,
    ) -> None:
        self._truncate_at = truncate_at
        self._mean = mean
        self._std = std

    def is_interpolating_kernel(self, spatial_dim: int) -> bool:
        return False

    def edge_continuity_schedule(self, spatial_dim: int, device: torch_device) -> Tensor:
        return zeros(  # Due to truncating, the kernel is not continuous at the edges
            (1, 3), device=device, dtype=torch_bool
        )

    def piece_edges(self, spatial_dim: int, dtype: torch_dtype, device: torch_device) -> Tensor:
        return linspace(
            -self._truncate_at[spatial_dim],
            self._truncate_at[spatial_dim],
            2,
            dtype=dtype,
            device=device,
        )

    def evaluate(self, spatial_dim: int, coordinates: Tensor) -> Tensor:
        std: Union[Tensor, float] = 1.0 if self._std is None else self._std[spatial_dim]
        mean: Union[Tensor, float] = 0.0 if self._mean is None else self._mean[spatial_dim]
        values = exp(-((coordinates - mean) ** 2) / (2 * std**2))
        values = values / values.sum()
        return values


class GaussianSampler(SeparableSampler):
    """Interpolation with gaussian kernel."""

    def __init__(
        self,
        truncate_at: Sequence[int | float],
        mean: Sequence[int | float] | None = None,
        std: Sequence[int | float] | None = None,
    ) -> None:
        self._mean = mean
        self._std = std
        self._truncate_at = truncate_at
        super().__init__(
            kernel=GaussianKernel(truncate_at=truncate_at, mean=mean, std=std),
            extrapolation_mode="zeros",
            mask_extrapolated_regions=False,
            limit_direction=LimitDirection.average(),
        )
