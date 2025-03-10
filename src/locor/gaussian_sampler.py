"""Gaussian smoothing sampler."""

from typing import Sequence, Tuple, Union

from composable_mapping import BaseSeparableSampler, LimitDirection
from composable_mapping.sampler.base import ISeparableKernelSupport
from torch import Tensor, exp


class GaussianKernelSupport(ISeparableKernelSupport):
    """Kernel support for Gaussian smoothing."""

    def __init__(self, truncate_at: float) -> None:
        self.truncate_at = truncate_at

    def __call__(self, limit_direction: LimitDirection) -> Tuple[float, float, bool, bool]:
        return -self.truncate_at, self.truncate_at, True, True

    def derivative(self) -> "GaussianKernelSupport":
        return self


class GaussianSampler(BaseSeparableSampler):
    """Interpolation with gaussian kernel."""

    def __init__(
        self,
        truncate_at: Sequence[int | float],
        mean: Sequence[int | float] | None = None,
        std: Sequence[int | float] | None = None,
        extrapolation_mode: str = "zeros",
        mask_extrapolated_regions: bool = False,
        convolution_threshold: float = 1e-3,
        mask_threshold: float = 1e-5,
    ) -> None:
        self._mean = mean
        self._std = std
        self._truncate_at = truncate_at
        super().__init__(
            extrapolation_mode=extrapolation_mode,
            convolution_threshold=convolution_threshold,
            mask_extrapolated_regions=mask_extrapolated_regions,
            mask_threshold=mask_threshold,
            limit_direction=LimitDirection.left(),
        )

    def _kernel_support(self, spatial_dim: int) -> GaussianKernelSupport:
        return GaussianKernelSupport(
            truncate_at=self._truncate_at[spatial_dim],
        )

    def _is_interpolating_kernel(self, spatial_dim: int) -> bool:
        return False

    def _left_limit_kernel(self, coordinates: Tensor, spatial_dim: int) -> Tensor:
        return self._right_limit_kernel(coordinates, spatial_dim)

    def _right_limit_kernel(self, coordinates: Tensor, spatial_dim: int) -> Tensor:
        std: Union[Tensor, float] = 1.0 if self._std is None else self._std[spatial_dim]
        mean: Union[Tensor, float] = 0.0 if self._mean is None else self._mean[spatial_dim]
        values = exp(-((coordinates - mean) ** 2) / (2 * std**2))
        values = values / values.sum()
        return values

    def sample_values(
        self,
        volume: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        raise NotImplementedError

    def sample_mask(
        self,
        mask: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        raise NotImplementedError
