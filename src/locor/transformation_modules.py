"""Transformation parametrizations defined as modules in order to use DistributedDataParallel."""

from typing import Sequence

from torch import zeros
from torch.nn import Module, Parameter

from .affine_transformation import (
    AffineTransformationTypeDefinition,
    calculate_n_parameters,
)


class AffineTransformationParameters(Module):
    """Affine transformation parameters."""

    def __init__(
        self,
        n_dims: int,
        transformation_type: AffineTransformationTypeDefinition,
    ):
        super().__init__()
        self.transformation_parameters = Parameter(
            zeros(
                (
                    1,
                    calculate_n_parameters(
                        n_dims=n_dims,
                        transformation_type=transformation_type,
                    ),
                ),
            )
        )

    def forward(self):
        """Return the parameters."""
        return self.transformation_parameters


class SplineParameters(Module):
    """Spline parameters."""

    def __init__(self, n_dims: int, spatial_shape: Sequence[int]):
        super().__init__()
        self.transformation_parameters = Parameter(
            zeros(
                (
                    1,
                    n_dims,
                    *spatial_shape,
                )
            )
        )

    def forward(self):
        """Return the parameters."""
        return self.transformation_parameters
