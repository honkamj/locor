"""Transformation parametrizations defined as modules in order to use DistributedDataParallel."""

from typing import Sequence

import jax.numpy as jnp

from .affine_transformation import (
    AffineTransformationTypeDefinition,
    calculate_n_parameters,
)


def initialize_affine_transformation_parameters(
    n_dims: int,
    transformation_type: AffineTransformationTypeDefinition,
) -> jnp.ndarray:
    """Initialize affine transformation parameters."""
    return jnp.zeros(
        (
            1,
            calculate_n_parameters(
                n_dims=n_dims,
                transformation_type=transformation_type,
            ),
        ),
    )


def initialize_spline_parameters(
    n_dims: int,
    spatial_shape: Sequence[int],
) -> jnp.ndarray:
    """Initialize spline parameters."""
    return jnp.zeros(
        (
            1,
            n_dims,
            *spatial_shape,
        )
    )
