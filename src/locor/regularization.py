"""Regularization functions for image registration."""

from itertools import combinations_with_replacement

import jax.numpy as jnp
import numpy as np
from jaxmorph import (
    CoordinateSystem,
    GridComposableMapping,
    LimitDirection,
    LinearInterpolator,
    OriginalShape,
    estimate_coordinate_mapping_spatial_derivatives,
    mappable,
    samplable_volume,
    stack_mappable_tensors,
)

Array = jnp.ndarray | np.ndarray


class BendingEnergy:
    """Bending energy penalty for registration."""

    def __call__(
        self,
        mapping: GridComposableMapping,
    ) -> Array:
        jacobian_coordinates = mapping.coordinate_system.reformat(spatial_shape=OriginalShape() - 2)
        jacobian_matrices = self._estimate_jacobians(
            mapping,
            derivation_coordinates=jacobian_coordinates,
            limit_direction=LimitDirection.average(),
        )
        return self._affinity(
            jacobian_matrices,
            jacobian_coordinates=jacobian_coordinates,
            derivation_coordinates=jacobian_coordinates.reformat(spatial_shape=OriginalShape() - 2),
            limit_direction=LimitDirection.average(),
        ).mean()

    @staticmethod
    def _estimate_jacobians(
        deformation: GridComposableMapping,
        derivation_coordinates: CoordinateSystem,
        limit_direction: LimitDirection,
    ) -> Array:
        return stack_mappable_tensors(
            *(
                estimate_coordinate_mapping_spatial_derivatives(
                    deformation,
                    spatial_dim=spatial_dim,
                    target=derivation_coordinates,
                    sampler=LinearInterpolator(
                        mask_extrapolated_regions=False, limit_direction=limit_direction
                    ),
                )
                for spatial_dim in range(len(deformation.coordinate_system.spatial_shape))
            ),
            channel_index=-1,
        ).generate_values()

    @staticmethod
    def _affinity(
        jacobian_matrices: Array,
        jacobian_coordinates: CoordinateSystem,
        derivation_coordinates: CoordinateSystem,
        limit_direction: LimitDirection,
    ) -> Array:
        n_dims = len(jacobian_coordinates.spatial_shape)
        loss = jnp.zeros(1, dtype=jacobian_matrices.dtype)
        n_terms = n_dims**2
        for dim_to_derive, derived_dim in combinations_with_replacement(range(n_dims), 2):
            derivative = jacobian_matrices[:, :, dim_to_derive]
            gradient_volume = (
                samplable_volume(
                    derivative,
                    coordinate_system=jacobian_coordinates,
                    sampler=LinearInterpolator(
                        limit_direction=limit_direction, mask_extrapolated_regions=False
                    ).derivative(
                        spatial_dim=derived_dim,
                    ),
                ).sample_to(derivation_coordinates)
                / mappable(derivation_coordinates.grid_spacing()[..., None, derived_dim])
            ).generate_values()
            if dim_to_derive == derived_dim:
                loss = loss + jnp.mean(gradient_volume**2, axis=1) / n_terms
            else:
                loss = loss + jnp.mean(2 * gradient_volume**2, axis=1) / n_terms
        return loss
