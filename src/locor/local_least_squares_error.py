"""Local least squares"""

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxmorph import CoordinateSystem, GridComposableMapping, ISampler, samplable_volume
from jaxmorph.util import combine_optional_masks


def local_least_squares_error(
    source: GridComposableMapping,
    target: GridComposableMapping,
    sampler: ISampler,
    coordinates: CoordinateSystem,
    regularization: int | float | None = None,
    eps: int | float = 1e-4,
) -> jnp.ndarray:
    """Compute the local least squares loss between two images.

    Returns:
        The local least squares loss and the local weights if `return_weights` is `True`.
    """
    source_sampled = source.sample()
    target_sampled = target.sample()
    source_image, source_mask = source_sampled.generate(generate_missing_mask=False)
    target_image, target_mask = target_sampled.generate(generate_missing_mask=False)
    mask = combine_optional_masks(source_mask, target_mask)
    if mask is not None:
        source_image = source_image * mask
        target_image = target_image * mask
    n_source_features = source_image.shape[1]
    n_target_features = target_image.shape[1]

    source_product_indices_1, source_product_indices_2 = jnp.triu_indices(
        n=n_source_features, m=n_source_features, k=1
    )
    n_source_products = source_product_indices_1.shape[0]
    source_indices = jnp.arange(n_source_features)

    moving_averages = jnp.moveaxis(
        samplable_volume(
            jnp.concatenate(
                (
                    source_image[:, source_product_indices_1]
                    * source_image[:, source_product_indices_2],
                    (source_image[:, :, None] * target_image[:, None, :]).reshape(
                        source_sampled.batch_shape[0], -1, *source_sampled.spatial_shape
                    ),
                    source_image**2,
                    target_image**2,
                ),
                axis=1,
            ),
            coordinate_system=source.coordinate_system,
            sampler=sampler,
        )
        .sample_to(coordinates)
        .generate_values(),
        1,
        -1,
    )

    (
        source_product_avg,
        source_target_avg,
        source_squared_avg,
        target_squared_avg,
    ) = jnp.split(
        moving_averages,
        np.cumsum(
            [
                n_source_products,
                n_source_features * n_target_features,
                n_source_features,
                n_target_features,
            ]
        )[:-1],
        axis=-1,
    )
    avg_spatial_shape = moving_averages.shape[1:-1]
    source_target_avg = source_target_avg.reshape(
        source_sampled.batch_shape[0],
        *avg_spatial_shape,
        n_source_features,
        n_target_features,
    )
    least_squares_matrix = jnp.empty(
        (source_sampled.batch_shape[0], *avg_spatial_shape, n_source_features, n_source_features),
        dtype=source.dtype,
    )
    least_squares_matrix = least_squares_matrix.at[
        ..., source_product_indices_1, source_product_indices_2
    ].set(source_product_avg)
    least_squares_matrix = least_squares_matrix.at[
        ..., source_product_indices_2, source_product_indices_1
    ].set(source_product_avg)
    least_squares_matrix = least_squares_matrix.at[..., source_indices, source_indices].set(
        source_squared_avg
    )
    if regularization is not None:
        regularization_diagonal = jnp.full((n_source_features,), regularization, dtype=source.dtype)
        least_squares_matrix = least_squares_matrix + jnp.diag(regularization_diagonal)
    weights = jsp.linalg.solve(least_squares_matrix, source_target_avg)

    loss = jnp.mean(
        jnp.log(
            (
                jnp.sum(
                    2
                    * weights[..., source_product_indices_1, :]
                    * weights[..., source_product_indices_2, :]
                    * source_product_avg[..., None],
                    axis=-2,
                )
                - jnp.sum(2 * weights * source_target_avg, axis=-2)
                + jnp.sum(weights**2 * source_squared_avg[..., None], axis=-2)
                + target_squared_avg
            )
            + eps
        ),
        axis=-1,
    )
    return loss[:, None]
