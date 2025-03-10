"""Local least squares"""

from composable_mapping import (
    CoordinateSystem,
    GridComposableMapping,
    ISampler,
    samplable_volume,
)
from composable_mapping.util import combine_optional_masks
from torch import Tensor, arange, cat, diag, empty, full, triu_indices
from torch.linalg import solve_ex


def local_least_squares_error(
    source: GridComposableMapping,
    target: GridComposableMapping,
    sampler: ISampler,
    coordinates: CoordinateSystem,
    regularization: int | float | None = None,
    eps: int | float = 1e-4,
) -> Tensor:
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
    n_source_features = source_image.size(1)
    n_target_features = target_image.size(1)

    source_product_indices_1, source_product_indices_2 = triu_indices(
        n_source_features, n_source_features, offset=1, device=source.device
    )
    n_source_products = source_product_indices_1.size(0)
    source_indices = arange(n_source_features, device=source.device)

    moving_averages = (
        samplable_volume(
            cat(
                (
                    source_image[:, source_product_indices_1]
                    * source_image[:, source_product_indices_2],
                    (source_image[:, :, None] * target_image[:, None, :]).view(
                        source_sampled.batch_shape[0], -1, *source_sampled.spatial_shape
                    ),
                    source_image**2,
                    target_image**2,
                ),
                dim=1,
            ),
            coordinate_system=source.coordinate_system,
            sampler=sampler,
        )
        .sample_to(coordinates)
        .generate_values()
    ).moveaxis(1, -1)

    (
        source_product_avg,
        source_target_avg,
        source_squared_avg,
        target_squared_avg,
    ) = moving_averages.split(
        [
            n_source_products,
            n_source_features * n_target_features,
            n_source_features,
            n_target_features,
        ],
        dim=-1,
    )
    avg_spatial_shape = moving_averages.shape[1:-1]
    source_target_avg = source_target_avg.view(
        source_sampled.batch_shape[0],
        *avg_spatial_shape,
        n_source_features,
        n_target_features,
    )
    least_squares_matrix = empty(
        source_sampled.batch_shape[0],
        *avg_spatial_shape,
        n_source_features,
        n_source_features,
        device=source.device,
        dtype=source.dtype,
    )
    least_squares_matrix[..., source_product_indices_1, source_product_indices_2] = (
        source_product_avg
    )
    least_squares_matrix[..., source_product_indices_2, source_product_indices_1] = (
        source_product_avg
    )
    least_squares_matrix[..., source_indices, source_indices] = source_squared_avg
    if regularization is not None:
        regularization_diagonal = full(
            (n_source_features,), regularization, dtype=source.dtype, device=source.device
        )
        least_squares_matrix = least_squares_matrix + diag(regularization_diagonal).view(
            1, *(1 for _ in avg_spatial_shape), n_source_features, n_source_features
        )
    weights, _ = solve_ex(least_squares_matrix, source_target_avg)  # pylint:disable=not-callable

    loss = (
        (
            (
                (
                    2
                    * weights[..., source_product_indices_1, :]
                    * weights[..., source_product_indices_2, :]
                    * source_product_avg[..., None]
                ).sum(dim=-2)
                - (2 * weights * source_target_avg).sum(dim=-2)
                + (weights**2 * source_squared_avg[..., None]).sum(dim=-2)
                + target_squared_avg
            )
            + eps
        ).log()
    ).mean(dim=-1)
    return loss[:, None]
