"""Obtain coordinate system excluding areas outside mask."""

from typing import Iterable, Sequence

import jax.numpy as jnp
from jaxmorph import CoordinateSystem


def optimal_coordinates(
    masks_and_paddings: Iterable[tuple[jnp.ndarray, int | None]],
    original_coordinates: CoordinateSystem,
) -> CoordinateSystem:
    """Obtain coordinate system excluding areas outside mask."""
    original_shape = original_coordinates.spatial_shape
    combined_bounding_box = [(0, dim_size - 1) for dim_size in original_shape]
    for mask, padding in masks_and_paddings:
        if padding is None:
            continue
        bounding_box = [
            (
                min_idx - padding,
                max_idx + padding,
            )
            for min_idx, max_idx in _get_bounding_box(mask)
        ]
        combined_bounding_box = [
            (
                max(min_idx, min_combined_idx),
                min(max_idx, max_combined_idx),
            )
            for (min_combined_idx, max_combined_idx), (min_idx, max_idx) in zip(
                combined_bounding_box, bounding_box
            )
        ]
    bounding_box_shape = tuple(
        (max_idx - min_idx + 1) for min_idx, max_idx in combined_bounding_box
    )
    return original_coordinates.reformat(
        spatial_shape=bounding_box_shape,
        reference=[min_idx for (min_idx, _max_index) in combined_bounding_box],
        target_reference=0,
    )


def _get_bounding_box(mask: jnp.ndarray) -> Sequence[tuple[int, int]]:
    bounding_box = []
    for dim in range(mask.ndim):
        indices = jnp.nonzero(jnp.any(mask, axis=tuple(i for i in range(mask.ndim) if i != dim)))[0]
        if indices.size == 0:
            raise ValueError(
                "Mask is empty. This could be due to the images not overlapping initially. "
                "If that is the case, consider using the --initialize-at-center option, or "
                "perform initial alignment with some other method."
            )
        min_idx = int(indices.min())
        max_idx = int(indices.max())
        bounding_box.append((min_idx, max_idx))
    return bounding_box
