"""Affine transformation related algorithms"""

from math import sqrt

import jax.numpy as jnp
import jax.scipy as jsp
from jaxmorph.affine_transformation.matrix import (
    embed_matrix,
    generate_scale_matrix,
    generate_translation_matrix,
)


class AffineTransformationTypeDefinition:
    """Affine transformation type definition

    Corresponds to the parametrization:
    Kaji, Shizuo, and Hiroyuki Ochiai. "A concise parametrization of affine transformation." (2016)

    Arguments:
        translation: Translation included
        rotation: Rotation included
        scale: Scale included
        shear: Shear included
    """

    def __init__(self, translation: bool, rotation: bool, scale: bool, shear: bool) -> None:
        self.translation = translation
        self.rotation = rotation
        self.scale = scale
        self.shear = shear

    @staticmethod
    def full():
        """Generate full affine transformation"""
        return AffineTransformationTypeDefinition(True, True, True, True)

    @staticmethod
    def rigid():
        """Generate rigid transformation"""
        return AffineTransformationTypeDefinition(True, True, False, False)

    @staticmethod
    def only_translation():
        """Generate translation only transformation"""
        return AffineTransformationTypeDefinition(True, False, False, False)

    @staticmethod
    def only_rotation():
        """Generate rotation only transformation"""
        return AffineTransformationTypeDefinition(False, True, False, False)

    @staticmethod
    def only_scale():
        """Generate shear only transformation"""
        return AffineTransformationTypeDefinition(False, False, True, False)

    @staticmethod
    def only_shear():
        """Generate shear only transformation"""
        return AffineTransformationTypeDefinition(False, False, False, True)


def calculate_n_parameters(
    n_dims: int, transformation_type: AffineTransformationTypeDefinition
) -> int:
    """Calculate number of parameters from number of dims

    Args:
        n_dims: Number of dimensions
    """
    return int(
        int(transformation_type.translation) * n_dims
        + int(transformation_type.rotation) * (n_dims**2 - n_dims) / 2
        + int(transformation_type.scale) * n_dims
        + int(transformation_type.shear) * (n_dims**2 - n_dims) / 2
    )


def _calculate_n_dims(
    n_parameters: int, transformation_type: AffineTransformationTypeDefinition
) -> int:
    if transformation_type.rotation or transformation_type.shear:
        square_coefficient = int(transformation_type.rotation) + int(transformation_type.shear)
        linear_coffecient = (
            2 * int(transformation_type.translation)
            - int(transformation_type.rotation)
            + 2 * int(transformation_type.scale)
            - int(transformation_type.shear)
        )
        n_dims = (
            -linear_coffecient + sqrt(linear_coffecient**2 + 8 * square_coefficient * n_parameters)
        ) / (2 * square_coefficient)
    elif transformation_type.translation or transformation_type.scale:
        n_dims = n_parameters / (
            int(transformation_type.translation) + int(transformation_type.scale)
        )
    else:
        raise ValueError("At least one transformation type must be True.")
    if n_dims % 1 != 0:
        raise ValueError("Could not infer dimensionality")
    return int(n_dims)


def _generate_rotation_matrix(rotations: jnp.ndarray) -> jnp.ndarray:
    batch_size = rotations.shape[0]
    n_dims = _calculate_n_dims(
        rotations.shape[1], AffineTransformationTypeDefinition.only_rotation()
    )
    triu_rows, triu_cols = jnp.triu_indices(n=n_dims, m=n_dims, k=1)
    tril_rows, tril_cols = jnp.tril_indices(n=n_dims, m=n_dims, k=-1)
    non_diagonal_rows = jnp.concatenate((triu_rows, tril_rows))
    non_diagonal_cols = jnp.concatenate((triu_cols, tril_cols))
    log_rotation_matrix = jnp.zeros((batch_size, n_dims, n_dims), dtype=rotations.dtype)
    log_rotation_matrix = log_rotation_matrix.at[:, non_diagonal_rows, non_diagonal_cols].set(
        jnp.concatenate((rotations, -rotations), axis=1)
    )
    rotation_matrix = jsp.linalg.expm(log_rotation_matrix)
    return rotation_matrix


def _generate_scale_and_shear_matrix(
    scales_and_shears: jnp.ndarray,
) -> jnp.ndarray:
    batch_size = scales_and_shears.shape[0]
    n_dims = _calculate_n_dims(
        scales_and_shears.shape[1],
        AffineTransformationTypeDefinition(
            translation=False, rotation=False, scale=True, shear=True
        ),
    )
    diag_rows, diag_cols = jnp.diag_indices(n_dims)
    triu_rows, triu_cols = jnp.triu_indices(n=n_dims, m=n_dims, k=1)
    tril_rows, tril_cols = jnp.tril_indices(n=n_dims, m=n_dims, k=-1)
    non_diagonal_rows = jnp.concatenate((triu_rows, tril_rows))
    non_diagonal_cols = jnp.concatenate((triu_cols, tril_cols))
    diagonal = scales_and_shears[:, :n_dims]
    off_diagonal = scales_and_shears[:, n_dims:]
    log_scale_and_shear_matrix = jnp.empty(
        (batch_size, n_dims, n_dims), dtype=scales_and_shears.dtype
    )
    log_scale_and_shear_matrix = log_scale_and_shear_matrix.at[:, diag_rows, diag_cols].set(
        diagonal
    )
    log_scale_and_shear_matrix = log_scale_and_shear_matrix.at[
        :, non_diagonal_rows, non_diagonal_cols
    ].set(jnp.concatenate((off_diagonal, off_diagonal), axis=1))
    scale_and_shear_matrix = jsp.linalg.expm(log_scale_and_shear_matrix)
    return scale_and_shear_matrix


def _update_transformation(
    transformation: jnp.ndarray | None, new_transformation: jnp.ndarray, inverse: bool
) -> jnp.ndarray | None:
    if transformation is not None:
        if transformation.shape != new_transformation.shape:
            transformation = embed_matrix(transformation, new_transformation.shape[1:])
        if inverse:
            transformation = jnp.matmul(new_transformation, transformation)
        else:
            transformation = jnp.matmul(transformation, new_transformation)  # type: ignore
    else:
        transformation = new_transformation
    return transformation


def generate_affine_transformation_matrix(
    parameters: jnp.ndarray,
    transformation_type: AffineTransformationTypeDefinition,
    inverse: bool = False,
) -> jnp.ndarray:
    """Generates affine transformation matrix from correspoding
    euclidean space

    When translation, rotation, and shear are all True:
    For n_dims == 2, n_params = 2 + 1 + 3 = 6
    For n_dims == 3, n_params = 3 + 3 + 6 = 12

    Args:
        parameters: Tensor with shape (batch_size, n_params)
        transformation_type: Type of affine transformation matrix to generate

    Returns: Tensor with shape (batch_size, n_dims + 1, n_dims + 1, ...)
    """
    if inverse:
        parameters = -parameters
    n_dims = _calculate_n_dims(parameters.shape[1], transformation_type)
    transformation: jnp.ndarray | None = None
    n_parameters_used = 0
    if transformation_type.shear:
        if not transformation_type.scale:
            raise NotImplementedError(
                "Used parametrization method does not allow generating only shear without scaling"
            )
        n_scale_and_shear_params = calculate_n_parameters(
            n_dims,
            AffineTransformationTypeDefinition(
                translation=False, rotation=False, scale=True, shear=True
            ),
        )
        scale_and_shear_parameters = parameters[:, :n_scale_and_shear_params]
        transformation = _update_transformation(
            transformation,
            _generate_scale_and_shear_matrix(
                scale_and_shear_parameters,
            ),
            inverse,
        )
        n_parameters_used += n_scale_and_shear_params
    elif transformation_type.scale:
        n_scale_params = calculate_n_parameters(
            n_dims, AffineTransformationTypeDefinition.only_scale()
        )
        scale_parameters = parameters[:, :n_scale_params]
        transformation = _update_transformation(
            transformation,
            generate_scale_matrix(scale_parameters),
            inverse,
        )
        n_parameters_used += n_scale_params
    if transformation_type.rotation:
        n_rotation_params = calculate_n_parameters(
            n_dims, AffineTransformationTypeDefinition.only_rotation()
        )
        rotation_parameters = parameters[
            :, n_parameters_used : n_parameters_used + n_rotation_params
        ]
        transformation = _update_transformation(
            transformation,
            _generate_rotation_matrix(rotation_parameters),
            inverse,
        )
        n_parameters_used += n_rotation_params
    if transformation_type.translation:
        translation_parameters = parameters[:, n_parameters_used:]
        transformation = _update_transformation(
            transformation, generate_translation_matrix(translation_parameters), inverse
        )
    if transformation is None:
        raise RuntimeError("Empty transformation is not allowed")
    affine_transformation_matrix = embed_matrix(transformation, [n_dims + 1, n_dims + 1])
    return affine_transformation_matrix
