"""Affine transformation related algorithms"""

from math import sqrt
from typing import List, Optional, Tuple

from torch import Tensor, cat, diag_embed, eye, matmul
from torch import matrix_exp as torch_matrix_exp
from torch import ones, tril_indices, triu_indices, zeros
from torch.jit import script


@script
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


@script
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


def _embed_transformation(matrix: Tensor, target_shape: List[int]) -> Tensor:
    if len(target_shape) != 2:
        raise ValueError("Matrix shape must be two dimensional.")
    matrix = _move_channels_last(matrix, 2)
    matrix, batch_dimensions_shape = _merge_batch_dimensions(matrix, 2)
    batch_size = matrix.size(0)
    n_rows_needed = target_shape[0] - matrix.size(1)
    n_cols_needed = target_shape[1] - matrix.size(2)
    if n_rows_needed == 0 and n_cols_needed == 0:
        return matrix
    rows = cat(
        [
            zeros(
                n_rows_needed,
                min(matrix.size(2), matrix.size(1)),
                device=matrix.device,
                dtype=matrix.dtype,
            ),
            eye(
                n_rows_needed,
                max(0, matrix.size(2) - matrix.size(1)),
                device=matrix.device,
                dtype=matrix.dtype,
            ),
        ],
        dim=1,
    ).expand(batch_size, -1, -1)
    cols = cat(
        [
            zeros(
                min(target_shape[0], matrix.size(2)),
                n_cols_needed,
                device=matrix.device,
                dtype=matrix.dtype,
            ),
            eye(
                max(0, target_shape[0] - matrix.size(2)),
                n_cols_needed,
                device=matrix.device,
                dtype=matrix.dtype,
            ),
        ],
        dim=0,
    ).expand(batch_size, -1, -1)
    embedded_matrix = cat([cat([matrix, rows], dim=1), cols], dim=2)
    embedded_matrix = _unmerge_batch_dimensions(
        embedded_matrix, batch_dimensions_shape=batch_dimensions_shape, num_channel_dims=2
    )
    return _move_channels_first(embedded_matrix, 2)


def _convert_to_homogenous_coordinates(coordinates: Tensor) -> Tensor:
    coordinates = _move_channels_last(coordinates)
    coordinates, batch_dimensions_shape = _merge_batch_dimensions(coordinates)
    homogenous_coordinates = cat(
        [
            coordinates,
            ones(1, device=coordinates.device, dtype=coordinates.dtype).expand(
                coordinates.size(0), 1
            ),
        ],
        dim=-1,
    )
    homogenous_coordinates = _unmerge_batch_dimensions(
        homogenous_coordinates, batch_dimensions_shape=batch_dimensions_shape
    )
    return _move_channels_first(homogenous_coordinates)


def _generate_translation_matrix(translations: Tensor) -> Tensor:
    translations = _move_channels_last(translations)
    translations, batch_dimensions_shape = _merge_batch_dimensions(translations)
    batch_size = translations.size(0)
    n_dims = translations.size(1)
    homogenous_translation = _convert_to_homogenous_coordinates(coordinates=translations)
    translation_matrix = cat(
        [
            cat(
                [
                    eye(n_dims, device=translations.device, dtype=translations.dtype),
                    zeros(1, n_dims, device=translations.device, dtype=translations.dtype),
                ],
                dim=0,
            ).expand(batch_size, -1, -1),
            homogenous_translation[..., None],
        ],
        dim=2,
    ).view(-1, n_dims + 1, n_dims + 1)
    translation_matrix = _unmerge_batch_dimensions(
        translation_matrix, batch_dimensions_shape=batch_dimensions_shape, num_channel_dims=2
    )
    return _move_channels_first(translation_matrix, 2)


def _generate_rotation_matrix(rotations: Tensor) -> Tensor:
    rotations = _move_channels_last(rotations)
    rotations, batch_dimensions_shape = _merge_batch_dimensions(rotations)
    batch_size = rotations.size(0)
    n_dims = _calculate_n_dims(
        rotations.size(1), AffineTransformationTypeDefinition.only_rotation()
    )
    non_diagonal_indices = cat(
        (triu_indices(n_dims, n_dims, 1), tril_indices(n_dims, n_dims, -1)), dim=1
    )
    log_rotation_matrix = zeros(
        batch_size, n_dims, n_dims, device=rotations.device, dtype=rotations.dtype
    )
    log_rotation_matrix[:, non_diagonal_indices[0], non_diagonal_indices[1]] = cat(
        (rotations, -rotations), dim=1
    )
    rotation_matrix = torch_matrix_exp(log_rotation_matrix)
    rotation_matrix = _unmerge_batch_dimensions(
        rotation_matrix, batch_dimensions_shape=batch_dimensions_shape, num_channel_dims=2
    )
    return _move_channels_first(rotation_matrix, 2)


def _generate_scale_and_shear_matrix(
    scales_and_shears: Tensor,
) -> Tensor:
    scales_and_shears = _move_channels_last(scales_and_shears)
    scales_and_shears, batch_dimensions_shape = _merge_batch_dimensions(scales_and_shears)
    n_dims = _calculate_n_dims(
        scales_and_shears.size(1),
        AffineTransformationTypeDefinition(
            translation=False, rotation=False, scale=True, shear=True
        ),
    )
    non_diagonal_indices = cat(
        (triu_indices(n_dims, n_dims, 1), tril_indices(n_dims, n_dims, -1)), dim=1
    )
    diagonal = scales_and_shears[:, :n_dims]
    off_diagonal = scales_and_shears[:, n_dims:]
    log_scale_and_shear_matrix = diag_embed(diagonal)
    log_scale_and_shear_matrix[:, non_diagonal_indices[0], non_diagonal_indices[1]] = cat(
        (off_diagonal, off_diagonal), dim=1
    )
    scale_and_shear_matrix = torch_matrix_exp(log_scale_and_shear_matrix)
    scale_and_shear_matrix = _unmerge_batch_dimensions(
        scale_and_shear_matrix, batch_dimensions_shape=batch_dimensions_shape, num_channel_dims=2
    )
    return _move_channels_first(scale_and_shear_matrix, 2)


def _generate_scale_matrix(
    scales: Tensor,
) -> Tensor:
    scales = _move_channels_last(scales)
    scale_matrix = diag_embed(scales.exp())
    return _move_channels_first(scale_matrix, num_channel_dims=2)


def _update_transformation(
    transformation: Optional[Tensor], new_transformation: Tensor, inverse: bool
) -> Optional[Tensor]:
    if transformation is not None:
        if transformation.shape != new_transformation.shape:
            transformation = _embed_transformation(
                transformation, list(new_transformation.shape[1:])
            )
        if inverse:
            transformation = matmul(new_transformation, transformation)
        else:
            transformation = matmul(transformation, new_transformation)  # type: ignore
    else:
        transformation = new_transformation
    return transformation


def _move_channels_first(tensor: Tensor, num_channel_dims: int = 1) -> Tensor:
    if tensor.ndim == num_channel_dims:
        return tensor
    return tensor.permute(
        [0] + list(range(-num_channel_dims, 0)) + list(range(1, tensor.ndim - num_channel_dims))
    )


def _move_channels_last(tensor: Tensor, num_channel_dims: int = 1) -> Tensor:
    if tensor.ndim == num_channel_dims:
        return tensor
    return tensor.permute(
        [0] + list(range(num_channel_dims + 1, tensor.ndim)) + list(range(1, num_channel_dims + 1))
    )


def _merge_batch_dimensions(tensor: Tensor, num_channel_dims: int = 1) -> Tuple[Tensor, List[int]]:
    batch_dimensions_shape = list(tensor.shape[:-num_channel_dims])
    if num_channel_dims == 0:
        channels_shape: List[int] = []
    else:
        channels_shape = list(tensor.shape[-num_channel_dims:])
    return tensor.reshape([-1] + channels_shape), batch_dimensions_shape


def _unmerge_batch_dimensions(
    tensor: Tensor, batch_dimensions_shape: List[int], num_channel_dims: int = 1
) -> Tensor:
    if num_channel_dims == 0:
        channels_shape: List[int] = []
    else:
        channels_shape = list(tensor.shape[-num_channel_dims:])
    return tensor.view(batch_dimensions_shape + channels_shape)


@script
def generate_affine_transformation_matrix(
    parameters: Tensor,
    transformation_type: AffineTransformationTypeDefinition,
    inverse: bool = False,
) -> Tensor:
    """Generates affine transformation matrix from correspoding
    euclidean space

    When translation, rotation, and shear are all True:
    For n_dims == 2, n_params = 2 + 1 + 3 = 6
    For n_dims == 3, n_params = 3 + 3 + 6 = 12

    Args:
        parameters: Tensor with shape (batch_size, n_params, ...)
        transformation_type: Type of affine transformation matrix to generate

    Returns: Tensor with shape (batch_size, n_dims + 1, n_dims + 1, ...)
    """
    if inverse:
        parameters = -parameters
    parameters = _move_channels_last(parameters, 1)
    parameters, batch_dimensions_shape = _merge_batch_dimensions(parameters, 1)
    n_dims = _calculate_n_dims(parameters.size(1), transformation_type)
    transformation: Optional[Tensor] = None
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
            _generate_scale_matrix(scale_parameters),
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
            transformation, _generate_translation_matrix(translation_parameters), inverse
        )
    if transformation is None:
        raise RuntimeError("Emtpy transformation is not allowed")
    affine_transformation_matrix = _embed_transformation(transformation, [n_dims + 1, n_dims + 1])
    affine_transformation_matrix = _unmerge_batch_dimensions(
        affine_transformation_matrix,
        batch_dimensions_shape=batch_dimensions_shape,
        num_channel_dims=2,
    )
    return _move_channels_first(affine_transformation_matrix, 2)
