"""Configuration parameters for the image registration tool."""

from dataclasses import dataclass
from typing import Callable, Sequence

from composable_mapping import GridComposableMapping, SamplableVolume
from torch import Tensor

from .affine_transformation import AffineTransformationTypeDefinition


@dataclass
class ImageParameters:
    """Configuration parameters for a single stage related to either reference
    or moving image."""

    image_sampling_spacing: Sequence[int | float] | Tensor
    similarity_sliding_window_stride: Sequence[int] | Tensor
    similarity_sliding_window_std: Sequence[int | float] | Tensor

    sampling_coordinates_padding: int | None = None

    matrix_solve_epsilon: int | float = 1e-4
    similarity_logarithm_epsilon: int | float = 1e-4
    truncate_sliding_window_at_n_stds: int | float = 3.0
    truncate_image_smoothing_at_n_stds: int | float = 4.0


@dataclass
class RegularizationParameters:
    """Configuration parameters for the regularization."""

    weight: int | float
    loss: Callable[[GridComposableMapping], Tensor]


@dataclass
class AffineStageParameters:
    """Configuration parameters for the affine stage."""

    n_iterations: int
    feature_learning_rate: int | float
    transformation_learning_rate: int | float

    reference_image_parameters: ImageParameters
    moving_image_parameters: ImageParameters

    transformation_type: AffineTransformationTypeDefinition


@dataclass
class DenseStageParameters:
    """Configuration parameters for a single dense stage."""

    n_iterations: int
    feature_learning_rate: int | float
    deformation_learning_rate: int | float
    deformation_sampling_spacing: Sequence[int | float] | Tensor

    spline_grid_spacing: Sequence[int | float] | Tensor

    reference_image_parameters: ImageParameters
    moving_image_parameters: ImageParameters

    reference_regularization_parameters: RegularizationParameters | None
    moving_regularization_parameters: RegularizationParameters | None

    n_scalings_and_squarings: int = 6


@dataclass
class FeatureExtractionParameters:
    """Feature extraction parameters."""

    n_features: int
    n_hidden_features: Sequence[int]


@dataclass
class RegistrationParameters:
    """Configuration parameters for the image registration tool."""

    feature_extraction_parameters_reference: FeatureExtractionParameters
    feature_extraction_parameters_moving: FeatureExtractionParameters

    affine_stage_parameters: AffineStageParameters | None
    dense_stage_parameters: Sequence[DenseStageParameters]


@dataclass
class ConfigBuildingArguments:
    """Arguments for building the configuration parameters dynamically."""

    reference: SamplableVolume
    moving: SamplableVolume
