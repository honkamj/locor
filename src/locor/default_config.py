"""Default configuration for the image registration tool.

Configurations for the image registration tool are defined as Python files that
include a function with name `build_config` that is then called to dynamically
build the configuration parameters based on the inputs.
"""

from locor.affine_transformation import AffineTransformationTypeDefinition
from locor.config_parameters import (
    AffineStageParameters,
    ConfigBuildingArguments,
    DenseStageParameters,
    FeatureExtractionParameters,
    ImageParameters,
    RegistrationParameters,
    RegularizationParameters,
)
from locor.regularization import BendingEnergy


def build_config(arguments: ConfigBuildingArguments) -> RegistrationParameters:
    """Build default configuration for the image registration tool."""
    feature_learning_rate = 1e-2
    n_dims = len(arguments.reference.coordinate_system.spatial_shape)
    similarity_sliding_window_stride = [3] * n_dims

    reference_voxel_spacing = arguments.reference.coordinate_system.grid_spacing_cpu()
    moving_voxel_spacing = arguments.moving.coordinate_system.grid_spacing_cpu()

    # Regularization base weight is scaled by the square of the image sampling
    # grid spacing for each stage to account for the larger voxel size causing
    # reduced relative bending energy weight.
    min_voxel_spacing_reference = reference_voxel_spacing.amin().item()

    regularization_base_weight = 5.0e2
    regularization_loss = BendingEnergy()

    feature_extraction_parameters = FeatureExtractionParameters(
        n_features=4,
        n_hidden_features=[16, 16],
    )

    return RegistrationParameters(
        feature_extraction_parameters_reference=feature_extraction_parameters,
        feature_extraction_parameters_moving=feature_extraction_parameters,
        affine_stage_parameters=AffineStageParameters(
            n_iterations=80,
            feature_learning_rate=feature_learning_rate,
            transformation_learning_rate=5e-3,
            reference_image_parameters=ImageParameters(
                image_sampling_spacing=2 * reference_voxel_spacing,
                similarity_sliding_window_stride=similarity_sliding_window_stride,
                similarity_sliding_window_std=2 * reference_voxel_spacing,
            ),
            moving_image_parameters=ImageParameters(
                image_sampling_spacing=2 * moving_voxel_spacing,
                similarity_sliding_window_stride=similarity_sliding_window_stride,
                similarity_sliding_window_std=2 * moving_voxel_spacing,
            ),
            transformation_type=AffineTransformationTypeDefinition.full(),
        ),
        dense_stage_parameters=[
            DenseStageParameters(
                n_iterations=80,
                feature_learning_rate=feature_learning_rate,
                deformation_learning_rate=16e-2,
                spline_grid_spacing=32 * reference_voxel_spacing,
                deformation_sampling_spacing=4 * reference_voxel_spacing,
                reference_image_parameters=ImageParameters(
                    image_sampling_spacing=4 * reference_voxel_spacing,
                    similarity_sliding_window_stride=similarity_sliding_window_stride,
                    similarity_sliding_window_std=4 * reference_voxel_spacing,
                ),
                moving_image_parameters=ImageParameters(
                    image_sampling_spacing=4 * moving_voxel_spacing,
                    similarity_sliding_window_stride=similarity_sliding_window_stride,
                    similarity_sliding_window_std=4 * moving_voxel_spacing,
                ),
                reference_regularization_parameters=RegularizationParameters(
                    weight=regularization_base_weight * (4 * min_voxel_spacing_reference) ** 2,
                    loss=regularization_loss,
                ),
                moving_regularization_parameters=None,
            ),
            DenseStageParameters(
                n_iterations=80,
                feature_learning_rate=feature_learning_rate,
                deformation_learning_rate=16e-2,
                spline_grid_spacing=16 * reference_voxel_spacing,
                deformation_sampling_spacing=4 * reference_voxel_spacing,
                reference_image_parameters=ImageParameters(
                    image_sampling_spacing=4 * reference_voxel_spacing,
                    similarity_sliding_window_stride=similarity_sliding_window_stride,
                    similarity_sliding_window_std=4 * reference_voxel_spacing,
                    sampling_coordinates_padding=10,
                ),
                moving_image_parameters=ImageParameters(
                    image_sampling_spacing=4 * moving_voxel_spacing,
                    similarity_sliding_window_stride=similarity_sliding_window_stride,
                    similarity_sliding_window_std=4 * moving_voxel_spacing,
                    sampling_coordinates_padding=10,
                ),
                reference_regularization_parameters=RegularizationParameters(
                    weight=regularization_base_weight * (4 * min_voxel_spacing_reference) ** 2,
                    loss=regularization_loss,
                ),
                moving_regularization_parameters=None,
            ),
            DenseStageParameters(
                n_iterations=40,
                feature_learning_rate=feature_learning_rate,
                deformation_learning_rate=8e-2,
                spline_grid_spacing=8 * reference_voxel_spacing,
                deformation_sampling_spacing=2 * reference_voxel_spacing,
                reference_image_parameters=ImageParameters(
                    image_sampling_spacing=2 * reference_voxel_spacing,
                    similarity_sliding_window_stride=similarity_sliding_window_stride,
                    similarity_sliding_window_std=2 * reference_voxel_spacing,
                    sampling_coordinates_padding=10,
                ),
                moving_image_parameters=ImageParameters(
                    image_sampling_spacing=2 * moving_voxel_spacing,
                    similarity_sliding_window_stride=similarity_sliding_window_stride,
                    similarity_sliding_window_std=2 * moving_voxel_spacing,
                    sampling_coordinates_padding=10,
                ),
                reference_regularization_parameters=RegularizationParameters(
                    weight=regularization_base_weight * (2 * min_voxel_spacing_reference) ** 2,
                    loss=regularization_loss,
                ),
                moving_regularization_parameters=None,
            ),
            DenseStageParameters(
                n_iterations=40,
                feature_learning_rate=feature_learning_rate,
                deformation_learning_rate=8e-2,
                spline_grid_spacing=4 * reference_voxel_spacing,
                deformation_sampling_spacing=2 * reference_voxel_spacing,
                reference_image_parameters=ImageParameters(
                    image_sampling_spacing=2 * reference_voxel_spacing,
                    similarity_sliding_window_stride=similarity_sliding_window_stride,
                    similarity_sliding_window_std=2 * reference_voxel_spacing,
                    sampling_coordinates_padding=10,
                ),
                moving_image_parameters=ImageParameters(
                    image_sampling_spacing=2 * moving_voxel_spacing,
                    similarity_sliding_window_stride=similarity_sliding_window_stride,
                    similarity_sliding_window_std=2 * moving_voxel_spacing,
                    sampling_coordinates_padding=10,
                ),
                reference_regularization_parameters=RegularizationParameters(
                    weight=regularization_base_weight * (2 * min_voxel_spacing_reference) ** 2,
                    loss=regularization_loss,
                ),
                moving_regularization_parameters=None,
            ),
            DenseStageParameters(
                n_iterations=40,
                feature_learning_rate=feature_learning_rate,
                deformation_learning_rate=4e-2,
                spline_grid_spacing=4 * reference_voxel_spacing,
                deformation_sampling_spacing=1 * reference_voxel_spacing,
                reference_image_parameters=ImageParameters(
                    image_sampling_spacing=1 * reference_voxel_spacing,
                    similarity_sliding_window_stride=similarity_sliding_window_stride,
                    similarity_sliding_window_std=1 * reference_voxel_spacing,
                    sampling_coordinates_padding=5,
                ),
                moving_image_parameters=ImageParameters(
                    image_sampling_spacing=1 * moving_voxel_spacing,
                    similarity_sliding_window_stride=similarity_sliding_window_stride,
                    similarity_sliding_window_std=1 * moving_voxel_spacing,
                    sampling_coordinates_padding=5,
                ),
                reference_regularization_parameters=RegularizationParameters(
                    weight=regularization_base_weight * (1 * min_voxel_spacing_reference) ** 2,
                    loss=regularization_loss,
                ),
                moving_regularization_parameters=None,
            ),
        ],
    )
