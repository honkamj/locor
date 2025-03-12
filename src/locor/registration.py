"""Registration functions"""

from os import environ
from typing import Any, Sequence

from composable_mapping import (
    Affine,
    Center,
    CoordinateSystem,
    CubicSplineSampler,
    DataFormat,
    GridComposableMapping,
    LimitDirection,
    LinearInterpolator,
    OriginalFOV,
    OriginalShape,
    SamplableVolume,
    clear_sampling_cache,
    no_sampling_cache,
    samplable_volume,
    sampling_cache,
    stack_mappable_tensors,
)
from composable_mapping.util import combine_optional_masks
from torch import Tensor, cat
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import float32, rand, stack, tensor
from torch.cuda import set_device as set_cuda_device
from torch.distributed import init_process_group
from torch.multiprocessing import Pool, set_start_method
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from tqdm import tqdm  # type: ignore

from .bounding_box import optimal_coordinates
from .config_parameters import (
    AffineStageParameters,
    DenseStageParameters,
    ImageParameters,
    RegistrationParameters,
)
from .feature_extractor import FeatureExtractor
from .gaussian_sampler import GaussianSampler
from .local_least_squares_error import local_least_squares_error
from .symmetric_deformation_model import SymmetricDeformationModel
from .transformation_modules import AffineTransformationParameters, SplineParameters


def register(
    reference: GridComposableMapping,
    moving: GridComposableMapping,
    parameters: RegistrationParameters,
    devices: Sequence[torch_device] | None = None,
) -> tuple[GridComposableMapping, GridComposableMapping]:
    """Register the images and return the coordinate mappings in both directions.

    Coordinate mapping is returned in both directions but the non-affine
    component is learned in the coordinates of the reference image.
    """
    if devices is None or not devices:
        devices = [reference.device]
    if len(devices) == 1:
        return tuple(_register(None, reference, moving, parameters, device=devices[0]))
    if len(devices) == 2:
        set_start_method("spawn")
        with Pool(2) as pool:
            return tuple(
                pool.starmap(
                    _register,
                    [
                        (0, reference, moving, parameters, devices[0]),
                        (1, reference, moving, parameters, devices[1]),
                    ],
                )
            )
    raise NotImplementedError(
        "Distributed registration is not implemented for more than two devices."
    )


def _register(
    rank: int | None,
    reference: GridComposableMapping,
    moving: GridComposableMapping,
    parameters: RegistrationParameters,
    device: torch_device,
) -> Any:
    if device.type == "cuda":
        set_cuda_device(device)
    if rank is not None:
        environ.setdefault("MASTER_ADDR", "127.0.0.1")
        environ.setdefault("MASTER_PORT", "29500")
        init_process_group(backend="gloo", rank=rank, world_size=2)
        process_rank_postfix = f" (process {rank})"
    else:
        process_rank_postfix = ""

    reference = _normalize(reference)
    moving = _normalize(moving)

    n_reference_channels = reference.sample().channels_shape[0]
    n_moving_channels = moving.sample().channels_shape[0]

    n_dims = len(reference.coordinate_system.spatial_shape)

    feature_extractors: list[FeatureExtractor] = []
    deformations: list[SymmetricDeformationModel] = []
    full_deformation_references: list[CoordinateSystem] = []
    if rank in (0, None):
        feature_extractors.append(
            FeatureExtractor(
                n_dims=n_dims,
                n_input_channels=2 * n_reference_channels,
                n_hidden_features=(
                    parameters.feature_extraction_parameters_reference.n_hidden_features
                ),
                n_output_channels=parameters.feature_extraction_parameters_moving.n_features,
            ).to(device=device, dtype=reference.dtype)
        )
        deformations.append(SymmetricDeformationModel())
        full_deformation_references.append(reference.coordinate_system.cast(device=device))
    if rank in (1, None):
        feature_extractors.append(
            FeatureExtractor(
                n_dims=n_dims,
                n_input_channels=2 * n_moving_channels,
                n_hidden_features=parameters.feature_extraction_parameters_moving.n_hidden_features,
                n_output_channels=parameters.feature_extraction_parameters_moving.n_features,
            ).to(device=device, dtype=moving.dtype)
        )
        deformations.append(SymmetricDeformationModel(inverse=True))
        full_deformation_references.append(moving.coordinate_system.cast(device=device))

    if parameters.affine_stage_parameters is not None:
        print(f"Starting affine stage{process_rank_postfix}")
        deformations = _register_affine(
            initial_deformations=deformations,
            feature_extractors=feature_extractors,
            reference=reference,
            moving=moving,
            parameters=parameters.affine_stage_parameters,
            device=device,
            rank=rank,
        )

    for index, dense_stage_parameters in enumerate(parameters.dense_stage_parameters):
        print(f"Starting dense stage {index + 1}{process_rank_postfix}")
        deformations = _register_dense(
            initial_deformations=deformations,
            feature_extractors=feature_extractors,
            reference=reference,
            moving=moving,
            parameters=dense_stage_parameters,
            device=device,
            rank=rank,
        )

    output_deformations: list[GridComposableMapping] = []
    deformation_coordinates = reference.coordinate_system.cast(device=device)
    for deformation, full_deformation_reference in zip(deformations, full_deformation_references):
        full_deformation, _ = deformation.build_full_deformation(deformation_coordinates)
        output_deformations.append(full_deformation.assign_coordinates(full_deformation_reference))

    if rank is not None:
        assert len(output_deformations) == 1
        return output_deformations[0]

    return output_deformations


def _register_affine(
    initial_deformations: Sequence[SymmetricDeformationModel],
    feature_extractors: Sequence[FeatureExtractor],
    reference: GridComposableMapping,
    moving: GridComposableMapping,
    parameters: AffineStageParameters,
    device: torch_device,
    rank: int | None,
) -> list[SymmetricDeformationModel]:
    affine_parameters = AffineTransformationParameters(
        n_dims=len(reference.coordinate_system.spatial_shape),
        transformation_type=parameters.transformation_type,
    )
    affine_parameters.to(device=device, dtype=reference.dtype)
    if rank is not None:
        affine_parameters_distributed: Module = DistributedDataParallel(affine_parameters)
        process_rank_postfix = f" (process {rank})"
    else:
        affine_parameters_distributed = affine_parameters
        process_rank_postfix = ""
    optimizer = Adam(
        affine_parameters.parameters(),
        lr=parameters.transformation_learning_rate,
    )
    feature_optimizers: list[Adam] = []
    for feature_extractor in feature_extractors:
        feature_optimizers.append(
            Adam(
                feature_extractor.parameters(),
                lr=parameters.feature_learning_rate,
            )
        )
    registration_inputs = _initialize_registration_stage(
        initial_deformations=initial_deformations,
        reference=reference,
        moving=moving,
        parameters=parameters,
        device=device,
    )
    normalizing_affine, inverse_normalizing_affine = _normalizing_affine(
        moving.coordinate_system.cast(device=device)
    )
    progress_bar = tqdm(range(parameters.n_iterations), position=rank)
    for _ in progress_bar:
        with sampling_cache():
            optimizer.zero_grad()
            for feature_optimizer in feature_optimizers:
                feature_optimizer.zero_grad()
            similarity_losses = []
            for (
                (reference_initialized, moving_initialized, similarity_coordinates),
                feature_extractor,
                initial_deformation,
            ) in zip(registration_inputs, feature_extractors, initial_deformations):
                reference_image_parameters = (
                    parameters.reference_image_parameters,
                    parameters.moving_image_parameters,
                )[initial_deformation.inverse]
                updated_deformation = initial_deformation.set_affine(
                    affine_parameters=affine_parameters_distributed(),
                    affine_transformation_type=parameters.transformation_type,
                    normalizing_affine=normalizing_affine,
                    normalizing_affine_inverse=inverse_normalizing_affine,
                )
                registered_moving = (moving_initialized @ updated_deformation.affine).resample_to(
                    reference_initialized
                )
                registered_moving_values, registered_moving_mask = (
                    registered_moving.sample().generate()
                )
                features_moving = feature_extractor(registered_moving_values)
                similarity_sampler = _similarity_sampler(
                    reference_initialized,
                    reference_image_parameters,
                )
                # We disable the cache since the similarity sampler differs per iteration.
                with no_sampling_cache():
                    similarity_loss = local_least_squares_error(
                        samplable_volume(
                            features_moving,
                            mask=registered_moving_mask,
                            coordinate_system=reference_initialized.coordinate_system,
                        ),
                        reference_initialized,
                        sampler=similarity_sampler,
                        coordinates=similarity_coordinates,
                        regularization=reference_image_parameters.matrix_solve_epsilon,
                        eps=reference_image_parameters.similarity_logarithm_epsilon,
                    ).mean()
                similarity_losses.append(similarity_loss)
            loss = stack(similarity_losses).mean()
            loss.backward()
            optimizer.step()
            for feature_optimizer in feature_optimizers:
                feature_optimizer.step()
            progress_bar.set_description(f"Loss{process_rank_postfix}: {loss.item():.4e}")
    clear_sampling_cache()
    return [
        initial_deformation.set_affine(
            affine_parameters=affine_parameters.transformation_parameters.data,
            affine_transformation_type=parameters.transformation_type,
            normalizing_affine=normalizing_affine,
            normalizing_affine_inverse=inverse_normalizing_affine,
        )
        for initial_deformation in initial_deformations
    ]


def _normalizing_affine(coordinates: CoordinateSystem) -> tuple[Affine, Affine]:
    centered_normalized_coordinates = CoordinateSystem.centered_normalized(
        spatial_shape=coordinates.spatial_shape,
        voxel_size=coordinates.grid_spacing_cpu(),
        dtype=coordinates.dtype,
        device=coordinates.device,
    )
    forward = Affine.from_matrix(
        (
            centered_normalized_coordinates.from_voxel_coordinates
            @ coordinates.to_voxel_coordinates
        ).as_affine_matrix()
    )
    inverse = Affine.from_matrix(
        (
            coordinates.from_voxel_coordinates
            @ centered_normalized_coordinates.to_voxel_coordinates
        ).as_affine_matrix()
    )
    return forward, inverse


def _register_dense(
    initial_deformations: Sequence[SymmetricDeformationModel],
    feature_extractors: Sequence[FeatureExtractor],
    reference: GridComposableMapping,
    moving: GridComposableMapping,
    parameters: DenseStageParameters,
    device: torch_device,
    rank: int | None,
) -> list[SymmetricDeformationModel]:
    deformation_coordinates = reference.coordinate_system.cast(device=device)
    deformation_sampling_coordinates = deformation_coordinates.reformat(
        voxel_size=parameters.deformation_sampling_spacing,
        reference=Center(),
        spatial_shape=OriginalFOV(fitting_method="ceil"),
    )
    spline_svf_coordinates = deformation_coordinates.reformat(
        voxel_size=parameters.spline_grid_spacing,
        reference=Center(),
        spatial_shape=OriginalFOV(fitting_method="ceil") + 1,
    )
    spline_parameters: Module = SplineParameters(
        n_dims=len(reference.coordinate_system.spatial_shape),
        spatial_shape=spline_svf_coordinates.spatial_shape,
    )
    spline_parameters.to(device=device, dtype=reference.dtype)
    if rank is not None:
        spline_parameters_distributed: Module = DistributedDataParallel(spline_parameters)
        process_rank_postfix = f" (process {rank})"
    else:
        spline_parameters_distributed = spline_parameters
        process_rank_postfix = ""
    optimizer = Adam(
        spline_parameters.parameters(),
        lr=parameters.deformation_learning_rate,
    )
    feature_optimizers: list[Adam] = []
    for feature_extractor in feature_extractors:
        feature_optimizers.append(
            Adam(
                feature_extractor.parameters(),
                lr=parameters.feature_learning_rate,
            )
        )
    registration_inputs = _initialize_registration_stage(
        initial_deformations=initial_deformations,
        reference=reference,
        moving=moving,
        parameters=parameters,
        device=device,
    )
    progress_bar = tqdm(range(parameters.n_iterations), position=rank)
    for _ in progress_bar:
        with sampling_cache():
            optimizer.zero_grad()
            for feature_optimizer in feature_optimizers:
                feature_optimizer.zero_grad()
            update_svf = samplable_volume(
                spline_parameters_distributed(),
                coordinate_system=spline_svf_coordinates,
                data_format=DataFormat.voxel_displacements(),
                sampler=CubicSplineSampler(mask_extrapolated_regions=False),
            ).resample_to(
                deformation_sampling_coordinates,
            )
            similarity_losses = []
            regularization_losses = []
            for (
                (reference_initialized, moving_initialized, similarity_coordinates),
                feature_extractor,
                initial_deformation,
            ) in reversed(list(zip(registration_inputs, feature_extractors, initial_deformations))):
                reference_image_parameters = (
                    parameters.reference_image_parameters,
                    parameters.moving_image_parameters,
                )[initial_deformation.inverse]
                reference_regularization_parameters = (
                    parameters.reference_regularization_parameters,
                    parameters.moving_regularization_parameters,
                )[initial_deformation.inverse]
                updated_deformation = initial_deformation.update(
                    update_svf=update_svf,
                    n_scalings_and_squarings=parameters.n_scalings_and_squarings,
                )
                deformation_to_moving, deformation_to_moving_without_affine = (
                    updated_deformation.build_full_deformation(deformation_sampling_coordinates)
                )
                registered_moving = (moving_initialized @ deformation_to_moving).resample_to(
                    reference_initialized
                )
                registered_moving_values, registered_moving_mask = (
                    registered_moving.sample().generate()
                )
                features_moving = feature_extractor(registered_moving_values)
                similarity_sampler = _similarity_sampler(
                    reference_initialized,
                    reference_image_parameters,
                )
                # We disable the cache since the similarity sampler differs per iteration.
                with no_sampling_cache():
                    similarity = local_least_squares_error(
                        samplable_volume(
                            features_moving,
                            mask=registered_moving_mask,
                            coordinate_system=reference_initialized.coordinate_system,
                        ),
                        reference_initialized,
                        sampler=similarity_sampler,
                        coordinates=similarity_coordinates,
                        regularization=reference_image_parameters.matrix_solve_epsilon,
                        eps=reference_image_parameters.similarity_logarithm_epsilon,
                    ).mean()

                if reference_regularization_parameters is not None:
                    regularity = (
                        reference_regularization_parameters.weight
                        * reference_regularization_parameters.loss(
                            deformation_to_moving_without_affine
                        ).mean()
                    )
                else:
                    regularity = tensor(0.0, device=device, dtype=similarity.dtype)

                similarity_losses.append(similarity)
                regularization_losses.append(regularity)
            mean_similarity_loss = stack(similarity_losses).mean()
            mean_regularization_loss = stack(regularization_losses).mean()
            loss = mean_similarity_loss + mean_regularization_loss

            loss.backward()
            optimizer.step()
            for feature_optimizer in feature_optimizers:
                feature_optimizer.step()
            progress_bar.set_description(
                f"Loss{process_rank_postfix}: {loss.item():.4e} "
                f"(sim: {mean_similarity_loss.item():.4e}, "
                f"reg: {float(mean_regularization_loss):.4e})"
            )
    final_update_svf = samplable_volume(
        spline_parameters.transformation_parameters.data,
        coordinate_system=spline_svf_coordinates,
        data_format=DataFormat.voxel_displacements(),
        sampler=CubicSplineSampler(mask_extrapolated_regions=False),
    ).resample_to(
        deformation_coordinates,
    )
    clear_sampling_cache()
    return [
        initial_deformation.update(
            update_svf=final_update_svf,
            n_scalings_and_squarings=parameters.n_scalings_and_squarings,
        ).resample(deformation_coordinates)
        for initial_deformation in initial_deformations
    ]


def _initialize_registration_stage(
    initial_deformations: Sequence[SymmetricDeformationModel],
    reference: GridComposableMapping,
    moving: GridComposableMapping,
    parameters: AffineStageParameters | DenseStageParameters,
    device: torch_device,
) -> Sequence[tuple[GridComposableMapping, GridComposableMapping, CoordinateSystem]]:
    registration_inputs: list[
        tuple[GridComposableMapping, GridComposableMapping, CoordinateSystem]
    ] = []
    for initial_deformation in initial_deformations:
        reference_parameters = (
            parameters.reference_image_parameters
            if not initial_deformation.inverse
            else parameters.moving_image_parameters
        )
        reference_initialized, moving_initialized = _initialize_image_pair(
            reference=reference if not initial_deformation.inverse else moving,
            moving=moving if not initial_deformation.inverse else reference,
            reference_parameters=reference_parameters,
            initial_deformation=initial_deformation,
            deformation_coordinates=reference.coordinate_system,
            device=device,
        )
        similarity_coordinates = _similarity_sampling_coordinates(
            reference_initialized,
            reference_parameters,
        )
        registration_inputs.append(
            (reference_initialized, moving_initialized, similarity_coordinates)
        )
    return registration_inputs


def _initialize_image_pair(
    reference: GridComposableMapping,
    moving: GridComposableMapping,
    reference_parameters: ImageParameters,
    initial_deformation: SymmetricDeformationModel,
    deformation_coordinates: CoordinateSystem,
    device: torch_device,
) -> tuple[GridComposableMapping, GridComposableMapping]:
    reference = reference.cast(device=device)
    moving = moving.cast(device=device)
    deformation_coordinates = deformation_coordinates.cast(device=device)

    reference_smoothed = _smoothed_mapping(
        reference,
        sampling_spacing=_tensor(reference_parameters.image_sampling_spacing, reference.dtype),
        truncate_at_n_stds=reference_parameters.truncate_image_smoothing_at_n_stds,
    )
    moving_smoothed = _smoothed_mapping(
        moving,
        sampling_spacing=_tensor(reference_parameters.image_sampling_spacing, reference.dtype),
        truncate_at_n_stds=reference_parameters.truncate_image_smoothing_at_n_stds,
    )

    deformation_to_moving, _ = initial_deformation.build_full_deformation(deformation_coordinates)
    registered_moving = moving_smoothed @ deformation_to_moving

    moving_smoothed = _restrict_moving_to_masked_region(moving_smoothed)
    reference_resampled = _resample_reference(
        reference_smoothed, registered_moving, reference_parameters
    )

    return reference_resampled, moving_smoothed


def _restrict_moving_to_masked_region(
    moving: GridComposableMapping,
) -> GridComposableMapping:
    restricted_coordinates = optimal_coordinates(
        masks_and_paddings=[(moving.sample().generate_mask()[0, 0], 0)],
        original_coordinates=moving.coordinate_system,
    )
    return moving.resample_to(restricted_coordinates)


def _resample_reference(
    reference: GridComposableMapping,
    registered_moving: GridComposableMapping,
    parameters: ImageParameters,
) -> GridComposableMapping:
    restricted_coordinates = reference.coordinate_system.reformat(
        voxel_size=parameters.image_sampling_spacing,
        reference=Center(),
        spatial_shape=OriginalFOV(fitting_method="ceil"),
    )
    restricted_coordinates = optimal_coordinates(
        masks_and_paddings=[
            (reference.sample_to(restricted_coordinates).generate_mask()[0, 0], 0),
            (
                registered_moving.sample_to(restricted_coordinates).generate_mask()[0, 0],
                parameters.sampling_coordinates_padding,
            ),
        ],
        original_coordinates=restricted_coordinates,
    )
    return reference.resample_to(restricted_coordinates)


def _similarity_sampling_coordinates(
    sampled_reference: GridComposableMapping,
    parameters: ImageParameters,
) -> CoordinateSystem:
    grid_spacing_cpu = sampled_reference.coordinate_system.grid_spacing_cpu()
    stds = (
        _tensor(parameters.similarity_sliding_window_std, sampled_reference.dtype)
        / grid_spacing_cpu
    ).tolist()
    similarity_sampling_coordinates = sampled_reference.coordinate_system.reformat(
        downsampling_factor=parameters.similarity_sliding_window_stride,
        reference=Center(),
        spatial_shape=[
            OriginalFOV(fitting_method="ceil")
            + int((parameters.truncate_sliding_window_at_n_stds * std) // stride)
            for std, stride in zip(
                stds,
                parameters.similarity_sliding_window_stride,
            )
        ],
    )
    return similarity_sampling_coordinates


def _similarity_sampler(
    sampled_reference: GridComposableMapping,
    parameters: ImageParameters,
) -> GaussianSampler:
    grid_spacing_cpu = sampled_reference.coordinate_system.grid_spacing_cpu()
    stds = (
        _tensor(parameters.similarity_sliding_window_std, sampled_reference.dtype)
        / grid_spacing_cpu
    )
    return GaussianSampler(
        truncate_at=[
            stride / 2 + parameters.truncate_sliding_window_at_n_stds * std
            for std, stride in zip(
                stds,
                parameters.similarity_sliding_window_stride,
            )
        ],
        std=stds.tolist(),
        mean=[
            (2 * rand(1, device=torch_device("cpu"), dtype=float32).item() - 1) / 2 * int(stride)
            for stride in parameters.similarity_sliding_window_stride
        ],
    )


def _smoothed_mapping(
    image: GridComposableMapping,
    sampling_spacing: Tensor,
    truncate_at_n_stds: int | float,
) -> GridComposableMapping:
    voxel_size = image.coordinate_system.grid_spacing_cpu()
    smoothing_stds = 2.0 * sampling_spacing / voxel_size / 6.0
    smoothing_stds[sampling_spacing / voxel_size <= 1.0 + 1e-3] = 1e-6
    smoothing_sampler = GaussianSampler(
        truncate_at=[truncate_at_n_stds * std for std in smoothing_stds],
        std=smoothing_stds.tolist(),
    )
    sampled_image = image.sample()
    values, mask = sampled_image.generate(generate_missing_mask=False, cast_mask=False)
    if mask is not None:
        values = values * mask
    smoothed_values = (
        samplable_volume(
            values,
            coordinate_system=image.coordinate_system,
            sampler=smoothing_sampler,
        )
        .sample()
        .generate_values()
    )
    if mask is not None:
        smoothed_mask = (
            samplable_volume(
                mask.to(image.dtype),
                coordinate_system=image.coordinate_system,
                sampler=smoothing_sampler,
            )
            .sample()
            .generate_values()
        )
        smoothed_values = smoothed_values / (smoothed_mask + 1e-6)

    smoothed_image = samplable_volume(
        smoothed_values,
        coordinate_system=image.coordinate_system,
        mask=mask,
        sampler=LinearInterpolator(limit_direction=LimitDirection.average()),
    )
    coordinates = image.coordinate_system.reformat(
        reference=Center(),
        spatial_shape=OriginalShape() - 2,
    )
    derivatives = stack_mappable_tensors(
        *(
            smoothed_image.modify_sampler(
                sampler=LinearInterpolator().derivative(spatial_dim=spatial_dim)
            ).sample_to(coordinates)
            for spatial_dim in range(len(smoothing_stds))
        ),
        channel_index=0,
    )
    derivative_magnitude = derivatives.generate_values().square().sum(dim=1).sqrt()
    smoothed_values, smoothed_mask = smoothed_image.sample_to(coordinates).generate()
    joint_mapping = samplable_volume(
        cat([smoothed_values, derivative_magnitude], dim=1),
        coordinate_system=coordinates,
        mask=combine_optional_masks(
            smoothed_mask,
            derivatives.generate_mask()[:, 0],
        ),
    )
    return joint_mapping


def _normalize(
    image: GridComposableMapping,
) -> SamplableVolume:
    values, mask = image.sample().generate(generate_missing_mask=True, cast_mask=False)
    assert values.size(0) == 1
    valid_voxels = values[0][mask[0].broadcast_to(values.shape[1:])]
    image_min = valid_voxels.amin()
    image_max = valid_voxels.amax()
    values = values.clamp(min=image_min, max=image_max)
    values = (values - image_min) / (image_max - image_min)
    return samplable_volume(
        values,
        mask=mask,
        coordinate_system=image.coordinate_system,
    )


def _tensor(item: Tensor | Sequence[float | int], dtype: torch_dtype) -> Tensor:
    if isinstance(item, Tensor):
        return item
    return tensor(item, device=torch_device("cpu"), dtype=dtype)
