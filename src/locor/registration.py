"""Registration functions"""

from multiprocessing import Pool, set_start_method
from typing import Any, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxmorph import (
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
    samplable_volume,
    stack_mappable_tensors,
)
from jaxmorph.util import combine_optional_masks
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
from .transformation_parameters import (
    initialize_affine_transformation_parameters,
    initialize_spline_parameters,
)


def register(
    reference: GridComposableMapping,
    moving: GridComposableMapping,
    parameters: RegistrationParameters,
    devices: Sequence[jax.Device],
) -> tuple[GridComposableMapping, GridComposableMapping]:
    """Register the images and return the coordinate mappings in both directions.

    Coordinate mapping is returned in both directions but the non-affine
    component is learned in the coordinates of the reference image.
    """
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
    device: jax.Device,
) -> Any:
    with jax.default_device(device):
        if rank is not None:
            jax.distributed.initialize()
            process_rank_postfix = f" (process {rank})"
        else:
            process_rank_postfix = ""

        reference = _normalize(reference)
        moving = _normalize(moving)

        n_reference_channels = reference.sample().channels_shape[0]
        n_moving_channels = moving.sample().channels_shape[0]

        n_dims = len(reference.coordinate_system.spatial_shape)

        key = jax.random.PRNGKey(0)
        key_1, key_2 = jax.random.split(key, num=2)

        feature_extractors: list[FeatureExtractor] = []
        deformations: list[SymmetricDeformationModel] = []
        full_deformation_references: list[CoordinateSystem] = []
        if rank in (0, None):
            key_1, feature_extractor_key_1 = jax.random.split(key_1)
            feature_extractors.append(
                FeatureExtractor(
                    n_dims=n_dims,
                    n_input_channels=2 * n_reference_channels,
                    n_hidden_features=(
                        parameters.feature_extraction_parameters_reference.n_hidden_features
                    ),
                    n_output_channels=parameters.feature_extraction_parameters_moving.n_features,
                    key=feature_extractor_key_1,
                )
            )
            deformations.append(SymmetricDeformationModel())
            full_deformation_references.append(
                reference.coordinate_system.device_put(device=device)
            )
        if rank in (1, None):
            key_2, feature_extractor_key_2 = jax.random.split(key_2)
            feature_extractors.append(
                FeatureExtractor(
                    n_dims=n_dims,
                    n_input_channels=2 * n_moving_channels,
                    n_hidden_features=parameters.feature_extraction_parameters_moving.n_hidden_features,
                    n_output_channels=parameters.feature_extraction_parameters_moving.n_features,
                    key=feature_extractor_key_2,
                )
            )
            deformations.append(SymmetricDeformationModel(inverse=True))
            full_deformation_references.append(moving.coordinate_system.device_put(device=device))

        if parameters.affine_stage_parameters is not None:
            print(f"Starting affine stage{process_rank_postfix}")
            subkey_1, key_1 = jax.random.split(key_1)
            subkey_2, key_2 = jax.random.split(key_2)
            deformations = _register_affine(
                initial_deformations=deformations,
                feature_extractors=feature_extractors,
                keys=[subkey_1, subkey_2],
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
        deformation_coordinates = reference.coordinate_system.device_put(device=device)
        for deformation, full_deformation_reference in zip(
            deformations, full_deformation_references
        ):
            full_deformation, _ = deformation.build_full_deformation(deformation_coordinates)
            output_deformations.append(
                full_deformation.assign_coordinates(full_deformation_reference)
            )

        if rank is not None:
            assert len(output_deformations) == 1
            return output_deformations[0]

        return output_deformations


def _register_affine(
    initial_deformations: Sequence[SymmetricDeformationModel],
    feature_extractors: Sequence[FeatureExtractor],
    keys: Sequence[jnp.ndarray],
    reference: GridComposableMapping,
    moving: GridComposableMapping,
    parameters: AffineStageParameters,
    device: jax.Device,
    rank: int | None,
) -> list[SymmetricDeformationModel]:
    affine_parameters = initialize_affine_transformation_parameters(
        n_dims=len(reference.coordinate_system.spatial_shape),
        transformation_type=parameters.transformation_type,
    )
    if rank is not None:
        process_rank_postfix = f" (process {rank})"
    else:
        process_rank_postfix = ""
    optimizer = optax.adam(
        learning_rate=parameters.transformation_learning_rate,
    )
    optimizer_state = optimizer.init(affine_parameters)
    feature_optimizers: list[optax.GradientTransformation] = []
    feature_optimizer_states: list[optax.OptState] = []
    for feature_extractor in feature_extractors:
        feature_optimizer = optax.adam(
            learning_rate=parameters.feature_learning_rate,
        )
        feature_optimizers.append(feature_optimizer)
        feature_optimizer_states.append(
            feature_optimizer.init(eqx.filter(feature_extractor, eqx.is_array))
        )

    registration_inputs = _initialize_registration_stage(
        initial_deformations=initial_deformations,
        reference=reference,
        moving=moving,
        parameters=parameters,
        device=device,
    )
    normalizing_affine, inverse_normalizing_affine = _normalizing_affine(
        moving.coordinate_system.device_put(device=device)
    )
    progress_bar = tqdm(range(parameters.n_iterations), position=rank)

    def loss(
        affine_parameters: jnp.ndarray,
        feature_extractors: Sequence[FeatureExtractor],
        keys: Sequence[jnp.ndarray],
    ) -> jnp.ndarray:
        similarity_losses = []
        for (
            (reference_initialized, moving_initialized, similarity_coordinates),
            feature_extractor,
            initial_deformation,
            key,
        ) in zip(registration_inputs, feature_extractors, initial_deformations, keys):
            reference_image_parameters = (
                parameters.reference_image_parameters,
                parameters.moving_image_parameters,
            )[initial_deformation.inverse]
            updated_deformation = initial_deformation.set_affine(
                affine_parameters=affine_parameters,
                affine_transformation_type=parameters.transformation_type,
                normalizing_affine=normalizing_affine,
                normalizing_affine_inverse=inverse_normalizing_affine,
            )
            registered_moving = (moving_initialized @ updated_deformation.affine).resample_to(
                reference_initialized
            )
            registered_moving_values, registered_moving_mask = registered_moving.sample().generate()
            features_moving = feature_extractor(registered_moving_values)
            similarity_sampler = _similarity_sampler(
                reference_initialized,
                reference_image_parameters,
                key=key,
            )
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
        loss = jnp.stack(similarity_losses).mean()
        return loss

    @jax.jit
    def step(
        affine_parameters: jnp.ndarray,
        feature_extractors: Sequence[FeatureExtractor],
        optimizer_state: optax.OptState,
        feature_optimizer_states: Sequence[optax.OptState],
        keys: Sequence[jnp.ndarray],
    ) -> tuple[
        jnp.ndarray,
        jnp.ndarray,
        optax.GradientTransformation,
        Sequence[optax.GradientTransformation],
    ]:
        feature_extractors = list(feature_extractors)
        feature_optimizer_states = list(feature_optimizer_states)
        loss_value, grads = jax.value_and_grad(loss, argnums=(0, 1))(
            affine_parameters, feature_extractors, keys
        )
        updates, optimizer_state = optimizer.update(grads[0], optimizer_state)
        affine_parameters = optax.apply_updates(affine_parameters, updates)
        for i, feature_optimizer in enumerate(feature_optimizers):
            updates, feature_optimizer_states[i] = feature_optimizer.update(
                grads[1][i], feature_optimizer_states[i]
            )
            feature_extractors[i] = optax.apply_updates(feature_extractors[i], updates)
        return loss_value, affine_parameters, optimizer_state, feature_optimizer_states

    key_1, key_2 = keys
    for _ in progress_bar:
        subkey_1, key_1 = jax.random.split(key_1)
        subkey_2, key_2 = jax.random.split(key_2)
        print("Taking first step!")
        loss_value, affine_parameters, optimizer_state, feature_optimizer_states = step(
            affine_parameters,
            feature_extractors,
            optimizer_state,
            feature_optimizer_states,
            [subkey_1, subkey_2],
        )
        progress_bar.set_description(f"Loss{process_rank_postfix}: {float(loss_value):.4e}")
    return [
        initial_deformation.set_affine(
            affine_parameters=affine_parameters,
            affine_transformation_type=parameters.transformation_type,
            normalizing_affine=normalizing_affine,
            normalizing_affine_inverse=inverse_normalizing_affine,
        )
        for initial_deformation in initial_deformations
    ]


def _normalizing_affine(coordinates: CoordinateSystem) -> tuple[Affine, Affine]:
    centered_normalized_coordinates = CoordinateSystem.centered_normalized(
        spatial_shape=coordinates.spatial_shape,
        voxel_size=coordinates.grid_spacing(),
        dtype=coordinates.dtype,
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
    device: jax.Device,
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
    device: jax.Device,
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
    device: jax.Device,
) -> tuple[GridComposableMapping, GridComposableMapping]:
    reference = reference.device_put(device=device)
    moving = moving.device_put(device=device)
    deformation_coordinates = deformation_coordinates.device_put(device=device)

    reference_smoothed = _smoothed_mapping(
        reference,
        sampling_spacing=jnp.asarray(
            reference_parameters.image_sampling_spacing, dtype=reference.dtype
        ),
        truncate_at_n_stds=reference_parameters.truncate_image_smoothing_at_n_stds,
    )
    moving_smoothed = _smoothed_mapping(
        moving,
        sampling_spacing=jnp.asarray(
            reference_parameters.image_sampling_spacing, dtype=reference.dtype
        ),
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
    grid_spacing = sampled_reference.coordinate_system.grid_spacing()
    stds = (
        np.asarray(parameters.similarity_sliding_window_std, dtype=sampled_reference.dtype)
        / grid_spacing
    )
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
    key: jnp.ndarray,
) -> GaussianSampler:
    grid_spacing = sampled_reference.coordinate_system.grid_spacing()
    n_dims = len(parameters.similarity_sliding_window_stride)
    stds = (
        np.asarray(parameters.similarity_sliding_window_std, sampled_reference.dtype) / grid_spacing
    )
    strides = jnp.asarray(parameters.similarity_sliding_window_stride)
    means = (2 * jax.random.uniform(key=key, shape=(n_dims,), dtype=stds.dtype) - 1) * strides / 2
    return GaussianSampler(
        truncate_at=[
            stride / 2 + parameters.truncate_sliding_window_at_n_stds * std
            for std, stride in zip(
                stds,
                parameters.similarity_sliding_window_stride,
            )
        ],
        std=jnp.asarray(stds),
        mean=means,
    )


def _smoothed_mapping(
    image: GridComposableMapping,
    sampling_spacing: jnp.ndarray,
    truncate_at_n_stds: int | float,
) -> GridComposableMapping:
    voxel_size = image.coordinate_system.grid_spacing()
    smoothing_stds = 2.0 * sampling_spacing / voxel_size / 6.0
    smoothing_stds = smoothing_stds.at[sampling_spacing / voxel_size <= 1.0 + 1e-3].set(1e-6)
    smoothing_sampler = GaussianSampler(
        truncate_at=[truncate_at_n_stds * std for std in smoothing_stds],
        std=jnp.asarray(smoothing_stds),
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
                mask.astype(image.dtype),
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
    derivative_magnitude = jnp.sqrt(jnp.sum(derivatives.generate_values() ** 2, axis=1))
    smoothed_values, smoothed_mask = smoothed_image.sample_to(coordinates).generate()
    joint_mapping = samplable_volume(
        jnp.concatenate([smoothed_values, derivative_magnitude], axis=1),
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
    assert values.shape[0] == 1
    valid_voxels = values[0][jnp.broadcast_to(mask[0], values.shape[1:])]
    image_min = jnp.amin(valid_voxels)
    image_max = jnp.amax(valid_voxels)
    values = jnp.clip(values, min=image_min, max=image_max)
    values = (values - image_min) / (image_max - image_min)
    return samplable_volume(
        values,
        mask=mask,
        coordinate_system=image.coordinate_system,
    )
