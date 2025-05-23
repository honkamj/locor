"""Registration script for the image registration tool."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
from os import makedirs
from os.path import dirname

from composable_mapping import (
    ComposableMapping,
    CoordinateSystem,
    DataFormat,
    GridComposableMapping,
    ICoordinateSystemContainer,
    Identity,
    LinearInterpolator,
    SamplableVolume,
    from_file,
    mappable,
)
from nibabel import Nifti1Image  # type: ignore
from nibabel import save as nib_save  # type: ignore
from torch import Tensor
from torch import device as torch_device
from torch import float16, float32, float64, tensor
from torch.cuda import is_available as cuda_is_available

from .config_parameters import ConfigBuildingArguments, RegistrationParameters
from .default_config import build_config as build_default_config
from .registration import register


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Tool for affine and deformable image registration."
    )
    parser.add_argument(
        "reference",
        help="Path to the reference image.",
        type=str,
    )
    parser.add_argument(
        "moving",
        help="Path to the moving image.",
        type=str,
    )

    output_arguments = parser.add_argument_group("Output arguments")
    output_arguments.add_argument(
        "-o",
        "--output-moving",
        help="Path to the registered moving image.",
        type=str,
    )
    output_arguments.add_argument(
        "-d",
        "--displacement-field-reference",
        help=(
            "Path to the obtained displacements from reference to moving image "
            "(in the coordinates of the reference image)."
        ),
        type=str,
    )
    output_arguments.add_argument(
        "--output-reference",
        help="Path to the registered reference image.",
        type=str,
    )
    output_arguments.add_argument(
        "--displacement-field-moving",
        help=(
            "Path to the obtained displacement field from moving to reference image "
            "(in the coordinates of the moving image). Note that since the deformation "
            "is learned in the coordinates of the reference image, this might contain "
            "extrapolated values."
        ),
        type=str,
    )
    output_arguments.add_argument(
        "--extrapolation-mask-moving",
        help="Path to the mask of the extrapolated regions for the registered moving image.",
        type=str,
    )
    output_arguments.add_argument(
        "--extrapolation-mask-reference",
        help="Path to the mask of the extrapolated regions for the registered reference image.",
        type=str,
    )

    masking_arguments = parser.add_argument_group("Masking arguments")
    masking_arguments.add_argument(
        "--mask-reference",
        help="Path to the registration mask of the reference image.",
        type=str,
    )
    masking_arguments.add_argument(
        "--mask-moving",
        help="Path to the registration mask of the moving image.",
        type=str,
    )

    registration_arguments = parser.add_argument_group("Registration arguments")
    registration_arguments.add_argument(
        "-c",
        "--config",
        help="Path to the custom config file.",
        type=str,
    )
    registration_arguments.add_argument(
        "--initialize-at-center",
        help="Initialize by centering the moving image with respect to the reference image.",
        action="store_true",
    )
    registration_arguments.add_argument(
        "--device",
        help=(
            "Device to use for registration, defaults to 'cuda:0' if available, else cpu. "
            "To use more than one device, provide the option multiple times "
            "(currently defining up to 2 devices are supported)."
        ),
        action="append",
        type=str,
    )
    registration_arguments.add_argument(
        "--dtype",
        help=("PyTorch data type to use for registration, defaults to 'float32'."),
        type=str,
        default="float32",
    )

    args = parser.parse_args()

    if args.device is None:
        devices = [torch_device("cuda:0" if cuda_is_available() else "cpu")]
    else:
        devices = [torch_device(device) for device in args.device]

    dtype = {
        "float16": float16,
        "float32": float32,
        "float64": float64,
    }[args.dtype]

    reference = from_file(args.reference, mask_path=args.mask_reference, dtype=dtype)
    moving = from_file(args.moving, mask_path=args.mask_moving, dtype=dtype)

    initialized_moving, initializing_affine = _initialize(
        args.initialize_at_center, moving, reference
    )

    parameters = _build_registration_parameters(
        args.config, ConfigBuildingArguments(reference=reference, moving=moving)
    )

    if (
        args.output_moving is None
        and args.output_reference is None
        and args.displacement_field_reference is None
        and args.displacement_field_moving is None
        and args.extrapolation_mask_moving is None
        and args.extrapolation_mask_reference is None
    ):
        raise ValueError("No outputs specified.")

    deformation_to_moving, deformation_to_reference = register(
        reference=reference,
        moving=initialized_moving,
        parameters=parameters,
        devices=devices,
    )

    moving = moving.cast(device=deformation_to_moving.device).modify_sampler(
        LinearInterpolator(extrapolation_mode="zeros")
    )
    reference = reference.cast(device=deformation_to_reference.device).modify_sampler(
        LinearInterpolator(extrapolation_mode="zeros")
    )
    deformation_to_moving = (
        initializing_affine.cast(device=deformation_to_moving.device) @ deformation_to_moving
    ).resample_to(reference.coordinate_system.cast(device=deformation_to_moving.device))
    deformation_to_reference = (
        deformation_to_reference
        @ initializing_affine.cast(device=deformation_to_reference.device).invert()
    ).resample_to(moving.coordinate_system.cast(device=deformation_to_reference.device))

    if args.output_moving is not None:
        _save_deformed_image(moving, deformation_to_moving, reference, args.output_moving)
    if args.output_reference is not None:
        _save_deformed_image(reference, deformation_to_reference, moving, args.output_reference)

    if args.extrapolation_mask_moving is not None:
        _save_extrapolation_mask(
            moving, deformation_to_moving, reference, args.extrapolation_mask_moving
        )
    if args.extrapolation_mask_reference is not None:
        _save_extrapolation_mask(
            reference, deformation_to_reference, moving, args.extrapolation_mask_reference
        )

    if args.displacement_field_reference is not None:
        _save_displacement_field(deformation_to_moving, args.displacement_field_reference)
    if args.displacement_field_moving is not None:
        _save_displacement_field(deformation_to_reference, args.displacement_field_moving)


def _save_deformed_image(
    image: SamplableVolume,
    deformation: GridComposableMapping,
    target: ICoordinateSystemContainer,
    path: str,
) -> None:
    deformed_image = (image @ deformation).resample_to(
        target.coordinate_system.cast(device=image.device)
    )
    deformed_values = deformed_image.sample().generate_values()
    nib_image = Nifti1Image(
        deformed_values[0].moveaxis(0, -1).squeeze(-1).numpy(force=True),
        affine=deformed_image.coordinate_system.from_voxel_coordinates.as_affine_matrix().numpy(
            force=True
        ),
    )
    makedirs(dirname(path), exist_ok=True)
    nib_save(nib_image, path)


def _save_extrapolation_mask(
    image: SamplableVolume,
    deformation: GridComposableMapping,
    target: ICoordinateSystemContainer,
    path: str,
) -> None:
    deformed_image = (image.clear_mask() @ deformation).resample_to(
        target.coordinate_system.cast(device=image.device)
    )
    extrapolation_mask = deformed_image.sample().generate_mask(generate_missing_mask=True)
    nib_image = Nifti1Image(
        extrapolation_mask[0].moveaxis(0, -1).squeeze(-1).numpy(force=True).astype("uint8"),
        affine=deformed_image.coordinate_system.from_voxel_coordinates.as_affine_matrix().numpy(
            force=True
        ),
    )
    makedirs(dirname(path), exist_ok=True)
    nib_save(nib_image, path)


def _save_displacement_field(deformation: GridComposableMapping, path: str) -> None:
    displacement_field = deformation.sample(
        data_format=DataFormat.voxel_displacements()
    ).generate_values()
    image = Nifti1Image(
        displacement_field[0].moveaxis(0, -1).numpy(force=True),
        affine=deformation.coordinate_system.from_voxel_coordinates.as_affine_matrix().numpy(
            force=True
        ),
    )
    makedirs(dirname(path), exist_ok=True)
    nib_save(image, path)


def _build_registration_parameters(
    config_path: str | None, arguments: ConfigBuildingArguments
) -> RegistrationParameters:
    if config_path is None:
        parameters: RegistrationParameters = build_default_config(arguments)
    else:
        config_module_spec = spec_from_file_location("config", config_path)
        if config_module_spec is None or config_module_spec.loader is None:
            raise ValueError(f"Could not load config from {config_path}.")
        config_module = module_from_spec(config_module_spec)
        config_module_spec.loader.exec_module(config_module)
        parameters = config_module.build_config(arguments)
    return parameters


def _initialize(
    initialize_at_center: bool, moving: SamplableVolume, reference: SamplableVolume
) -> tuple[SamplableVolume, ComposableMapping]:
    if initialize_at_center:
        moving_center = _center_coordinate(moving.coordinate_system)
        reference_center = _center_coordinate(reference.coordinate_system)
        centered_moving_coordinates = moving.coordinate_system + reference_center - moving_center
        initializing_affine: ComposableMapping = (
            moving.coordinate_system.from_voxel_coordinates
            @ centered_moving_coordinates.to_voxel_coordinates
        )
        moving = SamplableVolume(
            moving.sample(),
            coordinate_system=centered_moving_coordinates,
        )
    else:
        initializing_affine = Identity()
    return moving, initializing_affine


def _center_coordinate(coordinates: CoordinateSystem) -> Tensor:
    return coordinates.from_voxel_coordinates(
        mappable(
            (
                tensor(
                    coordinates.spatial_shape, device=coordinates.device, dtype=coordinates.dtype
                )
                - 1
            )
            / 2
        )
    ).generate_values()


_main()
