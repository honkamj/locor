"""Symmetric deformation builder class"""

from composable_mapping import (
    Affine,
    ComposableMapping,
    CoordinateSystem,
    DataFormat,
    GridComposableMapping,
    Identity,
    LinearInterpolator,
    SamplableVolume,
    ScalingAndSquaring,
)
from torch import Tensor

from .affine_transformation import (
    AffineTransformationTypeDefinition,
    generate_affine_transformation_matrix,
)


class SymmetricDeformationModel:
    """Utility class for building deformations in symmetric manner"""

    def __init__(
        self,
        affine: ComposableMapping | None = None,
        from_reference: ComposableMapping | None = None,
        to_moving: ComposableMapping | None = None,
        inverse: bool = False,
    ) -> None:
        self.affine = Identity() if affine is None else affine
        self.inverse = inverse
        self._from_reference = Identity() if from_reference is None else from_reference
        self._to_moving = Identity() if to_moving is None else to_moving

    def set_affine(
        self,
        affine_parameters: Tensor,
        affine_transformation_type: AffineTransformationTypeDefinition,
        normalizing_affine: Affine,
        normalizing_affine_inverse: Affine,
    ) -> "SymmetricDeformationModel":
        """Set the affine transformation"""
        transformation_matrix = generate_affine_transformation_matrix(
            affine_parameters,
            transformation_type=affine_transformation_type,
            inverse=self.inverse,
        )
        affine_transformation = Affine(
            (
                normalizing_affine_inverse
                @ Affine.from_matrix(transformation_matrix)
                @ normalizing_affine
            ).as_affine_transformation()
        )
        return SymmetricDeformationModel(
            affine=affine_transformation,
            from_reference=self._from_reference,
            to_moving=self._to_moving,
            inverse=self.inverse,
        )

    def update(
        self,
        update_svf: SamplableVolume,
        n_scalings_and_squarings: int,
    ) -> "SymmetricDeformationModel":
        """Update the deformation"""
        sampler = LinearInterpolator(mask_extrapolated_regions=False)
        svf_deformation = update_svf.modify_sampler(
            sampler=ScalingAndSquaring(sampler=sampler, steps=n_scalings_and_squarings)
        )
        if self.inverse:
            svf_deformation = svf_deformation.invert()
        update_deformation = svf_deformation.resample(sampler=sampler)
        return SymmetricDeformationModel(
            affine=self.affine,
            from_reference=update_deformation @ self._from_reference,
            to_moving=self._to_moving @ update_deformation,
            inverse=self.inverse,
        )

    def build_full_deformation(
        self, coordinates: CoordinateSystem
    ) -> tuple[GridComposableMapping, GridComposableMapping]:
        """Build full deformations"""
        full_dense = (self._to_moving @ self._from_reference).resample_to(
            coordinates,
            sampler=LinearInterpolator(mask_extrapolated_regions=False),
            data_format=DataFormat.voxel_displacements(),
        )
        if self.inverse:
            return full_dense @ self.affine, full_dense
        return self.affine @ full_dense, full_dense

    def resample(self, coordinates: CoordinateSystem) -> "SymmetricDeformationModel":
        """Resample the deformations"""
        sampler = LinearInterpolator(mask_extrapolated_regions=False)
        return SymmetricDeformationModel(
            affine=self.affine,
            from_reference=self._from_reference.resample_to(
                coordinates, sampler=sampler, data_format=DataFormat.voxel_displacements()
            ),
            to_moving=self._to_moving.resample_to(
                coordinates, sampler=sampler, data_format=DataFormat.voxel_displacements()
            ),
            inverse=self.inverse,
        )
