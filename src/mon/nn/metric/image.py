#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements image metrics."""

from __future__ import annotations

__all__ = [
    "ErrorRelativeGlobalDimensionlessSynthesis",
    "LearnedPerceptualImagePatchSimilarity",
    "MemorizationInformedFrechetInceptionDistance",
    "MultiScaleStructuralSimilarityIndexMeasure",
    "PeakSignalNoiseRatio",
    "PeakSignalNoiseRatioWithBlockedEffect",
    "PerceptualPathLength",
    "RelativeAverageSpectralError",
    "RootMeanSquaredErrorUsingSlidingWindow",
    "SpatialCorrelationCoefficient",
    "SpatialDistortionIndex",
    "SpectralAngleMapper",
    "SpectralDistortionIndex",
    "StructuralSimilarityIndexMeasure",
    "TotalVariation",
    "UniversalImageQualityIndex",
    "VisualInformationFidelity",
]

import torchmetrics

from mon.globals import METRICS

# region Image Metric

ErrorRelativeGlobalDimensionlessSynthesis    = torchmetrics.image.ErrorRelativeGlobalDimensionlessSynthesis
LearnedPerceptualImagePatchSimilarity        = torchmetrics.image.LearnedPerceptualImagePatchSimilarity
MemorizationInformedFrechetInceptionDistance = torchmetrics.image.MemorizationInformedFrechetInceptionDistance
MultiScaleStructuralSimilarityIndexMeasure   = torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure
PeakSignalNoiseRatio                         = torchmetrics.image.PeakSignalNoiseRatio
PeakSignalNoiseRatioWithBlockedEffect        = torchmetrics.image.PeakSignalNoiseRatioWithBlockedEffect
PerceptualPathLength                         = torchmetrics.image.PerceptualPathLength
RelativeAverageSpectralError                 = torchmetrics.image.RelativeAverageSpectralError
RootMeanSquaredErrorUsingSlidingWindow       = torchmetrics.image.RootMeanSquaredErrorUsingSlidingWindow
SpatialCorrelationCoefficient                = torchmetrics.image.SpatialCorrelationCoefficient
SpatialDistortionIndex                       = torchmetrics.image.SpatialDistortionIndex
SpectralAngleMapper                          = torchmetrics.image.SpectralAngleMapper
SpectralDistortionIndex                      = torchmetrics.image.SpectralDistortionIndex
StructuralSimilarityIndexMeasure             = torchmetrics.image.StructuralSimilarityIndexMeasure
TotalVariation                               = torchmetrics.image.TotalVariation
UniversalImageQualityIndex                   = torchmetrics.image.UniversalImageQualityIndex
VisualInformationFidelity                    = torchmetrics.image.VisualInformationFidelity

METRICS.register(name="error_relative_global_dimensionless_synthesis",    module=ErrorRelativeGlobalDimensionlessSynthesis)
METRICS.register(name="learned_perceptual_image_patch_similarity",        module=LearnedPerceptualImagePatchSimilarity)
METRICS.register(name="lpips",                                            module=LearnedPerceptualImagePatchSimilarity)
METRICS.register(name="memorization_informed_frechet_inception_distance", module=MemorizationInformedFrechetInceptionDistance)
METRICS.register(name="multiscale_structural_similarity_index_measure",   module=MultiScaleStructuralSimilarityIndexMeasure)
METRICS.register(name="multiscale_ssim",                                  module=MultiScaleStructuralSimilarityIndexMeasure)
METRICS.register(name="peak_signal_noise_ratio",                          module=PeakSignalNoiseRatio)
METRICS.register(name="psnr",                                             module=PeakSignalNoiseRatio)
METRICS.register(name="peak_signal_noise_ratio_with_blocked_effect",      module=PeakSignalNoiseRatioWithBlockedEffect)
METRICS.register(name="perceptual_path_length",                           module=PerceptualPathLength)
METRICS.register(name="relative_average_spectral_error",                  module=RelativeAverageSpectralError)
METRICS.register(name="root_mean_squared_error_using_sliding_window",     module=RootMeanSquaredErrorUsingSlidingWindow)
METRICS.register(name="spatial_correlation_coefficient",                  module=SpatialCorrelationCoefficient)
METRICS.register(name="spatial_distortion_index",                         module=SpatialDistortionIndex)
METRICS.register(name="spectral_angle_mapper",                            module=SpectralAngleMapper)
METRICS.register(name="spectral_distortion_index",                        module=SpectralDistortionIndex)
METRICS.register(name="structural_similarity_index_measure",              module=StructuralSimilarityIndexMeasure)
METRICS.register(name="ssim",                                             module=StructuralSimilarityIndexMeasure)
METRICS.register(name="total_variation",                                  module=TotalVariation)
METRICS.register(name="universal_image_quality_index",                    module=UniversalImageQualityIndex)
METRICS.register(name="visual_information_fidelity",                      module=VisualInformationFidelity)

# endregion
