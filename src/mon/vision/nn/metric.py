#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements evaluation metrics for training vision deep learning
models.

This module is built on top of :mod:`mon.nn.metric`.
"""

from __future__ import annotations

__all__ = [
    "CriticalSuccessIndex",
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
    "calculate_efficiency_score",
]

import time
from copy import deepcopy

import thop
import torch
import torchmetrics
from fvcore.nn import FlopCountAnalysis, parameter_count

from mon import core, nn
from mon.globals import METRICS
# noinspection PyUnresolvedReferences
from mon.nn.metric import *
from mon.vision import core

console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region Image Metric

CriticalSuccessIndex                         = torchmetrics.image.CriticalSuccessIndex
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

METRICS.register(name="critical_success_index",                           module=CriticalSuccessIndex)
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


# region Parameters

def calculate_efficiency_score(
    model     : nn.Module | nn.Model,
    image_size: int | list[int] = 512,
    channels  : int             = 3,
    runs      : int             = 100,
    use_cuda  : bool            = True,
    verbose   : bool            = False,
):
    # Define input tensor
    h, w  = core.get_hw(image_size)
    input = torch.rand(1, channels, h, w)
    
    # Deploy to cuda
    if use_cuda:
        input = input.cuda()
        model = model.cuda()
     
    # Get FLOPs and Params
    flops, params = thop.profile(deepcopy(model), inputs=(input, ), verbose=verbose)
    flops         = FlopCountAnalysis(model, input).total() if flops == 0 else flops
    params        = model.params if hasattr(model, "params") and params == 0 else params
    params        = parameter_count(model) if hasattr(model, "params") else params
    params        = sum(list(params.values())) if isinstance(params, dict) else params
    g_flops       = flops  * 1e-9
    m_params      = params * 1e-6
    
    # Get time
    start_time = time.time()
    for i in range(runs):
        _ = model(input)
    runtime    = time.time() - start_time
    avg_time   = runtime / runs
    
    # Print
    if verbose:
        console.log(f"FLOPs (G)  = {flops:.4f}")
        console.log(f"Params (M) = {params:.4f}")
        console.log(f"Time (s)   = {avg_time:.4f}")
    
    return flops, params, avg_time
    
# endregion
