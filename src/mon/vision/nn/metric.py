#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements evaluation metrics for training vision deep learning
models.

This module is built on top of :mod:`mon.nn.metric`.
"""

from __future__ import annotations

__all__ = [
    "ErrorRelativeGlobalDimensionlessSynthesis",
    "MultiScaleStructuralSimilarityIndexMeasure", "PeakSignalNoiseRatio",
    "SpectralAngleMapper", "SpectralDistortionIndex",
    "StructuralSimilarityIndexMeasure", "TotalVariation",
    "UniversalImageQualityIndex", "calculate_efficiency_score"
]

import time
from copy import deepcopy
from typing import Literal, DefaultDict

import thop
import torch
import torchmetrics
from fvcore.nn import FlopCountAnalysis, parameter_count_table, parameter_count

from mon import core, nn
from mon.globals import METRICS
# noinspection PyUnresolvedReferences
from mon.nn.metric import *
from mon.vision import core

console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region Image Metric

ErrorRelativeGlobalDimensionlessSynthesis  = torchmetrics.ErrorRelativeGlobalDimensionlessSynthesis
MultiScaleStructuralSimilarityIndexMeasure = torchmetrics.MultiScaleStructuralSimilarityIndexMeasure
PeakSignalNoiseRatio                       = torchmetrics.PeakSignalNoiseRatio
SpectralAngleMapper                        = torchmetrics.SpectralAngleMapper
SpectralDistortionIndex                    = torchmetrics.SpectralDistortionIndex
StructuralSimilarityIndexMeasure           = torchmetrics.StructuralSimilarityIndexMeasure
TotalVariation                             = torchmetrics.TotalVariation
UniversalImageQualityIndex                 = torchmetrics.UniversalImageQualityIndex

METRICS.register(name="error_relative_global_dimensionless_synthesis",  module=ErrorRelativeGlobalDimensionlessSynthesis)
METRICS.register(name="multiscale_structural_similarity_index_measure", module=MultiScaleStructuralSimilarityIndexMeasure)
METRICS.register(name="multiscale_ssim",                                module=MultiScaleStructuralSimilarityIndexMeasure)
METRICS.register(name="peak_signal_noise_ratio",                        module=PeakSignalNoiseRatio)
METRICS.register(name="psnr",                                           module=PeakSignalNoiseRatio)
METRICS.register(name="spectral_angle_mapper",                          module=SpectralAngleMapper)
METRICS.register(name="spectral_distortion_index",                      module=SpectralDistortionIndex)
METRICS.register(name="structural_similarity_index_measure",            module=StructuralSimilarityIndexMeasure)
METRICS.register(name="ssim",                                           module=StructuralSimilarityIndexMeasure)
METRICS.register(name="total_variation",                                module=TotalVariation)
METRICS.register(name="universal_image_quality_index",                  module=UniversalImageQualityIndex)

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
