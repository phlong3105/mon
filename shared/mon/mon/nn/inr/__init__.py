#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A package for Implicit Neural Representations (INRs).

This package provides various modules and classes for building and training
Implicit Neural Representations (INRs) in deep learning applications.
"""

__all__ = [
    # Layer
    "ComplexGaborLayer",
    "ConvINRLayer",
    "DepthAwareSineLayer",
    "FFEncoding",
    "FINERLayer",
    "GaussLayer",
    "PosEncodingNeRF",
    "RealGaborLayer",
    "SineLayer",
    # MLP
    "ConvINR",
    "FFEncodingMLP",
    "FINER",
    "FINER_PP",
    "GAUSS",
    "PosEncodingMLP",
    "SIREN",
    "WIRE",
    # Utils
    "create_coords",
    "create_depth_aware_patches",
    "create_noisy_coords",
    "create_patches",
    "ff_embedding",
    "filter_up",
    "interpolate_image",
    "pair_downsampler",
]

from .conv_inr import ConvINR, ConvINRLayer
from .ffn import FFEncoding, FFEncodingMLP
from .finer import FINER, FINER_PP, FINERLayer
from .gauss import GAUSS, GaussLayer
from .pe import PosEncodingMLP, PosEncodingNeRF
from .siren import DepthAwareSineLayer, SineLayer, SineLayerBN, SIREN
from .utils import (
    create_coords,
    create_depth_aware_patches,
    create_noisy_coords,
    create_patches,
    ff_embedding,
    filter_up,
    interpolate_image,
    pair_downsampler,
)
from .wire import ComplexGaborLayer, RealGaborLayer, WIRE
