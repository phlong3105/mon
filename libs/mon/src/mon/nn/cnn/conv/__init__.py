#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A package for convolutional neural network layers.

This package includes various convolutional layers, including standard,
depth-aware, depthwise separable, and Ghost modules, as well as MobileOne blocks.
It also provides lazy initialization variants of convolutional layers and
a utility function for computing offsets.
"""

__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "DSConv2d",
    "DepthAwareAvgPool2d",
    "DepthAwareConv2d",
    "GhostBottleneck",
    "GhostBottleneckV2",
    "GhostModule",
    "GhostModuleV2",
    "LazyConv1d",
    "LazyConv2d",
    "LazyConv3d",
    "LazyConvTranspose1d",
    "LazyConvTranspose2d",
    "LazyConvTranspose3d",
    "MobileOneBlock",
    "compute_offset",
]

from .core import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    LazyConv1d,
    LazyConv2d,
    LazyConv3d,
    LazyConvTranspose1d,
    LazyConvTranspose2d,
    LazyConvTranspose3d,
)
from .depthaware import DepthAwareAvgPool2d, DepthAwareConv2d
from .dsconv import DSConv2d
from .ghost import GhostBottleneck, GhostBottleneckV2, GhostModule, GhostModuleV2
from .mobileone import MobileOneBlock
from .zacn import compute_offset
