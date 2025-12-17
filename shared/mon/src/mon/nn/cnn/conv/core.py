#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for core convolutional layers.

This module provides standard convolutional layers and their lazy initialization
variants.
"""

__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "LazyConv1d",
    "LazyConv2d",
    "LazyConv3d",
    "LazyConvTranspose1d",
    "LazyConvTranspose2d",
    "LazyConvTranspose3d",
]

from torch.nn.modules.conv import *
