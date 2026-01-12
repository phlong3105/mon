#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Basic convolutional layers.

This module provides standard convolutional layers from PyTorch.
"""

from __future__ import annotations

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

from torch.nn.modules.conv import (
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
