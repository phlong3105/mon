#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for core normalization layers.

This module provides various normalization layers commonly used in convolutional
neural networks (CNNs).
"""

__all__ = [
    "CrossMapLRN2d",
    "GroupNorm",
    "LayerNorm",
    "LocalResponseNorm",
    "RMSNorm",
]

from torch.nn.modules.normalization import *
