#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Upsampling layers.

This module implements various upsampling layers commonly used for increasing
the resolution of feature maps.
"""

__all__ = [
    "Upsample",
    "UpsamplingBilinear2d",
    "UpsamplingNearest2d",
]

from torch.nn.modules.upsampling import *
