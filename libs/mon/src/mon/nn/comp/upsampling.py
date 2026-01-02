#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for upsampling layers.

This module implements classes for upsampling operations in neural networks.
"""

__all__ = [
    "Upsample",
    "UpsamplingBilinear2d",
    "UpsamplingNearest2d",
]

from torch.nn.modules.upsampling import *
