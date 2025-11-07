#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements various upsampling layers."""

__all__ = [
    "Upsample",
    "UpsamplingBilinear2d",
    "UpsamplingNearest2d",
]

from torch.nn.modules.upsampling import *
