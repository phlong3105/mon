#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements upsampling layers."""

__all__ = [
    "Upsample",
    "UpsamplingBilinear2d",
    "UpsamplingNearest2d",
]

from torch.nn.modules.upsampling import *
