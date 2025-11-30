#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A package for linear layers.

This package implements various linear layer components used in neural networks.
"""

__all__ = [
    "Bilinear",
    "DepthAwareLinear",
    "Identity",
    "LazyLinear",
    "Linear",
]

from .core import Bilinear, Identity, LazyLinear, Linear
from .depthlinear import DepthAwareLinear
