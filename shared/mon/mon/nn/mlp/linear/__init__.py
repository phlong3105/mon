#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements various linear layers."""

__all__ = [
    "Bilinear",
    "DepthAwareLinear",
    "Identity",
    "LazyLinear",
    "Linear",
]

from .core import Bilinear, Identity, LazyLinear, Linear
from .depthlinear import DepthAwareLinear
