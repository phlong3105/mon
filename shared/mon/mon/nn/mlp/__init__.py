#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A package for multi-layer perceptrons (MLPs).

This package implements various multi-layer perceptron (MLP) components and
architectures.
"""

__all__ = [
    "Bilinear",
    "DepthAwareLinear",
    "Identity",
    "LazyLinear",
    "Linear",
]

from .linear import Bilinear, DepthAwareLinear, Identity, LazyLinear, Linear
