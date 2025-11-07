#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements various multi-layer perceptron (MLP) components."""

__all__ = [
    "Bilinear",
    "DepthAwareLinear",
    "Identity",
    "LazyLinear",
    "Linear",
]

from .linear import Bilinear, DepthAwareLinear, Identity, LazyLinear, Linear
