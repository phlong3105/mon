#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements multilayer perceptron (mlp)."""

__all__ = [
    "Bilinear",
    "DepthAwareLinear",
    "Identity",
    "LazyLinear",
    "Linear",
]

from .linear import Bilinear, DepthAwareLinear, Identity, LazyLinear, Linear
