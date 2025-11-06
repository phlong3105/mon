#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements MultiLayer Perceptron (MLP), the first type of neural network."""

__all__ = [
    "Bilinear",
    "DepthAwareLinear",
    "Identity",
    "LazyLinear",
    "Linear",
]

from .linear import Bilinear, DepthAwareLinear, Identity, LazyLinear, Linear
