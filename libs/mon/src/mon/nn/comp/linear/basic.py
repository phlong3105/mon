#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Basic linear layers from PyTorch.

This module implements various linear layers commonly used in MLP and deep
neural networks.
"""

__all__ = [
    "Bilinear",
    "Identity",
    "LazyLinear",
    "Linear",
]

from torch.nn.modules.linear import *
