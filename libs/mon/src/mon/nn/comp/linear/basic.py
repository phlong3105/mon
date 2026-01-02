#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Basic linear layers.

This module implements various basic linear layers from PyTorch.
"""

__all__ = [
    "Bilinear",
    "Identity",
    "LazyLinear",
    "Linear",
]

from torch.nn.modules.linear import *
