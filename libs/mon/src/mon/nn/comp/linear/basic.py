#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Basic linear layers.

This module provides various basic linear layers from PyTorch.
"""

from __future__ import annotations

__all__ = [
    "Bilinear",
    "Identity",
    "LazyLinear",
    "Linear",
]

from torch.nn.modules.linear import Bilinear, Identity, LazyLinear, Linear
