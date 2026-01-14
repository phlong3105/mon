#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Basic normalization layers.

This module provides various basic normalization layers from PyTorch.
"""

from __future__ import annotations

__all__ = [
    "CrossMapLRN2d",
    "GroupNorm",
    "LayerNorm",
    "LocalResponseNorm",
    "RMSNorm",
]

from torch.nn.modules.normalization import (
    CrossMapLRN2d,
    GroupNorm,
    LayerNorm,
    LocalResponseNorm,
    RMSNorm,
)
