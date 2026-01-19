#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Upsampling layers.

This module provides various upsampling layers commonly used for increasing
the resolution of feature maps.
"""

from __future__ import annotations

__all__ = [
    "Upsample",
    "UpsamplingBilinear2d",
    "UpsamplingNearest2d",
]

from torch.nn.modules.upsampling import (
    Upsample,
    UpsamplingBilinear2d,
    UpsamplingNearest2d,
)


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
