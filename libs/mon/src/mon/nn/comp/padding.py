#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Padding layers.

This module provides various padding layers used for padding feature maps to
ensure that they are of the same size without changing the spatial resolution.
"""

from __future__ import annotations

__all__ = [
    "CircularPad1d",
    "CircularPad2d",
    "CircularPad3d",
    "ConstantPad1d",
    "ConstantPad2d",
    "ConstantPad3d",
    "ReflectionPad1d",
    "ReflectionPad2d",
    "ReflectionPad3d",
    "ReplicationPad1d",
    "ReplicationPad2d",
    "ReplicationPad3d",
    "ZeroPad1d",
    "ZeroPad2d",
    "ZeroPad3d",
]

from torch.nn.modules.padding import (
    CircularPad1d,
    CircularPad2d,
    CircularPad3d,
    ConstantPad1d,
    ConstantPad2d,
    ConstantPad3d,
    ReflectionPad1d,
    ReflectionPad2d,
    ReflectionPad3d,
    ReplicationPad1d,
    ReplicationPad2d,
    ReplicationPad3d,
    ZeroPad1d,
    ZeroPad2d,
    ZeroPad3d,
)


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
