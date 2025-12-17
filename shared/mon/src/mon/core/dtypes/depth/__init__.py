#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A package for depth map data type.

This package provides a base class for handling depth map data along with
utility functions for processing depth maps.
"""

__all__ = [
    "DepthMap",
    "to_color",
]

from .core import DepthMap
from .processing import to_color
