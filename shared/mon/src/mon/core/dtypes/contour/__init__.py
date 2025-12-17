#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A package for contour data type.

This package provides functionalities for handling contour data, including
normalization, denormalization, and conversion between different formats.
"""

__all__ = [
    "convert",
    "denormalize",
    "normalize",
]

from .processing import convert, denormalize, normalize
