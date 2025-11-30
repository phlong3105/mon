#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A package for segmentation mask data type.

This package provides a data structure for handling segmentation masks,
which are used to represent pixel-wise class labels in images.
"""

__all__ = [
    "SemanticMask",
]

from .core import SemanticMask
