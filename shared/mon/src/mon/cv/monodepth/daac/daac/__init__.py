#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements DAAC model for depth estimation.

References:
    - Paper: "Depth Anything At Any Condition," arXiv 2025.
    - Code: https://github.com/HVision-NKU/DepthAnythingAC
"""

__all__ = [
    "DAAC",
]

from .model import DAAC
