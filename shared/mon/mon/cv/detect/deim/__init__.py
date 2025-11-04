#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements DEIM model for object detection.

References:
    - Paper: "DEIM: DETR with Improved Matching for Fast convergence," CVPR 2025.
    - Code: https://github.com/ShihuaHuang95/DEIM
"""

__all__ = [
    "DEIM",
]

from .engine import DEIM
