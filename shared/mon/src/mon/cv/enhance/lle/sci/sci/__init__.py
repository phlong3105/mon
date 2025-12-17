#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements SCI model for low-light image enhancement.

References:
    - Paper: "Toward Fast, Flexible, and Robust Low-Light Image Enhancement,"
      CVPR 2022.
    - Code: https://github.com/vis-opt-group/SCI
"""

__all__ = [
    "SCI",
]

from .model import SCI
