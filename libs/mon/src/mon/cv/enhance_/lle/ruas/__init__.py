#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements RUAS model for low-light image enhancement.

References:
    - Paper: "Retinex-inspired Unrolling with Cooperative Prior Architecture
      Search for Low-light Image Enhancement," 2021.
    - Code: https://github.com/KarelZhang/RUAS
"""

__all__ = [
    "RUAS",
]

from .ruas import RUAS
