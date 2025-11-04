#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements CoLIE model for low-light image enhancement.

References:
    - Paper: "Fast Context-Based Low-Light Image Enhancement via Neural Implicit
      Representations," ECCV 2024.
    - Code: https://github.com/ctom2/colie
"""

__all__ = [
    "CoLIE",
]

from .colie import CoLIE
