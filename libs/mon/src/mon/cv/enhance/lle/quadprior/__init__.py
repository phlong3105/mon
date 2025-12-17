#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements QuadPrior model for low-light image enhancement.

References:
    - Paper: "Zero-Reference Low-Light Enhancement via Physical Quadruple
      Priors," CVPR 2024.
    - Code: https://github.com/daooshee/QuadPrior
"""

__all__ = [
    "QuadPrior",
]

from .quadprior import QuadPrior
