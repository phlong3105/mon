#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements DarkIR model for low-light deblurring.

References:
    - Paper: "DarkIR: Robust Low-Light Image Restoration," CVPR 2025.
    - Code: https://github.com/cidautai/DarkIR
"""

__all__ = [
    "DarkIR",
]

from .archs import create_model
from .model import DarkIR
