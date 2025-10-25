#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements SRNO model for super-resolution.

References:
    - Paper: "Super-Resolution Neural Operator," CVPR 2023.
    - Code: https://github.com/2y7c3/Super-Resolution-Neural-Operator
"""

__all__ = [
    "SRNO",
]

from .model import SRNO
from .models import make
from .utils import make_coord
