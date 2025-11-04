#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements LIME model for low-light image enhancement.

References:
    - Paper: "LIME: Low-light Image Enhancement via Illumination Map Estimation,"
      TIP 2006.
    - Code: https://github.com/pvnieo/Low-light-Image-Enhancement
"""

__all__ = [
    "LIME",
]

from .lime import LIME
