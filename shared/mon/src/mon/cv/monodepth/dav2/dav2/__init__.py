#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements DAV2 model prediction pipeline for monocular depth estimation.

References:
    - Paper: "Depth Anything V2. A More Capable Foundation Model for Monocular
      Depth Estimation," NeurIPS 2024.
    - Code: https://github.com/DepthAnything/Depth-Anything-V2
"""

__all__ = [
    "DAV2",
    "DAV2_ViTS",
    "DAV2_ViTB",
    "DAV2_ViTL",
]

from .model import DAV2, DAV2_ViTB, DAV2_ViTL, DAV2_ViTS
