#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements NeurOP model for image retouching.

References:
    - Paper: "Neural Color Operators for Sequential Image Retouching," ECCV 2022.
    - Code: https://github.com/amberwangyili/neurop
"""

__all__ = [
    "NeurOP",
    "NeurOPInit",
]

from .neurop import NeurOP, NeurOPInit
