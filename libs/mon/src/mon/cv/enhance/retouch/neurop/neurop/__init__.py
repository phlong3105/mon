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

from .data import build_train_loader
from .model import NeurOP, NeurOPInit
from .models import build_model
from .utils import dict_to_nonedict, parse
