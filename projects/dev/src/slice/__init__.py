#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SLICE.

This package contains SLICE implementations, pre-trained weights, and utilities
for training and inference.

References:
    - Paper: "SLICE: Scale-Arbitrary Low-Light Enhancement via Depth-Aware
      Implicit Curve Estimation"
    - Code: https://github.com/phlong3105/slice
"""

from __future__ import annotations

from .model import *
from .predict import *
from .train import *
