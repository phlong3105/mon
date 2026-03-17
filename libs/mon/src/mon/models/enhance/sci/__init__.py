#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SCI.

This package contains SCI model implementations, pre-trained weights, and
utilities for training and inference.

References:
    - Paper: "Toward Fast, Flexible, and Robust Low-Light Image Enhancement,"
      CVPR 2022.
    - Code: https://github.com/vis-opt-group/SCI

    - Paper: "Learning with Self-Calibrator for Fast and Robust Low-Light
      Image Enhancement," TPAMI 2025.
    - Code: https://github.com/vis-opt-group/SCI
"""

from __future__ import annotations

from .model import *
from .predict import *
from .train import *
