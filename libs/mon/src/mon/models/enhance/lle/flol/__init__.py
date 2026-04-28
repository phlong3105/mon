#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FLOL.

This package contains FLOL model implementations, pre-trained weights, and
utilities for training and inference.

References:
    - Paper: "FLOL: Fast Baselines for Real-World Low-Light Enhancement," arXiv 2026.
    - Code: https://github.com/cidautai/FLOL
"""

from __future__ import annotations

from .model import *
from .predict import *
from .train import *
