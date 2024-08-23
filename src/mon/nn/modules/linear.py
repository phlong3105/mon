#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Linearity Layers.

This module implements linearity layers.
"""

from __future__ import annotations

__all__ = [
    "Bilinear",
    "Identity",
    "LazyLinear",
    "Linear",
]

from torch.nn.modules.linear import *
