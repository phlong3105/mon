#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Dropout layers.

This module provides various dropout layers used for regularization by randomly
setting activations to zero.
"""

from __future__ import annotations

__all__ = [
    "AlphaDropout",
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "FeatureAlphaDropout",
]

from torch.nn.modules.dropout import (
    AlphaDropout,
    Dropout,
    Dropout1d,
    Dropout2d,
    Dropout3d,
    FeatureAlphaDropout,
)
