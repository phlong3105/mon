#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Dropout layers.

This module implements various dropout layers used for regularization by randomly
setting activations to zero.
"""

__all__ = [
    "AlphaDropout",
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "FeatureAlphaDropout",
]

from torch.nn.modules.dropout import *
