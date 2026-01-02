#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for dropout layers.

This module implements various dropout layers commonly used in neural networks
to prevent overfitting by randomly setting a fraction of input units to zero
during training.
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
