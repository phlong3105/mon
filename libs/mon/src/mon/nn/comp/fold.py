#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fold and unfold layers.

This module implements various fold and unfold layers used for reducing the
spatial dimensionality of feature maps.
"""

__all__ = [
    "Fold",
    "Unfold",
]

from torch.nn.modules.fold import *
