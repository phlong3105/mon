#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ConvNeXt models for image classification.

References:
    - Paper: https://arxiv.org/abs/2201.03545
"""

__all__ = [
    "ConvNeXtBase",
    "ConvNeXtLarge",
    "ConvNeXtSmall",
    "ConvNeXtTiny",
]

from .model import ConvNeXtBase, ConvNeXtLarge, ConvNeXtSmall, ConvNeXtTiny
