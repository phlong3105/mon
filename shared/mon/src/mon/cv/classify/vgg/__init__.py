#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements VGG model for image classification.

References:
    - Paper: https://arxiv.org/abs/1409.1556
"""

__all__ = [
    "VGG11",
    "VGG11_BN",
    "VGG13",
    "VGG13_BN",
    "VGG16",
    "VGG16_BN",
    "VGG19",
    "VGG19_BN",
]

from .model import (
    VGG11,
    VGG11_BN,
    VGG13,
    VGG13_BN,
    VGG16,
    VGG16_BN,
    VGG19,
    VGG19_BN,
)
