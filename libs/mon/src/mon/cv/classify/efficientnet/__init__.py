#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements EfficientNet model for image classification.

References:
    - Paper: https://arxiv.org/abs/1905.11946
"""

__all__ = [
    "EfficientNet_B0",
    "EfficientNet_B1",
    "EfficientNet_B2",
    "EfficientNet_B3",
    "EfficientNet_B4",
    "EfficientNet_B5",
    "EfficientNet_B6",
    "EfficientNet_B7",
    "EfficientNet_V2_L",
    "EfficientNet_V2_M",
    "EfficientNet_V2_S",
]

from .model import (
    EfficientNet_B0,
    EfficientNet_B1,
    EfficientNet_B2,
    EfficientNet_B3,
    EfficientNet_B4,
    EfficientNet_B5,
    EfficientNet_B6,
    EfficientNet_B7,
    EfficientNet_V2_L,
    EfficientNet_V2_M,
    EfficientNet_V2_S
)
