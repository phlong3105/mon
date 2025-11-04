#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ResNet model for image classification.

References:
    - Paper: https://arxiv.org/abs/1512.03385
"""

__all__ = [
    "ResNeXt101_32X8D",
    "ResNeXt101_64X4D",
    "ResNeXt50_32X4D",
    "ResNet101",
    "ResNet152",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "WideResNet101_2",
    "WideResNet50_2",
]

from .model import (
    ResNet101,
    ResNet152,
    ResNet18,
    ResNet34,
    ResNet50,
    ResNeXt101_32X8D,
    ResNeXt101_64X4D,
    ResNeXt50_32X4D,
    WideResNet101_2,
    WideResNet50_2,
)
