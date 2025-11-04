#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements MobileNetV3 model for image classification.

References:
     - Paper: https://arxiv.org/abs/1905.02244
"""

__all__ = [
    "MobileNetV3Large",
    "MobileNetV3Small",
]

from .model import MobileNetV3Large, MobileNetV3Small
