#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements MobileOne models for image classification.

References:
    - Paper: "MobileOne: An Improved One millisecond Mobile Backbone," CVPR 2023.
    - Code: https://github.com/apple/ml-mobileone/tree/main
"""

__all__ = [
    "MobileOneS0",
    "MobileOneS1",
    "MobileOneS2",
    "MobileOneS3",
    "MobileOneS4",
]

from .model import *
