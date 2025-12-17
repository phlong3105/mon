#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Swin Transformer model for image classification.

References:
    - Paper: https://arxiv.org/pdf/2103.14030
"""

__all__ = [
    "Swin_B",
    "Swin_S",
    "Swin_T",
    "Swin_V2_B",
    "Swin_V2_S",
    "Swin_V2_T",
]

from .model import (
    Swin_B,
    Swin_S,
    Swin_T,
    Swin_V2_B,
    Swin_V2_S,
    Swin_V2_T,
)
