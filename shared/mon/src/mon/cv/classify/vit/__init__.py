#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Vision Transformer model for image classification.

References:
    - Paper: https://arxiv.org/abs/2010.11929
"""

__all__ = [
    "ViT_B_16",
    "ViT_B_32",
    "ViT_H_14",
    "ViT_L_16",
    "ViT_L_32",
]

from .model import (
    ViT_B_16,
    ViT_B_32,
    ViT_H_14,
    ViT_L_16,
    ViT_L_32,
)
