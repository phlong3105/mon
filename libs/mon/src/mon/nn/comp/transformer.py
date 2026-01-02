#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Transformer components.

This module implements various transformer components used for sequence modeling
tasks.
"""

__all__ = [
    "Transformer",
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerEncoderLayer",
]

from torch.nn.modules.transformer import *
