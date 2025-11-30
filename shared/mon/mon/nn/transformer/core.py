#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for Transformer architectures.

This module provides implementations of Transformer models, including encoder
and decoder layers, as well as the full Transformer architecture.
"""

__all__ = [
    "Transformer",
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerEncoderLayer",
]

from torch.nn.modules.transformer import *
