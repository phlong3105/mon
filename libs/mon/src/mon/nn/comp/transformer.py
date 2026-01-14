#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Transformer components.

This module provides various transformer components used for sequence modeling
tasks.
"""

from __future__ import annotations

__all__ = [
    "Transformer",
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerEncoderLayer",
]

from torch.nn.modules.transformer import (
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
