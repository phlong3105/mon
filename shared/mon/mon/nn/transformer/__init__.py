#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A package for Transformer components and architectures.

This package provides implementations of various Transformer components,
including attention mechanisms and encoder/decoder layers, as well as complete
Transformer architectures.

References:
    - Definition: https://www.ibm.com/think/topics/transformer-model#774698769
"""

__all__ = [
    "SEBlock",
    "SimAM",
    "Transformer",
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerEncoderLayer",
]

from .attention import SEBlock, SimAM
from .core import (
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
