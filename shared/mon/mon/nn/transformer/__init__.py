#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements various Transformer components and architectures.

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

from .core import (
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from .attention import (
    SEBlock,
    SimAM,
)
