#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Transformer models.

The transformer model is a type of neural network architecture that excels at
processing sequential data, most prominently associated with large language
models (LLMs). Transformer models have also achieved elite performance in other
fields of artificial intelligence (AI), such as computer vision, speech
recognition and time series forecasting.

References:
    - https://www.ibm.com/think/topics/transformer-model#774698769
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
