#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for sparse neural network components.

This module provides classes for embedding layers commonly used in neural
networks to handle sparse data representations.
"""

__all__ = [
    "Embedding",
    "EmbeddingBag",
]

from torch.nn.modules.sparse import *
