#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Sparse embedding layers.

This module implements various sparse embedding layers used for learning sparse
representations of words.
"""

__all__ = [
    "Embedding",
    "EmbeddingBag",
]

from torch.nn.modules.sparse import *
