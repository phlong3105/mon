#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Sparse embedding layers.

This module provides various sparse embedding layers used for learning sparse
representations of words.
"""

from __future__ import annotations

__all__ = [
    "Embedding",
    "EmbeddingBag",
]

from torch.nn.modules.sparse import Embedding, EmbeddingBag
