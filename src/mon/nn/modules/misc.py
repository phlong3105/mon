#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Miscellaneous Layers.

This module implements miscellaneous layers.
"""

from __future__ import annotations

__all__ = [
	"Embedding",
	"MLP",
	"Permute",
]

from torch.nn import Embedding
from torchvision.ops.misc import MLP, Permute
