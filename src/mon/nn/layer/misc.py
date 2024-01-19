#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements miscellaneous layers."""

from __future__ import annotations

__all__ = [
	"Embedding",
	"Permute",
]

from torch.nn import Embedding
from torchvision.ops.misc import Permute
