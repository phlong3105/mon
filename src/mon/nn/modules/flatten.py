#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Flatten/Unflatten Layers.

This module implements flatten/unflatten layers.
"""

from __future__ import annotations

__all__ = [
	"Flatten",
	"FlattenSingle",
	"Unflatten",
]

import torch
from torch import nn
from torch.nn.modules.flatten import *


# region Flatten

class FlattenSingle(nn.Module):
	"""Flatten a tensor along a single dimension.

	Args:
		dim: Dimension to flatten. Default: ``1``.
	"""
	
	def __init__(self, dim: int = 1):
		super().__init__()
		self.dim = dim
	
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		x = input
		y = torch.flatten(x, self.dim)
		return y

# endregion
