#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements feature extraction layers."""

from __future__ import annotations

__all__ = [
	"ExtractFeature",
	"ExtractFeatures",
	"ExtractItem",
	"ExtractItems",
	"Max",
]

from typing import Sequence

import torch
from torch import nn


class ExtractFeature(nn.Module):
	"""Extract a feature at :param:`index` in a tensor.
	
	Args:
		index: The index of the feature to extract.
	"""
	
	def __init__(self, index: int):
		super().__init__()
		self.index = index
	
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		if not input.ndim == 4:
			raise ValueError(
				f"input's number of dimensions must be == 4, but got {input.ndim}."
			)
		x = input
		y = x[:, self.index, :, :]
		return y
	
	@classmethod
	def parse_layer_args(cls, f: int, args: list, ch: list) -> tuple[list, list]:
		c2 = args[0]
		ch.append(c2)
		return args, ch


class ExtractFeatures(nn.Module):
	"""Extract features between :param:`start` index and :param:`end` index in a
	tensor.
	
	Args:
		start: The start index of the features to extract.
		end: The end index of the features to extract.
	"""
	
	def __init__(self, start: int, end: int):
		super().__init__()
		self.start = start
		self.end   = end
	
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		if not input.ndim == 4:
			raise ValueError(
				f"input's number of dimensions must be == 4, but got {input.ndim}."
			)
		x = input
		y = x[:, self.start:self.end, :, :]
		return y
	
	@classmethod
	def parse_layer_args(cls, f: int, args: list, ch: list) -> tuple[list, list]:
		c2 = args[1] - args[0]
		ch.append(c2)
		return args, ch


class ExtractItem(nn.Module):
	"""Extract an item (feature) at :param:`index` in a sequence of tensors.
	
	Args:
		index: The index of the item to extract.
	"""
	
	def __init__(self, index: int):
		super().__init__()
		self.index = index
	
	def forward(self, input: Sequence[torch.Tensor]) -> torch.Tensor:
		x = input
		if isinstance(x, torch.Tensor):
			return x
		elif isinstance(x, list | tuple):
			return x[self.index]
		else:
			raise TypeError(
				f"input must be a list or tuple of tensors, but got {type(input)}."
			)


class ExtractItems(nn.Module):
	"""Extract a :class:`list` of items (features) at `indexes` in a sequence of
	tensors.
	
	Args:
		indexes: The indexes of the items to extract.
	"""
	
	def __init__(self, indexes: Sequence[int]):
		super().__init__()
		self.indexes = indexes
	
	def forward(self, input: Sequence[torch.Tensor]) -> list[torch.Tensor]:
		x = input
		if isinstance(x, torch.Tensor):
			y = [x]
			return y
		elif isinstance(x, list | tuple):
			y = [x[i] for i in self.indexes]
			return y
		raise TypeError(
			f"input must be a list or tuple of tensors, but got {type(input)}."
		)


class Max(nn.Module):
	
	def __init__(self, dim: int, keepdim: bool = False):
		super().__init__()
		self.dim     = dim
		self.keepdim = keepdim
	
	def forward(self, input: torch.Tensor) -> torch.Tensor | int | float:
		x = input
		y = torch.max(input=x, dim=self.dim, keepdim=self.keepdim)
		return y
