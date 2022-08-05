#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Resolution Block used in paper: `Multi-Stage Progressive Image Restoration`.
"""

from __future__ import annotations

from torch import nn
from torch import Tensor

from one.core import Int2T
from one.core import to_2tuple
from one.nn.layer.attn import CAB

__all__ = [
	"OriginalResolutionBlock",
	"ORB",
]


# MARK: - Modules

class OriginalResolutionBlock(nn.Module):
	"""Original Resolution Block.
	
	Args:
		channels (int):
			Number of input and output channels.
		kernel_size (Int2T):
			Kernel size of the convolution layer.
		reduction (int):
			Reduction factor. Default: `16`.
		bias (bool):
			Default: `False`.
		act (nn.Module):
			Activation function.
		num_cab (int):
			Number of CAB modules used.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		channels   : int,
		kernel_size: Int2T,
		reduction  : int,
		bias       : bool,
		act        : nn.Module,
		num_cab    : int,
	):
		
		super().__init__()
		kernel_size = to_2tuple(kernel_size)
		padding 	= kernel_size[0] // 2
		body        = [CAB(channels, kernel_size, reduction, bias, act)
					   for _ in range(num_cab)]
		body.append(
			nn.Conv2d(channels, channels, kernel_size, stride=(1, 1),
					  padding=padding, bias=bias)
		)
		self.body = nn.Sequential(*body)
	
	# MARK: Forward Pass
	
	def forward(self, x: Tensor) -> Tensor:
		out  = self.body(x)
		out += x
		return out


# MARK: - Alias

ORB = OriginalResolutionBlock
