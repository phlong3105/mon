#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements channel attention modules."""

from __future__ import annotations

__all__ = [
	"ChannelAttention",
	"ECA",
	"ECA1d",
	"EfficientChannelAttention",
	"EfficientChannelAttention1d",
	"SimplifiedChannelAttention",
	"SqueezeExcitation",
	"SqueezeExciteC",
	"SqueezeExciteL",
]

from typing import Any

import torch
from torch import nn
from torchvision.ops import misc as torchvision_misc

from mon.globals import LAYERS
from mon.nn.layer import activation as act, base, conv, linear, pooling
from mon.nn.typing import _size_2_t


# region Efficient Channel Attention

@LAYERS.register()
class EfficientChannelAttention(base.PassThroughLayerParsingMixin, nn.Module):
	"""Constructs an Efficient Channel Attention (ECA) module.
	
	Args:
		channels: Number of channels of the input feature map.
		kernel_size: Adaptive selection of kernel size.
	"""
	
	def __init__(self, channels: int, kernel_size: _size_2_t = 3):
		super().__init__()
		self.avg_pool = pooling.AdaptiveAvgPool2d(1)
		self.conv     = conv.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
		self.sigmoid  = act.Sigmoid()
		self.channel  = channels
		self.k_size   = kernel_size
	
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		x = input
		# feature descriptor on the global spatial information
		y = self.avg_pool(x)
		# Two different branches of ECA module
		y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
		# Multi-scale information fusion
		y = self.sigmoid(y)
		return x * y.expand_as(x)
	
	def flops(self) -> int:
		flops  = 0
		flops += self.channel * self.channel * self.k_size
		return flops


@LAYERS.register()
class EfficientChannelAttention1d(base.PassThroughLayerParsingMixin, nn.Module):
	"""Constructs an Efficient Channel Attention (ECA) module.
	
	Args:
		channels: Number of channels of the input feature map.
		kernel_size: Adaptive selection of kernel size.
	"""
	
	def __init__(self, channels: int, kernel_size: _size_2_t = 3):
		super().__init__()
		self.avg_pool = pooling.AdaptiveAvgPool1d(1)
		self.conv     = conv.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
		self.sigmoid  = act.Sigmoid()
		self.channel  = channels
		self.k_size   = kernel_size
	
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		x = input
		# b hw c
		# feature descriptor on the global spatial information
		y = self.avg_pool(x.transpose(-1, -2))
		# Two different branches of ECA module
		y = self.conv(y.transpose(-1, -2))
		# Multi-scale information fusion
		y = self.sigmoid(y)
		return x * y.expand_as(x)
	
	def flops(self) -> int:
		flops = 0
		flops += self.channel * self.channel * self.k_size
		return flops


ECA   = EfficientChannelAttention
ECA1d = EfficientChannelAttention1d
LAYERS.register(name="ECA",   module=ECA)
LAYERS.register(name="ECA1d", module=ECA1d)

# endregion


# region Simplified Channel Attention

@LAYERS.register()
class SimplifiedChannelAttention(base.PassThroughLayerParsingMixin, nn.Module):
	"""Simplified channel attention layer proposed in the paper: "`Simple
	Baselines for Image Restoration <https://arxiv.org/pdf/2204.04676.pdf>`__".
	"""
	
	def __init__(
		self,
		channels: int,
		bias    : bool = True,
		device  : Any  = None,
		dtype   : Any  = None,
	):
		super().__init__()
		self.avg_pool   = pooling.AdaptiveAvgPool2d(1)  # squeeze
		self.excitation = conv.Conv2d(
			in_channels  = channels,
			out_channels = channels,
			kernel_size  = 1,
			stride       = 1,
			padding      = 0,
			bias         = bias,
			device       = device,
			dtype        = dtype,
		)
	
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		x = input
		b, c, _, _ = x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.excitation(y).view(b, c, 1, 1)
		y = x * y.expand_as(x)
		# y = self.avg_pool(x)
		# y = self.excitation(y)
		# y = y.view(-1, c, 1, 1)
		# y = x * y
		return y

# endregion


# region Squeeze Excitation

@LAYERS.register()
class SqueezeExciteC(base.PassThroughLayerParsingMixin, nn.Module):
	"""Squeeze and Excite layer from the paper: "`Squeeze and Excitation
	Networks <https://arxiv.org/pdf/1709.01507.pdf>`__"
	
	This implementation uses :class:`torch.nn.Conv2d` layer.
	
	References:
		- `<https://amaarora.github.io/2020/07/24/SeNet.html#squeeze-and-excitation-block-in-pytorch>`__
		- `<https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py>`__
	"""
	
	def __init__(
		self,
		channels       : int,
		reduction_ratio: int  = 16,
		bias           : bool = False,
		device         : Any  = None,
		dtype          : Any  = None,
	):
		super().__init__()
		self.avg_pool   = pooling.AdaptiveAvgPool2d(1)  # squeeze
		self.excitation = nn.Sequential(
			conv.Conv2d(
				in_channels  = channels,
				out_channels = channels  // reduction_ratio,
				kernel_size  = 1,
				stride       = 1,
				bias         = bias,
				device       = device,
				dtype        = dtype,
			),
			act.ReLU(inplace=True),
			conv.Conv2d(
				in_channels  = channels  // reduction_ratio,
				out_channels = channels,
				kernel_size  = 1,
				stride       = 1,
				bias         = bias,
				device       = device,
				dtype        = dtype,
			),
			act.Sigmoid()
		)
	
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		x = input
		b, c, _, _ = x.size()
		
		y = self.avg_pool(x).view(b, c)
		y = self.excitation(y).view(b, c, 1, 1)
		y = x * y.expand_as(x)
		
		# y = self.avg_pool(x)
		# y = self.excitation(y)
		# y = y.view(-1, c, 1, 1)
		# y = x * y
		
		return y


@LAYERS.register()
class SqueezeExciteL(base.PassThroughLayerParsingMixin, nn.Module):
	"""Squeeze and Excite layer from the paper: "`Squeeze and Excitation
	Networks <https://arxiv.org/pdf/1709.01507.pdf>`__"
	
	This implementation uses :class:`torch.nn.Linear` layer.
	
	References:
		- `<https://amaarora.github.io/2020/07/24/SeNet.html#squeeze-and-excitation-block-in-pytorch>`__
		- `<https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py>`__
	"""
	
	def __init__(
		self,
		channels       : int,
		reduction_ratio: int  = 16,
		bias           : bool = False,
		device         : Any  = None,
		dtype          : Any  = None,
	):
		super().__init__()
		self.avg_pool   = pooling.AdaptiveAvgPool2d(1)  # squeeze
		self.excitation = nn.Sequential(
			linear.Linear(
				in_features  = channels,
				out_features = channels  // reduction_ratio,
				bias         = bias,
				device       = device,
				dtype        = dtype,
			),
			act.ReLU(inplace=True),
			linear.Linear(
				in_features  = channels  // reduction_ratio,
				out_features = channels,
				bias         = bias,
				device       = device,
				dtype        = dtype,
			),
			act.Sigmoid()
		)
	
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		x = input
		b, c, _, _ = x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.excitation(y).view(b, c, 1, 1)
		y = x * y.expand_as(x)
		
		# y = self.avg_pool(x)
		# y = self.excitation(y)
		# y = y.view(-1, c, 1, 1)
		# y = x * y
		return y


@LAYERS.register()
class SqueezeExcitation(base.PassThroughLayerParsingMixin, torchvision_misc.SqueezeExcitation):
	pass


ChannelAttention = SqueezeExciteC
LAYERS.register(name="ChannelAttention", module=ChannelAttention)

# endregion
