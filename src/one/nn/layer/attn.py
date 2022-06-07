#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Attention Layers.
"""

from __future__ import annotations

from typing import Any
from typing import Optional

import torch
from torch import nn
from torch import Tensor

from one.core import ATTN_LAYERS
from one.core import Callable
from one.core import Int2T
from one.core import to_2tuple
from one.nn.layer.act import create_act_layer

__all__ = [
	"ChannelAttentionBlock",
	"ChannelAttentionLayer",
	"PixelAttentionLayer",
	"SupervisedAttentionModule",
	"CAB",
	"CAL",
	"PAL",
	"SAM",
]


# MARK: - Modules

@ATTN_LAYERS.register(name="channel_attention_layer")
class ChannelAttentionLayer(nn.Module):
	"""Channel Attention Layer.
	
	Attributes:
		channels (int):
			Number of input and output channels.
		reduction (int):
			Reduction factor. Default: `16`.
		bias (bool):
			Default: `False`.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		channels    : int,
		reduction   : int    = 16,
		stride      : Int2T = (1, 1),
		dilation    : Int2T = (1, 1),
		groups      : int    = 1,
		bias        : bool   = False,
		padding_mode: str    = "zeros",
        device      : Any    = None,
        dtype       : Any    = None,
		**_
	):
		super().__init__()
		# Global average pooling: feature --> point
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		# Feature channel downscale and upscale --> channel weight
		self.ca = nn.Sequential(
			nn.Conv2d(
				in_channels  = channels,
				out_channels = channels // reduction,
				kernel_size  = (1, 1),
				stride       = stride,
				padding      = 0,
				dilation     = dilation,
				groups       = groups,
				bias         = bias,
				padding_mode = padding_mode,
	            device       = device,
	            dtype        = dtype,
			),
			nn.ReLU(inplace=True),
			nn.Conv2d(
				in_channels  = channels // reduction,
				out_channels = channels,
				kernel_size  = (1, 1),
				stride       = stride,
				padding      = 0,
				dilation     = dilation,
				groups       = groups,
				bias         = bias,
				padding_mode = padding_mode,
	            device       = device,
	            dtype        = dtype,
			),
			nn.Sigmoid()
		)
	
	# MARK: Forward Pass
	
	def forward(self, x: Tensor) -> Tensor:
		out = self.avg_pool(x)
		out = self.ca(out)
		return x * out


@ATTN_LAYERS.register(name="channel_attention_block")
class ChannelAttentionBlock(nn.Module):
	"""Channel Attention Block."""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		channels    : int,
		reduction   : int,
		kernel_size : Int2T,
		stride      : Int2T              = (1, 1),
		dilation    : Int2T              = (1, 1),
        groups      : int                 = 1,
        bias        : bool                = True,
        padding_mode: str                 = "zeros",
        device      : Any                 = None,
        dtype       : Any                 = None,
		act         : Optional[Callable]  = nn.ReLU,
		**_
	):
		super().__init__()
		kernel_size = to_2tuple(kernel_size)
		stride      = to_2tuple(stride)
		dilation    = to_2tuple(dilation)
		padding     = kernel_size[0] // 2
		act         = create_act_layer(act_layer=act)
		self.ca     = CAL(channels, reduction, bias)
		self.body   = nn.Sequential(
			nn.Conv2d(
				in_channels  = channels,
				out_channels = channels,
				kernel_size  = kernel_size,
				stride       = stride,
				padding      = padding,
				dilation     = dilation,
	            groups       = groups,
	            bias         = bias,
	            padding_mode = padding_mode,
	            device       = device,
	            dtype        = dtype,
			),
			act,
			nn.Conv2d(
				in_channels  = channels,
				out_channels = channels,
				kernel_size  = kernel_size,
				stride       = stride,
				padding      = padding,
				dilation     = dilation,
	            groups       = groups,
	            bias         = bias,
	            padding_mode = padding_mode,
	            device       = device,
	            dtype        = dtype,
			),
		)
		
	# MARK: Forward Pass
	
	def forward(self, x: Tensor) -> Tensor:
		out = self.body(x)
		out = self.ca(out)
		out += x
		return out


@ATTN_LAYERS.register(name="pixel_attention_layer")
class PixelAttentionLayer(nn.Module):
	"""Pixel Attention Layer.
	
	Args:
		reduction (int):
			Reduction factor. Default: `16`.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		channels    : int,
		reduction   : int    = 16,
		stride      : Int2T = (1, 1),
		dilation    : Int2T = (1, 1),
        groups      : int    = 1,
        bias        : bool   = False,
        padding_mode: str    = "zeros",
        device      : Any    = None,
        dtype       : Any    = None,
        **_
	):
		super().__init__()
		stride   = to_2tuple(stride)
		dilation = to_2tuple(dilation)
		self.pa  = nn.Sequential(
			nn.Conv2d(
				in_channels  = channels,
				out_channels = channels // reduction,
				kernel_size  = (1, 1),
				stride       = stride,
				padding      = 0,
				dilation     = dilation,
				groups       = groups,
				bias         = bias,
				padding_mode = padding_mode,
	            device       = device,
	            dtype        = dtype,
			),
			nn.ReLU(inplace=True),
			nn.Conv2d(
				in_channels  = channels // reduction,
				out_channels = 1,
				kernel_size  = (1, 1),
				stride       = stride,
				padding      = 0,
				dilation     = dilation,
				groups       = groups,
				bias         = bias,
				padding_mode = padding_mode,
	            device       = device,
	            dtype        = dtype,
			),
			nn.Sigmoid()
		)
	
	# MARK: Forward Pass
	
	def forward(self, x: Tensor) -> Tensor:
		out = self.pa(x)
		return x * out


@ATTN_LAYERS.register(name="supervised_attention_module")
class SupervisedAttentionModule(nn.Module):
	"""Supervised Attention Module."""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		channels    : int,
		kernel_size : Int2T,
		dilation    : Int2T              = (1, 1),
        groups      : int                 = 1,
        bias        : bool                = False,
        padding_mode: str                 = "zeros",
        device      : Any                 = None,
        dtype       : Any                 = None,
        **_
	):
		super().__init__()
		kernel_size = to_2tuple(kernel_size)
		stride      = (1, 1)
		padding     = kernel_size[0] // 2
		dilation    = to_2tuple(dilation)
		self.conv1 = nn.Conv2d(
			in_channels  = channels,
			out_channels = channels,
			kernel_size  = kernel_size,
			stride       = stride,
			padding      = padding,
			dilation     = dilation,
			groups       = groups,
			bias         = bias,
			padding_mode = padding_mode,
			device       = device,
			dtype        = dtype,
		)
		self.conv2 = nn.Conv2d(
			in_channels  = channels,
			out_channels = 3,
			kernel_size  = kernel_size,
			stride       = stride,
			padding      = padding,
			dilation     = dilation,
			groups       = groups,
			bias         = bias,
			padding_mode = padding_mode,
			device       = device,
			dtype        = dtype,
		)
		self.conv3 = nn.Conv2d(
			in_channels  = 3,
			out_channels = channels,
			kernel_size  = kernel_size,
			stride       = stride,
			padding      = padding,
			dilation     = dilation,
			groups       = groups,
			bias         = bias,
			padding_mode = padding_mode,
			device       = device,
			dtype        = dtype,
		)
		
	# MARK: Forward Pass
	
	def forward(self, fx: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
		"""Run forward pass.

		Args:
			fx (Tensor):
				Output from previous steps.
			x (Tensor):
				Original input images.
			
		Returns:
			pred (Tensor):
				Output image.
			img (Tensor):
				Output image.
		"""
		x1    = self.conv1(fx)
		img   = self.conv2(fx) + x
		x2    = torch.sigmoid(self.conv3(img))
		x1   *= x2
		x1   += fx
		pred  = x1
		return pred, img


# MARK: - Alias

CAB = ChannelAttentionBlock
CAL = ChannelAttentionLayer
PAL = PixelAttentionLayer
SAM = SupervisedAttentionModule


# MARK: - Register

ATTN_LAYERS.register(name="cab",      module=CAB)
ATTN_LAYERS.register(name="cal",      module=CAL)
ATTN_LAYERS.register(name="identity", module=nn.Identity)
ATTN_LAYERS.register(name="pal",      module=PAL)
ATTN_LAYERS.register(name="sam",      module=SAM)
