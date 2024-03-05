#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements channel attention modules."""

from __future__ import annotations

__all__ = [
	"BAM",
	"CBAM",
	"ChannelAttentionModule",
	"SimAM",
]

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mon.core import _size_2_t
from mon.nn.layer import (
	activation as act, conv, flatten, linear, normalization as norm,
	pooling,
)


# region Channel Attention Module

class BAM(nn.Module):
	"""Bottleneck Attention Module from the paper: "BAM: Bottleneck Attention
	Module".
	
	References:
		- `<https://github.com/Jongchan/attention-module/blob/master/MODELS/bam.py>`__
	"""
	
	class Flatten(nn.Module):
		def forward(self, input: torch.Tensor) -> torch.Tensor:
			x = input
			y = x.view(x.size(0), -1)
			return y
	
	class ChannelAttention(nn.Module):
		def __init__(
			self,
			channels       : int,
			reduction_ratio: int = 16,
			num_layers     : int = 1,
		):
			super().__init__()
			gate_channels  = [channels]
			gate_channels += [channels // reduction_ratio] * num_layers
			gate_channels += [channels]
			
			self.c_gate = nn.Sequential()
			self.c_gate.add_module("flatten", self.Flatten())
			for i in range(len(gate_channels) - 2):
				self.c_gate.add_module(
					name   = "gate_c_fc_%d" % i,
					module = linear.Linear(
						in_features  = gate_channels[i],
						out_features = gate_channels[i + 1],
					)
				)
				self.c_gate.add_module(
					name   = "gate_c_bn_%d" % (i + 1),
					module = norm.BatchNorm1d(
						num_features=gate_channels[i + 1]
					)
				)
				self.c_gate.add_module(
					name   = "gate_c_relu_%d" % (i + 1),
					module = act.ReLU(),
				)
			
			self.c_gate.add_module(
				name   = "gate_c_fc_final",
				module = linear.Linear(
					in_features  = gate_channels[-2],
					out_features = gate_channels[-1],
				)
			)
		
		def forward(self, input: torch.Tensor) -> torch.Tensor:
			x = input
			y = F.avg_pool2d(x, x.size(2), stride=x.size(2))
			y = self.c_gate(y)
			y = y.unsqueeze(2).unsqueeze(3).expand_as(x)
			return y
	
	class SpatialAttention(nn.Module):
		def __init__(
			self,
			channels         : int,
			reduction_ratio  : int = 16,
			dilation_conv_num: int = 2,
			dilation_val     : int = 4,
			*args, **kwargs
		):
			super().__init__()
			self.s_gate = nn.Sequential()
			self.s_gate.add_module(
				name   = "gate_s_conv_reduce0",
				module = conv.Conv2d(
					in_channels  = channels,
					out_channels = channels  // reduction_ratio,
					kernel_size  = 1,
				)
			)
			self.s_gate.add_module(
				name   = "gate_s_bn_reduce0",
				module = norm.BatchNorm2d(
					num_features=channels // reduction_ratio
				)
			)
			self.s_gate.add_module(
				name   = "gate_s_relu_reduce0",
				module = act.ReLU()
			)
			for i in range(dilation_conv_num):
				self.s_gate.add_module(
					"gate_s_conv_di_%d" % i,
					conv.Conv2d(
						in_channels  = channels      // reduction_ratio,
						out_channels = channels      // reduction_ratio,
						kernel_size  = 3,
						padding      = dilation_val,
						dilation     = dilation_val,
					)
				)
				self.s_gate.add_module(
					name   = "gate_s_bn_di_%d" % i,
					module = norm.BatchNorm2d(
						num_features=channels // reduction_ratio
					)
				)
				self.s_gate.add_module(
					name   = "gate_s_relu_di_%d" % i,
					module = act.ReLU(),
				)
			self.s_gate.add_module(
				name   = "gate_s_conv_final",
				module = conv.Conv2d(
					in_channels  = channels // reduction_ratio,
					out_channels = 1,
					kernel_size  = 1,
				)
			)
		
		def forward(self, input: torch.Tensor) -> torch.Tensor:
			x = input
			y = self.s_gate(x).expand_as(x)
			return y
	
	def __init__(
		self,
		channels       : int,
		reduction_ratio: int = 16,
		num_layers     : int = 1,
	):
		super().__init__()
		self.channel = self.ChannelAttention(
			channels        = channels,
			reduction_ratio = reduction_ratio,
			num_layers      = num_layers,
		)
		self.spatial = self.SpatialAttention(
			channels        = channels,
			reduction_ratio = reduction_ratio,
		)
		self.sigmoid = act.Sigmoid()
	
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		x = input
		y = 1 + self.sigmoid(self.channel_att(x) * self.spatial_att(x))
		y = y * x
		return x


class CBAM(nn.Module):
	"""Convolutional Block Attention Module from the paper: "CBAM: Convolutional
	Block Attention Module".
	
	References:
		- `<https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py>`__
	
	Args:
		channels:
		reduction_ratio: Default: ``16``.
		pool_types: Pooling layer. One of ``'avg'``, `''lp''`, `''lse''`, or
			`''max''`. Defaults to ``["avg", "max"]``.
	"""
	
	class Flatten(nn.Module):
		def forward(self, input: torch.Tensor) -> torch.Tensor:
			x = input
			y = x.view(x.size(0), -1)
			return y
	
	# noinspection PyDefaultArgument
	class ChannelAttention(nn.Module):
		def __init__(
			self,
			channels       : int,
			reduction_ratio: int  = 16,
			pool_types     : list = ["avg", "max"],
		):
			super().__init__()
			self.channels = channels
			self.mlp      = nn.Sequential(
				flatten.Flatten(),
				linear.Linear(
					in_features  = channels,
					out_features = channels  // reduction_ratio,
				),
				act.ReLU(),
				linear.Linear(
					in_features  = channels  // reduction_ratio,
					out_features = channels,
				)
			)
			self.pool_types = pool_types
		
		def forward(self, input: torch.Tensor) -> torch.Tensor:
			x = input
			channel_att_sum = None
			channel_att_raw = None
			
			for pool_type in self.pool_types:
				if pool_type == "avg":
					avg_pool = F.avg_pool2d(
						input       = x,
						kernel_size = (x.size(2), x.size(3)),
						stride      = (x.size(2), x.size(3))
					)
					channel_att_raw = self.mlp(avg_pool)
				elif pool_type == "max":
					max_pool = F.max_pool2d(
						input       = x,
						kernel_size = (x.size(2), x.size(3)),
						stride      = (x.size(2), x.size(3))
					)
					channel_att_raw = self.mlp(max_pool)
				elif pool_type == "lp":
					lp_pool = F.lp_pool2d(
						input       = x,
						norm_type   = 2,
						kernel_size = (x.size(2), x.size(3)),
						stride      = (x.size(2), x.size(3))
					)
					channel_att_raw = self.mlp(lp_pool)
				elif pool_type == "lse":
					# LSE pool only
					lse_pool        = pooling.lse_pool2d(x)
					channel_att_raw = self.mlp(lse_pool)
				
				if channel_att_sum is None:
					channel_att_sum = channel_att_raw
				else:
					channel_att_sum = channel_att_sum + channel_att_raw
			
			y = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(
				3
			).expand_as(x)  # scale
			y = x * y
			return y
	
	class SpatialAttention(nn.Module):
		def __init__(self, *args, **kwargs):
			super().__init__()
			kernel_size = 7
			self.compress = pooling.ChannelPool()
			self.spatial  = conv.Conv2dNormAct(
				in_channels      = 2,
				out_channels     = 1,
				kernel_size      = kernel_size,
				stride           = 1,
				padding          = (kernel_size - 1) // 2,
				norm_layer       = norm.BatchNorm2d,
				activation_layer = None,
			)
			self.sigmoid = act.Sigmoid()
		
		def forward(self, input: torch.Tensor) -> torch.Tensor:
			x = input
			y = self.compress(x)  # compress
			y = self.spatial(y)  # spatial
			y = self.sigmoid(y)  # scale (broadcasting)
			y = x * y
			return y
	
	def __init__(
		self,
		channels       : int,
		reduction_ratio: int         = 16,
		pool_types     : list[str]   = ["avg", "max"],
		spatial        : bool | None = True,
	):
		super().__init__()
		self.channel = self.ChannelAttention(
			channels        = channels,
			reduction_ratio = reduction_ratio,
			pool_types      = pool_types,
		)
		self.spatial = self.SpatialAttention() if spatial is True else None
	
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		x = input
		y = x
		y = self.channel(y)
		if self.spatial is not None:
			y = self.spatial(y)
		return y


class ChannelAttentionModule(nn.Module):
	"""Channel Attention Module."""
	
	def __init__(
		self,
		channels       : int,
		reduction_ratio: int,
		stride         : _size_2_t       = 1,
		padding        : _size_2_t | str = 0,
		dilation       : _size_2_t       = 1,
		groups         : int             = 1,
		bias           : bool            = True,
		padding_mode   : str             = "zeros",
		device         : Any             = None,
		dtype          : Any             = None,
	):
		super().__init__()
		self.avg_pool   = pooling.AdaptiveAvgPool2d(1)
		self.excitation = nn.Sequential(
			conv.Conv2d(
				in_channels  = channels,
				out_channels = channels // reduction_ratio,
				kernel_size  = 1,
				stride       = stride,
				padding      = 0,
				dilation     = dilation,
				groups       = groups,
				bias         = True,
				padding_mode = padding_mode,
				device       = device,
				dtype        = dtype,
			),
			act.ReLU(inplace=True),
			conv.Conv2d(
				in_channels  = channels // reduction_ratio,
				out_channels = channels,
				kernel_size  = 1,
				padding      = 0,
				dilation     = dilation,
				groups       = groups,
				bias         = True,
				padding_mode = padding_mode,
				device       = device,
				dtype=dtype,
			),
			act.Sigmoid(),
		)
	
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		x = input
		y = self.avg_pool(x)
		y = self.excitation(y)
		y = x * y
		return y

# endregion


# region SimAm

class SimAM(nn.Module):
	"""SimAM adopted from paper: "SimAM: A Simple, Parameter-Free Attention
	Module for Convolutional Neural Networks".
	
	References:
		- `<https://github.com/ZjjConan/SimAM>`__
	"""
	
	def __init__(self, e_lambda: float = 1e-4, *args, **kwargs):
		super().__init__()
		self.e_lambda = e_lambda
		self.act      = act.Sigmoid()
	
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		x = input
		# Spatial size
		b, c, h, w = x.size()
		n = w * h - 1
		# Square of (t - u)
		d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
		# d.sum() / n is channel variance
		v = d.sum(dim=[2, 3], keepdim=True) / n
		# E_inv groups all important of x
		e_inv = d / (4 * (v + self.e_lambda)) + 0.5
		# Attended features
		y = x * self.act(e_inv)
		return y

# endregion
