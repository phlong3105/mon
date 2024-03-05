#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements pixel attention modules."""

from __future__ import annotations

__all__ = [
	"PAM",
	"PixelAttentionModule",
]

from typing import Any

import torch
from torch import nn

from mon.core import _size_2_t
from mon.nn.layer import activation as act, conv


# region Pixel Attention Module

class PixelAttentionModule(nn.Module):
	"""Pixel Attention Module."""
	
	def __init__(
		self,
		channels       : int,
		reduction_ratio: int,
		kernel_size    : _size_2_t,
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
		self.fc = nn.Sequential(
			conv.Conv2d(
				in_channels  = channels,
				out_channels = channels // reduction_ratio,
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
			act.ReLU(inplace=True),
			conv.Conv2d(
				in_channels  = channels // reduction_ratio,
				out_channels = 1,
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
		self.act = act.Sigmoid()
	
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		x = input
		y = self.fc(x)
		y = self.act(y)
		y = torch.mul(x, y)
		return y


PAM = PixelAttentionModule

# endregion
