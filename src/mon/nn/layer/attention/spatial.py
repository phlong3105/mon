#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements spatial attention modules."""

from __future__ import annotations

__all__ = [
	"ShiftedWindowAttention",
	"ShiftedWindowAttentionV2",
	"WindowAttention",
]

from typing import Any

import torch
from einops import repeat
from torch import nn
from torch.nn import functional as F

from mon.core import math
from mon.globals import LAYERS
from mon.nn.layer import activation as act, dropout as drop, linear, projection


# region Spatial Attention

def shifted_window_attention(
	input                 : torch.Tensor,
	qkv_weight            : torch.Tensor,
	proj_weight           : torch.Tensor,
	relative_position_bias: torch.Tensor,
	window_size           : list[int],
	num_heads             : int,
	shift_size            : list[int],
	attention_dropout     : float               = 0.0,
	dropout               : float               = 0.0,
	qkv_bias              : torch.Tensor | None = None,
	proj_bias             : torch.Tensor | None = None,
	logit_scale           : torch.Tensor | None = None,
	training              : bool                = True,
) -> torch.Tensor:
	"""Window-based multi-head self-attention (W-MSA) module with relative
	position bias. It supports both shifted and non-shifted windows.
	
	Args:
		input: An input of shape :math:`[N, C, H, W]`.
		qkv_weight: The weight tensor of query, key, value of shape
			:math:`[in_dim, out_dim]`.
		proj_weight: The weight tensor of projection of shape
			:math:`[in_dim, out_dim]`.
		relative_position_bias: The learned relative position bias added to
			attention.
		window_size: Window size.
		num_heads: Number of attention heads.
		shift_size: Shift size for shifted window attention.
		attention_dropout: Dropout ratio of attention weight. Default: ``0.0``.
		dropout: Dropout ratio of output. Default: ``0.0``.
		qkv_bias: The bias tensor of query, key, value. Default: ``None``.
		proj_bias: The bias tensor of projection. Default: ``None``.
		logit_scale: Logit scale of cosine attention for Swin Transformer V2.
			Default: ``None``.
		training: Training flag used by the dropout parameters. Default: ``True``.
	
	Returns:
		The output tensor after shifted window attention of shape :math:`[N, C, H, W]`.
	"""
	b, h, w, c = input.shape
	# Pad feature maps to multiples of window size
	pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
	pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
	x     = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
	_, pad_h, pad_w, _ = x.shape
	
	shift_size = shift_size.copy()
	# If window size is larger than feature size, there is no need to shift window
	if window_size[0] >= pad_h:
		shift_size[0] = 0
	if window_size[1] >= pad_w:
		shift_size[1] = 0
	
	# Cyclic shift
	if sum(shift_size) > 0:
		x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
	
	# partition windows
	num_windows = (pad_h // window_size[0]) * (pad_w // window_size[1])
	x = x.view(b, pad_h // window_size[0], window_size[0], pad_w // window_size[1], window_size[1], c)
	x = x.permute(0, 1, 3, 2, 4, 5).reshape(b * num_windows, window_size[0] * window_size[1], c)  # B*nW, Ws*Ws, C
	
	# Multi-head attention
	if logit_scale is not None and qkv_bias is not None:
		qkv_bias = qkv_bias.clone()
		length   = qkv_bias.numel() // 3
		qkv_bias[length : 2 * length].zero_()
	qkv     = F.linear(x, qkv_weight, qkv_bias)
	qkv     = qkv.reshape(x.size(0), x.size(1), 3, num_heads, c // num_heads).permute(2, 0, 3, 1, 4)
	q, k, v = qkv[0], qkv[1], qkv[2]
	if logit_scale is not None:
		# Cosine attention
		attn        = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
		logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
		attn        = attn * logit_scale
	else:
		q    = q * (c // num_heads) ** -0.5
		attn = q.matmul(k.transpose(-2, -1))
	# Add relative position bias
	attn = attn + relative_position_bias
	
	if sum(shift_size) > 0:
		# Generate attention mask
		attn_mask = x.new_zeros((pad_h, pad_w))
		h_slices  = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
		w_slices  = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
		count     = 0
		for h in h_slices:
			for w in w_slices:
				attn_mask[h[0] : h[1], w[0] : w[1]] = count
				count += 1
		attn_mask = attn_mask.view(pad_h // window_size[0], window_size[0], pad_w // window_size[1], window_size[1])
		attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
		attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
		attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
		attn      = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
		attn      = attn + attn_mask.unsqueeze(1).unsqueeze(0)
		attn      = attn.view(-1, num_heads, x.size(1), x.size(1))
	
	attn = F.softmax(attn, dim=-1)
	attn = F.dropout(attn, p=attention_dropout, training=training)
	
	x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), c)
	x = F.linear(x, proj_weight, proj_bias)
	x = F.dropout(x, p=dropout, training=training)
	
	# Reverse windows
	x = x.view(b, pad_h // window_size[0], pad_w // window_size[1], window_size[0], window_size[1], c)
	x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, pad_h, pad_w, c)
	
	# Reverse cyclic shift
	if sum(shift_size) > 0:
		x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
	
	# Unpad features
	x = x[:, :h, :w, :].contiguous()
	return x


@LAYERS.register()
class ShiftedWindowAttention(nn.Module):
	"""See Also :func:`shifted_window_attention`."""
	
	def __init__(
		self,
		channels         : int,
		window_size      : list[int],
		shift_size       : list[int],
		num_heads        : int,
		qkv_bias         : bool  = True,
		proj_bias        : bool  = True,
		attention_dropout: float = 0.0,
		dropout          : float = 0.0,
		*args, **kwargs
	):
		super().__init__()
		if len(window_size) != 2 or len(shift_size) != 2:
			raise ValueError(f":param:`window_size` and :param:`shift_size` must be of length ``2``.")
		self.channels          = channels
		self.window_size       = window_size
		self.shift_size        = shift_size
		self.num_heads         = num_heads
		self.attention_dropout = attention_dropout
		self.dropout           = dropout
		
		self.qkv  = linear.Linear(channels, channels * 3, bias=qkv_bias)
		self.proj = linear.Linear(channels, channels, bias=proj_bias)
		
		self.define_relative_position_bias_table()
		self.define_relative_position_index()
	
	def define_relative_position_bias_table(self):
		# define a parameter table of relative position bias
		self.relative_position_bias_table = nn.Parameter(
			torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads)
		)  # 2*Wh-1 * 2*Ww-1, nH
		torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
	
	def define_relative_position_index(self):
		# Get pair-wise relative position index for each token inside the window
		coords_h = torch.arange(self.window_size[0])
		coords_w = torch.arange(self.window_size[1])
		coords   = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
		coords_flatten  = torch.flatten(coords, 1)  # 2, Wh*Ww
		relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
		relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
		relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
		relative_coords[:, :, 1] += self.window_size[1] - 1
		relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
		relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww
		self.register_buffer("relative_position_index", relative_position_index)
	
	def calculate_relative_position_bias(
		self,
		relative_position_bias_table: torch.Tensor,
		relative_position_index     : torch.Tensor,
		window_size                 : list[int],
	) -> torch.Tensor:
		n = window_size[0] * window_size[1]
		relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
		relative_position_bias = relative_position_bias.view(n, n, -1)
		relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
		return relative_position_bias
	
	def get_relative_position_bias(self) -> torch.Tensor:
		return self.calculate_relative_position_bias(
			self.relative_position_bias_table,
			self.relative_position_index,
			self.window_size  # type: ignore[arg-type]
		)
	
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		"""
		
		Args:
			input: Tensor of shape :math:`[B, H, W, C]`.
	  
		Returns:
			Tensor of shape :math:`[B, H, W, C]`.
		"""
		x = input
		relative_position_bias = self.get_relative_position_bias()
		return shifted_window_attention(
			input                  = x,
			qkv_weight             = self.qkv.weight,
			proj_weight            = self.proj.weight,
			relative_position_bias = relative_position_bias,
			window_size            = self.window_size,
			num_heads              = self.num_heads,
			shift_size             = self.shift_size,
			attention_dropout      = self.attention_dropout,
			dropout                = self.dropout,
			qkv_bias               = self.qkv.bias,
			proj_bias              = self.proj.bias,
			training               = self.training,
		)


@LAYERS.register()
class ShiftedWindowAttentionV2(ShiftedWindowAttention):
	"""See Also: :class:`ShiftedWindowAttention`."""
	
	def __init__(
		self,
		channels         : int,
		window_size      : list[int],
		shift_size       : list[int],
		num_heads        : int,
		qkv_bias         : bool  = True,
		proj_bias        : bool  = True,
		attention_dropout: float = 0.0,
		dropout          : float = 0.0,
	):
		super().__init__(
			channels          = channels,
			window_size       = window_size,
			shift_size        = shift_size,
			num_heads         = num_heads,
			qkv_bias          = qkv_bias,
			proj_bias         = proj_bias,
			attention_dropout = attention_dropout,
			dropout           = dropout,
		)
		
		self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
		# MLP to generate continuous relative position bias
		self.cpb_mlp = nn.Sequential(
			linear.Linear(2, 512, bias=True),
			act.ReLU(inplace=True),
			linear.Linear(512, num_heads, bias=False)
		)
		if qkv_bias:
			length = self.qkv.bias.numel() // 3
			self.qkv.bias[length : 2 * length].data.zero_()
	
	def define_relative_position_bias_table(self):
		# Get relative_coords_table
		relative_coords_h     = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
		relative_coords_w     = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
		relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w], indexing="ij"))
		relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
		
		relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
		relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
		
		relative_coords_table *= 8  # normalize to -8, 8
		relative_coords_table = (torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / 3.0)
		self.register_buffer("relative_coords_table", relative_coords_table)
	
	def get_relative_position_bias(self) -> torch.Tensor:
		relative_position_bias = self.calculate_relative_position_bias(
			self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads),
			self.relative_position_index,  # type: ignore[arg-type]
			self.window_size,
		)
		relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
		return relative_position_bias
	
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		"""
		
		Args:
			input: Tensor of shape :math:`[B, H, W, C]`.
	  
		Returns:
			Tensor of shape :math:`[B, H, W, C]`.
		"""
		x = input
		relative_position_bias = self.get_relative_position_bias()
		return shifted_window_attention(
			input                  = x,
			qkv_weight             = self.qkv.weight,
			proj_weight            = self.proj.weight,
			relative_position_bias = relative_position_bias,
			window_size            = self.window_size,
			num_heads              = self.num_heads,
			shift_size             = self.shift_size,
			attention_dropout      = self.attention_dropout,
			dropout                = self.dropout,
			qkv_bias               = self.qkv.bias,
			proj_bias              = self.proj.bias,
			training               = self.training,
		)


@LAYERS.register()
class WindowAttention(nn.Module):
	
	def __init__(
		self,
		channels        : int,
		window_size     : list[int],
		num_heads       : int,
		token_projection: str   = "linear",
		qkv_bias        : bool  = True,
		qk_scale        : Any   = None,
		attn_drop       : float = 0.0,
		proj_drop       : float = 0.0,
	):
		super().__init__()
		self.channels    = channels
		self.window_size = window_size  # Wh, Ww
		self.num_heads   = num_heads
		head_dim         = channels // num_heads
		self.scale       = qk_scale or head_dim ** -0.5
		
		# Define a parameter table of relative position bias
		self.relative_position_bias_table = nn.Parameter(
			torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
		)  # 2*Wh-1 * 2*Ww-1, nH
		
		# Get pair-wise relative position index for each token inside the window
		coords_h = torch.arange(self.window_size[0])  # [0,...,Wh-1]
		coords_w = torch.arange(self.window_size[1])  # [0,...,Ww-1]
		coords   = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
		coords_flatten  = torch.flatten(coords, 1)  # 2, Wh*Ww
		relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
		relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
		relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
		relative_coords[:, :, 1] += self.window_size[1] - 1
		relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
		relative_position_index   = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
		self.register_buffer("relative_position_index", relative_position_index)
		torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
		
		if token_projection == "conv":
			self.qkv = projection.ConvProjection(channels, num_heads, channels // num_heads, bias=qkv_bias)
		elif token_projection == "linear":
			self.qkv = projection.LinearProjection(channels, num_heads, channels // num_heads, bias=qkv_bias)
		else:
			raise Exception("Projection error!")
		
		self.token_projection = token_projection
		self.attn_drop        = drop.Dropout(attn_drop)
		self.proj             = linear.Linear(channels, channels)
		self.proj_drop        = drop.Dropout(proj_drop)
		self.softmax          = act.Softmax(dim=-1)
	
	def forward(
		self,
		input  : torch.Tensor,
		attn_kv: torch.Tensor | None = None,
		mask   : torch.Tensor | None = None,
	) -> torch.Tensor:
		x        = input
		b_, n, c = x.shape
		q, k, v  = self.qkv(x, attn_kv)
		q        = q * self.scale
		attn     = (q @ k.transpose(-2, -1))
		
		relative_position_bias = self.relative_position_bias_table[
			self.relative_position_index.view(-1)].view(
			self.window_size[0] * self.window_size[1],
			self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
		relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
		ratio = attn.size(-1) // relative_position_bias.size(-1)
		relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)
		
		attn = attn + relative_position_bias.unsqueeze(0)
		
		if mask is not None:
			nW   = mask.shape[0]
			mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
			attn = attn.view(b_ // nW, nW, self.num_heads, n, n * ratio) + mask.unsqueeze(1).unsqueeze(0)
			attn = attn.view(-1, self.num_heads, n, n * ratio)
			attn = self.softmax(attn)
		else:
			attn = self.softmax(attn)
		
		attn = self.attn_drop(attn)
		
		x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x
	
	def extra_repr(self) -> str:
		return f"dim={self.channels}, win_size={self.window_size}, num_heads={self.num_heads}"
	
	def flops(self, h, w):
		# Calculate flops for 1 window with token length of N
		# print(N, self.dim)
		flops = 0
		N     = self.window_size[0] * self.window_size[1]
		nW    = h * w / N
		# qkv = self.qkv(x)
		# flops += N * self.dim * 3 * self.dim
		flops += self.qkv.flops(h * w, h * w)
		# attn = (q @ k.transpose(-2, -1))
		flops += nW * self.num_heads * N * (self.channels // self.num_heads) * N
		#  x = (attn @ v)
		flops += nW * self.num_heads * N * N * (self.channels // self.num_heads)
		# x = self.proj(x)
		flops += nW * N * self.channels * self.channels
		# print("W-MSA:{%.2f}" % (flops / 1e9))
		return flops

# endregion
