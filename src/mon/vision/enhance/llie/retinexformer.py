#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Retinexformer models."""

from __future__ import annotations

__all__ = [
	"Retinexformer_RE",
]

import math
from typing import Any, Literal

import torch
from einops import rearrange

from mon import core, nn
from mon.core import _callable, _size_2_t
from mon.globals import MODELS, Scheme
from mon.nn import functional as F
from mon.vision.enhance.llie import base

console       = core.console
error_console = core.error_console


# region Module

def trunc_normal_(
	tensor: torch.Tensor,
	mean  : float = 0.0,
	std   : float = 1.0,
	a     : float = -2.0,
	b     : float = 2.0,
) -> torch.Tensor:
	def norm_cdf(x):
		return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
	
	if (mean < a - 2 * std) or (mean > b + 2 * std):
		error_console.log(
			f":param:`mean` is more than 2 std from [a, b] in "
			f":func:`torch.nn.init.trunc_normal_`. "
			f"The distribution of values may be incorrect.",
		)
	with torch.no_grad():
		l = norm_cdf((a - mean) / std)
		u = norm_cdf((b - mean) / std)
		tensor.uniform_(2 * l - 1, 2 * u - 1)
		tensor.erfinv_()
		tensor.mul_(std * math.sqrt(2.))
		tensor.add_(mean)
		tensor.clamp_(min=a, max=b)
		return tensor


def variance_scaling_(
	tensor      : torch.Tensor,
	scale       : float = 1.0,
	mode        : Literal["fan_in", "fan_out", "fan_avg"]		   = "fan_in",
	distribution: Literal["truncated_normal", "normal", "uniform"] = "normal"
):
	fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
	if mode == "fan_in":
		denom = fan_in
	elif mode == "fan_out":
		denom = fan_out
	elif mode == "fan_avg":
		denom = (fan_in + fan_out) / 2
	else:
		raise ValueError(
			f":param:`mode` must be one of ``'fan_in'``, ``'fan_out'``, or "
			f"``'fan_avg'``, but got {mode}."
		)
	variance = scale / denom
	if distribution == "truncated_normal":
		trunc_normal_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
	elif distribution == "normal":
		tensor.normal_(std=math.sqrt(variance))
	elif distribution == "uniform":
		bound = math.sqrt(3 * variance)
		tensor.uniform_(-bound, bound)
	else:
		raise ValueError(
			f":param:`distribution` must be one of ``'truncated_normal'``, "
			f"``'normal'``, or  ``'uniform'``, but got {distribution}."
		)
	

def lecun_normal_(tensor: torch.Tensor):
	variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


def shift_back(inputs: torch.Tensor, step: int = 2) -> torch.Tensor:
	"""input [bs,28,256,310] --> output [bs, 28, 256, 256]."""
	b, c, h, w  = inputs.shape
	down_sample = 256 // h
	step 		= float(step) / float(down_sample * down_sample)
	out_col 	= h
	for i in range(c):
		inputs[:, i, :, :out_col] = inputs[:, i, :, int(step * i):int(step * i) + out_col]
	return inputs[:, :, :, :out_col]


def conv(
	in_channels : int,
	out_channels: int,
	kernel_size : _size_2_t,
	stride      : _size_2_t = 1,
	bias        : bool      = False,
) -> nn.Module:
	return nn.Conv2d(
		in_channels  = in_channels,
		out_channels = out_channels,
		kernel_size  = kernel_size,
		stride       = stride,
		padding      = (kernel_size // 2),
		bias         = bias,
	)


class PreNorm(nn.Module):
	
	def __init__(self, normalized_shape: int | list[int], fn: _callable):
		super().__init__()
		self.fn   = fn
		self.norm = nn.LayerNorm(normalized_shape)
	
	def forward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
		x = input
		x = self.norm(x)
		return self.fn(x, *args, **kwargs)


class IlluminationEstimator(nn.Module):
	
	def __init__(
		self,
		mid_channels: int,
		in_channels : int = 4,
		out_channels: int = 3,
	):
		super().__init__()
		self.conv1 		= nn.Conv2d(in_channels,  mid_channels, 1, bias=True)
		self.depth_conv = nn.Conv2d(mid_channels, mid_channels, 5, padding=2, bias=True, groups=in_channels)
		self.conv2 		= nn.Conv2d(mid_channels, out_channels, 1, bias=True)
	
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		x 	     = input
		mean_c 	 = x.mean(dim=1).unsqueeze(1)
		x 	 	 = torch.cat([x, mean_c], dim=1)
		x_1 	 = self.conv1(x)
		illu_fea = self.depth_conv(x_1)
		illu_map = self.conv2(illu_fea)
		return illu_fea, illu_map


class IGMSA(nn.Module):
	"""Illumination-Guided Multi-head Self-Attention."""
	
	def __init__(
		self,
		in_channels  : int,
		head_channels: int = 64,
		num_heads    : int = 8,
	):
		super().__init__()
		self.num_heads = num_heads
		self.dim_head  = head_channels
		self.to_q 	   = nn.Linear(in_channels, head_channels * num_heads, bias=False)
		self.to_k 	   = nn.Linear(in_channels, head_channels * num_heads, bias=False)
		self.to_v 	   = nn.Linear(in_channels, head_channels * num_heads, bias=False)
		self.rescale   = nn.Parameter(torch.ones(num_heads, 1, 1))
		self.proj      = nn.Linear(head_channels * num_heads, in_channels, bias=True)
		self.pos_emb   = nn.Sequential(
			nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False, groups=in_channels),
			nn.GELU(),
			nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False, groups=in_channels),
		)
		self.dim = in_channels
	
	def forward(self, input: torch.Tensor, illu_fea_trans: torch.Tensor) -> torch.Tensor:
		x 		   = input
		b, h, w, c = x.shape
		x 		   = x.reshape(b, h * w, c)
		q_inp 	   = self.to_q(x)
		k_inp 	   = self.to_k(x)
		v_inp 	   = self.to_v(x)
		illu_attn  = illu_fea_trans  # illu_fea: b,c,h,w -> b,h,w,c
		q, k, v, illu_attn = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))
		v     = v * illu_attn
		# q: b,heads,hw,c
		q     = q.transpose(-2, -1)
		k     = k.transpose(-2, -1)
		v     = v.transpose(-2, -1)
		q     = F.normalize(q, dim=-1, p=2)
		k     = F.normalize(k, dim=-1, p=2)
		attn  = (k @ q.transpose(-2, -1))  # A = K^T*Q
		attn  = attn * self.rescale
		attn  = attn.softmax(dim=-1)
		x     = attn @ v  # b,heads,d,hw
		x     = x.permute(0, 3, 1, 2)  # Transpose
		x     = x.reshape(b, h * w, self.num_heads * self.dim_head)
		out_c = self.proj(x).view(b, h, w, c)
		out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
		y = out_c + out_p
		return y


class FeedForward(nn.Module):
	
	def __init__(self, in_channels: int, multiplier: int = 4):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_channels, in_channels * multiplier, 1, 1, bias=False),
			nn.GELU(),
			nn.Conv2d(in_channels * multiplier, in_channels * multiplier, 3, 1, 1, bias=False, groups=in_channels * multiplier),
			nn.GELU(),
			nn.Conv2d(in_channels * multiplier, in_channels, 1, 1, bias=False),
		)
	
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		x = input
		y = self.net(x.permute(0, 3, 1, 2))
		y = y.permute(0, 2, 3, 1)
		return y
		

class IGAB(nn.Module):
	
	def __init__(
		self,
		in_channels  : int,
		head_channels: int = 64,
		num_heads    : int = 8,
		num_blocks   : int = 2,
	):
		super().__init__()
		self.blocks = nn.ModuleList([])
		for _ in range(num_blocks):
			self.blocks.append(
				nn.ModuleList([
					IGMSA(in_channels=in_channels, head_channels=head_channels, num_heads=num_heads),
					PreNorm(in_channels, FeedForward(in_channels=in_channels))
				])
			)
	
	def forward(self, input: torch.Tensor, illu_fea: torch.Tensor) -> torch.Tensor:
		x = input
		x = x.permute(0, 2, 3, 1)
		for (attn, ff) in self.blocks:
			x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
			x = ff(x) + x
		y = x.permute(0, 3, 1, 2)
		return y


class Denoiser(nn.Module):
	
	def __init__(
		self,
		in_channels : int       = 3,
		out_channels: int       = 3,
		num_channels: int       = 31,
		level       : int       = 2,
		num_blocks  : list[int] = [2, 4, 4],
	):
		super().__init__()
		self.num_channels = num_channels
		self.level        = level
		
		# Input projection
		self.embedding = nn.Conv2d(in_channels, self.num_channels, 3, 1, 1, bias=False)
		
		# Encoder
		self.encoder_layers = nn.ModuleList([])
		dim_level = num_channels
		for i in range(level):
			self.encoder_layers.append(
				nn.ModuleList([
					IGAB(in_channels=dim_level, num_blocks=num_blocks[i], head_channels=num_channels, num_heads=dim_level // num_channels),
					nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
					nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
				])
			)
			dim_level *= 2
		
		# Bottleneck
		self.bottleneck = IGAB(in_channels=dim_level, head_channels=num_channels, num_heads=dim_level // num_channels, num_blocks=num_blocks[-1])
		
		# Decoder
		self.decoder_layers = nn.ModuleList([])
		for i in range(level):
			self.decoder_layers.append(
				nn.ModuleList([
					nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
					nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
					IGAB(in_channels=dim_level // 2, num_blocks=num_blocks[level - 1 - i], head_channels=num_channels, num_heads=(dim_level // 2) // num_channels),
				])
			)
			dim_level //= 2
		
		# Output projection
		self.mapping = nn.Conv2d(self.num_channels, out_channels, 3, 1, 1, bias=False)
		
		# activation function
		self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
		self.apply(self._init_weights)
	
	def _init_weights(self, m: nn.Module):
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=0.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				torch.nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			torch.nn.init.constant_(m.bias, 0)
			torch.nn.init.constant_(m.weight, 1.0)
	
	def forward(self, input: torch.Tensor, illu_fea: torch.Tensor) -> torch.Tensor:
		x = input
		
		# Embedding
		fea = self.embedding(x)
		
		# Encoder
		fea_encoder   = []
		illu_fea_list = []
		for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
			fea = IGAB(fea, illu_fea)
			fea_encoder.append(fea)
			illu_fea_list.append(illu_fea)
			fea 	 = FeaDownSample(fea)
			illu_fea = IlluFeaDownsample(illu_fea)
		
		# Bottleneck
		fea = self.bottleneck(fea, illu_fea)
		
		# Decoder
		for i, (FeaUpSample, FeaFusion, LeWinBlock) in enumerate(self.decoder_layers):
			fea 	 = FeaUpSample(fea)
			fea		 = FeaFusion(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
			illu_fea = illu_fea_list[self.level-1-i]
			fea 	 = LeWinBlock(fea, illu_fea)
		
		# Mapping
		y = self.mapping(fea) + x
		return y


class RetinexFormerSingleStage(nn.Module):
	
	def __init__(
		self,
		in_channels : int       = 3,
		out_channels: int       = 3,
		num_channels: int       = 31,
		level       : int       = 2,
		num_blocks  : list[int] = [1, 1, 1],
	):
		super().__init__()
		self.estimator = IlluminationEstimator(num_channels)
		self.denoiser  = Denoiser(
			in_channels  = in_channels,
			out_channels = out_channels,
			num_channels = num_channels,
			level        = level,
			num_blocks   = num_blocks,
		)
	
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		x = input
		illu_fea, illu_map = self.estimator(x)
		input_img = x * illu_map + x
		y 		  = self.denoiser(input_img, illu_fea)
		return y
	
# endregion


# region Model

@MODELS.register(name="retinexformer_re", arch="retinexformer")
class Retinexformer_RE(base.LowLightImageEnhancementModel):
	"""`Retinexformer: One-stage Retinex-based Transformer for Low-light Image
	Enhancement <https://arxiv.org/abs/2303.06705>`__.
	
	References:
		`<https://github.com/caiyuanhao1998/Retinexformer>`__

	See Also: :class:`base.LowLightImageEnhancementModel`
	"""
	
	arch   : str  = "retinexformer"
	schemes: list[Scheme] = [Scheme.SUPERVISED]
	zoo    : dict = {
		"fivek" : {
			"url"         : None,
			"path"        : "retinexformer/retinexformer_fivek",
			"channels"    : 3,
			"num_channels": 40,
			"stage"	      : 1,
			"num_blocks"  : [1, 2, 2],
			"num_classes" : None,
			"map": {},
		},
		"lol_v1": {
			"url"         : None,
			"path"        : "retinexformer/retinexformer_lol_v1",
			"channels"    : 3,
			"num_channels": 40,
			"stage"	      : 1,
			"num_blocks"  : [1, 2, 2],
			"num_classes" : None,
			"map": {},
		},
		"lol_v2_real" : {
			"url"         : None,
			"path"        : "retinexformer/retinexformer_lol_v2_real",
			"channels"    : 3,
			"num_channels": 40,
			"stage"	      : 1,
			"num_blocks"  : [1, 2, 2],
			"num_classes" : None,
			"map": {},
		},
		"lol_v2_syn"  : {
			"url"         : None,
			"path"        : "retinexformer/retinexformer_lol_v2_syn",
			"channels"    : 3,
			"num_channels": 40,
			"stage"	      : 1,
			"num_blocks"  : [1, 2, 2],
			"num_classes" : None,
			"map": {},
		},
		"sdsd_indoor" : {
			"url"         : None,
			"path"        : "retinexformer/retinexformer_sdsd_indoor",
			"channels"    : 3,
			"num_channels": 40,
			"stage"	      : 1,
			"num_blocks"  : [1, 2, 2],
			"num_classes" : None,
			"map": {},
		},
		"sdsd_outdoor": {
			"url"         : None,
			"path"        : "retinexformer/retinexformer_sdsd_outdoor",
			"channels"    : 3,
			"num_channels": 40,
			"stage"	      : 1,
			"num_blocks"  : [1, 2, 2],
			"num_classes" : None,
			"map": {},
		},
		"sid" : {
			"url"         : None,
			"path"        : "retinexformer/retinexformer_sid",
			"channels"    : 3,
			"num_channels": 40,
			"stage"	      : 1,
			"num_blocks"  : [1, 2, 2],
			"num_classes" : None,
			"map": {},
		},
		"smid": {
			"url"         : None,
			"path"        : "retinexformer/retinexformer_smid",
			"channels"    : 3,
			"num_channels": 40,
			"stage"	      : 1,
			"num_blocks"  : [1, 2, 2],
			"num_classes" : None,
			"map": {},
		},
	}
	
	def __init__(
		self,
		in_channels : int       = 3,
		num_channels: int       = 31,
		stage       : int       = 3,
		num_blocks  : list[int] = [1, 1, 1],
		weights     : Any       = None,
		*args, **kwargs
	):
		super().__init__(
			name        = "retinexformer_re",
			in_channels = in_channels,
			weights     = weights,
			*args, **kwargs
		)
		
		# Populate hyperparameter values from pretrained weights
		if isinstance(self.weights, dict):
			in_channels  = self.weights.get("in_channels" , in_channels)
			num_channels = self.weights.get("num_channels", num_channels)
			stage        = self.weights.get("stage"       , stage)
			num_blocks   = self.weights.get("num_blocks"  , num_blocks)
		self.in_channels  = in_channels
		self.num_channels = num_channels
		self.stage        = stage
		self.num_blocks   = num_blocks
		
		# Construct model
		self.body = nn.Sequential(*[
			RetinexFormerSingleStage(
				in_channels  = self.in_channels,
				out_channels = self.in_channels,
				num_channels = self.num_channels,
				level        = 2,
				num_blocks   = self.num_blocks,
			)
			for _ in range(stage)
		])
		
		# Load weights
		if self.weights:
			self.load_weights()
		else:
			self.apply(self.init_weights)
	
	def init_weights(self, m: nn.Module):
		pass
	
	def forward(
		self,
		input    : torch.Tensor,
		augment  : _callable = None,
		profile  : bool      = False,
		out_index: int       = -1,
		*args, **kwargs
	) -> torch.Tensor:
		x = input
		y = self.body(x)
		return y

# endregion
