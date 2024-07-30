#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements MPRNet (Multi-Stage Progressive Image Restoration)
models.
"""

from __future__ import annotations

__all__ = [
	"MPRNet",
]

from typing import Any, Sequence

import torch

from mon import core, nn
from mon.core import _callable, _size_2_t
from mon.globals import MODELS, Scheme, Task
from mon.vision.enhance.multitask import base

console = core.console


# region Module

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
		padding      = (kernel_size  // 2),
		bias         = bias,
	)


class DownSample(nn.Module):
	
	def __init__(self, in_channels: int, s_factor: int):
		super().__init__()
		self.down = nn.Sequential(
			nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=False),
			nn.Conv2d(
				in_channels  = in_channels,
				out_channels = in_channels + s_factor,
				kernel_size  = 1,
				stride       = 1,
				padding      = 0,
				bias         = False,
			)
		)
	
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		x = input
		y = self.down(x)
		return y


class UpSample(nn.Module):
	
	def __init__(self, in_channels: int, s_factor: int):
		super().__init__()
		self.up = nn.Sequential(
			nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
			nn.Conv2d(
				in_channels  = in_channels+s_factor,
				out_channels = in_channels,
				kernel_size  = 1,
				stride       = 1,
				padding      = 0,
				bias         = False,
			)
		)
	
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		x = input
		y = self.up(x)
		return y


class SkipUpSample(nn.Module):
	
	def __init__(self, in_channels: int, s_factor: int):
		super().__init__()
		self.up = nn.Sequential(
			nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
			nn.Conv2d(
				in_channels  = in_channels+s_factor,
				out_channels = in_channels,
				kernel_size  = 1,
				stride       = 1,
				padding      = 0,
				bias         = False
			)
		)
	
	def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		x = self.up(x)
		x = x + y
		return x
	

class CALayer(nn.Module):
	
	def __init__(self, channels: int, reduction: int = 16, bias: bool = False):
		super().__init__()
		# Global average pooling: feature -> point
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		# Feature channel downscale and upscale -> channel weight
		self.conv_du  = nn.Sequential(
			nn.Conv2d(channels, channels // reduction, 1, padding=0, bias=bias),
			nn.ReLU(inplace=True),
			nn.Conv2d(channels // reduction, channels, 1, padding=0, bias=bias),
			nn.Sigmoid()
		)
	
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		x = input
		y = self.avg_pool(x)
		y = self.conv_du(y)
		return x * y


class CAB(nn.Module):
	
	def __init__(
		self,
		channels   : int,
		kernel_size: _size_2_t,
		reduction  : int,
		bias       : bool,
		act_layer  : _callable,
	):
		super().__init__()
		modules_body = []
		modules_body.append(conv(channels, channels, kernel_size, bias=bias))
		modules_body.append(act_layer)
		modules_body.append(conv(channels, channels, kernel_size, bias=bias))
		self.CA   = CALayer(channels, reduction, bias=bias)
		self.body = nn.Sequential(*modules_body)
	
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		x = input
		y = self.body(x)
		y = self.CA(y)
		y += x
		return y


class SAM(nn.Module):
	
	def __init__(self, channels: int, kernel_size: _size_2_t, bias: bool):
		super().__init__()
		self.conv1 = conv(channels, channels, kernel_size, bias=bias)
		self.conv2 = conv(channels, 3, kernel_size, bias=bias)
		self.conv3 = conv(3, channels, kernel_size, bias=bias)
	
	def forward(
		self,
		x    : torch.Tensor,
		x_img: torch.Tensor
	) -> tuple[torch.Tensor, torch.Tensor]:
		x1  = self.conv1(x)
		img = self.conv2(x) + x_img
		x2  = torch.sigmoid(self.conv3(img))
		x1  = x1 * x2
		x1  = x1 + x
		return x1, img


class Encoder(nn.Module):
	
	def __init__(
		self,
		channels       : int,
		kernel_size    : _size_2_t,
		reduction      : int,
		act_layer      : _callable,
		bias           : bool,
		scale_unetfeats: int,
		csff           : bool
	):
		super().__init__()
		self.encoder_level1 = [CAB(channels, 						 kernel_size, reduction, bias=bias, act_layer=act_layer) for _ in range(2)]
		self.encoder_level2 = [CAB(channels + scale_unetfeats, 		 kernel_size, reduction, bias=bias, act_layer=act_layer) for _ in range(2)]
		self.encoder_level3 = [CAB(channels + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act_layer=act_layer) for _ in range(2)]
		self.encoder_level1 = nn.Sequential(*self.encoder_level1)
		self.encoder_level2 = nn.Sequential(*self.encoder_level2)
		self.encoder_level3 = nn.Sequential(*self.encoder_level3)
		self.down12  		= DownSample(channels, scale_unetfeats)
		self.down23  		= DownSample(channels + scale_unetfeats, scale_unetfeats)
		
		# Cross Stage Feature Fusion (CSFF)
		if csff:
			self.csff_enc1 = nn.Conv2d(channels, 						 channels, kernel_size=1, bias=bias)
			self.csff_enc2 = nn.Conv2d(channels + scale_unetfeats, 		 channels + scale_unetfeats, kernel_size=1, bias=bias)
			self.csff_enc3 = nn.Conv2d(channels + (scale_unetfeats * 2), channels + (scale_unetfeats * 2), kernel_size=1, bias=bias)
			self.csff_dec1 = nn.Conv2d(channels, 						 channels, kernel_size=1, bias=bias)
			self.csff_dec2 = nn.Conv2d(channels + scale_unetfeats, 		 channels + scale_unetfeats, kernel_size=1, bias=bias)
			self.csff_dec3 = nn.Conv2d(channels + (scale_unetfeats * 2), channels + (scale_unetfeats * 2), kernel_size=1, bias=bias)
	
	def forward(
		self,
		input		: torch.Tensor,
		encoder_outs: torch.Tensor | None = None,
		decoder_outs: torch.Tensor | None = None,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		x = input
		enc1 = self.encoder_level1(x)
		if (encoder_outs is not None) and (decoder_outs is not None):
			enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])
		
		x = self.down12(enc1)
		
		enc2 = self.encoder_level2(x)
		if (encoder_outs is not None) and (decoder_outs is not None):
			enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])
		
		x = self.down23(enc2)
		
		enc3 = self.encoder_level3(x)
		if (encoder_outs is not None) and (decoder_outs is not None):
			enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])
		
		return [enc1, enc2, enc3]


class Decoder(nn.Module):
	
	def __init__(
		self,
		channels         : int,
		kernel_size    : _size_2_t,
		reduction      : int,
		act_layer      : _callable,
		bias           : bool,
		scale_unetfeats: int
	):
		super().__init__()
		self.decoder_level1 = [CAB(channels, 						 kernel_size, reduction, bias=bias, act_layer=act_layer) for _ in range(2)]
		self.decoder_level2 = [CAB(channels + scale_unetfeats, 		 kernel_size, reduction, bias=bias, act_layer=act_layer) for _ in range(2)]
		self.decoder_level3 = [CAB(channels + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act_layer=act_layer) for _ in range(2)]
		self.decoder_level1 = nn.Sequential(*self.decoder_level1)
		self.decoder_level2 = nn.Sequential(*self.decoder_level2)
		self.decoder_level3 = nn.Sequential(*self.decoder_level3)
		self.skip_attn1 	= CAB(channels, 				  kernel_size, reduction, bias=bias, act_layer=act_layer)
		self.skip_attn2 	= CAB(channels + scale_unetfeats, kernel_size, reduction, bias=bias, act_layer=act_layer)
		self.up21  			= SkipUpSample(channels, scale_unetfeats)
		self.up32  			= SkipUpSample(channels + scale_unetfeats, scale_unetfeats)
	
	def forward(
		self,
		input: Sequence[torch.Tensor]
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		enc1, enc2, enc3 = input
		dec3 = self.decoder_level3(enc3)
		x 	 = self.up32(dec3, self.skip_attn2(enc2))
		dec2 = self.decoder_level2(x)
		x 	 = self.up21(dec2, self.skip_attn1(enc1))
		dec1 = self.decoder_level1(x)
		return [dec1,dec2,dec3]


class ORB(nn.Module):
	"""Original Resolution Block (ORB)."""
	
	def __init__(
		self,
		channels   : int,
		kernel_size: _size_2_t,
		reduction  : int,
		act_layer  : _callable,
		bias       : bool,
		num_cab    : int,
	):
		super().__init__()
		modules_body = [
			CAB(channels, kernel_size, reduction, bias=bias, act_layer=act_layer)
			for _ in range(num_cab)
		]
		modules_body.append(conv(channels, channels, kernel_size))
		self.body = nn.Sequential(*modules_body)
	
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		x = input
		y = self.body(x)
		y += x
		return y


class ORSNet(nn.Module):
	
	def __init__(
		self,
		channels         : int,
		scale_orsnetfeats: int,
		kernel_size      : _size_2_t,
		reduction        : int,
		act_layer        : _callable,
		bias             : bool,
		scale_unetfeats  : int,
		num_cab          : int,
	):
		super().__init__()
		self.orb1 	   = ORB(channels + scale_orsnetfeats, kernel_size, reduction, act_layer, bias, num_cab)
		self.orb2 	   = ORB(channels + scale_orsnetfeats, kernel_size, reduction, act_layer, bias, num_cab)
		self.orb3      = ORB(channels + scale_orsnetfeats, kernel_size, reduction, act_layer, bias, num_cab)
		self.up_enc1   = UpSample(channels, scale_unetfeats)
		self.up_dec1   = UpSample(channels, scale_unetfeats)
		self.up_enc2   = nn.Sequential(
			UpSample(channels + scale_unetfeats, scale_unetfeats),
			UpSample(channels, scale_unetfeats)
		)
		self.up_dec2   = nn.Sequential(
			UpSample(channels + scale_unetfeats, scale_unetfeats),
			UpSample(channels, scale_unetfeats)
		)
		self.conv_enc1 = nn.Conv2d(channels, channels + scale_orsnetfeats, kernel_size=1, bias=bias)
		self.conv_enc2 = nn.Conv2d(channels, channels + scale_orsnetfeats, kernel_size=1, bias=bias)
		self.conv_enc3 = nn.Conv2d(channels, channels + scale_orsnetfeats, kernel_size=1, bias=bias)
		self.conv_dec1 = nn.Conv2d(channels, channels + scale_orsnetfeats, kernel_size=1, bias=bias)
		self.conv_dec2 = nn.Conv2d(channels, channels + scale_orsnetfeats, kernel_size=1, bias=bias)
		self.conv_dec3 = nn.Conv2d(channels, channels + scale_orsnetfeats, kernel_size=1, bias=bias)
	
	def forward(
		self,
		input       : torch.Tensor,
		encoder_outs: torch.Tensor,
		decoder_outs: torch.Tensor,
	) -> torch.Tensor:
		x = input
		x = self.orb1(x)
		x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])
		x = self.orb2(x)
		x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))
		x = self.orb3(x)
		x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))
		return x

# endregion


# region Model

@MODELS.register(name="mprnet", arch="mprnet")
class MPRNet(base.MultiTaskImageEnhancementModel):
	"""Multi-Stage Progressive Image Restoration.
	
	Args:
		in_channels: The first layer's input channel. Default: ``3`` for RGB image.
		num_channels: Output channels for subsequent layers. Default: ``64``.
		depth: The depth of the network. Default: ``5``.
		relu_slope: The slope of the ReLU activation. Default: ``0.2``,
		in_pos_left: The layer index to begin applying the Instance
			Normalization. Default: ``0``.
		in_pos_right: The layer index to end applying the Instance
			Normalization. Default: ``4``.
		
	References:
		`<https://github.com/swz30/MPRNet/tree/main>`__

	See Also: :class:`base.MultiTaskImageEnhancementModel`
	"""
	
	arch   : str  = "mprnet"
	tasks  : list[Task]   = [Task.DEBLUR, Task.DENOISE, Task.DERAIN, Task.DESNOW]
	schemes: list[Scheme] = [Scheme.SUPERVISED]
	zoo    : dict = {}
	
	def __init__(
		self,
		in_channels      : int       = 3,
		num_channels     : int       = 96,
		scale_unetfeats  : int       = 48,
		scale_orsnetfeats: int       = 32,
		num_cab          : int       = 8,
		kernel_size      : _size_2_t = 3,
		reduction        : int       = 4,
		bias             : bool      = False,
		weights          : Any       = None,
		loss 		     : Any       = nn.EdgeCharbonnierLoss(edge_loss_weight=0.05),
		*args, **kwargs
	):
		super().__init__(
			name        = "mprnet",
			in_channels = in_channels,
			weights     = weights,
			loss        = loss,
			*args, **kwargs
		)
		
		# Populate hyperparameter values from pretrained weights
		if isinstance(self.weights, dict):
			in_channels       = self.weights.get("in_channels"      , in_channels)
			num_channels      = self.weights.get("num_channels"     , num_channels)
			scale_unetfeats   = self.weights.get("scale_unetfeats"  , scale_unetfeats)
			scale_orsnetfeats = self.weights.get("scale_orsnetfeats", scale_orsnetfeats)
			num_cab           = self.weights.get("num_cab"          , num_cab)
			kernel_size       = self.weights.get("kernel_size"      , kernel_size)
			reduction         = self.weights.get("reduction"        , reduction)
			bias              = self.weights.get("bias"             , bias)
			
		self.in_channels       = in_channels
		self.num_channels      = num_channels
		self.scale_unetfeats   = scale_unetfeats
		self.scale_orsnetfeats = scale_orsnetfeats
		self.num_cab           = num_cab
		self.kernel_size       = kernel_size
		self.reduction         = reduction
		self.bias              = bias
	
		# Construct model
		act_layer = nn.PReLU()
		self.shallow_feat1 = nn.Sequential(
			conv(self.in_channels, self.num_channels, kernel_size, bias=bias),
			CAB(self.num_channels, kernel_size, reduction, bias=bias, act_layer=act_layer)
		)
		self.shallow_feat2 = nn.Sequential(
			conv(self.in_channels, self.num_channels, kernel_size, bias=bias),
			CAB(self.num_channels, kernel_size, reduction, bias=bias, act_layer=act_layer)
		)
		self.shallow_feat3 = nn.Sequential(
			conv(self.in_channels, self.num_channels, kernel_size, bias=bias),
			CAB(self.num_channels, kernel_size, reduction, bias=bias, act_layer=act_layer)
		)
		# Cross Stage Feature Fusion (CSFF)
		self.stage1_encoder = Encoder(self.num_channels, kernel_size, reduction, act_layer, bias, scale_unetfeats, csff=False)
		self.stage1_decoder = Decoder(self.num_channels, kernel_size, reduction, act_layer, bias, scale_unetfeats)
		self.stage2_encoder = Encoder(self.num_channels, kernel_size, reduction, act_layer, bias, scale_unetfeats, csff=True)
		self.stage2_decoder = Decoder(self.num_channels, kernel_size, reduction, act_layer, bias, scale_unetfeats)
		self.stage3_orsnet  = ORSNet(self.num_channels, scale_orsnetfeats, kernel_size, reduction, act_layer, bias, scale_unetfeats, num_cab)
		self.sam12		    = SAM(self.num_channels, kernel_size=1, bias=bias)
		self.sam23 		    = SAM(self.num_channels, kernel_size=1, bias=bias)
		self.concat12 	    = conv(self.num_channels * 2, self.num_channels, kernel_size, bias=bias)
		self.concat23  		= conv(self.num_channels * 2, self.num_channels + scale_orsnetfeats, kernel_size, bias=bias)
		self.tail      		= conv(self.num_channels+scale_orsnetfeats, self.in_channels, kernel_size, bias=bias)
		
		# Load weights
		if self.weights:
			self.load_weights()
		else:
			self.apply(self.init_weights)
	
	def init_weights(self, m: nn.Module):
		pass
	
	def forward_loss(
		self,
		input : torch.Tensor,
		target: torch.Tensor | None,
		*args, **kwargs
	) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
		pred = self.forward(input=input, *args, **kwargs)
		if self.loss:
			loss = 0
			for p in pred:
				loss += self.loss(p, target)
		else:
			loss = None
		return pred[-1], loss
	
	def forward(
		self,
		input    : torch.Tensor,
		augment  : bool = False,
		profile  : bool = False,
		out_index: int  = -1,
		*args, **kwargs
	) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
		x = input
	
		# Original-resolution Image for Stage 3
		h = x.size(2)
		w = x.size(3)
		
		# Multi-Patch Hierarchy: Split Image into four non-overlapping patches
		# Two Patches for Stage 2
		x2top_img  = x[:, :, 0:int(h / 2), :]
		x2bot_img  = x[:, :, int(h / 2):h, :]
		# Four Patches for Stage 1
		x1ltop_img = x2top_img[:, :, :, 0: int(w / 2)]
		x1rtop_img = x2top_img[:, :, :, int(w / 2): w]
		x1lbot_img = x2bot_img[:, :, :, 0: int(w / 2)]
		x1rbot_img = x2bot_img[:, :, :, int(w / 2): w]
		
		# Stage 1
		# Compute Shallow Features
		x1ltop     = self.shallow_feat1(x1ltop_img)
		x1rtop     = self.shallow_feat1(x1rtop_img)
		x1lbot     = self.shallow_feat1(x1lbot_img)
		x1rbot 	   = self.shallow_feat1(x1rbot_img)
		# Process features of all 4 patches with Encoder of Stage 1
		feat1_ltop = self.stage1_encoder(x1ltop)
		feat1_rtop = self.stage1_encoder(x1rtop)
		feat1_lbot = self.stage1_encoder(x1lbot)
		feat1_rbot = self.stage1_encoder(x1rbot)
		# Concat deep features
		feat1_top  = [torch.cat((k, v), 3) for k, v in zip(feat1_ltop, feat1_rtop)]
		feat1_bot  = [torch.cat((k, v), 3) for k, v in zip(feat1_lbot, feat1_rbot)]
		# Pass features through Decoder of Stage 1
		res1_top   = self.stage1_decoder(feat1_top)
		res1_bot   = self.stage1_decoder(feat1_bot)
		# Apply Supervised Attention Module (SAM)
		x2top_samfeats, stage1_img_top = self.sam12(res1_top[0], x2top_img)
		x2bot_samfeats, stage1_img_bot = self.sam12(res1_bot[0], x2bot_img)
		# Output image at Stage 1
		stage1_img = torch.cat([stage1_img_top, stage1_img_bot], 2)
		
		# Stage 2
		# Compute Shallow Features
		x2top     = self.shallow_feat2(x2top_img)
		x2bot     = self.shallow_feat2(x2bot_img)
		# Concatenate SAM features of Stage 1 with shallow features of Stage 2
		x2top_cat = self.concat12(torch.cat([x2top, x2top_samfeats], 1))
		x2bot_cat = self.concat12(torch.cat([x2bot, x2bot_samfeats], 1))
		# Process features of both patches with Encoder of Stage 2
		feat2_top = self.stage2_encoder(x2top_cat, feat1_top, res1_top)
		feat2_bot = self.stage2_encoder(x2bot_cat, feat1_bot, res1_bot)
		# Concat deep features
		feat2	  = [torch.cat((k, v), 2) for k, v in zip(feat2_top, feat2_bot)]
		# Pass features through Decoder of Stage 2
		res2 	  = self.stage2_decoder(feat2)
		# Apply SAM
		x3_samfeats, stage2_img = self.sam23(res2[0], x)
		
		# Stage 3
		# Compute Shallow Features
		x3     	   = self.shallow_feat3(x)
		# Concatenate SAM features of Stage 2 with shallow features of Stage 3
		x3_cat     = self.concat23(torch.cat([x3, x3_samfeats], 1))
		x3_cat 	   = self.stage3_orsnet(x3_cat, feat2, res2)
		stage3_img = self.tail(x3_cat)
		
		return [stage1_img, stage2_img, stage3_img + x]

# endregion
