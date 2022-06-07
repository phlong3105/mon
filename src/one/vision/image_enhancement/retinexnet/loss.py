#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Decomposition loss and enhancement loss according to RetinexNet
paper (https://arxiv.org/abs/1808.04560).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from one.core import LOSSES
from one.imgproc import avg_gradient
from one.imgproc import gradient

__all__ = [
	"decom_loss",
	"DecomLoss",
	"enhance_loss",
	"EnhanceLoss",
	"retinex_loss",
	"RetinexLoss"
]


def decom_loss(
	input : Tensor,
	target: Tensor,
	r_low : Tensor,
	r_high: Tensor,
	i_low : Tensor,
	i_high: Tensor,
	**_
) -> Tensor:
	"""Calculate the decomposition loss according to RetinexNet paper
	(https://arxiv.org/abs/1808.04560).

	Args:
		input (Tensor):
			Low-light images.
		target (Tensor):
			Normal-light images.
		r_low (Tensor):
			Reflectance map extracted from low-light images.
		r_high (Tensor):
			Reflectance map extracted from normal-light images.
		i_low (Tensor):
			Illumination map extracted from low-light images as a single
			channel.
		i_high (Tensor):
			Illumination map extracted from normal-light images as a single
			channel.
			
	Returns:
		loss (Tensor):
			Loss.
	"""
	i_low_3                = torch.cat(tensors=(i_low,  i_low,  i_low),  dim=1)
	i_high_3               = torch.cat(tensors=(i_high, i_high, i_high), dim=1)
	recon_loss_low         = F.l1_loss(r_low * i_low_3, input)
	recon_loss_high        = F.l1_loss(r_high * i_high_3, target)
	recon_loss_mutual_low  = F.l1_loss(r_high * i_low_3, input)
	recon_loss_mutual_high = F.l1_loss(r_low * i_high_3, target)
	equal_r_loss           = F.l1_loss(r_low, r_high)
	i_smooth_loss_low      = _smooth(i_low, r_low)
	i_smooth_loss_high     = _smooth(i_high, r_high)
	
	loss = (recon_loss_low +
		   recon_loss_high +
		   0.001 * recon_loss_mutual_low +
		   0.001 * recon_loss_mutual_high +
		   0.1   * i_smooth_loss_low +
		   0.1   * i_smooth_loss_high +
		   0.01  * equal_r_loss)
	return loss


def enhance_loss(
	target   : Tensor,
	r_low    : Tensor,
	i_delta  : Tensor,
	i_delta_3: Optional[Tensor] = None,
	pred     : Optional[Tensor] = None,
	**_
) -> Tensor:
	"""Calculate the enhancement loss according to RetinexNet paper
	(https://arxiv.org/abs/1808.04560).

	Args:
		target (Tensor):
			Normal-light images.
		r_low (Tensor):
			Reflectance map extracted from low-light images.
		i_delta (Tensor):
			Enhanced illumination map produced from low-light images as a
			single-channel.
		i_delta_3 (Tensor, optional):
			Enhanced illumination map produced from low-light images as a
			3-channels. Default: `None`.
		pred (Tensor, optional):
			Enhanced low-light images. Default: `None`.
			
	Returns:
		loss (Tensor):
			Loss.
	"""
	i_delta_3    		= (torch.cat(tensors=(i_delta, i_delta, i_delta), dim=1)
						   if i_delta_3 is None else i_delta_3)
	pred  		 		= (r_low * i_delta_3) if pred is None else pred
	relight_loss 		= F.l1_loss(pred, target)
	i_smooth_loss_delta = _smooth(i_delta, r_low)
	loss 				= relight_loss + 3 * i_smooth_loss_delta
	return loss


def retinex_loss(
	input  : Tensor,
	target : Tensor,
	r_low  : Tensor,
	r_high : Tensor,
	i_low  : Tensor,
	i_high : Tensor,
	i_delta: Tensor,
	**_
) -> Tensor:
	"""Calculate the combined decom loss and enhance loss.

	Args:
		input (Tensor):
			Low-light images.
		target (Tensor):
			Normal-light images.
		r_low (Tensor):
			Reflectance map extracted from low-light images.
		r_high (Tensor):
			Reflectance map extracted from normal-light images.
		i_low (Tensor):
			Illumination map extracted from low-light images as a single
			channel.
		i_high (Tensor):
			Illumination map extracted from normal-light images as a single
			channel.
		i_delta (Tensor):
			Enhanced illumination map produced from low-light images as a
			single-channel.

	Returns:
		loss (Tensor):
			Loss.
	"""
	loss1 = decom_loss(
		input  = input,
		target = target,
		r_low  = r_low,
		r_high = r_high,
		i_low  = i_low,
		i_high = i_high,
	)
	loss2 = enhance_loss(
		target  = target,
		r_low   = r_low,
		i_delta = i_delta,
	)
	loss = loss1 + loss2
	return loss


def _smooth(i: Tensor, r: Tensor) -> Tensor:
	"""Get the smooth reconstructed image from the given illumination map and reflectance map.
	
	Args:
		i (Tensor):
			Illumination map.
		r (Tensor):
			Reflectance map.

	Returns:
		grad (Tensor):
			Smoothed reconstructed image.
	"""
	r    = ((0.299 * r[:, 0, :, :])
	        + (0.587 * r[:, 1, :, :])
	        + (0.114 * r[:, 2, :, :]))
	r    = torch.unsqueeze(input=r, dim=1)
	grad = gradient(image=i, direction="x") * torch.exp(-10 * avg_gradient(
		image=r, direction="x")) + \
		   gradient(image=i, direction="y") * torch.exp(-10 * avg_gradient(
		image=r, direction="y"))
	return torch.mean(input=grad)


# noinspection PyMethodMayBeStatic
@LOSSES.register(name="decom_loss")
class DecomLoss(nn.Module):
	"""Calculate the decomposition loss according to RetinexNet paper
	(https://arxiv.org/abs/1808.04560).
	"""
	
	# MARK: Magic Functions
	
	def __init__(self):
		super().__init__()
		self.name = "decom_loss"
	
	# MARK: Forward Pass
	
	def forward(
		self,
		input : Tensor,
		target: Tensor,
		r_low : Tensor,
		r_high: Tensor,
		i_low : Tensor,
		i_high: Tensor,
		**_
	) -> Tensor:
		"""Forward pass.
		
		Args:
			input (Tensor):
				Low-light images.
			target (Tensor):
				Normal-light images.
			r_low (Tensor):
				Reflectance map extracted from low-light images.
			r_high (Tensor):
				Reflectance map extracted from normal-light images.
			i_low (Tensor):
				Illumination map extracted from low-light images as a
				single channel.
			i_high (Tensor):
				Illumination map extracted from normal-light images as a
				single channel.
				
		Returns:
			loss (Tensor):
				Loss.
		"""
		return decom_loss(
			input  = input,
			target = target,
			r_low  = r_low,
			r_high = r_high,
			i_low  = i_low,
			i_high = i_high,
		)
	
	
# noinspection PyMethodMayBeStatic
@LOSSES.register(name="enhance_loss")
class EnhanceLoss(nn.Module):
	"""Calculate the enhancement loss according to RetinexNet paper
	(https://arxiv.org/abs/1808.04560).
	"""
	
	# MARK: Magic Functions
	
	def __init__(self):
		super().__init__()
		self.name = "enhance_loss"
	
	# MARK: Forward Pass
	
	def forward(
		self,
		target   : Tensor,
		r_low    : Tensor,
		i_delta  : Tensor,
		i_delta_3: Optional[Tensor] = None,
		pred     : Optional[Tensor] = None,
		**_
	) -> Tensor:
		"""Run forward pass.

		Args:
			target (Tensor):
				Normal-light images.
			r_low (Tensor):
				Reflectance map extracted from low-light images.
			i_delta (Tensor):
				Enhanced illumination map produced from low-light images
				as a single-channel.
			i_delta_3 (Tensor, optional):
				Enhanced illumination map produced from low-light images
				as a 3-channels. Default: `None`.
			pred (Tensor, optional):
				Enhanced low-light images. Default: `None`.
				
		Returns:
			loss (Tensor):
				Loss.
		"""
		return enhance_loss(
			target    = target,
			r_low     = r_low,
			i_delta   = i_delta,
			i_delta_3 = i_delta_3,
			pred      = pred,
		)


# noinspection PyMethodMayBeStatic
@LOSSES.register(name="retinex_loss")
class RetinexLoss(nn.Module):
	"""Calculate the combined decomposition loss and enhancement loss."""
	
	# MARK: Magic Functions
	
	def __init__(self):
		super().__init__()
		self.name = "retinex_loss"
		
	# MARK: Forward Pass
	
	def forward(
		self,
		input  : Tensor,
		target : Tensor,
		r_low  : Tensor,
		r_high : Tensor,
		i_low  : Tensor,
		i_high : Tensor,
		i_delta: Tensor,
		**_
	) -> Tensor:
		"""Run forward pass.

		Args:
			input (Tensor):
				Low-light images.
			target (Tensor):
				Normal-light images.
			r_low (Tensor):
				Reflectance map extracted from low-light images.
			r_high (Tensor):
				Reflectance map extracted from normal-light images.
			i_low (Tensor):
				Illumination map extracted from low-light images as a
				single channel.
			i_high (Tensor):
				Illumination map extracted from normal-light images as a
				single channel.
			i_delta (Tensor):
				Enhanced illumination map produced from low-light images
				as a single-channel.
	
		Returns:
			loss (Tensor):
				Loss.
		"""
		return retinex_loss(
			input   = input,
			target  = target,
			r_low   = r_low,
			r_high  = r_high,
			i_low   = i_low,
			i_high  = i_high,
			i_delta = i_delta,
		)
