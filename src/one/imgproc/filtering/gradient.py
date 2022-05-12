#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Gradient Operations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
	"avg_gradient",
	"gradient",
]


# MARK: - Functional

def gradient(image: Tensor, direction: str) -> Tensor:
	"""Calculate the gradient in the image with the desired direction.

	Args:
		image (Tensor[B, C, H, W]):
			Input image.
		direction (str):
			Direction to calculate the gradient. Can be ["x", "y"].

	Returns:
		grad (Tensor[B, C, H, W]):
			Gradient.
	"""
	if direction not in ["x", "y"]:
		raise ValueError
	
	smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2))
	smooth_kernel_y = torch.transpose(smooth_kernel_x, 2, 3)
	
	if direction == "x":
		kernel = smooth_kernel_x
	else:
		kernel = smooth_kernel_y
	
	kernel = kernel.cuda()
	grad   = torch.abs(F.conv2d(input=image, weight=kernel, stride=1, padding=1))
	return grad


def avg_gradient(image: Tensor, direction: str) -> Tensor:
	"""Calculate the average gradient in the image with the desired direction.

	Args:
		image (Tensor[B, C, H, W]):
			Input image.
		direction (str):
			Direction to calculate the gradient. Can be ["x", "y"].

	Returns:
		avg_gradient (Tensor[B, C, H, W]):
			Average gradient.
	"""
	return F.avg_pool2d(
		gradient(image=image, direction=direction),
		kernel_size = 3,
		stride      = 1,
		padding     = 1
	)
