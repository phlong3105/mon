#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import numpy as np
import torch
import torchvision
from multipledispatch import dispatch
from torch import Tensor

from one.core.globals import TensorOrArray
from one.vision.transformation import is_channel_first
from one.vision.transformation import to_channel_last


# MARK: - Functional

@dispatch(Tensor, int)
def make_image_grid(images: Tensor, nrow: int = 1) -> Tensor:
	"""Concatenate multiple images into a single image.

	Args:
		images (Tensor):
			Images can be:
				- A 4D mini-batch image of shape [B, C, H, W].
				- A 3D RGB image of shape [C, H, W].
				- A 2D grayscale image of shape [H, W].
		nrow (int):
			Number of images in each row of the grid. Final grid size is
			`[B / nrow, nrow]`. Default: `1`.

	Returns:
		cat_image (Tensor):
			Concatenated image.
	"""
	return torchvision.utils.make_grid(tensor=images, nrow=nrow)


@dispatch(np.ndarray, int)
def make_image_grid(images: np.ndarray, nrow: int = 1) -> np.ndarray:
	"""Concatenate multiple images into a single image.

	Args:
		images (np.array):
			Images can be:
				- A 4D mini-batch image of shape [B, C, H, W].
				- A 3D RGB image of shape [C, H, W].
				- A 2D grayscale image of shape [H, W].
		nrow (int):
			Number of images in each row of the grid. Final grid size is
			`[B / nrow, nrow]`. Default: `1`.

	Returns:
		cat_image (np.ndarray):
			Concatenated image.
	"""
	# NOTE: Type checking
	if images.ndim == 3:
		return images
	
	# NOTE: Conversion (just for sure)
	if is_channel_first(images):
		images = to_channel_last(images)
	
	b, c, h, w = images.shape
	ncols = nrow
	nrows = (b // nrow) if (b // nrow) > 0 else 1
	cat_image = np.zeros((c, int(h * nrows), w * ncols))
	for idx, im in enumerate(images):
		j = idx // ncols
		i = idx % ncols
		cat_image[:, j * h: j * h + h, i * w: i * w + w] = im
	return cat_image


@dispatch(list, int)
def make_image_grid(images: list, nrow: int = 1) -> TensorOrArray:
	"""Concatenate multiple images into a single image.

	Args:
		images (list):
			A list of images of shape [C, H, W].
		nrow (int):
			Number of images in each row of the grid. Final grid size is
			`[B / nrow, nrow]`. Default: `1`.

	Returns:
		cat_image (Image):
			Concatenated image.
	"""
	if isinstance(images, list) and all(torch.is_tensor(t) for t in images):
		return torchvision.utils.make_grid(tensor=images, nrow=nrow)
	else:
		raise TypeError(
			f"`image` must be a `list` of `Tensor`. But got: {type(images)}."
		)


@dispatch(dict, int)
def make_image_grid(images: dict, nrow: int = 1) -> TensorOrArray:
	"""Concatenate multiple images into a single image.

	Args:
		images (dict):
			A dict of images of the same shape [C, H, W].
		nrow (int, None):
			Number of images in each row of the grid. Final grid size is
			`[B / nrow, nrow]`. Default: `1`.

	Returns:
		cat_image (Image):
			Concatenated image.
	"""
	if (isinstance(images, dict)
		and all(torch.is_tensor(t) for k, t in images.items())):
		values = list(tuple(images.values()))
		return torchvision.utils.make_grid(values, nrow)
	else:
		raise TypeError(
			f"`image` must be a `dict` of `Tensor`. But got: {type(images)}."
		)
