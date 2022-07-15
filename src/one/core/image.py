# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import inspect
import multiprocessing
import os
import sys
from copy import copy
from copy import deepcopy
from typing import Union

import cv2
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from joblib import delayed
from joblib import Parallel
from multipledispatch import dispatch
from PIL import ExifTags
from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import _is_numpy
from torchvision.transforms.functional_pil import _is_pil_image

from one.core.collection import to_size
from one.core.file import create_dirs
from one.core.globals import FloatAnyT
from one.core.globals import Int2Or3T
from one.core.globals import Int2T
from one.core.globals import Int3T
from one.core.globals import PaddingMode
from one.core.globals import TensorOrArray
from one.core.globals import TRANSFORMS
from one.core.rich import error_console
from one.math import make_divisible

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
	if ExifTags.TAGS[orientation] == "Orientation":
		break


# MARK: - Functional

def add_weighted(
	src1 : TensorOrArray,
	alpha: float,
	src2 : TensorOrArray,
	beta : float,
	gamma: float = 0.0,
) -> Tensor:
	"""Calculate the weighted sum of two Tensors.
	
	Function calculates the weighted sum of two Tensors as follows:
		out = src1 * alpha + src2 * beta + gamma

	Args:
		src1 (TensorOrArray[B, C, H, W]):
			First image.
		alpha (float):
			Weight of the src1 elements.
		src2 (TensorOrArray[B, C, H, W]):
			Tensor of same size and channel number as src1 [*, H, W].
		beta (float):
			Weight of the src2 elements.
		gamma (float):
			Scalar added to each sum. Default: `0.0`.

	Returns:
		add (Tensor[B, C, H, W]):
			Weighted tensor.

	Example:
		>>> input1 = torch.rand(1, 1, 5, 5)
		>>> input2 = torch.rand(1, 1, 5, 5)
		>>> output = add_weighted(input1, 0.5, input2, 0.5, 1.0)
		>>> output.shape
		torch.Size([1, 1, 5, 5])
	"""
	if not isinstance(src1, Tensor):
		raise TypeError(f"`src1` must be a `Tensor`. But got: {type(src1)}.")
	if not isinstance(src2, Tensor):
		raise TypeError(f"`src2` must be a `Tensor`. But got: {type(src2)}.")
	if src1.shape != src2.shape:
		raise ValueError(f"`src1` and `src2` must have the same shape. "
						 f"But got: {src1.shape} != {src2.shape}.")
	if not isinstance(alpha, float):
		raise TypeError(f"`alpha` must be a `float`. But got: {type(alpha)}.")
	if not isinstance(beta, float):
		raise TypeError(f"`beta` must be a `float`. But got: {type(beta)}.")
	if not isinstance(gamma, float):
		raise TypeError(f"`gamma` must be a `float`. But got: {type(gamma)}.")

	return src1 * alpha + src2 * beta + gamma


@dispatch(Tensor, Tensor, float, float)
def blend_images(
	overlays: Tensor,
	images  : Tensor,
	alpha   : float,
	gamma   : float = 0.0
) -> Tensor:
	"""Blends 2 images together. dst = image1 * alpha + image2 * beta + gamma

	Args:
		overlays (Tensor[B, C, H, W]):
			Images we want to overlay on top of the original image.
		images (Tensor[B, C, H, W]):
			Source images.
		alpha (float):
			Alpha transparency of the overlay.
		gamma (float):
			Default: `0.0`.

	Returns:
		blend (Tensor[B, C, H, W]):
			Blended image.
	"""
	overlays_np = overlays.numpy()
	images_np   = images.numpy()
	blends      = blend_images(overlays_np, images_np, alpha, gamma)
	blends      = torch.from_numpy(blends)
	return blends


@dispatch(np.ndarray, np.ndarray, float, float)
def blend_images(
	overlays: np.ndarray,
	images  : np.ndarray,
	alpha   : float,
	gamma   : float = 0.0
) -> np.ndarray:
	"""Blends 2 images together. dst = image1 * alpha + image2 * beta + gamma

	Args:
		overlays (np.ndarray[B, C, H, W]):
			Images we want to overlay on top of the original image.
		images (np.ndarray[B, C, H, W]):
			Source images.
		alpha (float):
			Alpha transparency of the overlay.
		gamma (float):
			Default: `0.0`.

	Returns:
		blend (np.ndarray[B, C, H, W]):
			Blended image.
	"""
	# NOTE: Type checking
	if overlays.ndim != images.ndim:
		raise ValueError(f"`overlays` and `images` must have the same ndim. "
						 f"But got: {overlays.ndim} != {images.ndim}")
	
	# NOTE: Convert to channel-first
	overlays = to_channel_first(overlays)
	images   = to_channel_first(images)
	
	# NOTE: Unnormalize images
	images = denormalize_naive(images)
	
	# NOTE: Convert overlays to same data type as images
	images   = images.astype(np.uint8)
	overlays = overlays.astype(np.uint8)
	
	# NOTE: If the images are of shape [CHW]
	if overlays.ndim == 3 and images.ndim == 3:
		return cv2.addWeighted(overlays, alpha, images, 1.0 - alpha, gamma)
	
	# NOTE: If the images are of shape [BCHW]
	if overlays.ndim == 4 and images.ndim == 4:
		if overlays.shape[0] != images.shape[0]:
			raise ValueError(
				f"`overlays` and `images` must have the same batch sizes. "
				f"But got: {overlays.shape[0]} != {images.shape[0]}"
			)
		blends = []
		for overlay, image in zip(overlays, images):
			blends.append(cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, gamma))
		blends = np.stack(blends, axis=0).astype(np.uint8)
		return blends


def check_image_size(image_size: Int2Or3T, stride: int = 32) -> int:
	"""Verify image size is a multiple of stride and return the new size.
	
	Args:
		image_size (Int2Or3T):
			Image size.
		stride (int):
			Stride. Default: `32`.
	
	Returns:
		new_size (int):
			Appropriate size.
	"""
	if isinstance(image_size, (list, tuple)):
		if len(image_size) == 3:  # [H, W, C]
			image_size = image_size[1]
		elif len(image_size) == 2:  # [H, W]
			image_size = image_size[0]
		
	new_size = make_divisible(image_size, int(stride))  # ceil gs-multiple
	if new_size != image_size:
		error_console.log(
			"WARNING: image_size %g must be multiple of max stride %g, "
			"updating to %g" % (image_size, stride, new_size)
		)
	return new_size


def denormalize(
	data: Tensor,
	mean: Union[Tensor, float],
	std : Union[Tensor, float]
) -> Tensor:
	"""Denormalize an image/video image with mean and standard deviation.
	
	input[channel] = (input[channel] * std[channel]) + mean[channel]
		
		where `mean` is [M_1, ..., M_n] and `std` [S_1, ..., S_n] for `n`
		channels,

	Args:
		data (Tensor[B, C, *, *]):
			Image.
		mean (Tensor[B, C, *, *], float):
			Mean for each channel.
		std (Tensor[B, C, *, *], float):
			Standard deviations for each channel.

	Return:
		out (Tensor[B, N, *, *]):
			Denormalized image with same size as input.

	Examples:
		>>> x   = torch.rand(1, 4, 3, 3)
		>>> out = denormalize(x, 0.0, 255.)
		>>> out.shape
		torch.Size([1, 4, 3, 3])

		>>> x    = torch.rand(1, 4, 3, 3, 3)
		>>> mean = torch.zeros(1, 4)
		>>> std  = 255. * torch.ones(1, 4)
		>>> out  = denormalize(x, mean, std)
		>>> out.shape
		torch.Size([1, 4, 3, 3, 3])
	"""
	shape = data.shape

	if isinstance(mean, float):
		mean = torch.tensor([mean] * shape[1], device=data.device,
							dtype=data.dtype)
	if isinstance(std, float):
		std  = torch.tensor([std] * shape[1], device=data.device,
							dtype=data.dtype)
	if not isinstance(data, Tensor):
		raise TypeError(f"`data` should be a `Tensor`. But got: {type(data)}")
	if not isinstance(mean, Tensor):
		raise TypeError(f"`mean` should be a `Tensor`. But got: {type(mean)}")
	if not isinstance(std, Tensor):
		raise TypeError(f"`std` should be a `Tensor`. But got: {type(std)}")

	# Allow broadcast on channel dimension
	if mean.shape and mean.shape[0] != 1:
		if mean.shape[0] != data.shape[-3] and mean.shape[:2] != data.shape[:2]:
			raise ValueError(f"`mean` and `data` must have the same shape. "
							 f"But got: {mean.shape} and {data.shape}.")

	# Allow broadcast on channel dimension
	if std.shape and std.shape[0] != 1:
		if std.shape[0] != data.shape[-3] and std.shape[:2] != data.shape[:2]:
			raise ValueError(f"`std` and `data` must have the same shape. "
							 f"But got: {std.shape} and {data.shape}.")

	mean = torch.as_tensor(mean, device=data.device, dtype=data.dtype)
	std  = torch.as_tensor(std,  device=data.device, dtype=data.dtype)

	if mean.shape:
		mean = mean[..., :, None]
	if std.shape:
		std  = std[..., :, None]

	out = (data.view(shape[0], shape[1], -1) * std) + mean
	return out.view(shape)


@dispatch((Tensor, np.ndarray))
def denormalize_naive(image: TensorOrArray) -> TensorOrArray:
	if isinstance(image, Tensor):
		return torch.clamp(image * 255, 0, 255).to(torch.uint8)
	elif isinstance(image, np.ndarray):
		return np.clip(image * 255, 0, 255).astype(np.uint8)
	else:
		raise TypeError(f"Do not support: {type(image)}.")
	

@dispatch(list)
def denormalize_naive(image: list) -> list:
	# NOTE: List of np.ndarray
	if all(isinstance(i, np.ndarray) and i.ndim == 3 for i in image):
		return list(denormalize_naive(np.array(image)))
	if all(isinstance(i, np.ndarray) and i.ndim == 4 for i in image):
		return [denormalize_naive(i) for i in image]
	
	# NOTE: List of Tensor
	if all(isinstance(i, Tensor) and i.ndim == 3 for i in image):
		return list(denormalize_naive(torch.stack(image)))
	if all(isinstance(i, Tensor) and i.ndim == 4 for i in image):
		return [denormalize_naive(i) for i in image]
	
	raise TypeError(f"Do not support {type(image)}.")


@dispatch(tuple)
def denormalize_naive(image: tuple) -> tuple:
	image = list(image)
	image = denormalize_naive(image)
	return tuple(image)


@dispatch(dict)
def denormalize_naive(image: dict) -> dict:
	if not all(isinstance(v, (tuple, list, Tensor, np.ndarray))
			   for k, v in image.items()):
		raise ValueError()
	
	for k, v in image.items():
		image[k] = denormalize_naive(v)
	
	return image


def get_exif_size(image: Image) -> Int2T:
	"""Return the exif-corrected PIL size."""
	size = image.size  # (width, height)
	try:
		rotation = dict(image._getexif().items())[orientation]
		if rotation == 6:  # rotation 270
			size = (size[1], size[0])
		elif rotation == 8:  # rotation 90
			size = (size[1], size[0])
	except:
		pass
	return size[1], size[0]


def get_image_center(image: TensorOrArray) -> TensorOrArray:
	"""Get image center as (x=h/2, y=w/2).
	
	Args:
		image (TensorOrArray[B, C, H, W]):
			Image.
   
	Returns:
		center (TensorOrArray):
			Image center as (x=h/2, y=w/2).
	"""
	h, w   = get_image_hw(image)
	center = np.array((h / 2, w / 2))
	
	if isinstance(image, Tensor):
		return torch.from_numpy(center)
	elif isinstance(image, np.ndarray):
		return center
	else:
		TypeError(f"Unexpected type {type(image)}")


def get_image_center4(image: TensorOrArray) -> TensorOrArray:
	"""Get image center as (x=h/2, y=w/2, x=h/2, y=w/2).
	
	Args:
		image (TensorOrArray[B, C, H, W]):
			Image.
   
	Returns:
		center (TensorOrArray):
			Image center as (x=h/2, y=w/2, x=h/2, y=w/2).
	"""
	h, w   = get_image_hw(image)
	center = np.array((h / 2, w / 2))
	center = np.hstack((center, center))
	
	if isinstance(image, Tensor):
		return torch.from_numpy(center)
	elif isinstance(image, np.ndarray):
		return center
	else:
		TypeError(f"Unexpected type {type(image)}")


def get_image_hw(image: Union[Tensor, np.ndarray, PIL.Image]) -> Int2T:
	"""Returns the size of an image as [H, W].
	
	Args:
		image (Tensor, np.ndarray, PIL Image):
			The image to be checked.
   
	Returns:
		size (Int2T):
			Image size as [H, W].
	"""
	if isinstance(image, (Tensor, np.ndarray)):
		if is_channel_first(image):  # [.., C, H, W]
			return [image.shape[-2], image.shape[-1]]
		else:  # [.., H, W, C]
			return [image.shape[-3], image.shape[-2]]
	elif _is_pil_image(image):
		return list(image.size)
	else:
		raise TypeError(
			f"`image` must be `Tensor`, `np.ndarray`, or `PIL.Image. "
			f"But got: {type(image)}."
		)
	

def get_image_shape(image: Union[Tensor, np.ndarray, PIL.Image]) -> Int3T:
	"""Returns the shape of an image as [H, W, C].

	Args:
		image (Tensor, np.ndarray, PIL Image):
			Image.

	Returns:
		shape (Int3T):
			Image shape as [H, W, C].
	"""
	if isinstance(image, (Tensor, np.ndarray)):
		if is_channel_first(image):  # [.., C, H, W]
			return [image.shape[-2], image.shape[-1], image.shape[-3]]
		else:  # [.., H, W, C]
			return [image.shape[-3], image.shape[-2], image.shape[-1]]
	elif _is_pil_image(image):
		return list(image.size)
	else:
		raise TypeError(
			f"`image` must be `Tensor`, `np.ndarray`, or `PIL.Image. "
			f"But got: {type(image)}."
		)

def get_num_channels(image: TensorOrArray) -> int:
	"""Get number of channels of the image."""
	if image.ndim == 4:
		if is_channel_first(image):
			_, c, h, w = list(image.shape)
		else:
			_, h, w, c = list(image.shape)
		return c
	elif image.ndim == 3:
		if is_channel_first(image):
			c, h, w = list(image.shape)
		else:
			h, w, c = list(image.shape)
		return c
	else:
		raise ValueError(f"`image.ndim` must be == 3 or 4. But got: {image.ndim}.")


def is_channel_first(image: TensorOrArray) -> bool:
	"""Check if the image is in channel first format."""
	if image.ndim == 5:
		_, _, s2, s3, s4 = list(image.shape)
		if (s2 < s3) and (s2 < s4):
			return True
		elif (s4 < s2) and (s4 < s3):
			return False
	elif image.ndim == 4:
		_, s1, s2, s3 = list(image.shape)
		if (s1 < s2) and (s1 < s3):
			return True
		elif (s3 < s1) and (s3 < s2):
			return False
	elif image.ndim == 3:
		s0, s1, s2 = list(image.shape)
		if (s0 < s1) and (s0 < s2):
			return True
		elif (s2 < s0) and (s2 < s1):
			return False
	
	raise ValueError(f"`image.ndim` must be == 3, 4, or 5. But got: {image.ndim}.")


def is_channel_last(image: TensorOrArray) -> bool:
	"""Check if the image is in channel last format."""
	return not is_channel_first(image)


def is_integer_image(image: TensorOrArray) -> bool:
	"""Check if the given image is integer-encoded."""
	c = get_num_channels(image)
	if c == 1:
		return True
	return False


def is_normalized(image: TensorOrArray) -> TensorOrArray:
	if isinstance(image, Tensor):
		return abs(torch.max(image)) <= 1.0
	elif isinstance(image, np.ndarray):
		return abs(np.amax(image)) <= 1.0
	else:
		raise TypeError(f"Do not support: {type(image)}.")


def is_one_hot_image(image: TensorOrArray) -> bool:
	"""Check if the given image is one-hot encoded."""
	c = get_num_channels(image)
	if c > 1:
		return True
	return False


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
	ncols      = nrow
	nrows      = (b // nrow) if (b // nrow) > 0 else 1
	cat_image  = np.zeros((c, int(h * nrows), w * ncols))
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
			A list of images of the same shape [C, H, W].
		nrow (int):
			Number of images in each row of the grid. Final grid size is
			`[B / nrow, nrow]`. Default: `1`.

	Returns:
		cat_image (Image):
			Concatenated image.
	"""
	if (isinstance(images, list) and
		all(isinstance(t, np.ndarray) for t in images)):
		cat_image = np.concatenate([images], axis=0)
		return make_image_grid(cat_image, nrow)
	elif isinstance(images, list) and all(torch.is_tensor(t) for t in images):
		return torchvision.utils.make_grid(tensor=images, nrow=nrow)
	else:
		raise TypeError(f"Do not support {type(images)}.")


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
	if (isinstance(images, dict) and
		all(isinstance(t, np.ndarray) for k, t in images.items())):
		cat_image = np.concatenate([image for key, image in images.items()], axis=0)
		return make_image_grid(cat_image, nrow)
	elif (isinstance(images, dict) and
		  all(torch.is_tensor(t) for k, t in images.items())):
		values = list(tuple(images.values()))
		return torchvision.utils.make_grid(values, nrow)
	else:
		raise TypeError(f"Do not support {type(images)}.")


def normalize_min_max(
	image  : Tensor,
	min_val: float = 0.0,
	max_val: float = 1.0,
	eps    : float = 1e-6
) -> Tensor:
	"""Normalise an image/video image by MinMax and re-scales the value
	between a range.

	Args:
		image (Tensor[B, C, *, *]):
			Image to be normalized.
		min_val (float):
			Minimum value for the new range.
		max_val (float):
			Maximum value for the new range.
		eps (float):
			Float number to avoid zero division.

	Returns:
		x_out (Tensor[B, C, *, *]):
			Fnormalized tensor image with same shape.

	Example:
		>>> x      = torch.rand(1, 5, 3, 3)
		>>> x_norm = normalize_min_max(image, min_val=-1., max_val=1.)
		>>> x_norm.min()
		image(-1.)
		>>> x_norm.max()
		image(1.0000)
	"""
	if not isinstance(image, Tensor):
		raise TypeError(f"data should be a image. But got: {type(image)}.")
	if not isinstance(min_val, float):
		raise TypeError(f"`min_val` should be a `float`. But got: {type(min_val)}.")
	if not isinstance(max_val, float):
		raise TypeError(f"`max_val` should be a `float`. But got: {type(max_val)}.")
	if image.ndim < 3:
		raise ValueError(f"`image.ndim` must be >= 3. But got: {image.shape}.")

	shape = image.shape
	B, C  = shape[0], shape[1]

	x_min = image.view(B, C, -1).min(-1)[0].view(B, C, 1)
	x_max = image.view(B, C, -1).max(-1)[0].view(B, C, 1)

	x_out = ((max_val - min_val) * (image.view(B, C, -1) - x_min) /
			 (x_max - x_min + eps) + min_val)
	return x_out.view(shape)


@dispatch((Tensor, np.ndarray))
def normalize_naive(image: TensorOrArray) -> TensorOrArray:
	"""Convert image from `torch.uint8` type and range [0, 255] to `torch.float`
	type and range of [0.0, 1.0].
	"""
	if isinstance(image, Tensor):
		if abs(torch.max(image)) > 1.0:
			return image.to(torch.get_default_dtype()).div(255.0)
		else:
			return image.to(torch.get_default_dtype())
	elif isinstance(image, np.ndarray):
		if abs(np.amax(image)) > 1.0:
			return image.astype(np.float32) / 255.0
		else:
			return image.astype(np.float32)
	else:
		raise TypeError(f"Do not support: {type(image)}.")
	

@dispatch(list)
def normalize_naive(image: list) -> list:
	# NOTE: List of np.ndarray
	if all(isinstance(i, np.ndarray) and i.ndim == 3 for i in image):
		image = normalize_naive(np.array(image))
		return list(image)
	if all(isinstance(i, np.ndarray) and i.ndim == 4 for i in image):
		image = [normalize_naive(i) for i in image]
		return image
	
	# NOTE: List of Tensor
	if all(isinstance(i, Tensor) and i.ndim == 3 for i in image):
		image = normalize_naive(torch.stack(image))
		return list(image)
	if all(isinstance(i, Tensor) and i.ndim == 4 for i in image):
		image = [normalize_naive(i) for i in image]
		return image

	raise TypeError(f"Do not support {type(image)}.")


@dispatch(tuple)
def normalize_naive(image: tuple) -> tuple:
	image = list(image)
	image = normalize_naive(image)
	return tuple(image)


@dispatch(dict)
def normalize_naive(image: dict) -> dict:
	if not all(isinstance(v, (tuple, list, Tensor, np.ndarray))
			   for k, v in image.items()):
		raise ValueError()
	
	for k, v in image.items():
		image[k] = normalize_naive(v)
	
	return image


def pad_image(
	image   : TensorOrArray,
	pad_size: Int2Or3T,
	mode    : Union[PaddingMode, str] = "constant",
	value   : Union[FloatAnyT, None]  = 0.0,
) -> TensorOrArray:
	"""Pad image with `value`.
	
	Args:
		image (TensorOrArray[B, C, H, W]/[B, H, W, C]):
			Image to be padded.
		pad_size (Int2Or3T[H, W, *]):
			Padded image size.
		mode (PaddingMode, str):
			One of the padding modes defined in `PaddingMode`.
			Default: `constant`.
		value (FloatAnyT, None):
			Fill value for `constant` padding. Default: `0.0`.
			
	Returns:
		image (TensorOrArray[B, C, H, W]/[B, H, W, C]):
			Padded image.
	"""
	if image.ndim not in (3, 4):
		raise ValueError(f"`image.ndim` must be 3 or 4. "
						 f"But got: {image.ndim}")
	if isinstance(mode, str) and mode not in PaddingMode.values():
		raise ValueError(f"`mode` must be one of: {PaddingMode.values()}. "
						 f"But got {mode}.")
	elif isinstance(mode, PaddingMode):
		if mode not in PaddingMode:
			raise ValueError(f"`mode` must be one of: {PaddingMode}. "
							 f"But got: {mode}.")
		mode = mode.value
	if isinstance(image, Tensor):
		if mode not in ("constant", "circular", "reflect", "replicate"):
			raise ValueError()
	if isinstance(image, np.ndarray):
		if mode not in ("constant", "edge", "empty", "linear_ramp", "maximum",
						"mean", "median", "minimum", "symmetric", "wrap"):
			raise ValueError()
	
	h0, w0 = get_image_size(image)
	h1, w1 = to_size(pad_size)
	# Image size > pad size, do nothing
	if (h0 * w0) >= (h1 * w1):
		return image
	
	if value is None:
		value = 0
	pad_h = int(abs(h0 - h1) / 2)
	pad_w = int(abs(w0 - w1) / 2)

	if isinstance(image, Tensor):
		if is_channel_first(image):
			pad = (pad_w, pad_w, pad_h, pad_h)
		else:
			pad = (0, 0, pad_w, pad_w, pad_h, pad_h)
		return F.pad(input=image, pad=pad, mode=mode, value=value)
	elif isinstance(image, np.ndarray):
		if is_channel_first(image):
			if image.ndim == 3:
				pad_width = ((0, 0), (pad_h, pad_h), (pad_w, pad_w))
			else:
				pad_width = ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w))
		else:
			if image.ndim == 3:
				pad_width = ((pad_h, pad_h), (pad_w, pad_w), (0, 0))
			else:
				pad_width = ((pad_h, pad_h), (pad_w, pad_w), (0, 0), (0, 0))
		return np.pad(array=image, pad_width=pad_width, mode=mode, constant_values=value)
	
	return image


@dispatch(Tensor, keep_dims=bool)
def to_channel_first(image: Tensor, keep_dims: bool = True) -> Tensor:
	"""Convert image to channel first format.
	
	Args:
		image (Tensor):
			Image.
		keep_dims (bool):
			If `False` unsqueeze the image to match the shape [B, H, W, C].
			Default: `True`.
	"""
	image = copy(image)
	if is_channel_first(image):
		pass
	elif image.ndim == 2:
		image     = image.unsqueeze(0)
	elif image.ndim == 3:
		image     = image.permute(2, 0, 1)
	elif image.ndim == 4:
		image     = image.permute(0, 3, 1, 2)
		keep_dims = True
	elif image.ndim == 5:
		image     = image.permute(0, 1, 4, 2, 3)
		keep_dims = True
	else:
		raise ValueError(f"`image.ndim` must be == 2, 3, 4, or 5. But got: {image.ndim}.")

	return image.unsqueeze(0) if not keep_dims else image


@dispatch(np.ndarray, keep_dims=bool)
def to_channel_first(image: np.ndarray, keep_dims: bool = True) -> np.ndarray:
	"""Convert image to channel first format.
	
	Args:
		image (np.ndarray):
			Image.
		keep_dims (bool):
			If `False` unsqueeze the image to match the shape [B, H, W, C].
			Default: `True`.
	"""
	image = copy(image)
	if is_channel_first(image):
		pass
	elif image.ndim == 2:
		image    = np.expand_dims(image, 0)
	elif image.ndim == 3:
		image    = np.transpose(image, (2, 0, 1))
	elif image.ndim == 4:
		image    = np.transpose(image, (0, 3, 1, 2))
		keep_dims = True
	elif image.ndim == 5:
		image    = np.transpose(image, (0, 1, 4, 2, 3))
		keep_dims = True
	else:
		raise ValueError(f"`image.ndim` must be == 2, 3, 4, or 5. But got: {image.ndim}.")

	return np.expand_dims(image, 0) if not keep_dims else image


@dispatch(Tensor, keep_dims=bool)
def to_channel_last(image: Tensor, keep_dims: bool = True) -> Tensor:
	"""Convert image to channel last format.
	
	Args:
		image (Tensor):
			Image.
		keep_dims (bool):
			If `False` squeeze the input image to match the shape [H, W, C] or
			[H, W]. Default: `True`.
	"""
	image       = copy(image)
	input_shape = image.shape
	
	if is_channel_last(image):
		pass
	elif image.ndim == 2:
		pass
	elif image.ndim == 3:
		if input_shape[0] == 1:
			# Grayscale for proper plt.imshow needs to be [H, W]
			image = image.squeeze()
		else:
			image = image.permute(1, 2, 0)
	elif image.ndim == 4:  # [B, C, H, W] -> [B, H, W, C]
		image = image.permute(0, 2, 3, 1)
		if input_shape[0] == 1 and not keep_dims:
			image = image.squeeze(0)
		if input_shape[1] == 1:
			image = image.squeeze(-1)
	elif image.ndim == 5:
		image = image.permute(0, 1, 3, 4, 2)
		if input_shape[0] == 1 and not keep_dims:
			image = image.squeeze(0)
		if input_shape[2] == 1:
			image = image.squeeze(-1)
	else:
		raise ValueError(f"`image.ndim` must be == 2, 3, 4, or 5. But got: {image.ndim}.")
	
	return image
	

@dispatch(np.ndarray, keep_dims=bool)
def to_channel_last(image: np.ndarray, keep_dims: bool = True) -> np.ndarray:
	"""Convert image to channel last format.
	
	Args:
		image (np.ndarray):
			Image.
		keep_dims (bool):
			If `False` squeeze the input image to match the shape [H, W, C] or
			[H, W]. Default: `True`.
	"""
	image       = copy(image)
	input_shape = image.shape
	
	if is_channel_last(image):
		pass
	elif image.ndim == 2:
		pass
	elif image.ndim == 3:
		if input_shape[0] == 1:
			# Grayscale for proper plt.imshow needs to be [H, W]
			image = image.squeeze()
		else:
			image = np.transpose(image, (1, 2, 0))
	elif image.ndim == 4:
		image = np.transpose(image, (0, 2, 3, 1))
		if input_shape[0] == 1 and not keep_dims:
			image = image.squeeze(0)
		if input_shape[1] == 1:
			image = image.squeeze(-1)
	elif image.ndim == 5:
		image = np.transpose(image, (0, 1, 3, 4, 2))
		if input_shape[0] == 1 and not keep_dims:
			image = image.squeeze(0)
		if input_shape[2] == 1:
			image = image.squeeze(-1)
	else:
		raise ValueError(f"`image.ndim` must be == 2, 3, 4, or 5. But got: {image.ndim}.")
   
	return image


def to_image(
	tensor     : Tensor,
	keep_dims  : bool = True,
	denormalize: bool = False
) -> np.ndarray:
	"""Converts a PyTorch tensor to a numpy image. In case the image is in the
	GPU, it will be copied back to CPU.

	Args:
		tensor (Tensor):
			Image of the form [H, W], [C, H, W] or [B, H, W, C].
		keep_dims (bool):
			If `False` squeeze the input image to match the shape [H, W, C] or
			[H, W]. Default: `True`.
		denormalize (bool):
			If `True`, converts the image in the range [0.0, 1.0] to the range
			[0, 255]. Default: `False`.
		
	Returns:
		image (np.ndarray):
			Image of the form [H, W], [H, W, C] or [B, H, W, C].
	"""
	if not torch.is_tensor(tensor):
		error_console.log(f"Input type is not a Tensor. Got: {type(tensor)}.")
		return tensor
	if tensor.ndim > 4 or tensor.ndim < 2:
		raise ValueError(f"`tensor.ndim` must be == 2, 3, 4, or 5. But got: {tensor.ndim}.")

	image = tensor.cpu().detach().numpy()
	
	# NOTE: Channel last format
	image = to_channel_last(image, keep_dims=keep_dims)
	
	# NOTE: Denormalize
	if denormalize:
		image = denormalize_naive(image)
		
	return image.astype(np.uint8)


def to_pil_image(image: TensorOrArray) -> PIL.Image:
	"""Convert image from `np.ndarray` or `Tensor` to PIL image."""
	if torch.is_tensor(image):
		# Equivalent to: `np_image = image.numpy()` but more efficient
		return torchvision.transforms.ToPILImage()(image)
	elif isinstance(image, np.ndarray):
		return PIL.Image.fromarray(image.astype(np.uint8), "RGB")
	raise TypeError(f"Do not support {type(image)}.")


def to_tensor(
	image    : Union[np.ndarray, PIL.Image],
	keep_dims: bool = True,
	normalize: bool = False,
) -> Tensor:
	"""Convert a `PIL Image` or `np.ndarray` image to a 4d tensor.
	
	Args:
		image (np.ndarray, PIL.Image):
			Image in [H, W, C], [H, W] or [B, H, W, C].
		keep_dims (bool):
			If `False` unsqueeze the image to match the shape [B, H, W, C].
			Default: `True`.
		normalize (bool):
			If `True`, converts the tensor in the range [0, 255] to the range
			[0.0, 1.0]. Default: `False`.
	
	Returns:
		img (Tensor):
			Converted image.
	"""
	if not isinstance(image, (Tensor, np.ndarray)) or _is_pil_image(image):
		raise TypeError(
			f"`image` must be `Tensor`, `np.ndarray`, or `PIL.Image. "
			f"But got: {type(image)}."
		)
	
	if ((_is_numpy(image) or torch.is_tensor(image))
		and (image.ndim > 4 or image.ndim < 2)):
		raise ValueError(f"`image.ndim` must be == 2, 3, or 4. But got: {image.ndim}.")

	# img = image
	img = deepcopy(image)
	
	# NOTE: Handle PIL Image
	if _is_pil_image(img):
		mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
		img = np.array(img, mode_to_nptype.get(img.mode, np.uint8), copy=True)
		if image.mode == "1":
			img = 255 * img
	
	# NOTE: Handle numpy array
	if _is_numpy(img):
		img = torch.from_numpy(img).contiguous()
	
	# NOTE: Channel first format
	img = to_channel_first(img, keep_dims=keep_dims)
   
	# NOTE: Normalize
	if normalize:
		img = normalize_naive(img)
	
	if isinstance(img, torch.ByteTensor):
		return img.to(dtype=torch.get_default_dtype())
	return img


@dispatch(np.ndarray, str, str, str, str)
def write_image(
	image    : np.ndarray,
	dir      : str,
	name     : str,
	prefix   : str = "",
	extension: str = ".png"
):
	"""Save the image using `PIL`.

	Args:
		image (np.ndarray):
			A single image.
		dir (str):
			Saving directory.
		name (str):
			Name of the image file.
		prefix (str):
			Filename prefix. Default: ``.
		extension (str):
			Image file extension. One of [`.jpg`, `.jpeg`, `.png`, `.bmp`].
	"""
	if image.ndim not in [2, 3]:
		raise ValueError(f"`image.ndim` must be 2 or 3. But got: {image.ndim}.")
	
	# NOTE: Unnormalize
	image = denormalize_naive(image)
	
	# NOTE: Convert to channel first
	if is_channel_first(image):
		image = to_channel_last(image)
	
	# NOTE: Convert to PIL image
	if not Image.isImageType(t=image):
		image = Image.fromarray(image.astype(np.uint8))
	
	# NOTE: Write image
	if create_dirs(paths=[dir]) == 0:
		base, ext = os.path.splitext(name)
		if ext:
			extension = ext
		if "." not in extension:
			extension = f".{extension}"
		if prefix in ["", None]:
			filepath = os.path.join(dir, f"{base}{extension}")
		else:
			filepath = os.path.join(dir, f"{prefix}_{base}{extension}")
		image.save(filepath)


@dispatch(Tensor, str, str, str, str)
def write_image(
	image    : Tensor,
	dir      : str,
	name     : str,
	prefix   : str = "",
	extension: str = ".png"
):
	"""Save the image using `torchvision`.

	Args:
		image (Tensor):
			A single image.
		dir (str):
			Saving directory.
		name (str):
			Name of the image file.
		prefix (str):
			Filename prefix. Default: ``.
		extension (str):
			Image file extension. One of: [`.jpg`, `.jpeg`, `.png`].
	"""
	if image.dim() not in [2, 3]:
		raise ValueError(f"`image.ndim` must be 2 or 3. But got: {image.ndim}.")
	
	# NOTE: Convert image
	image = denormalize_naive(image)
	image = to_channel_last(image)
	
	# NOTE: Write image
	if create_dirs(paths=[dir]) == 0:
		base, ext = os.path.splitext(name)
		if ext:
			extension = ext
		if "." not in extension:
			extension = f".{extension}"
		if prefix in ["", None]:
			filepath = os.path.join(dir, f"{base}{extension}")
		else:
			filepath = os.path.join(dir, f"{prefix}_{base}{extension}")
		
		if extension in [".jpg", ".jpeg"]:
			torchvision.io.image.write_jpeg(input=image, filename=filepath)
		elif extension in [".png"]:
			torchvision.io.image.write_png(input=image, filename=filepath)


@dispatch(np.ndarray, str, str, str)
def write_images(
	images   : np.ndarray,
	dir      : str,
	name     : str,
	extension: str = ".png"
):
	"""Save multiple images using `PIL`.

	Args:
		images (np.ndarray):
			A batch of images.
		dir (str):
			Saving directory.
		name (str):
			Name of the image file.
		extension (str):
			Image file extension. One of [`.jpg`, `.jpeg`, `.png`, `.bmp`].
	"""
	if images.ndim != 4:
		raise ValueError(f"`images.ndim` must be 4. But got: {images.ndim}.")
	
	num_jobs = multiprocessing.cpu_count()
	Parallel(n_jobs=num_jobs)(
		delayed(write_image)(image, dir, name, f"{index}", extension)
		for index, image in enumerate(images)
	)


@dispatch(Tensor, str, str, str)
def write_images(
	images   : Tensor,
	dir      : str,
	name     : str,
	extension: str = ".png"
):
	"""Save multiple images using `torchvision`.

	Args:
		images (Tensor):
			A image of image.
		dir (str):
			Saving directory.
		name (str):
			Name of the image file.
		extension (str):
			Image file extension. One of: [`.jpg`, `.jpeg`, `.png`].
	"""
	if images.dim() != 4:
		raise ValueError(f"`images.ndim` must be 4. But got: {images.ndim}.")
	
	num_jobs = multiprocessing.cpu_count()
	Parallel(n_jobs=num_jobs)(
		delayed(write_image)(image, dir, name, f"{index}", extension)
		for index, image in enumerate(images)
	)


@dispatch(list, str, str, str)
def write_images(
	images   : list,
	dir      : str,
	name     : str,
	extension: str = ".png"
):
	"""Save multiple images.

	Args:
		images (list):
			A list of images.
		dir (str):
			Saving directory.
		name (str):
			Name of the image file.
		extension (str):
			Image file extension. One of: [`.jpg`, `.jpeg`, `.png`].
	"""
	if (isinstance(images, list) and
		all(isinstance(image, np.ndarray) for image in images)):
		cat_image = np.concatenate([images], axis=0)
		write_images(cat_image, dir, name, extension)
	elif (isinstance(images, list) and
		  all(torch.is_tensor(image) for image in images)):
		cat_image = torch.stack(images)
		write_images(cat_image, dir, name, extension)
	else:
		raise TypeError(f"Do not support {type(images)}.")


@dispatch(dict, str, str, str)
def write_images(
	images   : dict,
	dir      : str,
	name     : str,
	extension: str = ".png"
):
	"""Save multiple images.

	Args:
		images (dict):
			A list of images.
		dir (str):
			Saving directory.
		name (str):
			Name of the image file.
		extension (str):
			Image file extension. One of: [`.jpg`, `.jpeg`, `.png`].
	"""
	if (isinstance(images, dict) and
		all(isinstance(image, np.ndarray) for _, image in images.items())):
		cat_image = np.concatenate([image for key, image in images.items()],
								   axis=0)
		write_images(cat_image, dir, name, extension)
	elif (isinstance(images, dict) and
		  all(torch.is_tensor(image) for _, image in images)):
		values    = list(tuple(images.values()))
		cat_image = torch.stack(values)
		write_images(cat_image, dir, name, extension)
	else:
		raise TypeError


# MARK: - Modules

@TRANSFORMS.register(name="add_weighted")
class AddWeighted(nn.Module):
	"""Calculate the weighted sum of two Tensors. Function calculates the
	weighted sum of two Tensors as follows:
		out = src1 * alpha + src2 * beta + gamma

	Args:
		alpha (float):
			Weight of the src1 elements.
		beta (float):
			Weight of the src2 elements.
		gamma (float):
			Scalar added to each sum.

	Example:
		>>> input1 = torch.rand(1, 1, 5, 5)
		>>> input2 = torch.rand(1, 1, 5, 5)
		>>> output = AddWeighted(0.5, 0.5, 1.0)(input1, input2)
		>>> output.shape
		torch.Size([1, 1, 5, 5])
	"""
	
	# MARK: Magic Functions
	
	def __init__(self, alpha: float, beta: float, gamma: float):
		super().__init__()
		self.alpha = alpha
		self.beta  = beta
		self.gamma = gamma

	# MARK: Forward Pass
	
	def forward(self, src1: Tensor, src2: Tensor) -> Tensor:
		return add_weighted(src1, self.alpha, src2, self.beta, self.gamma)


@TRANSFORMS.register(name="denormalize")
class Denormalize(nn.Module):
	"""Denormalize a tensor image with mean and standard deviation.
 
	Args:
		mean (Tensor[B, C, *, *], float):
			Mean for each channel.
		std (Tensor[B, C, *, *], float):
			Standard deviations for each channel.

	Examples:
		>>> x   = torch.rand(1, 4, 3, 3)
		>>> out = Denormalize(0.0, 255.)(x)
		>>> out.shape
		torch.Size([1, 4, 3, 3])

		>>> x    = torch.rand(1, 4, 3, 3, 3)
		>>> mean = torch.zeros(1, 4)
		>>> std  = 255. * torch.ones(1, 4)
		>>> out  = Denormalize(mean, std)(x)
		>>> out.shape
		torch.Size([1, 4, 3, 3, 3])
	"""

	# MARK: Magic Functions
	
	def __init__(self, mean: Union[Tensor, float], std: Union[Tensor, float]):
		super().__init__()
		self.mean = mean
		self.std  = std

	def __repr__(self):
		repr = f"(mean={self.mean}, std={self.std})"
		return self.__class__.__name__ + repr

	# MARK: Forward Pass
	
	def forward(self, image: Tensor) -> Tensor:
		return denormalize(image, self.mean, self.std)
	

@TRANSFORMS.register(name="to_image")
class ToImage(nn.Module):
	"""Converts a PyTorch tensor to a numpy image. In case the image is in the
	GPU, it will be copied back to CPU.

	Args:
		keep_dims (bool):
			If `False` squeeze the input image to match the shape [H, W, C] or
			[H, W]. Default: `True`.
		denormalize (bool):
			If `True`, converts the image in the range [0.0, 1.0] to the range
			[0, 255]. Default: `False`.
	"""

	def __init__(self, keep_dims: bool = True, denormalize: bool = False):
		super().__init__()
		self.keep_dims    = keep_dims
		self.denormalize = denormalize

	def forward(self, image: Tensor) -> np.ndarray:
		return to_image(image, self.keep_dims, self.denormalize)


@TRANSFORMS.register(name="to_tensor")
class ToTensor(nn.Module):
	"""Convert a `PIL Image` or `np.ndarray` image to a 4d tensor.

	Args:
		keep_dims (bool):
			If `False` unsqueeze the image to match the shape [B, H, W, C].
			Default: `True`.
		normalize (bool):
			If `True`, converts the tensor in the range [0, 255] to the range
			[0.0, 1.0]. Default: `False`.
	"""

	def __init__(self, keep_dims: bool = False, normalize: bool = False):
		super().__init__()
		self.keep_dims  = keep_dims
		self.normalize = normalize

	def forward(self, image: Union[np.ndarray, PIL.Image]) -> Tensor:
		return to_tensor(image, self.keep_dims, self.normalize)


# MARK: - Alias

get_image_size = get_image_hw


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
