#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Video IO Module.

This module implements the basic I/O functionalities of video data.
"""

from __future__ import annotations

__all__ = [
	"read_video_ffmpeg",
	"write_video_ffmpeg",
]

import numpy as np
import torch

from mon.core import image as ci


# region Read

def read_video_ffmpeg(
	process,
	height   : int,
	width    : int,
	to_tensor: bool = False,
	normalize: bool = False,
) -> torch.Tensor | np.ndarray:
	"""Read raw bytes from a video stream using :mod`ffmpeg`. Optionally,
	convert it to :obj:`torch.Tensor` type of shape `[1, C, H, W]`.
	
	Args:
		process: The subprocess that manages :obj:`ffmpeg` instance.
		height: The height of the video frame.
		width: The width of the video.
		to_tensor: If ``True`` convert the image from :obj:`numpy.ndarray` to
			:obj:`torch.Tensor`. Default: ``False``.
		normalize: If ``True``, normalize the image to ``[0.0, 1.0]``.
			Default: ``False``.
	
	Return:
		A :obj:`numpy.ndarray` image of shape `[H, W, C]` with value in
		range ``[0, 255]`` or a :obj:`torch.Tensor` image of shape
		`[1, C, H, W]` with value in range ``[0.0, 1.0]``.
	"""
	# RGB24 == 3 bytes per pixel.
	img_size = height * width * 3
	in_bytes = process.stdout.read(img_size)
	if len(in_bytes) == 0:
		image = None
	else:
		if len(in_bytes) != img_size:
			raise ValueError()
		image = (
			np
			.frombuffer(in_bytes, np.uint8)
			.reshape([height, width, 3])
		)  # Numpy
		if to_tensor:
			image = ci.to_image_tensor(
				image= image,
				keepdim   = False,
				normalize = normalize
			)
	return image

# endregion


# region Write

def write_video_ffmpeg(
	process,
	image      : torch.Tensor | np.ndarray,
	denormalize: bool = False
):
	"""Write an image to a video file using :obj:`ffmpeg`.

	Args:
		process: A subprocess that manages :obj:``ffmpeg``.
		image: A frame/image of shape `[1, C, H, W]`.
		denormalize: If ``True``, convert image to ``[0, 255]``. Default: ``False``.
	"""
	if isinstance(image, np.ndarray):
		if ci.is_normalized_image(image=image):
			image = ci.denormalize_image(image=image)
		if ci.is_channel_first_image(image=image):
			image = ci.to_channel_last_image(image=image)
	elif isinstance(image, torch.Tensor):
		image = ci.to_image_nparray(
			image= image,
			keepdim     = False,
			denormalize = denormalize
		)
	else:
		raise ValueError(
			f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
			f"but got {type(image)}."
		)
	process.stdin.write(
		image
		.astype(np.uint8)
		.tobytes()
	)

# endregion
