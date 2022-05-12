#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implement processing wrapper functions.
"""

from __future__ import annotations

import functools

import numpy as np
from torch import Tensor

from one.core import Callable
from one.core import is_channel_first
from one.core import TensorOrArray
from one.core import to_4d_array
from one.core import to_4d_tensor
from one.core import to_channel_first
from one.core import to_channel_last

__all__ = [
	"batch_image_processing",
	"channel_last_processing",
]


# MARK: - Functional

def batch_image_processing(func: Callable):
	"""Mostly used in cv2 related functions where it only accepts single image
	as input.
	"""
	
	@functools.wraps(func)
	def wrapper(image: TensorOrArray, *args, **kwargs) -> TensorOrArray:
		if not isinstance(image, (Tensor, np.ndarray)):
			raise ValueError(f"`image` must be a `Tensor` or `np.ndarray`. "
			                 f"But got: {type(image)}.")
		if image.ndim not in [3, 4]:
			raise ValueError(f"`image.ndim must be 3 or 4. But got: {image.ndim}.")
			
		if isinstance(image, Tensor):
			img = image.clone()
			img = to_4d_tensor(img)
		elif isinstance(image, np.ndarray):
			img = image.copy()
			img = to_4d_array(img)
 
		img = [func(i, *args, **kwargs) for i in img]
		
		if isinstance(image, Tensor):
			img = to_4d_tensor(img)
			if image.ndim == 3:
				img = img[0]
		elif isinstance(image, np.ndarray):
			img = to_4d_array(img)
			if image.ndim == 3:
				img = img[0]
			
		return img
		
	return wrapper


def channel_last_processing(func: Callable):
	"""Mostly used in cv2 related functions where input image is in channel last
	format.
	"""
	
	@functools.wraps(func)
	def wrapper(image: TensorOrArray, *args, **kwargs) -> TensorOrArray:
		img           = image.copy()
		channel_first = is_channel_first(img)
		
		if not isinstance(img, (Tensor, np.ndarray)):
			raise ValueError(f"Do not support {type(img)}.")
		if channel_first:
			img = to_channel_last(img)

		img = func(img, *args, **kwargs)
		
		if channel_first:
			img = to_channel_first(img)
		
		return img
		
	return wrapper
