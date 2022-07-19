#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implement processing wrapper functions.
"""

from __future__ import annotations

import functools
import inspect
import sys

import numpy as np
from torch import Tensor

from one.core.numpy import to_4d_array
from one.core.tensor import to_4d_tensor
from one.core.types import Callable
from one.core.types import TensorOrArray


# MARK: - Functional

def batch_image_processing(func: Callable):
	"""Process batch of images.
	"""
	
	@functools.wraps(func)
	def wrapper(image: TensorOrArray, *args, **kwargs) -> TensorOrArray:
		if not isinstance(image, (Tensor, np.ndarray)):
			raise TypeError(
				f"`image` must be a `Tensor` or `np.ndarray`. "
				f"But got: {type(image)}."
			)
		if not 3 <= image.ndim <= 4:
			raise ValueError(
				f"Require 3 <= image.ndim <= 4. But got: {image.ndim}."
			)
			
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
	"""Process channel last images.
	"""
	
	@functools.wraps(func)
	def wrapper(image: TensorOrArray, *args, **kwargs) -> TensorOrArray:
		img           = image.copy()
		from one.vision.transformation import is_channel_first
		channel_first = is_channel_first(img)
		
		if not isinstance(image, (Tensor, np.ndarray)):
			raise TypeError(
				f"`image` must be a `Tensor` or `np.ndarray`. "
				f"But got: {type(image)}."
			)
		
		if channel_first:
			from one.vision.transformation import to_channel_last
			img = to_channel_last(img)

		img = func(img, *args, **kwargs)
		
		if channel_first:
			from one.vision.transformation import to_channel_first
			img = to_channel_first(img)
		
		return img
		
	return wrapper


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
