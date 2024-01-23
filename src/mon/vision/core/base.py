#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base classes for image, video, and pointcloud
data.
"""

from __future__ import annotations

__all__ = [
	"Loader",
	"Writer",
]

from abc import ABC, abstractmethod

import numpy as np
import torch

from mon.vision import core


class Loader(ABC):
	"""The base class for all image and video loaders.
	
	Args:
		source: A data source. It can be a path to a single image file, a
			directory, a video file, or a stream. It can also be a path pattern.
		max_samples: The maximum number of datapoints from the given
			:param:`source` to process. Default: ``None``.
		batch_size: The number of samples in a single forward pass.
			Default: ``1``.
		to_rgb: If ``True``, convert the image from BGR to RGB.
			Default: ``False``.
		to_tensor: If ``True``, convert the image from :class:`numpy.ndarray` to
			:class:`torch.Tensor`. Default: ``False``.
		normalize: If ``True``, normalize the image to :math:`[0.0, 1.0]`.
			Default: ``True``.
		verbose: Verbosity. Default: ``False``.
	"""
	
	def __init__(
		self,
		source     : core.Path,
		max_samples: int | None = None,
		batch_size : int        = 1,
		to_rgb     : bool       = True,
		to_tensor  : bool       = False,
		normalize  : bool       = False,
		verbose    : bool       = False,
		*args, **kwargs
	):
		super().__init__()
		self.source      = core.Path(source)
		self.batch_size  = batch_size
		self.to_rgb      = to_rgb
		self.to_tensor   = to_tensor
		self.normalize   = normalize
		self.verbose     = verbose
		self.index       = 0
		self.max_samples = max_samples
		self.num_images  = 0
		self.init()
	
	def __iter__(self):
		"""Return an iterator object starting at index ``0``."""
		self.reset()
		return self
	
	def __len__(self) -> int:
		"""Return the number of images in the dataset."""
		return self.num_images
	
	@abstractmethod
	def __next__(self):
		pass
	
	def __del__(self):
		"""Close."""
		self.close()
	
	def batch_len(self) -> int:
		"""Return the number of batches."""
		return int(self.__len__() / self.batch_size)
	
	@abstractmethod
	def init(self):
		"""Initialize the data source."""
		pass
	
	@abstractmethod
	def reset(self):
		"""Reset and start over."""
		pass
	
	@abstractmethod
	def close(self):
		"""Stop and release."""
		pass


class Writer(ABC):
	"""The base class for all image and video writers that save images to a
	destination directory.

	Args:
		destination: A directory to save images.
		image_size: An output image size of shape :math:`[H, W]`.
			Default: :math:`[480, 640]`.
		denormalize: If ``True``, convert image to :math:`[0, 255]`.
			Default: ``False``.
		verbose: Verbosity. Default: ``False``.
	"""
	
	def __init__(
		self,
		destination: core.Path,
		image_size : int | list[int] = [480, 640],
		denormalize: bool = False,
		verbose    : bool = False,
		*args, **kwargs
	):
		super().__init__()
		self.dst         = core.Path(destination)
		self.img_size    = core.get_hw(size=image_size)
		self.denormalize = denormalize
		self.verbose     = verbose
		self.index       = 0
		self.init()
	
	def __len__(self) -> int:
		"""Return the number frames of already written frames."""
		return self.index
	
	def __del__(self):
		"""Close."""
		self.close()
	
	@abstractmethod
	def init(self):
		"""Initialize the output handler."""
		pass
	
	@abstractmethod
	def close(self):
		"""Close."""
		pass
	
	@abstractmethod
	def write(
		self,
		image      : torch.Tensor | np.ndarray,
		path       : core.Path | None = None,
		denormalize: bool = False
	):
		"""Write an image to :attr:`dst`.

		Args:
			image: An image.
			path: An image file path with an extension. Default: ``None``.
			denormalize: If ``True``, convert image to :math:`[0, 255]`.
				Default: ``False``.
		"""
		pass
	
	@abstractmethod
	def write_batch(
		self,
		images     : list[torch.Tensor  | np.ndarray],
		paths      : list[core.Path] | None = None,
		denormalize: bool = False
	):
		"""Write a batch of images to :attr:`dst`.

		Args:
			images: A :class:`list` of images.
			paths: A :class:`list` of image file paths with extensions.
				Default: ``None``.
			denormalize: If ``True``, convert image to :math:`[0, 255]`.
				Default: ``False``.
		"""
		pass
