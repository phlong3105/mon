#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements image-only datasets."""

from __future__ import annotations

__all__ = [
	"ImageEnhancementDataset",
	"ImageLoader",
	"LabeledImageDataset",
	"UnlabeledImageDataset",
]

import glob
from abc import ABC, abstractmethod
from typing import Any

import albumentations as A

from mon import core
from mon.data.datastruct import annotation as anno
from mon.data.datastruct.dataset import base
from mon.globals import Split

console         = core.console
ClassLabels     = anno.ClassLabels
ImageAnnotation = anno.ImageAnnotation


# region Unlabeled Image Dataset

class UnlabeledImageDataset(base.Dataset, ABC):
	"""The base class for datasets that represent an unlabeled collection of
	images. This is mainly used for unsupervised learning tasks.
	
	Attributes:
		datapoint_attrs: A :class:`dict` of datapoint attributes with the keys
			are the attribute names and the values are the attribute types. By
			default, it must contain ``{'input', 'target'}``.
	
	See Also: :class:`mon.data.datastruct.dataset.base.Dataset`.
	"""
	
	datapoint_attrs : dict = {
		"input" : ImageAnnotation,
		"target": ImageAnnotation,
	}
	
	def __getitem__(self, index: int) -> dict:
		"""Returns a dictionary containing the datapoint and metadata at the
		given :param:`index`. The dictionary must contain the following keys:
		{'input', 'target', 'meta'}.
		"""
		
		image = self.images[index].data
		meta  = self.images[index].meta
		
		if self.transform is not None:
			transformed = self.transform(image=image)
			image	    = transformed["image"]
		if self.to_tensor:
			image       = core.to_image_tensor(input=image, keepdim=False, normalize=True)
		
		return {
			"input" : image,
			"target": None,
			"meta"  : meta,
		}
	
	@abstractmethod
	def get_data(self):
		"""Get datapoints."""
		pass
	
	def reset(self):
		"""Reset and start over."""
		self.index = 0
	
	def close(self):
		"""Stop and release."""
		pass
	
	@staticmethod
	def collate_fn(batch) -> dict:
		"""Collate function used to fused input items together when using
		:attr:`batch_size` > 1. This is used in :class:`torch.utils.data.DataLoader` wrapper.
		
		Args:
			batch: A :class:`list` of :class:`dict` of {`input`, `target`, `meta`}.
		"""
		zipped = {k: list(v) for k, v in zip(batch[0].keys(), zip(*[b.values() for b in batch]))}
		input  = core.to_4d_image(zipped.get("input"))
		target = None
		meta   = zipped.get("meta")
		return {
			"input" : input,
			"target": target,
			"meta"  : meta,
		}


class ImageLoader(UnlabeledImageDataset):
	"""A general image loader that retrieves and loads image(s) from a file
	path, file path pattern, or directory.
	
	See Also: :class:`UnlabeledImageDataset`.
	"""
	
	def __init__(
		self,
		root       : core.Path,
		split      : Split       	    = Split.PREDICT,
		classlabels: ClassLabels | None = None,
		transform  : A.Compose   | None = None,
		to_tensor  : bool               = False,
		cache_data : bool               = False,
		verbose    : bool               = True,
		*args, **kwargs
	):
		super().__init__(
			root        = root,
			split		= split,
			classlabels = classlabels,
			transform   = transform,
			to_tensor   = to_tensor,
			cache_data	= cache_data,
			verbose     = verbose,
			*args, **kwargs
		)
	
	def get_data(self):
		# A single image
		if self.root.is_image_file():
			paths = [self.root]
		# A directory of images
		elif self.root.is_dir() and self.root.exists():
			paths = list(self.root.rglob("*"))
		# A file path pattern
		elif "*" in str(self.root):
			paths = [core.Path(i) for i in glob.glob(str(self.root))]
		else:
			raise IOError(f"Error when listing image files.")
		
		self.images: list[ImageAnnotation] = []
		with core.get_progress_bar() as pbar:
			for path in pbar.track(
				sorted(paths),
				description=f"[bright_yellow]Listing {self.__class__.__name__} {self.split_str} images"
			):
				if path.is_image_file():
					self.images.append(ImageAnnotation(path=path))

# endregion


# region Labeled Image Dataset

class LabeledImageDataset(base.Dataset, ABC):
	"""The base class for datasets that represent a labeled collection of images.
	
	Args:
		root: A root directory where the data is stored.
		split: The data split to use. Default: ``'Split.TRAIN'``.
		classlabels: :class:`ClassLabels` object. Default: ``None``.
		transform: Transformations performed on both the input and target.
		to_tensor: If ``True``, convert input and target to :class:`torch.Tensor`.
			Default: ``False``.
		cache_data: If ``True``, cache data to disk for faster loading next
			time. Default: ``False``.
		verbose: Verbosity. Default: ``True``.
	
	See Also: :class:`LabeledDataset`.
	"""
	
	def __init__(
		self,
		root       : core.Path,
		split      : Split         	    = Split.TRAIN,
		classlabels: ClassLabels | None = None,
		transform  : A.Compose   | None = None,
		to_tensor  : bool               = False,
		cache_data : bool               = False,
		verbose    : bool               = True,
		*args, **kwargs
	):
		super().__init__(
			root        = root,
			split       = split,
			classlabels = classlabels,
			transform   = transform,
			to_tensor   = to_tensor,
			verbose     = verbose,
			*args, **kwargs
		)
		self.images     : list[ImageAnnotation] = []
		self.annotations: list[Any]		        = []
		if not hasattr(self, "annotations"):
			self.annotations = []
		
		# Get images and annotations from disk or cache
		cache_file = self.root / f"{self.split_str}.cache"
		if cache_data and cache_file.is_file():
			self.load_cache(path=cache_file)
		else:
			self.get_images()
			if self.has_annotations:
				self.get_annotations()
		
		# Filter and verify data
		self.filter()
		self.verify()
		
		# Cache data
		if cache_data:
			self.cache_data(path=cache_file)
		else:
			core.delete_cache(cache_file)
	
	@abstractmethod
	def __getitem__(self, index: int) -> dict:
		"""Returns a dictionary containing the datapoint and metadata at the
		given :param:`index`. The dictionary must contain the following keys:
		{'input', 'target', 'meta'}.
		"""
		pass
	
	@abstractmethod
	def get_images(self):
		"""Get image files."""
		pass
	
	@abstractmethod
	def get_annotations(self):
		"""Get annotations files."""
		pass
	
	def reset(self):
		"""Reset and start over."""
		self.index = 0
	
	def close(self):
		"""Stop and release."""
		pass


class ImageEnhancementDataset(LabeledImageDataset, ABC):
	"""The base class for datasets that represent a collection of images, and a
	set of associated enhanced images.
	
	See Also: :class:`LabeledImageDataset`.
	"""
	
	def __init__(
		self,
		root       : core.Path,
		split      : Split          	= Split.TRAIN,
		classlabels: ClassLabels | None = None,
		transform  : A.Compose   | None = None,
		to_tensor  : bool               = False,
		cache_data : bool               = False,
		verbose    : bool               = True,
		*args, **kwargs
	):
		self.annotations: list[ImageAnnotation] = []
		super().__init__(
			root        = root,
			split       = split,
			classlabels = classlabels,
			transform   = transform,
			to_tensor   = to_tensor,
			cache_data  = cache_data,
			verbose     = verbose,
			*args, **kwargs
		)
	
	def __getitem__(self, index: int) -> dict:
		"""Returns a dictionary containing the datapoint and metadata at the
		given :param:`index`. The dictionary must contain the following keys:
		{'input', 'target', 'meta'}.
		"""
		image  = self.images[index].data
		target = self.annotations[index].data if self.has_annotations else None
		meta   = self.images[index].meta
		
		if self.transform is not None:
			if self.has_annotations:
				transformed = self.transform(image=image, mask=target)
				image       = transformed["image"]
				target      = transformed["mask"]
			else:
				transformed = self.transform(image=image)
				image       = transformed["image"]
		
		if self.to_tensor:
			if self.has_annotations:
				image  = core.to_image_tensor(input=image,  keepdim=False, normalize=True)
				target = core.to_image_tensor(input=target, keepdim=False, normalize=True)
			else:
				image = core.to_image_tensor(input=image, keepdim=False, normalize=True)
		
		return {
			"input"   : image,
			"target"  : target,
			"metadata": meta,
		}
		
	def filter(self):
		'''
		keep = []
		for i, (img, lab) in enumerate(zip(self._images, self._labels)):
			if img.path.is_image_file() and lab.path.is_image_file():
				keep.append(i)
		self._images = [img for i, img in enumerate(self._images) if i in keep]
		self._labels = [lab for i, lab in enumerate(self._labels) if i in keep]
		'''
		pass
	
	@staticmethod
	def collate_fn(batch) -> dict:
		"""Collate function used to fused input items together when using
		:attr:`batch_size` > 1. This is used in the
		:class:`torch.utils.data.DataLoader` wrapper.
		
		Args:
			batch: A :class:`list` of :class:`dict` of {`input`, `target`, `meta`}.
		"""
		zipped = {k: list(v) for k, v in zip(batch[0].keys(), zip(*[b.values() for b in batch]))}
		input  = core.to_4d_image(zipped.get("input"))
		target = core.to_4d_image(zipped.get("target"))
		meta   = zipped.get("meta")
		return {
			"input" : input,
			"target": target,
			"meta"  : meta,
		}

# endregion
