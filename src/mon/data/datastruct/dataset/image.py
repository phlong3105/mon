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
import numpy as np
import torch

from mon import core
from mon.data.datastruct import annotation as anno
from mon.data.datastruct.dataset import base
from mon.globals import Split

console         = core.console
ClassLabels     = anno.ClassLabels
ImageAnnotation = anno.ImageAnnotation


# region Unlabeled Image Dataset

class UnlabeledImageDataset(base.UnlabeledDataset, ABC):
	"""The base class for datasets that represent an unlabeled collection of
	images. This is mainly used for unsupervised learning tasks.
	
	Args:
		root: A root directory where the data is stored.
		split: The data split to use. Default: ``'Split.TRAIN'``.
		classlabels: :class:`ClassLabels` object. Default: ``None``.
		transform: Transformations performed on both the input and target. We
			use `albumentations <https://albumentations.ai/docs/api_reference/full_reference>`__
		to_tensor: If True, convert input and target to :class:`torch.Tensor`.
			Default: ``False``.
		cache_data: If ``True``, cache data to disk for faster loading next
			time. Default: ``False``.
		verbose: Verbosity. Default: ``True``.
	
	See Also: :class:`UnlabeledDataset`.
	"""
	
	def __init__(
		self,
		root       : core.Path,
		split      : Split              = Split.TRAIN,
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
		self.images: list[ImageAnnotation] = []
		
		# Get image from disk or cache
		cache_file = self.root / f"{self.split_str}.cache"
		if cache_data and cache_file.is_cache_file():
			self.load_cache(path=cache_file)
		else:
			self.get_images()
		
		# Filter and verify data
		self.filter()
		self.verify()
		
		# Cache data
		if cache_data:
			self.cache_data(path=cache_file)
		else:
			core.delete_cache(cache_file)
		
	def __len__(self) -> int:
		return len(self.images)
	
	def __getitem__(self, index: int) -> tuple[
		torch.Tensor | np.ndarray,
		torch.Tensor | np.ndarray | None,
		dict | None
	]:
		image = self.images[index].data
		meta  = self.images[index].meta
		
		if self.transform is not None:
			transformed = self.transform(image=image)
			image	    = transformed["image"]
		if self.to_tensor:
			image = core.to_image_tensor(input=image, keepdim=False, normalize=True)

		return image, None, meta
	
	@property
	def hash(self) -> int:
		"""Return the total hash value of all the files (if it has one).
		Hash values are integers (in bytes) of all files.
		"""
		return sum(i.meta.get("hash", 0) for i in self.images) if self.images else 0
	
	@abstractmethod
	def get_images(self):
		"""Get image files."""
		pass
	
	def filter(self):
		"""Filter unwanted samples."""
		pass
	
	def verify(self):
		"""Verify and check data."""
		if not len(self.images) > 0:
			raise RuntimeError(f"No images in dataset.")
		if self.verbose:
			console.log(f"Number of samples: {len(self.images)}.")
	
	def cache_data(self, path: core.Path):
		"""Cache data to :param:`path`."""
		hash_ = 0
		if path.is_cache_file():
			cache = torch.load(path)
			hash_ = cache.get("hash", 0)
			
		if self.hash != hash_:
			cache = {
				"hash"  : self.hash,
				"images": self.images,
			}
			torch.save(cache, str(path))
			if self.verbose:
				console.log(f"Cached data to: {path}")
		
	def load_cache(self, path: core.Path):
		"""Load cache data from :param:`path`."""
		cache       = torch.load(path)
		self.images = cache["images"]
	
	def reset(self):
		"""Reset and start over."""
		self.index = 0
	
	def close(self):
		"""Stop and release."""
		pass
	
	@staticmethod
	def collate_fn(batch) -> tuple[
		torch.Tensor | np.ndarray,
		torch.Tensor | np.ndarray | None,
		list | None
	]:
		"""Collate function used to fused input items together when using
		:attr:`batch_size` > 1. This is used in
		:class:`torch.utils.data.DataLoader` wrapper.
		
		Args:
			batch: A list of tuples of (input, meta).
		"""
		input, target, meta = zip(*batch)  # Transposed
		
		if all(isinstance(i, torch.Tensor)   and i.ndim == 3 for i in input):
			input = torch.stack(input, dim=0)
		elif all(isinstance(i, torch.Tensor) and i.ndim == 4 for i in input):
			input = torch.cat(input, dim=0)
		elif all(isinstance(i, np.ndarray)   and i.ndim == 3 for i in input):
			input = np.array(input)
		elif all(isinstance(i, np.ndarray)   and i.ndim == 4 for i in input):
			input = np.concatenate(input, axis=0)
		else:
			raise ValueError(f"input's number of dimensions must be between ``2`` and ``4``.")
		
		target = None
		return input, target, meta


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
	
	def get_images(self):
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

class LabeledImageDataset(base.LabeledDataset, ABC):
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
		cache_images: If ``True``, cache images into memory for faster training
			(WARNING: large datasets may exceed system RAM). Default: ``False``.
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
	
	def __len__(self) -> int:
		return len(self.images)
	
	@abstractmethod
	def __getitem__(self, index: int) -> tuple[
		torch.Tensor | np.ndarray,
		Any,
		dict | None
	]:
		"""Return the image, ground-truth, and metadata, optionally transformed
		by the respective transforms.
		"""
		pass
	
	@property
	def hash(self) -> int:
		"""Return the total hash value of all the files (if it has one).
		Hash values are integers (in bytes) of all files.
		"""
		hash_  = 0
		hash_ += sum(i.meta.get("hash", 0) for i in self.images) if self.images else 0
		hash_ += sum(i.meta.get("hash", 0) for i in self.annotations) if self.annotations else 0
		return hash_
	
	@abstractmethod
	def get_images(self):
		"""Get image files."""
		pass
	
	@abstractmethod
	def get_annotations(self):
		"""Get annotations files."""
		pass
	
	def filter(self):
		"""Filter unwanted samples."""
		pass
	
	def verify(self):
		"""Verify and check data."""
		if not len(self.images) > 0:
			raise RuntimeError(f"No images in dataset.")
		if self.has_annotations and not len(self.images) == len(self.annotations):
			raise RuntimeError(
				f"Number of images and annotations must be the same, but got "
				f"{len(self.images)} and {len(self.annotations)}."
			)
		if self.verbose:
			console.log(f"Number of {self.split_str} samples: {len(self.images)}.")
	
	def cache_data(self, path: core.Path):
		"""Cache data to :param:`path`."""
		hash_ = 0
		if path.is_cache_file():
			cache = torch.load(path)
			hash_ = cache.get("hash", 0)
			
		if self.hash != hash_:
			cache = {
				"hash"	     : self.hash,
				"images"     : self.images,
				"annotations": self.annotations,
			}
			torch.save(cache, str(path))
			if self.verbose:
				console.log(f"Cached data to: {path}")
	
	def load_cache(self, path: core.Path):
		"""Load cache data from :param:`path`."""
		cache            = torch.load(path)
		self.images      = cache["images"]
		self.annotations = cache["annotations"]
	
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
	
	def __getitem__(self, index: int) -> tuple[
		torch.Tensor | np.ndarray,
		torch.Tensor | np.ndarray | None,
		dict | None
	]:
		image = self.images[index].data
		mask  = self.annotations[index].data if self.has_annotations else None
		meta  = self.images[index].meta
		
		if self.transform is not None:
			if self.has_annotations:
				transformed = self.transform(image=image, mask=mask)
				image       = transformed["image"]
				mask        = transformed["mask"]
			else:
				transformed = self.transform(image=image)
				image       = transformed["image"]
		
		if self.to_tensor:
			if self.has_annotations:
				image = core.to_image_tensor(input=image, keepdim=False, normalize=True)
				mask  = core.to_image_tensor(input=mask,  keepdim=False, normalize=True)
			else:
				image = core.to_image_tensor(input=image, keepdim=False, normalize=True)
		
		return image, mask, meta
		
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
	def collate_fn(batch) -> tuple[
		torch.Tensor | np.ndarray,
		torch.Tensor | np.ndarray | list | None,
		list | None
	]:
		"""Collate function used to fused input items together when using
		:attr:`batch_size` > 1. This is used in the
		:class:`torch.utils.data.DataLoader` wrapper.
		
		Args:
			batch: a list of tuples of (input, target, meta).
		"""
		input, target, meta = zip(*batch)  # Transposed
		
		if all(isinstance(i, torch.Tensor)   and i.ndim == 3 for i in input):
			input = torch.stack(input, dim=0)
		elif all(isinstance(i, torch.Tensor) and i.ndim == 4 for i in input):
			input = torch.cat(input, dim=0)
		elif all(isinstance(i, np.ndarray)   and i.ndim == 3 for i in input):
			input = np.array(input)
		elif all(isinstance(i, np.ndarray)   and i.ndim == 4 for i in input):
			input = np.concatenate(input, axis=0)
		else:
			raise ValueError(f"input's number of dimensions must be between ``2`` and ``4``.")
		
		if all(isinstance(t, torch.Tensor)   and t.ndim == 3 for t in target):
			target = torch.stack(target, dim=0)
		elif all(isinstance(t, torch.Tensor) and t.ndim == 4 for t in target):
			target = torch.cat(target, dim=0)
		elif all(isinstance(t, np.ndarray)   and t.ndim == 3 for t in target):
			target = np.array(target)
		elif all(isinstance(t, np.ndarray)   and t.ndim == 4 for t in target):
			target = np.concatenate(target, axis=0)
		else:
			target = None
		
		return input, target, meta

# endregion
