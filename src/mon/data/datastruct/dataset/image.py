#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements image-only datasets."""

from __future__ import annotations

__all__ = [
	"ImageEnhancementDataset",
	"ImageLoader",
	"LabeledImageDataset",
	"LabeledImageInpaintingDataset",
	"UnlabeledImageDataset",
	"UnlabeledImageInpaintingDataset",
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

console	    = core.console
ClassLabels = anno.ClassLabels
FrameLabel  = anno.FrameAnnotation
ImageLabel  = anno.ImageAnnotation


# region Unlabeled Image Dataset

class UnlabeledImageDataset(base.UnlabeledDataset, ABC):
	"""The base class for datasets that represent an unlabeled collection of
	images. This is mainly used for unsupervised learning tasks.
	
	Args:
		root: A root directory where the data is stored.
		split: The data split to use. One of: [``'train'``, ``'val'``,
			``'test'``, ``'predict'``]. Default: ``'train'``.
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
		split      : Split 			    = Split.TRAIN,
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
		self._images: list[ImageLabel] = []
		
		cache_file = self.root / f"{self.split_str}.cache"
		if cache_data and cache_file.is_cache_file():
			self.load_cache(path=cache_file)
		else:
			self._get_images()
		
		self._filter()
		self._verify()
		
		if cache_data:
			self.cache_data(path=cache_file)
		else:
			core.delete_cache(cache_file)
		
	def __len__(self) -> int:
		return len(self._images)
	
	def __getitem__(self, index: int) -> tuple[
		torch.Tensor | np.ndarray,
		torch.Tensor | np.ndarray | None,
		dict | None
	]:
		image = self._images[index].data
		meta  = self._images[index].meta
		
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
		return sum(i.meta.get("hash", 0) for i in self._images) if self._images else 0
	
	@abstractmethod
	def _get_images(self):
		"""Get image files."""
		pass
	
	def _filter(self):
		"""Filter unwanted samples."""
		pass
	
	def _verify(self):
		"""Verify and check data."""
		if not len(self._images) > 0:
			raise RuntimeError(f"No images in dataset.")
		if self.verbose:
			console.log(f"Number of samples: {len(self._images)}.")
	
	def cache_data(self, path: core.Path):
		"""Cache data to :param:`path`."""
		hash = 0
		if path.is_cache_file():
			_cache = torch.load(path)
			hash   = _cache.get("hash", 0)
			
		if self.hash != hash:
			cache = {
				"hash"  : self.hash,
				"images": self._images,
			}
			torch.save(cache, str(path))
			if self.verbose:
				console.log(f"Cached data to: {path}")
		
	def load_cache(self, path: core.Path):
		"""Load cache data from :param:`path`."""
		cache       = torch.load(path)
		self._images = cache["images"]
	
	def reset(self):
		"""Reset and start over."""
		self._index = 0
	
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


class UnlabeledImageInpaintingDataset(base.UnlabeledDataset, ABC):
	"""The base class for inpainting datasets that represent a collection of
	images, and associated masks.
 
	Args:
		root: A root directory where the data is stored.
		split: The data split to use. One of: [``'train'``, ``'val'``,
			``'test'``, ``'predict'``]. Default: ``'train'``.
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
		split      : Split 			    = Split.TRAIN,
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
		self._images: list[ImageLabel] = []
		self._masks : list[ImageLabel] = []
		
		cache_file = self.root / f"{self.split_str}.cache"
		if cache_data and cache_file.is_cache_file():
			self.load_cache(path=cache_file)
		else:
			self._get_images()
			self._get_masks()
		
		self._filter()
		self._verify()
		
		if cache_data:
			self.cache_data(path=cache_file)
		else:
			core.delete_cache(cache_file)
		
	def __len__(self) -> int:
		return len(self._images)
	
	def __getitem__(self, index: int) -> tuple[
		torch.Tensor | np.ndarray,
		torch.Tensor | np.ndarray,
		torch.Tensor | np.ndarray | None,
		dict | None
	]:
		image = self._images[index].data
		mask  = self._masks[index].data
		meta  = self._images[index].meta
		
		if self.transform is not None:
			transformed = self.transform(image=image, mask=mask)
			image       = transformed["image"]
			mask        = transformed["mask"]
		
		if self.to_tensor:
			image = core.to_image_tensor(input=image, keepdim=False, normalize=True)
			mask  = core.to_image_tensor(input=mask,  keepdim=False, normalize=True)
		
		return image, mask, None, meta
	
	@property
	def hash(self) -> int:
		"""Return the total hash value of all the files (if it has one).
		Hash values are integers (in bytes) of all files.
		"""
		hash  = 0
		hash += sum(i.meta.get("hash", 0) for i in self._images) if self._images else 0
		hash += sum(i.meta.get("hash", 0) for i in self._masks)  if self._masks  else 0
		return hash
	
	@abstractmethod
	def _get_images(self):
		"""Get image files."""
		pass
	
	@abstractmethod
	def _get_masks(self):
		"""Get mask files."""
		pass
	
	def _filter(self):
		"""Filter unwanted samples."""
		pass
	
	def _verify(self):
		"""Verify and check data."""
		if not len(self._images) > 0:
			raise RuntimeError(f"No images in dataset.")
		if not len(self._images) == len(self._masks):
			raise RuntimeError(
				f"Number of images and masks must be the same, but got "
				f"{len(self._images)} and {len(self._masks)}."
			)
		if self.verbose:
			console.log(f"Number of {self.split_str} samples: {len(self._images)}.")
	
	def cache_data(self, path: core.Path):
		"""Cache data to :param:`path`."""
		hash = 0
		if path.is_cache_file():
			_cache = torch.load(path)
			hash   = _cache.get("hash", 0)
			
		if self.hash != hash:
			cache = {
				"hash"  : self.hash,
				"images": self._images,
				"masks" : self._masks,
			}
			torch.save(cache, str(path))
			if self.verbose:
				console.log(f"Cached data to: {path}")
		
	def load_cache(self, path: core.Path):
		"""Load cache data from :param:`path`."""
		cache        = torch.load(path)
		self._images = cache["images"]
		self._masks  = cache["masks"]
		
	def reset(self):
		"""Reset and start over."""
		self._index = 0
	
	def close(self):
		"""Stop and release."""
		pass
	
	@staticmethod
	def collate_fn(batch) -> tuple[
		torch.Tensor | np.ndarray,
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
		input, mask, target, meta = zip(*batch)  # Transposed
		
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
		
		if all(isinstance(i, torch.Tensor)   and i.ndim == 3 for i in input):
			mask = torch.stack(input, dim=0)
		elif all(isinstance(i, torch.Tensor) and i.ndim == 4 for i in input):
			mask = torch.cat(input, dim=0)
		elif all(isinstance(i, np.ndarray)   and i.ndim == 3 for i in input):
			mask = np.array(input)
		elif all(isinstance(i, np.ndarray)   and i.ndim == 4 for i in input):
			mask = np.concatenate(input, axis=0)
		else:
			raise ValueError(f"input's number of dimensions must be between ``2`` and ``4``.")
		
		target = None
		return input, mask, target, meta


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
	
	def _get_images(self):
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
		
		self._images: list[ImageLabel] = []
		with core.get_progress_bar() as pbar:
			for path in pbar.track(
				sorted(paths),
				description=f"[bright_yellow]Listing {self.__class__.__name__} {self.split_str} images"
			):
				if path.is_image_file():
					self._images.append(ImageLabel(path=path))
	
# endregion


# region Labeled Image Dataset

class LabeledImageDataset(base.LabeledDataset, ABC):
	"""The base class for datasets that represent a labeled collection of images.
	
	Args:
		root: A root directory where the data is stored.
		split: The data split to use. One of: [``'train'``, ``'val'``,
			``'test'``, ``'predict'``]. Default: ``'train'``.
		classlabels: :class:`ClassLabels` object. Default: ``None``.
		has_test_label: If ``True``, the test set has ground-truth labels.
			Default: ``False``.
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
		self._images: list[ImageLabel] = []
		self._labels: list[Any]		   = []
		if not hasattr(self, "labels"):
			self._labels = []
		
		cache_file = self.root / f"{self.split_str}.cache"
		if cache_data and cache_file.is_file():
			self.load_cache(path=cache_file)
		else:
			self._get_images()
			if self.has_test_label:
				self._get_labels()
		
		self._filter()
		self._verify()
		
		if cache_data:
			self.cache_data(path=cache_file)
		else:
			core.delete_cache(cache_file)
	
	def __len__(self) -> int:
		return len(self._images)
	
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
		hash  = 0
		hash += sum(i.meta.get("hash", 0) for i in self._images) if self._images else 0
		hash += sum(i.meta.get("hash", 0) for i in self._labels) if self._labels else 0
		return hash
	
	@abstractmethod
	def _get_images(self):
		"""Get image files."""
		pass
	
	@abstractmethod
	def _get_labels(self):
		"""Get label files."""
		pass
	
	def _filter(self):
		"""Filter unwanted samples."""
		pass
	
	def _verify(self):
		"""Verify and check data."""
		if not len(self._images) > 0:
			raise RuntimeError(f"No images in dataset.")
		if self.has_test_label and not len(self._images) == len(self._labels):
			raise RuntimeError(
				f"Number of images and labels must be the same, but got "
				f"{len(self._images)} and {len(self._labels)}."
			)
		if self.verbose:
			console.log(f"Number of {self.split_str} samples: {len(self._images)}.")
	
	def cache_data(self, path: core.Path):
		"""Cache data to :param:`path`."""
		hash = 0
		if path.is_cache_file():
			_cache = torch.load(path)
			hash   = _cache.get("hash", 0)
			
		if self.hash != hash:
			cache = {
				"hash"	: self.hash,
				"images": self._images,
				"labels": self._labels,
			}
			torch.save(cache, str(path))
			if self.verbose:
				console.log(f"Cached data to: {path}")
	
	def load_cache(self, path: core.Path):
		"""Load cache data from :param:`path`."""
		cache        = torch.load(path)
		self._images = cache["images"]
		self._labels = cache["labels"]
	
	def reset(self):
		"""Reset and start over."""
		self._index = 0
	
	def close(self):
		"""Stop and release."""
		pass


class LabeledImageInpaintingDataset(base.LabeledDataset, ABC):
	"""The base class for inpainting datasets that represent a collection of
	images, associated masks, and target.
	
	Args:
		root: A root directory where the data is stored.
		split: The data split to use. One of: [``'train'``, ``'val'``,
			``'test'``, ``'predict'``]. Default: ``'train'``.
		classlabels: :class:`ClassLabels` object. Default: ``None``.
		has_test_label: If ``True``, the test set has ground-truth labels.
			Default: ``False``.
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
		self._images: list[ImageLabel] = []
		self._masks : list[ImageLabel] = []
		self._labels: list[ImageLabel] = []
		if not hasattr(self, "labels"):
			self._labels = []
		
		cache_file = self.root / f"{self.split_str}.cache"
		if cache_data and cache_file.is_file():
			self.load_cache(path=cache_file)
		else:
			self._get_images()
			self._get_masks()
			if self.has_test_label:
				self._get_labels()
		
		self._filter()
		self._verify()
		
		if cache_data:
			self.cache_data(path=cache_file)
		else:
			core.delete_cache(cache_file)
	
	def __len__(self) -> int:
		return len(self._images)
	
	def __getitem__(self, index: int) -> tuple[
		torch.Tensor | np.ndarray,
		torch.Tensor | np.ndarray,
		torch.Tensor | np.ndarray | None,
		dict | None
	]:
		image = self._images[index].data
		mask  = self._masks[index].data
		label = self._labels[index].data if self.has_test_label else None
		meta  = self._images[index].meta
		
		if self.transform is not None:
			if self.has_test_label:
				transformed = self.transform(image=image, masks=[mask, label])
				image       = transformed["image"]
				masks       = transformed["masks"]
				mask        = masks[0]
				label       = masks[1]
			else:
				transformed = self.transform(image=image, mask=mask)
				image       = transformed["image"]
				mask        = transformed["mask"]
			
		if self.to_tensor:
			if self.has_test_label:
				image = core.to_image_tensor(input=image, keepdim=False, normalize=True)
				mask  = core.to_image_tensor(input=mask,  keepdim=False, normalize=True)
				label = core.to_image_tensor(input=label, keepdim=False, normalize=True)
			else:
				image = core.to_image_tensor(input=image, keepdim=False, normalize=True)
				mask  = core.to_image_tensor(input=mask,  keepdim=False, normalize=True)
		
		return image, mask, label, meta
	
	@property
	def hash(self) -> int:
		"""Return the total hash value of all the files (if it has one).
		Hash values are integers (in bytes) of all files.
		"""
		hash  = 0
		hash += sum(i.meta.get("hash", 0) for i in self._images) if self._images else 0
		hash += sum(i.meta.get("hash", 0) for i in self._masks)  if self._masks  else 0
		hash += sum(i.meta.get("hash", 0) for i in self._labels) if self._labels else 0
		return hash
	
	@abstractmethod
	def _get_images(self):
		"""Get image files."""
		pass
	
	@abstractmethod
	def _get_masks(self):
		"""Get mask files."""
		pass
	
	@abstractmethod
	def _get_labels(self):
		"""Get label files."""
		pass
	
	def _filter(self):
		"""Filter unwanted samples."""
		pass
	
	def _verify(self):
		"""Verify and check data."""
		if not len(self._images) > 0:
			raise RuntimeError(f"No images in dataset.")
		if (
			self.has_test_label
			and not len(self._images) == len(self._masks)
			and not len(self._images) == len(self._labels)
		):
			raise RuntimeError(
				f"Number of images, masks and labels must be the same, but got "
				f"{len(self._images)}, {len(self._masks)}and {len(self._labels)}."
			)
		if self.verbose:
			console.log(f"Number of {self.split_str} samples: {len(self._images)}.")
	
	def cache_data(self, path: core.Path):
		"""Cache data to :param:`path`."""
		hash = 0
		if path.is_cache_file():
			_cache = torch.load(path)
			hash   = _cache.get("hash", 0)
		
		if self.hash != hash:
			cache = {
				"hash"	: self.hash,
				"images": self._images,
				"masks" : self._masks,
				"labels": self._labels,
			}
			torch.save(cache, str(path))
			if self.verbose:
				console.log(f"Cached data to: {path}")
	
	def load_cache(self, path: core.Path):
		"""Load cache data from :param:`path`."""
		cache        = torch.load(path)
		self._images = cache["images"]
		self._masks  = cache["masks"]
		self._labels = cache["labels"]
	
	def reset(self):
		"""Reset and start over."""
		self._index = 0
	
	def close(self):
		"""Stop and release."""
		pass
	
	@staticmethod
	def collate_fn(batch) -> tuple[
		torch.Tensor | np.ndarray,
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
		input, mask, target, meta = zip(*batch)  # Transposed
		
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
		
		if all(isinstance(i, torch.Tensor)   and i.ndim == 3 for i in input):
			mask = torch.stack(input, dim=0)
		elif all(isinstance(i, torch.Tensor) and i.ndim == 4 for i in input):
			mask = torch.cat(input, dim=0)
		elif all(isinstance(i, np.ndarray)   and i.ndim == 3 for i in input):
			mask = np.array(input)
		elif all(isinstance(i, np.ndarray)   and i.ndim == 4 for i in input):
			mask = np.concatenate(input, axis=0)
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
			
		return input, mask, target, meta
	

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
		self._labels: list[ImageLabel] = []
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
		image = self._images[index].data
		label = self._labels[index].data if self.has_test_label else None
		meta  = self._images[index].meta
		
		if self.transform is not None:
			if self.has_test_label:
				transformed = self.transform(image=image, mask=label)
				image       = transformed["image"]
				label       = transformed["mask"]
			else:
				transformed = self.transform(image=image)
				image       = transformed["image"]
		
		if self.to_tensor:
			if self.has_test_label:
				image = core.to_image_tensor(input=image, keepdim=False, normalize=True)
				label = core.to_image_tensor(input=label, keepdim=False, normalize=True)
			else:
				image = core.to_image_tensor(input=image, keepdim=False, normalize=True)
		
		return image, label, meta
		
	def _filter(self):
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
