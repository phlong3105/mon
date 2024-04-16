#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements base classes for all datasets.

For transformation operations, we use
`albumentations <https://albumentations.ai/docs/api_reference/full_reference>`__
"""

from __future__ import annotations

__all__ = [
	"ChainDataset",
	"ConcatDataset",
	"Dataset",
	"DetectionDatasetCOCO",
	"DetectionDatasetVOC",
	"DetectionDatasetYOLO",
	"ImageClassificationDataset",
	"ImageClassificationDirectoryTree",
	"ImageDetectionDataset",
	"ImageEnhancementDataset",
	"ImageLoader",
	"ImageSegmentationDataset",
	"IterableDataset",
	"LabeledDataset",
	"LabeledImageDataset",
	"LabeledImageInpaintingDataset",
	"Subset",
	"TensorDataset",
	"UnlabeledDataset",
	"UnlabeledImageDataset",
	"UnlabeledImageInpaintingDataset",
	"UnlabeledVideoDataset",
	"VideoLoaderCV",
	"VideoLoaderFFmpeg",
	"random_split",
]

import glob
import subprocess
import uuid
from abc import ABC, abstractmethod
from typing import Any

import albumentations as A
import cv2
import ffmpeg
import numpy as np
import torch
from torch.utils.data import dataset
from torch.utils.data.dataset import *

from mon import core
from mon.data.base import label as L
from mon.globals import BBoxFormat, Split, Task

console	= core.console

ClassLabels         = L.ClassLabels
ClassificationLabel = L.ClassificationLabel
DetectionsLabel     = L.DetectionsLabel
FrameLabel          = L.FrameLabel
ImageLabel          = L.ImageLabel
SegmentationLabel   = L.SegmentationLabel
VOCDetectionsLabel  = L.DetectionsLabelVOC
YOLODetectionsLabel = L.DetectionsLabelYOLO


# region Base

class Dataset(dataset.Dataset, ABC):
	"""The base class of all datasets.
	
	Attributes:
		_tasks: A list of tasks that the dataset supports.
		_splits: A list of splits that the dataset supports.
	
	Args:
		root: The root directory where the data is stored.
		split: The data split to use. One of: [``'train'``, ``'val'``,
			``'test'``, ``'predict'``]. Default: ``'train'``.
		classlabels: :class:`ClassLabels` object. Default: ``None``.
		transform: Transformations performed on both the input and target.
		to_tensor: If ``True``, convert input and target to :class:`torch.Tensor`.
			Default: ``False``.
		verbose: Verbosity. Default: ``True``.
	
	See Also: :mod:`dataset.Dataset`.
	"""
	
	_tasks : list[Task]  = []
	_splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST, Split.PREDICT]
	
	def __init__(
		self,
		root       : core.Path,
		split      : Split          	= Split.TRAIN,
		classlabels: ClassLabels | None = None,
		transform  : Any                = None,
		to_tensor  : bool               = False,
		verbose    : bool               = False,
		*args, **kwargs
	):
		super().__init__()
		self._root        = core.Path(root)
		self._classlabels = ClassLabels.from_value(value=classlabels)
		self._split       = None
		self._init_split(split)
		self._transform   = transform
		self.to_tensor    = to_tensor
		self.verbose      = verbose
		self._index		  = 0  # Use with :meth:`__iter__` and :meth`__next__`
		
	def __iter__(self):
		"""Returns an iterator starting at the index ``0``."""
		self.reset()
		return self
	
	@abstractmethod
	def __len__(self) -> int:
		"""Return the total number of datapoints in the dataset."""
		pass
	
	@abstractmethod
	def __getitem__(self, index: int) -> Any:
		"""Returns the datapoint and metadata at the given :param:`index`."""
		pass
	
	def __next__(self) -> Any:
		"""Returns the next datapoint and metadata when using :meth:`__iter__`."""
		if self._index >= self.__len__():
			raise StopIteration
		else:
			result = self.__getitem__(self._index)
			self._index += 1
			return result
	
	def __repr__(self) -> str:
		head = "Dataset " + self.__class__.__name__
		body = [f"Number of datapoints: {self.__len__()}"]
		if self.root is not None:
			body.append(f"Root location: {self.root}")
		if hasattr(self, "transform") and self.transform is not None:
			body += [repr(self.transform)]
		lines = [head]  # + [" " * self._repr_indent + line for line in body]
		return "\n".join(lines)
	
	def __del__(self):
		self.close()
	
	@classmethod
	@property
	def tasks(cls) -> list[str]:
		return cls._tasks
	
	@classmethod
	@property
	def splits(cls) -> list[str]:
		return cls._splits
	
	@property
	def root(self) -> core.Path:
		return self._root
	
	@property
	def split(self) -> str:
		return self._split
	
	@property
	def split_str(self) -> str:
		return self._split.value
	
	def _init_split(self, split: Split):
		split = Split[split] if isinstance(split, str) else split
		if split in self._splits:
			self._split = split
		else:
			raise ValueError(f":param:`split` must be one of {self._splits}, but got {split}.")
	
	@property
	def classlabels(self) -> ClassLabels | None:
		return self._classlabels
	
	@property
	def transform(self) -> Any:
		return self._transform
	
	@property
	def index(self) -> int:
		return self._index
	
	@property
	def disable_pbar(self) -> bool:
		return not self.verbose
		
	@abstractmethod
	def reset(self):
		"""Resets and starts over."""
		pass
	
	@abstractmethod
	def close(self):
		"""Stops and releases."""
		pass


class UnlabeledDataset(Dataset, ABC):
	"""The base class for all datasets that represent an unlabeled collection of
	data samples.
	
	See Also: :class:`Dataset`.
	"""
	pass


class LabeledDataset(Dataset, ABC):
	"""The base class for datasets that represent an unlabeled collection of
	data samples.
	
	Attributes:
		_has_test_label: If ``True``, the test set has ground-truth labels.
			Default: ``False``.
		
	Args:
		root: The root directory where the data is stored.
		split: The data split to use. One of: [``'train'``, ``'val'``,
			``'test'``, ``'predict'``]. Default: ``'train'``.
		classlabels: :class:`ClassLabels` object. Default: ``None``.
		has_test_label: If ``True``, the test set has ground-truth labels.
			Default: ``False``.
		transform: Transformations performed on both the input and target.
		to_tensor: If ``True``, convert input and target to :class:`torch.Tensor`.
			Default: ``False``.
		verbose: Verbosity. Default: ``True``.
	
	See Also: :class:`Dataset`.
	"""
	
	_has_test_label: bool = False
	
	def __init__(
		self,
		root       : core.Path,
		split      : Split          	= Split.TRAIN,
		classlabels: ClassLabels | None = None,
		transform  : Any                = None,
		to_tensor  : bool               = False,
		verbose    : bool               = False,
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
		
	@property
	def has_test_label(self) -> bool:
		return (
			(self._has_test_label and self._split == Split.TEST) or
			self._split in [Split.TRAIN, Split.VAL]
		)

# endregion


# region Image-Only Dataset

# region Unlabeled

class UnlabeledImageDataset(UnlabeledDataset, ABC):
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


class UnlabeledImageInpaintingDataset(UnlabeledDataset, ABC):
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


# region Labeled

class LabeledImageDataset(LabeledDataset, ABC):
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


class LabeledImageInpaintingDataset(LabeledDataset, ABC):
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

# endregion


# region Video-Only Dataset

# region Unlabeled

class UnlabeledVideoDataset(UnlabeledDataset, ABC):
	"""The base class for datasets that represent an unlabeled video. This is
	mainly used for unsupervised learning tasks.
	
	Args:
		root: A data source. It can be a path to a single video file or a
			stream.
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
		split      : Split          	= Split.PREDICT,
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
		self._num_frames = 0
		self._init_video()
	
	def __len__(self) -> int:
		return self._num_frames
	
	@abstractmethod
	def __getitem__(self, item: int) -> tuple[
		torch.Tensor | np.ndarray,
		torch.Tensor | np.ndarray | None,
		dict | None
	]:
		pass
	
	@property
	@abstractmethod
	def fourcc(self) -> str:
		"""Return the 4-character code of codec."""
		pass
	
	@property
	@abstractmethod
	def fps(self) -> int:
		"""Return the frame rate."""
		pass
	
	@property
	@abstractmethod
	def frame_height(self) -> int:
		"""Return the height of the frames in the video stream."""
		pass
	
	@property
	@abstractmethod
	def frame_width(self) -> int:
		"""Return the width of the frames in the video stream."""
		pass
	
	@property
	def is_stream(self) -> bool:
		"""Return ``True`` if it is a video stream, i.e., unknown :attr:`frame_count`. """
		return self._num_frames == -1
	
	@property
	def shape(self) -> list[int]:
		"""Return the shape of the frames in the video stream in
		:math:`[H, W, C]` format.
		"""
		return [self.frame_height, self.frame_width, 3]
	
	@property
	def image_size(self) -> list[int]:
		"""Return the shape of the frames in the video stream in
		:math:`[H, W]` format.
		"""
		return [self.frame_height, self.frame_width]
	
	@property
	def imgsz(self) -> list[int]:
		"""Return the shape of the frames in the video stream in
		:math:`[H, W]` format.
		"""
		return self.image_size
	
	@abstractmethod
	def _init_video(self):
		"""Initialize the video capture object."""
		pass
	
	@abstractmethod
	def reset(self):
		"""Reset and start over."""
		pass
	
	@abstractmethod
	def close(self):
		"""Stop and release."""
		pass


class VideoLoaderCV(UnlabeledVideoDataset):
	"""A video loader that retrieves and loads frame(s) from a video or a stream
	using :mod:`cv2`.
	
	See Also: :class:`UnlabeledVideoDataset`.
	"""
	
	def __init__(
		self,
		root       : core.Path,
		split      : Split      	    = Split.TRAIN,
		classlabels: ClassLabels | None = None,
		transform  : A.Compose   | None = None,
		to_tensor  : bool               = False,
		cache_data : bool               = False,
		verbose    : bool               = True,
		*args, **kwargs
	):
		self._video_capture = None
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
	
	def __getitem__(self, item: int) -> tuple[
		torch.Tensor | np.ndarray,
		torch.Tensor | np.ndarray | None,
		dict | None
	]:
		pass
	
	def __next__(self) -> tuple[
		torch.Tensor | np.ndarray,
		torch.Tensor | np.ndarray | None,
		dict | None
	]:
		if not self.is_stream and self.index >= self._num_frames:
			self.close()
			raise StopIteration
		else:
			# Read the next frame
			if isinstance(self._video_capture, cv2.VideoCapture):
				ret_val, frame = self._video_capture.read()
			else:
				raise RuntimeError(f":attr`_video_capture` has not been initialized.")
			if frame is not None:
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				frame = FrameLabel(index=self.index, path=self.root, frame=frame)
			self._index += 1
			
			# Get data
			image = frame.data
			meta  = frame.meta
			if self.transform is not None:
				transformed = self.transform(image=image)
				image	    = transformed["image"]
			if self.to_tensor:
				image = core.to_image_tensor(input=image, keepdim=False, normalize=True)
			return image, None, meta
			
	@property
	def format(self):  # Flag=8
		"""Return the format of the Mat objects (see Mat::type()) returned by
		VideoCapture::retrieve(). Set value -1 to fetch undecoded RAW video
		streams (as Mat 8UC1).
		"""
		return self._video_capture.get(cv2.CAP_PROP_FORMAT)
	
	@property
	def fourcc(self) -> str:  # Flag=6
		"""Return the 4-character code of codec."""
		return str(self._video_capture.get(cv2.CAP_PROP_FOURCC))
	
	@property
	def fps(self) -> int:  # Flag=5
		"""Return the frame rate."""
		return int(self._video_capture.get(cv2.CAP_PROP_FPS))
	
	@property
	def frame_count(self) -> int:  # Flag=7
		"""Return the number of frames in the video file."""
		if isinstance(self._video_capture, cv2.VideoCapture):
			return int(self._video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
		elif isinstance(self._video_capture, list):
			return len(self._video_capture)
		else:
			return -1
	
	@property
	def frame_height(self) -> int:  # Flag=4
		"""Return the height of the frames in the video stream."""
		return int(self._video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	
	@property
	def frame_width(self) -> int:  # Flag=3
		"""Return the width of the frames in the video stream."""
		return int(self._video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
	
	@property
	def mode(self):  # Flag=10
		"""Return the backend-specific value indicating the current capture mode."""
		return self._video_capture.get(cv2.CAP_PROP_MODE)
	
	@property
	def pos_avi_ratio(self) -> int:  # Flag=2
		"""Return the relative position of the video file: ``0``=start of the
		film, ``1``=end of the film.
		"""
		return int(self._video_capture.get(cv2.CAP_PROP_POS_AVI_RATIO))
	
	@property
	def pos_msec(self) -> int:  # Flag=0
		"""Return the current position of the video file in milliseconds."""
		return int(self._video_capture.get(cv2.CAP_PROP_POS_MSEC))
	
	@property
	def pos_frames(self) -> int:  # Flag=1
		"""Return the 0-based index of the frame to be decoded/captured next."""
		return int(self._video_capture.get(cv2.CAP_PROP_POS_FRAMES))
	
	def _init_video(self):
		root = core.Path(self.root)
		if root.is_video_file():
			self._video_capture = cv2.VideoCapture(str(root), cv2.CAP_FFMPEG)
			num_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
		elif root.is_video_stream():
			self.video_capture = cv2.VideoCapture(str(root), cv2.CAP_FFMPEG)  # stream
			num_frames = -1
		else:
			raise IOError(f"Error when reading input stream or video file!")
		
		if self._num_frames == 0:
			self._num_frames = num_frames
		
	def reset(self):
		"""Reset and start over."""
		self._index = 0
		if isinstance(self._video_capture, cv2.VideoCapture):
			self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.index)
	
	def close(self):
		"""Stop and release the current attr:`_video_capture` object."""
		if isinstance(self._video_capture, cv2.VideoCapture):
			self._video_capture.release()


class VideoLoaderFFmpeg(UnlabeledVideoDataset):
	"""A video loader that retrieves and loads frame(s) from a video or a stream
	using :mod:`ffmpeg`.
	
	References:
		`<https://github.com/kkroening/ffmpeg-python/tree/master/examples>`__
	
	See Also: :class:`UnlabeledVideoDataset`.
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
		self._ffmpeg_cmd     = None
		self._ffmpeg_process = None
		self._ffmpeg_kwargs  = kwargs
		self._video_info     = None
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
	
	def __getitem__(self, item: int) -> tuple[
		torch.Tensor | np.ndarray,
		torch.Tensor | np.ndarray | None,
		dict | None
	]:
		pass
	
	def __next__(self) -> tuple[
		torch.Tensor | np.ndarray,
		torch.Tensor | np.ndarray | None,
		dict | None
	]:
		if not self.is_stream and self.index >= self.frame_count:
			self.close()
			raise StopIteration
		else:
			# Read the next frame
			if self._ffmpeg_process:
				frame = core.read_video_ffmpeg(
					process = self._ffmpeg_process,
					width   = self.frame_width,
					height  = self.frame_height
				)  # Already in RGB
			else:
				raise RuntimeError(f":attr`_video_capture` has not been initialized.")
			if frame is not None:
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				frame = FrameLabel(index=self.index, path=self.root, frame=frame)
			self._index += 1
			
			# Get data
			image = frame.data
			meta  = frame.meta
			if self.transform is not None:
				transformed = self.transform(image=image)
				image	    = transformed["image"]
			if self.to_tensor:
				image = core.to_image_tensor(input=image, keepdim=False, normalize=True)
			
			return image, None, meta
	
	@property
	def fourcc(self) -> str:
		"""Return the 4-character code of codec."""
		return self._video_info["codec_name"]
	
	@property
	def fps(self) -> int:
		"""Return the frame rate."""
		return int(self._video_info["avg_frame_rate"].split("/")[0])
	
	@property
	def frame_count(self) -> int:
		"""Return the number of frames in the video file."""
		if self.root.is_video_file():
			return int(self._video_info["nb_frames"])
		else:
			return -1
	
	@property
	def frame_width(self) -> int:
		"""Return the width of the frames in the video stream."""
		return int(self._video_info["width"])
	
	@property
	def frame_height(self) -> int:
		"""Return the height of the frames in the video stream."""
		return int(self._video_info["height"])
	
	def _init_video(self):
		"""Initialize ``ffmpeg`` cmd."""
		source = str(self.root)
		probe  = ffmpeg.probe(source, **self._ffmpeg_kwargs)
		self._video_info = next(
			s for s in probe["streams"] if s["codec_type"] == "video"
		)
		if self.verbose:
			self._ffmpeg_cmd = (
				ffmpeg
				.input(source, **self._ffmpeg_kwargs)
				.output("pipe:", format="rawvideo", pix_fmt="rgb24")
				.compile()
			)
		else:
			self._ffmpeg_cmd = (
				ffmpeg
				.input(source, **self._ffmpeg_kwargs)
				.output("pipe:", format="rawvideo", pix_fmt="rgb24")
				.global_args("-loglevel", "quiet")
				.compile()
			)
	
	def reset(self):
		"""Reset and start over."""
		self.close()
		self._index = 0
		if self._ffmpeg_cmd:
			self._ffmpeg_process = subprocess.Popen(
				self._ffmpeg_cmd,
				stdout  = subprocess.PIPE,
				bufsize = 10 ** 8
			)
	
	def close(self):
		"""Stop and release the current :attr:`ffmpeg_process`."""
		if self._ffmpeg_process and self._ffmpeg_process.poll() is not None:
			# os.killpg(os.getpgid(self.ffmpeg_process.pid), signal.SIGTERM)
			self._ffmpeg_process.terminate()
			self._ffmpeg_process.wait()
			self._ffmpeg_process = None

# endregion

# endregion


# region Classification Dataset

class ImageClassificationDataset(LabeledImageDataset, ABC):
	"""The base class for labeled datasets consisting of images, and their
	associated classification labels stored in a simple JSON format.
	
	See Also: :class:`LabeledImageDataset`.
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
		self._labels: list[ClassificationLabel] = []
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
		torch.Tensor | np.ndarray | list | None,
		dict | None
	]:
		image = self._images[index].data
		label = self._labels[index].data if self.has_test_label else None
		meta  = self._images[index].meta
		
		if self.transform is not None:
			if self.has_test_label:
				transformed = self.transform(image=image, mask=label)
				label       = transformed["mask"]
			else:
				transformed = self.transform(image=image)
				image       = transformed["image"]
		
		if self.to_tensor:
			if self.has_test_label:
				image = core.to_image_tensor(input=image, keepdim=False, normalize=True)
				label = torch.Tensor(label)
			else:
				image = core.to_image_tensor(input=image, keepdim=False, normalize=True)
		
		return image, label, meta
		
	@staticmethod
	def collate_fn(batch) -> tuple[
		torch.Tensor | np.ndarray,
		torch.Tensor | np.ndarray | None,
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
		
		if all(isinstance(t, torch.Tensor) for t in target):
			target = torch.cat(target, dim=0)
		elif all(isinstance(t, np.ndarray) for t in target):
			target = np.concatenate(target, axis=0)
		else:
			target = None
		return input, target, meta


class ImageClassificationDirectoryTree(ImageClassificationDataset):
	"""A directory tree whose sub-folders define an image classification
	dataset.
	
	See Also: :class:`ImageClassificationDataset`.
	"""
	
	def _get_images(self):
		pass
	
	def _get_labels(self):
		pass
	
	def _filter(self):
		pass

# endregion


# region Detection Dataset

class ImageDetectionDataset(LabeledImageDataset, ABC):
	"""The base class for datasets that represent a collection of images, and a
	set of associated detections.
	
	See Also: :class:`LabeledImageDataset`.
	
	Args:
		bbox_format: A bounding box format specified in :class:`BBoxFormat`.
	"""
	
	def __init__(
		self,
		root       : core.Path,
		split      : Split          	= Split.TRAIN,
		bbox_format: BBoxFormat         = BBoxFormat.XYXY,
		classlabels: ClassLabels | None = None,
		transform  : A.Compose   | None = None,
		to_tensor  : bool               = False,
		cache_data : bool               = False,
		verbose    : bool               = True,
		*args, **kwargs
	):
		self._bbox_format = BBoxFormat.from_value(value=bbox_format)
		if isinstance(transform, A.Compose):
			if "bboxes" not in transform.processors:
				transform = A.Compose(
					transforms  = transform.transforms,
					bbox_params = A.BboxParams(format=str(self._bbox_format.value))
				)
		self._labels: list[DetectionsLabel] = []
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
		image  = self._images[index].data
		bboxes = self._labels[index].data if self.has_test_label else None
		meta   = self._images[index].meta
		
		if self.transform is not None:
			if self.has_test_label:
				transformed = self.transform(image=image, bboxes=bboxes)
				image       = transformed["image"]
				bboxes      = transformed["bboxes"]
			else:
				transformed = self.transform(image=image, bboxes=bboxes)
				image       = transformed["image"]
		
		if self.to_tensor:
			if self.has_test_label:
				image  = core.to_image_tensor(input=image, keepdim=False, normalize=True)
				bboxes = torch.Tensor(bboxes)
			else:
				image  = core.to_image_tensor(input=image, keepdim=False, normalize=True)
		
		return image, bboxes, meta
	
	def _filter(self):
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
			raise ValueError(
				f"input's number of dimensions must be between ``2`` and ``4``."
			)
		
		if any(t is None for t in target):
			target = None
		else:
			for i, l in enumerate(target):
				l[:, -1] = i  # add target image index for build_targets()
		return input, target, meta


class DetectionDatasetCOCO(ImageDetectionDataset, ABC):
	"""The base class for labeled datasets consisting of images, and their
	associated object detections saved in `COCO Object Detection Format
	<https://cocodataset.org/#format-data>`.
	
	See Also: :class:`ImageDetectionDataset`.
	"""
	
	def __init__(
		self,
		root       : core.Path,
		split      : Split          	= Split.TRAIN,
		bbox_format: BBoxFormat         = BBoxFormat.XYXY,
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
			bbox_format = bbox_format,
			classlabels = classlabels,
			transform   = transform,
			to_tensor   = to_tensor,
			cache_data  = cache_data,
			verbose     = verbose,
			*args, **kwargs
		)
	
	def _get_labels(self):
		json_file = self.annotation_file()
		if not json_file.is_json_file():
			raise ValueError(
				f":param:`json_file` must be a valid path to a ``.json`` file, "
				f"but got {json_file}."
			)
		json_data = core.read_from_file(json_file)
		if not isinstance(json_data, dict):
			raise TypeError(f":param:`json_data` must be a :class:`dict`, but got {type(json_data)}.")
		
		info	    = json_data.get("info", 	   None)
		licenses    = json_data.get("licenses",    None)
		categories  = json_data.get("categories",  None)
		images	    = json_data.get("images",	   None)
		annotations = json_data.get("annotations", None)
		
		for img in images:
			id       = img.get("id",        uuid.uuid4().int)
			filename = img.get("file_name", "")
			height   = img.get("height",     0)
			width    = img.get("width",      0)
			index    = -1
			for idx, im in enumerate(self._images):
				if im.name == filename:
					index = idx
					break
			self._images[index].id            = id
			self._images[index].coco_url      = img.get("coco_url", "")
			self._images[index].flickr_url    = img.get("flickr_url", "")
			self._images[index].license       = img.get("license", 0)
			self._images[index].date_captured = img.get("date_captured", "")
			self._images[index].shape         = (height, width, 3)
		
		for ann in annotations:
			id          = ann.get("id"         , uuid.uuid4().int)
			image_id    = ann.get("image_id"   , None)
			bbox        = ann.get("bbox"       , None)
			category_id = ann.get("category_id", -1)
			area        = ann.get("area"       , 0)
			iscrowd     = ann.get("iscrowd"    , False)
	
	@abstractmethod
	def annotation_file(self) -> core.Path:
		pass
	
	def _filter(self):
		pass


class DetectionDatasetVOC(ImageDetectionDataset, ABC):
	"""The base class for labeled datasets consisting of images, and their
	associated object detections saved in `PASCAL VOC format
	<https://host.robots.ox.ac.uk/pascal/VOC>`.
	
	See Also: :class:`ImageDetectionDataset`.
	"""
	
	def __init__(
		self,
		root       : core.Path,
		split      : Split        	    = Split.TRAIN,
		bbox_format: BBoxFormat         = BBoxFormat.XYXY,
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
			bbox_format = bbox_format,
			classlabels = classlabels,
			transform   = transform,
			to_tensor   = to_tensor,
			cache_data  = cache_data,
			verbose     = verbose,
			*args, **kwargs
		)
	
	def _get_labels(self):
		files = self.annotation_files()
		
		if not len(self._images) > 0:
			raise RuntimeError(f"No images in dataset.")
		if not len(self._images) == len(files):
			raise RuntimeError(
				f"Number of images and files must be the same, but got "
				f"{len(self._images)} and {len(files)}."
			)
		
		self.labels: list[VOCDetectionsLabel] = []
		with core.get_progress_bar() as pbar:
			for f in pbar.track(
				files,
				description=f"Listing {self.__class__.__name__} {self.split_str} labels"
			):
				self.labels.append(
					VOCDetectionsLabel.from_file(
						path        = f,
						classlabels = self._classlabels
					)
				)
	
	@abstractmethod
	def annotation_files(self) -> list[core.Path]:
		pass
	
	def _filter(self):
		pass


class DetectionDatasetYOLO(ImageDetectionDataset, ABC):
	"""The base class for labeled datasets consisting of images, and their
	associated object detections saved in `YOLO format`.
	
	See Also: :class:`ImageDetectionDataset`.
	"""
	
	def __init__(
		self,
		root       : core.Path,
		split      : Split	            = Split.TRAIN,
		bbox_format: BBoxFormat         = BBoxFormat.XYXY,
		classlabels: ClassLabels | None = None,
		transform  : A.Compose   | None = None,
		to_tensor  : bool               = False,
		cache_data : bool               = False,
		verbose    : bool               = True,
		*args         , **kwargs
	):
		super().__init__(
			root        = root,
			split       = split,
			bbox_format = bbox_format,
			classlabels = classlabels,
			transform   = transform,
			to_tensor   = to_tensor,
			cache_data  = cache_data,
			verbose     = verbose,
			*args, **kwargs
		)
	
	def _get_labels(self):
		files = self.annotation_files()
		
		if not len(self._images) > 0:
			raise RuntimeError(f"No images in dataset.")
		if not len(self._images) == len(files):
			raise RuntimeError(
				f"Number of images and files must be the same, but got "
				f"{len(self._images)} and {len(files)}."
			)
		
		self._labels: list[YOLODetectionsLabel] = []
		with core.get_progress_bar() as pbar:
			for f in pbar.track(
				files,
				description=f"Listing {self.__class__.__name__} {self.split_str} labels"
			):
				self._labels.append(YOLODetectionsLabel.from_file(path=f))
	
	@abstractmethod
	def annotation_files(self) -> list[core.Path]:
		pass
	
	def _filter(self):
		pass

# endregion


# region Segmentation

class ImageSegmentationDataset(LabeledImageDataset, ABC):
	"""The base class for datasets that represent a collection of images and a
	set of associated semantic segmentations.
	
	See Also: :class:`LabeledImageDataset`.
	"""
	
	def __init__(
		self,
		root       : core.Path,
		split      : Split	            = Split.TRAIN,
		classlabels: ClassLabels | None = None,
		transform  : A.Compose   | None = None,
		to_tensor  : bool               = False,
		cache_data : bool               = False,
		verbose    : bool               = True,
		*args, **kwargs
	):
		self._labels: list[SegmentationLabel] = []
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
				transformed = self.transform(image=image, mask=label)
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
		for i, (img, lab) in enumerate(zip(self._images, self.labels)):
			if img.path.is_image_file() and lab.path.is_image_file():
				keep.append(i)
		self._images = [img for i, img in enumerate(self._images) if i in keep]
		self._labels = [lab for i, lab in enumerate(self.labels) if i in keep]
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
			raise ValueError(
				f"input's number of dimensions must be between ``2`` and ``4``."
			)
		
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
