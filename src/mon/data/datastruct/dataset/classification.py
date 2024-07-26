#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements classification datasets."""

from __future__ import annotations

__all__ = [
	"ImageClassificationDataset",
	"ImageClassificationDirectoryTree",
]

from abc import ABC

import albumentations as A
import numpy as np
import torch

from mon import core
from mon.data.datastruct import annotation as anno
from mon.data.datastruct.dataset import image as img
from mon.globals import Split

console             = core.console
ClassLabels         = anno.ClassLabels
ClassificationLabel = anno.ClassificationAnnotation


# region Image Classification Dataset

class ImageClassificationDataset(img.LabeledImageDataset, ABC):
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
