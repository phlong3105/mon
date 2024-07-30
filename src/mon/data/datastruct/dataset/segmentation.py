#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements segmentation datasets."""

from __future__ import annotations

__all__ = [
	"ImageSegmentationDataset",
]

from abc import ABC

import albumentations as A
import numpy as np
import torch

from mon import core
from mon.data.datastruct import annotation as anno
from mon.data.datastruct.dataset import image as img
from mon.globals import Split

console                = core.console
ClassLabels            = anno.ClassLabels
SegmentationAnnotation = anno.SegmentationAnnotation


# region Image Segmentation Dataset

class ImageSegmentationDataset(img.LabeledImageDataset, ABC):
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
		self.labels: list[SegmentationAnnotation] = []
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
		label = self.labels[index].data if self.has_test_label else None
		meta  = self.images[index].meta
		
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
		
	def filter(self):
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
