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
		self.annotations: list[SegmentationAnnotation] = []
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
		
		return {
			"input"   : image,
			"target"  : mask,
			"metadata": meta,
		}
		
	def filter(self):
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
