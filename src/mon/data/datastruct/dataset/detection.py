#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements base classes for all datasets.

For transformation operations, we use
`albumentations <https://albumentations.ai/docs/api_reference/full_reference>`__
"""

from __future__ import annotations

__all__ = [
	"ImageDetectionDataset",
]

from abc import ABC

import albumentations as A
import torch
from albumentations.core import bbox_utils
from mon import core
from mon.data.datastruct import annotation
from mon.data.datastruct.dataset import image as I
from mon.globals import BBoxFormat

console             = core.console
BBoxesLabel         = annotation.BBoxesAnnotation
ClassLabels         = annotation.ClassLabels
DatapointAttributes = annotation.DatapointAttributes
ImageAnnotation     = annotation.ImageAnnotation


# region Detection Dataset

class ImageDetectionDataset(I.ImageDataset, ABC):
	"""The base class for all object detection datasets.
	
	Attributes:
		datapoint_attrs: A :class:`dict` of datapoint attributes with the keys
			are the attribute names and the values are the attribute types.
			Must contain: {``'image'``: :class:`ImageAnnotation`,
			``'bboxes'``: :class:`BBoxesLabel`}. Note that to comply with
			:class:`albumentations.Compose`, we will treat the first key as the
			main image attribute.
			
	Args:
		bbox_format: A bounding box format specified in :class:`BBoxFormat`.
	
	See Also: :class:`mon.data.datastruct.dataset.base.Dataset`.
	"""
	
	datapoint_attrs = DatapointAttributes({
		"image" : ImageAnnotation,
		"bboxes": BBoxesLabel,
	})
	
	def __init__(
		self,
		bbox_format: BBoxFormat = BBoxFormat.XYXY,
		*args, **kwargs
	):
		self.bbox_format = BBoxFormat.from_value(value=bbox_format)
		super().__init__(*args, **kwargs)
	
	def __getitem__(self, index: int) -> dict:
		"""Returns a dictionary containing the datapoint and metadata at the
		given :param:`index`. The dictionary must contain the following keys:
		{'input', 'target', 'meta'}.
		"""
		image  = self.images[index].data
		bboxes = self.annotations[index].data if self.has_annotations else None
		meta   = self.images[index].meta
		
		if self.transform:
			if self.has_annotations:
				transformed = self.transform(image=image, bboxes=bboxes)
				image       = transformed["image"]
				bboxes      = transformed["bboxes"]
			else:
				transformed = self.transform(image=image, bboxes=bboxes)
				image       = transformed["image"]
		
		if self.to_tensor:
			if self.has_annotations:
				image  = core.to_image_tensor(image=image, keepdim=False, normalize=True)
				bboxes = torch.Tensor(bboxes)
			else:
				image  = core.to_image_tensor(image=image, keepdim=False, normalize=True)
		
		return {
			"input" : image,
			"target": None,
			"meta"  : meta,
		}
	
	def init_transform(self, transform: A.Compose | Any = None):
		super().init_transform(transform=transform)
		# Add additional targets
		if isinstance(self.transform, A.Compose):
			if "bboxes" not in self.transform.processors:
				self.transform.processors["bboxes"] = A.core.bbox_utils.BboxProcessor(
					A.BboxParams(
						format=str(self.bbox_format.value),
					)
				)
	
	@classmethod
	def collate_fn(cls, batch: list[dict]) -> dict:
		"""Collate function used to fused input items together when using
		:attr:`batch_size` > 1. This is used in :class:`torch.utils.data.DataLoader` wrapper.

		Args:
			batch: A :class:`list` of :class:`dict`.
		"""
		zipped = super().collate_fn(batch=batch)
		bboxes = zipped["bboxes"]
		if bboxes is not None:
			for i, l in enumerate(bboxes):
				l[:, -1] = i  # add target image index for build_targets()
			zipped["bboxes"] = bboxes
		return zipped

# endregion
