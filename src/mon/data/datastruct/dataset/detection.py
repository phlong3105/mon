#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements base classes for all datasets.

For transformation operations, we use
`albumentations <https://albumentations.ai/docs/api_reference/full_reference>`__
"""

from __future__ import annotations

__all__ = [
	"DetectionDatasetCOCO",
	"DetectionDatasetVOC",
	"DetectionDatasetYOLO",
	"ImageDetectionDataset",
]

import uuid
from abc import ABC, abstractmethod

import albumentations as A
import numpy as np
import torch

from mon import core
from mon.data.datastruct import annotation as anno
from mon.data.datastruct.dataset import image as img
from mon.globals import BBoxFormat, Split

console	= core.console

ClassLabels     = anno.ClassLabels
BBoxesLabel     = anno.BBoxesAnnotation
BBoxesLabelVOC  = anno.BBoxesAnnotationVOC
BBoxesLabelYOLO = anno.BBoxesAnnotationYOLO


# region Detection Dataset

class ImageDetectionDataset(img.LabeledImageDataset, ABC):
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
		self._labels: list[BBoxesLabel] = []
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
		annos       = json_data.get("annotations", None)
		
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
		
		for ann in annos:
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
		
		self.labels: list[BBoxesLabelVOC] = []
		with core.get_progress_bar() as pbar:
			for f in pbar.track(
				files,
				description=f"Listing {self.__class__.__name__} {self.split_str} labels"
			):
				self.labels.append(
					BBoxesLabelVOC.from_file(
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
		
		self._labels: list[BBoxesLabelYOLO] = []
		with core.get_progress_bar() as pbar:
			for f in pbar.track(
				files,
				description=f"Listing {self.__class__.__name__} {self.split_str} labels"
			):
				self._labels.append(BBoxesLabelYOLO.from_file(path=f))
	
	@abstractmethod
	def annotation_files(self) -> list[core.Path]:
		pass
	
	def _filter(self):
		pass

# endregion
