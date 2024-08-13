#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements image-only datasets."""

from __future__ import annotations

__all__ = [
	"ImageDataset",
	"ImageLoader",
]

import glob
from abc import ABC, abstractmethod
from typing import Any

import albumentations as A

from mon import core
from mon.data.datastruct import annotation
from mon.data.datastruct.dataset import base
from mon.globals import Split

console             = core.console
ClassLabels         = annotation.ClassLabels
DatapointAttributes = annotation.DatapointAttributes
ImageAnnotation     = annotation.ImageAnnotation


# region Image Dataset

class ImageDataset(base.Dataset, ABC):
	"""The base class for all image-based datasets.
	
	Attributes:
		datapoint_attrs: A :class:`dict` of datapoint attributes with the keys
			are the attribute names and the values are the attribute types.
			Must contain: {``'image'``: :class:`ImageAnnotation`}. Note that to
			comply with :class:`albumentations.Compose`, we will treat the first
			key as the main image attribute.
		
	See Also: :class:`mon.data.datastruct.dataset.base.Dataset`.
	"""
	
	datapoint_attrs = DatapointAttributes({
		"image": ImageAnnotation,
	})
	
	def __getitem__(self, index: int) -> dict:
		"""Returns a dictionary containing the datapoint and metadata at the
		given :param:`index`.
		"""
		# Get datapoint at the given index
		datapoint = self.get_datapoint(index=index)
		meta      = self.get_meta(index=index)
		# Transform
		if self.transform:
			main_attr      = self.main_attribute
			args           = {k: v for k, v in datapoint.items() if v}
			args["image"]  = args.pop(main_attr)
			transformed    = self.transform(**args)
			transformed[main_attr] = transformed.pop("image")
			datapoint     |= transformed
		if self.to_tensor:
			for k, v in datapoint.items():
				to_tensor_fn = self.datapoint_attrs.get_tensor_fn(k)
				if to_tensor_fn and v:
					datapoint[k] = to_tensor_fn(input=v, keepdim=False, normalize=True)
		# Return
		return datapoint | {"meta": meta}
	
	def __len__(self) -> int:
		"""Return the total number of datapoints in the dataset."""
		return len(self.datapoints[self.main_attribute])
		
	def init_transform(self, transform: A.Compose | Any = None):
		super().init_transform(transform=transform)
		# Add additional targets
		if self.transform:
			additional_targets = self.datapoint_attrs.albumentation_target_types()
			additional_targets.pop(self.main_attribute, None)
			additional_targets.pop("meta", None)
			self.transform.add_targets(additional_targets)
	
	def filter_data(self):
		"""Filter unwanted datapoints."""
		pass
	
	def verify_data(self):
		"""Verify dataset."""
		if self.__len__() <= 0:
			raise RuntimeError(f"No datapoints in the dataset.")
		for k, v in self.datapoints.items():
			if k not in self.datapoint_attrs:
				raise RuntimeError(
					f"Attribute ``{k}`` has not been defined in :attr:`datapoint_attrs`. "
					f"If this is not an error, please define the attribute in the class."
				)
			if self.datapoint_attrs[k]:
				if v is None:
					raise RuntimeError(f"No ``{k}`` attributes has been defined.")
				if v and len(v) != self.__len__():
					raise RuntimeError(f"Number of {k} attributes does not match the number of datapoints.")
		if self.verbose:
			console.log(f"Number of {self.split_str} datapoints: {self.__len__()}.")
	
	def reset(self):
		"""Reset and start over."""
		self.index = 0
	
	def close(self):
		"""Stop and release."""
		pass
	
	def get_datapoint(self, index: int) -> dict:
		"""Get a datapoint at the given :param:`index`."""
		datapoint = self.new_datapoint
		for k, v in self.datapoints.items():
			if v and v[index] and hasattr(v[index], "data"):
				datapoint[k] = v[index].data
		return datapoint
	
	def get_meta(self, index: int) -> dict:
		"""Get metadata at the given :param:`index`. By default, we will use
		the first attribute in :attr:`datapoint_attrs` as the main image attribute.
		"""
		return self.datapoints[self.main_attribute][index].meta
		

class ImageLoader(ImageDataset):
	"""A general image loader that retrieves and loads image(s) from a file
	path, file path pattern, or directory.
	
	See Also: :class:`ImageDataset`.
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
		
		images: list[ImageAnnotation] = []
		with core.get_progress_bar() as pbar:
			for path in pbar.track(
				sorted(paths),
				description=f"[bright_yellow]Listing {self.__class__.__name__} {self.split_str} images"
			):
				if path.is_image_file():
					images.append(ImageAnnotation(path=path))
		self.datapoints["image"] = images

# endregion
