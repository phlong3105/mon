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
	"IterableDataset",
	"Subset",
	"TensorDataset",
	"random_split",
]

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import dataset
from torch.utils.data.dataset import *

from mon import core
from mon.data.datastruct import annotation as anno
from mon.data.transform import albumentation as A
from mon.globals import Split, Task

console	    = core.console
ClassLabels = anno.ClassLabels


# region Base

class Dataset(dataset.Dataset, ABC):
	"""The base class of all datasets.
	
	Attributes:
		tasks: A :class:`list` of tasks that the dataset supports.
		splits: A :class:`list` of splits that the dataset supports.
		has_test_targets: If ``True``, the test set has ground-truth labels.
			Default: ``False``.
		datapoint_attrs: A :class:`dict` of datapoint attributes with the keys
			are the attribute names and the values are the attribute types. By
			default, it must contain ``{'input', 'target'}``.
	
	Args:
		root: The root directory where the data is stored.
		split: The data split to use. Default: ``'Split.TRAIN'``.
		classlabels: :class:`ClassLabels` object. Default: ``None``.
		transform: Transformations performed on both the input and target.
		to_tensor: If ``True``, convert input and target to :class:`torch.Tensor`.
			Default: ``False``.
		cache_data: If ``True``, cache data to disk for faster loading next
			time. Default: ``False``.
		verbose: Verbosity. Default: ``True``.
	
	See Also: :mod:`dataset.Dataset`.
	"""
	
	tasks : list[Task]  = []
	splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST, Split.PREDICT]
	has_test_annotations: bool = False
	datapoint_attrs     : dict = {
		"input" : Any,
		"target": Any,
	}
	
	def __init__(
		self,
		root       : core.Path,
		split      : Split              = Split.TRAIN,
		classlabels: ClassLabels | None = None,
		transform  : A.Compose   | Any  = None,
		to_tensor  : bool               = False,
		cache_data : bool               = False,
		verbose    : bool               = False,
		*args, **kwargs
	):
		super().__init__(*args, **kwargs)
		self.root            = core.Path(root)
		self.split 	         = split
		self.classlabels     = ClassLabels.from_value(classlabels)
		self.transform       = transform
		self.to_tensor       = to_tensor
		self.verbose         = verbose
		self.index		     = 0  # Use with :meth:`__iter__` and :meth`__next__`
		self.datapoint_attrs = {"input": Any, "target": Any} if not self.datapoint_attrs else self.datapoint_attrs
		self.datapoints      = {k: list[v]() for k, v in self.datapoint_attrs.items()}
		
		# Get image from disk or cache
		cache_file = self.root / f"{self.split_str}.cache"
		if cache_data and cache_file.is_cache_file():
			self.load_cache(path=cache_file)
		else:
			self.get_data()
		
		# Filter and verify data
		self.filter()
		self.verify()
		
		# Cache data
		if cache_data:
			self.cache_data(path=cache_file)
		else:
			core.delete_cache(cache_file)
		
	def __iter__(self):
		"""Returns an iterator starting at the index ``0``."""
		self.reset()
		return self
	
	def __len__(self) -> int:
		"""Return the total number of datapoints in the dataset."""
		return len(self.datapoints["input"])
	
	@abstractmethod
	def __getitem__(self, index: int) -> dict:
		"""Returns a dictionary containing the datapoint and metadata at the
		given :param:`index`. The dictionary must contain the following keys:
		{'input', 'target', 'meta'}.
		"""
		pass
	
	def __next__(self) -> Any:
		"""Returns the next datapoint and metadata when using :meth:`__iter__`."""
		if self.index >= self.__len__():
			raise StopIteration
		else:
			result      = self.__getitem__(self.index)
			self.index += 1
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
	
	@property
	def split(self) -> Split:
		return self._split
	
	@split.setter
	def split(self, split: Split):
		split = Split[split] if isinstance(split, str) else split
		if split in self.splits:
			self._split = split
		else:
			raise ValueError(f":param:`split` must be one of {self.splits}, but got {split}.")
	
	@property
	def split_str(self) -> str:
		return self.split.value
	
	@property
	def has_annotations(self) -> bool:
		"""Returns ``True`` if the images has accompany annotations, otherwise ``False``."""
		return (
			(self.has_test_annotations and self.split in [Split.TEST, Split.PREDICT])
			or (self.split in [Split.TRAIN, Split.VAL])
		)
	
	@property
	def hash(self) -> int:
		"""Return the total hash value of all the files (if it has one).
		Hash values are integers (in bytes) of all files.
		"""
		# return sum(i.meta.get("hash", 0) for i in self.images) if self.images else 0
		sum = 0
		for k, v in self.datapoints.items():
			if isinstance(v, list):
				for a in v:
					if a is not None and hasattr(a, "meta"):
						sum += a.meta.get("hash", 0)
		return sum
	
	@property
	def disable_pbar(self) -> bool:
		return not self.verbose
	
	@abstractmethod
	def get_data(self):
		"""Get datapoints."""
		pass
	
	def cache_data(self, path: core.Path):
		"""Cache data to :param:`path`."""
		hash_ = 0
		if path.is_cache_file():
			cache = torch.load(path)
			hash_ = cache.get("hash", 0)
		
		if self.hash != hash_:
			cache = self.datapoints | {"hash": self.hash}
			torch.save(cache, str(path))
			if self.verbose:
				console.log(f"Cached data to: {path}")
	
	def load_cache(self, path: core.Path):
		"""Load cache data from :param:`path`."""
		self.datapoints = torch.load(path)
		self.datapoints.pop("hash", None)
	
	def filter(self):
		"""Filter unwanted samples."""
		pass
	
	def verify(self):
		"""Verify and check data."""
		if not self.__len__() > 0:
			raise RuntimeError(f"No datapoints in dataset.")
		for k, v in self.datapoints.items():
			if not len(v) == self.__len__():
				raise RuntimeError(f"Number of {k} attributes does not match the number of datapoints.")
		if self.verbose:
			console.log(f"Number of {self.split_str} datapoints: {self.__len__()}.")
	
	@abstractmethod
	def reset(self):
		"""Resets and starts over."""
		pass
	
	@abstractmethod
	def close(self):
		"""Stops and releases."""
		pass

# endregion
