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
	"LabeledDataset",
	"Subset",
	"TensorDataset",
	"UnlabeledDataset",
	"random_split",
]

from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import dataset
from torch.utils.data.dataset import *

from mon import core
from mon.data.datastruct import annotation as anno
from mon.globals import Split, Task

console	    = core.console
ClassLabels = anno.ClassLabels


# region Base

class Dataset(dataset.Dataset, ABC):
	"""The base class of all datasets.
	
	Attributes:
		tasks: A list of tasks that the dataset supports.
		splits: A list of splits that the dataset supports.
	
	Args:
		root: The root directory where the data is stored.
		split: The data split to use. Default: ``'Split.TRAIN'``.
		classlabels: :class:`ClassLabels` object. Default: ``None``.
		transform: Transformations performed on both the input and target.
		to_tensor: If ``True``, convert input and target to :class:`torch.Tensor`.
			Default: ``False``.
		verbose: Verbosity. Default: ``True``.
	
	See Also: :mod:`dataset.Dataset`.
	"""
	
	tasks : list[Task]  = []
	splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST, Split.PREDICT]
	
	def __init__(
		self,
		root       : core.Path,
		split      : Split              = Split.TRAIN,
		classlabels: ClassLabels | None = None,
		transform  : Any                = None,
		to_tensor  : bool               = False,
		verbose    : bool               = False,
		*args, **kwargs
	):
		super().__init__()
		self.root        = core.Path(root)
		self.split 	     = split
		self.classlabels = ClassLabels.from_value(classlabels)
		self.transform   = transform
		self.to_tensor   = to_tensor
		self.verbose     = verbose
		self.index		 = 0  # Use with :meth:`__iter__` and :meth`__next__`
		
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

# endregion


# region Unlabeled Datasets

class UnlabeledDataset(Dataset, ABC):
	"""The base class for all datasets that represent an unlabeled collection of
	data samples.
	
	See Also: :class:`Dataset`.
	"""
	pass

# endregion


# region Labeled Datasets

class LabeledDataset(Dataset, ABC):
	"""The base class for datasets that represent an unlabeled collection of
	data samples.
	
	Attributes:
		has_test_annotations: If ``True``, the test set has ground-truth labels.
			Default: ``False``.
		
	Args:
		root: The root directory where the data is stored.
		split: The data split to use. Default: ``'Split.TRAIN'``.
		classlabels: :class:`ClassLabels` object. Default: ``None``.
		has_test_label: If ``True``, the test set has ground-truth labels.
			Default: ``False``.
		transform: Transformations performed on both the input and target.
		to_tensor: If ``True``, convert input and target to :class:`torch.Tensor`.
			Default: ``False``.
		verbose: Verbosity. Default: ``True``.
	
	See Also: :class:`Dataset`.
	"""
	
	has_test_annotations: bool = False
	
	def __init__(
		self,
		root       : core.Path,
		split      : Split = Split.TRAIN,
		classlabels: Any   = None,
		transform  : Any   = None,
		to_tensor  : bool  = False,
		verbose    : bool  = False,
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
	def has_annotations(self) -> bool:
		"""Returns ``True`` if the images has accompany annotations, otherwise ``False``."""
		return (
			(self.has_test_annotations and self.split in [Split.TEST, Split.PREDICT])
			or (self.split in [Split.TRAIN, Split.VAL])
		)
		
# endregion
