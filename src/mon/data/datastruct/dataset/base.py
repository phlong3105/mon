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
