#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Dataset.

This module implements base classes for all datasets.
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

from mon.core import pathlib
from mon.core.data import annotation
from mon.core.rich import console
from mon.core.transform import albumentation as A
from mon.globals import Split, Task

ClassLabels         = annotation.ClassLabels
DatapointAttributes = annotation.DatapointAttributes


# region Base

class Dataset(dataset.Dataset, ABC):
    """The base class of all datasets.
    
    Attributes:
        tasks: A :obj:`list` of tasks that the dataset supports.
        splits: A :obj:`list` of splits that the dataset supports.
        has_test_annotations: If ``True``, the test set has ground-truth labels.
            Default: ``False``.
        datapoint_attrs: A :obj:`dict` of datapoint attributes with the keys
            are the attribute names and the values are the attribute types.
    
    Args:
        root: The root directory where the data is stored.
        split: The data split to use. Default: ``'Split.TRAIN'``.
        classlabels: :obj:`ClassLabels` object. Default: ``None``.
        transform: Transformations performed on both the input and target. We use
            `albumentations <https://albumentations.ai/docs/api_reference/full_reference>`__
        to_tensor: If ``True``, convert input and target to :obj:`torch.Tensor`.
            Default: ``False``.
        cache_data: If ``True``, cache data to disk for faster loading next
            time. Default: ``False``.
        verbose: Verbosity. Default: ``True``.
    
    """
    
    tasks : list[Task]  = []
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST, Split.PREDICT]
    datapoint_attrs     = DatapointAttributes({})
    has_test_annotations: bool = False

    def __init__(
        self,
        root       : pathlib.Path,
        split      : Split           = Split.TRAIN,
        classlabels: ClassLabels     = None,
        transform  : A.Compose | Any = None,
        to_tensor  : bool            = False,
        cache_data : bool            = False,
        verbose    : bool            = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.root        = pathlib.Path(root)
        self.split 	     = split
        self.classlabels = ClassLabels.from_value(classlabels)
        self.transform   = transform
        self.to_tensor   = to_tensor
        self.verbose     = verbose
        self.index		 = 0  # Use with :obj:`__iter__` and :meth`__next__`
        self.datapoints  = {}
        self.init_transform()
        self.init_datapoints()
        
        # Get image from disk or cache
        cache_file = self.root / f"{self.split_str}.cache"
        if cache_data and cache_file.is_cache_file():
            self.load_cache(path=cache_file)
        else:
            self.get_data()
        
        # Filter and verify data
        self.filter_data()
        self.verify_data()
        
        # Cache data
        if cache_data:
            self.cache_data(path=cache_file)
        else:
            pathlib.delete_cache(cache_file)
    
    # region Magic Methods
    
    def __del__(self):
        self.close()
    
    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        """Returns a dictionary containing the datapoint and metadata at the
        given :obj:`index`.
        """
        pass
    
    def __iter__(self):
        """Returns an iterator starting at the index ``0``."""
        self.reset()
        return self
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of datapoints in the dataset."""
        pass
    
    def __next__(self) -> dict:
        """Returns the next datapoint and metadata when using :obj:`__iter__`."""
        if self.index >= self.__len__():
            raise StopIteration
        else:
            result      = self.__getitem__(self.index)
            self.index += 1
            return result
    
    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root:
            body.append(f"Root location: {self.root}")
        if hasattr(self, "transform") and self.transform:
            body += [repr(self.transform)]
        lines = [head]  # + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)
    
    # endregion
    
    # region Properties
    
    @property
    def disable_pbar(self) -> bool:
        return not self.verbose
    
    @property
    def has_annotations(self) -> bool:
        """Returns ``True`` if the images has accompanied annotations, otherwise
        ``False``.
        """
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
                    if a and hasattr(a, "meta"):
                        sum += a.meta.get("hash", 0)
        return sum
    
    @property
    def main_attribute(self) -> str:
        """Return the main attribute of the dataset as the first key in
        :obj:`datapoint_attrs`.
        """
        return next(iter(self.datapoint_attrs.keys()))
    
    @property
    def new_datapoint(self) -> dict:
        """Return a new datapoint with default values."""
        return {k: None for k in self.datapoint_attrs.keys()}
    
    @property
    def split(self) -> Split:
        return self._split
    
    @split.setter
    def split(self, split: Split):
        split = Split[split] if isinstance(split, str) else split
        if split in self.splits:
            self._split = split
        else:
            raise ValueError(
                f"`split` must be one of {self.splits}, but got {split}."
            )
    
    @property
    def split_str(self) -> str:
        return self.split.value
    
    # endregion
    
    # region Initialization
    
    def init_transform(self, transform: A.Compose | Any = None):
        """Initialize transformation operations."""
        self.transform = transform or self.transform
    
    def init_datapoints(self):
        """Initialize datapoints dictionary."""
        if not self.datapoint_attrs:
            raise ValueError(f"`datapoint_attrs` has no defined attributes.")
        self.datapoints = {k: list[v]() for k, v in self.datapoint_attrs.items()}
    
    @abstractmethod
    def get_data(self):
        """Get datapoints."""
        pass
    
    def cache_data(self, path: pathlib.Path):
        """Cache data to :obj:`path`."""
        hash_ = 0
        if path.is_cache_file():
            cache = torch.load(path)
            hash_ = cache.get("hash", 0)
        
        if self.hash != hash_:
            cache = self.datapoints | {"hash": self.hash}
            torch.save(cache, str(path))
            if self.verbose:
                console.log(f"Cached data to: {path}")
    
    def load_cache(self, path: pathlib.Path):
        """Load cache data from :obj:`path`."""
        self.datapoints = torch.load(path)
        self.datapoints.pop("hash", None)
    
    @abstractmethod
    def filter_data(self):
        """Filter unwanted datapoints."""
        pass
    
    @abstractmethod
    def verify_data(self):
        """Verify dataset."""
        pass
    
    @abstractmethod
    def reset(self):
        """Resets and starts over."""
        pass
    
    @abstractmethod
    def close(self):
        """Stops and releases."""
        pass
    
    # endregion
    
    # region Retrieve Data
    
    @abstractmethod
    def get_datapoint(self, index: int) -> dict:
        """Get a datapoint at the given :obj:`index`."""
        pass
    
    @abstractmethod
    def get_meta(self, index: int) -> dict:
        """Get metadata at the given :obj:`index`."""
        pass
    
    @classmethod
    def collate_fn(cls, batch: list[dict]) -> dict:
        """Collate function used to fused input items together when using
		:obj:`batch_size` > 1. This is used in :obj:`torch.utils.data.DataLoader`
		wrapper.

		Args:
			batch: A :obj:`list` of :obj:`dict`.
		"""
        zipped = {
            k: list(v)
            for k, v in zip(batch[0].keys(), zip(*[b.values() for b in batch]))
        }
        for k, v in zipped.items():
            collate_fn = cls.datapoint_attrs.get_collate_fn(k)
            if collate_fn and v:
                zipped[k] = collate_fn(batch=v)
        return zipped
    
    # endregion
    
# endregion
