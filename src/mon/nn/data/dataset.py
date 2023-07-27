#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for all datasets."""

from __future__ import annotations

__all__ = [
    "ChainDataset", "ConcatDataset", "Dataset", "IterableDataset",
    "LabeledDataset", "Subset", "TensorDataset", "UnlabeledDataset",
    "random_split",
]

from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import dataset
from torch.utils.data.dataset import *

from mon.core import pathlib


# region Dataset

# noinspection PyMethodMayBeStatic
class Dataset(dataset.Dataset, ABC):
    """The base class of all datasets.
    
    See Also: :mod:`dataset.Dataset`.
    
    Args:
        root: The root directory where the data is stored.
        split: The data split to use. One of: ['train', 'val', 'test',
            'predict']. Default: 'train'.
        transform: Transformations performing on both the input and target.
        to_tensor: If True, convert input and target to :class:`torch.Tensor`.
            Default: False.
        verbose: Verbosity. Default: True.
    """
    
    def __init__(
        self,
        root     : pathlib.Path,
        split    : str  = "train",
        transform: Any  = None,
        to_tensor: bool = False,
        verbose  : bool = True,
        *args, **kwargs
    ):
        super().__init__()
        self.root      = pathlib.Path(root)
        self.split     = split
        self.transform = transform
        self.to_tensor = to_tensor
        self.verbose   = verbose
    
    def __iter__(self):
        """Returns an iterator starting at index 0."""
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
    
    @abstractmethod
    def reset(self):
        """Resets and starts over."""
        pass
    
    @abstractmethod
    def close(self):
        """Stops and releases."""
        pass


class UnlabeledDataset(Dataset, ABC):
    """The base class for all datasets that represent an unlabeled collection of
    data samples.
    
    See Also: :class:`Dataset`.
    """
    pass


class LabeledDataset(Dataset, ABC):
    """The base class for datasets that represent an unlabeled collection of
    data samples.
    
    See Also: :class:`Dataset`.
    """
    pass

# endregion
