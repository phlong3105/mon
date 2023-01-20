#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for all datasets."""

from __future__ import annotations

__all__ = [
    "ChainDataset", "ConcatDataset", "Dataset", "IterableDataset", "Subset",
    "TensorDataset", "random_split",
]

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from torch.utils.data import dataset
from torch.utils.data.dataset import *

from mon.foundation import pathlib

if TYPE_CHECKING:
    from mon.coreml.typing import TransformsType, Ints, PathType


# region Dataset

# noinspection PyMethodMayBeStatic
class Dataset(dataset.Dataset, ABC):
    """The base class of all datasets.
    
    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching
    a data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`torch.utils.data.Sampler` implementations and the default options
    of :class:`torch.utils.data.DataLoader`.
    
    Args:
        name: The dataset name.
        root: The root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
        shape: The data shape.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str,
        root            : PathType,
        split           : str,
        shape           : Ints,
        transform       : TransformsType | None = None,
        target_transform: TransformsType | None = None,
        transforms      : TransformsType | None = None,
        verbose         : bool                  = True,
        *args, **kwargs
    ):
        self.name    = name
        self.root    = pathlib.Path(root)
        self.split   = split
        self.shape   = shape
        self.verbose = verbose
        
        if transform is None:
            transform = transform
        elif isinstance(transform, transform.ComposeTransform):
            transform = transform
        else:
            transform = transform.ComposeTransform(transforms=transform)

        if target_transform is None:
            target_transform = target_transform
        elif isinstance(target_transform, transform.ComposeTransform):
            target_transform = target_transform
        else:
            target_transform = transform.ComposeTransform(transforms=target_transform)
        
        if transforms is None:
            transforms = transforms
        elif isinstance(transforms, transform.ComposeTransform):
            transforms = transforms
        else:
            transforms = transform.ComposeTransform(transforms=transforms)
        
        self.transform        = transform
        self.target_transform = target_transform
        self.transforms       = transforms
        """
        has_transforms         = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError(
                "Only transforms or transform/target_transform can be passed "
                "as argument."
            )

        self.transform        = transform
        self.target_transform = target_transform
        
        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms
        """
    
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
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)
    
    def __del__(self):
        self.close()
        
    def _format_transform_repr(
        self,
        transform: object,
        head     : str
    ) -> list[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] \
            + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

    def extra_repr(self) -> str:
        return ""
    
    @abstractmethod
    def reset(self):
        """Resets and starts over."""
        pass
    
    @abstractmethod
    def close(self):
        """Stops and releases."""
        pass

# endregion
