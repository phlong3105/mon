#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base dataset classes and mixins.

Provides a skeleton for defining datasets and mixins for various operations on
datasets.
"""

__all__ = [
    "Dataset",
]

import abc
from typing import Any

from torch.utils.data import dataset

from mon.core import log, Path
from mon.core.dtypes import Classes


# --- Abstract Dataset ---
class Dataset(dataset.Dataset, abc.ABC):
    """An abstract class for all datasets.
    
    A class representing a dataset. This is an abstract class that should be
    subclassed to create specific dataset implementations.
    
    Attributes:
        _datapoints (dict): A dictionary containing lists of datapoints for
            each modality.
        classes (Classes): The dataset classes/labels. Defaults to None and
            should be overridden in subclasses.
        verbose (bool): If True, enables verbose output.
    """
    
    _classes: Classes = None
    
    def __init__(
        self,
        datapoints: dict[str, list[Any]] = None,
        classes   : Path | Classes       = None,
        verbose   : bool                 = True,
        *args, **kwargs
    ):
        """Initialize the Dataset instance.
        
        Args:
            datapoints (dict, optional): A dictionary containing lists of
                datapoints for each modality. Defaults to None.
            classes (Path or Classes, optional): Either a path to a .yaml file
                containing class label definitions, or a ``Classes`` instance.
                If given, this will override any ``classes`` defined in the
                subclass. Defaults to None.
            verbose (bool, optional): If True, enables verbose output. Defaults
                to True.
        """
        super().__init__(*args, **kwargs)
        self.verbose     = verbose
        self.classes     = classes
        self._datapoints = datapoints if datapoints is not None else {}
    
    # --- Magic Methods ---
    @abc.abstractmethod
    def __del__(self):
        """Close the dataset loading mechanism and releases resources."""
        pass
    
    @abc.abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a datapoint and metadata at the specified ``index``.
        
        Args:
            index (int): Index of datapoint.
            
        Returns:
            dict[str, Any]: A dictionary containing the datapoint and metadata.
        """
        pass
    
    def __iter__(self):
        """Initialize the dataset iterator."""
        self._iter_idx = 0
        return self

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the length of the dataset.
        
        Returns:
            int: The number of datapoints in the dataset.
        """
        pass
    
    def __next__(self) -> dict[str, Any]:
        """Return the next datapoint in the dataset iteration.
        
        Returns:
            dict[str, Any]: A dictionary containing the next datapoint.
        
        Raises:
            StopIteration: If ``_iter_idx`` exceeds the dataset length.
        """
        if self._iter_idx < self.__len__():
            item = self.__getitem__(self._iter_idx)
            self._iter_idx += 1
            return item
        else:
            raise StopIteration
    
    def __repr__(self) -> str:
        """Return the string representation of the dataset.
        
        Returns:
            str: String representation of the dataset.
        """
        lines  = ["Dataset " + self.__class__.__name__]
        lines += [f"Number of datapoints: {self.__len__()}"]
        return "\n".join(lines)
    
    # --- Properties ---
    @property
    def datapoints(self) -> dict[str, list[Any]]:
        """Getter for the dataset datapoints.
        
        Returns:
            dict[str, list[Any]]: A dictionary containing lists of datapoints
                for each modality.
        """
        return self._datapoints
    
    @property
    def classes(self) -> Classes:
        """Getter for the dataset classes.
        
        Returns:
            Classes: The dataset classes/labels.
        """
        return self._classes
    
    @classes.setter
    def classes(self, classes: Path | Classes = None):
        """Setter for the dataset classes.
        
        Args:
            classes (Path or Classes, optional): Either a path to a .yaml file
                containing class label definitions, or a ``Classes`` instance.
                If given, this will override any ``classes`` defined in the
                subclass. Defaults to None.
        
        Raises:
            TypeError: If ``classes`` is not a valid type.
        """
        changed = False
        if classes is not None and isinstance(classes, Path | Classes):
            self._classes = Classes(classes)
            changed  = True
        
        if self.verbose and changed:
            log(f"``classes`` is updated with {classes}.")
    
    @property
    def verbose(self) -> bool:
        """Getter for the verbosity mode.
        
        Returns:
            bool: True if verbose output is enabled, False otherwise.
        """
        return self._verbose
    
    @verbose.setter
    def verbose(self, verbose: bool):
        """Setter for the verbosity mode.
        
        Args:
            verbose (bool): If True, enables verbose output.
        """
        self._verbose = bool(verbose)
    
    @property
    def disable_pbar(self) -> bool:
        """Getter for disabling progress bars.
        
        Returns:
            bool: True if progress bars are disabled, False otherwise.
        """
        return not self.verbose
    
    # --- Access ---
    @abc.abstractmethod
    def _get_datapoint(self, index: int) -> dict[str, Any]:
        """Get a datapoint at the specified ``index``.

        Args:
            index (int): Index of datapoint.
            
        Returns:
            dict[str, Any]: A dictionary containing the datapoint.
        """
        pass
    
    @abc.abstractmethod
    def _get_meta(self, index: int) -> dict[str, Any]:
        """Get metadata at the specified ``index``.

        Args:
            index (int): Index of datapoint.
            
        Returns:
            dict[str, Any]: A dictionary containing the metadata.
        """
        pass
