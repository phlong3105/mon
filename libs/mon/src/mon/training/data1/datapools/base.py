#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base classes for data pools.

Note:
    Datapools are different from datasets. Datapools are responsible for managing
    the labels/annotations in the dataset, while datasets are responsible for
    loading and preprocessing the datapoints for training. One example is that
    datapools do not apply any augmentations to the data, while datasets do.
    
    To make thing easier to manage and debug, we separate the data pools and
    datasets into different implementations.
"""

__all__ = [
    "DataPool",
]

import abc
from typing import Any

from torch.utils.data import dataset

from mon.core import log, Path
from ..classes import Classes


# --- Abstract Data Pool ---
class DataPool(dataset.Dataset, abc.ABC):
    """An abstract class for all data pools.
    
    This class defines the common interface for initializing, iterating, and
    accessing datapoints in a dataset. Most methods are left abstract and must
    be implemented by subclasses.
    
    Attributes:
        _datapoints (dict): A dictionary containing lists of datapoints for
            each modality.
        classes (Classes): The dataset classes/labels.
        verbose (bool): If True, enables verbose output.
    """
    
    _classes: Classes = None
    
    def __init__(
        self,
        classes: Path | Classes = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initializes the DataPool instance.
        
        Args:
            classes (Path or Classes, optional): Either a path to a .yaml file
                containing class label definitions, or a ``Classes`` instance.
                If given, this will override any ``classes`` defined in the
                subclass. Defaults to None.
            verbose (bool, optional): If True, enables verbose output. Defaults
                to True.
        """
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.classes = classes
        self._datapoints: dict[str, list[Any]] = {}
        
        # Loading pipeline
        self.load()
        if hasattr(self, "on_load_end"):  # An optional hook after loading (implemented in Mixins)
            self.on_load_end()
        self.verify()
    
    # --- Magic Methods ---
    @abc.abstractmethod
    def __del__(self):
        """Closes the dataset loading mechanism and releases resources."""
        pass
    
    def __getitem__(self, index: int) -> dict:
        """Gets a datapoint and metadata at the specified ``index``.
        
        Args:
            index (int): Index of datapoint.
            
        Returns:
            dict[str, Any]: A dictionary containing the datapoint and metadata.
        """
        data = self._get_datapoint(index)
        meta = self._get_meta(index)
        return data | {"meta": meta}
    
    def __iter__(self):
        """Initializes the dataset iterator."""
        self._iter_idx = 0
        return self
    
    @abc.abstractmethod
    def __len__(self) -> int:
        """Returns the length of the dataset.
        
        Returns:
            int: The number of datapoints in the dataset.
        """
        pass
    
    def __next__(self) -> dict[str, Any]:
        """Returns the next datapoint in the dataset iteration.
        
        Returns:
            dict[str, Any]: A dictionary containing the next datapoint.
        
        Raises:
            StopIteration: If index exceeds the dataset length.
        """
        if self._iter_idx < self.__len__():
            item = self.__getitem__(self._iter_idx)
            self._iter_idx += 1
            return item
        else:
            raise StopIteration
    
    def __repr__(self) -> str:
        """Returns the string representation of the dataset.
        
        Returns:
            str: String representation of the dataset.
        """
        lines  = ["Dataset " + self.__class__.__name__]
        lines += [f"Number of datapoints: {self.__len__()}"]
        return "\n".join(lines)
    
    # --- Properties ---
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
    def datapoints(self) -> dict[str, list[Any]]:
        """Getter for the dataset datapoints.
        
        Returns:
            dict[str, list[Any]]: A dictionary containing lists of datapoints
                for each modality.
        """
        return self._datapoints
    
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
    
    # --- Initialize ---
    @abc.abstractmethod
    def load(self):
        """Initializes and loads all datapoints in the dataset from disk.
        
        After calling this, ``self._datapoints`` will be populated with all
        modalities' data lists. This method can be called internally or externally
        to reload the data if needed.
        """
        pass
    
    @abc.abstractmethod
    def verify(self):
        """Verifies dataset integrity after loading.
        
        Raises:
            RuntimeError: If no datapoints or attributes invalid.
        """
        pass
    
    # --- Access ---
    @abc.abstractmethod
    def _get_datapoint(self, index: int) -> dict[str, Any]:
        """Gets a datapoint at the specified ``index``.

        Args:
            index (int): Index of datapoint.
            
        Returns:
            dict[str, Any]: A dictionary containing the datapoint.
        """
        pass
    
    @abc.abstractmethod
    def _get_meta(self, index: int) -> dict[str, Any]:
        """Gets metadata at the specified ``index``.

        Args:
            index (int): Index of datapoint.
            
        Returns:
            dict[str, Any]: A dictionary containing the metadata.
        """
        pass
