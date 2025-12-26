#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base dataset classes and mixins.

Provides a skeleton for defining datasets and mixins for various operations on
datasets.
"""

__all__ = [
    "Dataset",
    "Modalities",
    "Modality",
]

import abc
from collections import namedtuple
from typing import Any, Dict, TypeAlias

from torch.utils.data import dataset

from mon.core import log, Path
from mon.core.dtypes import ClassList


# ==============================================================================
# GLOBAL CONFIGURATIONS (Constants)
# ==============================================================================

# --- Constants (Global defaults, versioning) ---


# --- Environment ---


# ==============================================================================
# TYPE DEFINITIONS & PROTOCOLS (Interfaces)
# ==============================================================================

# --- Type Aliases ---
Modality  = namedtuple(
    typename    = "Modality",
    field_names = [
        "name",     # The name of the directory that contains the modality data.
        "type",     # Albumentations target type (e.g. "image", "mask", ...) for augmentations.
        "module",   # The tensor class that performs I/O operations.
        "train",    # If ``True``, this modality is included in train/val set.
        "test",     # If ``True``, this modality is included in test set.
        "primary"   # If ``True``, this is the primary modality.
    ],
    defaults    = [None, None, True, False, False])
Modalities: TypeAlias = Dict[str, Modality]

# --- Structural Protocols ---


# ==============================================================================
# BASE CLASSES & MIXINS (Behaviors)
# ==============================================================================

# --- Structural Bases ---
class Dataset(dataset.Dataset, abc.ABC):
    """An abstract class for all datasets.
    
    A class representing a dataset. This is an abstract class that should be
    subclassed to create specific dataset implementations.
    
    Attributes:
        _datapoints (dict): A dictionary containing lists of datapoints for
            each modality.
        _classlist (ClassList): The dataset object classes. Defaults to None and
            should be overridden in subclasses.
        verbose (bool): If True, enables verbose output.
    """
    
    _classlist: ClassList = None
    
    def __init__(
        self,
        datapoints: dict[str, list[Any]] = None,
        classlist : Path | ClassList     = None,
        verbose   : bool                 = True,
        *args, **kwargs
    ):
        """Initialize a new instance.
        
        Args:
            datapoints: A dictionary containing lists of datapoints for each
                modality. Defaults to None.
            classlist: Either a .yaml file containing the classes definitions,
                or a ``ClassList`` instance. If given, this will override any
                ``classes`` defined in the subclass. Defaults to None.
            verbose: If True, enables verbose output. Defaults to True.
        """
        super().__init__(*args, **kwargs)
        self.verbose     = verbose
        self.classlist   = classlist
        self._datapoints = datapoints if datapoints is not None else {}
    
    # --- Magic Methods ---
    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the length of the dataset (i.e., number of datapoints)."""
        pass
    
    @abc.abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the datapoint at the specified ``index`` in ``_datapoints``.
        
        Args:
            index (int): Index of datapoint.
            
        Returns:
            A dictionary containing the datapoint and its metadata.
        """
        pass
    
    def __iter__(self):
        """Initialize the dataset iterator."""
        self._iter_idx = 0
        return self

    def __next__(self) -> dict[str, Any]:
        """Return the next datapoint in the dataset iteration.
        
        Returns:
            A dictionary containing the datapoint and its metadata.
        
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
        """Return the string representation of the dataset."""
        lines  = ["Dataset " + self.__class__.__name__]
        lines += [f"Number of datapoints: {self.__len__()}"]
        return "\n".join(lines)
    
    @abc.abstractmethod
    def __del__(self):
        """Close the dataset loading mechanism and releases resources."""
        pass
    
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
    def classlist(self) -> ClassList:
        """Return the dataset's class definitions."""
        return self._classes
    
    @classlist.setter
    def classlist(self, classlist: Path | ClassList = None):
        """Setter for the dataset classes.
        
        Args:
            classlist: Either a .yaml file containing the classes definitions,
                or a ``ClassList`` instance. If given, this will override any
                ``classes`` defined in the subclass. Defaults to None.
        
        Raises:
            TypeError: If ``classlist`` is not a valid type.
        """
        changed = False
        if classlist is not None and isinstance(classlist, Path | ClassList):
            self._classlist = ClassList(classlist)
            changed  = True
        
        if self.verbose and changed:
            log(f"``_classlist`` is updated with {classlist}.")
    
    @property
    def disable_pbar(self) -> bool:
        """Return True if progress bars are disabled, False otherwise."""
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


# --- Lifecycle Mixins ---


# --- Compute Mixins ---
