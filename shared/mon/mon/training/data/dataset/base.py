#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for dataset abstract classes.

This module provides abstract base classes for datasets, defining common
interfaces and attributes for dataset handling, including loading, accessing,
and iterating over datapoints. It supports multiple modalities and is designed
to be extended for specific dataset implementations.
"""

__all__ = [
    "Dataset",
    "BaseDataset",
    "Modalities",
    "Modality",
]

import abc
import os
from collections import namedtuple
from typing import Any, Dict, TypeAlias

import numpy as np
import torch
from torch.utils.data import dataset

from mon.core import create_progress_bar, log, Path, Split, Task
from ..classes import Classes

Modality  = namedtuple("Modality", [
    "name",     # The name of the directory that contains the modality data.
    "type",     # Albumentations target type (e.g. "image", "mask", ...) for augmentations.
    "module",   # The tensor class that performs I/O operations.
    "train",    # If ``True``, this modality is included in train/val set.
    "test",     # If ``True``, this modality is included in test set.
    "primary"   # If ``True``, this is the primary modality.
], defaults=[None, None, True, False, False])
Modalities: TypeAlias = Dict[str, Modality]


# ----- Abstract Dataset -----
class Dataset(dataset.Dataset, abc.ABC):
    """An abstract class for all datasets.
    
    This class defines the common interface for initializing, iterating, and
    accessing datapoints in a dataset. Most methods are left abstract and must
    be implemented by subclasses.
    
    The primary use case for this is in training pipelines (train, val, test)
    and dataset workflows that require full access to all properties of the
    dataset.
    
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
        """Initializes the Dataset instance.
        
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
        if hasattr(self, "on_load_end"):  # Optional hook after loading
            self.on_load_end()
        self.verify()
        
    # ----- Magic Methods -----
    @abc.abstractmethod
    def __del__(self):
        """Closes the dataset loading mechanism and releases resources."""
        pass
    
    @abc.abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]:
        """Gets a datapoint and metadata at the specified ``index``.
        
        Args:
            index (int): Index of datapoint.
            
        Returns:
            dict[str, Any]: A dictionary containing the datapoint and metadata.
        """
        pass
    
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
    
    # ----- Properties -----
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
    
    # ----- Initialize -----
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
    
    # ----- Access -----
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
    
    # ----- Utils -----
    @abc.abstractmethod
    def collate_fn(self, batch: list[dict]) -> dict[str, Any]:
        """Collates a batch of input items for torch.utils.data.dataset.DataLoader.
        
        By default, batch is a list of dicts, where each dict is a datapoint.
        We need to collate these into a single dict where each key corresponds to
        a modality and the values are stacked tensors or arrays.

        Args:
            batch (list[dict]): List of dicts, each dict is a datapoint.

        Returns:
            dict[str, Any]: Collated dictionary for torch.utils.data.dataset.DataLoader.
        """
        pass


# ----- Base Dataset -----
class BaseDataset(Dataset, abc.ABC):
    """A partial abstract class for all datasets.
    
    This class extends the ``Dataset`` abstract class and provides common
    attributes for categorizing datasets, such as supported tasks, splits,
    modalities, and classes.
    
    This is primarily used for factory-based dataset initialization.
    
    Attributes:
        _root_name (str): The name of the dataset root directory.
        _tasks (list[Task]): A list of supported tasks.
        _splits (list[Split]): A list of supported splits.
        _modalities (Modalities): A dictionary defining the dataset modalities.
        root (Path): The dataset root directory.
        split (Split): The current dataset split.
        transform (Any): The dataset transformations.
    """
    
    _root_name : str         = None
    _tasks     : list[Task]  = []
    _splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST, Split.PREDICT]
    _modalities: Modalities  = {}
    
    def __init__(
        self,
        root     : Path,
        split    : Split = Split.TRAIN,
        transform: Any   = None,
        classes  : Path | Classes = None,
        verbose  : bool  = False,
        *args, **kwargs
    ):
        """Initializes the BaseDataset instance.
        
        Args:
            root (Path): Absolute path to the dataset root directory.
            split (Split): Data split subset to use. One of: Split.TRAIN,
                Split.VAL, Split.TEST, or Split.PREDICT. Defaults to Split.TRAIN.
            transform (Any, optional): Transformations to apply. Defaults to None.
            classes (Path or Classes, optional): Either a path to a .yaml file
                containing class label definitions, or a Classes instance.
                If given, this will override any classes defined in the
                subclass. Defaults to None.
            verbose (bool): If True, enables verbose output. Defaults to False.
        
        Raises:
            ValueError: If ``modalities`` has no defined attributes.
        """
        # Validate modalities
        if not self.modalities:
            raise ValueError("``modalities`` has no defined attributes.")

        self.root      = root
        self.split     = split
        self.transform = transform
        
        super().__init__(classes=classes, verbose=verbose, *args, **kwargs)
    
    # ----- Magic Methods -----
    def __repr__(self) -> str:
        """Returns the string representation of the dataset.
        
        Returns:
            str: String representation of the dataset.
        """
        lines  = ["Dataset " + self.__class__.__name__]
        lines += [f"Number of datapoints: {self.__len__()}"]
        if self.root:
            lines += [f"Root location: {self.root}"]
        if hasattr(self, "transform") and self.transform:
            lines += [repr(self.transform)]
        return "\n".join(lines)
    
    # ----- Properties -----
    @property
    def root_name(self) -> str:
        """Getter for the dataset root directory name.
        
        Returns:
            str: The name of the dataset root directory.
        """
        return self._root_name
    
    @property
    def tasks(self) -> list[Task]:
        """Getter for the list of supported tasks.
        
        Returns:
            list[Task]: The list of supported tasks.
        """
        return self._tasks
    
    @property
    def splits(self) -> list[Split]:
        """Getter for the list of supported splits.
        
        Returns:
            list[Split]: The list of supported splits.
        """
        return self._splits
    
    @property
    def modalities(self) -> Modalities:
        """Getter for the dataset modalities.
        
        Returns:
            Modalities: The dataset modalities.
        """
        return self._modalities
    
    @property
    def root(self) -> Path:
        """Getter for the dataset root directory.
        
        Returns:
            Path: The dataset root directory.
        """
        return self._root
    
    @root.setter
    def root(self, root: Path):
        """Setter for the dataset root directory.
        
        Args:
            root (Path): Absolute path to the dataset root directory.
            
        Raises:
            FileNotFoundError: If the specified root directory does not exist.
        """
        root = Path(root)
        if self._root_name not in [None, ""] and root.name != self._root_name:
            root = root / self._root_name
        if not root.is_dir():
            raise FileNotFoundError(f"``root`` directory not found: {root}.")
        self._root = root
    
    @property
    def split(self) -> Split:
        """Getter for the current dataset split.
        
        Returns:
            Split: The current dataset split.
        """
        return self._split
    
    @split.setter
    def split(self, split: Split):
        """Setter for the current dataset split.
        
        Args:
            split (Split): Data split subset to use. One of: Split.TRAIN,
                Split.VAL, Split.TEST, or Split.PREDICT.
                
        Raises:
            ValueError: If ``split`` is not one of the supported splits.
        """
        split = Split(split)
        if split not in self._splits:
            raise ValueError(f"``split`` must be one of {self._splits}, got {split}.")
        self._split = split
    
    @property
    def split_str(self) -> str:
        """Getter for the current dataset split as a string.
        
        Returns:
            str: The current dataset split as a string.
        """
        return self.split.value
    
    @property
    @abc.abstractmethod
    def transform(self) -> Any:
        """Getter for the dataset transformations.
        
        Returns:
            Any: The dataset transformations.
        """
        pass
    
    @transform.setter
    @abc.abstractmethod
    def transform(self, transform: Any):
        """Setter for the dataset transformations.
        
        This method is abstract and must be implemented by subclasses.
        
        Args:
            transform (Any): Transformations to apply.
        """
        pass
    
    @property
    def primary_modality(self) -> tuple[str, Modality]:
        """Getter for the primary modality.
        
        Returns:
            tuple[str, Modality]: A tuple containing the key and Modality
                instance of the primary modality.
                
        Raises:
            ValueError: If no primary modality is defined.
        """
        for k, v in self.modalities.items():
            if v.primary:
                return k, v
        raise ValueError(f"``modalities`` has no primary modality. "
                         f"Please set `primary=True` for one of the modalities.")
    
    # ----- Initialize -----
    def load(self):
        """Initializes and loads all datapoints in the dataset from disk.
        
        After calling this, ``self._datapoints`` will be populated with all
        modalities' data lists. This method can be called internally or externally
        to reload the data if needed.
        """
        # Initialize empty datapoints dictionary with modalities
        datapoints = {}
        for k, v in self._modalities.items():
            if ((v.type  is None  or  v.module is None) or
                (v.train is False and self.split in [Split.TRAIN, Split.VAL]) or
                (v.test  is False and self.split in [Split.TEST,  Split.PREDICT])):
                continue
            datapoints[k] = []
        self._datapoints = datapoints
        
        # List data
        pk, _ = self.primary_modality
        self._datapoints[pk] = self._load_primary_data()  # List primary modality
        for k, v in self._datapoints.items():             # List other modalities
            if k != pk:
                self._datapoints[k] = self._load_modality_data(k)
    
    @abc.abstractmethod
    def _load_primary_data(self) -> list[Any]:
        """Loads primary modality data files in the dataset.
        
        This method is abstract and must be implemented by subclasses.
        
        Returns:
            list[Any]: A list of primary modality data files.
        """
        pass
    
    def _load_modality_data(self, key: str) -> list[Any]:
        """Loads modality data files in the dataset.
        
        Args:
            key (str): The modality key to load.
            
        Returns:
            list[Any]: A list of modality data files.
        """
        pk, pk_modality = self.primary_modality
        pk_name  = pk_modality.name
        pk_files = self.datapoints[pk]
        
        modality = self.modalities[key]
        name     = modality.name
        module   = modality.module
        files    = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            for file in pbar.track(
                sequence    = pk_files,
                description = f"Listing {self.__class__.__name__} {self.split_str} {key}(s)"
            ):
                path = file.path.replace_part(f"{os.sep}{pk_name}{os.sep}", f"{os.sep}{name}{os.sep}")
                files.append(module(path=path, root=file.root))
                
        return files
    
    def verify(self):
        """Verifies dataset integrity.
        
        Raises:
            RuntimeError: If no datapoints or attributes invalid.
        """
        if self.__len__() <= 0:
            raise RuntimeError("No datapoints in the dataset!")
        
        pk, _ = self.primary_modality
        for k, v in self.datapoints.items():
            if k not in self.modalities:
                raise RuntimeError(f"Modality ``{k}`` is not defined in ``modalities``. "
                                   f"Define it in the class if intentional.")
            if self.modalities[k]:
                if v in [None, []]:
                    raise RuntimeError(f"``datapoints`` has no ``{k}`` attributes!")
                elif len(v) != self.__len__():
                    raise RuntimeError(f"Number of ``{k}`` items does not match number "
                                       f"of ``{pk}``, got: {len(v)} != {self.__len__()}")
                
        if self.verbose:
            log(f"Number of {self.split_str} datapoints: {self.__len__()}.")
    
    # ----- Utils -----
    def collate_fn(self, batch: list[dict]) -> dict:
        """Collates a batch of input items for torch.utils.data.dataset.DataLoader.
        
        By default, batch is a list of dicts, where each dict is a datapoint.
        We need to collate these into a single dict where each key corresponds to
        a modality and the values are stacked tensors or arrays.

        Args:
            batch (list[dict]): List of dicts, each dict is a datapoint.

        Returns:
            dict[str, Any]: Collated dictionary for torch.utils.data.dataset.DataLoader.
        """
        zipped = {
            k: list(v)
            for k, v in zip(batch[0].keys(), zip(*[b.values() for b in batch]))
        }

        for k, v in zipped.items():
            if k not in self._modalities:  # i.e., metadata
                continue
            if v is None:
                zipped[k] = None
            elif isinstance(v[0], torch.Tensor):
                zipped[k] = torch.stack(v, dim=0)
            elif isinstance(v[0], np.ndarray):
                zipped[k] = np.stack(v, axis=0)

        return zipped
