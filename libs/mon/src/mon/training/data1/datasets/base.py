#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for dataset abstract classes and mixins.

This module provides abstract base classes for datasets, defining common
interfaces and attributes for dataset handling, including loading, accessing,
and iterating over datapoints. It supports multiple modalities and is designed
to be extended for specific dataset implementations.
"""

__all__ = [
    "DataLoaderMixin",
    "Dataset",
    "DatasetLoadingMixin",
    "DatasetMetadataMixin",
    "DatasetMultimodalLoadingMixin",
    "Modalities",
    "Modality",
]

import abc
from collections import namedtuple
from typing import Any, Dict, TypeAlias

from torch.utils.data import dataset

from mon.core import log, Path, Split, Task
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


# --- Abstract Dataset ---
class Dataset(dataset.Dataset, abc.ABC):
    """An abstract class for all datasets.
    
    This class stores datapoints and class-labels, and provides basic interfaces
    for accessing each datapoint and metadata.
    
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
        """Initializes the Dataset instance.
        
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


# --- General Mixins ---
class DatasetMetadataMixin(abc.ABC):
    """A mixin class that adds metadata to ``Dataset`` class.
    
    This class defines common dataset attributes for categorization, such as
    supported tasks. This is useful for factory-related operations.
    
    Attributes:
        _tasks (list[Task]): A list of supported tasks. Defaults to an empty
            list and should be overridden in subclasses.
    """
    
    _tasks: list[Task] = []
    
    # --- Properties ---
    @property
    def tasks(self) -> list[Task]:
        """Getter for the list of supported tasks.
        
        Returns:
            list[Task]: The list of supported tasks.
        """
        return self._tasks


class DataLoaderMixin(abc.ABC):
    """A mixin class that adds DataLoader's functionality to ``Dataset``.
    
    This class defines the collate function for batching datapoints when using
    a DataLoader.
    """
    
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


# --- Data Loading Mixins ---
class DatasetLoadingMixin(abc.ABC):
    """A mixin class that adds data loading functionality to ``Dataset``.
    
    This class defines a skeleton for data loading pipeline while allowing
    subclasses to override specific steps without altering the overall structure.
    
    The calling sequence includes:
        1. ``load()``          : Main method to load all datapoints.
        2. ``_on_load_start()``: Hook before loading (extensible).
        3. ``_core_load()``    : Core loading mechanism (extensible).
        4. ``_on_load_end()``  : Hook after loading (extensible).
        5. ``verify()``        : Verify dataset integrity (extensible).
    """
    
    def __init__(self, *args, **kwargs):
        """Initializes the DataLoadingMixin instance."""
        self.load()
        self.verify()
    
    # --- Data Loading ---
    # noinspection PyAttributeOutsideInit
    def load(self):
        """Main method to loads all datapoints in the dataset from disk.
        
        Calls ``_on_load_start()`` hook, then ``_core_load()`` (extensible), then
        ``_on_load_end()`` hook. This method should generally not be overridden;
        extend`` _core_load()`` instead.
        
        After calling this, ``self._datapoints`` will be populated with all
        modalities' data lists. This method can be called internally or externally
        to reload the data if needed.
        """
        if not hasattr(self, "_datapoints"):
            raise AttributeError("``DataLoadingMixin`` requires ``_datapoints`` attribute from parent class.")
        
        self._on_load_start()
        self._datapoints = self._core_load()
        self._on_load_end()
    
    @abc.abstractmethod
    def _core_load(self) -> dict[str, Any]:
        """Core data loading method for the dataset.
        
        This method is abstract and must be implemented by subclasses.
        
        Returns:
            dict[str, Any]: A dictionary containing lists of datapoints for
            each modality.
        """
        pass
    
    @abc.abstractmethod
    def _on_load_start(self):
        """A hook method called at the start of the data loading process.
        
        This method can be overridden by subclasses to perform additional
        operations before the dataset is loaded.
        """
        pass
    
    @abc.abstractmethod
    def _on_load_end(self):
        """A hook method called at the end of the data loading process.
        
        This method can be overridden by subclasses to perform additional
        operations after the dataset has been loaded.
        """
        pass
    
    @abc.abstractmethod
    def verify(self):
        """Verifies dataset integrity after loading.
        
        Raises:
            RuntimeError: If no datapoints or attributes invalid.
        """
        pass


class DatasetMultimodalLoadingMixin(DatasetLoadingMixin, abc.ABC):
    """A mixin class that adds multimodal data loading functionality to
    ``Dataset``.
    
    This class defines a skeleton for multimodal data loading pipeline while
    allowing subclasses to override specific steps without altering the overall
    structure. It extends ``DatasetLoadingMixin`` pipeline by introducing
    methods for loading primary and other modality data.
    
    The calling sequence includes:
        1. ``load()``               : Main method to load all datapoints.
        2. ``_on_load_start()``     : Hook before loading (extensible).
        3. ``_core_load()``         : Core loading mechanism.
        4. ``_load_primary_data()`` : Loads primary modality data (extensible).
        5. ``_load_modality_data()``: Loads other modality data (extensible).
        6. ``_on_load_end()``       : Hook after loading (extensible).
        7. ``verify()``             : Verify dataset integrity (extensible).
        
    Attributes:
        _subset (str): The name of the dataset's subset directory. Since the
            given attribute ``root`` may only set the dataset root directory,
            this attribute defines the actual folder name of the sub-dataset
            within the root directory (e.g., dataset with multiple versions).
            Defaults to None and should be overridden in subclasses.
        _splits (list[Split]): A list of supported splits. This is used to
            validate the given attribute ``split``. Defaults to an empty list
            and should be overridden in subclasses.
        _modalities (Modalities): A dictionary defining the dataset modalities.
            Defaults to an empty dictionary and should be overridden in
            subclasses to accommodate additional modalities (e.g., depth maps,
            segmentation masks, bounding boxes, captions, or other sensor data).
    """
    
    _subset    : str         = None
    _splits    : list[Split] = []
    _modalities: Modalities  = {}
    
    def __init__(self, root: Path, split: Split, *args, **kwargs):
        """Initializes the DataLoadingMixin instance.
        
        Args:
            root (Path): Absolute path to the dataset root directory.
            split (Split): Data split subset to use.
        """
        self.root  = root
        self.split = split
        super().__init__(*args, **kwargs)
        
    # --- Properties ---
    @property
    def subset(self) -> str:
        """Getter for the dataset's subset name.
        
        Returns:
            str: The name of the dataset subset.
        """
        return self._subset
    
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
        if self._subset not in [None, ""] and root.name != self._subset:
            root = root / self._subset
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
    
    # --- Data Loading ---
    def _core_load(self) -> dict[str, Any]:
        """Core data loading mechanism for the dataset."""
        # Initialize empty datapoints dictionary with modalities
        datapoints = {}
        for k, v in self._modalities.items():
            if ((v.type  is None  or  v.module is None) or
                (v.train is False and self.split in [Split.TRAIN, Split.VAL]) or
                (v.test  is False and self.split in [Split.TEST,  Split.PREDICT])):
                continue
            datapoints[k] = []
        
        # List data
        pk, _          = self.primary_modality
        datapoints[pk] = self._load_primary_data()  # List primary modality
        for k, v in datapoints.items():             # List other modalities
            if k != pk:
                datapoints[k] = self._load_modality_data(k)
        
        return datapoints
    
    @abc.abstractmethod
    def _load_primary_data(self) -> list[Any]:
        """Loads primary modality data files in the dataset.
        
        This method is abstract and must be implemented by subclasses.
        
        Returns:
            list[Any]: A list of primary modality data files.
        """
        pass
    
    @abc.abstractmethod
    def _load_modality_data(self, key: str) -> list[Any]:
        """Loads modality data files in the dataset.
        
        This method is abstract and must be implemented by subclasses.
        
        Args:
            key (str): The modality key to load.
            
        Returns:
            list[Any]: A list of modality data files.
        """
        pass
