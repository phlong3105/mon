#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for mixins that provide data retrieval capabilities for data
containers.

This module defines mixin classes that can be used to add data retrieval
functionality to various data container classes. These mixins include methods
for retrieving data samples, batches, and metadata from the containers.
"""

__all__ = [
    "DataFetchMixin",
    "InputTargetFetchMixin",
    "Modalities",
    "Modality",
    "MultimodalDataFetchMixin",
    "RootFetchMixin",
]

import abc
from collections import namedtuple
from typing import Any, Dict, TypeAlias

from mon.core import Path, Split

Modality  = namedtuple("Modality", [
    "name",     # The name of the directory that contains the modality data.
    "type",     # Albumentations target type (e.g. "image", "mask", ...) for augmentations.
    "module",   # The tensor class that performs I/O operations.
    "train",    # If ``True``, this modality is included in train/val set.
    "test",     # If ``True``, this modality is included in test set.
    "primary"   # If ``True``, this is the primary modality.
], defaults=[None, None, True, False, False])
Modalities: TypeAlias = Dict[str, Modality]


# --- Data Fetching Mixin ---
class DataFetchMixin(abc.ABC):
    """A mixin class that adds data fetching functionality to data containers.

    This class defines a skeleton for data fetching pipeline from disk to memory.
    It allows subclasses to override specific steps without altering the overall
    structure.

    The calling sequence includes:
        1. ``fetch()``          : Main method to fetch all datapoints.
        2. ``_on_fetch_start()``: Hook before fetching (extensible).
        3. ``_core_fetch()``    : Core fetching mechanism (extensible).
        4. ``_on_fetch_end()``  : Hook after fetching (extensible).
        5. ``verify()``         : Verify dataset integrity (extensible).
    """

    # --- Data Fetching ---
    # noinspection PyAttributeOutsideInit
    def fetch(self):
        """Main method to fetch all datapoints in the dataset from disk.

        Calls ``_on_fetch_start()`` hook, then ``_fetch_data()`` (extensible), then
        ``_on_fetch_end()`` hook. This method should generally not be overridden;
        extend`` _core_fetch()`` instead.

        After calling this, ``self._datapoints`` will be populated with all
        modalities' data lists. This method can be called internally or externally
        to reload the data if needed.
        """
        if not hasattr(self, "_datapoints"):
            raise AttributeError("``DataLoadingMixin`` requires ``_datapoints`` attribute from parent class.")

        self._on_fetch_start()
        self._datapoints = self._fetch_data()
        self._on_fetch_end()

    @abc.abstractmethod
    def _fetch_data(self) -> dict[str, Any]:
        """Core data fetching method for the dataset.

        This method is abstract and must be implemented by subclasses.

        Returns:
            dict[str, Any]: A dictionary containing lists of datapoints for
            each modality.
        """
        pass

    def _on_fetch_start(self):
        """A hook method called at the start of the data fetching process.

        This method can be overridden by subclasses to perform additional
        operations before the dataset is fetched.
        """
        pass

    def _on_fetch_end(self):
        """A hook method called at the end of the data fetching process.

        This method can be overridden by subclasses to perform additional
        operations after the dataset has been fetched.
        """
        pass

    def verify(self):
        """Verifies dataset integrity after loading.

        This method can be overridden by subclasses to perform additional
        operations after the dataset has been fetched.

        Raises:
            RuntimeError: If no datapoints or attributes invalid.
        """
        pass


# --- Directory-Aware Fetch Mixin ---
class RootFetchMixin(DataFetchMixin, abc.ABC):
    """A mixin class that adds data fetching functionality from a given ``root``
    directory to data containers.

    This class extends ``DataFetchMixin`` by introducing ``root`` and ``split``
    attributes, along with validation for these attributes. The data is located
    at: ``root/subset/split/...``.

    Attributes:
        _subset (str): The name of the dataset's subset directory. Since the
            given attribute ``root`` may only set the dataset root directory,
            this attribute defines the actual folder name of the sub-dataset
            within the root directory (e.g., dataset with multiple versions).
            Defaults to None and should be overridden in subclasses.
        _splits (list[Split]): A list of supported splits. This is used to
            validate the given attribute ``split``. Defaults to an empty list
            and should be overridden in subclasses.
    """

    _subset: str         = None
    _splits: list[Split] = []

    def __init__(self, root: Path, split: Split, *args, **kwargs):
        """Initializes the DataLoadingMixin instance.

        Args:
            root (Path): Absolute path to the dataset root directory.
            split (Split): Data split subset to use.
        """
        super().__init__(*args, **kwargs)
        self.root  = root
        self.split = split

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


class InputTargetFetchMixin(DataFetchMixin, abc.ABC):
    """A mixin class that adds data fetching functionality from two given
    ``input_dir`` and ``label_dir`` directories to data containers.

    This class extends ``DataFetchMixin`` by introducing ``input_dir`` and
    ``label_dir`` attributes, along with validation for these attributes. The
    data is located at: ``input_dir/...`` and ``label_dir/...``.

    Attributes:
        input_dir (Path): Absolute path to the input directory.
        label_dir (Path): Absolute path to the label directory.
    """

    def __init__(self, input_dir: Path, label_dir: Path, *args, **kwargs):
        """Initializes the DualPathFetchMixin instance.

        Args:
            input_dir (Path): Absolute path to the input directory.
            label_dir (Path): Absolute path to the label directory.
        """
        super().__init__(*args, **kwargs)
        self.input_dir = input_dir
        self.label_dir = label_dir

    # --- Properties ---
    @property
    def input_dir(self) -> Path:
        """Getter for the input directory.

        Returns:
            Path: Path to the input directory.
        """
        return self._input_dir

    @input_dir.setter
    def input_dir(self, input_dir: Path):
        """Setter for the input directory.

        Args:
            input_dir (Path): Path to the input directory.

        Raises:
            FileNotFoundError: If the specified directory does not exist.
        """
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise FileNotFoundError(f"``input_dir`` directory not found: {input_dir}.")
        self._input_dir = input_dir

    @property
    def label_dir(self) -> Path:
        """Getter for the label directory.

        Returns:
            Path: Path to the label directory.
        """
        return self._label_dir

    @label_dir.setter
    def label_dir(self, label_dir: Path):
        """Setter for the label directory.

        Args:
            label_dir (Path): Path to the label directory.

        Raises:
            FileNotFoundError: If the specified directory does not exist.
        """
        label_dir = Path(label_dir)
        if not label_dir.is_dir():
            raise FileNotFoundError(f"``label_dir`` directory not found: {label_dir}.")
        self._label_dir = label_dir


# --- Multimodal Fetch Mixin ---
class MultimodalDataFetchMixin(RootFetchMixin, abc.ABC):
    """A mixin class that adds multimodal data fetching functionality from a
    given ``root`` directory to data containers.

    This class defines a skeleton for multimodal data loading pipeline while
    allowing subclasses to override specific steps without altering the overall
    structure. It extends ``RootFetchMixin`` by introducing a ``_modalities``
    attribute that defines the dataset modalities.

    The calling sequence includes:
        1. ``fetch()``               : Main method to fetch all datapoints.
        2. ``_on_fetch_start()``     : Hook before fetching (extensible).
        3. ``_fetch_data()``         : Core loading mechanism.
        4. ``_fetch_primary_data()`` : Fetches primary modality data (extensible).
        5. ``_fetch_modality_data()``: Fetches other modality data (extensible).
        6. ``_on_fetch_end()``       : Hook after fetching (extensible).
        7. ``verify()``              : Verify dataset integrity (extensible).

    Attributes:
        _modalities (Modalities): A dictionary defining the dataset modalities.
            Defaults to an empty dictionary and should be overridden in
            subclasses to accommodate additional modalities (e.g., depth maps,
            segmentation masks, bounding boxes, captions, or other sensor data).
    """

    _modalities: Modalities = {}

    # --- Properties ---
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

    # --- Data Fetching ---
    def _fetch_data(self) -> dict[str, Any]:
        """Core data fetching mechanism for the dataset."""
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
        datapoints[pk] = self._fetch_primary_data()  # Fetch primary modality
        for k, v in datapoints.items():              # getch other modalities
            if k != pk:
                datapoints[k] = self._fetch_modality_data(k)

        return datapoints

    @abc.abstractmethod
    def _fetch_primary_data(self) -> list[Any]:
        """Fetches primary modality data files in the dataset.

        This method is abstract and must be implemented by subclasses.

        Returns:
            list[Any]: A list of primary modality data files.
        """
        pass

    @abc.abstractmethod
    def _fetch_modality_data(self, key: str) -> list[Any]:
        """Fetches modality data files in the dataset.

        This method is abstract and must be implemented by subclasses.

        Args:
            key (str): The modality key to fetch.

        Returns:
            list[Any]: A list of modality data files.
        """
        pass
