#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Mixins for I/O operations.

This module provides mixins for input and output operations for data containers.
"""

__all__ = [
    "DataLoadMixin",
    "InputTargetLoadMixin",
    "MultimodalDataLoadMixin",
    "RootLoadMixin",
]

import abc
import os
from typing import Any

from mon.core import create_progress_bar, Path, Split
from ...base import Modalities, Modality


# ==============================================================================
# RESOURCE RESOLVERS (Path/URL Handling)
# ==============================================================================

# --- Path Handling (Resolving URIs, Local Paths) ---


# --- Backend Selection (Selecting PIL vs. OpenCV vs. TurboJPEG) ---


# ==============================================================================
# HYDRATION & DESERIALIZATION (Read/Load)
# ==============================================================================

# --- Deserialize (Bytes to Object) ---


# --- Loaders (Standard Disk-to-RAM logic) ---
class DataLoadMixin(abc.ABC):
    """A mixin class that adds data loading functionality to data containers.

    Define a skeleton for the data loading pipeline from disk to memory.
    Allow subclasses to override specific steps without altering the overall
    structure.

    The calling sequence includes:
        1. ``load()``          : Main method to load all datapoints.
        2. ``_on_load_start()``: Hook before loading (extensible).
        3. ``_core_load()``    : Core loading mechanism (extensible).
        4. ``_on_load_end()``  : Hook after loading (extensible).
        5. ``verify()``        : Verify dataset integrity (extensible).
    """

    # --- Data Loading ---
    # noinspection PyAttributeOutsideInit
    def load(self):
        """Main method to load all datapoints in the dataset from the disk.

        Call ``_on_load_start()`` hook, then ``_load_data()`` (extensible),
        then ``_on_load_end()`` hook. This method can be called internally or
        externally to reload the data if needed. After calling this,
        ``self._datapoints`` will be populated with all modalities' data lists.
        For example, {"image": [...], ..., "meta": [...]}.
        
        This method should generally not be overridden; extend`` _load_data()``
        instead.
        """
        if not hasattr(self, "_datapoints"):
            raise AttributeError("``DataLoadingMixin`` requires ``_datapoints`` attribute from parent class.")

        self._on_load_start()
        self._datapoints = self._load_data()
        self._on_load_end()

    @abc.abstractmethod
    def _load_data(self) -> dict[str, Any]:
        """Core data loading method for the dataset.

        This method is abstract and must be implemented by subclasses.

        Returns:
            A dictionary containing lists of datapoints for each modality.
        """
        pass

    def _on_load_start(self):
        """A hook method called at the start of the data loading process.

        This method can be overridden by subclasses to perform additional
        operations before the dataset is loaded.
        """
        pass

    def _on_load_end(self):
        """A hook method called at the end of the data loading process.

        This method can be overridden by subclasses to perform additional
        operations after the dataset has been loaded.
        """
        pass

    def verify(self):
        """Verify dataset integrity after loading.

        This method can be overridden by subclasses to perform additional
        operations after the dataset has been loaded.

        Raises:
            RuntimeError: If no datapoints or attributes are invalid.
        """
        pass


class RootLoadMixin(DataLoadMixin, abc.ABC):
    """A mixin class that adds data loading functionality from a given ``root``
    directory to data containers.

    Extend ``DataLoadMixin`` by introducing ``root`` and ``split`` attributes,
    along with validation for these attributes. The data is located at:
    ``root/subset/split/...``.

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

    # --- Lifecycle & Initialization ---
    def __init__(self, root: Path, split: Split, *args, **kwargs):
        """Initialize a new instance.

        Args:
            root: Absolute path to the dataset root directory.
            split: Data split subset to use.
        """
        super().__init__(*args, **kwargs)
        self.root  = root
        self.split = split

    # --- Properties ---
    @property
    def subset(self) -> str:
        """Return the dataset's subset name."""
        return self._subset

    @property
    def splits(self) -> list[Split]:
        """Return the list of supported splits."""
        return self._splits

    @property
    def root(self) -> Path:
        """Return for the dataset root directory."""
        return self._root

    @root.setter
    def root(self, root: Path):
        """Setter for the dataset root directory.

        Args:
            root: Absolute path to the dataset root directory.

        Raises:
            FileNotFoundError: If the ``root`` directory does not exist.
        """
        root = Path(root)
        if self._subset not in [None, ""] and root.name != self._subset:
            root = root / self._subset
        if not root.is_dir():
            raise FileNotFoundError(f"``root`` directory not found: {root}.")
        self._root = root

    @property
    def split(self) -> Split:
        """Return the current dataset split."""
        return self._split

    @split.setter
    def split(self, split: Split):
        """Setter for the current dataset split.

        Args:
            split: Data split subset to use. One of: Split.TRAIN, Split.VAL,
                Split.TEST, or Split.PREDICT.

        Raises:
            ValueError: If ``split`` is not one of the supported splits.
        """
        split = Split(split)
        if split not in self._splits:
            raise ValueError(f"``split`` must be one of {self._splits}, got {split}.")
        self._split = split

    @property
    def split_str(self) -> str:
        """Return the current dataset split as a string."""
        return self.split.value


class InputTargetLoadMixin(DataLoadMixin, abc.ABC):
    """A mixin class that adds data loading functionality from two given
    ``input_dir`` and ``target_dir`` directories to data containers.

    Extend ``DataLoadMixin`` by introducing ``input_dir`` and ``target_dir``
    attributes, along with validation for these attributes. The data is located
    at: ``input_dir/...`` and ``target_dir/...``.

    Attributes:
        input_dir (Path): Absolute path to the input directory.
        target_dir (Path): Absolute path to the target directory.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, input_dir: Path, target_dir: Path, *args, **kwargs):
        """Initialize a new instance.

        Args:
            input_dir: Absolute path to the input directory.
            target_dir: Absolute path to the target directory.
        """
        super().__init__(*args, **kwargs)
        self.input_dir  = input_dir
        self.target_dir = target_dir

    # --- Properties ---
    @property
    def input_dir(self) -> Path:
        """Return the input directory."""
        return self._input_dir

    @input_dir.setter
    def input_dir(self, input_dir: Path):
        """Setter for the input directory.

        Args:
            input_dir: Path to the input directory.

        Raises:
            FileNotFoundError: If the ``input_dir`` directory does not exist.
        """
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise FileNotFoundError(f"``input_dir`` directory not found: {input_dir}.")
        self._input_dir = input_dir

    @property
    def target_dir(self) -> Path:
        """Return the target directory."""
        return self._target_dir

    @target_dir.setter
    def target_dir(self, target_dir: Path):
        """Setter for the target directory.

        Args:
            target_dir: Path to the target directory.

        Raises:
            FileNotFoundError: If the ``target_dir`` directory does not exist.
        """
        if target_dir is not None:
            target_dir = Path(target_dir)
            if not target_dir.is_dir():
                raise FileNotFoundError(f"``target_dir`` directory not found: {target_dir}.")
        self._target_dir = target_dir
    
    @property
    def has_target(self) -> bool:
        """Indicates whether the dataset has target data.
        
        Returns:
            bool: True if target data is available, False otherwise.
        """
        return self.target_dir is not None and self.target_dir.is_dir()
    
    @property
    def label_dir(self) -> Path:
        """An alias to ``target_dir`` for better readability in certain contexts."""
        return self.target_dir


class MultimodalDataLoadMixin(RootLoadMixin):
    """A mixin class that adds multimodal data loading functionality from a
    given ``root`` directory to data containers.

    Define a skeleton for a multimodal data loading pipeline while allowing
    subclasses to override specific steps without altering the overall
    structure. Extend ``RootLoadMixin`` by introducing a ``_modalities``
    attribute that defines the dataset modalities.

    The calling sequence includes:
        1. ``load()``               : Main method to load all datapoints.
        2. ``_on_load_start()``     : Hook before loading (extensible).
        3. ``_load_data()``         : Core loading mechanism.
        4. ``_load_primary_data()`` : Load primary modality data (extensible).
        5. ``_load_modality_data()``: Load other modality data (extensible).
        6. ``_on_load_end()``       : Hook after loading (extensible).
        7. ``verify()``             : Verify dataset integrity (extensible).

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
        """Return the dataset modalities."""
        return self._modalities

    @property
    def primary_modality(self) -> tuple[str, Modality]:
        """Return a tuple containing the key and Modality instance of the
        primary modality.

        Raises:
            ValueError: If no primary modality is defined.
        """
        for k, v in self.modalities.items():
            if v.primary:
                return k, v
        raise ValueError(f"``modalities`` has no primary modality. "
                         f"Please set `primary=True` for one of the modalities.")

    # --- Data Loading ---
    def _load_data(self) -> dict[str, list[Any]]:
        """Core data loading mechanism for the dataset."""
        # Initialize empty datapoints dictionary with modalities
        datapoints = {}
        for k, v in self._modalities.items():
            # Skips modalities when unavailable or split‑incompatible
            if ((v.type  is None  or  v.module is None) or
                (v.train is False and self.split in [Split.TRAIN, Split.VAL]) or
                (v.test  is False and self.split in [Split.TEST,  Split.PREDICT])):
                continue
            datapoints[k] = []

        # List data
        pk, _          = self.primary_modality
        primary_data   = self._load_primary_data()  # Load primary modality
        datapoints[pk] = primary_data
        for k, v in datapoints.items():             # Load other modalities
            if k != pk:
                datapoints[k] = self._load_modality_data(primary_data, k)
                
        # List metadata
        datapoints["meta"] = [d.meta for d in primary_data]
        
        return datapoints

    def _load_primary_data(self) -> list[Any]:
        """Load primary modality data files in the dataset.

        Returns:
            A list of primary modality data files.
        """
        if not hasattr(self, "disable_pbar"):
            raise AttributeError("``MultimodalDataLoadMixin`` requires ``disable_pbar`` attribute from parent class.")
        
        pk, pk_modality = self.primary_modality
        pk_name   = pk_modality.name
        pk_module = pk_modality.module
        patterns  = [self._root / self.split_str / pk_name]
        files     = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} {pk}(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        files.append(pk_module(data=path, root=pattern))
        
        return files

    def _load_modality_data(self, primary_data: list[Any], key: str) -> list[Any]:
        """Load modality data files in the dataset.

        Args:
            primary_data: A list of primary modality data files.
            key: The modality key to load.

        Returns:
            A list of modality data files.
        """
        if not hasattr(self, "disable_pbar"):
            raise AttributeError("``MultimodalDataLoadMixin`` requires ``disable_pbar`` attribute from parent class.")
        
        pk, pk_modality = self.primary_modality
        pk_name  = pk_modality.name
        
        modality = self.modalities[key]
        name     = modality.name
        module   = modality.module
        files    = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            for file in pbar.track(
                sequence    = primary_data,
                description = f"Listing {self.__class__.__name__} {self.split_str} {key}(s)"
            ):
                path = file.path.replace_part(f"{os.sep}{pk_name}{os.sep}", f"{os.sep}{name}{os.sep}")
                files.append(module(path=path, root=file.root))
        
        return files


# ==============================================================================
# PERSISTENCE & EXPORT (Write/Commit)
# ==============================================================================

# --- Serialize (Object to Bytes) ---


# --- Commit (Saving to Disk/Cloud) ---
