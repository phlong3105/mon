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
        # Ensure the container exists
        if not hasattr(self, "_datapoints"):
             self._datapoints = {}
            
        self._on_load_start()
        
        # Core loading
        datapoints = self._load_data()
        if not isinstance(datapoints, dict):
            raise TypeError(f"Expected ``_load_data()`` to return a dict, "
                            f"but got {type(datapoints).__name__}.")
        
        self._datapoints = datapoints
        
        self._on_load_end()
        
        # Auto-verify after loading is complete
        self.verify()

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
    def __init__(self, root: Path | str, split: Split | str, *args, **kwargs):
        """Initialize a new instance.

        Args:
            root: Absolute path to the dataset root directory.
            split: Data split subset to use.
        """
        # Initialize attributes to None first to avoid AttributeError
        # during setter logic if super().__init__ triggers something
        self._root  = None
        self._split = None
        
        # Run setter logic
        self.root  = root
        self.split = split
        
        # Continue the initialization chain
        super().__init__(*args, **kwargs)
        
    def __init_subclass__(cls, *args, **kwargs):
        """Called when inheriting from this class."""
        super().__init_subclass__(*args, **kwargs)
        
        # Check for EXPLICIT definition in the subclass (not inherited)
        for attr in ["_splits"]:
            if attr not in cls.__dict__:
                raise AttributeError(f"Class {cls.__name__} must explicitly define "
                                     f"class attribute ``{attr}``.")
        
        # Ensure that any class using RootLoadMixin defines its splits
        if not cls._splits:
            raise AttributeError(f"Expected '{cls.__name__}' to define '_splits' "
                                 f"attribute, but got None.")
    
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
        """Return the dataset root directory."""
        return self._root

    @root.setter
    def root(self, value: Path | str):
        """Setter for the dataset root directory.

        Args:
            value: Absolute path to the dataset root directory.

        Raises:
            FileNotFoundError: If the ``root`` directory does not exist.
        """
        root = Path(value).normalize()  # Ensure absolute, clean path
        
        # Logic for subset appending
        if self._subset not in [None, ""]:
            # Check if current root ends with subset; if not, try to append
            if root.name != self._subset:
                sub_path = root / self._subset
                if sub_path.is_dir():
                    root = sub_path
                    
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root directory not found: {root}")
        
        self._root = root

    @property
    def split(self) -> Split:
        """Return the current dataset split."""
        return self._split

    @split.setter
    def split(self, split: Split | str):
        """Setter for the current dataset split.

        Args:
            split: Data split subset to use. One of: Split.TRAIN, Split.VAL,
                Split.TEST, or Split.PREDICT.

        Raises:
            ValueError: If ``split`` is not one of the supported splits.
        """
        # Cast to Enum if it's a string
        split = Split(split)
        if split not in self._splits:
            raise ValueError(f"Expected 'split' in {self._splits}, but got '{split}'.")
        self._split = split

    @property
    def split_str(self) -> str:
        """Return the current dataset split as a string."""
        return self._split.value


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
    def __init__(self, input_dir: Path | str, target_dir: Path | str, *args, **kwargs):
        """Initialize a new instance.

        Args:
            input_dir: Absolute path to the input directory.
            target_dir: Absolute path to the target directory.
        """
        # Initialize attributes to None first to avoid AttributeError
        # during setter logic if super().__init__ triggers something
        self._input_dir  = None
        self._target_dir = None
        
        # Run setter logic
        self.input_dir  = input_dir
        self.target_dir = target_dir
        
        # Continue the initialization chain
        super().__init__(*args, **kwargs)
        
    # --- Properties ---
    @property
    def input_dir(self) -> Path:
        """Return the input directory."""
        return self._input_dir

    @input_dir.setter
    def input_dir(self, value: Path | str):
        """Setter for the input directory.

        Args:
            value: Path to the input directory.

        Raises:
            FileNotFoundError: If the ``input_dir`` directory does not exist.
        """
        if value is None:
            raise ValueError("Expected 'input_dir' to be a valid path, but got None.")
        
        input_dir = Path(value).normalize(exist=True)
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}.")
        self._input_dir = input_dir

    @property
    def target_dir(self) -> Path:
        """Return the target directory."""
        return self._target_dir

    @target_dir.setter
    def target_dir(self, value: Path | str):
        """Setter for the target directory.

        Args:
            value: Path to the target directory.

        Raises:
            FileNotFoundError: If the ``target_dir`` directory does not exist.
        """
        if value is not None:
            target_dir = Path(value).normalize(exist=True)
            if not target_dir.is_dir():
                raise FileNotFoundError(f"Target directory not found: {target_dir}.")
            self._target_dir = target_dir
        else:
            self._target_dir = None
    
    @property
    def has_target(self) -> bool:
        """Indicates whether the dataset has target data.
        
        Returns:
            bool: True if target data is available, False otherwise.
        """
        return self._target_dir is not None and self._target_dir.is_dir()
    
    @property
    def label_dir(self) -> Path:
        """An alias to ``target_dir`` for better readability in certain contexts."""
        return self._target_dir


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
    
    # --- Lifecycle & Initialization ---
    def __init_subclass__(cls, *args, **kwargs):
        """Called when inheriting from this class."""
        super().__init_subclass__(*args, **kwargs)
        
        # Check for EXPLICIT definition in the subclass (not inherited)
        for attr in ["_modalities"]:
            if attr not in cls.__dict__:
                raise TypeError(f"Class {cls.__name__} must explicitly define class "
                                f"attribute '{attr}' of type Modalities.")
        
        # Check for VALID values
        if not cls._modalities:  # Checks for None, empty list [], or empty tuple ()
            raise ValueError(f"Expected '{cls.__name__}' to define non-empty '_modalities' "
                             f"attribute, but got None or empty list or tuple.")
        
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
        try:
            return next((k, v) for k, v in self._modalities.items() if v.primary)
        except StopIteration:
            raise ValueError(f"Expected '{self.__class__.__name__}' to define a "
                             f"primary modality, but got none.")

    # --- Data Loading ---
    def _load_data(self) -> dict[str, list[Any]]:
        """Core data loading mechanism for the dataset."""
        pk, _ = self.primary_modality
        
        # Initialize empty datapoints dictionary with modalities
        datapoints = {
            k: [] for k, v in self._modalities.items()
            if v.type and v.module and (
                (v.train is not False  if self._split in [Split.TRAIN, Split.VAL]
                 else v.test is not False)
            )
        }
        
        # Load primary modality
        primary_data   = self._load_primary_data()
        datapoints[pk] = primary_data
        
        # Load other modalities using the cached primary_data and key
        for k, v in datapoints.items():
            if k != pk:
                datapoints[k] = self._load_modality_data(primary_data, k, pk)
                
        # List metadata
        datapoints["meta"] = [d.meta for d in primary_data]
        return datapoints

    def _load_primary_data(self) -> list[Any]:
        """Load primary modality data files in the dataset.

        Returns:
            A list of primary modality data files.
        """
        disable_pbar = getattr(self, "disable_pbar", False)
        
        pk, pk_modality = self.primary_modality
        pk_name   = pk_modality.name
        pk_module = pk_modality.module
        patterns  = [self._root / self.split_str / pk_name]
        files     = []
        with create_progress_bar(disable=disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} {pk}(s)"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        files.append(pk_module(data=path, root=pattern))
        
        return files

    def _load_modality_data(self, primary_data: list[Any], key: str, pk_key: str) -> list[Any]:
        """Load modality data files in the dataset.

        Args:
            primary_data: A list of primary modality data files.
            key: The modality key to load.
            pk_key: The primary modality key.

        Returns:
            A list of modality data files.
        """
        disable_pbar = getattr(self, "disable_pbar", False)
        
        # Pre-calculate strings and lookup modules outside the loop
        pk_name      = self._modalities[pk_key].name
        target       = self._modalities[key]
        name, module = target.name, target.module
        
        old_part = f"{os.sep}{pk_name}{os.sep}"
        new_part = f"{os.sep}{name}{os.sep}"
        
        files = []
        with create_progress_bar(disable=disable_pbar) as pbar:
            desc = f"Listing {self.__class__.__name__} {self.split_str} {key}(s)"
            for file in pbar.track(sequence=primary_data, description=desc):
                path = file.path.replace_part(old_part, new_part)
                files.append(module(path=path, root=file.root))
        
        return files


# ==============================================================================
# PERSISTENCE & EXPORT (Write/Commit)
# ==============================================================================

# --- Serialize (Object to Bytes) ---


# --- Commit (Saving to Disk/Cloud) ---
