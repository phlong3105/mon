#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""I/O operation mixins for data containers.

This module provides mixins for input and output operations for data containers.
"""

from __future__ import annotations

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
# region DISCOVERY
# ==============================================================================


# endregion


# ==============================================================================
# region CONNECTION
# ==============================================================================


# endregion


# ==============================================================================
# region INPUT
# ==============================================================================

class DataLoadMixin(abc.ABC):
    """Data loading mixin.

    Define a skeleton for the data loading pipeline from disk to memory. Allow
    subclasses to override specific steps without altering the overall
    structure.

    The calling sequence includes:
        1. ``load()``          : Main method to load all datapoints.
        2. ``_on_load_start()``: Hook before loading (extensible).
        3. ``_load_data()``    : Core loading mechanism (extensible).
        4. ``_on_load_end()``  : Hook after loading (extensible).
        5. ``verify()``        : Verify dataset integrity (extensible).
    """

    # --- Data Loading ---
    # noinspection PyAttributeOutsideInit
    def load(self):
        """Load all datapoints in the dataset from the disk.

        Call ``_on_load_start()`` hook, then ``_load_data()``, then
        ``_on_load_end()`` hook. After calling this, ``_datapoints`` will be
        populated with all modalities' data lists.

        Raises:
            TypeError: If ``_load_data()`` does not return a dict.
        """
        # Ensure the container exists
        if not hasattr(self, "_datapoints"):
            self._datapoints = {}
            
        self._on_load_start()
        
        # Core loading
        datapoints = self._load_data()
        if not isinstance(datapoints, dict):
            raise TypeError(
                f"Expected '_load_data' to return a dict, but got {type(datapoints).__name__}."
            )
        
        self._datapoints = datapoints
        
        self._on_load_end()
        
        # Auto-verify after loading is complete
        self.verify()

    @abc.abstractmethod
    def _load_data(self) -> dict[str, Any]:
        """Load core data for the dataset.

        Returns:
            Dictionary containing lists of datapoints for each modality.
        """
        pass

    def _on_load_start(self):
        """Execute hook at the start of the data loading process."""
        pass

    def _on_load_end(self):
        """Execute hook at the end of the data loading process."""
        pass

    def verify(self):
        """Verify dataset integrity after loading."""
        pass


class RootLoadMixin(DataLoadMixin, abc.ABC):
    """Root directory data loading mixin.

    Extend ``DataLoadMixin`` by introducing ``root`` and ``split`` attributes,
    along with validation for these attributes.

    Attributes:
        _subset (str | None): Name of the dataset's subset directory.
            Defaults to None.
        _splits (list[Split]): List of supported splits. Defaults to [].
        _root (mon.core.pathlib.Path | None): Dataset root directory.
            Defaults to None.
        _split (mon.core.enum.Split | None): Current dataset split.
            Defaults to None.
    """
    
    _subset: str | None  = None
    _splits: list[Split] = []

    # --- Lifecycle & Initialization ---
    def __init__(self, root: Path | str, split: Split | str, *args, **kwargs):
        """Initialize a new instance.

        Args:
            root: Absolute path to the dataset root directory.
            split: Data split subset to use.
            *args: Positional arguments.
            **kwargs: Keyword arguments.
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
        """Validate subclass attributes on inheritance.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Raises:
            TypeError: If ``_splits`` is not defined in the subclass.
        """
        super().__init_subclass__(*args, **kwargs)
        
        # Check for EXPLICIT definition in the subclass (not inherited)
        for attr in ["_splits"]:
            if attr not in cls.__dict__:
                raise TypeError(f"Class {cls.__name__} must define '{attr}' attribute.")
        
        # Ensure that any class using RootLoadMixin defines its splits
        if not cls._splits:
            raise TypeError(f"Class {cls.__name__} must define '_splits' attribute.")
    
    # --- Properties ---
    @property
    def subset(self) -> str | None:
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
        """Set the dataset root directory.

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
            raise FileNotFoundError(f"Dataset root not found at: {root}")
        
        self._root = root
    
    @property
    def split(self) -> Split:
        """Return the current dataset split."""
        return self._split

    @split.setter
    def split(self, split: Split | str):
        """Set the current dataset split.

        Args:
            split: Data split subset to use. One of: [Split.TRAIN,  Split.VAL,
                Split.TEST, Split.PREDICT].

        Raises:
            ValueError: If ``split`` is not one of the supported splits.
        """
        # Cast to Enum if it's a string
        split = Split(split)
        if split not in self._splits:
            raise ValueError(f"Unsupported 'split': {split}. Must be one of: {self._splits}.")
        self._split = split

    @property
    def split_str(self) -> str:
        """Return the current dataset split as a string."""
        return self._split.value


class InputTargetLoadMixin(DataLoadMixin, abc.ABC):
    """Input and target directory data loading mixin.

    Extend ``DataLoadMixin`` by introducing ``input_dir`` and ``target_dir``
    attributes, along with validation for these attributes.

    Attributes:
        _input_dir (Path | None): Absolute path to the input directory.
            Defaults to None.
        _target_dir (Path | None): Absolute path to the target directory.
            Defaults to None.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, input_dir: Path | str, target_dir: Path | str, *args, **kwargs):
        """Initialize a new instance.

        Args:
            input_dir: Absolute path to the input directory.
            target_dir: Absolute path to the target directory.
            *args: Positional arguments.
            **kwargs: Keyword arguments.
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
        """Set the input directory.

        Args:
            value: Path to the input directory.

        Raises:
            TypeError: If ``value`` is None.
            FileNotFoundError: If the ``input_dir`` directory does not exist.
        """
        if value is None:
            raise TypeError(
                f"Expected 'input_dir' to be a Path or str, but got {type(value).__name__}."
            )
        
        input_dir = Path(value).normalize(exist=True)
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found at: {input_dir}")
        self._input_dir = input_dir

    @property
    def target_dir(self) -> Path | None:
        """Return the target directory."""
        return self._target_dir

    @target_dir.setter
    def target_dir(self, value: Path | str | None):
        """Set the target directory.

        Args:
            value: Path to the target directory. Defaults to None.

        Raises:
            FileNotFoundError: If the ``target_dir`` directory does not exist.
        """
        if value is not None:
            target_dir = Path(value).normalize(exist=True)
            if not target_dir.is_dir():
                raise FileNotFoundError(f"Target directory not found at: {target_dir}")
            self._target_dir = target_dir
        else:
            self._target_dir = None
    
    @property
    def has_target(self) -> bool:
        """Check if the dataset has target data."""
        return self._target_dir is not None and self._target_dir.is_dir()
    
    @property
    def label_dir(self) -> Path | None:
        """An alias to ``target_dir`` for better readability in certain contexts."""
        return self._target_dir


class MultimodalDataLoadMixin(RootLoadMixin):
    """Multimodal data loading mixin.

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
        _modalities (Modalities): Dictionary defining the dataset modalities.
            Defaults to {}.
    """

    _modalities: Modalities = {}
    
    # --- Lifecycle & Initialization ---
    def __init_subclass__(cls, *args, **kwargs):
        """Validate subclass attributes on inheritance.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Raises:
            TypeError: If ``_modalities`` is not defined in the subclass.
        """
        super().__init_subclass__(*args, **kwargs)
        
        # Check for EXPLICIT definition in the subclass (not inherited)
        for attr in ["_modalities"]:
            if attr not in cls.__dict__:
                raise TypeError(f"Class {cls.__name__} must define '{attr}' attribute.")
        
        # Check for VALID values
        if not cls._modalities:  # Checks for None, empty list [], or empty tuple ()
            raise TypeError(f"Class {cls.__name__} must define '_modalities' attribute.")
        
    # --- Properties ---
    @property
    def modalities(self) -> Modalities:
        """Return the dataset modalities."""
        return self._modalities

    @property
    def primary_modality(self) -> tuple[str, Modality]:
        """Return the primary modality.

        Raises:
            ValueError: If no primary modality is defined.
        """
        try:
            return next((k, v) for k, v in self._modalities.items() if v.primary)
        except StopIteration:
            raise ValueError(f"Primary modality not found in {self.__class__.__name__}.")

    # --- Data Loading ---
    def _load_data(self) -> dict[str, list[Any]]:
        """Load core data for the dataset.

        Returns:
            Dictionary containing lists of datapoints for each modality.
        """
        pk, _ = self.primary_modality
        
        # Initialize empty datapoints dictionary with modalities
        datapoints = {
            k: [] for k, v in self._modalities.items()
            if v.type and v.module and (
                (v.train is not False if self._split in [Split.TRAIN, Split.VAL]
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
            List of primary modality data files.
        """
        disable_pbar = getattr(self, "disable_pbar", False)
        
        pk, pk_modality = self.primary_modality
        pk_name   = pk_modality.name
        pk_module = pk_modality.module
        pattern   = self._root / self.split_str / pk_name
        
        files = []
        with create_progress_bar(disable=disable_pbar) as pbar:
            paths = sorted(pattern.rglob("*"))
            desc  = f"Listing {self.__class__.__name__} {self.split_str} {pk}(s)"
            for path in pbar.track(sequence=paths, description=desc):
                if path.is_image_file():
                    files.append(pk_module(data=path, root=pattern))
        
        return files

    def _load_modality_data(
        self,
        primary_data: list[Any],
        key         : str,
        pk_key      : str
    ) -> list[Any]:
        """Load modality data files in the dataset.

        Args:
            primary_data: List of primary modality data files.
            key: Modality key to load.
            pk_key: Primary modality key.

        Returns:
            List of modality data files.
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

# endregion


# ==============================================================================
# region OUTPUT
# ==============================================================================


# endregion
