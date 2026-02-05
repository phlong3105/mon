#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Mixins for additional data handling functionalities.

This module provides mixins for additional data handling functionalities.
"""

from __future__ import annotations

__all__ = [
    "BatchCollateMixin",
    "DataLoadMixin",
    "InputTargetLoadMixin",
    "MultimodalDataLoadMixin",
    "RootLoadMixin",
]

import abc
import os
from typing import Any

import numpy as np
import torch

from mon.core import create_progress_bar, is_valid_str, Path, Split
from ..base import Modalities, Modality


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
        if not hasattr(self, "datapoints"):
            self.datapoints = {}

        self._on_load_start()

        # Core loading
        datapoints = self._load_data()
        if not isinstance(datapoints, dict):
            raise TypeError(
                f"Expected '_load_data' to return a dict, but got {type(datapoints).__name__}."
            )

        self.datapoints = datapoints

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
        """Execute a hook at the start of the data loading process."""
        pass

    def _on_load_end(self):
        """Execute a hook at the end of the data loading process."""
        pass

    def verify(self):
        """Verify dataset integrity after loading."""
        pass


class RootLoadMixin(DataLoadMixin, abc.ABC):
    """Root directory data loading mixin.

    Extend ``DataLoadMixin`` by introducing ``root`` and ``split`` attributes,
    along with validation for these attributes.

    Attributes:
        subset: Name of the dataset's subset directory. `Should be defined in
            subclasses.`
        splits: List of supported splits. `Should be defined in subclasses.`
        root: Dataset root directory.
        split: Current dataset split. Must be one of ``splits``.
    """

    subset: str | None  = None
    splits: list[Split] = []

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        root : Path  | str,
        split: Split | str,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            root: Absolute path to the dataset root directory.
            split: Data split subset to use.
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        # Assign attributes
        # Initialize attributes to None first to avoid AttributeError
        # during setter logic if super().__init__ triggers something
        self.root  = None
        self.split = None
        self.set_root(root)
        self.set_split(split)

        # Continue the initialization chain
        super().__init__(*args, **kwargs)

    def __init_subclass__(cls, *args, **kwargs):
        """Validate subclass attributes on inheritance.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Raises:
            TypeError: If ``splits`` is not defined in the subclass.
        """
        super().__init_subclass__(*args, **kwargs)

        # Check for EXPLICIT definition in the subclass (not inherited)
        for attr in ["splits"]:
            if not hasattr(cls, attr):  # or getattr(cls, attr) is None:
                raise TypeError(
                    f"Class {cls.__name__} must define '{attr}' attribute "
                    f"(defined locally or inherited)."
                )

    # --- Properties ---
    def set_root(self, value: Path | str):
        """Set the dataset root directory.

        Args:
            value: Absolute path to the dataset root directory.

        Raises:
            FileNotFoundError: If the ``root`` directory does not exist.
        """
        root = Path(value).normalize()  # Ensure an absolute, clean path

        # Logic for subset appending
        if is_valid_str(self.subset):
            # Check if current root ends with subset; if not, try to append
            if root.name != self.subset:
                sub_path = root / self.subset
                if sub_path.is_dir():
                    root = sub_path

        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root not found at: {root}")

        self.root = root

    def set_split(self, split: Split | str):
        """Set the current dataset split.

        Args:
            split: Data split subset to use. Must be one of ``Split``.

        Raises:
            ValueError: If ``split`` is not one of the supported splits.
        """
        # Cast to Enum if it's a string
        split = Split(split)
        if split not in self.splits:
            raise ValueError(f"Unsupported 'split': {split}. Must be one of: {self.splits}.")

        self.split = split

    @property
    def split_str(self) -> str:
        """Return the current dataset split as a string."""
        return self._split.value


class InputTargetLoadMixin(DataLoadMixin, abc.ABC):
    """Input and target directory data loading mixin.

    Extend ``DataLoadMixin`` by introducing ``input_dir`` and ``target_dir``
    attributes, along with validation for these attributes.

    Attributes:
        input_dir: Absolute path to the input directory.
        target_dir: Absolute path to the target directory.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        input_dir : Path | str,
        target_dir: Path | str,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            input_dir: Absolute path to the input directory.
            target_dir: Absolute path to the target directory.
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        # Assign attributes
        # Initialize attributes to None first to avoid AttributeError
        # during setter logic if super().__init__ triggers something
        self.input_dir  = None
        self.target_dir = None
        self.set_input_dir(input_dir)
        self.set_target_dir(target_dir)

        # Continue the initialization chain
        super().__init__(*args, **kwargs)

    # --- Properties ---
    def set_input_dir(self, value: Path | str):
        """Set the input directory.

        Args:
            value: Path to the input directory.

        Raises:
            TypeError: If ``value`` is None.
            FileNotFoundError: If the ``input_dir`` directory does not exist.
        """
        if value is None:
            raise TypeError(
                f"Expected 'input_dir' to be a Path or str, "
                f"but got {type(value).__name__}."
            )

        input_dir = Path(value).normalize(exist=True)
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found at: {input_dir}")

        self.input_dir = input_dir

    def set_target_dir(self, value: Path | str | None):
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
            self.target_dir = target_dir
        else:
            self.target_dir = None

    @property
    def has_target(self) -> bool:
        """Check if the dataset has target data."""
        return self.target_dir is not None and self.target_dir.is_dir()

    @property
    def label_dir(self) -> Path | None:
        """An alias to ``target_dir`` for better readability in certain contexts."""
        return self.target_dir


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
        subset: Name of the dataset's subset directory. `Should be defined in
            subclasses.`
        splits: List of supported splits. `Should be defined in subclasses.`
        modalities: Dictionary defining the dataset modalities. `Must be defined
            in subclasses.`
    """

    subset    : str | None  = None
    splits    : list[Split] = []
    modalities: Modalities  = {}

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
        for attr in ["modalities"]:
            if not hasattr(cls, attr):  #  or getattr(cls, attr) is None:
                raise TypeError(
                    f"Class {cls.__name__} must define '{attr}' attribute "
                    f"(defined locally or inherited)."
                )

    # --- Properties ---
    @property
    def primary_modality(self) -> tuple[str, Modality]:
        """Return the primary modality.

        Raises:
            ValueError: If no primary modality is defined.
        """
        try:
            return next((k, v) for k, v in self.modalities.items() if v.primary)
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
            k: [] for k, v in self.modalities.items()
            if v.type and v.module and (
                (v.train is not False if self.split in [Split.TRAIN, Split.VAL]
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
        pattern   = self.root / self.split_str / pk_name

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
        pk_name      = self.modalities[pk_key].name
        target       = self.modalities[key]
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


# ==============================================================================
# region TRANSFORMATION
# ==============================================================================

# --- Casting ---


# --- Encoding ---


# --- Standardization ---


# --- Structural ---

class BatchCollateMixin:
    """Batch collation mixin for data containers.

    Define the collate function for batching datapoints when using with a
    DataLoader.
    """

    # noinspection PyTypeChecker
    @staticmethod
    def collate_fn(batch: list[dict]) -> dict[str, Any]:
        """Collate a batch of input items.

        Args:
            batch: List of dictionaries, where each dictionary is a datapoint.

        Returns:
            Collated dictionary for torch.utils.data.dataset.DataLoader.
        """
        if not isinstance(batch, list):
            raise TypeError(f"Expected 'batch' to be a list, but got {type(batch).__name__}.")
        if not batch:
            return {}

        # Faster Transposition: Group items by key
        # This replaces the complex zip(*[b.values()]) logic
        keys = batch[0].keys()
        # Validates batch item types and key sets
        for i, d in enumerate(batch):
            if not isinstance(d, dict):
                raise TypeError(
                    f"Expected 'batch' item at index {i} to be a dict, "
                    f"but got {type(d).__name__}."
                )
            if set(d.keys()) != set(keys):
                raise ValueError(
                    f"Expected 'batch' item at index {i} to have keys {set(keys)}, "
                    f"but got {set(d.keys())}."
                )

        collated = {k: [d[k] for d in batch] for k in keys}

        for k, v in collated.items():
            # Immutability for metadata
            if k == "meta":
                collated[k] = tuple(v)  # Freeze metadata to prevent runtime modification
                continue

            # Performance: O(1) Type Checking
            # We check only the first element, assuming batch homogeneity
            first_item = v[0]

            if first_item is None:
                collated[k] = None
            elif isinstance(first_item, torch.Tensor):
                collated[k] = torch.stack(v, dim=0)
            elif isinstance(first_item, np.ndarray):
                collated[k] = np.stack(v, axis=0)
            # v remains a list for other types (like strings or custom objects)

        return collated


# --- Statistical ---


# --- Geometric ---


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
