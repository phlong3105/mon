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

import os
from abc import ABC, abstractmethod
from typing import Any, OrderedDict, override

import numpy as np
import torch
from torch import Tensor

from mon.core import create_progress_bar, is_valid_str, Path, Split
from ..base import Modality


# ==============================================================================
# region INPUT
# ==============================================================================

class DataLoadMixin(ABC):
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
        """Load all datapoints in the dataset from the disk. After calling this,
        ``datapoints`` will be populated.

        Raises:
            TypeError: If ``_load_data()`` does not return a dictionary.
        """
        # Ensure the container exists
        if not hasattr(self, "datapoints"):
            self.datapoints = {}

        self._on_load_start()

        # Core loading
        datapoints = self._load_data()
        if not isinstance(datapoints, dict):
            raise TypeError(
                f"Expected '_load_data' to return a dict, but got "
                f"{type(datapoints).__name__}."
            )

        self.datapoints = datapoints
        self.datapoints = datapoints

        self._on_load_end()

        # Auto-verify after loading is complete
        self.verify()

    @abstractmethod
    def _load_data(self) -> dict[str, list[Any]]:
        """Load the core data of the dataset.

        Returns:
            dict[str, list[Any]]: Dictionary containing lists of datapoints for
                each modality.
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


class RootLoadMixin(DataLoadMixin, ABC):
    """Root directory data loading mixin.

    Extend ``DataLoadMixin`` by introducing ``root`` and ``split`` attributes,
    along with validation for these attributes.

    Attributes:
        subroot (str, optional): Name of the subdirectory within the dataset's
            ``root`` (i.e., ``root/subroot``). Use this if the current dataset
            is a subset of another dataset. `Should be defined in subclasses.`
        splits (list[Split]): List of supported splits. `Must be defined in
            subclasses.`
    """

    subroot: str = ""
    splits: list[Split]

    # --- Lifecycle & Initialization ---
    def __init__(self, root: Path | str, split: Split | str, *args, **kwargs):
        """Initialize a new instance.

        Args:
            root (Path | str): Absolute path to the dataset root directory.
            split (Split | str): Data split subset to use. Must be one of the
                supported ``splits``.
        """
        # Assign attributes
        self.root = root
        self.split = split

        # Continue the initialization chain
        super().__init__(*args, **kwargs)

    def __init_subclass__(cls, *args, **kwargs):
        """Validate subclass attributes on inheritance.

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
    @property
    def root(self) -> Path:
        """Return the dataset root directory."""
        return self._root

    @root.setter
    def root(self, root: Path | str):
        """Set the dataset root directory.

        Args:
            root (Path | str): Absolute path to the dataset root directory.

        Raises:
            FileNotFoundError: If the ``root`` directory does not exist.
        """
        root = Path(root).normalize()  # Ensure an absolute, clean path

        # Logic for subset appending
        if is_valid_str(self.subroot):
            # Check if current root ends with subset; if not, try to append
            if root.name != self.subroot:
                sub_path = root / self.subroot
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
            split (Split | str): Data split subset to use. Must be one of the
                supported ``splits``.

        Raises:
            ValueError: If ``split`` is not one of the supported splits.
        """
        # Cast to Enum if it's a string
        split = Split(split)
        if split not in self.splits:
            raise ValueError(
                f"Unsupported 'split': {split}. Must be one of: {self.splits}."
            )

        self._split = split

    @property
    def split_str(self) -> str:
        """Return the string representation of the current split."""
        return self.split.value


class InputTargetLoadMixin(DataLoadMixin, ABC):
    """Input and target directory data loading mixin.

    Extend ``DataLoadMixin`` by introducing ``input_dir`` and ``target_dir``
    attributes, along with validation for these attributes.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        input_dir: Path | str,
        target_dir: Path | str,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            input_dir (Path | str): Absolute path to the input directory.
            target_dir (Path | str): Absolute path to the target directory.
        """
        # Assign attributes
        self.input_dir = input_dir
        self.target_dir = target_dir

        # Continue the initialization chain
        super().__init__(*args, **kwargs)

    # --- Properties ---
    @property
    def input_dir(self) -> Path:
        """Return the input directory."""
        return self._input_dir

    @input_dir.setter
    def input_dir(self, input_dir: Path | str):
        """Set the input directory.

        Args:
            input_dir (Path | str): Absolute path to the input directory.

        Raises:
            TypeError: If ``input_dir`` is None.
            FileNotFoundError: If the ``input_dir`` directory does not exist.
        """
        if input_dir is None:
            raise TypeError(
                f"Expected 'input_dir' to be a Path or str, "
                f"but got {type(input_dir).__name__}."
            )

        input_dir = Path(input_dir).normalize(exist=True)
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found at: {input_dir}")

        self._input_dir = input_dir

    @property
    def target_dir(self) -> Path | None:
        """Return the target directory."""
        return self._target_dir

    @target_dir.setter
    def target_dir(self, target_dir: Path | str | None):
        """Set the target directory.

        Args:
            target_dir (Path | str, optional): Absolute path to the target directory.

        Raises:
            FileNotFoundError: If the ``target_dir`` directory does not exist.
        """
        if target_dir is not None:
            target_dir = Path(target_dir).normalize(exist=True)
            if not target_dir.is_dir():
                raise FileNotFoundError(
                    f"Target directory not found at: {target_dir}"
                )
            self._target_dir = target_dir
        else:
            self._target_dir = None

    @property
    def has_target(self) -> bool:
        """Check if the dataset has target data."""
        return isinstance(self.target_dir, Path) and self.target_dir.is_dir()

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
        subroot (str, optional): Name of the subdirectory within the dataset's
            ``root`` (i.e., ``root/subroot``). Use this if the current dataset
            is a subset of another dataset. `Should be defined in subclasses.`
        splits (list[Split]): List of supported splits. `Must be defined in
            subclasses.`
        modalities (OrderedDict[str, Modality]): Dictionary defining the dataset
            modalities. The first key in the dictionary is the primary modality,
            which is loaded first and used to guide the loading of other
            modalities. `Must be defined in subclasses.`
    """

    subroot: str = ""
    splits: list[Split]
    modalities: OrderedDict[str, Modality]

    # --- Lifecycle & Initialization ---
    def __init_subclass__(cls, *args, **kwargs):
        """Validate subclass attributes on inheritance.

        Raises:
            TypeError: If ``modalities`` is not defined in the subclass.
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
        """Return the primary modality's key and ``Modality`` object.

        Raises:
            ValueError: If no primary modality is defined.
        """
        try:
            pk = next(iter(self.modalities))
            pm = self.modalities[pk]
            return pk, pm
        except StopIteration:
            raise ValueError("No primary modality defined in 'modalities'.")

    @property
    def new_datapoints(self) -> dict[str, list[Any]]:
        """Return an empty datapoints dictionary with modalities as keys."""
        datapoints = {}
        for k, v in self.modalities.items():
            if v.type and v.module:
                if self.split in [Split.TEST] and not v.test:
                    continue
                datapoints[k] = []
        return datapoints

    # --- Data Loading ---
    @override
    def _load_data(self) -> dict[str, list[Any]]:
        """Load the core data of the dataset.

        Returns:
            dict[str, list[Any]]: Dictionary containing lists of datapoints for
                each modality.
        """
        # Initialize empty datapoints dictionary with modalities
        datapoints = self.new_datapoints

        # Load primary modality
        pk, _ = self.primary_modality
        pk_data = self._load_primary_data()
        datapoints[pk] = pk_data

        # Load other modalities using the cached primary_data and key
        for k, v in datapoints.items():
            if k != pk:
                datapoints[k] = self._load_modality_data(k, pk_data)

        # List metadata
        datapoints["meta"] = [d.meta for d in pk_data]
        return datapoints

    def _load_primary_data(self) -> list[Any]:
        """Load primary modality data files in the dataset.

        Returns:
            list[Any]: List of primary modality data files.
        """
        disable_pbar = getattr(self, "disable_pbar", False)

        pk, pm = self.primary_modality
        pattern = self.root / self.split_str / pm.name

        files = []
        with create_progress_bar(disable=disable_pbar) as pbar:
            paths = sorted(pattern.rglob("*"))
            desc = f"Listing {self.__class__.__name__} {self.split_str} {pk}(s)"
            for path in pbar.track(sequence=paths, description=desc):
                if path.is_image_file():
                    files.append(pm.module(data=path, root=pattern))

        return files

    def _load_modality_data(self, key: str, pk_data: list[Any]) -> list[Any]:
        """Load modality data files in the dataset.

        Args:
            key (str): Modality key.
            pk_data (list[Any]): List of primary modality data files.

        Returns:
            list[Any]: List of modality data files.
        """
        disable_pbar = getattr(self, "disable_pbar", False)

        # Pre-calculate strings and lookup modules outside the loop
        pk, pm = self.primary_modality
        target = self.modalities[key]
        name, module = target.name, target.module

        old_part = f"{os.sep}{pm.name}{os.sep}"
        new_part = f"{os.sep}{name}{os.sep}"

        files = []
        with create_progress_bar(disable=disable_pbar) as pbar:
            desc = f"Listing {self.__class__.__name__} {self.split_str} {key}(s)"
            for file in pbar.track(sequence=pk_data, description=desc):
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
            batch (list[dict]): List of datapoints to collate. Each datapoint
                is expected to be a dictionary with consistent keys.

        Returns:
            dict[str, Any]: Collated batch where each key maps to a batched
                value. Tensors and NumPy arrays are stacked, while other types
                are collected into lists. The "meta" key is converted to a
                tuple to ensure immutability.
        """
        if not isinstance(batch, list):
            raise TypeError(
                f"Expected 'batch' to be a list, but got {type(batch).__name__}."
            )
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
                    f"Expected 'batch' item at index {i} to have keys "
                    f"{set(keys)}, but got {set(d.keys())}."
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
            elif isinstance(first_item, Tensor):
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
