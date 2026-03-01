#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data Structures.

This module provides data structures for handling various data types.
"""

from __future__ import annotations

__all__ = [
    "Data",
    "DeviceManagementMixin",
    "Loader",
    "Metadata",
    "MetadataDictList",
]

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Union

import torch
from numpy import ndarray

from mon.core.base import DictList
from mon.core.path import Path
from mon.core.typing import IntOrTuple
from mon.core.utils import is_valid_str


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

# --- Data & Metadata ---

@dataclass
class Data(ABC):
    """Base class for data structures."""

    # --- Representation ---
    def __repr__(self) -> str:
        """Return the official string representation for developers."""
        return f"{self.__class__.__name__}(shape={self.shape}, type={type(self.data)})"

    # --- Container / Sequence Methods ---
    @abstractmethod
    def __len__(self) -> int:
        """Return the length of the container."""
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Return the element at the given ``index``."""
        pass

    # --- Properties ---
    @property
    @abstractmethod
    def data(self) -> Any:
        """Return the underlying data."""
        pass

    @property
    @abstractmethod
    def shape(self) -> IntOrTuple:
        """Return the data shape."""
        pass

    @property
    @abstractmethod
    def meta(self) -> dict:
        """Return metadata describing the data."""
        pass


@dataclass
class Metadata:
    """Data structure that stores metadata about the ``Data``.

    Hold metadata about the data during the discovery process without loading
    the actual data (to save memory). Later the actual data can be loaded and
    the ``Data`` object can be initialized.

    Attributes:
        path (Path): Path to the data file.
        base_dir (Path, optional): Base directory for relative paths. This is
            useful to resolve other files related to the data. Defaults to None.
        shape (int_any_t, optional): Shape of the data (if known). This is used
            to initialize an empty tensor or array of the correct shape.
            Defaults to None.
    """

    path: Path
    base_dir: Path | None = None
    shape: tuple[int, ...] | None = None

    # --- Lifecycle & Initialization ---
    def __post_init__(self):
        """Perform post-initialization tasks."""
        # Validate inputs
        if is_valid_str(self.path):
            self.path = Path(self.path).normalize()
        if not self.path.exists():
            raise ValueError(f"Path not found at: '{self.path}'")

        if is_valid_str(self.base_dir):
            self.base_dir = Path(self.base_dir).normalize()
            if not self.base_dir.exists():
                raise ValueError(f"Base directory not found at: '{self.base_dir}'")


class MetadataDictList(DictList[str, list[Metadata]]):
    """A dictionary that stores lists of ``Metadata`` instances.

    Extend ``DictList`` to provide dictionary-like access to lists of ``Metadata``
    instances by their string keys.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        data: dict[str, Union[Metadata | list[Metadata]]] | None = None,
        **kwargs
    ):
        """Initialize a new instance.

        Args:
            data (dict[str, list[Metadata]], optional): Initial data for the
                dictionary. Defaults to None.
            **kwargs: Additional key-value pairs to initialize.
        """
        # We hardcode the item_type
        # Users just call MetadataDictList() without arguments
        _ = kwargs.pop("item_type", Metadata)
        super().__init__(item_type=Metadata, data=data, **kwargs)


# --- Loader ---

class Loader(ABC):
    """Base class for all data loaders.

    Notes: Each ``Data`` should have a corresponding ``Loader`` that can be
    used to load it.
    """

    # --- Callable & Context Manager ---
    def __call__(self, metadata: Metadata, *args, **kwargs) -> Data | None:
        """Allow the instance to be called like a function."""
        return self.load(metadata=metadata, *args, **kwargs)

    # --- Input ---
    @abstractmethod
    def load(self, metadata: Metadata, *args, **kwargs) -> Data | None:
        """Load data from the given ``metadata``.

        Args:
            metadata (Metadata): Metadata describing the data to be loaded.
            *args: Positional arguments to forward to the loading process.
            **kwargs: Keyword arguments to forward to the loading process.

        Returns:
            Data | None: A ``Data`` instance containing the loaded data,
                or None if loading fails.
        """
        pass

# endregion


# ==============================================================================
# region MIXINS
# ==============================================================================

class DeviceManagementMixin(ABC):
    """Mixin for device management operations."""

    @abstractmethod
    def to(self, device: str | torch.device, *args, **kwargs) -> Any:
        """Move or cast data to a specific device.

        Args:
            device: Target device.
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        pass

    @abstractmethod
    def cpu(self) -> Any:
        """Move data to CPU."""
        pass

    @abstractmethod
    def cuda(self) -> Any:
        """Move data to GPU."""
        pass

    @abstractmethod
    def mps(self) -> Any:
        """Move data to MPS."""
        pass

    @abstractmethod
    def numpy(self) -> ndarray:
        """Convert data to a NumPy array."""
        pass

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
