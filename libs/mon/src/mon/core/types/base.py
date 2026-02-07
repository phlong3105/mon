#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Abstract base classes and mixins.

This module provides abstract base classes and mixins for data types.
"""

from __future__ import annotations

__all__ = [
    "Data",
    "DeviceManagementMixin",
    "PersistentData",
]

import abc
from typing import Any, Iterator

import torch

from mon.core.pathlib import Path


# ==============================================================================
# region CONSTANTS
# ==============================================================================


# endregion


# ==============================================================================
# region TYPE DEFINITIONS & PROTOCOLS
# ==============================================================================

# --- Type Aliases ---


# --- Protocols ---


# endregion


# ==============================================================================
# region BASE CLASSES & MIXINS
# ==============================================================================

# --- Base Classes ---

class Data(abc.ABC):
    """Abstract base class for complex data objects.

    Attributes:
        _data (Any): Underlying data.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, data: Any, *args, **kwargs):
        """Initialize a new instance.

        Args:
            data: Underlying data.
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        self._data = data

    # --- Representation ---
    def __repr__(self) -> str:
        """Return the official string representation for developers."""
        return f"{self.__class__.__name__}(shape={self.shape}, type={type(self.data)})"

    # --- Container / Sequence Methods ---
    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the length of the container.

        Returns:
            Length of the container.
        """
        pass

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Return an item at the given ``index``.

        Args:
            index: Index to access.
        """
        pass

    def __iter__(self) -> Iterator:
        """Return an iterator for the container.

        Returns:
            Iterator for the container.
        """
        for i in range(len(self)):
            yield self[i]

    # --- Properties ---
    @property
    @abc.abstractmethod
    def data(self) -> Any:
        """Return the underlying data."""
        pass

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Return the data shape."""
        pass

    @property
    @abc.abstractmethod
    def meta(self) -> dict:
        """Return metadata describing the data."""
        pass


class PersistentData(Data, abc.ABC):
    """Abstract base class for data that can be loaded from and saved to disk.

    Extend ``Data`` to add persistence capabilities and lazy loading.

    Attributes:
        _path (Path | str | None): Path to load data from.
        _root (Path | str | None): Root directory for relative paths. Defaults
            to None.
        _persist (bool): If True, persist loaded data in memory. Defaults to
            False.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        data   : Any,
        path   : Path | str | None,
        root   : Path | str | None = None,
        persist: bool              = False,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            data: Underlying data.
            path: Path to load data from.
            root: Root directory for relative paths. Defaults to None.
            persist: If True, persist loaded data in memory. Defaults to False.
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        # Validate and set paths and persistence flag
        self._path    = Path(path).normalize(exist=True) if path else None
        self._root    = Path(root).normalize(exist=True) if root else None
        self._persist = persist

        # Continue the initialization chain
        super().__init__(data=data, *args, **kwargs)

    # --- Properties ---
    @property
    def data(self) -> Any:
        """Return the underlying data."""
        if self._persist:
            if self._data is None:
                self._data = self.load()
            return self._data
        return self.load()

    @property
    def path(self) -> Path:
        """Return the path to the data file."""
        return self._path

    @property
    def root(self) -> Path:
        """Return the root directory for relative paths."""
        return self._root

    @property
    def persist(self) -> bool:
        """Return whether the loaded data is persisted in memory."""
        return self._persist

    @persist.setter
    def persist(self, value: bool):
        """Set whether loaded data is persisted in memory.

        Args:
            value: If True, persist loaded data in memory. Else, clear it.
        """
        self._persist = value
        if not self._persist:
            self.clear()

    # --- Data Loading ---
    @abc.abstractmethod
    def load(self, reload: bool = False) -> Any:
        """Load data from disk to memory.

        Args:
            reload: If True, force reloading even if data is already in memory.
                Defaults to False.

        Returns:
            Loaded data object.
        """
        pass

    def clear(self):
        """Clear the loaded data from memory if not persisting."""
        if (
            not self._persist
            and (self._path and self._path.exists())
        ):
            self._data = None


# --- Mixins ---

class DeviceManagementMixin(abc.ABC):
    """Mixin for device management operations."""

    @abc.abstractmethod
    def to(self, device: str | torch.device, *args, **kwargs) -> Any:
        """Move or cast data to a specific device.

        Args:
            device: Target device.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Data moved or cast to the target device.
        """
        pass

    @abc.abstractmethod
    def cpu(self) -> Any:
        """Move data to CPU.

        Returns:
            Data moved to CPU.
        """
        pass

    @abc.abstractmethod
    def cuda(self) -> Any:
        """Move data to GPU.

        Returns:
            Data moved to GPU.
        """
        pass

    @abc.abstractmethod
    def mps(self) -> Any:
        """Move data to MPS.

        Returns:
            Data moved to MPS.
        """
        pass

    @abc.abstractmethod
    def numpy(self) -> Any:
        """Convert data to numpy.ndarray.

        Returns:
            Data converted to numpy.ndarray.
        """
        pass

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
