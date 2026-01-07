#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Abstract classes and mixins.

This module provides the abstract base classes and mixins for data types.
"""

__all__ = [
    "Data",
    "DataLoadMixin",
    "DeviceManagementMixin",
]

import abc
from typing import Any

from mon.core.pathlib import Path


# ==============================================================================
# TYPE DEFINITIONS & PROTOCOLS (Interfaces)
# ==============================================================================

# --- Type Aliases ---


# --- Structural Protocols ---


# ==============================================================================
# BASE CLASSES & MIXINS (Behaviors)
# ==============================================================================

# --- Structural Bases ---
class Data(abc.ABC):
    """An abstract base class for complex data objects.

    Attributes:
        _data (Any): Underlying data object.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(self, data: Any, *args, **kwargs):
        """Initialize a new instance.

        Args:
            data: Underlying data.
        """
        self._data = data
    
    # --- Container / Sequence Methods ---
    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the logical length."""
        pass

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """Return element(s) at the given index.

        Args:
            idx: Index or slice to select from the underlying data.
        """
        pass

    # --- Properties ---
    @property
    @abc.abstractmethod
    def data(self) -> Any:
        """Return the underlying data object."""
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


# --- Lifecycle Mixins ---
class DataLoadMixin(abc.ABC):
    """A mixin for data loading operations (i.e., reading from disk and parsing).
    
    Attributes:
        _path (Path): Path to load data from.
        _root (Path): Root directory for relative paths.
        _persist (bool): Whether to persist loaded data in memory. Defaults to False.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        path   : Path,
        root   : Path = None,
        persist: bool = False
    ):
        """Initialize a new instance.

        Args:
            path: Path to load data from.
            root: Root directory for relative paths. Defaults to None.
            persist: If True, persist loaded data in memory. Defaults to False.
        """
        path = Path(path) if path is not None else None
        root = Path(root) if root is not None else None
        if path is not None and not path.exists():
            raise FileNotFoundError(f"``path`` does not exist: {path}.")
        if root is not None and not root.exists():
            raise FileNotFoundError(f"``root`` does not exist: {root}.")
            
        self._path    = path
        self._root    = root
        self._persist = persist
    
    # --- Properties ---
    @property
    def path(self) -> Path:
        """Return the path to load data from."""
        return self._path
    
    @property
    def root(self) -> Path:
        """Return the root directory for relative paths."""
        return self._root
    
    @property
    def persist(self) -> bool:
        """Return whether loaded data is persisted in memory."""
        return self._persist
    
    @persist.setter
    def persist(self, persist: bool):
        """Set whether loaded data is persisted in memory.

        Args:
            persist: If True, persist loaded data in memory.
        """
        self._persist = persist
        if not self._persist:
            self.clear()
    
    # --- Data Loading ---
    @abc.abstractmethod
    def load(self, reload: bool = False) -> Any:
        """Load data from disk to memory (i.e., reading from disk and parsing).

        Args:
            reload: If True, force reloading even if data is already in memory.
            
        Returns:
            The loaded data object.
        """
        pass
    
    @abc.abstractmethod
    def clear(self):
        """Clear the loaded data from memory."""
        pass


# --- Compute Mixins ---
class DeviceManagementMixin(abc.ABC):
    """A mixin for device management operations."""

    @abc.abstractmethod
    def cpu(self) -> Any:
        """Move data to CPU."""
        pass

    @abc.abstractmethod
    def cuda(self) -> Any:
        """Move data to GPU."""
        pass

    @abc.abstractmethod
    def numpy(self) -> Any:
        """Convert data to numpy."""
        pass

    @abc.abstractmethod
    def to(self, *args, **kwargs) -> Any:
        """Move or cast data."""
        pass


# ==============================================================================
# CONCRETE IMPLEMENTATIONS (The Concrete Classes)
# ==============================================================================

# --- Primary Data Types ---
