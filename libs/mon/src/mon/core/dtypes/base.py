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
    
    def __init__(self, data: Any):
        """Initialize the Data object.

        Args:
            data: Underlying data.
        """
        self._data = data
    
    # --- Magic Methods ---
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
    """A mixin for data loading operations (i.e., reading from disk to memory
    + parsing).
    """
    
    @abc.abstractmethod
    def load(self, reload: bool = False) -> Any:
        """Load data from disk to memory (i.e., reading from disk and parsing).

        Args:
            reload: If True, force reloading even if data is already in memory.
            
        Returns:
            The loaded data object.
        """
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
