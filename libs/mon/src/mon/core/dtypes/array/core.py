#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Array-like base classes and mixins.

This module provides the base classes and mixins for array-like data types
which can be a torch.Tensor or numpy.ndarray.
"""

__all__ = [
    "TensorOrArray",
]

import numpy as np
import torch

from ..base import Data, DeviceManagementMixin


# ==============================================================================
# TYPE DEFINITIONS & PROTOCOLS (Interfaces)
# ==============================================================================

# --- Type Aliases ---


# --- Structural Protocols ---


# ==============================================================================
# BASE CLASSES & MIXINS (Behaviors)
# ==============================================================================

# --- Structural Bases ---


# --- Lifecycle Mixins ---


# --- Compute Mixins ---


# ==============================================================================
# CONCRETE IMPLEMENTATIONS (The Concrete Classes)
# ==============================================================================

# --- Primary Data Types ---
class TensorOrArray(Data, DeviceManagementMixin):
    """A basic class for tensor-like or ndarray-like data types.
    
    Extend Data to handle either torch.Tensor or numpy.ndarray and provide
    properties and methods related to both data types.
    
    Attributes:
        _data (np.ndarray | torch.Tensor): Either a torch.Tensor or numpy.ndarray.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(self, data: np.ndarray | torch.Tensor, *args, **kwargs):
        """Initialize a new instance.

        Args:
            data: Either a torch.Tensor or numpy.ndarray.
        """
        # Initialize parent classes and assign attributes
        super().__init__(data=data)  # This will call the data setter

    # --- Container / Sequence Methods ---
    def __len__(self) -> int:
        """Return the logical length."""
        return len(self.data)

    def __getitem__(self, idx: int | list[int] | torch.Tensor) -> "TensorOrArray":
        """Return element(s) at the given index.

        Args:
            idx: Index or slice to select from the underlying data.
        """
        return self.__class__(self.data[idx])

    # --- Properties ---
    @property
    def data(self) -> np.ndarray | torch.Tensor:
        """Return the underlying data."""
        return self._data

    @data.setter
    def data(self, data: np.ndarray | torch.Tensor):
        """Set the underlying data.

        Args:
            data: New data to store.

        Raises:
            TypeError: If ``data`` is not a torch.Tensor or numpy.ndarray.
        """
        if not isinstance(data, (torch.Tensor, np.ndarray)):
            raise TypeError(f"``data`` must be a torch.Tensor or numpy.ndarray, got {type(data)}.")
        self._data = data

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the data shape."""
        return self.data.shape

    @property
    def meta(self) -> dict:
        """Return the metadata."""
        return {
            "shape": self.shape,
            "dtype": self.data.dtype,
            "type" : type(self.data),
        }

    # --- Device Management ---
    def cpu(self) -> "TensorOrArray":
        """Move data to CPU."""
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu())

    def cuda(self) -> "TensorOrArray":
        """Move data to GPU."""
        if isinstance(self.data, np.ndarray):
            return self.__class__(torch.as_tensor(self.data).cuda())
        else:
            return self.__class__(self.data.cuda())

    def numpy(self) -> "TensorOrArray":
        """Convert data to numpy."""
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy())

    def to(self, *args, **kwargs) -> "TensorOrArray":
        """Move or cast data."""
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs))

