#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Array-like base classes and mixins.

This module provides the base classes and mixins for array-like data.
"""

from __future__ import annotations

__all__ = [
    "TensorOrArray",
]

import numpy as np
import torch

from mon.core.dtypes.base import Data, DeviceManagementMixin


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


# --- Mixins ---


# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

class TensorOrArray(Data, DeviceManagementMixin):
    """Tensor-like or ndarray-like data structure.

    Extend Data to handle either torch.Tensor or numpy.ndarray and provide
    properties and methods related to both data types.

    Attributes:
        _data (torch.Tensor | numpy.ndarray): Underlying data object.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(self, data: np.ndarray | torch.Tensor, *args, **kwargs):
        """Initialize a new instance.

        Args:
            data: Either a torch.Tensor or numpy.ndarray.
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        # Call the setter to ensure type validation on init
        self.data = data
        
        # Continue the initialization chain
        super().__init__(data=self.data, *args, **kwargs)

    # --- Container / Sequence Methods ---
    def __len__(self) -> int:
        """Return the length of the container."""
        return len(self.data)

    def __getitem__(self, index: int | slice | list[int] | np.ndarray | torch.Tensor) -> TensorOrArray:
        """Define behavior for when an item is accessed via the notation self[index]."""
        item = self.data[index]
        # Handle numpy scalars which are not ndarray instances
        if isinstance(item, (np.generic, int, float, bool)) and not isinstance(item, np.ndarray):
             item = np.array(item)
        return self.__class__(item)
    
    # --- Properties ---
    @property
    def data(self) -> np.ndarray | torch.Tensor:
        """Return the underlying data."""
        return self._data

    @data.setter
    def data(self, value: np.ndarray | torch.Tensor):
        """Set the underlying data.

        Args:
            value: New data to store.

        Raises:
            TypeError: If ``data`` is not a torch.Tensor or numpy.ndarray.
        """
        if not isinstance(value, (np.ndarray, torch.Tensor)):
            raise TypeError(
                f"Expected 'value' to be a torch.Tensor or numpy.ndarray, "
                f"but got {type(value).__name__}."
            )
        self._data = value
    
    @property
    def shape(self) -> tuple[int, ...]:
        """Return the data shape."""
        return self.data.shape
    
    @property
    def meta(self) -> dict:
        """Return metadata describing the data."""
        return {
            "shape": self.shape,
            "dtype": self.data.dtype,
            "type" : type(self.data),
        }
    
    # --- Device Management ---
    def to(self, *args, **kwargs) -> TensorOrArray:
        """Move or cast data to a specific device (cpu, cuda, mps, etc.),
        ensuring NumPy input is upgraded to Tensor.
        """
        new_data = self.data
        if isinstance(new_data, np.ndarray):
            new_data = torch.as_tensor(new_data)
        return self.__class__(new_data.to(*args, **kwargs))

    def cpu(self) -> TensorOrArray:
        """Move data to CPU."""
        if isinstance(self.data, np.ndarray):
            return self
        return self.__class__(self.data.cpu())
    
    def cuda(self) -> TensorOrArray:
        """Move data to GPU, ensuring NumPy arrays are converted to Tensors."""
        if isinstance(self.data, np.ndarray):
            # torch.as_tensor is safer than torch.tensor as it avoids copying if possible
            return self.__class__(torch.as_tensor(self.data).cuda())
        return self.__class__(self.data.cuda())
    
    def mps(self) -> TensorOrArray:
        """Move data to MPS."""
        if isinstance(self.data, np.ndarray):
            return self.__class__(torch.as_tensor(self.data).to("mps"))
        return self.__class__(self.data.to("mps"))
    
    def numpy(self) -> TensorOrArray:
        """Convert data to numpy."""
        if isinstance(self.data, np.ndarray):
            return self
        # .detach() is vital if the tensor is part of a computation graph
        return self.__class__(self.data.detach().cpu().numpy())

# endregion
