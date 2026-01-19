#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Array-like data structures.

This module provides base classes and mixins for array-like data.
"""

from __future__ import annotations

__all__ = [
    "TensorOrArray",
]

import numpy as np
import torch

from ..base import Data, DeviceManagementMixin


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
        """Return an item at the given ``index``.

        Args:
            index: Index to access.

        Returns:
            Item at the given ``index``.
        """
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
            TypeError: If ``value`` is not a torch.Tensor or numpy.ndarray.
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
        """Move or cast data to a specific device.

        Move or cast data to a specific device (cpu, cuda, mps, etc.), ensuring
        numpy.ndarray input is upgraded to torch.Tensor.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Data moved or cast to a specific device.
        """
        new_data = self.data
        if isinstance(new_data, np.ndarray):
            new_data = torch.as_tensor(new_data)
        return self.__class__(new_data.to(*args, **kwargs))

    def cpu(self) -> TensorOrArray:
        """Move data to CPU.

        Returns:
            Data moved to CPU.
        """
        if isinstance(self.data, np.ndarray):
            return self
        return self.__class__(self.data.cpu())

    def cuda(self) -> TensorOrArray:
        """Move data to GPU.

        Move data to GPU, ensuring numpy.ndarray arrays are converted to
        torch.Tensor.

        Returns:
            Data moved to GPU.
        """
        if isinstance(self.data, np.ndarray):
            # torch.as_tensor is safer than torch.tensor as it avoids copying if possible
            return self.__class__(torch.as_tensor(self.data).cuda())
        return self.__class__(self.data.cuda())

    def mps(self) -> TensorOrArray:
        """Move data to MPS.

        Returns:
            Data moved to MPS.
        """
        if isinstance(self.data, np.ndarray):
            return self.__class__(torch.as_tensor(self.data).to("mps"))
        return self.__class__(self.data.to("mps"))

    def numpy(self) -> TensorOrArray:
        """Convert data to numpy.ndarray.

        Returns:
            Data converted to numpy.ndarray.
        """
        if isinstance(self.data, np.ndarray):
            return self
        # .detach() is vital if the tensor is part of a computation graph
        return self.__class__(self.data.detach().cpu().numpy())

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
