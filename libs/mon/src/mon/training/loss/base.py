#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base classes and mixins for loss functions.

This module provides base classes and mixins for loss functions.
"""

from __future__ import annotations

__all__ = [
    "BaseLoss",
]

from abc import ABC, abstractmethod
from typing import override

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

from mon.core import depascalize


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

class BaseLoss(_Loss, ABC):
    """Loss function base class."""

    # --- Lifecycle & Initialization ---
    def __init__(self, reduction: str = "mean"):
        """Initialize a new instance.

        Args:
            reduction (str): Reduction method to apply to the loss. One of:
                ["mean", "sum", "none"]. Defaults to "mean".

        Raises:
            ValueError: If the provided ``reduction`` method is not supported.
        """
        super().__init__(reduction=reduction)
        # Assign the function once to avoid repeated dict lookups
        self._reduce_fn = {
            "mean": torch.mean,
            "sum": torch.sum,
            "none": lambda x: x,
        }[reduction]

    # --- Representation ---
    @override
    def __str__(self):
        """Return the string representation of the loss class."""
        return depascalize(self.__class__.__name__).lower()

    # --- Callable & Context Manager ---
    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """Calculate the loss.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Tensor: Loss value.
        """
        pass

    def reduce(self, loss: Tensor) -> Tensor:
        """Reduce the loss tensor according to the specified reduction method.

        Args:
            loss (Tensor): Loss tensor to reduce.

        Returns:
            Tensor: Reduced loss tensor.
        """
        return self._reduce_fn(loss)


# --- Mixins ---


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
