#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base classes and mixins for loss functions.

This module provides base classes and mixins for loss functions.
"""

from __future__ import annotations

__all__ = [
    "BaseLoss",
]

import abc

import torch
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

class BaseLoss(_Loss, abc.ABC):
    """Loss function base class.

    Attributes:
        _reduce_fn (Callable): Function to reduce the loss tensor based on the
            specified ``reduction`` method.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, reduction: str = "mean"):
        """Initialize a new instance.

        Args:
            reduction: Reduction method to apply to the loss. Can be one of
                ["none", "mean", "sum"]. Defaults to "mean".

        Raises:
            ValueError: If the provided ``reduction`` method is not supported.
        """
        super().__init__(reduction=reduction)
        # Assign the function once to avoid repeated dict lookups
        self._reduce_fn = {
            "mean": torch.mean,
            "sum" : torch.sum,
            "none": lambda x: x,
        }[reduction]

    # --- Representation ---
    def __str__(self):
        """Return the string representation of the loss class."""
        return depascalize(self.__class__.__name__).lower()

    # --- Callable & Context Manager ---
    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Calculate the loss.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Loss value.
        """
        pass

    def reduce(self, loss: torch.Tensor) -> torch.Tensor:
        """Reduce the loss tensor according to the specified reduction method.

        Args:
            loss: Loss tensor to be reduced.

        Returns:
            Reduced loss tensor.
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
