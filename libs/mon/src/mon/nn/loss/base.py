#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base classes and mixins for loss functions.

This module provides base classes and mixins for loss functions.
"""

from __future__ import annotations

__all__ = [
    "Loss",
]

from abc import ABC

from torch import mean, sum, Tensor
from torch.nn.modules.loss import _Loss


# ==============================================================================
# region BASE CLASSES & MIXINS
# ==============================================================================

# --- Base Classes ---

class Loss(_Loss, ABC):
    """Base class for loss functions.

    Extend PyTorch's ``_Loss`` base class to provide a convenient loss
    reduction method.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, reduction: str = "mean"):
        """Initialize a new instance.

        Args:
            reduction (str): Reduction method to apply to the loss. One of:
                ["mean", "sum", "none"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)

        # Assign attributes
        self._reduce_fn = {
            "mean": mean,
            "sum": sum,
            "none": lambda x: x,
        }[reduction]

    # --- Callable & Context Manager ---
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
