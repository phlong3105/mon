#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Loss base classes and mixins.

This module provides the base classes and mixins for loss functions.
"""

__all__ = [
    "BaseLoss",
]

import abc

import torch
from torch.nn.modules.loss import _Loss

from mon.core import depascalize


# ==============================================================================
# TYPE DEFINITIONS & PROTOCOLS (Interfaces)
# ==============================================================================

# --- Type Aliases ---


# --- Structural Protocols ---


# ==============================================================================
# BASE CLASSES & MIXINS (Behaviors)
# ==============================================================================

# --- Structural Bases ---
class BaseLoss(_Loss, abc.ABC):
    """A base class for all loss functions.
    
    Attributes:
        reductions (List[str]): List of supported reduction methods.
        reduction (str): Reduction method to apply to the loss. Can be one of
            "none", "mean", or "sum".
    """
    
    reductions = ["none", "mean", "sum"]
    
    def __init__(self, reduction: str = "mean"):
        """Initialize a new instance.
        
        Args:
            reduction: Reduction method to apply to the loss. Can be one of
                "none", "mean", or "sum". Defaults to "mean".
                
        Raises:
            ValueError: If the provided ``reduction`` method is not supported.
        """
        if reduction not in self.reductions:
            raise ValueError(f"``reduction`` must be one of: {self.reductions}, got {reduction}.")
        
        # Initialize the parent class and assign attributes
        super().__init__(reduction=reduction)
        
    # --- Magic Methods ---
    def __str__(self):
        """Return the string representation of the loss class."""
        return depascalize(self.__class__.__name__).lower()
    
    # --- Core Methods ---
    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Calculate the loss.

        Returns:
            Loss value, formatted according to the specified reduction method.
        """
        pass
    
    def reduce(self, loss: torch.Tensor) -> torch.Tensor:
        """Reduce the loss tensor according to the specified reduction method.
    
        Args:
            loss: Loss tensor to be reduced.
            
        Returns:
            Reduced loss tensor.
        """
        return {
            "mean": torch.mean,
            "sum" : torch.sum,
            "none": lambda x: x
        }[self.reduction](loss)


# --- Lifecycle Mixins ---


# --- Compute Mixins ---
