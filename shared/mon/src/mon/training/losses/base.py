#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for base loss functions.

This module provides the base class for all loss functions used in training
machine learning models. It defines the common interface and functionality that
all loss functions should implement.
"""

__all__ = [
    "BaseLoss",
]

import abc
from typing import Literal

import torch
from torch.nn.modules.loss import _Loss

from mon.core import depascalize


# ----- Base Loss -----
class BaseLoss(_Loss, abc.ABC):
    """A base class for all loss functions.
    
    Attributes:
        reductions (List[str]): List of supported reduction methods.
        reduction (str): Reduction method to apply to the loss. Can be one of
            "none", "mean", or "sum".
    """
    
    reductions = ["none", "mean", "sum"]
    
    def __init__(self, reduction: Literal["none", "mean", "sum"] = "mean"):
        """Initializes the BaseLoss instance.
        
        Args:
            reduction (str): Reduction method to apply to the loss. Can be one
                of "none", "mean", or "sum". Defaults to "mean".
                
        Raises:
            ValueError: If the provided reduction method is not supported.
        """
        super().__init__(reduction=reduction)
        if self.reduction not in self.reductions:
            raise ValueError(f"``reduction`` must be one of: {self.reductions}, got {reduction}.")
    
    # ----- Magic Methods -----
    def __str__(self):
        """Returns the string representation of the loss class."""
        return depascalize(self.__class__.__name__).lower()
    
    @abc.abstractmethod
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the loss between input and target.
    
        Args:
            input (torch.Tensor): Input tensor (predictions) of shape (B, C, H, W)
                with pixel values in the range [0.0, 1.0].
            target (torch.Tensor): Target tensor (ground truth) of shape (B, C, H, W)
                with pixel values in the range [0.0, 1.0].
                
        Returns:
            torch.Tensor: Calculated loss
        """
        pass
    
    def reduce(self, loss: torch.Tensor) -> torch.Tensor:
        """Reduces the loss tensor.
    
        Args:
            loss (torch.Tensor): Loss tensor to be reduced.
            
        Returns:
            torch.Tensor: Reduced loss tensor.
        """
        return {
            "mean": torch.mean,
            "sum" : torch.sum,
            "none": lambda x: x
        }[self.reduction](loss)
