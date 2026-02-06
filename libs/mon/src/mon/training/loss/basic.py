#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Basic loss functions.

This module provides various loss functions commonly used for training machine
learning models.
"""

from __future__ import annotations

__all__ = [
    "CharbonnierLoss",
    "CosineSimilarityLoss",
    "ExtendedL1Loss",
]

from typing import override

import torch
from torch import nn, Tensor

from .base import BaseLoss


# ==============================================================================
# region BASIC LOSSES
# ==============================================================================

class CharbonnierLoss(BaseLoss):
    """Differentiable variant of L1 loss."""

    # --- Lifecycle & Initialization ---
    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """Initialize a new instance.

        Args:
            eps (float): Small constant for numerical stability. Defaults to 1e-6.
            reduction (str): Reduction method to apply to the loss. One of:
                ["mean", "sum", "none"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.eps2 = eps ** 2

    # --- Callable & Context Manager ---
    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Calculate the loss between ``input`` and ``target``.

        Args:
            input (Tensor): Input (predictions) tensor of shape (B, C, H, W)
                and values ranging from 0.0 to 1.0.
            target (Tensor): Target (ground truth) tensor of shape (B, C, H, W)
                and values ranging from 0.0 to 1.0.

        Returns:
            Tensor: Loss value.
        """
        diff = input - target
        loss = torch.sqrt(diff * diff + self.eps2)
        loss = self.reduce(loss=loss)
        return loss


class CosineSimilarityLoss(BaseLoss):
    """Cosine Similarity loss function."""

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        dim: int = 1,
        eps: float = 1e-6,
        reduction: str = "mean",
    ):
        """Initialize a new instance.

        Args:
            dim (int): Dimension along which to compute the cosine similarity.
                Defaults to 1.
            eps (float): Small constant for numerical stability. Defaults to 1e-6.
            reduction (str): Reduction method to apply to the loss. One of:
                ["mean", "sum", "none"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        # dim=1 is standard for (B, C, H, W) images to compare color/feature vectors
        self.cos = nn.CosineSimilarity(dim=dim, eps=eps)

    # --- Callable & Context Manager ---
    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Calculate the loss between ``input`` and ``target``.

        Args:
            input (Tensor): Input (predictions) tensor of shape (B, C, H, W)
                and values ranging from 0.0 to 1.0.
            target (Tensor): Target (ground truth) tensor of shape (B, C, H, W)
                and values ranging from 0.0 to 1.0.

        Returns:
            Tensor: Loss value.
        """
        # cos() returns (B, H, W).
        # Loss is 1 - similarity, so similarity=1 means loss=0.
        loss = 1.0 - self.cos(input, target)
        loss = self.reduce(loss=loss)
        return loss


class ExtendedL1Loss(BaseLoss):
    """Extended L1 loss function that applies a mask to the input and target."""

    # --- Lifecycle & Initialization ---
    def __init__(self, eps: float = 1e-8, reduction: str = "mean"):
        """Initialize a new instance.

        Args:
            eps (float): Small constant for numerical stability. Defaults to 1e-8.
            reduction (str): Reduction method to apply to the loss. One of:
                ["mean", "sum", "none"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.eps = eps

    # --- Callable & Context Manager ---
    @override
    def forward(self, input: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """Calculate the loss between ``input`` and ``target``.

        Args:
            input (Tensor): Input (predictions) tensor of shape (B, C, H, W)
                and values ranging from 0.0 to 1.0.
            target (Tensor): Target (ground truth) tensor of shape (B, C, H, W)
                and values ranging from 0.0 to 1.0.
            mask (Tensor): Mask tensor of shape (B, 1, H, W) with binary values
                indicating the regions to consider in the loss calculation.

        Returns:
            Tensor: Loss value.
        """
        # Calculate absolute difference
        abs_diff = torch.abs(input - target)

        # Apply mask
        masked_diff = abs_diff * mask

        # Proper Normalization (Masked Mean)
        # Instead of dividing by the total number of pixels,
        # we divide by the number of active pixels in the mask.
        if self.reduction == "mean":
            # Sum of active pixels
            denom = torch.sum(mask) + self.eps
            loss = torch.sum(masked_diff) / denom
        else:
            # If reduction is 'none' or 'sum', use the base reduction logic
            loss = self.reduce(masked_diff)

        return loss


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
