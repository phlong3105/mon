#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Basic loss functions from PyTorch.

This module provides various loss functions commonly used for training machine
learning models.
"""

__all__ = [
    "BCELoss",
    "BCEWithLogitsLoss",
    "CTCLoss",
    "CharbonnierLoss",
    "CosineEmbeddingLoss",
    "CosineSimilarityLoss",
    "CrossEntropyLoss",
    "ExtendedL1Loss",
    "GaussianNLLLoss",
    "HingeEmbeddingLoss",
    "HuberLoss",
    "KLDivLoss",
    "L1Loss",
    "MSELoss",
    "MarginRankingLoss",
    "MultiLabelMarginLoss",
    "MultiLabelSoftMarginLoss",
    "MultiMarginLoss",
    "NLLLoss",
    "NLLLoss2d",
    "PoissonNLLLoss",
    "SmoothL1Loss",
    "SoftMarginLoss",
    "TripletMarginLoss",
    "TripletMarginWithDistanceLoss",
]

import torch
import torch.nn as nn
from torch.nn.modules.loss import *  # Expose all losses from ``torch.nn.modules.loss``

from .base import BaseLoss


# ==============================================================================
# BASIC & ATOMIC LOSSES
# ==============================================================================

# --- Regression ---
class CharbonnierLoss(BaseLoss):
    """A differentiable variant of L1 loss.
    
    Attributes:
        eps2 (float): Small constant for numerical stability.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """Initializes the CharbonnierLoss instance.
        
        Args:
            eps: Small constant for numerical stability. Defaults to 1e-6.
            reduction: Reduction method to apply to the loss. Can be one
                of "none", "mean", or "sum". Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.eps2 = eps ** 2
    
    # --- Callable & Context Manager ---
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the Charbonnier loss between input and target.
        
        Args:
           input: Input (predictions), formatted as a torch.Tensor of shape
                (B, C, H, W) and values ranging from 0.0 to 1.0.
           target: Target (ground truth), formatted as a torch.Tensor of shape
                (B, C, H, W) and values ranging from 0.0 to 1.0.
        
        Returns:
            Loss value, formatted according to the specified reduction method.
        """
        diff = input - target
        loss = torch.sqrt(diff * diff + self.eps2)
        loss = self.reduce(loss=loss)
        return loss
    

# --- Relational & Vector ---
class CosineSimilarityLoss(BaseLoss):
    """Cosine Similarity loss function.
    
    Attributes:
        cos (torch.nn.CosineSimilarity): Cosine similarity module.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(self, dim: int = 1, eps: float = 1e-6, reduction: str = "mean"):
        """Initialize a new instance.
        
        Args:
            dim: Dimension along which to compute cosine similarity. Defaults to 1.
            eps: Small constant for numerical stability. Defaults to 1e-6.
            reduction: Reduction method to apply to the loss. Can be one of
                "none", "mean", or "sum". Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        # dim=1 is standard for (B, C, H, W) images to compare color/feature vectors
        self.cos = nn.CosineSimilarity(dim=dim, eps=eps)
    
    # --- Callable & Context Manager ---
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculates the Cosine Similarity loss between input and target.
        
        Args:
           input: Input (predictions), formatted as a torch.Tensor of shape
                (B, C, H, W) and values ranging from 0.0 to 1.0.
            target: Target (ground truth), formatted as a torch.Tensor of shape
                (B, C, H, W) and values ranging from 0.0 to 1.0.
        
        Returns:
            Loss value, formatted according to the specified reduction method.
        """
        # cos() returns (B, H, W).
        # Loss is 1 - similarity, so similarity=1 means loss=0.
        loss = 1.0 - self.cos(input, target)
        loss = self.reduce(loss=loss)
        return loss
        
        # TODO: Delete later
        """
        b, c, h, w = input.shape
        x    = input.permute(0, 2, 3, 1).view(-1, c)
        y    = target.permute(0, 2, 3, 1).view(-1, c)
        loss = 1.0 - self.cos(x, y).sum() / (1.0 * b * h * w)
        loss = self.reduce(loss=loss)
        return loss
        """
        

# --- Specialized & Masked ---
class ExtendedL1Loss(BaseLoss):
    """Extended L1 loss function that applies a mask to the input and target.
    
    Attributes:
        eps (float): Small constant for numerical stability.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(self, eps: float = 1e-8, reduction: str = "mean"):
        """Initialize a new instance.
        
        Args:
            eps: Small constant for numerical stability. Defaults to 1e-8.
            reduction: Reduction method to apply to the loss. Can be one of
                "none", "mean", or "sum". Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.eps = eps
    
    # --- Callable & Context Manager ---
    # noinspection PyMethodOverriding
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor,
        mask  : torch.Tensor
    ) -> torch.Tensor:
        """Calculate the Extended L1 loss between input and target using a mask.
        
        Args:
            input: Input (predictions), formatted as a torch.Tensor of shape
                (B, C, H, W) and values ranging from 0.0 to 1.0.
            target: Target (ground truth), formatted as a torch.Tensor of shape
                (B, C, H, W) and values ranging from 0.0 to 1.0.
            mask: Mask, formatted as a torch.Tensor of shape (B, 1, H, W)
                with binary values indicating the regions to consider in the
                loss calculation.
                
        Returns:
            Loss value, formatted according to the specified reduction method.
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
            loss  = torch.sum(masked_diff) / denom
        else:
            # If reduction is 'none' or 'sum', use the base reduction logic
            loss  = self.reduce(masked_diff)
        
        # TODO: Delete later
        """
        norm = self.loss_l1(mask, torch.zeros_like(mask))
        loss = self.loss_l1(mask * input, mask * target) / norm
        loss = self.reduce(loss=loss)
        return loss
        """
        return loss
        
