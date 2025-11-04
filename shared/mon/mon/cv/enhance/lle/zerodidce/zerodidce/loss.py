#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "PiecewiseNonReferenceLoss",
]

import torch

from mon.core import nn


class PiecewiseNonReferenceLoss(nn.BaseLoss):
    
    def __init__(
        self,
        K : int   = 16,
        Q1: float = 0.2,
        Q2: float = 0.8,
        W1: float = 1.0,
        W2: float = 0.5,
        m : float = 1.0,
        E : float = 0.6,
    ):
        super().__init__()
        self.K  = K   # Patch size
        self.Q1 = Q1  # Threshold for extremely low light
        self.Q2 = Q2  # Threshold for overexposed
        self.W1 = W1  # Weight for extreme light conditions
        self.W2 = W2  # Weight for general light conditions
        self.m  = m   # Weight control for L2 loss
        self.E  = E   # Well-lighted value in RGB space (0.6 as per paper)

    def extract_patches(self, image: torch.Tensor) -> torch.Tensor:
        """Extract patches of size patch_size x patch_size from the input image."""
        patches = image.unfold(2, self.K, self.K).unfold(3, self.K, self.K)
        patches = patches.contiguous().view(image.size(0), image.size(1), -1, self.K, self.K)
        return patches

    def compute_region_light_quality(self, patches: torch.Tensor) -> torch.Tensor:
        """Compute mean light quality for each patch."""
        return patches.mean(dim=[1, 3, 4])  # Mean across channels and spatial dimensions

    def L1_loss(self, Y_e: torch.Tensor) -> torch.Tensor:
        """Compute L1 loss for extreme light conditions."""
        return torch.mean((Y_e - self.E) ** 2)

    def L2_loss(self, Y_e: torch.Tensor, Y_i: torch.Tensor) -> torch.Tensor:
        """Compute L2 loss for general light conditions."""
        return torch.mean((Y_e - self.E) ** 2 / (1 + self.m * Y_i))

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute piecewise non-reference loss."""
        # Extract patches
        input_patches  = self.extract_patches(input)
        target_patches = self.extract_patches(target)

        # Compute light quality for input and enhanced patches
        Y_i = self.compute_region_light_quality(input_patches)   # Input image light quality
        Y_e = self.compute_region_light_quality(target_patches)  # Enhanced image light quality

        # Define masks for extreme and general light conditions
        extreme_mask = (Y_i <= self.Q1) | (Y_i >= self.Q2)  # Boolean mask for extreme conditions
        general_mask = ~extreme_mask  # Boolean mask for general conditions

        # L1 loss for extreme light conditions: mean((Y_e - E)^2)
        L1 = torch.mean((Y_e - self.E) ** 2 * extreme_mask.float(), dim=1)

        # L2 loss for general light conditions: mean((Y_e - E)^2 / (1 + m * Y_i))
        L2 = torch.mean(((Y_e - self.E) ** 2 / (1 + self.m * Y_i)) * general_mask.float(), dim=1)

        # Combine losses with weights
        loss = self.W1 * L1 + self.W2 * L2
        loss = loss.mean()
        return loss
