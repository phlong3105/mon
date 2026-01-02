#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Positional Normalization and Moment Shortcut layers.

This module implements Positional Normalization (PONO) and Moment Shortcut (MS)
layers.

References:
    - Paper: "Positional Normalization," NeurIPS 2019.
    - Code: https://github.com/Boyiliee/Positional-Normalization
    
Pseudocode:
    # x is the features of shape [B, C, H, W]
    
    # In the Encoder
    def PONO(x, epsilon=1e-5):
        mean = x.mean(dim=1, keepdim=True)
        std  = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
        x    = (x - mean) / std
        return x, mean, std
        
    # In the Decoder, one can call MS(x, mean, std) with the mean and std are from a PONO in the encoder
    def MS(x, beta, gamma):
        return x * gamma + beta
"""

__all__ = [
    "PositionalNorm",
    "MomentShortcut",
]

import torch
import torch.nn as nn


class PositionalNorm(nn.Module):
    """Positional normalization layer."""
    
    def __init__(self, eps: float = 1e-5):
        """Initialize a new instance.
        
        Args:
            eps: A small value to avoid division by zero. Defaults to 1e-5.
        """
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor with dimensions (B, C, H, W) and values ranging
                from 0.0 to 1.0.
        
        Returns:
            A tuple containing:
                - The normalized tensor with dimensions (B, C, H, W).
                - The mean tensor with dimensions (B, 1, H, W).
                - The standard deviation tensor with dimensions (B, 1, H, W).
        """
        mean = x.mean(dim=1, keepdim=True)
        std  = (x.var(dim=1, keepdim=True) + self.eps).sqrt()
        x    = (x - mean) / std
        return x, mean, std


class MomentShortcut(nn.Module):
    """Moment shortcut layer."""
    
    def forward(
        self,
        x    : torch.Tensor,
        beta : torch.Tensor = None,
        gamma: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor with dimensions (B, C, H, W) and values ranging
                from 0.0 to 1.0.
            beta: The beta tensor with dimensions (B, 1, H, W). Defaults to None.
            gamma: The gamma tensor with dimensions (B, 1, H, W). Defaults to None.
        
        Returns:
            Output tensor with dimensions (B, C, H, W) and values ranging
                from 0.0 to 1.0.
        """
        if gamma is not None:
            x.mul_(gamma)
        if beta is not None:
            x.add_(beta)
        return x
