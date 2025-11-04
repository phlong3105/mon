#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Positional Normalization (PONO) and Moment Shortcut (MS).

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
    
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=1, keepdim=True)
        std  = (x.var(dim=1, keepdim=True) + self.eps).sqrt()
        x    = (x - mean) / std
        return x, mean, std


class MomentShortcut(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x    : torch.Tensor,
        beta : torch.Tensor = None,
        gamma: torch.Tensor = None
    ) -> torch.Tensor:
        if gamma is not None:
            x.mul_(gamma)
        if beta is not None:
            x.add_(beta)
        return x
