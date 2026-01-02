#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Instance normalization layers.

This module implements various instance normalization layers commonly used in
convolutional neural networks (CNNs) and deep learning models.
"""

__all__ = [
    "AdaptiveInstanceNorm2d",
    "HalfInstanceNorm2d",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LazyInstanceNorm1d",
    "LazyInstanceNorm2d",
    "LazyInstanceNorm3d",
]

import torch
import torch.nn as nn
from torch.nn.modules.instancenorm import *


class AdaptiveInstanceNorm2d(nn.Module):
    r"""Adaptive instance normalization layer."""

    def __init__(
        self,
        num_features: int,
        eps         : float = 0.999,
        momentum    : float = 0.001,
        *args, **kwargs
    ):
        """Initialize a new instance.
        
        Args:
            num_features: Number of features in the input tensor.
            eps: A small value to avoid division by zero. Default is 0.999.
            momentum: The momentum for the running mean and variance.
                Default is 0.001.
            args: Additional positional arguments for nn.InstanceNorm2d.
            kwargs: Additional keyword arguments for nn.InstanceNorm2d.
        """
        super().__init__()
        self.w0  = nn.Parameter(torch.tensor(1.0))
        self.w1  = nn.Parameter(torch.tensor(0.0))
        self.in_ = nn.InstanceNorm2d(num_features, eps, momentum, *args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            input: Input tensor with dimensions (B, C, H, W) and values ranging
                from 0.0 to 1.0.
        
        Returns:
            Output tensor with dimensions (B, C, H, W) and values ranging
                from 0.0 to 1.0.
        """
        return self.w0 * input + self.w1 * self.in_(input)
    

class HalfInstanceNorm2d(nn.Module):
    r"""A half-instance normalization layer.
    
    Apply Instance Normalization on the first half of the input tensor and
    concatenates it with the second half.
    
    .. math::
        
        y = \text{IN}(x_1) \oplus x_2
    
    where :math:`\oplus` is concatenation along the channel dimension.
    """

    def __init__(
        self,
        num_features: int,
        eps         : float = 1e-5,
        momentum    : float = 0.1,
        affine      : bool  = True,
        *args, **kwargs,
    ):
        """Initialize a new instance.
        
        Args:
            num_features: Number of features in the input tensor.
            eps: A small value to avoid division by zero. Defaults to 1e-5.
            momentum: The momentum for the running mean and variance.
                Defaults to 0.1.
            affine: If True, this module has learnable affine parameters.
                Defaults to True.
            args: Additional positional arguments for nn.InstanceNorm2d.
            kwargs: Additional keyword arguments for nn.InstanceNorm2d.
            
        Raises:
            ValueError: If ``num_features`` is not even.
        """
        super().__init__()
        if num_features % 2 != 0:
            raise ValueError(f"``num_features`` must be even, got {num_features}.")
        self.in_ = nn.InstanceNorm2d(int(num_features // 2), eps, momentum, *args, **kwargs)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            input: Input tensor with dimensions (B, C, H, W) and values ranging
                from 0.0 to 1.0.
                
        Returns:
            Output tensor with dimensions (B, C, H, W) and values ranging
                from 0.0 to 1.0.
        """
        y1, y2 = torch.chunk(input, 2, dim=1)
        y1     = self.in_(y1)
        y      = torch.cat([y1, y2], dim=1 if input.dim() == 4 else 0)
        return y
