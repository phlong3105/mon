#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for Instance Normalization layers.

This module provides various Instance Normalization layers commonly used in
convolutional neural networks (CNNs).
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
    r"""An adaptive instance normalization layer for 2D tensors."""

    def __init__(
        self,
        num_features: int,
        eps         : float = 0.999,
        momentum    : float = 0.001,
        *args, **kwargs
    ):
        """Initializes the AdaptiveInstanceNorm2d layer.
        
        Args:
            num_features (int): Number of features in the input tensor.
            eps (float): A small value to avoid division by zero. Default is 0.999.
            momentum (float): The momentum for the running mean and variance.
                Default is 0.001.
            args: Additional positional arguments for nn.InstanceNorm2d.
            kwargs: Additional keyword arguments for nn.InstanceNorm2d.
        """
        super().__init__()
        self.w0  = nn.Parameter(torch.tensor(1.0))
        self.w1  = nn.Parameter(torch.tensor(0.0))
        self.in_ = nn.InstanceNorm2d(num_features, eps, momentum, *args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the AdaptiveInstanceNorm2d layer.
        
        Args:
            input (torch.Tensor): Input tensor of shape (N, C, H, W).
        
        Returns:
            torch.Tensor: Output tensor after applying adaptive instance
                normalization of shape (N, C, H, W).
        """
        return self.w0 * input + self.w1 * self.in_(input)
    

class HalfInstanceNorm2d(nn.Module):
    r"""A half instance normalization layer for 2D tensors.
    
    IT applies Instance Normalization on the first half of input tensor and
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
        """Initializes the HalfInstanceNorm2d layer.
        
        Args:
            num_features (int): Number of features in the input tensor.
            eps (float): A small value to avoid division by zero. Default is 1e-5.
            momentum (float): The momentum for the running mean and variance.
                Default is 0.1.
            affine (bool): If True, this module has learnable affine parameters.
                Default is True.
            args: Additional positional arguments for nn.InstanceNorm2d.
            kwargs: Additional keyword arguments for nn.InstanceNorm2d.
            
        Raises:
            ValueError: If num_features is not even.
        """
        super().__init__()
        if num_features % 2 != 0:
            raise ValueError(f"``num_features`` must be even, got {num_features}.")
        self.in_ = nn.InstanceNorm2d(int(num_features // 2), eps, momentum, *args, **kwargs)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the HalfInstanceNorm2d layer.
        
        Args:
            input (torch.Tensor): Input tensor of shape (N, C, H, W) or (C, H, W).
            
        Returns:
            torch.Tensor: Output tensor after applying half instance normalization
                of shape (N, C, H, W) or (C, H, W).
        """
        if input.dim() == 3:
            y1, y2 = torch.chunk(input, 2, dim=0)
        else:
            y1, y2 = torch.chunk(input, 2, dim=1)
        y1 = self.in_(y1)
        y  = torch.cat([y1, y2], dim=1 if input.dim() == 4 else 0)
        return y
