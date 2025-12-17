#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for Batch Normalization layers.

This module implements various Batch Normalization layers commonly used in
convolutional neural networks (CNNs) and deep learning models.
"""

__all__ = [
    "AdaptiveBatchNorm2d",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "LazyBatchNorm1d",
    "LazyBatchNorm2d",
    "LazyBatchNorm3d",
    "SyncBatchNorm",
]

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import *


class AdaptiveBatchNorm2d(nn.Module):
    r"""An implementation of Adaptive Batch Normalization.
    
    It applies adaptive Batch Normalization over a 4D input.
    
    .. math::
        y = w_0 \cdot x + w_1 \cdot \text{BN}(x)

    References:
        - Paper: https://arxiv.org/abs/1709.00643
        - Code: https://github.com/nrupatunga/Fast-Image-Filters
    """

    def __init__(
        self,
        num_features: int,
        eps         : float = 0.999,
        momentum    : float = 0.001,
        *args, **kwargs
    ):
        """Initializes the Adaptive Batch Normalization layer.
        
        Args:
            num_features (int): Number of features in the input tensor.
            eps (float): A value added to the denominator for numerical stability.
                Defaults to 0.999.
            momentum (float): The value used for the running mean and variance
                computation. Defaults to 0.001.
            *args: Additional positional arguments for nn.BatchNorm2d.
            **kwargs: Additional keyword arguments for nn.BatchNorm2d.
        """
        super().__init__()
        self.w0 = nn.Parameter(torch.tensor(1.0))
        self.w1 = nn.Parameter(torch.tensor(0.0))
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, *args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Adaptive Batch Normalization layer.
        
        Args:
            input (torch.Tensor): Input tensor of shape (N, C, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, C, H, W).
        """
        return self.w0 * input + self.w1 * self.bn(input)
