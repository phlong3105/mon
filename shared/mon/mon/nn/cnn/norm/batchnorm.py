#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements batch normalization (batchnorm) layers."""

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
    r"""Applies adaptive Batch Normalization over a 4D input.
    
    .. math::
    
        y = w_0 \cdot x + w_1 \cdot \text{BN}(x)

    Args:
        num_features: :math:`C` from an expected input of size :math:`(N, C, H, W)`.
        eps: A value added to the denominator for numerical stability. Default: ``0.999``.
        momentum: Value used for the running_mean and running_var computation.
            Can be set to ``None`` for cumulative moving average (i.e., simple average).
            Default: ``0.001``.
        kwargs: Additional keyword arguments for ``torch.nn.BatchNorm2d``.
        
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
        super().__init__()
        self.w0 = nn.Parameter(torch.tensor(1.0))
        self.w1 = nn.Parameter(torch.tensor(0.0))
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, *args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.w0 * input + self.w1 * self.bn(input)
