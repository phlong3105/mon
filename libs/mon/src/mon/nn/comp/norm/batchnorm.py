#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Batch normalization layers.

This module provides various batch normalization layers used for normalizing
a batch of inputs in neural networks.
"""

from __future__ import annotations

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
from torch.nn import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    LazyBatchNorm1d,
    LazyBatchNorm2d,
    LazyBatchNorm3d,
    SyncBatchNorm,
)


class AdaptiveBatchNorm2d(nn.Module):
    r"""Adaptive batch normalization layer.

    Apply adaptive batch normalization to the input tensor.

    .. math::
        y = w_0 \cdot x + w_1 \cdot \text{BN}(x)

    References:
        - Paper: https://arxiv.org/abs/1709.00643
        - Code: https://github.com/nrupatunga/Fast-Image-Filters

    Attributes:
        w0 (torch.nn.parameter.Parameter): Weight for the identity connection.
        w1 (torch.nn.parameter.Parameter): Weight for the batch normalization
            connection.
        bn (torch.nn.BatchNorm2d): Batch normalization layer.
    """

    # --- Lifecycle & Initialization ---
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
            eps: A value added to the denominator for numerical stability.
                Defaults to 0.999.
            momentum: The value used for the running mean and variance
                computation. Defaults to 0.001.
            *args: Additional positional arguments for nn.BatchNorm2d.
            **kwargs: Additional keyword arguments for nn.BatchNorm2d.
        """
        super().__init__()
        self.w0 = nn.Parameter(torch.tensor(1.0))
        self.w1 = nn.Parameter(torch.tensor(0.0))
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, *args, **kwargs)

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Args:
            x: Input tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            Output tensor of shape (B, C, H, W) and values ranging
            from 0.0 to 1.0.
        """
        return self.w0 * x + self.w1 * self.bn(x)


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
