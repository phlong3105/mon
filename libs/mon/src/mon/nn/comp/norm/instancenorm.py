#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Instance normalization layers.

This module provides various instance normalization layers used for
normalizing.
"""

from __future__ import annotations

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
from torch.nn.modules.instancenorm import (
    InstanceNorm1d,
    InstanceNorm2d,
    InstanceNorm3d,
    LazyInstanceNorm1d,
    LazyInstanceNorm2d,
    LazyInstanceNorm3d,
)


class AdaptiveInstanceNorm2d(nn.Module):
    """Adaptive instance normalization layer.

    Apply adaptive instance normalization to the input tensor.

    Attributes:
        w0 (torch.nn.parameter.Parameter): Weight for the identity connection.
        w1 (torch.nn.parameter.Parameter): Weight for the instance normalization
            connection.
        norm (torch.nn.InstanceNorm2d): Instance normalization layer.
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
            eps: A small value to avoid division by zero. Defaults to 0.999.
            momentum: The momentum for the running mean and variance.
                Defaults to 0.001.
            *args: Additional positional arguments for nn.InstanceNorm2d.
            **kwargs: Additional keyword arguments for nn.InstanceNorm2d.
        """
        super().__init__()
        self.w0   = nn.Parameter(torch.tensor(1.0))
        self.w1   = nn.Parameter(torch.tensor(0.0))
        self.norm = nn.InstanceNorm2d(num_features, eps, momentum, *args, **kwargs)

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Args:
            x: Input tensor with dimensions (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            Output tensor with dimensions (B, C, H, W) and values ranging
                from 0.0 to 1.0.
        """
        return self.w0 * x + self.w1 * self.norm(x)


class HalfInstanceNorm2d(nn.Module):
    r"""Half-instance normalization layer.

    Apply Instance Normalization on the first half of the input tensor and
    concatenate it with the second half.

    .. math::

        y = \text{IN}(x_1) \oplus x_2

    where :math:`\oplus` is concatenation along the channel dimension.

    Attributes:
        norm (torch.nn.InstanceNorm2d): Instance normalization layer.
    """

    # --- Lifecycle & Initialization ---
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
            *args: Additional positional arguments for nn.InstanceNorm2d.
            **kwargs: Additional keyword arguments for nn.InstanceNorm2d.

        Raises:
            ValueError: If ``num_features`` is not even.
        """
        super().__init__()
        if num_features % 2 != 0:
            raise ValueError(f"``num_features`` must be even, got {num_features}.")

        self.norm = nn.InstanceNorm2d(
            num_features // 2, eps, momentum, affine=affine, *args, **kwargs
        )

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Args:
            x: Input tensor with dimensions (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            Output tensor with dimensions (B, C, H, W) and values ranging
                from 0.0 to 1.0.
        """
        y1, y2 = torch.chunk(x, chunks=2, dim=1)
        y1     = self.norm(y1)
        return torch.cat([y1, y2], dim=1)


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
