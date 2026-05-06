#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Normalization Layers.

This package contains various normalization layers commonly used in convolutional
neural networks (CNNs) and deep learning models.
"""

from __future__ import annotations

__all__ = [
    "AdaptiveBatchNorm2d",
    "AdaptiveInstanceNorm2d",
    "HalfInstanceNorm2d",
    "MomentShortcut",
    "PositionalNorm",
]

import torch
from torch import nn, Tensor


# ==============================================================================
# region BATCH NORMALIZATION
# ==============================================================================

class AdaptiveBatchNorm2d(nn.Module):
    r"""Adaptive batch normalization layer.

    .. math::
        y = w_0 \cdot x + w_1 \cdot \text{BN}(x)

    References:
        - Paper: https://arxiv.org/abs/1709.00643
        - Code: https://github.com/nrupatunga/Fast-Image-Filters
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        num_features: int,
        eps: float = 0.999,
        momentum: float = 0.001,
        *args, **kwargs,
    ):
        """Initialize a new instance.

        Args:
            num_features (int): Number of channels expected in input.
            eps (float, optional): A value added to the denominator for
                numerical stability. Defaults to 0.999.
            momentum (float, optional): The value used for the ``running_mean``
                and ``running_var`` computation. Defaults to 0.001.
        """
        super().__init__()
        self.w0 = nn.Parameter(torch.tensor(1.0))
        self.w1 = nn.Parameter(torch.tensor(0.0))
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, *args, **kwargs)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        """Forward the input through the layer.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            Tensor: Output tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.
        """
        return self.w0 * x + self.w1 * self.bn(x)

# endregion


# ==============================================================================
# region INSTANCE NORMALIZATION
# ==============================================================================

class AdaptiveInstanceNorm2d(nn.Module):
    """Adaptive instance normalization layer."""

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        num_features: int,
        eps: float = 0.999,
        momentum: float = 0.001,
        *args, **kwargs,
    ):
        """Initialize a new instance.

        Args:
            num_features (int): Number of channels expected in input.
            eps (float, optional): A small value to avoid division by zero.
                Defaults to 0.999.
            momentum (float, optional): The value used for the ``running_mean``
                and ``running_var`` computation. Defaults to 0.001.
        """
        super().__init__()
        self.w0 = nn.Parameter(torch.tensor(1.0))
        self.w1 = nn.Parameter(torch.tensor(0.0))
        self.norm = nn.InstanceNorm2d(num_features, eps, momentum, *args, **kwargs)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        """Forward the input through the layer.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            Tensor: Output tensor of shape (B, C, H, W) and values ranging
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
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        *args, **kwargs,
    ):
        """Initialize a new instance.

        Args:
            num_features (int): Number of channels expected in input.
            eps (float, optional): A small value to avoid division by zero.
                Defaults to 1e-5.
            momentum (float, optional): The value used for the ``running_mean``
                and ``running_var`` computation. Defaults to 0.1.
            affine (bool, optional): If True, this module has learnable affine
                parameters. Defaults to True.

        Raises:
            ValueError: If ``num_features`` is not even.
        """
        super().__init__()
        if num_features % 2 != 0:
            raise ValueError(f"expected num_features to be even, got {num_features}.")

        self.norm = nn.InstanceNorm2d(
            num_features // 2, eps, momentum, affine=affine, *args, **kwargs,
        )

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        """Forward the input through the layer.

        Args:
            x (Tensor): Input tensor of shape (B, 2C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            Tensor: Output tensor of shape (B, 2C, H, W) and values ranging
                from 0.0 to 1.0.
        """
        y1, y2 = torch.chunk(x, chunks=2, dim=1)
        y1 = self.norm(y1)
        return torch.cat([y1, y2], dim=1)

# endregion


# ==============================================================================
# region POSITIONAL NORMALIZATION
# ==============================================================================

class PositionalNorm(nn.Module):
    """Positional normalization layer."""

    # --- Lifecycle & Initialization ---
    def __init__(self, eps: float = 1e-5):
        """Initialize a new instance.

        Args:
            eps (float, optional): A small value to avoid division by zero.
                Defaults to 1e-5.
        """
        super().__init__()
        self.eps = eps

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward the input through the layer.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            tuple[Tensor, Tensor, Tensor]: Tuple containing the normalized tensor,
                the mean tensor, and the standard deviation tensor.
        """
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True) + self.eps).sqrt()
        x = (x - mean) / std
        return x, mean, std


class MomentShortcut(nn.Module):
    """Moment shortcut layer."""

    # --- Callable & Context Manager ---
    def forward(
        self,
        x: Tensor,
        beta: Tensor | None = None,
        gamma: Tensor | None = None,
    ) -> Tensor:
        """Forward the input through the layer.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.
            beta (Tensor | None, optional): The beta tensor of shape (B, 1, H, W)
                and values ranging from -1.0 to 1.0. Defaults to None.
            gamma (Tensor | None, optional): The gamma tensor of shape (B, 1, H, W)
                and values ranging from 0.0 to 1.0. Defaults to None.

        Returns:
            Tensor: Output tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.
        """
        if gamma is not None:
            x = x * gamma
        if beta is not None:
            x = x + beta
        return x

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
