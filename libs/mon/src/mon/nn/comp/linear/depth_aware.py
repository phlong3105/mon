#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Depth-aware linear layers.

This module provides depth-aware linear layers that incorporate depth
information into the linear transformation process.
"""

from __future__ import annotations

__all__ = [
    "DepthAwareLinear",
]

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# region DEPTH-AWARE LINEAR LAYERS
# ==============================================================================

class DepthAwareLinear(nn.Module):
    """A linear layer with depth-aware local RGB averaging.

    Apply a linear transformation to each pixel, augmenting the input with
    depth-similarity-weighted local RGB averages.

    For each pixel, a neighborhood is defined by ``kernel_size``. The depth map
    is used to compute similarity weights for the pixels in this neighborhood.
    These weights are then used to compute a weighted average of the RGB values
    in the neighborhood. The original RGB values, depth values, and the
    depth-weighted RGB averages are concatenated and passed through a linear
    layer.

    Attributes:
        in_features: Number of input features (RGB channels).
        out_features: Number of output features.
        kernel_size: Size of the square neighborhood for local averaging.
        alpha: Parameter controlling the sensitivity of depth similarity.
        linear: The underlying linear layer.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_features   : int,
        out_features  : int,
        depth_features: int,
        kernel_size   : int   = 3,
        alpha         : float = 8.3,
        bias          : bool  = True,
    ):
        """Initialize a new instance.

        Args:
            in_features: Number of input features (RGB channels).
            out_features: Number of output features.
            depth_features: Number of depth features (depth channels).
            kernel_size: Size of the square neighborhood for local averaging.
                Must be an odd positive integer. Defaults to 3.
            alpha: Parameter controlling the sensitivity of depth similarity.
                Defaults to 8.3.
            bias: If True, includes a bias term in the linear layer.
                Defaults to True.

        Raises:
            ValueError: If ``kernel_size`` is not an odd positive integer.
        """
        super().__init__()
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError(f"Expected 'kernel_size' to be an odd positive integer, but got {kernel_size}.")

        self.in_features  = in_features * 2 + depth_features  # 2 * RGB + depth
        self.out_features = out_features
        self.kernel_size  = kernel_size
        self.alpha        = alpha
        self.linear       = nn.Linear(self.in_features, self.out_features, bias=bias)

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Args:
            x: Image tensor of shape (H, W, C) or (B, H, W, C) and values ranging
                from 0.0 to 1.0.
            d: Depth tensor of shape (H, W, C_D) or (B, H, W, C_D) and values
                ranging from 0.0 to 1.0.

        Returns:
            Output tensor of shape (H, W, out_features) or (B, H, W, out_features).
        """
        # Handle dimensions: Ensure (B, H, W, C)
        is_batched = x.dim() == 4
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if d.dim() == 3:
            d = d.unsqueeze(0)

        if x.dim() != 4 or d.dim() != 4:
            raise ValueError(
                f"Expected 'x' and 'd' to be 3D (H, W, C) or 4D (B, H, W, C), but got {x.shape} and "
                f"{d.shape}."
            )

        B, H, W, C   = x.shape
        _, _, _, C_D = d.shape

        # Permute to (B, C, H, W) for unfolding
        input_nchw = x.permute(0, 3, 1, 2)
        depth_nchw = d.permute(0, 3, 1, 2)

        # Padding
        pad = self.kernel_size // 2
        input_padded = F.pad(input_nchw, (pad, pad, pad, pad), mode="replicate")
        depth_padded = F.pad(depth_nchw, (pad, pad, pad, pad), mode="replicate")

        # Unfold: (B, C * k*k, L) where L = H*W
        input_unfolded = F.unfold(input_padded, kernel_size=self.kernel_size)
        depth_unfolded = F.unfold(depth_padded, kernel_size=self.kernel_size)

        L  = H * W
        k2 = self.kernel_size ** 2

        # Reshape for processing
        input_windows = input_unfolded.view(B, C, k2, L)    # (B, C, k2, L)
        depth_windows = depth_unfolded.view(B, C_D, k2, L)  # (B, C_D, k2, L)

        # Center depth: (B, C_D, 1, L)
        depth_center = depth_nchw.view(B, C_D, 1, L)

        # Similarity
        diff    = depth_windows - depth_center
        dist_sq = torch.sum(diff ** 2, dim=1, keepdim=True)  # (B, 1, k2, L)
        sim     = torch.exp(-dist_sq / (2 * self.alpha ** 2))

        # Weighted sum of input
        # sim: (B, 1, k2, L)
        weighted_sum = torch.sum(input_windows * sim, dim=2)  # (B, C, L)
        sum_sim      = torch.sum(sim, dim=2)                  # (B, 1, L)

        # Normalize
        weighted_avg = weighted_sum / (sum_sim + 1e-8)

        # Reshape weighted_avg back to (B, H, W, C)
        weighted_avg = weighted_avg.view(B, C, H, W).permute(0, 2, 3, 1)

        # Concatenate: (B, H, W, C + C_D + C)
        combined = torch.cat([x, d, weighted_avg], dim=-1)

        # Linear layer
        output = self.linear(combined)

        if not is_batched:
            output = output.squeeze(0)

        return output

# endregion


# ==============================================================================
# region UNIT TESTS
# ==============================================================================

if __name__ == "__main__":
    # Test 3D input
    image  = torch.ones(256, 256, 49)
    depth  = torch.randn(256, 256, 1)
    linear = DepthAwareLinear(49, 256, 1, kernel_size=3, alpha=10.0)
    out    = linear(image, depth)
    print(f"3D Input Output Shape: {out.shape}")

    # Test 4D input
    image_batch = torch.ones(2, 256, 256, 49)
    depth_batch = torch.randn(2, 256, 256, 1)
    out_batch   = linear(image_batch, depth_batch)
    print(f"4D Input Output Shape: {out_batch.shape}")

# endregion
