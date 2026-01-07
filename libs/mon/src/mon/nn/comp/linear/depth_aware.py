#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Depth-aware linear layers.

This module implements depth-aware linear layers that incorporate depth
information into the linear transformation process.
"""

__all__ = [
    "DepthAwareLinear",
]

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            bias: If True, includes a bias term in the linear layer. Defaults to True.
                
        Raises:
            ValueError: If ``kernel_size`` is not an odd positive integer.
        """
        super().__init__()
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError(f"``kernel_size`` must be odd positive integer, got {kernel_size}.")
        
        self.in_features  = in_features * 2 + depth_features  # 2 * RGB + depth
        self.out_features = out_features
        self.kernel_size  = kernel_size
        self.alpha        = alpha
        self.linear       = nn.Linear(self.in_features, self.out_features, bias=bias)
    
    # --- Callable & Context Manager ---
    def forward(self, input: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        if input.dim() != 3 or depth.dim() != 3:
            raise ValueError(f"``input`` and ``depth`` must be 3D tensors, got {input.dim()}D and {depth.dim()}D.")
        
        H, W, C  = input.shape
        C_D      = depth.shape[2]
        r        = self.kernel_size // 2
        k2       = self.kernel_size ** 2
        L        = H * W
        d_center = depth.permute(2, 0, 1).reshape(C_D, L).unsqueeze(0).unsqueeze(2)  # (1, depth_channels, 1, L)
        
        # Prepare input
        input_4d = input.permute(2, 0, 1).unsqueeze(0)  # (1, image_channels, H, W)
        depth_4d = depth.permute(2, 0, 1).unsqueeze(0)  # (1, depth_channels, H, W)
        # Use ReplicationPad2d for replicate padding on spatial dimensions
        pad_layer    = nn.ReplicationPad2d((r, r, r, r))  # left, right, top, bottom
        pad_image_4d = pad_layer(input_4d)
        pad_depth_4d = pad_layer(depth_4d)
        # Unfold to get flattened windows
        window_image_flat = F.unfold(pad_image_4d, kernel_size=self.kernel_size, stride=1)  # (1, image_channels*k2, H*W)
        window_depth_flat = F.unfold(pad_depth_4d, kernel_size=self.kernel_size, stride=1)  # (1, depth_channels*k2, H*W)
        
        # Depth Similarity
        window_depth_reshaped = window_depth_flat.view(1, C_D, k2, L)  # (1, depth_channels, k2, L)
        diff    = window_depth_reshaped - d_center
        dist_sq = torch.sum(diff ** 2, dim=1)                  # Squared Euclidean distance: (1, k2, L)
        sim     = torch.exp(-dist_sq / (2 * self.alpha ** 2))  # Similarities: (1, k2, L)
        sum_sim = torch.sum(sim, dim=1)  # (1, L)
        
        # Compute weighted sum: reshape and multiply
        window_image_reshaped = window_image_flat.view(1, C, k2, L)             # (1, image_channels, k2, L)
        sim_reshaped  = sim.unsqueeze(1)                                        # (1, 1, k2, L)
        weighted_sum  = torch.sum(sim_reshaped * window_image_reshaped, dim=2)  # (1, image_channels, L)
        # Normalize where sum_sim > 0, else fallback to original image
        mask          = sum_sim > 0  # (1, L)
        normalized    = weighted_sum / sum_sim.clamp(min=1e-6).unsqueeze(1)     # (1, image_channels, L)
        image_flat    = input.permute(2, 0, 1).reshape(C, L).unsqueeze(0)       # (1, image_channels, L)
        weighted_flat = torch.where(mask.unsqueeze(1), normalized, image_flat)  # (1, image_channels, L)
        weighted      = weighted_flat.view(1, C, H, W).permute(0, 2, 3, 1).squeeze(0)  # (H, W, image_channels)
        
        # Apply linear
        # Concatenate: (H, W, in_features) -> [image (C_img), depth (C_depth), weighted (C_img)]
        input_data = torch.cat((input, depth, weighted), dim=2)  # (H, W, in_features)
        flat_input = input_data.view(-1, self.in_features)  # Flatten for linear: (H*W, in_features)
        y_flat     = self.linear(flat_input)
        # Reshape back to (H, W, out_features)
        output     = y_flat.view(H, W, self.out_features)
        
        return output


if __name__ == "__main__":
    image  = torch.ones(256, 256, 49)
    depth  = torch.randn(256, 256, 1)
    linear = DepthAwareLinear(49, 256, 1, kernel_size=3, alpha=10.0)
    out    = linear(image, depth)
    print(out.shape)
