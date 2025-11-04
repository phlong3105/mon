#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements depth-similarity linear layers."""

__all__ = [
    "DepthAwareLinear",
]

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthAwareLinear(nn.Module):
    """A module that applies a linear transformation y = Wx + b to each pixel,
    incorporating depth as a feature and extending with depth-similarity-weighted
    local RGB averages.
    
    For each pixel, computes similarities to neighboring depths in a
    `(kernel_size x kernel_size)` window using exponential decay similarity
    :math:`FD(i,j) = exp^{-alpha * |D_{i} - D_{j}|}` then uses them to
    weight-average the neighbor RGB values. This weighted RGB is concatenated to
    the input: [R, G, B, depth, weighted_R, weighted_G, weighted_B].
    
    Args:
        in_features: Size of each input sample.
        depth_features: Size of depth features.
        out_features: Size of each output sample.
        kernel_size: Odd integer for neighborhood size (e.g., 3 for 3x3). Default: ``3``.
        alpha: Controls similarity sensitivity (larger = stricter decay). Default: ``8.3``.
        bias: If set to ``False``, the layer will not learn an additive bias. Default: ``True``.
        
    Forward Args:
        image: RGB image of shape `(H, W, C)`.
        depth: Depth map of shape `(H, W, 1)` with values 0-1.
    
    Returns:
        Transformed output of shape `(H, W, out_features)`.
    """
    
    def __init__(
        self,
        in_features   : int,
        out_features  : int,
        depth_features: int,
        kernel_size   : int   = 3,
        alpha         : float = 8.3,
        bias          : bool  = True,
    ):
        super().__init__()
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError(f"``kernel_size`` must be odd positive integer, got {kernel_size}.")
        
        self.in_features  = in_features * 2 + depth_features  # 2 * RGB + depth
        self.out_features = out_features
        self.kernel_size  = kernel_size
        self.alpha        = alpha
        self.linear       = nn.Linear(self.in_features, self.out_features, bias=bias)
    
    def forward(self, image: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        if image.dim() != 3 or depth.dim() != 3:
            raise ValueError(f"``image`` and ``depth`` must be 3D tensors, got {image.dim()}D and {depth.dim()}D.")
        
        H, W, C  = image.shape
        C_D      = depth.shape[2]
        r        = self.kernel_size // 2
        k2       = self.kernel_size ** 2
        L        = H * W
        d_center = depth.permute(2, 0, 1).reshape(C_D, L).unsqueeze(0).unsqueeze(2)  # (1, depth_channels, 1, L)
        
        # Prepare input
        image_4d = image.permute(2, 0, 1).unsqueeze(0)  # (1, image_channels, H, W)
        depth_4d = depth.permute(2, 0, 1).unsqueeze(0)  # (1, depth_channels, H, W)
        # Use ReplicationPad2d for replicate padding on spatial dimensions
        pad_layer    = nn.ReplicationPad2d((r, r, r, r))  # left, right, top, bottom
        pad_image_4d = pad_layer(image_4d)
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
        image_flat    = image.permute(2, 0, 1).reshape(C, L).unsqueeze(0)       # (1, image_channels, L)
        weighted_flat = torch.where(mask.unsqueeze(1), normalized, image_flat)  # (1, image_channels, L)
        weighted      = weighted_flat.view(1, C, H, W).permute(0, 2, 3, 1).squeeze(0)  # (H, W, image_channels)
        
        # Apply linear
        # Concatenate: (H, W, in_features) -> [image (C_img), depth (C_depth), weighted (C_img)]
        input_data = torch.cat((image, depth, weighted), dim=2)  # (H, W, in_features)
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
