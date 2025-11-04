#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Depth-Aware Convolution and Depth-Aware Pooling layers.

References:
    - Paper: "Depth-aware CNN for RGB-D Segmentation," ECCV 2018.
    - Code: https://github.com/laughtervv/DepthAwareCNN
"""

__all__ = [
    "DepthAwareAvgPool2d",
    "DepthAwareConv2d",
]

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthAwareConv2d(nn.Module):
    """Depth-Aware Convolution Layer.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolution kernel.
        alpha: Scaling factor for depth similarity. Default: ``8.3`` (from paper).
        padding: Padding size for the convolution. Default: ``0``.
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : int,
        alpha       : float = 8.3,
        padding     : int   = 0,
    ):
        super().__init__()
        self.conv        = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.alpha       = alpha
        self.kernel_size = kernel_size
        self.padding     = padding

    def forward(self, x: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        # x    : [b, channels, h, w]
        # depth: [b, 1,        h, w]
        b, _, h, w = x.size()

        # Extract patches for depth similarity computation
        h_kernel, w_kernel = self.kernel_size, self.kernel_size
        padding            = self.padding
        depth_padded       = F.pad(depth, (padding, padding, padding, padding), mode="replicate")
        depth_unfolded     = F.unfold(depth_padded, kernel_size=(h_kernel, w_kernel), stride=1, padding=0)
        depth_unfolded     = depth_unfolded.view(b, 1, h_kernel * w_kernel, h * w)

        # Center depth values
        depth_center = depth.view(b, 1, 1, h * w)

        # Compute depth difference and similarity
        depth_diff = depth_unfolded - depth_center
        F_D = torch.exp(-self.alpha * torch.abs(depth_diff))  # [batch, 1, kernel_size^2, h*w]

        # Reshape F_D to match conv output for element-wise multiplication.
        F_D = F_D.view(b, 1, h_kernel, w_kernel, h, w)
        F_D = F_D.permute(0, 4, 1, 2, 3, 5).reshape(b, 1, h * w_kernel, w * h_kernel)
        F_D = F_D[:, :, padding:h + padding, padding:w + padding]  # Adjust for padding
        
        # Apply depth similarity to standard convolution output
        return self.conv(x) * F_D


class DepthAwareAvgPool2d(nn.Module):
    """Depth-Aware Average Pooling Layer.
    
    Args:
        kernel_size: Size of the pooling kernel.
        alpha: Scaling factor for depth similarity. Default: ``8.3`` (from paper).
        stride: Stride for the pooling operation. Default: ``1``.
        padding: Padding size for the pooling. Default: ``0``.
    """
    
    def __init__(self, kernel_size: int, alpha: float = 8.3, stride: int = 1, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride      = stride
        self.padding     = padding
        self.alpha       = alpha

    def forward(self, x: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        # x    : [b, c, h, w]
        # depth: [b, 1, h, w]
        b, c, h, w = x.size()

        # Pad depth and input for pooling
        depth_padded = F.pad(depth, (self.padding, self.padding, self.padding, self.padding), mode="replicate")
        x_padded     = F.pad(x,     (self.padding, self.padding, self.padding, self.padding), mode="replicate")

        # Extract patches using unfold
        x_unfolded     = F.unfold(x_padded,     kernel_size=self.kernel_size, stride=self.stride)
        depth_unfolded = F.unfold(depth_padded, kernel_size=self.kernel_size, stride=self.stride)

        # Reshape for computation
        b, c_in, h_out, w_out = x_unfolded.size(0), x_unfolded.size(1) // (self.kernel_size * self.kernel_size), x_unfolded.size(2), 1
        x_unfolded     =     x_unfolded.view(b, c, self.kernel_size * self.kernel_size, h_out * w_out)
        depth_unfolded = depth_unfolded.view(b, 1, self.kernel_size * self.kernel_size, h_out * w_out)

        # Center depth values
        depth_center = depth.unfold(2, self.stride, self.stride).unfold(3, self.stride, self.stride)
        depth_center = depth_center.contiguous().view(b, 1, 1, h_out * w_out)

        # Compute depth similarity
        depth_diff = depth_unfolded - depth_center
        F_D = torch.exp(-self.alpha * torch.abs(depth_diff))  # [b, 1, kernel_size^2, h_out*w_out]

        # Weighted average pooling
        weighted_sum = torch.sum(F_D * x_unfolded, dim=2, keepdim=True)  # Sum over kernel
        fd_sum = torch.sum(F_D, dim=2, keepdim=True)  # Normalize
        output = weighted_sum / (fd_sum + 1e-8)       # Avoid division by zero
        
        # Reshape to [b, c, h_out, w_out]
        output = output.view(b, c, h_out, w_out)
        return output


if __name__ == "__main__":
    # Test DepthAwareConv2d
    b, c, h, w = 1, 3, 5, 5
    x          = torch.randn(b, c, h, w)
    depth      = torch.randn(b, 1, h, w)
    dac        = DepthAwareConv2d(in_channels=c, out_channels=6, kernel_size=3, padding=1)
    out        = dac(x, depth)
    print("DepthAwareConv2d output shape:", out.shape)  # Expected: [b, 6, h, w]

    # Test DepthAwareAvgPool2d
    dap        = DepthAwareAvgPool2d(kernel_size=3, padding=1)
    out_pool   = dap(x, depth)
    print("DepthAwareAvgPool2d output shape:", out_pool.shape)  # Expected: [b, c, h, w]
