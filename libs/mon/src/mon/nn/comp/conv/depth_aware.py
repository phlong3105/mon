#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Depth-aware convolutional layers.

This module implements depth-aware convolutional and average pooling layers
that incorporate depth information into the convolution and pooling operations.

References:
    - Paper: "Depth-aware CNN for RGB-D Segmentation," ECCV 2018.
    - Code: https://github.com/laughtervv/DepthAwareCNN
"""

from __future__ import annotations

__all__ = [
    "DepthAwareAvgPool2d",
    "DepthAwareConv2d",
]

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# region DEPTH-AWARE LAYERS
# ==============================================================================

class DepthAwareConv2d(nn.Module):
    """A depth-aware 2D convolution operation.

    This class performs convolution by considering depth similarity, enabling the
    model to incorporate depth information into the convolution process. It modifies
    the standard 2D convolution by weighting it with depth-aware factors computed
    from the depth tensor.

    Attributes:
        conv (nn.Conv2d): Convolutional layer used to perform standard 2D convolution.
        kernel_size (int): Size of the convolutional kernel.
        padding (int): Padding size to be applied to the convolution operation.
        alpha (float): Scaling factor for depth similarity computation.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : int,
        padding     : int   = 0,
        alpha       : float = 8.3,
    ):
        """Initialize a new instance.
        
        Args:
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolutional kernel.
            padding: Padding size for the convolution. Defaults to 0.
            alpha: Scaling factor for depth similarity. Defaults to 8.3 (from paper).
        """
        super().__init__()
        self.conv        = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.kernel_size = kernel_size
        self.padding     = padding
        self.alpha       = alpha

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor with dimensions (B, C_in, H, W) and values ranging
                from 0.0 to 1.0.
            d: Depth tensor with dimensions (B, 1, H, W) and values ranging
                from 0.0 to 1.0.
            
        Returns:
            Output tensor with dimensions (B, C_out, H_out, W_out) and values
            ranging from 0.0 to 1.0.
        """
        # input: [b, channels, h, w]
        # depth: [b, 1,        h, w]
        b, _, h, w = x.size()

        # Extract patches for depth similarity computation
        h_kernel, w_kernel = self.kernel_size, self.kernel_size
        padding    = self.padding
        d_padded   = F.pad(d, (padding, padding, padding, padding), mode="replicate")
        d_unfolded = F.unfold(d_padded, kernel_size=(h_kernel, w_kernel), stride=1, padding=0)
        d_unfolded = d_unfolded.view(b, 1, h_kernel * w_kernel, h * w)

        # Center depth values
        d_center = d.view(b, 1, 1, h * w)

        # Compute depth difference and similarity
        depth_diff = d_unfolded - d_center
        F_D = torch.exp(-self.alpha * torch.abs(depth_diff))  # [batch, 1, kernel_size^2, h*w]

        # Reshape F_D to match conv output for element-wise multiplication.
        F_D = F_D.view(b, 1, h_kernel, w_kernel, h, w)
        F_D = F_D.permute(0, 4, 1, 2, 3, 5).reshape(b, 1, h * w_kernel, w * h_kernel)
        F_D = F_D[:, :, padding:h + padding, padding:w + padding]  # Adjust for padding
        
        # Apply depth similarity to standard convolution output
        return self.conv(x) * F_D


class DepthAwareAvgPool2d(nn.Module):
    """Depth-aware average pooling for 2D input tensors.

    This class implements a custom pooling layer that computes an average pooling
    operation by incorporating depth similarity as a weighting factor. The
    depth-aware pooling assigns higher weights to spatial values closer in depth,
    resulting in a more contextually aware aggregation of features. It is
    particularly useful for tasks that demand depth-awareness, such as depth-guided
    segmentation or reconstruction.

    Attributes:
        kernel_size (int): Size of the pooling kernel.
        stride (int): Stride of the pooling operation.
        padding (int): Padding size for the pooling operation.
        alpha (float): Scaling factor for depth similarity.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        kernel_size: int,
        stride     : int   = 1,
        padding    : int   = 0,
        alpha      : float = 8.3,
    ):
        """Initialize a new instance.
        
        Args:
            kernel_size: Size of the pooling kernel.
            stride: Stride of the pooling operation. Defaults to 1.
            padding: Padding size for the pooling operation. Defaults to 0.
            alpha: Scaling factor for depth similarity. Defaults to 8.3 (from paper).
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride      = stride
        self.padding     = padding
        self.alpha       = alpha

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor with dimensions (B, C, H, W) and values ranging
                from 0.0 to 1.0.
            d: Depth tensor with dimensions (B, 1, H, W) and values ranging
                from 0.0 to 1.0.
            
        Returns:
            Output tensor with dimensions (B, C, H_out, W_out) and values ranging
            from 0.0 to 1.0.
        """
        # input: [b, c, h, w]
        # depth: [b, 1, h, w]
        b, c, h, w = x.size()

        # Pad input and depth for pooling
        x_padded = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode="replicate")
        d_padded = F.pad(d, (self.padding, self.padding, self.padding, self.padding), mode="replicate")
        
        # Extract patches using unfold
        x_unfolded = F.unfold(x_padded, kernel_size=self.kernel_size, stride=self.stride)
        d_unfolded = F.unfold(d_padded, kernel_size=self.kernel_size, stride=self.stride)

        # Reshape for computation
        b, c_in, h_out, w_out = x_unfolded.size(0), x_unfolded.size(1) // (self.kernel_size * self.kernel_size), x_unfolded.size(2), 1
        x_unfolded = x_unfolded.view(b, c, self.kernel_size * self.kernel_size, h_out * w_out)
        d_unfolded = d_unfolded.view(b, 1, self.kernel_size * self.kernel_size, h_out * w_out)

        # Center depth values
        d_center = d.unfold(2, self.stride, self.stride).unfold(3, self.stride, self.stride)
        d_center = d_center.contiguous().view(b, 1, 1, h_out * w_out)

        # Compute depth similarity
        d_diff = d_unfolded - d_center
        F_D    = torch.exp(-self.alpha * torch.abs(d_diff))  # [b, 1, kernel_size^2, h_out*w_out]

        # Weighted average pooling
        weighted_sum = torch.sum(F_D * x_unfolded, dim=2, keepdim=True)  # Sum over kernel
        fd_sum       = torch.sum(F_D, dim=2, keepdim=True)  # Normalize
        y            = weighted_sum / (fd_sum + 1e-8)       # Avoid division by zero
        
        # Reshape to [b, c, h_out, w_out]
        y = y.view(b, c, h_out, w_out)
        return y

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

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

# endregion
