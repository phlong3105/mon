#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Vision-related Losses.

This module provides various loss functions commonly used in computer vision
tasks.
"""

from __future__ import annotations

__all__ = [
    "ColorConstancyLoss",
    "EdgeLoss",
    "EdgePreservingLoss",
    "ExposureControlLoss",
    "ExposureValueControlLoss",
    "PSNRLoss",
    "SpatialConsistencyLoss",
    "TotalVariationLoss",
]

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .base import Loss
from .common import CharbonnierLoss


# ==============================================================================
# region PIXEL & INTENSITY LOSSES
# ==============================================================================

class ExposureControlLoss(Loss):
    """Loss function for managing the average luminance.

    References:
        - https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py#L74
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        patch_size: int = 16,
        E: float = 0.6,
        required_grad: bool = True,
        channel_mean: bool = True,
        reduction: str = "mean",
    ):
        """Initialize a new instance.

        Args:
            patch_size (int, optional): Kernel size for pooling layer.
                Defaults to 16.
            E (float, optional): Well-exposedness level E. Defaults to 0.6.
            required_grad (bool, optional): If True, ``mean_val`` is learnable.
                Defaults to True.
            channel_mean (bool, optional): If True, compute the mean across
                channels before pooling. Defaults to True.
            reduction (str, optional): Reduction method to apply to the loss.
                One of: ["mean", "sum", "none"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.channel_mean = channel_mean

        # Registering as a buffer if not learnable to ensure it moves to the
        # correct device
        if not required_grad:
            self.register_buffer("target_exposure", torch.tensor([E]))
        else:
            self.target_exposure = nn.Parameter(torch.tensor([E]))

        self.pool = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)

    # --- Callable & Context Manager ---
    def forward(self, input: Tensor) -> Tensor:
        """Calculate the loss between the ``input`` and ``target_exposure``.

        Args:
            input (Tensor): Input (predictions) tensor of shape (B, C, H, W)
                and values ranging from 0.0 to 1.0.

        Returns:
            Tensor: Loss value.
        """
        # Compute local means
        x = torch.mean(input, dim=1, keepdim=True) if self.channel_mean else input
        local_mean = self.pool(x)
        # L2 distance to target exposure
        loss = torch.pow(local_mean - self.target_exposure, 2)
        # Apply reduction
        loss = self.reduce(loss=loss)
        return loss


class ExposureValueControlLoss(Loss):
    """Variation of ``ExposureControlLoss`` for non-linear exposure adjustment.

    References:
        - https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py#L74
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        patch_size: int = 16,
        E: float = 0.6,
        eps: float = 1e-6,
        required_grad: bool = True,
        channel_mean: bool = True,
        reduction: str = "mean",
    ):
        """Initialize a new instance.

        Args:
            patch_size (int, optional): Kernel size for pooling layer.
                Defaults to 16.
            E (float, optional): Well-exposedness level E. Defaults to 0.6.
            eps (float, optional): Small constant for numerical stability.
                Defaults to 1e-6.
            required_grad (bool, optional): If True, ``mean_val`` is learnable.
                Defaults to True.
            channel_mean (bool, optional): If True, compute the mean across
                channels before pooling. Defaults to True.
            reduction (str, optional): Reduction method to apply to the loss.
                One of: ["mean", "sum", "none"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.channel_mean = channel_mean
        self.eps = eps

        # Registering as a buffer if not learnable to ensure it moves to the correct device
        if not required_grad:
            self.register_buffer("target_exposure", torch.tensor([E]))
        else:
            self.target_exposure = nn.Parameter(torch.tensor([E]))

        self.pool = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)

    # --- Callable & Context Manager ---
    def forward(self, input: Tensor) -> Tensor:
        """Calculate the loss between the ``input`` and ``target_exposure``.

        Args:
            input (Tensor): Input (predictions) tensor of shape (B, C, H, W)
                and values ranging from 0.0 to 1.0.

        Returns:
            Tensor: Loss value.
        """
        # Compute local means
        x = torch.mean(input, dim=1, keepdim=True) if self.channel_mean else input
        local_mean = self.pool(x)
        local_mean = torch.sqrt(local_mean + self.eps)
        # L2 distance to target exposure
        loss = torch.pow(local_mean - self.target_exposure, 2)
        # Apply reduction
        loss = self.reduce(loss=loss)
        return loss

# endregion


# ==============================================================================
# region COLOR & FIDELITY LOSSES
# ==============================================================================

class ColorConstancyLoss(Loss):
    """Loss function for preventing color shifting by balancing RGB means.

    References:
        - https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py#L74
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """Initialize a new instance.

        Args:
            eps (float, optional): Small constant for numerical stability.
                Defaults to 1e-6.
            reduction (str, optional): Reduction method to apply to the loss.
                One of: ["mean", "sum", "none"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.eps = eps

    # --- Callable & Context Manager ---
    def forward(self, input: Tensor) -> Tensor:
        """Calculate the loss for the ``input``.

        Args:
            input (Tensor): Input (predictions) tensor of shape (B, C, H, W)
                and values ranging from 0.0 to 1.0.

        Returns:
            Tensor: Loss value.
        """
        # Calculate the mean of each channel (B, C, 1, 1)
        # Using [2, 3] flattens the spatial dimensions
        mean_rgb = torch.mean(input, dim=[2, 3], keepdim=True)

        # Extract channels for clear pairwise comparison
        # (Alternatively, one could use itertools.combinations for N-channels)
        r, g, b = mean_rgb[:, 0:1], mean_rgb[:, 1:2], mean_rgb[:, 2:3]

        # Calculate squared differences
        # Using L2 norm of the differences for stability and standard behavior
        d_rg = (r - g) ** 2
        d_rb = (r - b) ** 2
        d_gb = (g - b) ** 2

        # Final loss: sqrt of the sum of squared differences
        # This is essentially the standard deviation between channel means
        loss = torch.sqrt(d_rg + d_rb + d_gb + self.eps)

        # Apply reduction
        loss = self.reduce(loss=loss)
        return loss


class PSNRLoss(Loss):
    """Loss function based on Peak Signal-to-Noise Ratio (PSNR)."""

    # --- Lifecycle & Initialization ---
    def __init__(self, to_y: bool = False, reduction: str = "mean"):
        """Initialize a new instance.

        Args:
            to_y (bool, optional): If True, convert RGB to Y-channel before
                computing PSNR. Defaults to False.
            reduction (str, optional): Reduction method to apply to the loss.
                One of: ["mean", "sum", "none"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.to_y = to_y

        # Registering as buffer handles device movement and serialization
        coef = torch.tensor([65.481, 128.553, 24.966]).view(1, 3, 1, 1)
        self.register_buffer("coef", coef)

    # --- Callable & Context Manager ---
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Calculate the loss between ``input`` and ``target``.

        Args:
            input (Tensor): Input (predictions) tensor of shape (B, C, H, W)
                and values ranging from 0.0 to 1.0.
            target (Tensor): Target (ground truth) tensor of shape (B, C, H, W)
                and values ranging from 0.0 to 1.0.

        Returns:
            Tensor: Loss value.
        """
        # Handle Y-channel conversion if requested
        if self.to_y:
            # Scaled RGB to Y conversion (BT.601)
            input = (input * self.coef).sum(dim=1, keepdim=True) + 16.0
            target = (target * self.coef).sum(dim=1, keepdim=True) + 16.0
            input = input / 255.0
            target = target / 255.0

        # Calculate MSE per image in batch
        # We don't use F.mse_loss here to maintain per-sample control before log
        mse = torch.mean((input - target) ** 2, dim=(1, 2, 3))

        # Calculate PSNR
        # 1e-8 prevents log10(0)
        psnr = 20 * torch.log10(1.0 / (torch.sqrt(mse) + 1e-8))

        # Transform to Loss (Lower is better)
        # Based on your logic: 50dB -> 0.0 loss, 0dB -> 0.5 loss
        loss = (50.0 - psnr) / 100.0

        # Apply reduction
        loss = self.reduce(loss=loss)
        return loss

# endregion


# ==============================================================================
# region SPATIAL & STRUCTURAL LOSSES
# ==============================================================================

class EdgeLoss(Loss):
    """Loss function for penalizing blurry boundaries.

    Preserve edge details in images by computing the Laplacian edge maps and
    penalizing differences between the input and target images.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, reduction: str = "mean"):
        """Initialize a new instance.

        Args:
            reduction (str, optional): Reduction method to apply to the loss.
                One of: ["mean", "sum", "none"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        # Create 5x5 Gaussian Kernel
        k = Tensor([[0.05, 0.25, 0.4, 0.25, 0.05]])
        kernel = torch.matmul(k.t(), k).unsqueeze(0).unsqueeze(0)  # [1, 1, 5, 5]
        # Register as buffer to handle device placement automatically
        self.register_buffer("kernel", kernel.repeat(3, 1, 1, 1))

        self.charbonnier = CharbonnierLoss(eps=1e-3, reduction="none")

    # --- Callable & Context Manager ---
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Calculate the loss between ``input`` and ``target``.

        Args:
            input (Tensor): Input (predictions) tensor of shape (B, C, H, W)
                and values ranging from 0.0 to 1.0.
            target (Tensor): Target (ground truth) tensor of shape (B, C, H, W)
                and values ranging from 0.0 to 1.0.

        Returns:
            Tensor: Loss value.
        """
        # Extract edge maps
        input_edges = self._laplacian(input)
        target_edges = self._laplacian(target)

        # Calculate Charbonnier loss on the edge maps
        # Using your existing class preserves architectural consistency
        loss = self.charbonnier(input_edges, target_edges)

        # Apply reduction
        loss = self.reduce(loss=loss)
        return loss

    def _laplacian(self, image: Tensor) -> Tensor:
        """Compute the Laplacian edge map using a Gaussian pyramid.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.

        Returns:
            Tensor: Laplacian edge map.
        """
        filtered = self._gauss_conv(image)
        # Downsample and Upsample (Stride 2)
        down = filtered[:, :, ::2, ::2]
        up   = torch.zeros_like(filtered)
        up[:, :, ::2, ::2] = down * 4
        # Second blur to smooth the upsampled grid
        up_blurred = self._gauss_conv(up)
        return image - up_blurred

    def _gauss_conv(self, image: Tensor) -> Tensor:
        """Apply Gaussian convolution to the ``image``.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.

        Returns:
            Tensor: Gaussian filtered image.
        """
        # TODO: Delete later
        """
        b, c, w, h  = self.kernel.shape
        self.kernel = self.kernel.to(image.device)
        image       = F.pad(image, (w // 2, h // 2, w // 2, h // 2), mode="replicate")
        # gauss       = F.conv2d(image, self.kernel, groups=b)  # Old code
        gauss       = F.conv2d(image, self.kernel, groups=c)  # Groups=c for channel-wise convolution
        return gauss
        """
        # Replicate padding prevents edge artifacts in the laplacian map
        x = F.pad(image, (2, 2, 2, 2), mode="replicate")
        # groups=3 ensures each RGB channel is blurred independently
        return F.conv2d(x, self.kernel, groups=3)


class EdgePreservingLoss(Loss):
    """Loss function for preserving edge details by comparing Sobel gradients.

    Preserve edge details in images by computing the Sobel gradients and
    comparing the differences between the input and target images. This is
    particularly useful for tasks like denoising where maintaining sharp edges
    is crucial. If the network deletes text, the predicted edges will be empty,
    leading to a high loss and encouraging the model to retain those details.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, reduction: str = "mean"):
        """Initialize a new instance.

        Args:
            reduction (str, optional): Reduction method to apply to the loss.
                One of: ["mean", "sum", "none"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        # Sobel edge detection kernels
        self.kernel_x = torch.tensor([[-1.0,  0.0,  1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        self.kernel_y = torch.tensor([[-1.0, -2.0, -1.0], [ 0.0, 0.0, 0.0], [ 1.0, 2.0, 1.0]]).view(1, 1, 3, 3)

    # --- Callable & Context Manager ---
    def forward(self, input: Tensor, pred: Tensor) -> Tensor:
        """Calculate the loss between ``input`` and ``pred``.

        Args:
            input (Tensor): Input tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            pred (Tensor): Prediction tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.

        Returns:
            Tensor: Loss value.
        """
        c = input.shape[1]
        # Expand kernels to match image channels (e.g., 3 for RGB)
        kx = self.kernel_x.expand(c, 1, 3, 3).to(input.device)
        ky = self.kernel_y.expand(c, 1, 3, 3).to(input.device)

        # Calculate edges (gradients) for the original noisy image
        grad_x_noisy = F.conv2d(input, kx, padding=1, groups=c)
        grad_y_noisy = F.conv2d(input, ky, padding=1, groups=c)

        # Calculate edges for the predicted denoised image
        grad_x_denoised = F.conv2d(pred, kx, padding=1, groups=c)
        grad_y_denoised = F.conv2d(pred, ky, padding=1, groups=c)

        # The loss is the difference in edges.
        # If the network deleted text, grad_denoised will be empty here, causing a high loss.
        loss_x = torch.mean(torch.abs(grad_x_noisy - grad_x_denoised))
        loss_y = torch.mean(torch.abs(grad_y_noisy - grad_y_denoised))

        return loss_x + loss_y


class SpatialConsistencyLoss(Loss):
    """Loss function for maintaining local gradients (crucial for Zero-DCE
    architectures).

    Ensure spatial coherence in the enhanced image by penalizing discrepancies
    in local gradients between the enhanced and input images.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        num_regions: int = 4,
        patch_size: int = 4,
        reduction: str = "mean",
    ):
        """Initialize a new instance.

        Args:
            num_regions (int, optional): Number of regions to consider for
                gradient comparison. One of: [4, 8, 16, 24]. Defaults to 4.
            patch_size (int, optional): Patch size for local patch pooling.
                Defaults to 4.
            reduction (str, optional): Reduction method to apply to the loss.
                One of: ["mean", "sum", "none"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.num_regions = num_regions
        self.pool = nn.AvgPool2d(patch_size)

        # Initialize all 24 kernels (5x5 grid to accommodate 2-pixel jumps)
        kernels = self._get_24_kernels()

        # Shape: [24, 1, 5, 5]
        weight_stack = torch.stack(kernels).unsqueeze(1)
        self.register_buffer("weight_stack", weight_stack)

    def _get_24_kernels(self) -> list[Tensor]:
        """Return 24 kernels for gradient comparison."""
        # Center of 5x5 grid is (2, 2)
        base = torch.zeros(5, 5)
        base[2, 2] = 1
        kernels = []

        # Coordinates for 24 neighbors relative to (2,2)
        offsets = [
            # 1-pixel neighbors (8)
            (1,2), (3,2), (2,1), (2,3), (1,1), (1,3), (3,1), (3,3),
            # 2-pixel neighbors (8)
            (0,2), (4,2), (2,0), (2,4), (0,0), (0,4), (4,0), (4,4),
            # Mixed / Knight moves (8)
            (0,1), (0,3), (4,1), (4,3), (1,0), (1,4), (3,0), (3,4)
        ]

        for r, c in offsets[:self.num_regions]:
            k = base.clone()
            k[r, c] = -1
            kernels.append(k)
        return kernels

    # --- Callable & Context Manager ---
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Calculate the loss between ``input`` and ``target``.

        Args:
            input (Tensor): Input (predictions) tensor of shape (B, C, H, W)
                and values ranging from 0.0 to 1.0.
            target (Tensor): Target (ground truth) tensor of shape (B, C, H, W)
                and values ranging from 0.0 to 1.0.

        Returns:
            Tensor: Loss value.
        """
        # Convert to luminance
        mu_in = torch.mean(input, dim=1, keepdim=True)
        mu_out = torch.mean(target, dim=1, keepdim=True)

        # Patch-wise pooling
        p_in, p_out = self.pool(mu_in), self.pool(mu_out)

        # Single-pass 24-direction convolution
        # We use padding=2 because our kernels are 5x5
        grads_in = F.conv2d(p_in, self.weight_stack, padding=2)
        grads_out = F.conv2d(p_out, self.weight_stack, padding=2)

        # Squared difference across all 24 directions
        loss = torch.pow(grads_in - grads_out, 2)

        # Sum directions [B, 24, H, W] -> [B, 1, H, W]
        # loss = self.reduce(loss.sum(dim=1, keepdim=True))

        # Apply reduction
        loss = self.reduce(loss=loss)
        return loss


class TotalVariationLoss(Loss):
    """Loss function for reducing noise by encouraging piecewise smoothness.

    Encourage spatial smoothness in the enhanced image by penalizing large
    intensity variations between neighboring pixels.

    References:
        - https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, reduction: str = "mean"):
        """Initialize a new instance.

        Args:
            reduction (str, optional): Reduction method to apply to the loss.
                One of: ["mean", "sum", "none"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)

    # --- Callable & Context Manager ---
    def forward(self, input: Tensor) -> Tensor:
        """Calculate the loss for the ``input``.

        Args:
            input (Tensor): Input (predictions) tensor of shape (B, C, H, W)
                and values ranging from 0.0 to 1.0.

        Returns:
            Tensor: Loss value.
        """
        x = input

        # Calculate variations between adjacent pixels
        # h_diff: (B, C, H-1, W)
        # w_diff: (B, C, H, W-1)
        h_diff = x[:, :, 1:, :] - x[:, :, :-1, :]
        w_diff = x[:, :, :, 1:] - x[:, :, :, :-1]

        # L2-style Total Variation (Isotropic approximation)
        # We calculate the mean over C, H, W for each image in the batch
        # to keep the loss scale-invariant.
        h_tv = torch.mean(torch.pow(h_diff, 2), dim=(1, 2, 3))
        w_tv = torch.mean(torch.pow(w_diff, 2), dim=(1, 2, 3))

        # Combine and apply Loss reduction (mean/sum/none)
        loss = h_tv + w_tv

        # Apply reduction
        loss = self.reduce(loss=loss)
        return loss

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
