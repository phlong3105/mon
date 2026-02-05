#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image loss functions.

This module provides various loss functions commonly used in training deep
learning models where the outputs and targets are images.
"""

from __future__ import annotations

__all__ = [
    "ColorConstancyLoss",
    "DepthAwareIlluminationLoss",
    "EdgeLoss",
    "ExposureControlLoss",
    "ExposureValueControlLoss",
    "PSNRLoss",
    "SpatialConsistencyLoss",
    "StructureTextureDecompositionLoss",
    "TotalVariationLoss",
]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

from .base import BaseLoss
from .basic import CharbonnierLoss


# ==============================================================================
# region PIXEL & INTENSITY LOSSES
# ==============================================================================

class ExposureControlLoss(BaseLoss):
    """Loss function for managing the average luminance.

    References:
        - https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py#L74

    Attributes:
        channel_mean: If True, compute mean across channels before pooling.
        target_exposure: Well-exposedness level E.
        pool: Pooling layer for computing local means.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        patch_size   : int   = 16,
        mean_val     : float = 0.6,
        required_grad: bool  = True,
        channel_mean : bool  = True,
        reduction    : str   = "mean",
    ):
        """Initialize a new instance.

        Args:
            patch_size: Kernel size for pooling layer. Defaults to 16.
            mean_val: Well-exposedness level E. Defaults to 0.6.
            required_grad: If True, ``mean_val`` is learnable. Defaults to True.
            channel_mean: If True, compute the mean across channels before
                pooling. Defaults to True.
            reduction: Reduction method to apply to the loss. Can be one of
                ["none", "mean", "sum"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.channel_mean = channel_mean

        # Registering as a buffer if not learnable to ensure it moves to the correct device
        if not required_grad:
            self.register_buffer("target_exposure", torch.tensor([mean_val]))
        else:
            self.target_exposure = nn.Parameter(torch.tensor([mean_val]))

        self.pool = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)

    # --- Callable & Context Manager ---
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Calculate the loss between the ``input`` and the target exposure.

        Args:
            input: Predicted image, formatted as a torch.Tensor of shape
                (B, C, H, W) and values ranging from 0.0 to 1.0.

        Returns:
            Loss value.
        """
        # Compute local means
        x = torch.mean(input, dim=1, keepdim=True) if self.channel_mean else input

        # Patch-wise average intensity
        local_mean = self.pool(x)

        # L2 distance to target exposure
        loss = torch.pow(local_mean - self.target_exposure, 2)

        # TODO: Delete later
        """
        x = input
        if self.channel_mean:
            x = torch.mean(input, 1, keepdim=True)
        mean = self.pool(x)
        loss = torch.pow(mean - self.mean_val, 2)
        loss = self.reduce(loss=loss)
        """

        loss = self.reduce(loss=loss)
        return loss


class ExposureValueControlLoss(BaseLoss):
    """Variation of ExposureControlLoss for non-linear exposure adjustment.

    References:
        - https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py#L74

    Attributes:
        channel_mean: If True, compute mean across channels before pooling.
        target_exposure: Well-exposedness level E.
        pool: Pooling layer for computing local means.
    """

    # --- Lifecycle & Initialization ---

    def __init__(
        self,
        patch_size   : int   = 16,
        mean_val     : float = 0.6,
        eps          : float = 1e-6,
        required_grad: bool  = True,
        channel_mean : bool  = True,
        reduction    : str   = "mean",
    ):
        """Initialize a new instance.

        Args:
            patch_size: Kernel size for pooling layer. Defaults to 16.
            mean_val: Well-exposedness level E. Defaults to 0.6.
            eps: Small constant for numerical stability. Defaults to 1e-6.
            required_grad: If True, ``mean_val`` is learnable. Defaults to True.
            channel_mean: If True, compute the mean across channels before
                pooling. Defaults to True.
            reduction: Reduction method to apply to the loss. Can be one of
                ["none", "mean", "sum"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.channel_mean = channel_mean
        self.eps = eps

        # Registering as a buffer if not learnable to ensure it moves to the correct device
        if not required_grad:
            self.register_buffer("target_exposure", torch.tensor([mean_val]))
        else:
            self.target_exposure = nn.Parameter(torch.tensor([mean_val]))

        self.pool = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)

    # --- Callable & Context Manager ---
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Calculate the loss between the ``input`` and the target exposure.

        Args:
            input: Predicted image, formatted as a torch.Tensor of shape
                (B, C, H, W) and values ranging from 0.0 to 1.0.

        Returns:
            Loss value.
        """
        # Compute local means
        x = torch.mean(input, dim=1, keepdim=True) if self.channel_mean else input

        # Non-linear patch pooling
        # Adding eps prevents derivative issues at zero
        pooled_mean     = self.pool(x)
        non_linear_mean = torch.sqrt(pooled_mean + self.eps)

        # L2 distance to target exposure
        loss = torch.pow(non_linear_mean - self.target_exposure, 2)

        # TODO: Delete later
        """
        x = input
        if self.channel_mean:
            x = torch.mean(x, 1, keepdim=True)  # Channel-wise mean: [B, 1, H, W]
        mean = self.pool(x) ** 0.5              # Pooled mean:       [B, 1, H, W]
        loss = torch.pow((mean - self.mean_val), 2)
        loss = torch.abs(torch.mean(loss))
        """

        loss = self.reduce(loss=loss)
        return loss

# endregion


# ==============================================================================
# region COLOR & FIDELITY LOSSES
# ==============================================================================

class ColorConstancyLoss(BaseLoss):
    """Loss function for preventing color shifting by balancing RGB means.

    References:
        - https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py#L74

    Attributes:
        eps: Small constant for numerical stability.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """Initialize a new instance.

        Args:
            eps: Small constant for numerical stability. Defaults to 1e-6.
            reduction: Reduction method to apply to the loss. Can be one of
                ["none", "mean", "sum"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.eps = eps

    # --- Callable & Context Manager ---
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Calculate the loss for the ``input``.

        Args:
            input: Predicted image, formatted as a torch.Tensor of shape
                (B, C, H, W) and values ranging from 0.0 to 1.0.

        Returns:
            Loss value.
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

        # TODO: Delete later
        """
        mean_rgb   = torch.mean(input, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        d_rg       = torch.pow(torch.abs(mr - mg), 2)
        d_rb       = torch.pow(torch.abs(mr - mb), 2)
        d_gb       = torch.pow(torch.abs(mb - mg), 2)
        d_rg2      = torch.pow(d_rg, 2)
        d_rb2      = torch.pow(d_rb, 2)
        d_gb2      = torch.pow(d_gb, 2)
        loss       = d_rg2 + d_rb2 + d_gb2
        loss       = torch.pow(loss + self.eps, 0.5)
        """

        loss = self.reduce(loss=loss)
        return loss


class PSNRLoss(BaseLoss):
    """Loss function based on Peak Signal-to-Noise Ratio (PSNR).

    Attributes:
        to_y: If True, use Y-channel for computing PSNR.
    """

    # --- Lifecycle & Initialization ---

    def __init__(self, to_y: bool = False, reduction: str = "mean"):
        """Initialize a new instance.

        Args:
            to_y: If True, use Y-channel for computing PSNR. Defaults to False.
            reduction: Reduction method to apply to the loss. Can be one of
                ["none", "mean", "sum"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.to_y = to_y

        # Registering as buffer handles device movement and serialization
        coef = torch.tensor([65.481, 128.553, 24.966]).view(1, 3, 1, 1)
        self.register_buffer("coef", coef)

    # --- Callable & Context Manager ---
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the loss between ``input`` and ``target``.

        Args:
            input: Predicted image, formatted as a torch.Tensor of shape
                (B, C, H, W) and values ranging from 0.0 to 1.0.
            target: Target image, formatted as a torch.Tensor of shape
                (B, C, H, W) and values ranging from 0.0 to 1.0.

        Returns:
            Loss value.
        """
        # Handle Y-channel conversion if requested
        if self.to_y:
            # Scaled RGB to Y conversion (BT.601)
            input  = (input  * self.coef).sum(dim=1, keepdim=True) + 16.0
            target = (target * self.coef).sum(dim=1, keepdim=True) + 16.0
            input  = input  / 255.0
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

        loss = self.reduce(loss=loss)
        return loss

# endregion


# ==============================================================================
# region SPATIAL & STRUCTURAL LOSSES
# ==============================================================================

class SpatialConsistencyLoss(BaseLoss):
    """Loss function for maintaining local gradients (crucial for Zero-DCE
    architectures).

    Ensure spatial coherence in the enhanced image by penalizing discrepancies
    in local gradients between the enhanced and input images.

    Attributes:
        num_regions: Number of directional regions to consider for gradient comparison.
        pool: Pooling layer for blurring.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        num_regions: int = 4,
        patch_size : int = 4,
        reduction  : str = "mean",
    ):
        """Initialize a new instance.

        Args:
            num_regions: Number of directional regions to consider for gradient
                comparison. Can be one of [4, 8, 16, 24]. Defaults to 4.
            patch_size: Size of the Gaussian kernel for blurring. Defaults to 4.
            reduction: Reduction method to apply to the loss. Can be one of
                ["none", "mean", "sum"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.num_regions = num_regions
        self.pool        = nn.AvgPool2d(patch_size)

        # Initialize all 24 kernels (5x5 grid to accommodate 2-pixel jumps)
        kernels = self._get_24_kernels()

        # Shape: [24, 1, 5, 5]
        weight_stack = torch.stack(kernels).unsqueeze(1)
        self.register_buffer("weight_stack", weight_stack)

    def _get_24_kernels(self) -> list[torch.Tensor]:
        """Return 24 kernels for gradient comparison."""
        # Center of 5x5 grid is (2, 2)
        base       = torch.zeros(5, 5)
        base[2, 2] = 1
        kernels    = []

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
            k       = base.clone()
            k[r, c] = -1
            kernels.append(k)
        return kernels

    # --- Callable & Context Manager ---
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the loss between ``input`` and ``target``.

        Args:
            input: Predicted image, formatted as a torch.Tensor of shape
                (B, C, H, W) and values ranging from 0.0 to 1.0.
            target: Target image, formatted as a torch.Tensor of shape
                (B, C, H, W) and values ranging from 0.0 to 1.0.

        Returns:
            Loss value.
        """
        # Convert to luminance
        mu_in  = torch.mean(input,  dim=1, keepdim=True)
        mu_out = torch.mean(target, dim=1, keepdim=True)

        # Patch-wise pooling
        p_in, p_out = self.pool(mu_in), self.pool(mu_out)

        # Single-pass 24-direction convolution
        # We use padding=2 because our kernels are 5x5
        grads_in  = F.conv2d(p_in,  self.weight_stack, padding=2)
        grads_out = F.conv2d(p_out, self.weight_stack, padding=2)

        # Squared difference across all 24 directions
        loss = torch.pow(grads_in - grads_out, 2)

        # Sum directions [B, 24, H, W] -> [B, 1, H, W]
        # loss = self.reduce(loss.sum(dim=1, keepdim=True))

        loss = self.reduce(loss=loss)
        return loss


class TotalVariationLoss(BaseLoss):
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
            reduction: Reduction method to apply to the loss. Can be one of
                ["none", "mean", "sum"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)

    # --- Callable & Context Manager ---
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Calculate the loss for the ``input`` tensor.

        Args:
            input: Predicted image, formatted as a torch.Tensor of shape
                (B, C, H, W) and values ranging from 0.0 to 1.0.

        Returns:
            Loss value.
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

        # Combine and apply BaseLoss reduction (mean/sum/none)
        loss = h_tv + w_tv

        # TODO: Delete later
        """
        x          = input
        b, c, h, w = x.size()
        count_h    = (x.size()[2]-1) * x.size()[3]
        count_w    =  x.size()[2] * (x.size()[3] - 1)
        h_tv       = torch.pow((x[:, :, 1:,  :] - x[:, :, :h - 1, :]), 2).sum()
        w_tv       = torch.pow((x[:, :,  :, 1:] - x[:, :, :, :w - 1]), 2).sum()
        loss       = 2 * (h_tv / count_h + w_tv / count_w) / b
        """

        loss = self.reduce(loss=loss)
        return loss


class EdgeLoss(BaseLoss):
    """Loss function for penalizing blurry boundaries.

    Preserve edge details in images by computing the Laplacian edge maps and
    penalizing differences between the input and target images.

    Attributes:
        charbonnier: Charbonnier loss instance for edge map comparison.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, reduction: str = "mean"):
        """Initialize a new instance.

        Args:
            reduction: Reduction method to apply to the loss. Can be one of
                ["none", "mean", "sum"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        # Create 5x5 Gaussian Kernel
        k      = torch.Tensor([[0.05, 0.25, 0.4, 0.25, 0.05]])
        kernel = torch.matmul(k.t(), k).unsqueeze(0).unsqueeze(0)  # [1, 1, 5, 5]
        # Register as buffer to handle device placement automatically
        self.register_buffer("kernel", kernel.repeat(3, 1, 1, 1))

        self.charbonnier = CharbonnierLoss(eps=1e-3, reduction="none")

    # --- Callable & Context Manager ---
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the loss between ``input`` and ``target``.

        Args:
            input: Predicted image, formatted as a torch.Tensor of shape
                (B, C, H, W) and values ranging from 0.0 to 1.0.
            target: Target image, formatted as a torch.Tensor of shape
                (B, C, H, W) and values ranging from 0.0 to 1.0.

        Returns:
            Loss value.
        """
        # Extract edge maps
        input_edges  = self._laplacian(input)
        target_edges = self._laplacian(target)

        # Calculate Charbonnier loss on the edge maps
        # Using your existing class preserves architectural consistency
        loss = self.charbonnier(input_edges, target_edges)

        loss = self.reduce(loss=loss)
        return loss

    def _laplacian(self, image: torch.Tensor) -> torch.Tensor:
        """Compute the Laplacian edge map using a Gaussian pyramid.

        Args:
            image: An image, formatted as a torch.Tensor of shape
                (B, C, H, W) and values ranging from 0.0 to 1.0.

        Returns:
            Laplacian edge map.
        """
        filtered = self._gauss_conv(image)
        # Downsample and Upsample (Stride 2)
        down = filtered[:, :, ::2, ::2]
        up   = torch.zeros_like(filtered)
        up[:, :, ::2, ::2] = down * 4
        # Second blur to smooth the upsampled grid
        up_blurred = self._gauss_conv(up)
        return image - up_blurred

    def _gauss_conv(self, image: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian convolution to the ``image``.

        Args:
            image: An image, formatted as a torch.Tensor of shape
                (B, C, H, W) and values ranging from 0.0 to 1.0.

        Returns:
            Gaussian filtered image.
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

# endregion


# ==============================================================================
# region GEOMETRY & CONTEXT-AWARE LOSSES
# ==============================================================================

class DepthAwareIlluminationLoss(BaseLoss):
    """Loss function for smoothing lighting while respecting 3D depth boundaries.

    Encourage smoothness in the illumination map while preserving depth
    discontinuities.

    Attributes:
        alpha: Weighting factor for depth influence.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, alpha: float = 1.0, reduction: str = "mean"):
        """Initialize a new instance.

        Args:
            alpha: Weighting factor for depth influence. Defaults to 1.0.
            reduction: Reduction method to apply to the loss. Can be one of
                ["none", "mean", "sum"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.alpha = alpha

    # --- Callable & Context Manager ---
    def forward(self, input: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Calculate the loss between illumination map and depth map.

        Args:
            input: Illumination map, formatted as a torch.Tensor of shape
                (B, 1, H, W) and values ranging from 0.0 to 1.0.
            depth: Depth map, formatted as a torch.Tensor of shape
                (B, 1, H, W) and values ranging from 0.0 to 1.0.

        Returns:
            Loss value.
        """
        # Illumination gradients (L)
        L_dx = input[:, :, :, 1:] - input[:, :, :, :-1]
        L_dy = input[:, :, 1:, :] - input[:, :, :-1, :]

        # Depth gGradients (D)
        D_dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
        D_dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]

        # Calculate Weights
        # Higher depth gradient -> Smaller weight -> Lower penalty for lighting changes
        weight_dx = torch.exp(-self.alpha * torch.abs(D_dx))
        weight_dy = torch.exp(-self.alpha * torch.abs(D_dy))

        # Apply Weights
        # Using L1 variation (abs) is standard for edge-preserving smoothness
        loss_x = weight_dx * torch.abs(L_dx)
        loss_y = weight_dy * torch.abs(L_dy)

        # Pad gradients back to original size (optional but keeps shapes consistent)
        # or simply average them. Here we sum the directional components:
        # We take the mean over spatial dims for each image in batch first
        loss = loss_x.mean(dim=(1, 2, 3)) + loss_y.mean(dim=(1, 2, 3))

        loss = self.reduce(loss=loss)
        return loss


class StructureTextureDecompositionLoss(BaseLoss):
    """Loss function for separating structural edges from fine details.

    Separates an image into structure and texture components using Gaussian
    blurring and penalizes the texture component to encourage smoother textures
    in the enhanced image.

    Attributes:
        kernel_size: Size of the Gaussian kernel for blurring.
        sigma: Standard deviation for the Gaussian kernel.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        kernel_size: int   = 3,
        sigma      : float = 1.0,
        reduction  : str   = "mean",
    ):
        """Initialize a new instance.

        Args:
            kernel_size: Size of the Gaussian kernel for blurring. Defaults to 3.
            sigma: Standard deviation for the Gaussian kernel. Defaults to 1.0.
            reduction: Reduction method to apply to the loss. Can be one of
                ["none", "mean", "sum"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        # Use a list for kernel/sigma if required by the functional blur
        self.kernel_size = [kernel_size, kernel_size]
        self.sigma       = [sigma, sigma]

    # --- Callable & Context Manager ---
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Calculate the loss for the ``input`` tensor.

        Args:
            input: Predicted image, formatted as a torch.Tensor of shape
                (B, C, H, W) and values ranging from 0.0 to 1.0.

        Returns:
            Loss value.
        """
        # Extract the low-frequency "Structure"
        # Note: Ensure you have torchvision.transforms.functional.gaussian_blur
        # or similar imported as gaussian_blur
        structure = gaussian_blur(input, self.kernel_size, self.sigma)

        # Extract the high-frequency "Texture"
        texture = input - structure

        # Calculate L1 norm (Mean Absolute Error) of the texture
        # We calculate mean over C, H, W for each batch element
        loss = torch.mean(torch.abs(texture), dim=(1, 2, 3))

        loss = self.reduce(loss=loss)
        return loss

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
