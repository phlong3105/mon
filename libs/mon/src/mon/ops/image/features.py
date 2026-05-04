#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Vision Features.

This module provides vision features extraction or prior computation.
"""

from __future__ import annotations

__all__ = [
    "APSF",
    "BoundaryAwarePrior",
    "BrightnessAttentionMap",
]

import kornia
import torch
import torch.nn.functional as F
from torch import nn, Tensor


# ==============================================================================
# region PRIORS
# ==============================================================================

class APSF(nn.Module):
    """Atmospheric Point Spread Function (APSF) module.

    Model the physical scattering of light through different atmospheric
    conditions (air, aerosol, haze, mist, fog, rain).
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        q: float = 0.2,
        t: float = 1.2,
        k: float = 0.5,
        kernel_size: int = 101,
        learnable: bool = False,
    ):
        """Initialize a new instance.

        Args:
            q (float, optional): Forward scattering parameter (0.0 to 1.0).
                Higher = thicker weather:

                - 0.00-0.20: air
                - 0.20-0.70: aerosol
                - 0.70-0.80: haze
                - 0.80-0.85: mist
                - 0.85-0.90: fog
                - 0.90-1.00: rain

            t (float, optional): Optical thickness parameter. Possibly: [0.7, 1.2, 4].
                Defaults to 1.2.
            k (float, optional): Conversion parameter for the kernel. Defaults to 0.5.
            kernel_size (int, optional): Size of the 2D spatial kernel.
            learnable (bool, optional): If True, q, t, and k become network
                weights updated by backprop.
        """
        super().__init__()

        # 1. Ensure kernel size is safely odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size

        # 2. Setup parameters
        if learnable:
            # Allows the network to learn the weather conditions!
            self.q = nn.Parameter(torch.tensor(q, dtype=torch.float32))
            self.t = nn.Parameter(torch.tensor(t, dtype=torch.float32))
            self.k = nn.Parameter(torch.tensor(k, dtype=torch.float32))
        else:
            # Static values that automatically move to the correct device
            self.register_buffer("q", torch.tensor(q, dtype=torch.float32))
            self.register_buffer("t", torch.tensor(t, dtype=torch.float32))
            self.register_buffer("k", torch.tensor(k, dtype=torch.float32))

        # 3. Pre-compute the spatial grid (Saves computation during the forward pass)
        x = torch.linspace(-6, 6, steps=self.kernel_size)
        XX, YY = torch.meshgrid(x, x, indexing="ij")
        radius_sq = XX ** 2 + YY ** 2
        self.register_buffer("radius_sq", radius_sq)

    # --- Callable & Context Manager ---
    def forward(self, image: Tensor) -> Tensor:
        """Applies the atmospheric point spread function to the input image.

        Args:
            image (Tensor): RGB image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.

        Returns:
            Tensor: The blurred image of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.
        """
        # 1. Get the current kernel profile
        kernel_2d = self._compute_kernel()

        # 2. Reshape for depthwise convolution: [Channels, 1, H, W]
        channels = image.shape[1]
        kernel = kernel_2d.view(1, 1, self.kernel_size, self.kernel_size)
        kernel = kernel.repeat(channels, 1, 1, 1)

        # 3. Apply convolution (groups=channels ensures R, G, and B are processed independently)
        apsf_out = F.conv2d(image, kernel, padding="same", groups=channels)

        # 4. Ensure valid pixel range
        return torch.clamp(apsf_out, 0.0, 1.0)

    def _compute_kernel(self) -> Tensor:
        """Dynamically generates the normalized 2D PSF kernel."""
        # Clamp values to prevent math errors during gradient descent
        q = torch.clamp(self.q, min=1e-3, max=0.999)
        t = torch.clamp(self.t, min=1e-3)
        k = self.k

        p = k * t
        sigma = (1 - q) / q

        # THE FIX: Use PyTorch's Log-Gamma (lgamma).
        # We use log rules: ln(A/B) = ln(A) - ln(B)
        # This prevents float32 NaN overflows if 1/p becomes very large.
        # Mathematically equivalent to: sqrt((sigma^2 * gamma(1/p)) / gamma(3/p))
        A_val = torch.abs(sigma) * torch.exp(
            0.5 * (torch.lgamma(1 / p) - torch.lgamma(3 / p))
        )

        # Calculate the 2D profile using the pre-computed radius grid
        numerator = torch.exp(-(self.radius_sq ** (p / 2)) / (torch.abs(A_val) ** p))

        # Calculate the denominator using lgamma -> exp
        gamma_term = torch.exp(torch.lgamma(1 + 1 / p))
        denominator = (2 * gamma_term * A_val) ** 2

        apsf2d = numerator / denominator
        apsf2d = apsf2d / torch.sum(apsf2d) # Normalize to maintain image brightness

        return apsf2d


class BrightnessAttentionMap(nn.Module):
    """A module that computes the Brightness Attention Map (BAM) prior.

    Extract a self-attention map from the V-channel of an input image to guide
    the enhancement network. Brighter regions are given lower weights to avoid
    over-saturation, while preserving image details and enhancing contrast in
    dark regions effectively.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, gamma: float = 2.5, eps: float = 1e-8):
        """Initialize a new instance.

        Args:
            gamma (float, optional): The exponent used to compute the BAM.
                Higher values increase the contrast between dark and bright
                regions. Defaults to 2.5.
            eps (float, optional): A small constant is added to the BAM
                computation to prevent division by zero. Defaults to 1e-8.
        """
        super().__init__()
        # Assign attributes
        self.gamma = gamma
        self.eps = eps

    # --- Callable & Context Manager ---
    def forward(self, image: Tensor) -> Tensor:
        """Extract a Brightness Attention Map (BAM) from an RGB image.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0. Can be either RGB (C=3) or grayscale (C=1).

        Returns:
            Tensor: A BAM tensor of shape (B, 1, H, W) with values ranging from
                0.0 to 1.0, where higher values indicate higher attention (darker regions).
        """
        # Extract Intensity (V-channel)
        if image.shape[1] == 3:
            # RGB case: Use HSV Value channel
            hsv = kornia.color.rgb_to_hsv(image)
            v = hsv[:, 2:3, :, :]
        else:
            # Grayscale case: The image is the brightness
            v = image.mean(dim=1, keepdim=True)

        # Compute BAM: (1 - V)^gamma
        # High value = high attention (dark regions)
        bam = torch.pow((1.0 - v + self.eps), self.gamma)

        return torch.clamp(bam, 0, 1)


class BoundaryAwarePrior(nn.Module):
    """A module to get the boundary prior from an RGB or grayscale image."""

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        eps: float = 0.05,
        as_gradient: bool = False,
        normalized: bool = False
    ):
        """Initialize a new instance.

        Args:
            eps (float, optional): Threshold to remove weak edges. Defaults to 0.05.
            as_gradient (bool, optional): If True, returns the gradient image
                instead of a binary boundary. Defaults to False.
            normalized (bool, optional): L1 norm of the kernel is set to 1 if
                True. Defaults to False.
        """
        super().__init__()
        # Assign attributes
        self.eps = eps
        self.as_gradient = as_gradient
        self.sobel = kornia.filters.Sobel(normalized=normalized)

    # --- Callable & Context Manager ---
    def forward(self, image: Tensor) -> Tensor:
        """Computes the boundary prior from the input image.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
        """
        # Compute Gradient Magnitude
        gradient = self.sobel(image.to(torch.float32))

        # Per-sample Normalization
        # We find the max for each image in the batch (B, 1, 1, 1)
        # add 1e-8 to avoid division by zero
        b_max = gradient.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1)
        gradient = gradient / (b_max + 1e-8)

        # Output selection
        if self.as_gradient:
            return gradient

        # Binary mask (Hard Attention)
        return (gradient > self.eps).float()

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
