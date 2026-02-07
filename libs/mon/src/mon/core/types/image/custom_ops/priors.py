#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image priors.

This module provides functions to compute various image priors that can be used
in image processing and computer vision tasks.
"""

__all__ = [
    "BoundaryAwarePrior",
    "BrightnessAttentionMap",
    "ImageLocalMean",
    "ImageLocalStdDev",
    "ImageLocalVariance",
    "apsf",
]

import kornia
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# ATTENTION & ENHANCEMENT PRIORS
# ==============================================================================

# --- Lighting Guidance ---

class BrightnessAttentionMap(nn.Module):
    """A module that computes the Brightness Attention Map (BAM) prior.

    Extract a self-attention map from the V-channel of an input image to guide
    the enhancement network. Brighter regions are given lower weights to avoid
    over-saturation, while preserving image details and enhancing contrast in
    dark regions effectively.

    Attributes:
        gamma (float): Parameter controlling the curvature of the map.
        eps (float): Small value to avoid division by zero.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        gamma      : float = 2.5,
        kernel_size: int   = None,
        eps        : float = 1e-8,
    ):
        """Initialize a new instance.

        Args:
            gamma: Parameter controlling the curvature of the map. Defaults to 2.5.
            kernel_size: Window size for denoising operation. Defaults to None.
            eps: Small value to avoid division by zero. Defaults to 1e-8.
        """
        super().__init__()
        self.gamma   = gamma
        self.eps     = eps
        self.denoise = kornia.filters.MedianBlur(kernel_size=kernel_size) if kernel_size else None

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get the Brightness Attention Map (BAM) prior from an RGB image.

        Args:
            x: An RGB or grayscale image, formatted as a torch.Tensor of
                shape (B, C, H, W) and pixel values ranging from 0.0 to 1.0.

        Returns:
            The Brightness Attention Map, formatted as a torch.Tensor of shape
                (B, 1, H, W) and pixel values ranging from 0.0 to 1.0.
        """
        # Pre-process (Optional Denoising)
        # Median blur preserves edges while removing salt-and-pepper noise
        x = self.denoise(x) if self.denoise else x

        # Extract Intensity (V-channel)
        if x.shape[1] == 3:
            # RGB case: Use HSV Value channel
            hsv = kornia.color.rgb_to_hsv(x)
            v   = hsv[:, 2:3, :, :]
        else:
            # Grayscale case: The image is the brightness
            v   = x.mean(dim=1, keepdim=True)

        # 3. Compute BAM: (1 - V)^gamma
        # High value = high attention (dark regions)
        bam = torch.pow((1.0 - v + self.eps), self.gamma)

        return torch.clamp(bam, 0, 1)


# --- Atmospheric ---

def apsf(
    image: torch.Tensor,
    q    : float = 0.2,
    t    : float = 1.2,
    k    : float = 0.5,
) -> torch.Tensor:
    """Atmospheric point spread function (APSF) from an RGB image.

    Args:
        image: An RGB image, formatted as a torch.Tensor of shape (B, 3, H, W)
            and pixel values ranging from 0.0 to 1.0.
        q: Forward scattering params:

            - 0.00-0.20: air
            - 0.20-0.70: aerosol
            - 0.70-0.80: haze
            - 0.80-0.85: mist
            - 0.85-0.90: fog
            - 0.90-1.00: rain
            Defaults to 0.2.
        t: Optical thickness. Possibly: [0.7, 1.2, 4]. According to Narasimhan
            in CVPR03 paper:
            T = sigma * R (extinction coefficient * distance or depth),
            which is the same \beta d in haze modelling. Defaults to 1.2.
        k: Conversion param for kernel. Defaults to 0.5.

    Returns:
        The APSF applied image, formatted as a torch.Tensor of shape (B, 3, H, W)
            and pixel values ranging from 0.0 to 1.0.

    References:
        - Code: https://github.com/jinyeying/night-enhancement/blob/main/glow_rendering_code/repro_ICCV2007_Fig5.m
    """
    from scipy.special import gamma

    def A(p: float, sigma: float):
        return np.sqrt(sigma ** 2 * gamma(1 / p) / gamma(3 / p))

    p       = k * t        # Eq (9)
    sigma   = (1 - q) / q  # Eq (1)
    # Generate APSF kernel
    x       = torch.linspace(-6, 6, 100)
    XX, YY  = torch.meshgrid(x, x, indexing="ij")
    A_val   = A(p, sigma)
    APSF2D  = torch.exp(-((XX ** 2 + YY ** 2) ** (p / 2)) / abs(A_val) ** p) / (2 * gamma(1 + 1 / p) * A_val) ** 2
    APSF2D /= torch.sum(APSF2D)
    # Apply convolution
    kernel  = APSF2D.unsqueeze(0).unsqueeze(0)   # Shape: (1, 1, H, W)
    kernel  = kernel.repeat(3, 1, 1, 1)          # Shape: (3, 1, H, W) for RGB channels
    apsf    = F.conv2d(image, kernel, padding="same", groups=3)
    apsf    = torch.clamp(apsf, 0, 1)  # Ensure valid pixel range
    return apsf


# ==============================================================================
# STRUCTURAL & EDGE PRIORS
# ==============================================================================

# --- Boundary Detection ---

class BoundaryAwarePrior(nn.Module):
    """A module to get the boundary prior from an RGB or grayscale image.

    Attributes:
        eps (float): Threshold to remove weak edges.
        as_gradient (bool): If True, returns the gradient image instead of
            binary boundary.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        eps        : float = 0.05,
        as_gradient: bool  = False,
        normalized : bool  = False
    ):
        """Initialize a new instance.

        Args:
            eps: Threshold to remove weak edges. Defaults to 0.05.
            as_gradient: If True, returns the gradient image instead of binary
                boundary. Defaults to False.
            normalized: L1 norm of the kernel is set to 1 if True. Defaults to False.
        """
        super().__init__()
        self.sobel       = kornia.filters.Sobel(normalized=normalized)
        self.eps         = eps
        self.as_gradient = as_gradient

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the boundary prior from the input image.

        Args:
            x: An RGB or grayscale image, formatted as a torch.Tensor of
                shape (B, C, H, W) and pixel values ranging from 0.0 to 1.0.

        Returns:
            A boundary-aware prior map, formatted as a torch.Tensor of shape
                (B, C, H, W). Depending on the ``as_gradient`` attribute,
                it returns either a binary boundary map or the gradient image.
        """
        # Compute Gradient Magnitude
        gradient = self.sobel(x.to(torch.float32))

        # Per-sample Normalization
        # We find the max for each image in the batch (B, 1, 1, 1)
        # add 1e-8 to avoid division by zero
        b_max    = gradient.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1)
        gradient = gradient / (b_max + 1e-8)

        # Output selection
        if self.as_gradient:
            return gradient

        # Binary mask (Hard Attention)
        return (gradient > self.eps).float()


# ==============================================================================
# STRUCTURAL & EDGE PRIORS
# ==============================================================================

class ImageLocalMean(nn.Module):
    """A module to calculate the local mean of an image using a sliding window.

    Attributes:
        patch_size (int): Size of the sliding window.
        padding (int): Padding size for the sliding window.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, patch_size: int = 5):
        """Initialize a new instance.

        Args:
            patch_size: Size of the sliding window. Defaults to 5.
        """
        super().__init__()
        self.patch_size = patch_size
        self.padding    = patch_size // 2

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the local mean of the input image.

        Args:
            x: An RGB or grayscale image, formatted as a torch.Tensor of
                shape (B, C, H, W) and pixel values ranging from 0.0 to 1.0.

        Returns:
            Local means with the same type and format as ``image``.
        """
        # Apply reflection padding to maintain edge intensity
        # pad is (left, right, top, bottom)
        x = F.pad(
            x,
            (self.padding, self.padding, self.padding, self.padding),
            mode="reflect"
        )

        # Use Average Pooling to calculate the mean
        # stride=1 ensures we get a value for every original pixel
        return F.avg_pool2d(x, kernel_size=self.patch_size, stride=1)


class ImageLocalVariance(nn.Module):
    """A module to calculate the local variance of an image using a sliding window.

    Attributes:
        patch_size (int): Size of the sliding window.
        padding (int): Padding size for the sliding window.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, patch_size: int = 5):
        """Initialize a new instance.

        Args:
            patch_size: Size of the sliding window. Defaults to 5.
        """
        super().__init__()
        self.patch_size = patch_size
        self.padding    = patch_size // 2

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the local variance of the input image.

        Args:
            x: An RGB or grayscale image, formatted as a torch.Tensor of
                shape (B, C, H, W) and pixel values ranging from 0.0 to 1.0.

        Returns:
            Local variance with the same type and format as the ``image``.
        """
        # Pad image to maintain dimensions
        x_padded = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode="reflect")

        # Compute E[X] (Local Mean)
        mean_x = F.avg_pool2d(x_padded, kernel_size=self.patch_size, stride=1)

        # Compute E[X^2] (Local Mean of Squares)
        mean_x2 = F.avg_pool2d(x_padded**2, kernel_size=self.patch_size, stride=1)

        # Variance = E[X^2] - (E[X])^2
        # Use max(0) or a small epsilon to avoid negative values due to float precision
        variance = mean_x2 - mean_x**2

        # Ensures no negative values
        return torch.relu(variance)


class ImageLocalStdDev(nn.Module):
    """A module to calculate the local standard deviation of an image using a
    sliding window.

    Attributes:
        patch_size (int): Size of the sliding window.
        eps (float): Small value to avoid division by zero.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, patch_size: int = 5, eps: float = 1e-9):
        """Initialize a new instance.

        Args:
            patch_size: Size of the sliding window. Defaults to 5.
            eps: Small value to avoid division by zero. Defaults to 1e-9.
        """
        super().__init__()
        self.patch_size = patch_size
        self.padding    = patch_size // 2
        self.eps        = eps

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the local standard deviation of the input image.

        Args:
            x: An RGB or grayscale image, formatted as a torch.Tensor of shape
                (B, C, H, W) and pixel values ranging from 0.0 to 1.0.

        Returns:
            Local standard deviation with the same type and format as the ``image``.
        """
        #Reflection padding for edge consistency
        x_padded = F.pad(x, [self.padding]*4, mode="reflect")

        # Compute E[X] and E[X^2]
        mean_x  = F.avg_pool2d(x_padded,    kernel_size=self.patch_size, stride=1)
        mean_x2 = F.avg_pool2d(x_padded**2, kernel_size=self.patch_size, stride=1)

        # Variance = E[X^2] - (E[X])^2
        # Clamp at 0 to prevent tiny negative numbers from float imprecision
        variance = (mean_x2 - mean_x**2).clamp(min=0)

        # StdDev = sqrt(Var + eps)
        return torch.sqrt(variance + self.eps)


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
