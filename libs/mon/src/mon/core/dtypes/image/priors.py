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
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(self, gamma: float = 2.5, kernel_size: int = None):
        """Initialize a new instance.
        
        Args:
            gamma: Parameter controlling the curvature of the map. Defaults to 2.5.
            kernel_size: Window size for denoising operation. Defaults to None.
        """
        super().__init__()
        self.gamma   = gamma
        self.denoise = kornia.filters.MedianBlur(kernel_size=kernel_size) if kernel_size else None
    
    # --- Callable & Context Manager ---
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Get the Brightness Attention Map (BAM) prior from an RGB image.
        
        Args:
            image: An RGB or grayscale image, formatted as a torch.Tensor with
                dimensions (B, C, H, W) and pixel values ranging from 0.0 to 1.0.
                
        Returns:
            The Brightness Attention Map, formatted as a torch.Tensor with
            dimensions (B, 1, H, W) and pixel values ranging from 0.0 to 1.0.
        """
        if self.denoise:
            image = self.denoise(image)

        hsv = kornia.color.rgb_to_hsv(image)
        v   = hsv[:, 2:3, :, :]  # Extract the V-channel (brightness)
        bam = torch.pow((1 - v), self.gamma)
        return bam


# --- Atmospheric ---
def apsf(
    image: torch.Tensor,
    q    : float = 0.2,
    t    : float = 1.2,
    k    : float = 0.5,
) -> torch.Tensor:
    """Atmospheric point spread function (APSF) from an RGB image.
    
    Args:
        image: An RGB image, formatted as a torch.Tensor with dimensions
            (B, 3, H, W) and pixel values ranging from 0.0 to 1.0.
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
        The APSF applied image, formatted as a torch.Tensor with dimensions
        (B, 3, H, W) and pixel values ranging from 0.0 to 1.0.
    
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
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Computes the boundary prior from the input image.
        
        Args:
            image: An RGB or grayscale image, formatted as a torch.Tensor with
                dimensions (B, C, H, W) and pixel values ranging from 0.0 to 1.0.
            
        Returns:
            A boundary-aware prior map, formatted as a torch.Tensor with
            dimensions  (B, C, H, W). Depending on the ``as_gradient`` attribute,
            it returns either a binary boundary map or the gradient image.
        """
        image    = image.to(torch.float32)
        gradient = self.sobel(image)
        g_max    = torch.max(gradient)
        gradient = gradient / g_max
        boundary = (gradient > self.eps).float()
        if self.as_gradient:
            return gradient
        else:
            return boundary


# ==============================================================================
# STRUCTURAL & EDGE PRIORS
# ==============================================================================

class ImageLocalMean(nn.Module):
    """A module to calculate the local mean of an image using a sliding window.
    
    Attributes:
        patch_size (int): Size of the sliding window.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(self, patch_size: int = 5):
        """Initialize a new instance.
        
        Args:
            patch_size: Size of the sliding window. Defaults to 5.
        """
        super().__init__()
        self.patch_size = patch_size
    
    # --- Callable & Context Manager ---
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Calculate the local mean of the input image.

        Args:
            image: An RGB or grayscale image, formatted as a torch.Tensor with
                dimensions (B, C, H, W) and pixel values ranging from 0.0 to 1.0.

        Returns:
            Local mean with similar type and format as ``image``.
        """
        padding = self.patch_size // 2
        image   = F.pad(image, (padding, padding, padding, padding), mode="reflect")
        patches = image.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)
        return patches.mean(dim=(4, 5))


class ImageLocalVariance(nn.Module):
    """A module to calculate the local variance of an image using a sliding window.
    
    Attributes:
        patch_size (int): Size of the sliding window.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(self, patch_size: int = 5):
        """Initialize a new instance.
        
        Args:
            patch_size: Size of the sliding window. Defaults to 5.
        """
        super().__init__()
        self.patch_size = patch_size
    
    # --- Callable & Context Manager ---
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Calculate the local variance of the input image.
        
        Args:
            image: An RGB or grayscale image, formatted as a torch.Tensor with
                dimensions (B, C, H, W) and pixel values ranging from 0.0 to 1.0.
                
        Returns:
            Local variance with similar type and format as the ``image``.
        """
        padding = self.patch_size // 2
        image   = F.pad(image, (padding, padding, padding, padding), mode="reflect")
        patches = image.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)
        mean    = patches.mean(dim=(4, 5))
        return ((patches - mean.unsqueeze(4).unsqueeze(5)) ** 2).mean(dim=(4, 5))


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
        self.eps        = eps
    
    # --- Callable & Context Manager ---
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Calculate the local standard deviation of the input image.
        
        Args:
            image: An RGB or grayscale image, formatted as a torch.Tensor with
                dimensions (B, C, H, W) and pixel values ranging from 0.0 to 1.0.
                
        Returns:
            Local standard deviation with similar type and format as the ``image``.
        """
        padding        = self.patch_size // 2
        image          = F.pad(image, (padding, padding, padding, padding), mode="reflect")
        patches        = image.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)
        mean           = patches.mean(dim=(4, 5), keepdim=True)
        squared_diff   = (patches - mean) ** 2
        local_variance = squared_diff.mean(dim=(4, 5))
        local_stddev   = torch.sqrt(local_variance + self.eps)
        return local_stddev
