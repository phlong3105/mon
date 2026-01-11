#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Color processing operations.

This module provides operations for color processing.
"""

from __future__ import annotations

__all__ = [
    "RGBToHVI",
    "color_transfer",
]

import cv2
import numpy as np
import torch
import torch.nn as nn


# ==============================================================================
# region LEARNABLE COLOR SPACES
# ==============================================================================

# --- HVI Space (Perceptual Saturation & Intensity) ---
class RGBToHVI(nn.Module):
    """A module for converting RGB images to HVI color space and back.
    
    References:
        - Code: https://github.com/Fediory/HVI-CIDNet/blob/master/net/HVI_transform.py
        
    Attributes:
        eps (float): Epsilon value to avoid division by zero.
        density_k (nn.Parameter): Learnable parameter controlling the sensitivity
            of the color gamut relative to intensity.
        pi (float): Mathematical constant π.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(self, eps: float = 1e-8, requires_grad: bool = False):
        """Initialize a new instance.
        
        Args:
            eps: Epsilon value to avoid division by zero. Defaults to 1e-8.
            requires_grad: If True, allows gradient computation for ``density_k``.
                Defaults to False.
        """
        super().__init__()
        self.eps       = eps
        # Learnable 'k' controls the color gamut sensitivity relative to intensity
        self.density_k = nn.Parameter(torch.full([1], 0.1), requires_grad=requires_grad)
        self.pi        = 3.141592653589793
    
    # --- Callable & Context Manager ---
    def rgb_to_hvi(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert an RGB image to HVI color space.
        
        Args:
            rgb: An RGB image, formatted as a torch.Tensor of shape (B, 3, H, W)
                and pixel values ranging from 0.0 to 1.0.
                
        Returns:
            The HVI image, formatted as a torch.Tensor of shape (B, 3, H, W)
                and pixel values ranging from 0.0 to 1.0.
        """
        r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
        
        max_val, _ = rgb.max(1)
        min_val, _ = rgb.min(1)
        diff = max_val - min_val + self.eps
        
        # Standard Hue calculation (0-6 range)
        hue = torch.zeros_like(max_val)
        hue = torch.where(max_val == r, (g - b) / diff % 6, hue)
        hue = torch.where(max_val == g, (b - r) / diff + 2, hue)
        hue = torch.where(max_val == b, (r - g) / diff + 4, hue)
        hue = torch.where(diff < self.eps, torch.zeros_like(hue), hue)
        hue = hue / 6.0  # Normalized to [0, 1]
        
        # Saturation and Intensity
        # S = (max - min) / max
        s = diff / (max_val + self.eps)
        i = max_val # Intensity (Value)
        
        # Cartesian Mapping (The HVI specific logic)
        # Sensitivity curves the color response based on Intensity
        sensitivity = ((i * 0.5 * self.pi).sin() + self.eps).pow(self.density_k)
        
        # Mapping Hue/Saturation (Polar) -> H/V (Cartesian)
        h_coord = sensitivity * s * torch.cos(2.0 * self.pi * hue)
        v_coord = sensitivity * s * torch.sin(2.0 * self.pi * hue)
        
        return torch.stack([h_coord, v_coord, i], dim=1)
        
    def hvi_to_rgb(self, hvi: torch.Tensor) -> torch.Tensor:
        """Convert an HVI image to RGB color space.
        
        Args:
            hvi: An HVI image, formatted as a torch.Tensor of shape (B, 3, H, W)
                and H and V pixel values ranging from -1.0 to 1.0
                and I pixel values ranging from 0.0 to 1.0.

        Returns:
            An RGB image, formatted as a torch.Tensor of shape (B, 3, H, W) and
                pixel values ranging from 0.0 to 1.0.
        """
        h_coord, v_coord, i = hvi[:, 0, :, :], hvi[:, 1, :, :], hvi[:, 2, :, :]
        
        sensitivity = ((i * 0.5 * self.pi).sin() + self.eps).pow(self.density_k)
        
        # Reconstruct Hue (angle) and Saturation (magnitude)
        h = torch.atan2(v_coord, h_coord) / (2.0 * self.pi) % 1.0
        s = torch.sqrt(h_coord ** 2 + v_coord ** 2 + self.eps) / (sensitivity + self.eps)
        s = torch.clamp(s, 0, 1)

        # HSV to RGB Conversion (Simplified Vectorized)
        hi = (h * 6.0).floor()
        f  = h * 6.0 - hi
        p  = i * (1.0 - s)
        q  = i * (1.0 - (f * s))
        t  = i * (1.0 - ((1.0 - f) * s))

        # Reconstruct RGB channels based on Hue sector
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        # Hi indices: 0: R,Gt,Bp | 1: Rq,G,Bp | 2: Rp,G,Bt | 3: Rp,Gq,B | 4: Rt,Gp,B | 5: R,Gp,Bq
        mask = (hi == 0); r[mask], g[mask], b[mask] = i[mask], t[mask], p[mask]
        mask = (hi == 1); r[mask], g[mask], b[mask] = q[mask], i[mask], p[mask]
        mask = (hi == 2); r[mask], g[mask], b[mask] = p[mask], i[mask], t[mask]
        mask = (hi == 3); r[mask], g[mask], b[mask] = p[mask], q[mask], i[mask]
        mask = (hi == 4); r[mask], g[mask], b[mask] = t[mask], p[mask], i[mask]
        mask = (hi == 5); r[mask], g[mask], b[mask] = i[mask], p[mask], q[mask]

        return torch.stack([r, g, b], dim=1)

# endregion


# ==============================================================================
# region COLOR & STYLE (Domain Alignment)
# ==============================================================================

# --- Distribution Matching ---
def color_transfer(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Transfer the color distribution from the target image to the source
    image using the mean and standard deviation of the LAB color space.
    
    References:
        - Paper: "Color Transfer between Images".
        - Code: https://github.com/rinsa318/color-transfer
        - Code: https://github.com/chia56028/Color-Transfer-between-Images
        - Code: https://www.cnblogs.com/likethanlove/p/6003677.html
        - Code: https://pyimagesearch.com/2014/06/30/super-fast-color-transfer-images/
    
    Args:
        source: An RGB source image, formatted as a numpy.ndarray of shape
            (H, W, C) and pixel values ranging from 0 to 255.
        target: An RGB target image, formatted as a numpy.ndarray wof shape
            (H, W, C) and pixel values ranging from 0 to 255.

    Returns:
        The color transferred image.
    """
    # Convert to LAB
    # Note: cv2 expects BGR usually, but if your input is RGB, use RGB2LAB
    s_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    t_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    # Extract statistics
    s_mean, s_std = s_lab.mean(axis=(0, 1)), s_lab.std(axis=(0, 1))
    t_mean, t_std = t_lab.mean(axis=(0, 1)), t_lab.std(axis=(0, 1))
    
    # Perform transfer: (Source - Mean) * (Target_Std / Source_Std) + Target_Mean
    # We use an epsilon to avoid div by zero
    res_lab = (s_lab - s_mean) * (t_std / (s_std + 1e-8)) + t_mean
    
    # Cleanup: Clip to valid LAB ranges
    # L: [0, 100], a: [-127, 127], b: [-127, 127]
    # (Though OpenCV uint8 LAB uses 0-255 for all)
    res_lab = np.clip(res_lab, 0, 255).astype(np.uint8)
    
    return cv2.cvtColor(res_lab, cv2.COLOR_LAB2RGB)

# endregion
