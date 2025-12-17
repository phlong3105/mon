#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for HVI color space conversion.

This module implements a class to convert RGB images to HVI (Hue, Value, Intensity)
and vice versa.
"""

__all__ = [
    "RGBToHVI",
]

import torch
import torch.nn as nn


class RGBToHVI(nn.Module):
    """A class to convert RGB images to HVI color space and back.
    
    Attributes:
        eps (float): Epsilon value to avoid division by zero.
        density_k (nn.Parameter): Learnable parameter for color sensitivity.
        gated (bool): If True, applies gating to saturation during HVI to RGB conversion.
        gated2 (bool): If True, applies gating to RGB values during HVI to RGB conversion.
        alpha (float): Scaling factor for RGB values if ``gated2`` is True.
        alpha_s (float): Scaling factor for saturation if ``gated`` is True.
        this_k (float): Current value of ``density_k`` used in conversions.
    
    References:
        - Code: https://github.com/Fediory/HVI-CIDNet/blob/master/net/HVI_transform.py
    """
    
    def __init__(self, eps: float = 1e-8, requires_grad: bool = False):
        """Initializes the RGBToHVI instance.
        
        Args:
            eps (float): Epsilon value to avoid division by zero. Defaults to 1e-8.
            requires_grad (bool): If True, allows gradient computation for ``density_k``.
                Defaults to False.
        """
        super().__init__()
        self.eps       = eps
        self.density_k = nn.Parameter(torch.full([1], 0.1), requires_grad=requires_grad)  # k is reciprocal to the paper mentioned
        # self.density_k = nn.Parameter(torch.full([1], 0.2), requires_grad=requires_grad)  # k is reciprocal to the paper mentioned
        self.gated     = False
        self.gated2    = False
        self.alpha     = 1.0
        self.alpha_s   = 1.3
        self.this_k    = 0
    
    def rgb_to_hvi(self, image: torch.Tensor) -> torch.Tensor:
        """Converts an RGB image to HVI color space.
        
        Args:
            image (torch.Tensor): Input RGB image tensor of shape (B, 3, H, W)
                with pixel values in the range [0, 1].
                
        Returns:
            torch.Tensor: HVI image tensor of shape (B, 3, H, W).
        """
        pi      = 3.141592653589793
        device  = image.device
        dtypes  = image.dtype
        hue     = torch.Tensor(image.shape[0], image.shape[2], image.shape[3]).to(device).to(dtypes)
        value   = image.max(1)[0].to(dtypes)
        img_min = image.min(1)[0].to(dtypes)
        hue[image[:, 2] == value] =  4.0 + ((image[:, 0] - image[:, 1]) / (value - img_min + self.eps))[image[:, 2] == value]
        hue[image[:, 1] == value] =  2.0 + ((image[:, 2] - image[:, 0]) / (value - img_min + self.eps))[image[:, 1] == value]
        hue[image[:, 0] == value] = (0.0 + ((image[:, 1] - image[:, 2]) / (value - img_min + self.eps))[image[:, 0] == value]) % 6
        
        hue[image.min(1)[0] == value] = 0.0
        hue = hue / 6.0
        
        saturation = (value - img_min) / (value + self.eps)
        saturation[value == 0] = 0
        
        hue         = hue.unsqueeze(1)
        saturation  = saturation.unsqueeze(1)
        value       = value.unsqueeze(1)
        
        k = self.density_k#.clone()
        self.this_k = k.item()

        color_sensitive = ((value * 0.5 * pi).sin() + self.eps).pow(k)
        ch = (2.0 * pi * hue).cos()
        cv = (2.0 * pi * hue).sin()
        H  = color_sensitive * saturation * ch
        V  = color_sensitive * saturation * cv
        I  = value
        hvi = torch.cat([H, V, I], dim=1)
        return hvi
    
    def hvi_to_rgb(self, image: torch.Tensor) -> torch.Tensor:
        """Converts an HVI image to RGB color space.
        
        Args:
            image (torch.Tensor): Input HVI image tensor of shape (B, 3, H, W)
                with H and V in range [-1, 1] and I in range [0, 1].
                
        Returns:
            torch.Tensor: RGB image tensor of shape (B, 3, H, W) with pixel
                values in the range [0, 1].
        """
        pi      = 3.141592653589793
        H, V, I = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
        
        # Clip
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        I = torch.clamp(I, 0, 1)
        
        v = I
        k = self.this_k
        color_sensitive = ((v * 0.5 * pi).sin() + self.eps).pow(k)
        H = (H) / (color_sensitive + self.eps)
        V = (V) / (color_sensitive + self.eps)
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        h = torch.atan2(V + self.eps, H + self.eps) / (2 * pi)
        h = h % 1
        s = torch.sqrt(H ** 2 + V ** 2 + self.eps)
        
        if self.gated:
            s = s * self.alpha_s
        
        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)
        
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
        
        hi = torch.floor(h * 6.0)
        f  = h * 6.0 - hi
        p  = v * (1.0 - s)
        q  = v * (1.0 - (f * s))
        t  = v * (1.0 - ((1.0 - f) * s))
        
        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5
        
        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]
        
        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]
        
        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]
        
        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]
        
        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]
        
        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]
        
        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        if self.gated2:
            rgb = rgb * self.alpha
        return rgb
