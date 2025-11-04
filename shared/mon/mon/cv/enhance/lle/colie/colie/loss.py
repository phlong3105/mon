#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "L_exp",
    "L_texture",
    "L_tv",
]

import torch
from torchvision.transforms.functional import gaussian_blur

import mon.nn as nn


class L_exp(nn.Module):

    def __init__(self, patch_size: int, mean_val: float):
        super().__init__()
        self.pool     = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.pool(x) ** 0.5
        d    = torch.abs(torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).to(x.device), 2)))
        return d


class L_tv(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        count_h    = (x.size()[2] - 1) * x.size()[3]
        count_w    = x.size()[2] * (x.size()[3] - 1)
        h_tv       = torch.pow((x[:, :, 1:, :] - x[:, :, :h - 1, :]), 2).sum()
        w_tv       = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / b


class L_texture(nn.Module):
    """Structure-Texture Decomposition Loss for Denoising. Penalizes high-frequency
    texture/noise in the reflectance map.
    """
    
    def __init__(self, kernel_size: int = 3, sigma: float = 1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma       = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create a blurred version of the image to represent the "structure"
        structure = gaussian_blur(x, kernel_size=self.kernel_size, sigma=self.sigma)
        # The "texture" is the difference between the original and the structure
        texture   = x - structure
        # Penalize the L1 norm of the texture component
        return torch.mean(torch.abs(texture))
