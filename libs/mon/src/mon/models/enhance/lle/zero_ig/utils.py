#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities.

This module provides various utilities for Zero-IG.
"""

from __future__ import annotations

__all__ = [
    "LocalMean",
    "blur",
    "calculate_local_variance",
]

import torch
from torch import nn, Tensor
from torch.nn import functional as F


# ==============================================================================
# region UTILITIES
# ==============================================================================

def gauss_cdf(x: Tensor) -> Tensor:
    return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))


def gauss_kernel(kernel: int = 21, nsig: int = 3, channels: int = 1):
    interval = (2 * nsig + 1.0) / kernel
    x = torch.linspace(-nsig - interval / 2.0, nsig + interval / 2.0, kernel + 1).cuda()
    kern1d = torch.diff(gauss_cdf(x))
    kernel_raw = torch.sqrt(torch.outer(kern1d,kern1d))
    kernel = kernel_raw / torch.sum(kernel_raw)
    out_filter = kernel.view(1, 1, kernel, kernel)
    out_filter = out_filter.repeat(channels, 1, 1, 1)
    return out_filter


def blur(x: Tensor) -> Tensor:
    kernel_size = 21
    padding = kernel_size // 2
    kernel_var = gauss_kernel(kernel_size, 1, x.size(1)).to(x.device)
    x_padded = F.pad(x, (padding, padding, padding, padding), mode="reflect")
    return F.conv2d(x_padded, kernel_var, padding=0, groups=x.size(1))


def padr_tensor(image: Tensor) -> Tensor:
    pad_mod = nn.ConstantPad2d(2, 0)
    img_pad = pad_mod(image)
    return img_pad


def calculate_local_variance(train_noisy: Tensor) -> Tensor:
    b, c, w, h = train_noisy.shape
    avg_pool = torch.nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
    noisy_avg = avg_pool(train_noisy)
    noisy_avg_pad = padr_tensor(noisy_avg)
    train_noisy = padr_tensor(train_noisy)
    unfolded_noisy_avg = noisy_avg_pad.unfold(2,5,1).unfold(3,5,1)
    unfolded_noisy = train_noisy.unfold(2,5,1).unfold(3,5,1)
    unfolded_noisy_avg = unfolded_noisy_avg.reshape(unfolded_noisy_avg.shape[0], -1, 5, 5)
    unfolded_noisy = unfolded_noisy.reshape(unfolded_noisy.shape[0], -1, 5, 5)
    noisy_diff_squared = (unfolded_noisy - unfolded_noisy_avg) ** 2
    noisy_var = torch.mean(noisy_diff_squared, dim=(2, 3))
    noisy_var = noisy_var.view(b, c, w, h)
    return noisy_var


class LocalMean(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, patch_size: int = 5):
        """Initialize a new instance."""
        super().__init__()
        self.patch_size = patch_size
        self.padding = self.patch_size // 2

    # --- Callable & Context Manager ---
    def forward(self, image: Tensor) -> Tensor:
        p = self.padding
        image = F.pad(image, (p, p, p, p), mode="reflect")
        patches = image.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)
        return patches.mean(dim=(4, 5))

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
