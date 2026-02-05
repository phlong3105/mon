#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = [
    "filter_up",
    "get_coords",
    "get_image",
    "get_patches",
    "get_v_component",
    "interpolate_image",
    "replace_v_component",
]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from filter import FastGuidedFilter


def get_image(path: str) -> torch.Tensor:
    """Reads and returns RGB image, (1, 3, H, W)."""
    image = torch.from_numpy(np.array(Image.open(path))).float()
    image = image / torch.max(image)
    image = torch.movedim(image, -1, 0).unsqueeze(0)
    return image


def get_v_component(img_hsv: torch.Tensor) -> torch.Tensor:
    """Assumes (1, 3, H, W) HSV image."""
    return img_hsv[:,-1].unsqueeze(0)


def replace_v_component(img_hsv: torch.Tensor, v_new: torch.Tensor) -> torch.Tensor:
    """Replaces the V component of a HSV image (1, 3, H, W)."""
    img_hsv[:,-1] = v_new
    return img_hsv


def interpolate_image(img: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Reshapes the image based on new resolution."""
    return F.interpolate(img, size=(H,W))


def get_coords(H: int, W: int) -> torch.Tensor:
    """Creates a coordinates grid for INF."""
    coords = np.dstack(np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W)))
    coords = torch.from_numpy(coords).float()
    return coords


def get_patches(img: torch.Tensor, KERNEL_SIZE: int) -> torch.Tensor:
    """Creates a tensor where the channel contains patch information."""
    kernel = torch.zeros((KERNEL_SIZE ** 2, 1, KERNEL_SIZE, KERNEL_SIZE)).to(img.device)

    for i in range(KERNEL_SIZE):
        for j in range(KERNEL_SIZE):
            kernel[int(torch.sum(kernel).item()), 0, i, j] = 1

    pad       = nn.ReflectionPad2d(KERNEL_SIZE // 2)
    im_padded = pad(img)
    extracted = F.conv2d(im_padded, kernel, padding=0).squeeze(0)
    return torch.movedim(extracted, 0, -1)


def filter_up(x_lr: torch.Tensor, y_lr: torch.Tensor, x_hr: torch.Tensor, r: int = 1):
    """Applies the guided filter to upscale the predicted image."""
    guided_filter = FastGuidedFilter(r=r)
    y_hr = guided_filter(x_lr, y_lr, x_hr)
    y_hr = torch.clip(y_hr, 0, 1)
    return y_hr
