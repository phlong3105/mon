#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = [
    "filter_up",
    "get_coords",
    "get_h_component",
    "get_image",
    "get_patches",
    "get_s_component",
    "get_v_component",
    "hsv2rgb_torch",
    "interpolate_image",
    "replace_v_component",
    "rgb2hsv_torch",
]

import numpy as np
import torch
from PIL import Image
from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn import functional as F


# ==============================================================================
# region UTILITIES
# ==============================================================================

def get_image(path: str) -> Tensor:
    """Reads and returns RGB image, (1, 3, H, W)."""
    image = torch.from_numpy(np.array(Image.open(path))).float()
    image = image / torch.max(image)
    image = torch.movedim(image, -1, 0).unsqueeze(0)
    return image


# --- Color ---

def rgb2hsv_torch(rgb: Tensor) -> Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.0
    hsv_h /= 6.0
    hsv_s  = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v  = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)


def hsv2rgb_torch(hsv: Tensor) -> Tensor:
    hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    _c = hsv_l * hsv_s
    _x = _c * (- torch.abs(hsv_h * 6.0 % 2.0 - 1) + 1.0)
    _m = hsv_l - _c
    _o = torch.zeros_like(_c)
    idx = (hsv_h * 6.0).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsv)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb


def get_h_component(img_hsv: Tensor) -> Tensor:
    """Assumes (1, 3, H, W) HSV image."""
    return img_hsv[:, -3].unsqueeze(0)


def get_s_component(img_hsv: Tensor) -> Tensor:
    """Assumes (1, 3, H, W) HSV image."""
    return img_hsv[:, -2].unsqueeze(0)


def get_v_component(img_hsv: Tensor) -> Tensor:
    """Assumes (1, 3, H, W) HSV image."""
    return img_hsv[:, -1].unsqueeze(0)


def replace_v_component(img_hsv: Tensor, v_new: Tensor) -> Tensor:
    """Replaces the V component of a HSV image (1, 3, H, W)."""
    img_hsv[:,-1] = v_new
    return img_hsv


# --- Features ---

def get_coords(H: int, W: int) -> Tensor:
    """Creates a coordinates grid for INF."""
    coords = np.dstack(np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W)))
    coords = torch.from_numpy(coords).float()
    return coords


def get_patches(img: Tensor, KERNEL_SIZE: int) -> Tensor:
    """Creates a tensor where the channel contains patch information."""
    kernel = torch.zeros((KERNEL_SIZE ** 2, 1, KERNEL_SIZE, KERNEL_SIZE)).to(img.device)

    for i in range(KERNEL_SIZE):
        for j in range(KERNEL_SIZE):
            kernel[int(torch.sum(kernel).item()), 0, i, j] = 1

    pad = nn.ReflectionPad2d(KERNEL_SIZE // 2)
    im_padded = pad(img)
    extracted = F.conv2d(im_padded, kernel, padding=0).squeeze(0)
    return torch.movedim(extracted, 0, -1)


# --- Filter ---

def diff_x(input: Tensor, r: int) -> Tensor:
    assert input.dim() == 4
    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]
    output = torch.cat([left, middle, right], dim=2)
    return output


def diff_y(input: Tensor, r: int) -> Tensor:
    assert input.dim() == 4
    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]
    output = torch.cat([left, middle, right], dim=3)
    return output


class BoxFilter(nn.Module):

    def __init__(self, r: int):
        super().__init__()
        self.r = r

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 4
        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


class FastGuidedFilter(nn.Module):

    def __init__(self, r: int, eps: float = 1e-8):
        super().__init__()
        self.r          = r
        self.eps        = eps
        self.box_filter = BoxFilter(r)

    def forward(self, lr_x, lr_y, hr_x):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2 * self.r + 1 and w_lrx > 2 * self.r + 1

        ## N
        N = self.box_filter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        ## mean_x
        mean_x = self.box_filter(lr_x) / N
        ## mean_y
        mean_y = self.box_filter(lr_y) / N
        ## cov_xy
        cov_xy = self.box_filter(lr_x * lr_y) / N - mean_x * mean_y
        ## var_x
        var_x  = self.box_filter(lr_x * lr_x) / N - mean_x * mean_x

        ## A
        A = cov_xy / (var_x + self.eps)
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode="bilinear", align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode="bilinear", align_corners=True)

        return mean_A * hr_x + mean_b


# --- Resize ---

def interpolate_image(img: Tensor, H: int, W: int) -> Tensor:
    """Reshapes the image based on new resolution."""
    return F.interpolate(img, size=(H,W))


def filter_up(x_lr: Tensor, y_lr: Tensor, x_hr: Tensor, r: int = 1):
    """Applies the guided filter to upscale the predicted image."""
    guided_filter = FastGuidedFilter(r=r)
    y_hr = guided_filter(x_lr, y_lr, x_hr)
    y_hr = torch.clip(y_hr, 0, 1)
    return y_hr

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
