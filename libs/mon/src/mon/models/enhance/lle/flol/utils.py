#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities.

This module provides various utilities for FLOL.
"""

from __future__ import annotations

__all__ = [
    "default_init_weights",
    "flow_warp",
    "init_weights",
    "pixel_unshuffle",
    "resize_flow",
]

from typing import Literal

import torch
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
from torch import nn, Tensor


# ==============================================================================
# region UTILITIES
# ==============================================================================

# --- Initialization ---

@torch.no_grad()
def default_init_weights(
    module_list: list[nn.Module] | nn.Module,
    scale: float = 1.0,
    bias_fill: float = 0.0,
    **kwargs
):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Defaults to 1.0
        bias_fill (float): The value to fill bias. Defaults to 0.0
        kwargs (dict): Other arguments for the initialization function.
    """
    # Normalize inputs
    if not isinstance(module_list, list):
        module_list = [module_list]

    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def init_weights(net_l: list[nn.Module] | nn.Module, scale: float = 1.0):
    # Normalize inputs
    if not isinstance(net_l, list):
        net_l = [net_l]

    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


# --- Optical Flow ---

def flow_warp(
    x: Tensor,
    flow: Tensor,
    interp_mode: Literal["nearest", "bilinear"] = "bilinear",
    padding_mode: Literal["zeros", "border", "reflection"] = "zeros",
    align_corners: bool = True,
) -> Tensor:
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor of shape (B, C, H, W) and values ranging
            from 0.0 to 1.0.
        flow (Tensor): Tensor with size (B, H, W, 2) and values ranging
            from 0.0 to 1.0.
        interp_mode (str): Interpolation mode to calculate output values.
            Defaults to "bilinear".
        padding_mode (str): Padding mode for outside grid values.
            Defaults to "zeros".
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()

    # Create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(
        input=x,
        grid=vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    # TODO, what if align_corners=False
    return output


def resize_flow(
    flow: Tensor,
    size_type: Literal["ratio", "shape"],
    sizes: list[int | float],
    interp_mode: Literal["nearest", "bilinear"] = "bilinear",
    align_corners: bool = False,
) -> Tensor:
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): The type of resizing. Defaults to "ratio".
        sizes (list[int | float]): The ratio for resizing or the final output
            shape:

            1) The order of ratio should be [ratio_h, ratio_w]. For downsampling,
               the ratio should be smaller than 1.0 (i.e., ratio < 1.0).
               For upsampling, the ratio should be larger than 1.0 (i.e., ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): Interpolation mode to calculate output values.
            Defaults to "bilinear".
        align_corners (bool): Whether align corners. Defaults to False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == "ratio":
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == "shape":
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(f"Expected 'size_type' in ['ratio', 'shape'], but got: {size_type}.")

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow,
        size=(output_h, output_w),
        mode=interp_mode,
        align_corners=align_corners,
    )
    return resized_flow


# --- Shuffle ---

def pixel_unshuffle(x: Tensor, scale: int) -> Tensor:
    """Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (B, C, HH, HW).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale ** 2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
