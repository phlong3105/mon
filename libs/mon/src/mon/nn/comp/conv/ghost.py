#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""GhostNet's modules.

This module provides Ghost modules and Ghost bottlenecks as described in the
GhostNet and GhostNetV2 papers.

References:
    - Paper-V1: "GhostNet: More Features from Cheap Operations," CVPR 2020.
    - Code-V2: https://github.com/phlong3105/Efficient-AI-Backbones/tree/master/ghostnet_pytorch

    - Paper-V2: "GhostNetV2: Enhance Cheap Operation with Long-Range Attention," NeurIPS 2022.
    - Code-V2: https://github.com/phlong3105/Efficient-AI-Backbones/tree/master/ghostnetv2_pytorch
"""

from __future__ import annotations

__all__ = [
    "GhostBottleneck",
    "GhostBottleneckV2",
    "GhostModule",
    "GhostModuleV2",
]

import math
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# region UTILITIES
# ==============================================================================

def _make_divisible(v: int, divisor: int, min_value: int = None) -> int:
    """Ensure that all layers have a channel number that is divisible by 8.

    References:
        - Code: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """Hard sigmoid function."""
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(x + 3.0) / 6.0

# endregion


# ==============================================================================
# region MODULES
# ==============================================================================

class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block.

    Apply squeeze-and-excitation to the input tensor.

    Attributes:
        gate_fn (Callable): The gating function.
        avg_pool (torch.nn.AdaptiveAvgPool2d): Adaptive average pooling layer.
        conv_reduce (torch.nn.Conv2d): Reduction convolution layer.
        act1 (Callable): Activation function.
        conv_expand (torch.nn.Conv2d): Expansion convolution layer.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_channels     : int,
        se_ratio        : float    = 0.25,
        reduced_channels: int      = None,
        act_layer       : Callable = nn.ReLU,
        gate_fn         : Callable = hard_sigmoid,
        divisor         : int      = 4,
        **_
    ):
        """Initialize a new instance.

        Args:
            in_channels: Number of input channels.
            se_ratio: Squeeze-and-excitation ratio. Defaults to 0.25.
            reduced_channels: Number of reduced channels. Defaults to None.
            act_layer: Activation layer. Defaults to nn.ReLU.
            gate_fn: Gating function. Defaults to hard_sigmoid.
            divisor: Divisor for channel number. Defaults to 4.
        """
        super().__init__()
        self.gate_fn     = gate_fn
        reduced_channels = _make_divisible((reduced_channels or in_channels) * se_ratio, divisor)
        self.avg_pool    = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_channels, reduced_channels, 1, bias=True)
        self.act1        = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_channels, in_channels, 1, bias=True)

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x    = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):
    """Convolution-Batch Normalization-Activation block.

    Apply convolution, batch normalization, and activation to the input tensor.

    Attributes:
        conv (torch.nn.Conv2d): Convolution layer.
        bn1 (torch.nn.BatchNorm2d): Batch normalization layer.
        act1 (Callable): Activation function.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : int,
        stride      : int      = 1,
        act_layer   : Callable = nn.ReLU
    ):
        """Initialize a new instance.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolution kernel.
            stride: Stride of the convolution. Defaults to 1.
            act_layer: Activation layer. Defaults to nn.ReLU.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn1  = nn.BatchNorm2d(out_channels)
        self.act1 = act_layer(inplace=True)

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x

# endregion


# ==============================================================================
# region GHOST MODULES
# ==============================================================================

class GhostModule(nn.Module):
    """Ghost module.

    Generate more features from cheap operations.

    References:
        - Paper: "GhostNet: More Features from Cheap Operations," CVPR 2020.
        - Code: https://github.com/phlong3105/Efficient-AI-Backbones/tree/master/ghostnet_pytorch

    Attributes:
        out_channels (int): Number of output channels.
        primary_conv (torch.nn.Sequential): Primary convolution layer.
        cheap_operation (torch.nn.Sequential): Cheap operation layer.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : int  = 1,
        ratio       : int  = 2,
        dw_size     : int  = 3,
        stride      : int  = 1,
        relu        : bool = True
    ):
        """Initialize a new instance.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolution kernel. Defaults to 1.
            ratio: Ratio of primary to cheap channels. Defaults to 2.
            dw_size: Size of the depthwise convolution kernel. Defaults to 3.
            stride: Stride of the convolution. Defaults to 1.
            relu: If True, apply ReLU activation. Defaults to True.
        """
        super().__init__()
        self.out_channels = out_channels
        init_channels     = math.ceil(out_channels / ratio)
        new_channels      = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        y  = torch.cat([x1, x2], dim=1)
        return y[:, :self.out_channels, :, :]


class GhostBottleneck(nn.Module):
    """Ghost bottleneck w/ optional SE.

    Apply Ghost bottleneck with optional Squeeze-and-Excitation.

    References:
        - Paper: "GhostNet: More Features from Cheap Operations," CVPR 2020.
        - Code: https://github.com/phlong3105/Efficient-AI-Backbones/tree/master/ghostnet_pytorch

    Attributes:
        stride (int): Stride of the convolution.
        ghost1 (GhostModule): First Ghost module.
        conv_dw (torch.nn.Conv2d): Depthwise convolution layer.
        bn_dw (torch.nn.BatchNorm2d): Batch normalization layer.
        se (SqueezeExcite | None): Squeeze-and-Excitation block.
        ghost2 (GhostModule): Second Ghost module.
        shortcut (torch.nn.Sequential): Shortcut connection.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_channels   : int,
        mid_channels  : int,
        out_channels  : int,
        dw_kernel_size: int   = 3,
        stride        : int   = 1,
        se_ratio      : float = 0.0,
        relu          : bool  = True
    ):
        """Initialize a new instance.

        Args:
            in_channels: Number of input channels.
            mid_channels: Number of middle channels.
            out_channels: Number of output channels.
            dw_kernel_size: Size of the depthwise convolution kernel. Defaults
                to 3.
            stride: Stride of the convolution. Defaults to 1.
            se_ratio: Squeeze-and-excitation ratio. Defaults to 0.0.
            relu: If True, apply ReLU activation. Defaults to True.
        """
        super().__init__()
        has_se      = se_ratio is not None and se_ratio > 0.0
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_channels, mid_channels, relu=relu)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                in_channels  = mid_channels,
                out_channels = mid_channels,
                kernel_size  = dw_kernel_size,
                stride       = stride,
                padding      = (dw_kernel_size - 1) // 2,
                groups       = mid_channels,
                bias         = False
            )
            self.bn_dw = nn.BatchNorm2d(mid_channels)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_channels, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_channels, out_channels, relu=False)

        # Shortcut
        if in_channels == out_channels and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels  = in_channels,
                    out_channels = in_channels,
                    kernel_size  = dw_kernel_size,
                    stride       = stride,
                    padding      = (dw_kernel_size - 1) // 2,
                    groups       = in_channels,
                    bias         = False
                ),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        residual = x
        # 1st ghost bottleneck
        x = self.ghost1(x)
        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)
        # 2nd ghost bottleneck
        x  = self.ghost2(x)
        # Add shortcut
        x += self.shortcut(residual)
        return x


class GhostModuleV2(nn.Module):
    """Ghost module V2 with long-range attention.

    Generate more features from cheap operations with long-range attention.

    References:
        - Paper: "GhostNetV2: Enhance Cheap Operation with Long-Range Attention," NeurIPS 2022.
        - Code: https://github.com/phlong3105/Efficient-AI-Backbones/tree/master/ghostnetv2_pytorch

    Attributes:
        mode (str): Mode of operation ("original" or "attn").
        out_channels (int): Number of output channels.
        primary_conv (torch.nn.Sequential): Primary convolution layer.
        cheap_operation (torch.nn.Sequential): Cheap operation layer.
        gate_fn (torch.nn.Sigmoid): Gating function (only in "attn" mode).
        short_conv (torch.nn.Sequential): Short convolution layer (only in "attn" mode).
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : int  = 1,
        ratio       : int  = 2,
        dw_size     : int  = 3,
        stride      : int  = 1,
        relu        : bool = True,
        mode        : Literal["original", "attn"] = "original",
    ):
        """Initialize a new instance.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolution kernel. Defaults to 1.
            ratio: Ratio of primary to cheap channels. Defaults to 2.
            dw_size: Size of the depthwise convolution kernel. Defaults to 3.
            stride: Stride of the convolution. Defaults to 1.
            relu: If True, apply ReLU activation. Defaults to True.
            mode: Mode of operation ("original" or "attn"). Defaults to
                "original".
        """
        super().__init__()
        self.mode         = mode
        self.out_channels = out_channels
        init_channels     = math.ceil(out_channels / ratio)
        new_channels      = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        if self.mode == "attn":
            self.gate_fn    = nn.Sigmoid()
            self.short_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=out_channels, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=out_channels, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        y  = torch.cat([x1, x2], dim=1)
        y  = y[:, :self.out_channels, :, :]

        if self.mode == "attn":
            residual = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
            residual = self.gate_fn(residual)
            residual = F.interpolate(residual, size=(y.shape[-2], y.shape[-1]), mode="nearest")
            y        = y * residual

        return y


class GhostBottleneckV2(nn.Module):
    """Ghost bottleneck V2 with long-range attention and optional SE.

    Apply Ghost bottleneck V2 with long-range attention and optional
    Squeeze-and-Excitation.

    References:
        - Paper: "GhostNetV2: Enhance Cheap Operation with Long-Range Attention," NeurIPS 2022.
        - Code: https://github.com/phlong3105/Efficient-AI-Backbones/tree/master/ghostnetv2_pytorch

    Attributes:
        stride (int): Stride of the convolution.
        ghost1 (GhostModuleV2): First Ghost module.
        conv_dw (torch.nn.Conv2d): Depthwise convolution layer.
        bn_dw (torch.nn.BatchNorm2d): Batch normalization layer.
        se (SqueezeExcite | None): Squeeze-and-Excitation block.
        ghost2 (GhostModuleV2): Second Ghost module.
        shortcut (torch.nn.Sequential): Shortcut connection.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_channels   : int,
        mid_channels  : int,
        out_channels  : int,
        dw_kernel_size: int   = 3,
        stride        : int   = 1,
        se_ratio      : float = 0.0,
        layer_id      : int   = None,
        relu          : bool  = True
    ):
        """Initialize a new instance.

        Args:
            in_channels: Number of input channels.
            mid_channels: Number of middle channels.
            out_channels: Number of output channels.
            dw_kernel_size: Size of the depthwise convolution kernel. Defaults
                to 3.
            stride: Stride of the convolution. Defaults to 1.
            se_ratio: Squeeze-and-excitation ratio. Defaults to 0.0.
            layer_id: Layer ID. Defaults to None.
            relu: If True, apply ReLU activation. Defaults to True.
        """
        super().__init__()
        has_se      = se_ratio is not None and se_ratio > 0.0
        self.stride = stride

        # Point-wise expansion
        # DFC attention is usually applied in the first ghost module of the bottleneck
        # and typically not in the very first layers of the network.
        if layer_id is not None and layer_id <= 1:
            self.ghost1 = GhostModuleV2(in_channels, mid_channels, relu=relu, mode="original")
        else:
            self.ghost1 = GhostModuleV2(in_channels, mid_channels, relu=relu, mode="attn")

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                in_channels  = mid_channels,
                out_channels = mid_channels,
                kernel_size  = dw_kernel_size,
                stride       = stride,
                padding      = (dw_kernel_size - 1) // 2,
                groups       = mid_channels,
                bias         = False
            )
            self.bn_dw   = nn.BatchNorm2d(mid_channels)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_channels, se_ratio=se_ratio)
        else:
            self.se = None

        self.ghost2 = GhostModuleV2(mid_channels, out_channels, relu=False, mode="original")

        # Shortcut
        if in_channels == out_channels and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels  = in_channels,
                    out_channels = in_channels,
                    kernel_size  = dw_kernel_size,
                    stride       = stride,
                    padding      = (dw_kernel_size - 1) // 2,
                    groups       = in_channels,
                    bias         = False
                ),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if self.se is not None:
            x = self.se(x)
        x  = self.ghost2(x)
        x += self.shortcut(residual)
        return x

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
