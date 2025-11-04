#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Ghost modules.

References:
    - Paper-V1: "GhostNet: More Features from Cheap Operations," CVPR 2020.
    - Code-V2: https://github.com/phlong3105/Efficient-AI-Backbones/tree/master/ghostnet_pytorch
    
    - Paper-V2: "GhostNetV2: Enhance Cheap Operation with Long-Range Attention," NeurIPS 2022.
    - Code-V2: https://github.com/phlong3105/Efficient-AI-Backbones/tree/master/ghostnetv2_pytorch
"""

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


# ----- Utils -----
def _make_divisible(v: int, divisor: int, min_value: int = None) -> int:
    """This function ensures that all layers have a channel number that is
    divisible by ``8``.
    
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
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(x + 3.0) / 6.0


# ----- Modules -----
class SqueezeExcite(nn.Module):
    
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
        super().__init__()
        self.gate_fn     = gate_fn
        reduced_channels = _make_divisible((reduced_channels or in_channels) * se_ratio, divisor)
        self.avg_pool    = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_channels, reduced_channels, 1, bias=True)
        self.act1        = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_channels, in_channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x    = x * self.gate_fn(x_se)
        return x

    
class ConvBnAct(nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : int,
        stride      : int      = 1,
        act_layer   : Callable = nn.ReLU
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn1  = nn.BatchNorm2d(out_channels)
        self.act1 = act_layer(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


# ----- GhostModule -----
class GhostModule(nn.Module):
    """Ghost module.
    
    References:
        - Paper: "GhostNet: More Features from Cheap Operations," CVPR 2020.
        - Code: https://github.com/phlong3105/Efficient-AI-Backbones/tree/master/ghostnet_pytorch
    """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        y  = torch.cat([x1, x2], dim=1)
        return y[:,:self.out_channels, :, :]


class GhostBottleneck(nn.Module):
    """Ghost bottleneck w/ optional SE.
    
    References:
        - Paper: "GhostNet: More Features from Cheap Operations," CVPR 2020.
        - Code: https://github.com/phlong3105/Efficient-AI-Backbones/tree/master/ghostnet_pytorch
    """

    def __init__(
        self,
        in_channels   : int,
        mid_channels  : int,
        out_channels  : int,
        dw_kernel_size: int   = 3,
        stride        : int   = 1,
        se_ratio      : float = 0.0,
        relu          : bool = True
    ):
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
        
        # shortcut
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


# ----- GhostModuleV2 -----
class GhostModuleV2(nn.Module):
    """Ghost module V2 with long-range attention.
    
    References:
        - Paper: "GhostNetV2: Enhance Cheap Operation with Long-Range Attention," NeurIPS 2022.
        - Code: https://github.com/phlong3105/Efficient-AI-Backbones/tree/master/ghostnetv2_pytorch
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : int  = 1,
        ratio       : int  = 2,
        dw_size     : int  = 3,
        stride      : int  = 1,
        relu        : bool = True,
        mode        : Literal["original", "attn"] = None,
    ):
        super().__init__()
        self.mode    = mode
        self.gate_fn = nn.Sigmoid()

        if self.mode in ["original"]:
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
        elif self.mode in ["attn"]:
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
            self.short_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=out_channels, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=out_channels, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            raise NotImplementedError(f"Not implemented mode: {self.mode}.")
      
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode in ["original"]:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            y  = torch.cat([x1, x2], dim=1)
            return y[:,:self.out_channels, :, :]
        elif self.mode in ["attn"]:
            residual = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            y  = torch.cat([x1, x2], dim=1)
            return y[:,:self.out_channels, :, :] * F.interpolate(self.gate_fn(residual), size=(y.shape[-2], y.shape[-1]), mode="nearest")


class GhostBottleneckV2(nn.Module):
    """Ghost bottleneck V2 with long-range attention and optional SE.
    
    References:
        - Paper: "GhostNetV2: Enhance Cheap Operation with Long-Range Attention," NeurIPS 2022.
        - Code: https://github.com/phlong3105/Efficient-AI-Backbones/tree/master/ghostnetv2_pytorch
    """
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
        super().__init__()
        has_se      = se_ratio is not None and se_ratio > 0.0
        self.stride = stride

        # Point-wise expansion
        if layer_id <= 1:
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
        
        # shortcut
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
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
