#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Subspace Blueprint Separable Convolution layers.
"""

from __future__ import annotations

__all__ = [
    "ABSConv2dS",
    "ABSConv2dS1",
    "ABSConv2dS10",
    "ABSConv2dS11",
    "ABSConv2dS12",
    "ABSConv2dS13",
    "ABSConv2dS14",
    "ABSConv2dS15",
    "ABSConv2dS16",
    "ABSConv2dS17",
    "ABSConv2dS2",
    "ABSConv2dS3",
    "ABSConv2dS4",
    "ABSConv2dS5",
    "ABSConv2dS6",
    "ABSConv2dS7",
    "ABSConv2dS8",
    "ABSConv2dS9",
    "ABSConv2dU",
    "AttentionSubspaceBlueprintSeparableConv2d",
    "AttentionSubspaceBlueprintSeparableConv2d1",
    "AttentionSubspaceBlueprintSeparableConv2d10",
    "AttentionSubspaceBlueprintSeparableConv2d11",
    "AttentionSubspaceBlueprintSeparableConv2d12",
    "AttentionSubspaceBlueprintSeparableConv2d13",
    "AttentionSubspaceBlueprintSeparableConv2d14",
    "AttentionSubspaceBlueprintSeparableConv2d15",
    "AttentionSubspaceBlueprintSeparableConv2d16",
    "AttentionSubspaceBlueprintSeparableConv2d17",
    "AttentionSubspaceBlueprintSeparableConv2d2",
    "AttentionSubspaceBlueprintSeparableConv2d3",
    "AttentionSubspaceBlueprintSeparableConv2d4",
    "AttentionSubspaceBlueprintSeparableConv2d5",
    "AttentionSubspaceBlueprintSeparableConv2d6",
    "AttentionSubspaceBlueprintSeparableConv2d7",
    "AttentionSubspaceBlueprintSeparableConv2d8",
    "AttentionSubspaceBlueprintSeparableConv2d9",
    "AttentionUnconstrainedBlueprintSeparableConv2d",
    "BSConv2dS",
    "BSConv2dU",
    "SubspaceBlueprintSeparableConv2d",
    "UnconstrainedBlueprintSeparableConv2d",
]

import math
from typing import Any, Callable

import torch
from torch import nn

from mon.core import _size_2_t
from mon.nn.layer import activation, attention
from mon.nn.layer.conv import base as conv


# region Subspace Blueprint Separable Convolution

class SubspaceBlueprintSeparableConv2d(nn.Module):
    """Subspace Blueprint Separable Conv2d adopted from the paper: "Rethinking
    Depthwise Separable Convolutions: How Intra-Kernel Correlations Lead to
    Improved MobileNets".
    
    References:
        `<https://github.com/zeiss-microscopy/BSConv>`__
    """
    
    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        kernel_size     : _size_2_t,
        stride          : _size_2_t       = 1,
        padding         : _size_2_t | str = 0,
        dilation        : _size_2_t       = 1,
        groups          : int             = 1,
        bias            : bool            = True,
        padding_mode    : str             = "zeros",
        device          : Any             = None,
        dtype           : Any             = None,
        p               : float           = 0.25,
        min_mid_channels: int             = 4,
        act             : Callable        = None,
        *args, **kwargs
    ):
        super().__init__()
        mid_channels  = min(in_channels, max(min_mid_channels, math.ceil(p * in_channels)))
        self.pw_conv1 = conv.Conv2d(
            in_channels  = in_channels,
            out_channels = mid_channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            dilation     = 1,
            groups       = 1,
            bias         = False,
            padding_mode = "zeros",
            device       = device,
            dtype        = dtype,
        )
        self.act1 = activation.to_act_layer(
            act_layer= act,
            num_features = mid_channels
        )  # if norm else None
        self.pw_conv2 = conv.Conv2d(
            in_channels  = mid_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            dilation     = 1,
            groups       = 1,
            bias         = False,
            padding_mode = "zeros",
            device       = device,
            dtype        = dtype,
        )
        self.act2    = activation.to_act_layer(
            act_layer= act,
            num_features = out_channels
        )  # if norm else None
        self.dw_conv = conv.Conv2d(
            in_channels  = out_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = out_channels,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv1(x)
        if self.act1 is not None:
            y = self.act1(y)
        y = self.pw_conv2(y)
        if self.act2 is not None:
            y = self.act2(y)
        y = self.dw_conv(y)
        return y
    
    def regularization_loss(self):
        w   = self.pw_conv1.weight[:, :, 0, 0]
        wwt = torch.mm(w, torch.transpose(w, 0, 1))
        i   = torch.eye(wwt.shape[0], device=wwt.device)
        return torch.norm(wwt - i, p="fro")


BSConv2dS = SubspaceBlueprintSeparableConv2d

# endregion


# region Unconstrained Blueprint Separable Convolution

class UnconstrainedBlueprintSeparableConv2d(nn.Module):
    """Unconstrained Blueprint Separable Conv2d adopted from the paper:
    "Rethinking Depthwise Separable Convolutions: How Intra-Kernel Correlations
    Lead to Improved MobileNets," CVPR 2020.
    
    References:
        `<https://github.com/zeiss-microscopy/BSConv>`__
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        stride      : _size_2_t       = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t       = 1,
        groups      : int             = 1,
        bias        : bool            = True,
        padding_mode: str             = "zeros",
        device      : Any             = None,
        dtype       : Any             = None,
        act         : Callable        = None,
        *args, **kwargs
    ):
        super().__init__()
        self.pw_conv = conv.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            dilation     = 1,
            groups       = 1,
            bias         = False,
            padding_mode = "zeros",
            device       = device,
            dtype        = dtype,
        )
        self.act     = activation.to_act_layer(act_layer=act, num_features=out_channels)
        self.dw_conv = conv.Conv2d(
            in_channels  = out_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = out_channels,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv(x)
        if self.act is not None:
            y = self.act(y)
        y = self.dw_conv(y)
        return y


BSConv2dU = UnconstrainedBlueprintSeparableConv2d

# endregion


# region Attention Subspace Blueprint Separable Convolution

class AttentionSubspaceBlueprintSeparableConv2d(nn.Module):
    """Subspace Blueprint Separable Conv2d with Self-Attention adopted from the
    paper:
        "Rethinking Depthwise Separable Convolutions: How Intra-Kernel
        Correlations Lead to Improved MobileNets," CVPR 2020.
    
    References:
        `<https://github.com/zeiss-microscopy/BSConv>`__
    """
    
    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        kernel_size     : _size_2_t,
        stride          : _size_2_t       = 1,
        padding         : _size_2_t | str = 0,
        dilation        : _size_2_t       = 1,
        groups          : int             = 1,
        bias            : bool            = True,
        padding_mode    : str             = "zeros",
        device          : Any             = None,
        dtype           : Any             = None,
        p               : float           = 0.25,
        min_mid_channels: int             = 4,
        attn            : bool            = False,
        norm1           : Callable        = None,
        norm2           : Callable        = None,
    ):
        super().__init__()
        assert 0.0 <= p <= 1.0
        mid_channels  = min(in_channels, max(min_mid_channels, math.ceil(p * in_channels)))
        self.pw_conv1 = conv.Conv2d(
            in_channels  = in_channels,
            out_channels = mid_channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            dilation     = 1,
            groups       = 1,
            bias         = False,
            padding_mode = "zeros",
            device       = device,
            dtype        = dtype,
        )
        self.norm1    = norm1(num_features=mid_channels) if norm1 is not None else None
        self.pw_conv2 = conv.Conv2d(
            in_channels  = mid_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            dilation     = 1,
            groups       = 1,
            bias         = False,
            padding_mode = "zeros",
            device       = device,
            dtype        = dtype,
        )
        self.norm2   = norm2(num_features=out_channels) if norm2 is not None else None
        self.dw_conv = conv.Conv2d(
            in_channels  = out_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = out_channels,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.simam   = attention.SimAM() if attn else None
        # self.apply(self.init_weights)
        
    def init_weights(self, m: nn.Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                m.conv.weight.data.normal_(0.0, 0.02)
            elif hasattr(m, "dw_conv"):
                m.dw_conv.weight.data.normal_(0.0, 0.02)
            elif hasattr(m, "pw_conv"):
                m.pw_conv.weight.data.normal_(0.0, 0.02)
            else:
                m.weight.data.normal_(0.0, 0.02)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv1(x)
        if self.simam is not None:
            y = self.simam(y)
        if self.norm1 is not None:
            y = self.norm1(y)
        y = self.pw_conv2(y)
        if self.norm2 is not None:
            y = self.norm2(y)
        y = self.dw_conv(y)
        return y
    
    def regularization_loss(self):
        w   = self.pw_conv1.weight[:, :, 0, 0]
        wwt = torch.mm(w, torch.transpose(w, 0, 1))
        i   = torch.eye(wwt.shape[0], device=wwt.device)
        return torch.norm(wwt - i, p="fro")


class AttentionSubspaceBlueprintSeparableConv2d1(
    AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = self.pw_conv1(x)
        # if self.simam is not None:
        #    y = self.simam(y)
        # if self.norm1 is not None:
        #     x = self.norm1(x)
        x = self.pw_conv2(x)
        # if self.norm2 is not None:
        #     x = self.norm2(x)
        x = self.dw_conv(x)
        return x


class AttentionSubspaceBlueprintSeparableConv2d2(
    AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv1(x)
        # if self.simam is not None:
        #     y = self.simam(y)
        if self.norm1 is not None:
            y = self.norm1(y)
        y = self.pw_conv2(y)
        # if self.norm2 is not None:
        #    y = self.norm2(y)
        y = self.dw_conv(y)
        return y


class AttentionSubspaceBlueprintSeparableConv2d3(
    AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv1(x)
        # if self.simam is not None:
        #     y = self.simam(y)
        # if self.norm1 is not None:
        #    y = self.norm1(y)
        y = self.pw_conv2(y)
        if self.norm2 is not None:
            y = self.norm2(y)
        y = self.dw_conv(y)
        return y


class AttentionSubspaceBlueprintSeparableConv2d4(
    AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv1(x)
        # if self.simam is not None:
        #     y = self.simam(y)
        if self.norm1 is not None:
            y = self.norm1(y)
        y = self.pw_conv2(y)
        if self.norm2 is not None:
            y = self.norm2(y)
        y = self.dw_conv(y)
        return y


class AttentionSubspaceBlueprintSeparableConv2d5(
    AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv1(x)
        if self.simam is not None:
            y = self.simam(y)
        # if self.norm1 is not None:
        #    y = self.norm1(y)
        y = self.pw_conv2(y)
        # if self.norm2 is not None:
        #    y = self.norm2(y)
        y = self.dw_conv(y)
        return y


class AttentionSubspaceBlueprintSeparableConv2d6(
    AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv1(x)
        # if self.norm1 is not None:
        #    y = self.norm1(y)
        y = self.pw_conv2(y)
        if self.simam is not None:
            y = self.simam(y)
        # if self.norm2 is not None:
        #    y = self.norm2(y)
        y = self.dw_conv(y)
        return y


class AttentionSubspaceBlueprintSeparableConv2d7(
    AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv1(x)
        # if self.norm1 is not None:
        #    y = self.norm1(y)
        y = self.pw_conv2(y)
        # if self.norm2 is not None:
        #    y = self.norm2(y)
        y = self.dw_conv(y)
        if self.simam is not None:
            y = self.simam(y)
        return y


class AttentionSubspaceBlueprintSeparableConv2d8(
    AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv1(x)
        if self.simam is not None:
            y = self.simam(y)
        if self.norm1 is not None:
            y = self.norm1(y)
        y = self.pw_conv2(y)
        # if self.norm2 is not None:
        #    y = self.norm2(y)
        y = self.dw_conv(y)
        return y


class AttentionSubspaceBlueprintSeparableConv2d9(  # Last paper, this one is the best
    AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv1(x)
        if self.simam is not None:
            y = self.simam(y)
        # if self.norm1 is not None:
        #     y = self.norm1(y)
        y = self.pw_conv2(y)
        if self.norm2 is not None:
            y = self.norm2(y)
        y = self.dw_conv(y)
        return y


class AttentionSubspaceBlueprintSeparableConv2d10(
    AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv1(x)
        if self.norm1 is not None:
            y = self.norm1(y)
        y = self.pw_conv2(y)
        if self.simam is not None:
            y = self.simam(y)
        # if self.norm2 is not None:
        #     y = self.norm2(y)
        y = self.dw_conv(y)
        return y


class AttentionSubspaceBlueprintSeparableConv2d11(
    AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv1(x)
        # if self.norm1 is not None:
        #     y = self.norm1(y)
        y = self.pw_conv2(y)
        if self.simam is not None:
            y = self.simam(y)
        if self.norm2 is not None:
            y = self.norm2(y)
        y = self.dw_conv(y)
        return y


class AttentionSubspaceBlueprintSeparableConv2d12(
    AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv1(x)
        if self.norm1 is not None:
            y = self.norm1(y)
        y = self.pw_conv2(y)
        if self.norm2 is not None:
            y = self.norm2(y)
        y = self.dw_conv(y)
        if self.simam is not None:
            y = self.simam(y)
        return y


class AttentionSubspaceBlueprintSeparableConv2d13(
    AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv1(x)
        if self.simam is not None:
            y = self.simam(y)
        if self.norm1 is not None:
            y = self.norm1(y)
        y = self.pw_conv2(y)
        if self.norm2 is not None:
            y = self.norm2(y)
        y = self.dw_conv(y)
        return y


class AttentionSubspaceBlueprintSeparableConv2d14(
    AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.simam(x)
        y = self.pw_conv1(y)
        # if self.norm1 is not None:
        #    y = self.norm1(y)
        # y = self.simam(x)
        y = self.pw_conv2(y)
        # if self.norm2 is not None:
        #    y = self.norm2(y)
        # y = self.simam(x)
        y = self.dw_conv(y)
        # y = self.simam(x)
        return y
    

class AttentionSubspaceBlueprintSeparableConv2d15(
    AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.simam is not None:
            y = self.simam(x)
        y = self.pw_conv1(y)
        # if self.norm1 is not None:
        #     y = self.norm1(y)
        if self.simam is not None:
            y = self.simam(y)
        y = self.pw_conv2(y)
        # if self.norm2 is not None:
        #     y = self.norm2(y)
        # y = self.simam(y)
        y = self.dw_conv(y)
        # y = self.simam(y)
        return y
    

class AttentionSubspaceBlueprintSeparableConv2d16(
    AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.simam is not None:
            y = self.simam(x)
        y = self.pw_conv1(y)
        # if self.norm1 is not None:
        #     y = self.norm1(y)
        if self.simam is not None:
            y = self.simam(y)
        y = self.pw_conv2(y)
        # if self.norm2 is not None:
        #     y = self.norm2(y)
        if self.simam is not None:
            y = self.simam(y)
        y = self.dw_conv(y)
        # y = self.simam(y)
        return y


class AttentionSubspaceBlueprintSeparableConv2d17(
    AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.simam is not None:
            y = self.simam(x)
        y = self.pw_conv1(y)
        # if self.norm1 is not None:
        #     y = self.norm1(y)
        if self.simam is not None:
            y = self.simam(y)
        y = self.pw_conv2(y)
        # if self.norm2 is not None:
        #     y = self.norm2(y)
        if self.simam is not None:
            y = self.simam(y)
        y = self.dw_conv(y)
        if self.simam is not None:
            y = self.simam(y)
        return y


ABSConv2dS   = AttentionSubspaceBlueprintSeparableConv2d
ABSConv2dS1  = AttentionSubspaceBlueprintSeparableConv2d1
ABSConv2dS2  = AttentionSubspaceBlueprintSeparableConv2d2
ABSConv2dS3  = AttentionSubspaceBlueprintSeparableConv2d3
ABSConv2dS4  = AttentionSubspaceBlueprintSeparableConv2d4
ABSConv2dS5  = AttentionSubspaceBlueprintSeparableConv2d5
ABSConv2dS6  = AttentionSubspaceBlueprintSeparableConv2d6
ABSConv2dS7  = AttentionSubspaceBlueprintSeparableConv2d7
ABSConv2dS8  = AttentionSubspaceBlueprintSeparableConv2d8
ABSConv2dS9  = AttentionSubspaceBlueprintSeparableConv2d9
ABSConv2dS10 = AttentionSubspaceBlueprintSeparableConv2d10
ABSConv2dS11 = AttentionSubspaceBlueprintSeparableConv2d11
ABSConv2dS12 = AttentionSubspaceBlueprintSeparableConv2d12
ABSConv2dS13 = AttentionSubspaceBlueprintSeparableConv2d13
ABSConv2dS14 = AttentionSubspaceBlueprintSeparableConv2d14
ABSConv2dS15 = AttentionSubspaceBlueprintSeparableConv2d15
ABSConv2dS16 = AttentionSubspaceBlueprintSeparableConv2d16
ABSConv2dS17 = AttentionSubspaceBlueprintSeparableConv2d17

# endregion


# region Attention Unconstrained Blueprint Separable Convolution

class AttentionUnconstrainedBlueprintSeparableConv2d(nn.Module):
    """Subspace Blueprint Separable Conv2d with Self-Attention adopted from the
    paper:
        "Rethinking Depthwise Separable Convolutions: How Intra-Kernel
        Correlations Lead to Improved MobileNets," CVPR 2020.
    
    References:
        `<https://github.com/zeiss-microscopy/BSConv>`__
    """
    
    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        kernel_size     : _size_2_t,
        stride          : _size_2_t       = 1,
        padding         : _size_2_t | str = 0,
        dilation        : _size_2_t       = 1,
        groups          : int             = 1,
        bias            : bool            = True,
        padding_mode    : str             = "zeros",
        device          : Any             = None,
        dtype           : Any             = None,
        p               : float           = 0.25,
        min_mid_channels: int             = 4,
        norm            : Callable        = None,
    ):
        super().__init__()
        self.pw_conv = conv.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            dilation     = 1,
            groups       = 1,
            bias         = False,
            padding_mode = "zeros",
            device       = device,
            dtype        = dtype,
        )
        self.norm    = norm(num_features=out_channels) if norm is not None else None
        self.dw_conv = conv.Conv2d(
            in_channels  = out_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = out_channels,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.simam   = attention.SimAM()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv(x)
        y = self.simam(y)
        if self.norm is not None:
            y = self.norm(y)
        y = self.dw_conv(y)
        return y


ABSConv2dU = AttentionUnconstrainedBlueprintSeparableConv2d

# endregion
