#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements blueprint separable convolutional layers."""

from __future__ import annotations

__all__ = [
    "BSConv2dS",
    "BSConv2dU",
]

import math

import torch
from torch import nn

from mon.core import _size_2_t
from mon.nn.layer import normalization


# region Blueprint Separable Convolution

class BSConv2dS(nn.Module):
    """Unconstrained Blueprint Separable Conv2d adopted from the paper:
    `"Rethinking Depthwise Separable Convolutions: How Intra-Kernel Correlations
    Lead to Improved MobileNets" <https://arxiv.org/abs/2003.13549>`__
    
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
        bias            : bool            = True,
        padding_mode    : str             = "zeros",
        p               : float           = 0.25,
        min_mid_channels: int             = 4,
        with_bn         : bool            = False,
        bn_kwargs       : dict | None     = None,
        *args, **kwargs
    ):
        super().__init__()
        assert 0.0 <= p <= 1.0
        mid_channels = min(in_channels, max(min_mid_channels, math.ceil(p * in_channels)))
        if bn_kwargs is None:
            bn_kwargs = {}
        # Pointwise 1
        self.pw1 = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = mid_channels,
            kernel_size  = (1, 1),
            stride       = 1,
            padding      = 0,
            dilation     = 1,
            groups       = 1,
            bias         = False,
        )
        # Batchnorm
        if with_bn:
            self.bn1 = normalization.BatchNorm2d(num_features=mid_channels, **bn_kwargs)
        else:
            self.bn1 = None
        # Pointwise 2
        self.pw2 = nn.Conv2d(
            in_channels  = mid_channels,
            out_channels = out_channels,
            kernel_size  = (1, 1),
            stride       = 1,
            padding      = 0,
            dilation     = 1,
            groups       = 1,
            bias         = False,
        )
        # Batchnorm
        if with_bn:
            self.bn2 = normalization.BatchNorm2d(num_features=out_channels, **bn_kwargs)
        else:
            self.bn2 = None
        # Depthwise
        self.dw = nn.Conv2d(
            in_channels  = out_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = out_channels,
            bias         = bias,
            padding_mode = padding_mode,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw1(x)
        if self.bn1 is not None:
            y = self.bn1(y)
        y = self.pw2(y)
        if self.bn2 is not None:
            y = self.bn2(y)
        y = self.dw(y)
        return y
    
    def regularization_loss(self):
        w   = self.pw1.weight[:, :, 0, 0]
        wwt = torch.mm(w, torch.transpose(w, 0, 1))
        i   = torch.eye(wwt.shape[0], device=wwt.device)
        return torch.norm(wwt - i, p="fro")


class BSConv2dU(nn.Module):
    """Unconstrained Blueprint Separable Conv2d adopted from the paper:
    `"Rethinking Depthwise Separable Convolutions: How Intra-Kernel Correlations
    Lead to Improved MobileNets" <https://arxiv.org/abs/2003.13549>`__
    
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
        bias        : bool            = True,
        padding_mode: str             = "zeros",
        with_bn     : bool            = False,
        bn_kwargs   : dict | None     = None,
        *args, **kwargs
    ):
        super().__init__()
        if bn_kwargs is None:
            bn_kwargs = {}
        # Pointwise
        self.pw = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = (1, 1),
            stride       = 1,
            padding      = 0,
            dilation     = 1,
            groups       = 1,
            bias         = False,
        )
        # Batchnorm
        if with_bn:
            self.bn = normalization.BatchNorm2d(num_features=out_channels, **bn_kwargs)
        else:
            self.bn = None
        # Depthwise
        self.dw = nn.Conv2d(
            in_channels  = out_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = out_channels,
            bias         = bias,
            padding_mode = padding_mode,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw(x)
        if self.bn is not None:
            y = self.bn(y)
        y = self.dw(y)
        return y

# endregion
