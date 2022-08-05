#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Residual Block.
"""

from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Union

import torch
from torch import nn
from torch import Tensor
from torch.nn import init

from one.core import Callable
from one.core import Int2T
from one.core import Padding4T
from one.core import RESIDUAL_BLOCKS
from one.core import to_2tuple
from one.nn.layer.act import create_act_layer

__all__ = [
    "ResidualConvAct2d",
    "ResidualDenseBlock",
    "ResidualDenseBlock5ConvLReLU",
    "ResidualInResidualDenseBlock",
    "ResidualWideActivationBlock",
    "RDB",
    "RDB5ConvLReLu",
    "ResidualConvAct",
    "RRDB",
    "RWAB",
]


# MARK: - Modules

@RESIDUAL_BLOCKS.register(name="residual_conv_act2d")
class ResidualConvAct2d(nn.Module):
    """Basic Residual Conv2d + Act block."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels   : int,
        out_channels  : int,
        kernel_size   : Int2T,
        stride        : Int2T              = (1, 1),
        padding       : Optional[Padding4T] = 0,
        dilation      : Int2T              = (1, 1),
        groups        : int                 = 1,
        bias          : bool                = True,
        padding_mode  : str                 = "zeros",
        device        : Any                 = None,
        dtype         : Any                 = None,
        apply_act     : bool                = True,
        act_layer     : Optional[Callable]  = nn.ReLU,
        inplace       : bool                = True,
        **_
    ):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        dilation    = to_2tuple(dilation)

        self.conv = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype
        )
        self.act = create_act_layer(apply_act, act_layer, inplace)
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([x, self.act(self.conv(x))], 1)


@RESIDUAL_BLOCKS.register(name="residual_dense_block")
class ResidualDenseBlock(nn.Module):
    """Densely-Connected Residual block with activation layer. This is a more
    generalize version of:
        https://github.com/xinntao/ESRGAN/blob/master/RRDBNet_arch.py

    Args:
        num_layers (int):
            Number of conv layers in the residual block.
        in_channels (int):
            Number of channels in the input image.
        growth_channels (int):
            Growth channel, i.e. intermediate channels.
        kernel_size (Int2T):
            Size of the convolving kernel.
        lff_kernel_size (Int2T):
            Size of the convolving kernel for the last conv layer.
        stride (Int2T):
            Stride of the convolution. Default: `(1, 1)`.
        lff_stride (Int2T):
            Stride of the convolution of the last layer. Default: `(1, 1)`.
        padding (Padding4T, optional):
            Padding added to both sides of the input. Default: `0`.
        lff_padding:
            Padding added to both sides of the input of the last conv layer.
            Default: `0`.
        dilation (Int2T):
            Defaults: `(1, 1)`.
        groups (int):
            Default: `1`.
        bias (bool):
            Default: `True`.
        padding_mode (str):
            Defaults: `zeros`.
        device (Any):
            Defaults: `None`.
        dtype (Any):
            Defaults: `None`.
        apply_act (bool):
            Should use activation layer. Default: `True`.
        act_layer (Callable, optional):
            Activation layer or the name to build the activation layer.
        inplace (bool):
            Perform in-place activation. Default: `True`.
        residual_scale (float):
            It scales down the residuals by multiplying a constant between 0
            and 1 before adding them to the main path to prevent instability.
            Default: `0.2`.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        num_layers     : int,
        in_channels    : int,
        growth_channels: int,
        kernel_size    : Int2T,
        lff_kernel_size: Int2T,
        stride         : Int2T              = (1, 1),
        lff_stride     : Int2T              = (1, 1),
        padding        : Optional[Padding4T] = 0,
        lff_padding    : Optional[Padding4T] = 0,
        dilation       : Int2T              = (1, 1),
        groups         : int                 = 1,
        bias           : bool                = True,
        padding_mode   : str                 = "zeros",
        device         : Any                 = None,
        dtype          : Any                 = None,
        apply_act      : bool                = True,
        act_layer      : Optional[Callable]  = nn.ReLU(),
        inplace        : bool                = True,
        residual_scale : float               = 0.2,
        **_
    ):
        super().__init__()
        self.num_layers     = num_layers
        self.residual_scale = residual_scale
        self.layers = nn.Sequential(
            *[
                ResidualConvAct(
                    in_channels  = in_channels + i * growth_channels,
                    out_channels = growth_channels,
                    kernel_size  = kernel_size,
                    stride       = stride,
                    padding      = padding,
                    dilation     = dilation,
                    groups       = groups,
                    bias         = bias,
                    padding_mode = padding_mode,
                    device       = device,
                    dtype        = dtype,
                    apply_act    = apply_act,
                    act_layer    = act_layer,
                    inplace      = inplace,
                )
                for i in range(num_layers)
            ]
        )
        #  local feature fusion
        self.lff = nn.Conv2d(
            in_channels  = in_channels + num_layers * growth_channels,
            out_channels = growth_channels,
            kernel_size  = lff_kernel_size,
            stride       = lff_stride,
            padding      = lff_padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
 
    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        return x + self.residual_scale * self.lff(self.layers(x))


@RESIDUAL_BLOCKS.register(name="residual_dense_block_5conv_lrelu")
class ResidualDenseBlock5ConvLReLU(ResidualDenseBlock):
    """Densely-Connected Residual block with 5 convolution layers + Leaky ReLU.
    This is a similar implementation of:
        https://github.com/xinntao/ESRGAN/blob/master/RRDBNet_arch.py
    """

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels    : int                 = 64,
        growth_channels: int                 = 32,
        kernel_size    : Int2T              = (3, 3),
        lff_kernel_size: Int2T              = (3, 3),
        stride         : Int2T              = (1, 1),
        lff_stride     : Int2T              = (1, 1),
        padding        : Optional[Padding4T] = 0,
        lff_padding    : Optional[Padding4T] = 0,
        dilation       : Int2T              = (1, 1),
        groups         : int                 = 1,
        bias           : bool                = True,
        padding_mode   : str                 = "zeros",
        device         : Any                 = None,
        dtype          : Any                 = None,
        apply_act      : bool                = True,
        inplace        : bool                = True,
        residual_scale : float               = 0.2,
        **_
    ):
        super().__init__(
            num_layers      = 4,
            in_channels     = in_channels,
            growth_channels = growth_channels,
            kernel_size     = kernel_size,
            lff_kernel_size = lff_kernel_size,
            stride          = stride,
            lff_stride      = lff_stride,
            padding         = padding,
            lff_padding     = lff_padding,
            dilation        = dilation,
            groups          = groups,
            bias            = bias,
            padding_mode    = padding_mode,
            device          = device,
            dtype           = dtype,
            apply_act       = apply_act,
            act_layer       = nn.LeakyReLU(0.2, inplace=True),
            inplace         = inplace,
            residual_scale  = residual_scale,
        )
        
        # NOTE: Initialization
        self.initialize_weights([self.layers, self.lff], 0.1)
    
    # MARK: Configure

    # noinspection PyMethodMayBeStatic
    def initialize_weights(
        self, net_l: Union[list, nn.Module], scale: float = 1.0
    ):
        if not isinstance(net_l, list):
            net_l = [net_l]
       
        for net in net_l:
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                    m.weight.data *= scale  # For residual block
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


@RESIDUAL_BLOCKS.register(name="residual_in_residual_dense_block")
class ResidualInResidualDenseBlock(nn.Module):
    """Residual in Residual Dense Block with 3 Residual Dense Blocks.
    
    References:
        https://github.com/xinntao/ESRGAN/blob/master/RRDBNet_arch.py
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels    : int,
        growth_channels: int   = 32,
        residual_scale : float = 0.2,
        *args, **kwargs
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.rdb1 = RDB5ConvLReLu(
            in_channels, growth_channels, residual_scale=0.2, *args, **kwargs
        )
        self.rdb2 = RDB5ConvLReLu(
            in_channels, growth_channels, residual_scale=0.2, *args, **kwargs
        )
        self.rdb3 = RDB5ConvLReLu(
            in_channels, growth_channels, residual_scale=0.2, *args, **kwargs
        )

    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * self.residual_scale + x


@RESIDUAL_BLOCKS.register(name="residual_wide_activation_block")
class ResidualWideActivationBlock(nn.Module):
    """Conv2d + BN + Act + Conv2d + BN."""

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels    : int,
        expand         : int                = 4,
        kernel_size    : Int2T             = (3, 3),
        stride         : Int2T             = (1, 1),
        padding        : Padding4T          = 1,
        dilation       : Int2T             = (1, 1),
        groups         : int                = 1,
        bias           : bool               = False,
        padding_mode   : str                = "zeros",
        device         : Any                = None,
        dtype          : Any                = None,
        apply_act      : bool               = True,
        act_layer      : Optional[Callable] = nn.ReLU(inplace=True),
        inplace        : bool               = True,
        residual_scale : float              = 0.2,
        **_
    ):
        super().__init__()
        kernel_size         = to_2tuple(kernel_size)
        stride              = to_2tuple(stride)
        dilation            = to_2tuple(dilation)
        self.residual_scale = residual_scale
        
        self.conv1 = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels * expand,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.bn1   = nn.BatchNorm2d(in_channels * expand)
        self.act   = create_act_layer(apply_act, act_layer, inplace)
        self.conv2 = nn.Conv2d(
            in_channels  = in_channels * expand,
            out_channels = in_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.bn12  = nn.BatchNorm2d(in_channels)

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(x)
        out = self.act(x)
        out = self.conv2(x)
        out = self.bn1(x)
        out = out * self.residual_scale + x
        return out


# MARK: - Alias

RDB             = ResidualDenseBlock
RDB5ConvLReLu   = ResidualDenseBlock5ConvLReLU
ResidualConvAct = ResidualConvAct2d
RRDB            = ResidualInResidualDenseBlock
RWAB            = ResidualWideActivationBlock


# MARK: - Register

RESIDUAL_BLOCKS.register(name="rdb",               module=RDB)
RESIDUAL_BLOCKS.register(name="rdb_5conv_lrelu"  , module=RDB5ConvLReLu)
RESIDUAL_BLOCKS.register(name="residual_conv_act", module=ResidualConvAct)
RESIDUAL_BLOCKS.register(name="rrdb",              module=RRDB)
RESIDUAL_BLOCKS.register(name="rwab",              module=RWAB)
