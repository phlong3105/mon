#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convolution + Activation Layer.
"""

from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Union

from torch import nn

from one.core import Callable
from one.core import CONV_ACT_LAYERS
from one.core import Int2T
from one.core import Padding4T
from one.core import to_2tuple
from one.nn.layer.act import create_act_layer
from one.nn.layer.padding import autopad

__all__ = [
    "ConvAct2d",
    "ConvMish2d",
    "ConvReLU2d",
    "ConvSigmoid2d",
    "ConvTransposeAct2d",
    "ConvAct",
    "ConvMish",
    "ConvReLU",
    "ConvSigmoid",
    "ConvTransposeAct",
]


# MARK: - Modules

@CONV_ACT_LAYERS.register(name="conv_act2d")
class ConvAct2d(nn.Sequential):
    """Conv2d + Act.

    Args:
        in_channels (int):
            Number of channels in the input image.
        out_channels (int):
            Number of channels produced by the convolution.
        kernel_size (Int2T):
            Size of the convolving kernel.
        stride (Int2T):
            Stride of the convolution. Default: `(1, 1)`.
        padding (Padding4T, optional):
            Zero-padding added to both sides of the input. Default: `None`.
        groups (int):
            Default: `1`.
        apply_act (bool):
            Should use activation layer. Default: `True`.
        act_layer (nn.Module, str, optional):
            Activation layer or the name to build the activation layer.
        inplace (bool):
            Perform in-place activation. Default: `True`.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Int2T,
        stride      : Int2T              = (1, 1),
        padding     : Optional[Padding4T] = None,
        dilation    : Int2T              = (1, 1),
        groups      : int                 = 1,
        bias        : bool                = False,
        padding_mode: str                 = "zeros",
        device      : Any                 = None,
        dtype       : Any                 = None,
        apply_act   : bool                = True,
        act_layer   : Union[Callable]     = nn.ReLU,
        inplace     : bool                = True,
        **_
    ):
        super().__init__()
        act_layer   = create_act_layer(apply_act, act_layer, inplace)
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        self.add_module(
            "conv", nn.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = autopad(kernel_size, padding),
                dilation     = dilation,
                groups       = groups,
                bias         = bias,
                padding_mode = padding_mode,
                device       = device,
                dtype        = dtype,
            )
        )
        self.add_module("act", act_layer)


@CONV_ACT_LAYERS.register(name="conv_transposed_act2d")
class ConvTransposeAct2d(nn.Sequential):
    """ConvTranspose2d + Act.

    Args:
        in_channels (int):
            Number of channels in the input image.
        out_channels (int):
            Number of channels produced by the convolution.
        kernel_size (Int2T):
            Size of the convolving kernel.
        stride (Int2T):
            Stride of the convolution. Default: `(1, 1)`.
        padding (Padding4T, optional):
            Zero-padding added to both sides of the input. Default: `None`.
        groups (int):
            Default: `1`.
        apply_act (bool):
            Should use activation layer. Default: `True`.
        act_layer (nn.Module, str, optional):
            Activation layer or the name to build the activation layer.
        inplace (bool):
            Perform in-place activation. Default: `True`.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Int2T,
        stride      : Int2T              = (1, 1),
        padding     : Optional[Padding4T] = None,
        dilation    : Int2T              = (1, 1),
        groups      : int                 = 1,
        bias        : bool                = False,
        padding_mode: str                 = "zeros",
        device      : Any                 = None,
        dtype       : Any                 = None,
        apply_act   : bool                = True,
        act_layer   : Union[Callable]     = nn.ReLU,
        inplace     : bool                = True,
        **_
    ):
        super().__init__()
        act_layer   = create_act_layer(apply_act, act_layer, inplace)
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        self.add_module(
            "deconv", nn.ConvTranspose2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = autopad(kernel_size, padding),
                dilation     = dilation,
                groups       = groups,
                bias         = bias,
                padding_mode = padding_mode,
                device       = device,
                dtype        = dtype,
            )
        )
        self.add_module("act", act_layer)


@CONV_ACT_LAYERS.register(name="conv_mish2d")
class ConvMish2d(nn.Sequential):
    """Conv2d + Mish.

    Args:
        in_channels (int):
            Number of channels in the input image.
        out_channels (int):
            Number of channels produced by the convolution.
        kernel_size (Int2T):
            Size of the convolving kernel.
        stride (Int2T):
            Stride of the convolution. Default: `(1, 1)`.
        padding (Padding4T, optional):
            Zero-padding added to both sides of the input. Default: `None`.
        groups (int):
            Default: `1`.
        apply_act (bool):
            Should use activation layer. Default: `True`.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Int2T,
        stride      : Int2T              = (1, 1),
        padding     : Optional[Padding4T] = None,
        dilation    : Int2T              = (1, 1),
        groups      : int                 = 1,
        bias        : bool                = False,
        padding_mode: str                 = "zeros",
        device      : Any                 = None,
        dtype       : Any                 = None,
        apply_act   : bool                = True,
        **_
    ):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        self.add_module(
            "conv", nn.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = autopad(kernel_size, padding),
                dilation     = dilation,
                groups       = groups,
                bias         = bias,
                padding_mode = padding_mode,
                device       = device,
                dtype        = dtype,
            )
        )
        self.add_module("act", nn.Mish() if apply_act else nn.Identity())


@CONV_ACT_LAYERS.register(name="conv_relu2d")
class ConvReLU2d(nn.Sequential):
    """Conv2d + ReLU.

    Args:
        in_channels (int):
            Number of channels in the input image.
        out_channels (int):
            Number of channels produced by the convolution.
        kernel_size (Int2T):
            Size of the convolving kernel.
        stride (Int2T):
            Stride of the convolution. Default: `(1, 1)`.
        padding (Padding4T, optional):
            Zero-padding added to both sides of the input. Default: `0`.
        padding_mode (str):
            Default: `zeros`.
        apply_act (bool):
            Should use activation layer. Default: `True`.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Int2T,
        stride      : Int2T              = (1, 1),
        padding     : Optional[Padding4T] = 0,
        dilation    : Int2T              = (1, 1),
        groups      : int                 = 1,
        bias        : bool                = False,
        padding_mode: str                 = "zeros",
        device      : Any                 = None,
        dtype       : Any                 = None,
        apply_act   : bool                = True,
        **_
    ):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        self.add_module(
            "conv", nn.Conv2d(
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
                dtype        = dtype,
            )
        )
        self.add_module(
            "relu", nn.ReLU(inplace=True) if apply_act else nn.Identity()
        )

 
@CONV_ACT_LAYERS.register(name="conv_sigmoid2d")
class ConvSigmoid2d(nn.Sequential):
    """Conv2d + Sigmoid.

    Args:
        in_channels (int):
            Number of channels in the input image.
        out_channels (int):
            Number of channels produced by the convolution.
        kernel_size (Int2T):
            Size of the convolving kernel. Default: `(1, 1)`.
        stride (Int2T):
            Stride of the convolution. Default: `(1, 1)`.
        padding (Padding4T, optional):
            Zero-padding added to both sides of the input. Default: `None`.
        groups (int):
            Default: `1`.
        apply_act (bool):
            Should use activation layer. Default: `True`.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Int2T,
        stride      : Int2T              = (1, 1),
        padding     : Optional[Padding4T] = None,
        dilation    : Int2T              = (1, 1),
        groups      : int                 = 1,
        bias        : bool                = False,
        padding_mode: str                 = "zeros",
        device      : Any                 = None,
        dtype       : Any                 = None,
        apply_act   : bool                = True,
        **_
    ):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        self.add_module(
            "conv", nn.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = autopad(kernel_size, padding),
                dilation     = dilation,
                groups       = groups,
                bias         = bias,
                padding_mode = padding_mode,
                device       = device,
                dtype        = dtype,
            )
        )
        self.add_module("sigmoid", nn.Sigmoid() if apply_act else nn.Identity())
       
        
# MARK: - Alias

ConvAct          = ConvAct2d
ConvTransposeAct = ConvTransposeAct2d
ConvMish         = ConvMish2d
ConvReLU         = ConvReLU2d
ConvSigmoid      = ConvSigmoid2d


# MARK: - Register

CONV_ACT_LAYERS.register(name="conv_act",            module=ConvAct)
CONV_ACT_LAYERS.register(name="conv_transposed_act", module=ConvTransposeAct)
CONV_ACT_LAYERS.register(name="conv_mish",           module=ConvMish)
CONV_ACT_LAYERS.register(name="conv_relu",           module=ConvReLU)
CONV_ACT_LAYERS.register(name="conv_sigmoid",        module=ConvSigmoid)
