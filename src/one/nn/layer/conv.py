#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convolution Layers.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any
from typing import Optional
from typing import Union

import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from one.core import CONV_LAYERS
from one.core import Int2T
from one.core import Padding2T
from one.core import Padding4T
from one.core import to_2tuple
from one.nn.layer.padding import get_padding
from one.nn.layer.padding import get_padding_value
from one.nn.layer.padding import pad_same

__all__ = [
    "conv2d_same",
    "create_conv2d",
    "create_conv2d_pad",
    "CondConv2d",
    "Conv2d3x3",
    "Conv2dTf",
    "Conv2dSame",
    "CrossConv2d",
    "DepthwiseConv2d",
    "MixedConv2d",
    "PointwiseConv2d",
    "ScaledStdConv2d",
    "ScaledStdConv2dSame",
    "SeparableConv2d",
    "StdConv2d",
    "StdConv2dSame",
    "CondConv",
    "Conv3x3",
    "ConvTf",
    "ConvSame",
    "CrossConv",
    "DepthwiseConv",
    "MixedConv",
    "PointwiseConv",
    "ScaledStdConv",
    "ScaledStdConvSame",
    "SeparableConv",
    "StdConv",
    "StdConvSame",
]


# MARK: - Functional

def _split_channels(num_channels: int, num_groups: int):
    split     = [num_channels // num_groups for _ in range(num_groups)]
    split[0] += num_channels - sum(split)
    return split


def conv2d_same(
    x       : Tensor,
    weight  : Tensor,
    bias    : Optional[Tensor]    = None,
    stride  : Int2T              = (1, 1),
    padding : Optional[Padding4T] = 0,
    dilation: Int2T              = (1, 1),
    groups  : int                 = 1,
    **_
):
    """Functional interface for Same Padding Convolution 2D.

    Args:
        x (Tensor):
            Input.
        weight (Tensor):
            Weight.
        bias (Tensor, optional):
            Bias value.
        stride (Int2T):
            Stride of the convolution. Default: `(1, 1)`.
        padding (Padding4T, optional):
            Zero-padding added to both sides of the input. Default: `0`.
        dilation (Int2T):
            Spacing between kernel elements. Default: `(1, 1)`.
        groups (int):
            Number of blocked connections from input channels to output
            channels. Default: `1`.

    Returns:
        input (Tensor):
            Output image.
    """
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(x, weight, bias, stride, padding, dilation, groups)


# MARK: - Modules

# noinspection PyMethodMayBeStatic
@CONV_LAYERS.register(name="cond_conv2d")
class CondConv2d(nn.Module):
    """Conditionally Parameterized Convolution. Inspired by:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/condconv/condconv_layers.py

    Grouped convolution hackery for parallel execution of the per-sample kernel
    filtering inspired by this discussion:
    https://github.com/pytorch/pytorch/issues/17983
    """

    __constants__ = ["in_channels", "out_channels", "dynamic_padding"]

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Int2T,
        stride      : Int2T              = (1, 1),
        padding     : Optional[Padding4T] = "",
        dilation    : Int2T              = (1, 1),
        groups      : int                 = 1,
        bias        : Optional[bool]      = False,
        num_experts : int                 = 4,
        *args, **kwargs
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = to_2tuple(kernel_size)
        self.stride       = to_2tuple(stride)

        padding_val, is_padding_dynamic = get_padding_value(
			padding, kernel_size, stride=stride, dilation=dilation
		)
        # if in forward to work with torchscript
        self.dynamic_padding = is_padding_dynamic
        self.padding         = to_2tuple(padding_val)
        self.dilation        = to_2tuple(dilation)
        self.groups          = groups
        self.num_experts     = num_experts

        self.weight_shape = (
			(self.out_channels, self.in_channels // self.groups) +
			self.kernel_size
		)
        weight_num_param = 1
        for wd in self.weight_shape:
            weight_num_param *= wd
        self.weight = torch.nn.Parameter(
			Tensor(self.num_experts, weight_num_param)
		)

        if bias:
            self.bias_shape = (self.out_channels,)
            self.bias = torch.nn.Parameter(
				Tensor(self.num_experts, self.out_channels)
			)
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    # MARK: Configure

    def reset_parameters(self):
        init_weight = self.get_condconv_initializer(
            partial(nn.init.kaiming_uniform_, a=math.sqrt(5)),
			self.num_experts, self.weight_shape
        )
        init_weight(self.weight)
        if self.bias is not None:
            fan_in    = np.prod(self.weight_shape[1:])
            bound     = 1 / math.sqrt(fan_in)
            init_bias = self.get_condconv_initializer(
                partial(nn.init.uniform_, a=-bound, b=bound), self.num_experts,
				self.bias_shape
            )
            init_bias(self.bias)

    def get_condconv_initializer(self, initializer, num_experts: int, expert_shape):
        def condconv_initializer(weight: Tensor):
            """CondConv initializer function."""
            num_params = np.prod(expert_shape)
            if (
                len(weight.shape) != 2 or
                weight.shape[0] != num_experts or
                weight.shape[1] != num_params
            ):
                raise ValueError("CondConv variables must have shape [num_experts, num_params].")
            for i in range(num_experts):
                initializer(weight[i].view(expert_shape))
    
        return condconv_initializer
    
    # MARK: Forward Pass

    def forward(self, x: Tensor, routing_weights: Tensor) -> Tensor:
        b, c, h, w = x.shape
        weight     = torch.matmul(routing_weights, self.weight)
        new_weight_shape = (
			(b * self.out_channels, self.in_channels // self.groups) +
			self.kernel_size
		)
        weight = weight.view(new_weight_shape)
        bias   = None

        if self.bias is not None:
            bias = torch.matmul(routing_weights, self.bias)
            bias = bias.view(b * self.out_channels)
        # Move batch elements with channels so each batch element can be
		# efficiently convolved with separate kernel
        x = x.view(1, b * c, h, w)
        if self.dynamic_padding:
            out = conv2d_same(
                x, weight, bias, stride=self.stride, padding=self.padding,
				dilation=self.dilation, groups=self.groups * b
            )
        else:
            out = F.conv2d(
                x, weight, bias, stride=self.stride, padding=self.padding,
				dilation=self.dilation, groups=self.groups * b
            )
        out = out.permute([1, 0, 2, 3]).view(
            b, self.out_channels, out.shape[-2], out.shape[-1]
        )

        # Literal port (from TF definition)
        # input = torch.split(input, 1, 0)
        # weight = torch.split(weight, 1, 0)
        # if self.bias is not None:
        #     bias = torch.matmul(routing_weights, self.bias)
        #     bias = torch.split(bias, 1, 0)
        # else:
        #     bias = [None] * B
        # out = []
        # for xi, wi, bi in zip(input, weight, bias):
        #     wi = wi.view(*self.weight_shape)
        #     if bi is not None:
        #         bi = bi.view(*self.bias_shape)
        #     out.append(self.conv_fn(
        #         xi, wi, bi, stride=self.stride, padding=self.padding,
        #         dilation=self.dilation, groups=self.groups))
        # out = torch.cat(out, 0)
        return out


@CONV_LAYERS.register(name="conv2d3x3")
class Conv2d3x3(nn.Conv2d):
    """Conv2d with 3x3 kernel_size."""

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : Int2T              = (1, 1),
        padding     : Optional[Padding4T] = 0,
        dilation    : Int2T              = (1, 1),
        groups      : int                 = 1,
        bias        : bool                = True,
        padding_mode: str                 = "zeros",
        device      : Any                 = None,
        dtype       : Any                 = None,
        **_
    ):
        super().__init__(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = (3, 3),
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )


@CONV_LAYERS.register(name="conv2d_tf")
class Conv2dTf(nn.Conv2d):
    """Implementation of 2D convolution in TensorFlow with `padding` as "same",
    which applies padding to input (if needed) so that input image gets fully
    covered by filter and stride you specified. For stride `1`, this will
    ensure that output image size is same as input. For stride of 2, output
    dimensions will be half, for example.
    
    Args:
        in_channels (int):
            Number of channels in the input image.
        out_channels (int):
            Number of channels produced by the convolution.
        kernel_size (Int2T):
            Size of the convolving kernel
        stride (Int2T):
            Stride of the convolution. Default: `1`.
        padding (Padding2T, optional):
            Zero-padding added to both sides of the input. Default: `0`.
        dilation (str, Int2T, optional):
            Spacing between kernel elements. Default: `1`.
        groups (int):
            Number of blocked connections from input channels to output
            channels. Default: `1`.
        bias (bool):
            If `True`, adds a learnable bias to the output. Default: `True`.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Int2T,
        stride      : Int2T              = (1, 1),
        padding     : Optional[Padding2T] = 0,
        dilation    : Int2T              = (1, 1),
        groups      : int                 = 1,
        bias        : bool                = True,
        padding_mode: str                 = "zeros",
        device      : Any                 = None,
        dtype       : Any                 = None,
        **_
    ):
        super().__init__(
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
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x (Tensor):
                Input image.

        Returns:
            yhat (Tensor):
                Output image.
        """
        img_h, img_w       = x.size()[-2:]
        kernel_h, kernel_w = self.weight.size()[-2:]
        stride_h, stride_w = self.stride
        output_h = math.ceil(img_h / stride_h)
        output_w = math.ceil(img_w / stride_w)
        pad_h    = max((output_h - 1) * self.stride[0] + (kernel_h - 1) *
                       self.dilation[0] + 1 - img_h, 0)
        pad_w    = max((output_w - 1) * self.stride[1] + (kernel_w - 1) *
                       self.dilation[1] + 1 - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2,
                          pad_h // 2, pad_h - pad_h // 2])
        yhat = F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)
        return yhat


@CONV_LAYERS.register(name="conv2d_same")
class Conv2dSame(nn.Conv2d):
    """Tensorflow like `SAME` convolution wrapper for 2D convolutions."""

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
        bias        : bool                = True,
        padding_mode: str                 = "zeros",
        device      : Any                 = None,
        dtype       : Any                 = None,
        **_
    ):
        super().__init__(
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

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        return conv2d_same(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation,
            self.groups
        )


@CONV_LAYERS.register(name="cross_conv2d")
class CrossConv2d(nn.Module):
    """Cross Convolution Downsample."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : int                 = 3,
        stride      : int                 = 1,
        padding     : Optional[Padding4T] = None,
        dilation    : Int2T              = (1, 1),
        groups      : int                 = 1,
        bias        : bool                = False,
        padding_mode: str                 = "zeros",
        device      : Any                 = None,
        dtype       : Any                 = None,
        expansion   : float               = 1.0,
        shortcut    : bool                = False,
        **_
    ):
        super().__init__()
        c_ = int(out_channels * expansion)  # Hidden channels
        from one.nn import ConvBnMish
        self.cv1 = ConvBnMish(
            in_channels  = in_channels,
            out_channels = c_,
            kernel_size  = (1, kernel_size),
            stride       = (1, stride),
            padding      = padding,
            dilation     = dilation,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.cv2 = ConvBnMish(
            in_channels  = c_,
            out_channels = out_channels,
            kernel_size  = (kernel_size, 1),
            stride       = (stride     , 1),
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.add = shortcut and in_channels == out_channels
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


@CONV_LAYERS.register(name="depthwise_conv2d")
class DepthwiseConv2d(nn.Conv2d):
    """Depthwise Conv2d with 3x3 kernel size, 1 stride, and groups == out_channels."""

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        padding     : Optional[Padding4T] = 0,
        groups      : Optional[int]       = 0,
        bias        : bool                = True,
        padding_mode: str                 = "zeros",
        device      : Any                 = None,
        dtype       : Any                 = None,
        **_
    ):
        super().__init__(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = (3, 3),
            stride       = (1, 1),
            padding      = padding,
            groups       = groups if groups is not None else math.gcd(in_channels, out_channels),
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        

@CONV_LAYERS.register(name="mixed_conv2d")
class MixedConv2d(nn.ModuleDict):
    """Mixed Convolution from the paper `MixConv: Mixed Depthwise
    Convolutional Kernels` (https://arxiv.org/abs/1907.09595)

    Based on MDConv and GroupedConv in MixNet implementation:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Int2T,
        stride      : Int2T              = (1, 1),
        padding     : Optional[Padding4T] = "",
        dilation    : Int2T              = (1, 1),
        depthwise   : bool                = False,
        **kwargs
    ):
        super().__init__()
        kernel_size       = kernel_size
        stride            = to_2tuple(stride)
        dilation          = to_2tuple(dilation)
        num_groups        = len(kernel_size)
        in_splits         = _split_channels(in_channels, num_groups)
        out_splits        = _split_channels(out_channels, num_groups)
        self.in_channels  = sum(in_splits)
        self.out_channels = sum(out_splits)

        for idx, (k, in_ch, out_ch) in enumerate(
            zip(kernel_size, in_splits, out_splits)
        ):
            conv_groups = in_ch if depthwise else 1
            # Use add_module to keep key space clean
            self.add_module(
                str(idx),
                create_conv2d_pad(
                    in_ch, out_ch, k, stride=stride, padding=padding,
                    dilation=dilation, groups=conv_groups, **kwargs
                )
            )
        self.splits = in_splits

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        x_split = torch.split(x, self.splits, 1)
        x_out   = [c(x_split[i]) for i, c in enumerate(self.values())]
        x       = torch.cat(x_out, 1)
        return x


@CONV_LAYERS.register(name="pointwise_conv2d")
class PointwiseConv2d(nn.Conv2d):
    """Pointwise Conv2d with 1x1 kernel size, 1 stride, and groups == 1."""

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        padding     : Optional[Padding4T] = 0,
        bias        : bool                = True,
        padding_mode: str                 = "zeros",
        device      : Any                 = None,
        dtype       : Any                 = None,
        **_
    ):
        super().__init__(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = (1, 1),
            stride       = (1, 1),
            padding      = padding,
            groups       = 1,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
    

@CONV_LAYERS.register(name="scaled_std_conv2d")
class ScaledStdConv2d(nn.Conv2d):
    """Conv2d layer with Scaled Weight Standardization.

    Paper: `Characterizing signal propagation to close the performance gap in
    unnormalized ResNets` - https://arxiv.org/abs/2101.08692

    NOTE: the operations used in this impl differ slightly from the DeepMind
    Haiku impl. Fimpact is minor.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Int2T,
        stride      : Int2T              = 1,
        padding     : Optional[Padding4T] = None,
        dilation    : Int2T              = 1,
        groups      : int                 = 1,
        bias        : bool                = True,
        padding_mode: str                 = "zeros",
        device      : Any                 = None,
        dtype       : Any                 = None,
        gamma       : float               = 1.0,
        eps         : float               = 1e-6,
        gain_init   : float               = 1.0
    ):
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        dilation    = to_2tuple(dilation)
        if padding is None:
            padding = get_padding(kernel_size[0], stride[0], dilation[0])
            
        super().__init__(
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
        self.gain  = nn.Parameter(
            torch.full((self.out_channels, 1, 1, 1), gain_init)
        )
        # gamma * 1 / sqrt(fan-in)
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.eps   = eps

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        weight = F.batch_norm(
            self.weight.reshape(1, self.out_channels, -1), None, None,
            weight   = (self.gain * self.scale).view(-1),
            training = True,
            momentum = 0.0,
            eps      = self.eps
        ).reshape_as(self.weight)
        return F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation,
            self.groups
        )


@CONV_LAYERS.register(name="scaled_std_conv2d_same")
class ScaledStdConv2dSame(nn.Conv2d):
    """Conv2d layer with Scaled Weight Standardization and Tensorflow-like SAME
    padding support

    Paper: `Characterizing signal propagation to close the performance gap in
    unnormalized ResNets` - https://arxiv.org/abs/2101.08692

    NOTE: the operations used in this impl differ slightly from the DeepMind
    Haiku impl. impact is minor.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Int2T,
        stride      : Int2T              = (1, 1),
        padding     : Optional[Padding4T] = "same",
        dilation    : Int2T              = (1, 1),
        groups      : int                 = 1,
        bias        : bool                = True,
        padding_mode: str                 = "zeros",
        device      : Any                 = None,
        dtype       : Any                 = None,
        gamma       : float               = 1.0,
        eps         : float               = 1e-6,
        gain_init   : float               = 1.0
    ):
        padding, is_dynamic = get_padding_value(
            padding, kernel_size, stride=stride, dilation=dilation
        )
        super().__init__(
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
        self.gain = nn.Parameter(
            torch.full((self.out_channels, 1, 1, 1), gain_init)
        )
        self.scale    = gamma * self.weight[0].numel() ** -0.5
        self.same_pad = is_dynamic
        self.eps      = eps

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        if self.same_pad:
            x = pad_same(x, self.kernel_size, self.stride, self.dilation)
        weight = F.batch_norm(
            self.weight.reshape(1, self.out_channels, -1), None, None,
            weight   = (self.gain * self.scale).view(-1),
            training = True,
            momentum = 0.0,
            eps      = self.eps
        ).reshape_as(self.weight)
        return F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation,
            self.groups
        )


@CONV_LAYERS.register(name="separable_conv2d")
class SeparableConv2d(nn.Module):
    """Separable Conv."""

    # MARK: Magic Function

    def __init__(
        self,
        in_channels       : int,
        out_channels      : int,
        dw_kernel_size    : Int2T              = (3, 3),
        stride            : Int2T              = (1, 1),
        padding           : Optional[Padding4T] = "",
        dilation          : Int2T              = (1, 1),
        bias              : bool                = False,
        padding_mode      : str                 = "zeros",
        device            : Any                 = None,
        dtype             : Any                 = None,
        channel_multiplier: float               = 1.0,
        pw_kernel_size    : int                 = 1,
        **_
    ):
        super().__init__()

        self.conv_dw = create_conv2d(
            in_channels  = in_channels,
            out_channels = int(in_channels * channel_multiplier),
            kernel_size  = dw_kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
            depthwise    = True
        )
        self.conv_pw = create_conv2d(
            in_channels  = int(in_channels * channel_multiplier),
            out_channels = out_channels,
            kernel_size  = pw_kernel_size,
            padding      = padding,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )

    # MARK: Properties

    @property
    def in_channels(self):
        return self.conv_dw.in_channels

    @property
    def out_channels(self):
        return self.conv_pw.out_channels

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        return x


@CONV_LAYERS.register(name="std_conv2d")
class StdConv2d(nn.Conv2d):
    """Conv2d with Weight Standardization. Used for BiT ResNet-V2 models.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight
    Standardization` - https://arxiv.org/abs/1903.10520v2
    """

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Int2T,
        stride      : Int2T              = 1,
        padding     : Optional[Padding4T] = None,
        dilation    : Int2T              = 1,
        groups      : int                 = 1,
        bias        : bool                = False,
        padding_mode: str                 = "zeros",
        device      : Any                 = None,
        dtype       : Any                 = None,
        eps         : float               = 1e-6
    ):
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        dilation    = to_2tuple(dilation)
        if padding is None:
            padding = get_padding(kernel_size[0], stride[0], dilation[0])
        super().__init__(
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
        self.eps = eps

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        weight = F.batch_norm(
            self.weight.reshape(1, self.out_channels, -1), None, None,
            training=True, momentum=0.0, eps=self.eps
        ).reshape_as(self.weight)
        return F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation,
            self.groups
        )


@CONV_LAYERS.register(name="std_conv2d_same")
class StdConv2dSame(nn.Conv2d):
    """Conv2d with Weight Standardization. TF compatible SAME padding. Used for
     ViT Hybrid model.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight
    Standardization` - https://arxiv.org/abs/1903.10520v2
    """

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Int2T,
        stride      : Int2T              = (1, 1),
        padding     : Optional[Padding4T] = "same",
        dilation    : Int2T              = (1, 1),
        groups      : int                 = 1,
        bias        : bool                = False,
        padding_mode: str                 = "zeros",
        device      : Any                 = None,
        dtype       : Any                 = None,
        eps         : float               = 1e-6
    ):
        padding, is_dynamic = get_padding_value(
            padding, kernel_size, stride=stride, dilation=dilation
        )
        super().__init__(
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
        self.same_pad = is_dynamic
        self.eps      = eps

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        if self.same_pad:
            x = pad_same(x, self.kernel_size, self.stride, self.dilation)
        weight = F.batch_norm(
            self.weight.reshape(1, self.out_channels, -1), None, None,
            training=True, momentum=0.0, eps=self.eps
        ).reshape_as(self.weight)
        return F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation,
            self.groups
        )


# MARK: - Alias

CondConv          = CondConv2d
Conv3x3           = Conv2d3x3
ConvTf            = Conv2dTf
ConvSame          = Conv2dSame
CrossConv         = CrossConv2d
DepthwiseConv     = DepthwiseConv2d
MixedConv         = MixedConv2d
PointwiseConv     = PointwiseConv2d
ScaledStdConv     = ScaledStdConv2d
ScaledStdConvSame = ScaledStdConv2dSame
SeparableConv     = SeparableConv2d
StdConv           = StdConv2d
StdConvSame       = StdConv2dSame


# MARK: - Register

CONV_LAYERS.register(name="conv",                 module=nn.Conv2d)
CONV_LAYERS.register(name="conv1d",               module=nn.Conv1d)
CONV_LAYERS.register(name="conv2d",               module=nn.Conv2d)
CONV_LAYERS.register(name="conv3d",               module=nn.Conv3d)
CONV_LAYERS.register(name="cond3x3",              module=Conv3x3)
CONV_LAYERS.register(name="cond_conv",            module=CondConv)
CONV_LAYERS.register(name="conv_tf",              module=ConvTf)
CONV_LAYERS.register(name="conv_same",            module=ConvSame)
CONV_LAYERS.register(name="cross_conv",           module=CrossConv)
CONV_LAYERS.register(name="depthwise_conv",       module=DepthwiseConv)
CONV_LAYERS.register(name="mixed_conv",           module=MixedConv)
CONV_LAYERS.register(name="pointwise_conv",       module=PointwiseConv)
CONV_LAYERS.register(name="scaled_std_conv",      module=ScaledStdConv)
CONV_LAYERS.register(name="scaled_std_conv_same", module=ScaledStdConvSame)
CONV_LAYERS.register(name="separable_conv",       module=SeparableConv)
CONV_LAYERS.register(name="std_conv",             module=StdConv)
CONV_LAYERS.register(name="std_conv_same",        module=StdConvSame)


# MARK: - Builder

def create_conv2d_pad(
    in_channels: int, out_channels: int, kernel_size: Int2T, **kwargs
) -> Union[nn.Module, nn.Conv2d]:
    """Create 2D Convolution layer with padding."""
    padding = kwargs.pop("padding", "")
    kwargs.setdefault("bias", False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)

    if is_dynamic:
        return Conv2dSame(in_channels, out_channels, kernel_size, **kwargs)
    else:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs
        )


def create_conv2d(
    in_channels: int, out_channels: int, kernel_size: Int2T, **kwargs
):
    """Select a 2d convolution implementation based on arguments. Creates and
    returns one of `torch.nn.Conv2d`, `Conv2dSame`, `MixedConv2d`, or
    `CondConv2d`. Used extensively by EfficientNet, MobileNetv3 and related
    networks.
    """
    if isinstance(kernel_size, list):
        # MixNet + CondConv combo not supported currently
        if "num_experts" in kwargs:
            raise ValueError
        # MixedConv groups are defined by kernel list
        if "groups" in kwargs:
            raise ValueError
        # We're going to use only lists for defining the MixedConv2d kernel
        # groups, ints, tuples, other iterables will continue to pass to
        # normal conv and specify h, w.
        m = MixedConv2d(in_channels, out_channels, kernel_size, **kwargs)
    else:
        depthwise = kwargs.pop("depthwise", False)
        # for DW out_channels must be multiple of in_channels as must have
        # out_channels % groups == 0
        groups = in_channels if depthwise else kwargs.pop("groups", 1)
        if "num_experts" in kwargs and kwargs["num_experts"] > 0:
            m = CondConv2d(in_channels, out_channels, kernel_size, groups=groups,
                           **kwargs)
        else:
            m = create_conv2d_pad(in_channels, out_channels, kernel_size,
                                  groups=groups, **kwargs)
    return m
