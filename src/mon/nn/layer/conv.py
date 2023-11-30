#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements convolutional layers."""

from __future__ import annotations

__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv2dBn",
    "Conv2dNormAct",
    "Conv2dSame",
    "Conv2dTF",
    "Conv3d",
    "Conv3dNormAct",
    "ConvNormAct",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "DRConv2d",
    "DSConv2d",
    "DSConv2dReLU",
    "DepthwiseSeparableConv2d",
    "DepthwiseSeparableConv2dReLU",
    "LazyConv1d",
    "LazyConv2d",
    "LazyConv3d",
    "LazyConvTranspose1d",
    "LazyConvTranspose2d",
    "LazyConvTranspose3d",
    "conv2d_same",
]

from typing import Any

import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F
from torchvision.ops import misc

from mon.core import math
from mon.globals import LAYERS
from mon.nn.layer import (
    activation, base, normalization, padding as pad, pooling as pool,
)
from mon.nn.typing import _size_2_t, _size_any_t


# region Convolution

def conv2d_same(
    input   : torch.Tensor,
    weight  : torch.Tensor,
    bias    : torch.Tensor | None = None,
    stride  : _size_any_t         = 1,
    padding : _size_any_t | str   = 0,
    dilation: _size_any_t         = 1,
    groups  : int                 = 1,
):
    """Functional interface for Same Padding Convolution 2D."""
    x = input
    y = pad.pad_same(
        input       = x,
        kernel_size = weight.shape[-2: ],
        stride      = stride,
        dilation    = dilation
    )
    y = F.conv2d(
        input    = y,
        weight   = weight,
        bias     = bias,
        stride   = stride,
        padding  = padding,
        dilation = dilation,
        groups   = groups
    )
    return y


@LAYERS.register()
class Conv1d(base.ConvLayerParsingMixin, nn.Conv1d):
    pass


@LAYERS.register()
class Conv2d(base.ConvLayerParsingMixin, nn.Conv2d):
    pass


@LAYERS.register()
class Conv2dBn(base.ConvLayerParsingMixin, nn.Module):
    """Conv2d + BatchNorm."""
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        stride      : _size_2_t       = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t       = 1,
        groups      : int             = 1,
        bias        : bool            = False,
        padding_mode: str             = "zeros",
        device      : Any             = None,
        dtype       : Any             = None,
        bn          : bool | None     = True,
        eps         : float           = 1e-5,
        momentum    : float           = 0.01,
        affine      : bool            = True,
    ):
        super().__init__()
        self.conv = Conv2d(
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
        self.bn = normalization.BatchNorm2d(
            num_features = out_channels,
            eps          = eps,
            momentum     = momentum,
            affine       = affine,
        ) if bn is True else None
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.conv(x)
        if self.bn is not None:
            y = self.bn(y)
        return y


@LAYERS.register()
class Conv2dSame(base.ConvLayerParsingMixin, nn.Conv2d):
    """TensorFlow like ``SAME`` convolution wrapper for 2D convolutions."""
    
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
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = conv2d_same(
            input    = x,
            weight   = self.weight,
            bias     = self.bias,
            stride   = self.stride,
            padding  = self.padding,
            dilation = self.dilation,
            groups   = self.groups
        )
        return y


@LAYERS.register()
class Conv2dTF(base.ConvLayerParsingMixin, nn.Conv2d):
    """Implementation of 2D convolution in TensorFlow with :param:`padding` as
    ``'same'``, which applies padding to input (if needed) so that input image
    gets fully covered by filter and stride you specified. For stride of ``1``,
    this will ensure that the output image size is the same as input. For stride
    of ``2``, output dimensions will be half, for example.
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
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        img_h, img_w = x.size()[-2:]
        kernel_h, kernel_w = self.weight.size()[-2:]
        stride_h, stride_w = self.stride
        output_h = math.ceil(img_h / stride_h)
        output_w = math.ceil(img_w / stride_w)
        pad_h = max((output_h - 1) * self.stride[0] + (kernel_h - 1) * self.dilation[0] + 1 - img_h, 0)
        pad_w = max((output_w - 1) * self.stride[1] + (kernel_w - 1) * self.dilation[1] + 1 - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                input = x,
                pad   = [pad_w // 2, pad_w - pad_w // 2,
                         pad_h // 2, pad_h - pad_h // 2]
            )
        y = F.conv2d(
            input    = x,
            weight   = self.weight,
            bias     = self.bias,
            stride   = self.stride,
            padding  = self.padding,
            dilation = self.dilation,
            groups   = self.groups
        )
        return y


@LAYERS.register()
class Conv3d(base.ConvLayerParsingMixin, nn.Conv3d):
    pass


@LAYERS.register()
class ConvNormAct(base.ConvLayerParsingMixin, misc.ConvNormActivation):
    pass


@LAYERS.register()
class Conv2dNormAct(base.ConvLayerParsingMixin, misc.Conv2dNormActivation):
    pass


@LAYERS.register()
class Conv3dNormAct(base.ConvLayerParsingMixin, misc.Conv3dNormActivation):
    pass


@LAYERS.register()
class LazyConv1d(base.ConvLayerParsingMixin, nn.LazyConv1d):
    pass


@LAYERS.register()
class LazyConv2d(base.ConvLayerParsingMixin, nn.LazyConv2d):
    pass


@LAYERS.register()
class LazyConv3d(base.ConvLayerParsingMixin, nn.LazyConv3d):
    pass


LAYERS.register(name="ConvNormAct",   module=ConvNormAct)
LAYERS.register(name="Conv2dNormAct", module=Conv2dNormAct)
LAYERS.register(name="Conv3dNormAct", module=Conv3dNormAct)

# endregion


# region Depthwise Separable Convolution

@LAYERS.register()
class DepthwiseSeparableConv2d(base.ConvLayerParsingMixin, nn.Module):
    """Depthwise Separable Conv2d."""
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        dw_stride   : _size_2_t       = 1,
        dw_padding  : _size_2_t | str = 0,
        pw_stride   : _size_2_t       = 1,
        pw_padding  : _size_2_t | str = 0,
        dilation    : _size_2_t       = 1,
        groups      : int             = 1,
        bias        : bool            = True,
        padding_mode: str             = "zeros",
        device      : Any             = None,
        dtype       : Any             = None,
    ):
        super().__init__()
        self.dw_conv = Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels,
            kernel_size  = kernel_size,
            stride       = dw_stride,
            padding      = dw_padding,
            dilation     = dilation,
            groups       = in_channels,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.pw_conv = Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = pw_stride,
            padding      = pw_padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.dw_conv(x)
        y = self.pw_conv(y)
        return y


@LAYERS.register()
class DepthwiseSeparableConv2dReLU(base.ConvLayerParsingMixin, nn.Module):
    """Depthwise Separable Conv2d ReLU."""
    
    def __init__(
        self,
        in_channels   : int,
        out_channels  : int,
        kernel_size   : _size_2_t,
        dw_stride     : _size_2_t       = 1,
        dw_padding    : _size_2_t | str = 0,
        pw_stride     : _size_2_t       = 1,
        pw_padding    : _size_2_t | str = 0,
        dilation      : _size_2_t       = 1,
        groups        : int             = 1,
        bias          : bool            = True,
        padding_mode  : str             = "zeros",
        device        : Any             = None,
        dtype         : Any             = None,
    ):
        super().__init__()
        self.ds_conv = DepthwiseSeparableConv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            dw_stride    = dw_stride,
            dw_padding   = dw_padding,
            pw_stride    = pw_stride,
            pw_padding   = pw_padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.act = activation.ReLU(inplace=True)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.ds_conv(x)
        y = self.act(y)
        return y


DSConv2d     = DepthwiseSeparableConv2d
DSConv2dReLU = DepthwiseSeparableConv2dReLU
LAYERS.register(name="DSConv2d",     module=DSConv2d)
LAYERS.register(name="DSConv2dReLU", module=DSConv2dReLU)

# endregion


# region Dynamic Region-Aware Convolution

class DRConv2d(base.ConvLayerParsingMixin, nn.Module):
    """`Dynamic Region-Aware Convolution <https://arxiv.org/abs/2003.12243>`__
    
    References:
        `<https://github.com/shallowtoil/DRConv-PyTorch/tree/master>`__
    """
    
    class asign_index(torch.autograd.Function):
        
        @staticmethod
        def forward(ctx, kernel, guide_feature):
            ctx.save_for_backward(kernel, guide_feature)
            guide_mask = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True), 1).unsqueeze(2) # B x 3 x 1 x 25 x 25
            return torch.sum(kernel * guide_mask, dim=1)
        
        @staticmethod
        def backward(ctx, grad_output):
            kernel, guide_feature = ctx.saved_tensors
            guide_mask  = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True), 1).unsqueeze(2) # B x 3 x 1 x 25 x 25
            grad_kernel = grad_output.clone().unsqueeze(1) * guide_mask # B x 3 x 256 x 25 x 25
            grad_guide  = grad_output.clone().unsqueeze(1) * kernel     # B x 3 x 256 x 25 x 25
            grad_guide  = grad_guide.sum(dim=2)            # B x 3 x 25 x 25
            softmax     = F.softmax(guide_feature, 1) # B x 3 x 25 x 25
            grad_guide  = softmax * (grad_guide - (softmax * grad_guide).sum(dim=1, keepdim=True)) # B x 3 x 25 x 25
            return grad_kernel, grad_guide
    
    def xcorr_slow(x, kernel, kwargs):
        """For loop to calculate cross correlation."""
        batch = x.size()[0]
        out   = []
        for i in range(batch):
            px = x[i]
            pk = kernel[i]
            px = px.view(1, px.size()[0], px.size()[1], px.size()[2])
            pk = pk.view(-1, px.size()[1], pk.size()[1], pk.size()[2])
            po = F.conv2d(px, pk, **kwargs)
            out.append(po)
        out = torch.cat(out, 0)
        return out
    
    
    def xcorr_fast(x, kernel, kwargs):
        """Group conv2d to calculate cross correlation."""
        batch = kernel.size()[0]
        pk    = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
        px    = x.view(1, -1, x.size()[2], x.size()[3])
        po    = F.conv2d(px, pk, **kwargs, groups=batch)
        po    = po.view(batch, -1, po.size()[2], po.size()[3])
        return po
    
    
    class Corr(Function):
        
        @staticmethod
        def symbolic(g, x, kernel, groups):
            return g.op("Corr", x, kernel, groups_i=groups)
    
        @staticmethod
        def forward(self, x, kernel, groups, kwargs):
            """Group conv2d to calculate cross correlation."""
            batch   = x.size(0)
            channel = x.size(1)
            x       = x.view(1, -1, x.size(2), x.size(3))
            kernel  = kernel.view(-1, channel // groups, kernel.size(2), kernel.size(3))
            out     = F.conv2d(x, kernel, **kwargs, groups=groups * batch)
            out     = out.view(batch, -1, out.size(2), out.size(3))
            return out
    
    
    class Correlation(nn.Module):
        use_slow = True
        
        def __init__(self, use_slow=None):
            super().__init__()
            if use_slow is not None:
                self.use_slow = use_slow
            else:
                self.use_slow = Correlation.use_slow
    
        def extra_repr(self):
            return "xcorr_slow" if self.use_slow else "xcorr_fast"
    
        def forward(self, input: torch.Tensor, kernel, **kwargs) -> torch.Tensor:
            x = input
            if self.training:
                if self.use_slow:
                    return DRConv2d.xcorr_slow(x, kernel, kwargs)
                else:
                    return DRConv2d.xcorr_fast(x, kernel, kwargs)
            else:
                return Corr.apply(x, kernel, 1, kwargs)
    
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        region_num  : int = 8,
        *args, **kwargs,
    ):
        super().__init__()
        self.region_num  = region_num
        self.conv_kernel = nn.Sequential(
            pool.AdaptiveAvgPool2d((kernel_size, kernel_size)),
            Conv2d(
                in_channels  = in_channels,
                out_channels = region_num * region_num,
                kernel_size  = 1,
            ),
            activation.Sigmoid(),
            Conv2d(
                in_channels  = region_num * region_num,
                out_channels = region_num * in_channels * out_channels,
                kernel_size  = 1,
                groups       = region_num,
            )
        )
        self.conv_guide = Conv2d(
            in_channels  = in_channels,
            out_channels = region_num,
            kernel_size  = kernel_size,
            **kwargs
        )
        self.corr        = DRConv2d.Correlation(use_slow=False)
        self.asign_index = DRConv2d.asign_index.apply
        self.kwargs      = kwargs
    
    def forward(
        self,
        input      : torch.Tensor,
        guide_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kernel        = self.conv_kernel(input)
        kernel        = kernel.view(kernel.size(0), -1, kernel.size(2), kernel.size(3)) # B x (r*in*out) x W X H
        output        = self.corr(input, kernel, **self.kwargs)                         # B x (r*out) x W x H
        output        = output.view(output.size(0), self.region_num, -1, output.size(2), output.size(3)) # B x r x out x W x H
        guide_feature = self.conv_guide(input)
        output        = self.asign_index(output, guide_feature)
        return output

# endregion


# region Transposed Convolution

class ConvTranspose1d(base.ConvLayerParsingMixin, nn.ConvTranspose1d):
    pass


class ConvTranspose2d(base.ConvLayerParsingMixin, nn.ConvTranspose2d):
    pass


class ConvTranspose3d(base.ConvLayerParsingMixin, nn.ConvTranspose3d):
    pass


class LazyConvTranspose1d(base.ConvLayerParsingMixin, nn.LazyConvTranspose1d):
    pass


class LazyConvTranspose2d(base.ConvLayerParsingMixin, nn.LazyConvTranspose2d):
    pass


class LazyConvTranspose3d(base.ConvLayerParsingMixin, nn.LazyConvTranspose3d):
    pass

# endregion
