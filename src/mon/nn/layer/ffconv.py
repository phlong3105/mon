#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Fast-Fourier Convolution layers."""

from __future__ import annotations

__all__ = [
    "FFConv2d", "FFConv2dNormAct", "FFConv2dSE", "FastFourierConv2d",
    "FastFourierConv2dNormActivation", "FastFourierConv2dSE", "FourierUnit",
    "FourierUnit2d", "FourierUnit3d", "SpectralTransform2d",
]

from typing import Any

import torch
from torch import nn

from mon.globals import LAYERS
from mon.nn.layer import activation, base, conv, linear, normalization, pooling
from mon.nn.typing import _size_2_t


# region Fourier Transform

class FourierUnit(nn.Module):
    """Fourier transform unit proposed in the paper: "`Fast Fourier Convolution
    <https://github.com/pkumivision/FFC>`__".
    
    Args:
        ffc3d: For Fast Fourier Convolution3D.
        fft_norm: Normalization mode. For the backward transform
            (:func:`~torch.fft.irfft`), these correspond to:
            
            - ``'forward'``  - no normalization
            - ``'backward'`` - normalize by ``1/n``
            - ``'ortho'``    - normalize by ``1/sqrt(n)`` (making the real IFFT orthonormal)
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        groups      : int        = 1,
        ffc3d       : bool       = False,
        fft_norm    : str | None = "ortho",
    ):
        # bn_layer not used
        super().__init__()
        self.groups     = groups
        self.ffc3d      = ffc3d
        self.fft_norm   = fft_norm
        self.conv_layer = conv.Conv2d(
            in_channels  = in_channels * 2,
            out_channels = out_channels * 2,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            groups       = self.groups,
            bias         = False,
        )
        self.bn   = normalization.BatchNorm2d(out_channels * 2)
        self.relu = activation.ReLU(inplace=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x          = input
        b, c, h, w = x.size()
        r_size     = x.size()
        fft_dim    = (-3, -2, -1) if self.ffc3d else (-2, -1)
        
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)  # (b, c, h, w/2+1, 2)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (b, c, 2, h, w/2+1)
        ffted = ffted.view((b, -1,) + ffted.size()[3:])
        
        ffted = self.conv_layer(ffted)  # (b, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))
        
        ffted = ffted.view((b, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()  # (b, c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        
        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        y = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
        return y


class FourierUnit2d(FourierUnit):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        groups      : int        = 1,
        fft_norm    : str | None = "ortho",
    ):
        super().__init__(
            in_channels  = in_channels,
            out_channels = out_channels,
            groups       = groups,
            ffc3d        = False,
            fft_norm     = fft_norm,
        )
    

class FourierUnit3d(FourierUnit):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        groups      : int        = 1,
        fft_norm    : str | None = "ortho",
    ):
        super().__init__(
            in_channels  = in_channels,
            out_channels = out_channels,
            groups       = groups,
            ffc3d        = True,
            fft_norm     = fft_norm,
        )
    

class SpectralTransform2d(nn.Module):
    """Spectral transform unit proposed in the paper: "`Fast Fourier Convolution
    <https://github.com/pkumivision/FFC>`__".
    
    Args:
        fft_norm: Normalization mode. For the backward transform
            (:func:`~torch.fft.irfft`), these correspond to:
            
            - ``'forward'``  - no normalization
            - ``'backward'`` - normalize by ``1/n``
            - ``'ortho'``    - normalize by ``1/sqrt(n)`` (making the real IFFT orthonormal)
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : _size_2_t  = 1,
        groups      : int        = 1,
        enable_lfu  : bool       = True,
        fft_norm    : str | None = "ortho",
    ):
        super().__init__()
        
        # bn_layer not used
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = pooling.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = linear.Identity()

        self.stride = stride
        self.conv1  = nn.Sequential(
            conv.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels // 2,
                kernel_size  = 1,
                groups       = groups,
                bias         = False
            ),
            normalization.BatchNorm2d(out_channels // 2),
            activation.ReLU(inplace=True)
        )
        self.fu = FourierUnit2d(
            in_channels  = out_channels // 2,
            out_channels = out_channels // 2,
            groups       = groups,
            fft_norm     = fft_norm,
        )
        if self.enable_lfu:
            self.lfu = FourierUnit2d(
                in_channels  = out_channels // 2,
                out_channels = out_channels // 2,
                groups       = groups,
                fft_norm     = fft_norm,
            )
        self.conv2 = conv.Conv2d(
            in_channels  = out_channels // 2,
            out_channels = out_channels,
            kernel_size  = 1,
            groups       = groups,
            bias         = False,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = self.downsample(x)
        x = self.conv1(x)
        y = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no   = 2
            split_s_h  = h // split_no
            split_s_w  = w // split_no
            xs = torch.cat(torch.split(x[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        y = self.conv2(x + y + xs)
        return y

# endregion


# region Fast-Fourier Convolution

@LAYERS.register()
class FastFourierConv2d(base.ConvLayerParsingMixin, nn.Module):
    """Fast Fourier convolution proposed in the paper: "`Fast Fourier
    Convolution <https://github.com/pkumivision/FFC>`__".
    
    Args:
        fft_norm: Normalization mode. For the backward transform
            (:func:`~torch.fft.irfft`), these correspond to:
            
            - ``'forward'``  - no normalization
            - ``'backward'`` - normalize by ``1/n``
            - ``'ortho'``    - normalize by ``1/sqrt(n)`` (making the real IFFT orthonormal)
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        ratio_gin   : float,
        ratio_gout  : float,
        stride      : _size_2_t  = 1,
        padding     : _size_2_t  = 0,
        dilation    : _size_2_t  = 1,
        groups      : int        = 1,
        bias        : bool       = False,
        padding_mode: str        = "zeros",
        enable_lfu  : bool       = True,
        fft_norm    : str | None = "ortho",
    ):
        super().__init__()
        
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        
        in_cg  = int(in_channels * ratio_gin)
        in_cl  = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        # groups_global = 1 if groups == 1 else int(groups * ratio_gout)
        # groups_local  = 1 if groups == 1 else groups - groups_g
        
        self.ratio_gin  = ratio_gin
        self.ratio_gout = ratio_gout
        
        self.conv_l2l   = conv.Conv2d(
            in_channels  = in_cl,
            out_channels = out_cl,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
        ) if in_cl > 0 and out_cl > 0 else linear.Identity()
        self.conv_l2g   = conv.Conv2d(
            in_channels  = in_cl,
            out_channels = out_cg,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
        ) if in_cl > 0 and out_cg > 0 else linear.Identity()
        self.conv_g2l   = conv.Conv2d(
            in_channels  = in_cg,
            out_channels = out_cl,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
        ) if in_cg > 0 and out_cl > 0 else linear.Identity()
        self.conv_g2g   = SpectralTransform2d(
            in_channels  = in_cg,
            out_channels = out_cg,
            stride       = stride,
            groups       = 1 if groups == 1 else groups // 2,
            enable_lfu   = enable_lfu,
            fft_norm     = fft_norm,
        ) if in_cg > 0 and out_cg > 0 else linear.Identity()
    
    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x        = input
        x_l, x_g = x if isinstance(x, tuple | list) else (x, 0)
        y_l, y_g = 0, 0

        if self.ratio_gout != 1:
            y_l = self.conv_l2l(x_l) + self.conv_g2l(x_g)
        if self.ratio_gout != 0:
            y_g = self.conv_l2g(x_l) + self.conv_g2g(x_g)

        return y_l, y_g


@LAYERS.register()
class FastFourierConv2dNormActivation(base.ConvLayerParsingMixin, nn.Module):
    """Fast Fourier convolution + normalization + activation proposed in the
    paper: "`Fast Fourier Convolution <https://github.com/pkumivision/FFC>`__".
    
    Notes: Mimicking the naming of `torchvision.ops.misc.Conv2dNormActivation`.
    
    Args:
        fft_norm: Normalization mode. For the backward transform
            (:func:`~torch.fft.irfft`), these correspond to:
            
            - ``'forward'``  - no normalization
            - ``'backward'`` - normalize by ``1/n``
            - ``'ortho'``    - normalize by ``1/sqrt(n)`` (making the real IFFT orthonormal)
    """
    
    def __init__(
        self,
        in_channels  : int,
        out_channels : int,
        kernel_size  : _size_2_t,
        ratio_gin    : float,
        ratio_gout   : float,
        stride       : _size_2_t  = 1,
        padding      : _size_2_t  = 0,
        dilation     : _size_2_t  = 1,
        groups       : int        = 1,
        bias         : bool       = False,
        padding_mode : str        = "zeros",
        norm_layer   : Any        = normalization.BatchNorm2d,
        act_layer    : Any        = linear.Identity,
        enable_lfu   : bool       = True,
        fft_norm     : str | None = "ortho",
    ):
        super().__init__()
        self.ffc = FFConv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            ratio_gin    = ratio_gin,
            ratio_gout   = ratio_gout,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            enable_lfu   = enable_lfu,
            fft_norm     = fft_norm,
        )
        self.norm_l = linear.Identity() if ratio_gout == 1 else norm_layer(int(out_channels * (1 - ratio_gout)))
        self.norm_g = linear.Identity() if ratio_gout == 0 else norm_layer(int(out_channels * ratio_gout))
        self.act_l  = linear.Identity() if ratio_gout == 1 else act_layer(inplace=True)
        self.act_g  = linear.Identity() if ratio_gout == 0 else act_layer(inplace=True)
    
    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x        = input
        y_l, y_g = self.ffc(x)
        y_l      = self.act_l(self.norm_l(y_l))
        y_g      = self.act_g(self.norm_g(y_g))
        return y_l, y_g


@LAYERS.register()
class FastFourierConv2dSE(base.ConvLayerParsingMixin, nn.Module):
    """Squeeze and Excitation block for Fast Fourier convolution proposed in the
    paper: "`Fast Fourier Convolution <https://github.com/pkumivision/FFC>`__".
    """
    
    def __init__(
        self,
        channels: int,
        ratio_g : float,
    ):
        super().__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r     = 16
        self.avgpool  = pooling.AdaptiveAvgPool2d((1, 1))
        self.conv1    = conv.Conv2d(
            in_channels  = channels,
            out_channels = channels // r,
            kernel_size  = 1,
            bias         = True,
        )
        self.relu1    = activation.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else conv.Conv2d(
            in_channels  = channels // r,
            out_channels = in_cl,
            kernel_size  = 1,
            bias         = True,
        )
        self.conv_a2g = None if in_cg == 0 else conv.Conv2d(
            in_channels  = channels // r,
            out_channels = in_cg,
            kernel_size  = 1,
            bias         = True,
        )
        self.sigmoid  = activation.Sigmoid()
    
    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x          = input
        x          = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x   = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x   = self.avgpool(x)
        x   = self.relu1(self.conv1(x))
        y_l = 0 if self.conv_a2l is None else id_l * self.sigmoid(self.conv_a2l(x))
        y_g = 0 if self.conv_a2g is None else id_g * self.sigmoid(self.conv_a2g(x))
        return y_l, y_g


FFConv2d        = FastFourierConv2d
FFConv2dNormAct = FastFourierConv2dNormActivation
FFConv2dSE      = FastFourierConv2dSE
LAYERS.register(name="FFConv2d",        module=FFConv2d)
LAYERS.register(name="FFConv2dNormAct", module=FFConv2dNormAct)
LAYERS.register(name="FFConv2dSE",      module=FFConv2dSE)

# endregion
