#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements convolutional layers."""

from __future__ import annotations

__all__ = [
    "FFConv2d",
    "FFConv2dNormActivation",
    "FastFourierConvolution2d",
    "FourierUnit",
    "LearnableSpatialTransform",
    "SpectralTransform",
]

import torch
import torch.nn.functional as F
from kornia.geometry.transform import rotate as krotate
from torch import nn

from mon.coreml.layer import base
from mon.coreml.layer.typing import _size_2_t
from mon.globals import LAYERS


# region Fourier Transform

class FourierUnit(nn.Module):

    def __init__(
        self,
        in_channels          : int,
        out_channels         : int,
        groups               : int        = 1,
        spatial_scale_factor : int | None = None,
        spatial_scale_mode   : str        = "bilinear",
        spectral_pos_encoding: bool       = False,
        use_se               : bool       = False,
        reduction_ratio      : int        = 16,
        bias                 : bool       = False,
        ffc3d                : bool       = False,
        fft_norm             : str        = "ortho",
    ):
        super().__init__()
        self.groups     = groups
        self.conv_layer = base.Conv2d(
            in_channels  = in_channels * 2 + (2 if spectral_pos_encoding else 0),
            out_channels = out_channels * 2,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            groups       = self.groups,
            bias         = False
        )
        self.bn     = base.BatchNorm2d(out_channels * 2)
        self.relu   = base.ReLU(inplace=True)
        
        # Squeeze and excitation block
        self.use_se = use_se
        if use_se:
            self.se = base.SqueezeExciteL(
                channels        = self.conv_layer.in_channels,
                reduction_ratio = reduction_ratio,
                bias            = bias,
            )
        
        self.spatial_scale_factor  = spatial_scale_factor
        self.spatial_scale_mode    = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d                 = ffc3d
        self.fft_norm              = fft_norm

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x     = input
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(
                input         = x,
                scale_factor  = self.spatial_scale_factor,
                mode          = self.spatial_scale_mode,
                align_corners = False,
            )
        
        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted   = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted   = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted   = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted   = ffted.view((batch, -1,) + ffted.size()[3:])
        
        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert   = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor    = torch.linspace(0, 1,  width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted         = torch.cat((coords_vert, coords_hor, ffted), dim=1)
        
        if self.use_se:
            ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(
                input         = output,
                size          = orig_size,
                mode          = self.spatial_scale_mode,
                align_corners = False,
            )

        return output


class SpectralTransform(nn.Module):

    def __init__(
        self,
        in_channels             : int,
        out_channels            : int,
        stride                  : _size_2_t  = 1,
        groups                  : int        = 1,
        enable_lfu              : bool       = True,
        fu_spatial_scale_factor : int | None = None,
        fu_spatial_scale_mode   : str        = "bilinear",
        fu_spectral_pos_encoding: bool       = False,
        fu_use_se               : bool       = False,
        fu_reduction_ratio      : int        = 16,
        fu_bias                 : bool       = False,
        fu_ffc3d                : bool       = False,
        fu_fft_norm             : str        = "ortho",
    ):
        super().__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = base.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = base.Identity()

        self.stride = stride
        self.conv1  = nn.Sequential(
            base.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels // 2,
                kernel_size  = 1,
                groups       = groups,
                bias         = False,
            ),
            base.BatchNorm2d(out_channels // 2),
            base.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            in_channels           = out_channels  // 2,
            out_channels          = out_channels // 2,
            groups                = groups,
            spatial_scale_factor  = fu_spatial_scale_factor,
            spatial_scale_mode    = fu_spatial_scale_mode,
            spectral_pos_encoding = fu_spectral_pos_encoding,
            use_se                = fu_use_se,
            reduction_ratio       = fu_reduction_ratio,
            bias                  = fu_bias,
            ffc3d                 = fu_ffc3d,
            fft_norm              = fu_fft_norm,
        )
        if self.enable_lfu:
            self.lfu = FourierUnit(
                in_channels  = out_channels // 2,
                out_channels = out_channels // 2,
                groups       = groups,
            )
        self.conv2 = base.Conv2d(
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
            split_s    = h // split_no
            xs = torch.cat(torch.split(x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0
            
        y = self.conv2(x + y + xs)
        return y


class LearnableSpatialTransform(nn.Module):
    
    def __init__(
        self,
        impl,
        pad_coef        : float = 0.5,
        angle_init_range: int   = 80,
        train_angle     : bool  = True,
    ):
        super().__init__()
        self.impl  = impl
        self.angle = torch.rand(1) * angle_init_range
        if train_angle:
            self.angle = nn.Parameter(self.angle, requires_grad=True)
        self.pad_coef = pad_coef

    def forward(self, x):
        if torch.is_tensor(x):
            return self.inverse_transform(self.impl(self.transform(x)), x)
        elif isinstance(x, tuple):
            x_trans = tuple(self.transform(elem) for elem in x)
            y_trans = self.impl(x_trans)
            return tuple(self.inverse_transform(elem, orig_x) for elem, orig_x in zip(y_trans, x))
        else:
            raise ValueError(f"Unexpected input type {type(x)}")

    def transform(self, input: torch.Tensor) -> torch.Tensor:
        x                = input
        height, width    = x.shape[2:]
        pad_h, pad_w     = int(height * self.pad_coef), int(width * self.pad_coef)
        x_padded         = F.pad(x, [pad_w, pad_w, pad_h, pad_h], mode="reflect")
        x_padded_rotated = krotate(x_padded, angle=self.angle.to(x_padded))
        return x_padded_rotated

    def inverse_transform(self, y_padded_rotated, orig_x):
        height, width     = orig_x.shape[2:]
        pad_h, pad_w      = int(height * self.pad_coef), int(width * self.pad_coef)
        y_padded          = krotate(y_padded_rotated, angle=-self.angle.to(y_padded_rotated))
        y_height, y_width = y_padded.shape[2:]
        y = y_padded[:, :, pad_h : y_height - pad_h, pad_w : y_width - pad_w]
        return y

# endregion


# region Fast-Fourier Convolution

@LAYERS.register()
class FastFourierConvolution2d(base.ConvLayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        in_channels             : int,
        out_channels            : int,
        kernel_size             : _size_2_t,
        ratio_gin               : int,
        ratio_gout              : int,
        stride                  : _size_2_t  = 1,
        padding                 : _size_2_t  = 0,
        dilation                : _size_2_t  = 1,
        groups                  : int        = 1,
        bias                    : bool       = False,
        padding_type            : str        = "reflect",
        enable_lfu              : bool       = True,
        fu_spatial_scale_factor : int | None = None,
        fu_spatial_scale_mode   : str        = "bilinear",
        fu_spectral_pos_encoding: bool       = False,
        fu_use_se               : bool       = False,
        fu_reduction_ratio      : int        = 16,
        fu_bias                 : bool       = False,
        fu_ffc3d                : bool       = False,
        fu_fft_norm             : str        = "ortho",
        gated                   : bool       = False,
    ):
        super().__init__()
        
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        
        in_channels_global  = int(in_channels * ratio_gin)
        in_channels_local   = in_channels - in_channels_global
        out_channels_global = int(out_channels * ratio_gout)
        out_channels_local  = out_channels - out_channels_global
        # groups_global = 1 if groups == 1 else int(groups * ratio_gout)
        # groups_local  = 1 if groups == 1 else groups - groups_g
        
        self.ratio_gin     = ratio_gin
        self.ratio_gout    = ratio_gout
        self.global_in_num = in_channels_global
        
        self.conv_local2local = base.Conv2d(
            in_channels  = in_channels_local,
            out_channels = out_channels_local,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_type,
        ) if in_channels_local > 0 and out_channels_local > 0 else base.Identity()
        self.conv_local2global = base.Conv2d(
            in_channels  = in_channels_local,
            out_channels = out_channels_global,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_type,
        ) if in_channels_local > 0 and out_channels_local > 0 else base.Identity()
        self.conv_global2local = base.Conv2d(
            in_channels  = in_channels_global,
            out_channels = out_channels_local,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_type,
        ) if in_channels_local > 0 and out_channels_local > 0 else base.Identity()
        self.conv_global2global = SpectralTransform(
            in_channels              = in_channels_global,
            out_channels             = out_channels_global,
            stride                   = stride,
            groups                   = 1 if groups == 1 else groups // 2,
            enable_lfu               = enable_lfu,
            fu_spatial_scale_factor  = fu_spatial_scale_factor,
            fu_spatial_scale_mode    = fu_spatial_scale_mode,
            fu_spectral_pos_encoding = fu_spectral_pos_encoding,
            fu_use_se                = fu_use_se,
            fu_reduction_ratio       = fu_reduction_ratio,
            fu_bias                  = fu_bias,
            fu_ffc3d                 = fu_ffc3d,
            fu_fft_norm              = fu_fft_norm,
        ) if in_channels_local > 0 and out_channels_local > 0 else base.Identity()
        self.gated = gated
        self.gate  = base.Conv2d(
            in_channels  = in_channels,
            out_channels = 2,
            kernel_size  = 1,
        ) if in_channels_global > 0 and out_channels_local > 0 and self.gated else base.Identity()
    
    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = input
        x_local, x_global = x if type(x) is tuple else (x, 0)
        y_local, y_global = 0, 0

        if self.gated:
            total_input_parts = [x_local]
            if torch.is_tensor(x_global):
                total_input_parts.append(x_global)
            total_input = torch.cat(total_input_parts, dim=1)
            gates       = torch.sigmoid(self.gate(total_input))
            global2local_gate, local2global_gate = gates.chunk(2, dim=1)
        else:
            global2local_gate, local2global_gate = 1, 1

        if self.ratio_gout != 1:
            y_local = self.conv_local2local(x_local) \
                      + self.conv_global2local(x_global) * global2local_gate
        if self.ratio_gout != 0:
            y_global = self.conv_local2global(x_local) * local2global_gate \
                       + self.conv_global2global(x_global)
        return y_local, y_global


@LAYERS.register()
class FFConv2dNormActivation(base.ConvLayerParsingMixin, nn.Module):

    def __init__(
        self,
        in_channels             : int,
        out_channels            : int,
        kernel_size             : _size_2_t,
        ratio_gin               : int,
        ratio_gout              : int,
        stride                  : _size_2_t  = 1,
        padding                 : _size_2_t  = 0,
        dilation                : _size_2_t  = 1,
        groups                  : int        = 1,
        bias                    : bool       = False,
        norm_layer              : nn.Module  = nn.BatchNorm2d,
        activation_layer        : nn.Module  = nn.Identity,
        padding_type            : str        = "reflect",
        enable_lfu              : bool       = True,
        fu_spatial_scale_factor : int | None = None,
        fu_spatial_scale_mode   : str        = "bilinear",
        fu_spectral_pos_encoding: bool       = False,
        fu_use_se               : bool       = False,
        fu_reduction_ratio      : int        = 16,
        fu_bias                 : bool       = False,
        fu_ffc3d                : bool       = False,
        fu_fft_norm             : str        = "ortho",
        gated                   : bool       = False,
    ):
        super().__init__()
        self.ffc = FFConv2d(
            in_channels              = in_channels,
            out_channels             = out_channels,
            kernel_size              = kernel_size,
            ratio_gin                = ratio_gin,
            ratio_gout               = ratio_gout,
            stride                   = stride,
            padding                  = padding,
            dilation                 = dilation,
            groups                   = groups,
            bias                     = bias,
            padding_type             = padding_type,
            enable_lfu               = enable_lfu,
            fu_spatial_scale_factor  = fu_spatial_scale_factor,
            fu_spatial_scale_mode    = fu_spatial_scale_mode,
            fu_spectral_pos_encoding = fu_spectral_pos_encoding,
            fu_use_se                = fu_use_se,
            fu_reduction_ratio       = fu_reduction_ratio,
            fu_bias                  = fu_bias,
            fu_ffc3d                 = fu_ffc3d,
            fu_fft_norm              = fu_fft_norm,
            gated                    = gated,
        )
        local_norm      = nn.Identity if ratio_gout == 1 else norm_layer
        global_norm     = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_local   = local_norm(out_channels - global_channels)
        self.bn_global  = global_norm(global_channels)
        act_local       = nn.Identity if ratio_gout == 1 else activation_layer
        act_global      = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_local  = act_local(inplace=True)
        self.act_global = act_global(inplace=True)
    
    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = input
        x_local, x_global = self.ffc(x)
        y_local  = self.act_local(self.bn_local(x_local))
        y_global = self.act_global(self.bn_global(x_global))
        return y_local, y_global


FFConv2d = FastFourierConvolution2d
LAYERS.register(module=FFConv2d)

# endregion
