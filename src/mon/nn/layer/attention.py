#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements attention layers that are used to build deep learning
models.
"""

from __future__ import annotations

__all__ = [
    "AdditiveAttention",
    "BAM",
    "CBAM",
    "ChannelAttention",
    "ChannelAttentionModule",
    "DotProductAttention",
    "ECA",
    "ECA1d",
    "EfficientChannelAttention",
    "EfficientChannelAttention1d",
    "LocationAwareAttention",
    "MultiHeadAttention",
    "MultiHeadLocationAwareAttention",
    "PAM",
    "PixelAttentionModule",
    "RelativeMultiHeadAttention",
    "ScaledDotProductAttention",
    "ScaledDotProductAttention",
    "ShiftedWindowAttention",
    "ShiftedWindowAttentionV2",
    "SimAM",
    "SimplifiedChannelAttention",
    "SqueezeExcitation",
    "SqueezeExciteC",
    "SqueezeExciteL",
    "WindowAttention",
]

import math
from typing import Any

import numpy as np
import torch
from einops import repeat
from torch import nn
from torch.nn import functional as F
from torchvision.ops.misc import SqueezeExcitation

from mon.core import _size_2_t
from mon.nn.layer import (
    activation as act, conv, dropout as drop, flatten, linear, normalization as norm, pooling,
    projection,
)


# region Channel Attention

class EfficientChannelAttention(nn.Module):
    """Constructs an Efficient Channel Attention (ECA) module.
    
    Args:
        channels: Number of channels of the input feature map.
        kernel_size: Adaptive selection of kernel size.
    """
    
    def __init__(self, channels: int, kernel_size: _size_2_t = 3):
        super().__init__()
        self.avg_pool = pooling.AdaptiveAvgPool2d(1)
        self.conv     = conv.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid  = act.Sigmoid()
        self.channel  = channels
        self.k_size   = kernel_size
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)
    
    def flops(self) -> int:
        flops  = 0
        flops += self.channel * self.channel * self.k_size
        return flops


class EfficientChannelAttention1d(nn.Module):
    """Constructs an Efficient Channel Attention (ECA) module.
    
    Args:
        channels: Number of channels of the input feature map.
        kernel_size: Adaptive selection of kernel size.
    """
    
    def __init__(self, channels: int, kernel_size: _size_2_t = 3):
        super().__init__()
        self.avg_pool = pooling.AdaptiveAvgPool1d(1)
        self.conv     = conv.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid  = act.Sigmoid()
        self.channel  = channels
        self.k_size   = kernel_size
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        # b hw c
        # feature descriptor on the global spatial information
        y = self.avg_pool(x.transpose(-1, -2))
        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2))
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)
    
    def flops(self) -> int:
        flops = 0
        flops += self.channel * self.channel * self.k_size
        return flops


class SimplifiedChannelAttention(nn.Module):
    """Simplified channel attention layer proposed in the paper: "`Simple
    Baselines for Image Restoration <https://arxiv.org/pdf/2204.04676.pdf>`__".
    """
    
    def __init__(
        self,
        channels: int,
        bias    : bool = True,
        device  : Any  = None,
        dtype   : Any  = None,
    ):
        super().__init__()
        self.avg_pool   = pooling.AdaptiveAvgPool2d(1)  # squeeze
        self.excitation = conv.Conv2d(
            in_channels  = channels,
            out_channels = channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            bias         = bias,
            device       = device,
            dtype        = dtype,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        # y = self.avg_pool(x)
        # y = self.excitation(y)
        # y = y.view(-1, c, 1, 1)
        # y = x * y
        return y


ECA   = EfficientChannelAttention
ECA1d = EfficientChannelAttention1d

# endregion


# region Channel Attention Module

class BAM(nn.Module):
    """Bottleneck Attention Module from the paper: "BAM: Bottleneck Attention
    Module".
    
    References:
        - `<https://github.com/Jongchan/attention-module/blob/master/MODELS/bam.py>`__
    """
    
    class Flatten(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            x = input
            y = x.view(x.size(0), -1)
            return y
    
    class ChannelAttention(nn.Module):
        def __init__(
            self,
            channels       : int,
            reduction_ratio: int = 16,
            num_layers     : int = 1,
        ):
            super().__init__()
            gate_channels  = [channels]
            gate_channels += [channels // reduction_ratio] * num_layers
            gate_channels += [channels]
            
            self.c_gate = nn.Sequential()
            self.c_gate.add_module("flatten", self.Flatten())
            for i in range(len(gate_channels) - 2):
                self.c_gate.add_module(
                    name   = "gate_c_fc_%d" % i,
                    module = linear.Linear(
                        in_features  = gate_channels[i],
                        out_features = gate_channels[i + 1],
                    )
                )
                self.c_gate.add_module(
                    name   = "gate_c_bn_%d" % (i + 1),
                    module = norm.BatchNorm1d(
                        num_features=gate_channels[i + 1]
                    )
                )
                self.c_gate.add_module(
                    name   = "gate_c_relu_%d" % (i + 1),
                    module = act.ReLU(),
                )
            
            self.c_gate.add_module(
                name   = "gate_c_fc_final",
                module = linear.Linear(
                    in_features  = gate_channels[-2],
                    out_features = gate_channels[-1],
                )
            )
        
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            x = input
            y = F.avg_pool2d(x, x.size(2), stride=x.size(2))
            y = self.c_gate(y)
            y = y.unsqueeze(2).unsqueeze(3).expand_as(x)
            return y
    
    class SpatialAttention(nn.Module):
        def __init__(
            self,
            channels         : int,
            reduction_ratio  : int = 16,
            dilation_conv_num: int = 2,
            dilation_val     : int = 4,
            *args, **kwargs
        ):
            super().__init__()
            self.s_gate = nn.Sequential()
            self.s_gate.add_module(
                name   = "gate_s_conv_reduce0",
                module = conv.Conv2d(
                    in_channels  = channels,
                    out_channels = channels  // reduction_ratio,
                    kernel_size  = 1,
                )
            )
            self.s_gate.add_module(
                name   = "gate_s_bn_reduce0",
                module = norm.BatchNorm2d(
                    num_features=channels // reduction_ratio
                )
            )
            self.s_gate.add_module(
                name   = "gate_s_relu_reduce0",
                module = act.ReLU()
            )
            for i in range(dilation_conv_num):
                self.s_gate.add_module(
                    "gate_s_conv_di_%d" % i,
                    conv.Conv2d(
                        in_channels  = channels      // reduction_ratio,
                        out_channels = channels      // reduction_ratio,
                        kernel_size  = 3,
                        padding      = dilation_val,
                        dilation     = dilation_val,
                    )
                )
                self.s_gate.add_module(
                    name   = "gate_s_bn_di_%d" % i,
                    module = norm.BatchNorm2d(
                        num_features=channels // reduction_ratio
                    )
                )
                self.s_gate.add_module(
                    name   = "gate_s_relu_di_%d" % i,
                    module = act.ReLU(),
                )
            self.s_gate.add_module(
                name   = "gate_s_conv_final",
                module = conv.Conv2d(
                    in_channels  = channels // reduction_ratio,
                    out_channels = 1,
                    kernel_size  = 1,
                )
            )
        
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            x = input
            y = self.s_gate(x).expand_as(x)
            return y
    
    def __init__(
        self,
        channels       : int,
        reduction_ratio: int = 16,
        num_layers     : int = 1,
    ):
        super().__init__()
        self.channel = self.ChannelAttention(
            channels        = channels,
            reduction_ratio = reduction_ratio,
            num_layers      = num_layers,
        )
        self.spatial = self.SpatialAttention(
            channels        = channels,
            reduction_ratio = reduction_ratio,
        )
        self.sigmoid = act.Sigmoid()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = 1 + self.sigmoid(self.channel_att(x) * self.spatial_att(x))
        y = y * x
        return x


class CBAM(nn.Module):
    """Convolutional Block Attention Module from the paper: "CBAM: Convolutional
    Block Attention Module".
    
    References:
        - `<https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py>`__
    
    Args:
        channels:
        reduction_ratio: Default: ``16``.
        pool_types: Pooling layer. One of ``'avg'``, `''lp''`, `''lse''`, or
            `''max''`. Defaults to ``["avg", "max"]``.
    """
    
    class Flatten(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            x = input
            y = x.view(x.size(0), -1)
            return y
    
    # noinspection PyDefaultArgument
    class ChannelAttention(nn.Module):
        def __init__(
            self,
            channels       : int,
            reduction_ratio: int  = 16,
            pool_types     : list = ["avg", "max"],
        ):
            super().__init__()
            self.channels = channels
            self.mlp      = nn.Sequential(
                flatten.Flatten(),
                linear.Linear(
                    in_features  = channels,
                    out_features = channels  // reduction_ratio,
                ),
                act.ReLU(),
                linear.Linear(
                    in_features  = channels  // reduction_ratio,
                    out_features = channels,
                )
            )
            self.pool_types = pool_types
        
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            x = input
            channel_att_sum = None
            channel_att_raw = None
            
            for pool_type in self.pool_types:
                if pool_type == "avg":
                    avg_pool = F.avg_pool2d(
                        input       = x,
                        kernel_size = (x.size(2), x.size(3)),
                        stride      = (x.size(2), x.size(3))
                    )
                    channel_att_raw = self.mlp(avg_pool)
                elif pool_type == "max":
                    max_pool = F.max_pool2d(
                        input       = x,
                        kernel_size = (x.size(2), x.size(3)),
                        stride      = (x.size(2), x.size(3))
                    )
                    channel_att_raw = self.mlp(max_pool)
                elif pool_type == "lp":
                    lp_pool = F.lp_pool2d(
                        input       = x,
                        norm_type   = 2,
                        kernel_size = (x.size(2), x.size(3)),
                        stride      = (x.size(2), x.size(3))
                    )
                    channel_att_raw = self.mlp(lp_pool)
                elif pool_type == "lse":
                    # LSE pool only
                    lse_pool        = pooling.lse_pool2d(x)
                    channel_att_raw = self.mlp(lse_pool)
                
                if channel_att_sum is None:
                    channel_att_sum = channel_att_raw
                else:
                    channel_att_sum = channel_att_sum + channel_att_raw
            
            y = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(
                3
            ).expand_as(x)  # scale
            y = x * y
            return y
    
    class SpatialAttention(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            kernel_size = 7
            self.compress = pooling.ChannelPool()
            self.spatial  = conv.Conv2dNormAct(
                in_channels      = 2,
                out_channels     = 1,
                kernel_size      = kernel_size,
                stride           = 1,
                padding          = (kernel_size - 1) // 2,
                norm_layer       = norm.BatchNorm2d,
                activation_layer = None,
            )
            self.sigmoid = act.Sigmoid()
        
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            x = input
            y = self.compress(x)  # compress
            y = self.spatial(y)  # spatial
            y = self.sigmoid(y)  # scale (broadcasting)
            y = x * y
            return y
    
    def __init__(
        self,
        channels       : int,
        reduction_ratio: int         = 16,
        pool_types     : list[str]   = ["avg", "max"],
        spatial        : bool | None = True,
    ):
        super().__init__()
        self.channel = self.ChannelAttention(
            channels        = channels,
            reduction_ratio = reduction_ratio,
            pool_types      = pool_types,
        )
        self.spatial = self.SpatialAttention() if spatial is True else None
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = x
        y = self.channel(y)
        if self.spatial is not None:
            y = self.spatial(y)
        return y


class ChannelAttentionModule(nn.Module):
    """Channel Attention Module."""
    
    def __init__(
        self,
        channels       : int,
        reduction_ratio: int,
        stride         : _size_2_t       = 1,
        padding        : _size_2_t | str = 0,
        dilation       : _size_2_t       = 1,
        groups         : int             = 1,
        bias           : bool            = True,
        padding_mode   : str             = "zeros",
        device         : Any             = None,
        dtype          : Any             = None,
    ):
        super().__init__()
        self.avg_pool   = pooling.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            conv.Conv2d(
                in_channels  = channels,
                out_channels = channels // reduction_ratio,
                kernel_size  = 1,
                stride       = stride,
                padding      = 0,
                dilation     = dilation,
                groups       = groups,
                bias         = True,
                padding_mode = padding_mode,
                device       = device,
                dtype        = dtype,
            ),
            act.ReLU(inplace=True),
            conv.Conv2d(
                in_channels  = channels // reduction_ratio,
                out_channels = channels,
                kernel_size  = 1,
                padding      = 0,
                dilation     = dilation,
                groups       = groups,
                bias         = True,
                padding_mode = padding_mode,
                device       = device,
                dtype=dtype,
            ),
            act.Sigmoid(),
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.avg_pool(x)
        y = self.excitation(y)
        y = x * y
        return y

# endregion


# region Pixel Attention Module

class PixelAttentionModule(nn.Module):
    """Pixel Attention Module."""
    
    def __init__(
        self,
        channels       : int,
        reduction_ratio: int,
        kernel_size    : _size_2_t,
        stride         : _size_2_t       = 1,
        padding        : _size_2_t | str = 0,
        dilation       : _size_2_t       = 1,
        groups         : int             = 1,
        bias           : bool            = True,
        padding_mode   : str             = "zeros",
        device         : Any             = None,
        dtype          : Any             = None,
    ):
        super().__init__()
        self.fc = nn.Sequential(
            conv.Conv2d(
                in_channels  = channels,
                out_channels = channels // reduction_ratio,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = padding,
                dilation     = dilation,
                groups       = groups,
                bias         = bias,
                padding_mode = padding_mode,
                device       = device,
                dtype        = dtype,
            ),
            act.ReLU(inplace=True),
            conv.Conv2d(
                in_channels  = channels // reduction_ratio,
                out_channels = 1,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = padding,
                dilation     = dilation,
                groups       = groups,
                bias         = bias,
                padding_mode = padding_mode,
                device       = device,
                dtype        = dtype,
            ),
        )
        self.act = act.Sigmoid()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.fc(x)
        y = self.act(y)
        y = torch.mul(x, y)
        return y


PAM = PixelAttentionModule

# endregion


# region Spatial Attention

def shifted_window_attention(
    input                 : torch.Tensor,
    qkv_weight            : torch.Tensor,
    proj_weight           : torch.Tensor,
    relative_position_bias: torch.Tensor,
    window_size           : list[int],
    num_heads             : int,
    shift_size            : list[int],
    attention_dropout     : float               = 0.0,
    dropout               : float               = 0.0,
    qkv_bias              : torch.Tensor | None = None,
    proj_bias             : torch.Tensor | None = None,
    logit_scale           : torch.Tensor | None = None,
    training              : bool                = True,
) -> torch.Tensor:
    """Window-based multi-head self-attention (W-MSA) module with relative
    position bias. It supports both shifted and non-shifted windows.
    
    Args:
        input: An input of shape :math:`[N, C, H, W]`.
        qkv_weight: The weight tensor of query, key, value of shape
            :math:`[in_dim, out_dim]`.
        proj_weight: The weight tensor of projection of shape
            :math:`[in_dim, out_dim]`.
        relative_position_bias: The learned relative position bias added to
            attention.
        window_size: Window size.
        num_heads: Number of attention heads.
        shift_size: Shift size for shifted window attention.
        attention_dropout: Dropout ratio of attention weight. Default: ``0.0``.
        dropout: Dropout ratio of output. Default: ``0.0``.
        qkv_bias: The bias tensor of query, key, value. Default: ``None``.
        proj_bias: The bias tensor of projection. Default: ``None``.
        logit_scale: Logit scale of cosine attention for Swin Transformer V2.
            Default: ``None``.
        training: Training flag used by the dropout parameters. Default: ``True``.
    
    Returns:
        The output tensor after shifted window attention of shape :math:`[N, C, H, W]`.
    """
    b, h, w, c = input.shape
    # Pad feature maps to multiples of window size
    pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
    x     = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_h, pad_w, _ = x.shape
    
    shift_size = shift_size.copy()
    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_h:
        shift_size[0] = 0
    if window_size[1] >= pad_w:
        shift_size[1] = 0
    
    # Cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
    
    # partition windows
    num_windows = (pad_h // window_size[0]) * (pad_w // window_size[1])
    x = x.view(b, pad_h // window_size[0], window_size[0], pad_w // window_size[1], window_size[1], c)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(b * num_windows, window_size[0] * window_size[1], c)  # B*nW, Ws*Ws, C
    
    # Multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.clone()
        length   = qkv_bias.numel() // 3
        qkv_bias[length : 2 * length].zero_()
    qkv     = F.linear(x, qkv_weight, qkv_bias)
    qkv     = qkv.reshape(x.size(0), x.size(1), 3, num_heads, c // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    if logit_scale is not None:
        # Cosine attention
        attn        = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
        attn        = attn * logit_scale
    else:
        q    = q * (c // num_heads) ** -0.5
        attn = q.matmul(k.transpose(-2, -1))
    # Add relative position bias
    attn = attn + relative_position_bias
    
    if sum(shift_size) > 0:
        # Generate attention mask
        attn_mask = x.new_zeros((pad_h, pad_w))
        h_slices  = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        w_slices  = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        count     = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0] : h[1], w[0] : w[1]] = count
                count += 1
        attn_mask = attn_mask.view(pad_h // window_size[0], window_size[0], pad_w // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn      = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn      = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn      = attn.view(-1, num_heads, x.size(1), x.size(1))
    
    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout, training=training)
    
    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), c)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout, training=training)
    
    # Reverse windows
    x = x.view(b, pad_h // window_size[0], pad_w // window_size[1], window_size[0], window_size[1], c)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, pad_h, pad_w, c)
    
    # Reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
    
    # Unpad features
    x = x[:, :h, :w, :].contiguous()
    return x


class ShiftedWindowAttention(nn.Module):
    """See Also :func:`shifted_window_attention`."""
    
    def __init__(
        self,
        channels         : int,
        window_size      : list[int],
        shift_size       : list[int],
        num_heads        : int,
        qkv_bias         : bool  = True,
        proj_bias        : bool  = True,
        attention_dropout: float = 0.0,
        dropout          : float = 0.0,
        *args, **kwargs
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError(f":param:`window_size` and :param:`shift_size` must be of length ``2``.")
        self.channels          = channels
        self.window_size       = window_size
        self.shift_size        = shift_size
        self.num_heads         = num_heads
        self.attention_dropout = attention_dropout
        self.dropout           = dropout
        
        self.qkv  = linear.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = linear.Linear(channels, channels, bias=proj_bias)
        
        self.define_relative_position_bias_table()
        self.define_relative_position_index()
    
    def define_relative_position_bias_table(self):
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def define_relative_position_index(self):
        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords   = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten  = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
    
    def calculate_relative_position_bias(
        self,
        relative_position_bias_table: torch.Tensor,
        relative_position_index     : torch.Tensor,
        window_size                 : list[int],
    ) -> torch.Tensor:
        n = window_size[0] * window_size[1]
        relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
        relative_position_bias = relative_position_bias.view(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        return relative_position_bias
    
    def get_relative_position_bias(self) -> torch.Tensor:
        return self.calculate_relative_position_bias(
            self.relative_position_bias_table,
            self.relative_position_index,
            self.window_size  # type: ignore[arg-type]
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        
        Args:
            input: Tensor of shape :math:`[B, H, W, C]`.
      
        Returns:
            Tensor of shape :math:`[B, H, W, C]`.
        """
        x = input
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            input                  = x,
            qkv_weight             = self.qkv.weight,
            proj_weight            = self.proj.weight,
            relative_position_bias = relative_position_bias,
            window_size            = self.window_size,
            num_heads              = self.num_heads,
            shift_size             = self.shift_size,
            attention_dropout      = self.attention_dropout,
            dropout                = self.dropout,
            qkv_bias               = self.qkv.bias,
            proj_bias              = self.proj.bias,
            training               = self.training,
        )


class ShiftedWindowAttentionV2(ShiftedWindowAttention):
    """See Also: :class:`ShiftedWindowAttention`."""
    
    def __init__(
        self,
        channels         : int,
        window_size      : list[int],
        shift_size       : list[int],
        num_heads        : int,
        qkv_bias         : bool  = True,
        proj_bias        : bool  = True,
        attention_dropout: float = 0.0,
        dropout          : float = 0.0,
    ):
        super().__init__(
            channels          = channels,
            window_size       = window_size,
            shift_size        = shift_size,
            num_heads         = num_heads,
            qkv_bias          = qkv_bias,
            proj_bias         = proj_bias,
            attention_dropout = attention_dropout,
            dropout           = dropout,
        )
        
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        # MLP to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            linear.Linear(2, 512, bias=True),
            act.ReLU(inplace=True),
            linear.Linear(512, num_heads, bias=False)
        )
        if qkv_bias:
            length = self.qkv.bias.numel() // 3
            self.qkv.bias[length : 2 * length].data.zero_()
    
    def define_relative_position_bias_table(self):
        # Get relative_coords_table
        relative_coords_h     = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w     = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w], indexing="ij"))
        relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        
        relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
        relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
        
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / 3.0)
        self.register_buffer("relative_coords_table", relative_coords_table)
    
    def get_relative_position_bias(self) -> torch.Tensor:
        relative_position_bias = self.calculate_relative_position_bias(
            self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads),
            self.relative_position_index,  # type: ignore[arg-type]
            self.window_size,
        )
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        return relative_position_bias
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        
        Args:
            input: Tensor of shape :math:`[B, H, W, C]`.
      
        Returns:
            Tensor of shape :math:`[B, H, W, C]`.
        """
        x = input
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            input                  = x,
            qkv_weight             = self.qkv.weight,
            proj_weight            = self.proj.weight,
            relative_position_bias = relative_position_bias,
            window_size            = self.window_size,
            num_heads              = self.num_heads,
            shift_size             = self.shift_size,
            attention_dropout      = self.attention_dropout,
            dropout                = self.dropout,
            qkv_bias               = self.qkv.bias,
            proj_bias              = self.proj.bias,
            training               = self.training,
        )


class WindowAttention(nn.Module):
    
    def __init__(
        self,
        channels        : int,
        window_size     : list[int],
        num_heads       : int,
        token_projection: str   = "linear",
        qkv_bias        : bool  = True,
        qk_scale        : Any   = None,
        attn_drop       : float = 0.0,
        proj_drop       : float = 0.0,
    ):
        super().__init__()
        self.channels    = channels
        self.window_size = window_size  # Wh, Ww
        self.num_heads   = num_heads
        head_dim         = channels // num_heads
        self.scale       = qk_scale or head_dim ** -0.5
        
        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        
        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.window_size[1])  # [0,...,Ww-1]
        coords   = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten  = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index   = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        
        if token_projection == "conv":
            self.qkv = projection.ConvProjection(channels, num_heads, channels // num_heads, bias=qkv_bias)
        elif token_projection == "linear":
            self.qkv = projection.LinearProjection(channels, num_heads, channels // num_heads, bias=qkv_bias)
        else:
            raise Exception("Projection error!")
        
        self.token_projection = token_projection
        self.attn_drop        = drop.Dropout(attn_drop)
        self.proj             = linear.Linear(channels, channels)
        self.proj_drop        = drop.Dropout(proj_drop)
        self.softmax          = act.Softmax(dim=-1)
    
    def forward(
        self,
        input  : torch.Tensor,
        attn_kv: torch.Tensor | None = None,
        mask   : torch.Tensor | None = None,
    ) -> torch.Tensor:
        x        = input
        b_, n, c = x.shape
        q, k, v  = self.qkv(x, attn_kv)
        q        = q * self.scale
        attn     = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)
        
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW   = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(b_ // nW, nW, self.num_heads, n, n * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n * ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def extra_repr(self) -> str:
        return f"dim={self.channels}, win_size={self.window_size}, num_heads={self.num_heads}"
    
    def flops(self, h, w):
        # Calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        N     = self.window_size[0] * self.window_size[1]
        nW    = h * w / N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        flops += self.qkv.flops(h * w, h * w)
        # attn = (q @ k.transpose(-2, -1))
        flops += nW * self.num_heads * N * (self.channels // self.num_heads) * N
        #  x = (attn @ v)
        flops += nW * self.num_heads * N * N * (self.channels // self.num_heads)
        # x = self.proj(x)
        flops += nW * N * self.channels * self.channels
        # print("W-MSA:{%.2f}" % (flops / 1e9))
        return flops

# endregion


# region Squeeze Excitation

class SqueezeExciteC(nn.Module):
    """Squeeze and Excite layer from the paper: "`Squeeze and Excitation
    Networks <https://arxiv.org/pdf/1709.01507.pdf>`__"
    
    This implementation uses :class:`torch.nn.Conv2d` layer.
    
    References:
        - `<https://amaarora.github.io/2020/07/24/SeNet.html#squeeze-and-excitation-block-in-pytorch>`__
        - `<https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py>`__
    """
    
    def __init__(
        self,
        channels       : int,
        reduction_ratio: int  = 16,
        bias           : bool = False,
        device         : Any  = None,
        dtype          : Any  = None,
    ):
        super().__init__()
        self.avg_pool   = pooling.AdaptiveAvgPool2d(1)  # squeeze
        self.excitation = nn.Sequential(
            conv.Conv2d(
                in_channels  = channels,
                out_channels = channels  // reduction_ratio,
                kernel_size  = 1,
                stride       = 1,
                bias         = bias,
                device       = device,
                dtype        = dtype,
            ),
            act.ReLU(inplace=True),
            conv.Conv2d(
                in_channels  = channels  // reduction_ratio,
                out_channels = channels,
                kernel_size  = 1,
                stride       = 1,
                bias         = bias,
                device       = device,
                dtype        = dtype,
            ),
            act.Sigmoid()
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        # First implementation
        # b, c, _, _ = x.size()
        # y = self.avg_pool(x).view(b, c)
        # y = self.excitation(y).view(b, c, 1, 1)
        # y = x * y.expand_as(x)
        # Second implementation
        y = self.avg_pool(x)
        y = self.excitation(y)
        y = x * y
        return y


class SqueezeExciteL(nn.Module):
    """Squeeze and Excite layer from the paper: "`Squeeze and Excitation
    Networks <https://arxiv.org/pdf/1709.01507.pdf>`__"
    
    This implementation uses :class:`torch.nn.Linear` layer.
    
    References:
        - `<https://amaarora.github.io/2020/07/24/SeNet.html#squeeze-and-excitation-block-in-pytorch>`__
        - `<https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py>`__
    """
    
    def __init__(
        self,
        channels       : int,
        reduction_ratio: int  = 16,
        bias           : bool = False,
        device         : Any  = None,
        dtype          : Any  = None,
    ):
        super().__init__()
        self.avg_pool   = pooling.AdaptiveAvgPool2d(1)  # squeeze
        self.excitation = nn.Sequential(
            linear.Linear(
                in_features  = channels,
                out_features = channels  // reduction_ratio,
                bias         = bias,
                device       = device,
                dtype        = dtype,
            ),
            act.ReLU(inplace=True),
            linear.Linear(
                in_features  = channels  // reduction_ratio,
                out_features = channels,
                bias         = bias,
                device       = device,
                dtype        = dtype,
            ),
            act.Sigmoid()
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        # First implementation
        # b, c, _, _ = x.size()
        # y = self.avg_pool(x).view(b, c)
        # y = self.excitation(y).view(b, c, 1, 1)
        # y = x * y.expand_as(x)
        # Second implementation
        y = self.avg_pool(x)
        y = self.excitation(y)
        y = x * y
        return y


ChannelAttention = SqueezeExciteC

# endregion


# region SimAm

class SimAM(nn.Module):
    """SimAM adopted from paper: "SimAM: A Simple, Parameter-Free Attention
    Module for Convolutional Neural Networks".
    
    References:
        - `<https://github.com/ZjjConan/SimAM>`__
    """
    
    def __init__(self, e_lambda: float = 1e-4, *args, **kwargs):
        super().__init__()
        self.e_lambda = e_lambda
        self.act      = act.Sigmoid()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        # Spatial size
        b, c, h, w = x.size()
        n = w * h - 1
        # Square of (t - u)
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        # d.sum() / n is channel variance
        v = d.sum(dim=[2, 3], keepdim=True) / n
        # E_inv groups all important of x
        e_inv = d / (4 * (v + self.e_lambda)) + 0.5
        # Attended features
        y = x * self.act(e_inv)
        return y

# endregion


# Transformer

class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values.

    Args:
        dim: Dimension of attention.
        mask: Tensor containing indices to be masked.

    Inputs:
        q: Tensor containing projection vector for the decoder.
        k: Tensor containing projection vector for the encoder.
        v: Tensor containing features of the encoded input sequence.
        mask: Tensor containing indices to be masked.

    Returns:
        context: Tensor containing the context vector from attention mechanism.
        attn: Tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(
        self,
        q   : torch.Tensor,
        k   : torch.Tensor,
        v   : torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        score = torch.bmm(q, k.transpose(1, 2)) / self.sqrt_dim
        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))
        attn    = F.softmax(score, -1)
        context = torch.bmm(attn, v)
        return context, attn


class DotProductAttention(nn.Module):
    """Compute the dot products of the query with all values and apply a
    softmax function to obtain the weights on the values.
    """
    
    def __init__(self, dim: int):
        super().__init__()

    def forward(self, q: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, hidden_dim, input_size = q.size(0), q.size(2), q.size(1)
        score   = torch.bmm(q, v.transpose(1, 2))
        attn    = F.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(attn, v)
        return context, attn


class AdditiveAttention(nn.Module):
    """Applies a additive attention (bahdanau) mechanism on the output features
    from the decoder. Additive attention proposed in "Neural Machine Translation
    by Jointly Learning to Align and Translate" paper.

     Args:
         dim: Dimension of hidden state vector.

     Inputs:
        q: Tensor containing projection vector for the decoder.
        k: Tensor containing projection vector for the encoder.
        v: Tensor containing features of the encoded input sequence.

     Returns:
        context: Tensor containing the context vector from attention mechanism.
        attn: Tensor containing the alignment from the encoder outputs.
    """
    
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.key_proj   = nn.Linear(dim, dim, bias=False)
        self.bias       = nn.Parameter(torch.rand(dim).uniform_(-0.1, 0.1))
        self.score_proj = nn.Linear(dim, 1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        score   = self.score_proj(torch.tanh(self.key_proj(k) + self.query_proj(q) + self.bias)).squeeze(-1)
        attn    = F.softmax(score, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), v)
        return context, attn


class LocationAwareAttention(nn.Module):
    """Applies a location-aware attention mechanism on the output features from
    the decoder. Location-aware attention proposed in "Attention-Based Models
    for Speech Recognition" paper.
    
    The location-aware attention mechanism is performing well in speech recognition
    tasks. We refer to the implementation of ClovaCall Attention style.

    Args:
        dim: Dimension of hidden state vector.
        smoothing: Flag indication whether to use smoothing or not.

    Inputs:
        q: Tensor containing the output features from the decoder.
        v: Tensor containing features of the encoded input sequence.
        last_attn: Tensor containing previous timestep`s attention (alignment).

    Returns:
        output: Tensor containing the feature from encoder outputs
        attn: Tensor containing the attention (alignment) from the encoder outputs.

    References:
        - `Attention-Based Models for Speech Recognition <https://arxiv.org/abs/1506.07503>`__
        - `ClovaCall <https://github.com/clovaai/ClovaCall/blob/master/las.pytorch/models/attention.py>`__
    """
    
    def __init__(self, dim: int, smoothing: bool = True):
        super().__init__()
        self.dim        = dim
        self.conv1d     = nn.Conv1d(in_channels=1, out_channels=dim, kernel_size=3, padding=1)
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.value_proj = nn.Linear(dim, dim, bias=False)
        self.score_proj = nn.Linear(dim, 1, bias=True)
        self.bias       = nn.Parameter(torch.rand(dim).uniform_(-0.1, 0.1))
        self.smoothing  = smoothing

    def forward(
        self,
        q        : torch.Tensor,
        v        : torch.Tensor,
        last_attn: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, hidden_dim, seq_len = q.size(0), q.size(2), q.size(1)

        # Initialize previous attention (alignment) to zeros
        if last_attn is None:
            last_attn = v.new_zeros(batch_size, seq_len)

        conv_attn = torch.transpose(self.conv1d(last_attn.unsqueeze(1)), 1, 2)
        score     = self.score_proj(
            torch.tanh(
                  self.query_proj(q.reshape(-1, hidden_dim)).view(batch_size, -1, hidden_dim)
                + self.value_proj(v.reshape(-1, hidden_dim)).view(batch_size, -1, hidden_dim)
                + conv_attn
                + self.bias
            )
        ).squeeze(dim=-1)

        if self.smoothing:
            score = torch.sigmoid(score)
            attn  = torch.div(score, score.sum(dim=-1).unsqueeze(dim=-1))
        else:
            attn  = F.softmax(score, dim=-1)

        context = torch.bmm(attn.unsqueeze(dim=1), v).squeeze(dim=1)  # Bx1xT X BxTxD => Bx1xD => BxD
        return context, attn


class MultiHeadLocationAwareAttention(nn.Module):
    """Applies a multi-headed location-aware attention mechanism on the output
    features from the decoder. Location-aware attention proposed in
    "Attention-Based Models for Speech Recognition" paper.
    
    The location-aware attention mechanism is performing well in speech
    recognition tasks. In the above paper applied a single head, but we applied
    the multi-head concept.

    Args:
        dim: The number of expected features in the output.
        num_heads: The number of heads. (default: ).
        conv_out_channel: The number of out channels in a convolution.

    Inputs:
        q: tensor containing the output features from the decoder.
        v: tensor containing features of the encoded input sequence.
        prev_attn: tensor containing previous timestep`s attention (alignment).

    Returns: output, attn
        output: tensor containing the feature from encoder outputs.
        attn: tensor containing the attention (alignment) from the encoder outputs.

    References:
        - `Attention-Based Models for Speech Recognition <https://arxiv.org/abs/1506.07503>`__
        - `Attention-Based Models for Speech Recognition <https://arxiv.org/abs/1506.07503>`__
    """
    
    def __init__(
        self,
        in_channels     : int,
        num_heads       : int = 8,
        conv_out_channel: int = 10
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads   = num_heads
        self.dim         = int(in_channels / num_heads)
        self.conv1d      = nn.Conv1d(num_heads, conv_out_channel, kernel_size=3, padding=1)
        self.loc_proj    = nn.Linear(conv_out_channel, self.dim, bias=False)
        self.query_proj  = nn.Linear(in_channels, self.dim * num_heads, bias=False)
        self.value_proj  = nn.Linear(in_channels, self.dim * num_heads, bias=False)
        self.score_proj  = nn.Linear(self.dim, 1, bias=True)
        self.bias        = nn.Parameter(torch.rand(self.dim).uniform_(-0.1, 0.1))

    def forward(
        self,
        q        : torch.Tensor,
        v        : torch.Tensor,
        last_attn: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = v.size(0), v.size(1)

        if last_attn is None:
            last_attn = v.new_zeros(batch_size, self.num_heads, seq_len)

        loc_energy = torch.tanh(self.loc_proj(self.conv1d(last_attn).transpose(1, 2)))
        loc_energy = loc_energy.unsqueeze(1).repeat(1, self.num_heads, 1, 1).view(-1, seq_len, self.dim)

        query = self.query_proj(q).view(batch_size, -1, self.num_heads, self.dim).permute(0, 2, 1, 3)
        value = self.value_proj(v).view(batch_size, -1, self.num_heads, self.dim).permute(0, 2, 1, 3)
        query = query.contiguous().view(-1, 1, self.dim)
        value = value.contiguous().view(-1, seq_len, self.dim)
        
        score = self.score_proj(torch.tanh(value + query + loc_energy + self.bias)).squeeze(2)
        attn  = F.softmax(score, dim=1)

        value = value.view(batch_size, seq_len, self.num_heads, self.dim).permute(0, 2, 1, 3)
        value = value.contiguous().view(-1, seq_len, self.dim)

        context = torch.bmm(attn.unsqueeze(1), value).view(batch_size, -1, self.num_heads * self.dim)
        attn    = attn.view(batch_size, self.num_heads, -1)

        return context, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention proposed in "Attention Is All You Need".
   
    Instead of performing a single attention function with d_model-dimensional
    keys, values, and queries, project the queries, keys and values h times with
    different, learned linear projections to d_head dimensions.
   
    These are concatenated and once again projected, resulting in the final values.
    Multi-head attention allows the model to jointly attend to information from
    different representation subspaces at different positions.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_o
        where head_i = Attention(Q · W_q, K · W_k, V · W_v)

    Args:
        in_channels: The dimension of keys / values / queries (default: 512)
        num_heads: The number of attention heads. (default: 8)

    Inputs:
        - q: In a transformer, three different ways:
            Case 1: come from previous decoder layer
            Case 2: come from the input embedding
            Case 3: come from the output embedding (masked)

        - k: In a transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)

        - v: In a transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)

        - mask: tensor containing indices to be masked

    Returns:
        output: Tensor containing the attended output features.
        attn: Tensor containing the attention (alignment) from the encoder outputs.
    """
    
    def __init__(self, in_channels: int = 512, num_heads: int = 8):
        super().__init__()
        assert in_channels % num_heads == 0, "d_model % num_heads should be zero."
        self.dim             = int(in_channels / num_heads)
        self.num_heads       = num_heads
        self.scaled_dot_attn = ScaledDotProductAttention(self.dim)
        self.query_proj      = nn.Linear(in_channels, self.dim * num_heads)
        self.key_proj        = nn.Linear(in_channels, self.dim * num_heads)
        self.value_proj      = nn.Linear(in_channels, self.dim * num_heads)

    def forward(
        self,
        q   : torch.Tensor,
        k   : torch.Tensor,
        v   : torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = v.size(0)

        query = self.query_proj(q).view(batch_size, -1, self.num_heads, self.dim)    # BxQ_LENxNxD
        key   = self.key_proj(k).view(batch_size, -1, self.num_heads, self.dim)      # BxK_LENxNxD
        value = self.value_proj(v).view(batch_size, -1, self.num_heads, self.dim)    # BxV_LENxNxD

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.dim)    # BNxQ_LENxD
        key   = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.dim)      # BNxK_LENxD
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.dim)    # BNxV_LENxD

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # BxNxQ_LENxK_LEN

        context, attn = self.scaled_dot_attn(query, key, value, mask)

        context = context.view(self.num_heads, batch_size, -1, self.dim)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.dim)  # BxTxND

        return context, attn


class RelativeMultiHeadAttention(nn.Module):
    """Multi-head attention with relative positional encoding.
    
    This concept was proposed in the "Transformer-XL: Attentive Language Models
    Beyond a Fixed-Length Context"

    Args:
        in_channels: The dimension of model.
        num_heads: The number of attention heads.
        dropout_p: Probability of dropout.

    Inputs:
        q: Tensor containing query vector.
        k: Tensor containing key vector.
        v: Tensor containing value vector.
        pos_embedding: Positional embedding tensor.
        mask: Tensor containing indices to be masked.

    Returns:
        outputs: Tensor produced by relative multi head attention module.
    """
    
    def __init__(
        self,
        in_channels: int   = 512,
        num_heads  : int   = 16,
        dropout_p  : float = 0.1,
    ):
        super().__init__()
        assert in_channels % num_heads == 0, "in_channels % num_heads should be zero."
        self.in_channels = in_channels
        self.dim         = int(in_channels / num_heads)
        self.num_heads   = num_heads
        self.sqrt_dim    = math.sqrt(in_channels)

        self.query_proj = nn.Linear(in_channels, in_channels)
        self.key_proj   = nn.Linear(in_channels, in_channels)
        self.value_proj = nn.Linear(in_channels, in_channels)
        self.pos_proj   = nn.Linear(in_channels, in_channels, bias=False)

        self.dropout    = nn.Dropout(p=dropout_p)
        self.u_bias     = nn.Parameter(torch.Tensor(self.num_heads, self.dim))
        self.v_bias     = nn.Parameter(torch.Tensor(self.num_heads, self.dim))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)
        
        self.out_proj   = nn.Linear(in_channels, in_channels)

    def forward(
        self,
        q            : torch.Tensor,
        k            : torch.Tensor,
        v            : torch.Tensor,
        pos_embedding: torch.Tensor,
        mask         : torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = v.size(0)

        query = self.query_proj(q).view(batch_size, -1, self.num_heads, self.dim)
        key   = self.key_proj(k).view(batch_size, -1, self.num_heads, self.dim).permute(0, 2, 1, 3)
        value = self.value_proj(v).view(batch_size, -1, self.num_heads, self.dim).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.dim)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score     = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score     = self._compute_relative_positional_encoding(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.in_channels)

        return self.out_proj(context)

    def _compute_relative_positional_encoding(self, pos_score: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros            = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)
        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score        = padded_pos_score[:, :, 1:].view_as(pos_score)
        return pos_score

# endregion
