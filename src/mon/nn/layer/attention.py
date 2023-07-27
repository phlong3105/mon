#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements attention blocks."""

from __future__ import annotations

__all__ = [
    "BAM", "CBAM", "ChannelAttention", "ChannelAttentionModule", "GhostSAM",
    "GhostSupervisedAttentionModule", "PixelAttentionModule", "SAM", "SimAM",
    "SimplifiedChannelAttention", "SqueezeExcitation", "SqueezeExciteC",
    "SqueezeExciteL", "SupervisedAttentionModule",
]

from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional
from torchvision.ops import misc as torchvision_misc

from mon.globals import LAYERS
from mon.nn.layer import (
    activation, base, conv, ghost, linear, normalization, pooling,
)
from mon.nn.typing import _size_2_t


# region Channel Attention

@LAYERS.register()
class SqueezeExciteC(base.PassThroughLayerParsingMixin, nn.Module):
    """Squeeze and Excite layer from the paper: "`Squeeze and Excitation
    Networks <https://arxiv.org/pdf/1709.01507.pdf>`__"
    
    This implementation use :class:`torch.nn.Conv2d` layer.
    
    References:
        - https://amaarora.github.io/2020/07/24/SeNet.html#squeeze-and-excitation-block-in-pytorch
        - https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
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
            activation.ReLU(inplace=True),
            conv.Conv2d(
                in_channels  = channels  // reduction_ratio,
                out_channels = channels,
                kernel_size  = 1,
                stride       = 1,
                bias         = bias,
                device       = device,
                dtype        = dtype,
            ),
            activation.Sigmoid()
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


@LAYERS.register()
class SqueezeExciteL(base.PassThroughLayerParsingMixin, nn.Module):
    """Squeeze and Excite layer from the paper: "`Squeeze and Excitation
    Networks <https://arxiv.org/pdf/1709.01507.pdf>`__"
    
    This implementation use :class:`torch.nn.Linear` layer.
    
    References:
        - https://amaarora.github.io/2020/07/24/SeNet.html#squeeze-and-excitation-block-in-pytorch
        - https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
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
            activation.ReLU(inplace=True),
            linear.Linear(
                in_features  = channels  // reduction_ratio,
                out_features = channels,
                bias         = bias,
                device       = device,
                dtype        = dtype,
            ),
            activation.Sigmoid()
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


@LAYERS.register()
class SqueezeExcitation(base.PassThroughLayerParsingMixin, torchvision_misc.SqueezeExcitation):
    pass


@LAYERS.register()
class SimplifiedChannelAttention(base.PassThroughLayerParsingMixin, nn.Module):
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
    

ChannelAttention = SqueezeExciteC
LAYERS.register(module=ChannelAttention)

# endregion


# region Channel-Spatial Attention

@LAYERS.register()
class BAM(base.PassThroughLayerParsingMixin, nn.Module):
    """Bottleneck Attention Module from the paper: "BAM: Bottleneck Attention
    Module".
    
    References:
        https://github.com/Jongchan/attention-module/blob/master/MODELS/bam.py
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
                    module = normalization.BatchNorm1d(
                        num_features=gate_channels[i + 1]
                    )
                )
                self.c_gate.add_module(
                    name   = "gate_c_relu_%d" % (i + 1),
                    module = activation.ReLU(),
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
            y = functional.avg_pool2d(x, x.size(2), stride=x.size(2))
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
                module = normalization.BatchNorm2d(
                    num_features=channels // reduction_ratio
                )
            )
            self.s_gate.add_module(
                name   = "gate_s_relu_reduce0",
                module = activation.ReLU()
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
                    module = normalization.BatchNorm2d(
                        num_features=channels // reduction_ratio
                    )
                )
                self.s_gate.add_module(
                    name   = "gate_s_relu_di_%d" % i,
                    module = activation.ReLU(),
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
        self.sigmoid = activation.Sigmoid()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = 1 + self.sigmoid(self.channel_att(x) * self.spatial_att(x))
        y = y * x
        return x


@LAYERS.register()
class CBAM(base.PassThroughLayerParsingMixin, nn.Module):
    """Convolutional Block Attention Module from the paper: "CBAM: Convolutional
    Block Attention Module".
    
    References:
        https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py
    
    Args:
        channels:
        reduction_ratio: Default: 16.
        pool_type: Pooling layer. One of: ["avg", "lp", "lse", "max"]. Defaults
            to ["avg", "max"].
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
                self.Flatten(),
                linear.Linear(
                    in_features  = channels,
                    out_features = channels  // reduction_ratio,
                ),
                activation.ReLU(),
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
                    avg_pool = functional.avg_pool2d(
                        input       = x,
                        kernel_size = (x.size(2), x.size(3)),
                        stride      = (x.size(2), x.size(3))
                    )
                    channel_att_raw = self.mlp(avg_pool)
                elif pool_type == "max":
                    max_pool = functional.max_pool2d(
                        input       = x,
                        kernel_size = (x.size(2), x.size(3)),
                        stride      = (x.size(2), x.size(3))
                    )
                    channel_att_raw = self.mlp(max_pool)
                elif pool_type == "lp":
                    lp_pool = functional.lp_pool2d(
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
            self.spatial  = conv.Conv2dNormActivation(
                in_channels      = 2,
                out_channels     = 1,
                kernel_size      = kernel_size,
                stride           = 1,
                padding          = (kernel_size - 1) // 2,
                norm_layer       = normalization.BatchNorm2d,
                activation_layer = None,
            )
            self.sigmoid = activation.Sigmoid()
        
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
        pool_types     : list        = ["avg", "max"],
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
        y = self.channel(x)
        if self.spatial is not None:
            y = self.spatial(y)
        return y


@LAYERS.register()
class ChannelAttentionModule(base.PassThroughLayerParsingMixin, nn.Module):
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
            activation.ReLU(inplace=True),
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
            activation.Sigmoid(),
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.avg_pool(x)
        y = self.excitation(y)
        y = x * y
        return y


@LAYERS.register()
class SimAM(base.PassThroughLayerParsingMixin, nn.Module):
    """SimAM adopted from paper: "SimAM: A Simple, Parameter-Free Attention
    Module for Convolutional Neural Networks".
    
    References:
        https://github.com/ZjjConan/SimAM
    """
    
    def __init__(self, e_lambda: float = 1e-4, *args, **kwargs):
        super().__init__()
        self.e_lambda = e_lambda
        self.act      = activation.Sigmoid()
    
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


# region Pixel Attention

@LAYERS.register()
class PixelAttentionModule(base.SameChannelsLayerParsingMixin, nn.Module):
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
            activation.ReLU(inplace=True),
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
        self.act = activation.Sigmoid()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.fc(x)
        y = self.act(y)
        y = torch.mul(x, y)
        return y

# endregion


# region Supervised Attention

@LAYERS.register()
class GhostSupervisedAttentionModule(base.SameChannelsLayerParsingMixin, nn.Module):
    """Ghost Supervised Attention Module."""
    
    def __init__(
        self,
        channels    : int,
        kernel_size : _size_2_t,
        stride      : _size_2_t = 1,
        dilation    : _size_2_t = 1,
        groups      : int       = 1,
        bias        : bool      = True,
        padding_mode: str       = "zeros",
        device      : Any       = None,
        dtype       : Any       = None,
    ):
        super().__init__()
        padding = kernel_size[0] // 2 \
            if isinstance(kernel_size, Sequence) \
            else kernel_size // 2
        
        self.conv1 = ghost.GhostConv2d(
            in_channels  = channels,
            out_channels = channels,
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
        self.conv2 = ghost.GhostConv2d(
            in_channels  = channels,
            out_channels = 3,
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
        self.conv3 = ghost.GhostConv2d(
            in_channels  = 3,
            out_channels = channels,
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
        self.act = activation.Sigmoid()
    
    def forward(self, input: list[torch.Tensor]) -> list[torch.Tensor]:
        """Run forward pass.

        Args:
            input: A list of 2 tensors. The first tensor is the output from
                the previous layer. The second tensor is the current step input.
            
        Returns:
            Supervised attention features.
            Output feature for the next layer.
        """
        if not (isinstance(input, list | tuple) and len(input) == 2):
            raise ValueError(
                f"input must be a list of 2 torch.Tensor, but got {type(input)}."
            )
        fy  = input[0]
        x   = input[1]
        y1  = self.conv1(fy)
        img = self.conv2(fy) + x
        y2  = self.act(self.conv3(img))
        y   = y1 * y2
        y   = y + fy
        return [y, img]


@LAYERS.register()
class SupervisedAttentionModule(base.SameChannelsLayerParsingMixin, nn.Module):
    """Supervised Attention Module."""
    
    def __init__(
        self,
        channels    : int,
        kernel_size : _size_2_t,
        stride      : _size_2_t = 1,
        dilation    : _size_2_t = 1,
        groups      : int       = 1,
        bias        : bool      = True,
        padding_mode: str       = "zeros",
        device      : Any       = None,
        dtype       : Any       = None,
    ):
        super().__init__()
        padding = kernel_size[0] // 2 \
            if isinstance(kernel_size, Sequence) \
            else kernel_size // 2
        
        self.conv1 = conv.Conv2d(
            in_channels  = channels,
            out_channels = channels,
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
        self.conv2 = conv.Conv2d(
            in_channels  = channels,
            out_channels = 3,
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
        self.conv3 = conv.Conv2d(
            in_channels  = 3,
            out_channels = channels,
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
        self.act = activation.Sigmoid()
    
    def forward(self, input: list[torch.Tensor]) -> list[torch.Tensor]:
        """Run forward pass.

        Args:
            input: A list of 2 tensors. The first tensor is the output from
                previous layer. The second tensor is the current step input.
            
        Returns:
            Supervised attention features.
            Output feature for the next layer.
        """
        if not (isinstance(input, list | tuple) and len(input) == 2):
            raise ValueError(
                f"input must be a list of 2 torch.Tensor, but got "
                f"{type(input)}."
            )
        fy  = input[0]
        x   = input[1]
        y1  = self.conv1(fy)
        img = self.conv2(fy) + x
        y2  = self.act(self.conv3(img))
        y   = y1 * y2
        y   = y + fy
        return [y, img]


GhostSAM = GhostSupervisedAttentionModule
SAM      = SupervisedAttentionModule
LAYERS.register(module=GhostSAM)
LAYERS.register(module=SAM)

# endregion
