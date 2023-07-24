#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements layers and blocks especially used for MobileOne
models.
"""

from __future__ import annotations

__all__ = [
    "MobileOneConv2d", "MobileOneStage",
]

from typing import Any

import torch
from torch import nn

from mon.coreml.layer import (
    activation, attention, base, conv, linear, normalization,
)
from mon.coreml.layer.typing import _size_2_t
from mon.globals import LAYERS


# region MobileOne Convolution

@LAYERS.register()
class MobileOneConv2d(base.ConvLayerParsingMixin, nn.Module):
    """MobileOneConv2d from the paper: "An Improved One millisecond Mobile
    Backbone". This block has a multi-branched architecture at train-time and
    plain-CNN style architecture at inference time. It is similar to a Conv2d.
    
    References:
        https://github.com/apple/ml-mobileone/blob/main/mobileone.py
    """
    
    def __init__(
        self,
        in_channels      : int,
        out_channels     : int,
        kernel_size      : _size_2_t,
        stride           : _size_2_t = 1,
        padding          : _size_2_t = 0,
        dilation         : _size_2_t = 1,
        groups           : int       = 1,
        bias             : bool      = True,
        padding_mode     : str       = "zeros",
        device           : Any       = None,
        dtype            : Any       = None,
        inference_mode   : bool      = False,
        se               : bool      = False,
        num_conv_branches: int       = 1,
    ):
        super().__init__()
        self.in_channels       = in_channels
        self.out_channels      = out_channels
        self.kernel_size       = kernel_size
        self.stride            = stride
        self.padding           = padding
        self.dilation          = dilation
        self.groups            = groups
        self.bias              = bias
        self.padding_mode      = padding_mode
        self.device            = device
        self.dtype             = dtype
        self.inference_mode    = inference_mode
        self.num_conv_branches = num_conv_branches
        
        # Check if SE-ReLU is requested
        if se is True:
            self.se = attention.SqueezeExciteC(
                channels        = out_channels,
                reduction_ratio = 16,
                bias            = True,
            )
        else:
            self.se = linear.Identity()
        self.act = activation.ReLU()
        
        self.reparam_conv = None
        self.rbr_skip     = None
        self.rbr_conv     = None
        self.rbr_scale    = None
        if inference_mode:
            self.reparam_conv = conv.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = padding,
                dilation     = dilation,
                groups       = groups,
                bias         = True,
                padding_mode = padding_mode,
                device       = device,
                dtype        = dtype,
            )
        else:
            # Re-parameterizable skip connection
            self.rbr_skip = normalization.BatchNorm2d(
                num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None
            
            # Re-parameterizable conv branches
            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(
                    conv.Conv2dBn(
                        in_channels  = in_channels,
                        out_channels = out_channels,
                        kernel_size  = kernel_size,
                        stride       = stride,
                        padding      = padding,
                        dilation     = dilation,
                        groups       = groups,
                        bias         = False,
                        padding_mode = padding_mode,
                        device       = device,
                        dtype        = dtype,
                    )
                )
            self.rbr_conv = nn.ModuleList(rbr_conv)
            
            # Re-parameterizable scale branch
            if kernel_size > 1:
                self.rbr_scale = conv.Conv2dBn(
                    in_channels  = in_channels,
                    out_channels = out_channels,
                    kernel_size  = 1,
                    stride       = stride,
                    padding      = 0,
                    dilation     = dilation,
                    groups       = groups,
                    bias         = False,
                    padding_mode = padding_mode,
                    device       = device,
                    dtype        = dtype,
                )
    
    def _get_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Method to obtain re-parameterized kernel and bias.
        
        Reference:
            https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83
        
        Returns:
            Tuple of (kernel, bias) after fusing branches.
        """
        # Get weights and bias of scale branch
        kernel_scale = 0
        bias_scale   = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad          = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(
                kernel_scale,
                [pad, pad, pad, pad]
            )
        
        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity   = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)
        
        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv   = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            kernel_conv   += _kernel
            bias_conv     += _bias
        
        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final   = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final
    
    def _fuse_bn_tensor(self, branch) -> tuple[torch.Tensor, torch.Tensor]:
        """Method to fuse batchnorm layer with preceeding conv layer.
        
        Reference:
            https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        Returns:
            Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, conv.Conv2dBn):
            kernel       = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var  = branch.bn.running_var
            gamma        = branch.bn.weight
            beta         = branch.bn.bias
            eps          = branch.bn.eps
        else:
            assert isinstance(branch, normalization.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                    dtype  = branch.weight.dtype,
                    device = branch.weight.devices,
                )
                for i in range(self.in_channels):
                    kernel_value[
                        i, i % input_dim,
                           self.kernel_size // 2,
                           self.kernel_size // 2
                    ] = 1
                self.id_tensor = kernel_value
            kernel       = self.id_tensor
            running_mean = branch.running_mean
            running_var  = branch.running_var
            gamma        = branch.weight
            beta         = branch.bias
            eps          = branch.eps
        std = (running_var + eps).sqrt()
        t   = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
    
    def reparameterize(self):
        """Following works like "RepVGG: Making VGG-style ConvNets Great Again"
        (https://arxiv.org/pdf/2101.03697.pdf). We re-parameterize
        multi-branched architecture used at training time to obtain a plain
        CNN-like structure for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = conv.Conv2d(
            in_channels  = self.rbr_conv[0].conv.in_channels,
            out_channels = self.rbr_conv[0].conv.out_channels,
            kernel_size  = self.rbr_conv[0].conv.kernel_size,
            stride       = self.rbr_conv[0].conv.stride,
            padding      = self.rbr_conv[0].conv.padding,
            dilation     = self.rbr_conv[0].conv.dilation,
            groups       = self.rbr_conv[0].conv.groups,
            bias         = True
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data   = bias
        
        # Delete un-used branches
        for param in self.parameters():
            param.detach_()
        self.__delattr__("rbr_conv")
        self.__delattr__("rbr_scale")
        if hasattr(self, "rbr_skip"):
            self.__delattr__("rbr_skip")
        
        self.inference_mode = True
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        # Inference mode forward pass.
        if self.inference_mode:
            y = self.act(self.se(self.reparam_conv(x)))
            return y
        # Multi-branched train-time forward pass
        else:
            # Skip branch output
            y_identity = 0
            if self.rbr_skip is not None:
                y_identity = self.rbr_skip(x)
            # Scale branch output
            y_scale = 0
            if self.rbr_scale is not None:
                y_scale = self.rbr_scale(x)
            # Other branches
            y = y_scale + y_identity
            for ix in range(self.num_conv_branches):
                y += self.rbr_conv[ix](x)
            # Final output
            y = self.act(self.se(y))
            return y

# endregion


# region MobuleOne Module

@LAYERS.register()
class MobileOneStage(base.ConvLayerParsingMixin, nn.Module):
    """MobileOneStage used to construct the MobileOne Model from the paper:
    "An Improved One millisecond Mobile Backbone" (https://arxiv.org/pdf/2206.04040.pdf).
    
    References:
        https://github.com/apple/ml-mobileone/blob/main/mobileone.py
    """
    
    def __init__(
        self,
        in_channels      : int,
        out_channels     : int,
        num_blocks       : int,
        num_se_blocks    : int,
        inference_mode   : bool = False,
        num_conv_branches: int  = 1,
    ):
        super().__init__()
        strides = [2] + [1] * (num_blocks - 1)
        convs   = []
        for ix, stride in enumerate(strides):
            se = False
            if num_se_blocks > num_blocks:
                raise ValueError(
                    f"Require number of SE blocks less than number of layers. "
                    f"But got: {num_se_blocks} > {num_blocks}."
                )
            if ix >= (num_blocks - num_se_blocks):
                se = True
            
            # Depthwise
            convs.append(
                MobileOneConv2d(
                    in_channels       = in_channels,
                    out_channels      = in_channels,
                    kernel_size       = 3,
                    stride            = stride,
                    padding           = 1,
                    groups            = in_channels,
                    inference_mode    = inference_mode,
                    se                = se,
                    num_conv_branches = num_conv_branches,
                )
            )
            # Pointwise
            convs.append(
                MobileOneConv2d(
                    in_channels       = in_channels,
                    out_channels      = out_channels,
                    kernel_size       = 1,
                    stride            = 1,
                    padding           = 0,
                    groups            = 1,
                    inference_mode    = inference_mode,
                    se                = se,
                    num_conv_branches = num_conv_branches,
                )
            )
            in_channels = out_channels
        self.convs = nn.Sequential(*convs)
    
    def reparameterize(self):
        for module in self.convs.modules():
            if hasattr(module, "reparameterize"):
                module.reparameterize()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.convs(x)
        return y

# endregion
