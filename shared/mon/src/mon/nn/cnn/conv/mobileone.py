#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for MobileOne building block.

This module implements the MobileOneBlock class, which is a building block
for the MobileOne architecture. The block features a multi-branched structure
during training and a re-parameterized single-branch structure for inference.

References:
    - Paper: "MobileOne: An Improved One millisecond Mobile Backbone," CVPR 2023.
    - Code: https://github.com/apple/ml-mobileone/tree/main
"""

__all__ = [
    "MobileOneBlock",
]

import copy

import torch
import torch.nn as nn

from ...transformer.attention import SEBlock


def reparameterize_model(model: nn.Module) -> nn.Module:
    """Re-parameterizes all re-parameterizable modules in the model for inference.
    
    Args:
        model (nn.Module): The model containing re-parameterizable modules.
        
    Returns:
        nn.Module: The re-parameterized model ready for inference.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()
    return model


class MobileOneBlock(nn.Module):
    """A MobileOne building block.

    This block has a multi-branched architecture at train-time and plain-CNN
    style architecture at inference time.
    """
    
    def __init__(
        self,
        in_channels      : int,
        out_channels     : int,
        kernel_size      : int,
        stride           : int  = 1,
        padding          : int  = 0,
        dilation         : int  = 1,
        groups           : int  = 1,
        inference        : bool = False,
        use_se           : bool = False,
        use_act          : bool = True,
        num_conv_branches: int  = 1
    ):
        """Initializes the MobileOneBlock.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride of the convolution. Defaults to 1.
            padding (int): Padding for the convolution. Defaults to 0.
            dilation (int): Dilation for the convolution. Defaults to 1.
            groups (int): Number of groups for grouped convolution. Defaults to 1.
            inference (bool): If True, initializes in inference mode.
                Defaults to False.
            use_se (bool): If True, includes SE block. Defaults to False.
            use_act (bool): If True, includes ReLU activation. Defaults to True.
            num_conv_branches (int): Number of convolutional branches during
                training. Defaults to 1.
        """
        super().__init__()
        self.inference         = inference
        self.groups            = groups
        self.stride            = stride
        self.kernel_size       = kernel_size
        self.in_channels       = in_channels
        self.out_channels      = out_channels
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()
        # Check if activation is requested
        if use_act:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()
        
        if inference:
            self.reparam_conv = nn.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = padding,
                dilation     = dilation,
                groups       = groups,
                bias         = True
            )
        else:
            # Re-parameterizable skip connection
            self.rbr_skip = nn.BatchNorm2d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None

            # Re-parameterizable conv branches
            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(self._conv_bn(kernel_size=kernel_size, padding=padding))
            self.rbr_conv = nn.ModuleList(rbr_conv)

            # Re-parameterizable scale branch
            self.rbr_scale = None
            if kernel_size > 1:
                self.rbr_scale = self._conv_bn(kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MobileOneBlock.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, H_out, W_out).
        """
        # Inference mode forward pass.
        if self.inference:
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))
    
    def reparameterize(self):
        """Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
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
        for para in self.parameters():
            para.detach_()
        self.__delattr__("rbr_conv")
        self.__delattr__("rbr_scale")
        if hasattr(self, "rbr_skip"):
            self.__delattr__("rbr_skip")

        self.inference = True

    def _get_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuses all branches to obtain equivalent kernel and bias for
        re-parameterized conv layer.
        
        References:
            - Code: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        Returns:
             tuple[torch.Tensor, torch.Tensor]: Tuple of kernel and bias tensors.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale   = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale, [pad, pad, pad, pad])

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
        """Fuses batchnorm parameters into convolutional kernel and bias.
        
        References:
            -Code: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95
    
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of kernel and bias tensors.
        """
        if isinstance(branch, nn.Sequential):
            kernel       = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var  = branch.bn.running_var
            gamma        = branch.bn.weight
            beta         = branch.bn.bias
            eps          = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim    = self.in_channels // self.groups
                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                    dtype  = branch.weight.dtype,
                    device = branch.weight.device
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
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

    def _conv_bn(self, kernel_size: int, padding: int) -> nn.Sequential:
        """Creates a convolutional layer followed by batch normalization.
        
        Args:
            kernel_size (int): Size of the convolutional kernel.
            padding (int): Padding for the convolution.
            
        Returns:
            nn.Sequential: A sequential container with conv and batchnorm layers.
        """
        mod_list = nn.Sequential()
        mod_list.add_module(
            "conv",
            nn.Conv2d(
                in_channels  = self.in_channels,
                out_channels = self.out_channels,
                kernel_size  = kernel_size,
                stride       = self.stride,
                padding      = padding,
                groups       = self.groups,
                bias         = False
            )
        )
        mod_list.add_module("bn", nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list
