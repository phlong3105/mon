#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "ConvBlock",
    "DSConv",
    "DenoiseNet",
    "MobileOneConv",
    "normalize_minmax",
    "reparameterize_model",
    "weights_init",
]

import copy

import torch

from mon.core import nn


# ----- Utils -----
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def reparameterize_model(model: nn.Module) -> nn.Module:
    """Method returns a model where a multi-branched structure used in training
    is re-parameterized into a single branch for inference.

    Args:
        model: Model to re-parameterize.
    
    Returns:
        Re-parameterized model.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()
    return model


def normalize_minmax(x: torch.Tensor, scale: float = 1) -> torch.Tensor:
    x = x * scale
    return (x - x.min()) / (x.max() - x.min())


# ----- Conv -----
class DSConv(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels,
            kernel_size  = 3,
            stride       = 1,
            padding      = 1,
            groups       = in_channels
        )
        self.point_conv = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            groups       = 1
        )
        self.depth_conv.apply(weights_init)
        self.point_conv.apply(weights_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depth_conv(x)
        y = self.point_conv(y)
        return y


class ConvBlock(nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        norm        : nn.Module = nn.AdaptiveBatchNorm2d,
        use_se      : bool      = True,
    ):
        super().__init__()
        self.conv = DSConv(in_channels, out_channels)
        if norm:
            self.norm = norm(out_channels)
        else:
            self.norm = nn.Identity()
        if use_se:
            self.se = nn.SEBlock(out_channels)
        else:
            self.se = nn.Identity()
       
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.se(self.norm(self.conv(input)))


class MobileOneConv(nn.Module):
    
    def __init__(
        self,
        in_channels      : int,
        out_channels     : int,
        inference        : bool = False,
        use_se           : bool = False,
        use_act          : bool = True,
        num_conv_branches: int  = 1,
    ):
        super().__init__()
        self.depth_conv = nn.MobileOneBlock(
            in_channels       = in_channels,
            out_channels      = in_channels,
            kernel_size       = 3,
            stride            = 1,
            padding           = 1,
            groups            = in_channels,
            inference         = inference,
            use_se            = use_se,
            use_act           = use_act,
            num_conv_branches = num_conv_branches,
        )
        self.point_conv = nn.MobileOneBlock(
            in_channels       = in_channels,
            out_channels      = out_channels,
            kernel_size       = 1,
            stride            = 1,
            padding           = 0,
            groups            = 1,
            inference         = inference,
            use_se            = use_se,
            use_act           = use_act,
            num_conv_branches = num_conv_branches,
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        y = self.depth_conv(input)
        y = self.point_conv(y)
        return y


# ----- Network -----
class DenoiseNet(nn.Module):
    
    def __init__(self, in_channels: int, embedded_channels: int = 48):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,       embedded_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(embedded_channels, embedded_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(embedded_channels, in_channels,       1)
        self.act   = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x
