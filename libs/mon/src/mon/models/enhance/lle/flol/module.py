#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules.

This module provides various layers, blocks, and modules for the FLOL model.
"""

from __future__ import annotations

__all__ = [
    "AmplitudeNet_skip",
    "ResidualBlock_noBN",
    "SFNet",
    "make_layer",
]

import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .utils import default_init_weights, init_weights


# ==============================================================================
# region MODULES
# ==============================================================================

# --- Layers ---

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        b, c, h, w = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, c, 1, 1) * y + bias.view(1, c, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        b, c, h, w = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, c, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)

        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None
        )


class LayerNorm2d(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class Upsample(nn.Sequential):
    """Upsample module."""

    # --- Lifecycle & Initialization ---
    def __init__(self, scale: int, channels: int):
        """Initialize a new instance.

        Args:
            scale (int): Scale factor. Supported scales: 2^n and 3.
            channels (int): Channel number of intermediate features.
        """
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(channels, 4 * channels, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(channels, 9 * channels, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"Expected scale to be 2^n or 3, but got: {scale}.")

        # Continue the initialization chain
        super().__init__(*m)


# --- Residual Blocks ---

class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        channels: int = 64,
        res_scale: float = 1.0,
        pytorch_init: bool = False
    ):
        """Initialize a new instance.

        Args:
            channels (int): Channel number of intermediate features. Defaults to 64
            res_scale (float): Residual scale. Defaults to 1.0
            pytorch_init (bool): If set to True, use pytorch default init,
                otherwise, use default_init_weights. Defaults to False.
        """
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class ResidualBlock_noBN(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)

        # Initialization
        init_weights([self.conv1, self.conv2], 0.1)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class ResidualBlock(nn.Module):
    """Residual block w/o BN."""

    # --- Lifecycle & Initialization ---
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.bn = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)

        # Initialization
        init_weights([self.conv1, self.conv2], 0.1)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = F.relu(self.bn(self.conv1(x)), inplace=True)
        out = self.conv2(out)
        return identity + out


# --- Blocks ---

class SimpleGate(nn.Module):

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SGE(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, channels: int):
        super().__init__()
        self.dwc = nn.Conv2d(
            in_channels=channels // 2,
            out_channels=channels // 2,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=channels // 2,
            bias=True,
        )

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwc(x1)
        return x1 * x2


class SpaBlock(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        dw_channel = channels * dw_expand
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True,
        )  # the dconv
        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=channels,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dw_channel // 2,
                out_channels=dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = ffn_expand * channels
        self.conv4 = nn.Conv2d(
            in_channels=channels,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=channels,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)

        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        x = self.norm1(x)  # size [B, C, H, W]

        x = self.conv1(x)  # size [B, 2*C, H, W]
        x = self.conv2(x)  # size [B, 2*C, H, W]
        x = self.sg(x)  # size [B, C, H, W]
        x = x * self.sca(x)  # size [B, C, H, W]
        x = self.conv3(x)  # size [B, C, H, W]

        x = self.dropout1(x)

        y = x + x * self.beta  # size [B, C, H, W]

        x = self.conv4(self.norm2(y))  # size [B, 2*C, H, W]
        x = self.sg(x)  # size [B, C, H, W]
        x = self.conv5(x)  # size [B, C, H, W]

        x = self.dropout2(x)

        return y + x * self.gamma


class FreBlock(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, channels: int):
        super().__init__()
        self.fpre = nn.Conv2d(channels, channels, 1, 1, 0)
        self.process1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 1, 1, 0),
        )
        self.process2 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 1, 1, 0),
        )

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        _, _, h, w = x.shape
        x_freq = torch.fft.rfft2(self.fpre(x), norm="backward")
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        pha = self.process2(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(h, w), norm="backward")
        return x_out + x


class SFBlock(nn.Module):

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2
    ):
        super().__init__()
        dw_channel = channels * dw_expand
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True,
        )  # the dconv
        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=channels,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.fatt = FreBlock(dw_channel // 2)
        self.sge = SGE(dw_channel)

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = ffn_expand * channels
        self.conv4 = nn.Conv2d(
            in_channels=channels,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=channels,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)

        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        x = self.norm1(x)  # size [B, C, H, W]

        x = self.conv1(x)  # size [B, 2*C, H, W]
        x = self.conv2(x)  # size [B, 2*C, H, W]
        x = self.sge(x)    # size [B, C, H, W]

        x = self.fatt(x)
        x = self.conv3(x)  # size [B, C, H, W]

        y = x + x * self.beta  # size [B, C, H, W]

        x = self.conv4(self.norm2(y))  # size [B, 2*C, H, W]
        x = self.sg(x)   # size [B, C, H, W]
        x = self.conv5(x)  # size [B, C, H, W]

        return y + x * self.gamma


class ProcessBlock(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, in_channels: int, spatial: bool = True):
        super().__init__()
        self.spatial = spatial
        self.spatial_process = SpaBlock(in_channels) if spatial else nn.Identity()
        self.frequency_process = FreBlock(in_channels)
        self.cat = (nn.Conv2d(2 * in_channels, in_channels, 1, 1, 0)) \
            if spatial else nn.Conv2d(in_channels, in_channels, 1, 1, 0)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        xori = x
        x_freq = self.frequency_process(x)
        x_spatial = self.spatial_process(x)
        xcat = torch.cat([x_spatial, x_freq], 1)
        x_out = self.cat(xcat) if self.spatial else self.cat(x_freq)
        return x_out + xori


class SFNet(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, channels: int, n: int = 5):
        super().__init__()

        self.list_block = list()
        for index in range(n):
            self.list_block.append(ProcessBlock(channels, spatial=False))

        self.block = nn.Sequential(*self.list_block)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        x_ori = x
        x_out = self.block(x_ori)
        xout = x_ori + x_out
        return xout


class AmplitudeNet_skip(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, channels, n: int = 1):
        super().__init__()
        self.conv_init = nn.Conv2d(3, channels, 1, 1, 0)
        self.conv1 = SFBlock (channels)
        self.conv2 = SFBlock (channels)
        self.conv3 = SFBlock (channels)
        self.conv_out = nn.Conv2d(channels, 3, 1, 1, 0)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        x_lr = F.interpolate(x, scale_factor=0.5, mode='bilinear') # Resize and Normalize SNR map
        x_lr = self.conv_init(x_lr)
        x_lr = self.conv1(x_lr)
        x_lr = self.conv2(x_lr)
        x_lr = self.conv3(x_lr)
        x_lr = self.conv_out(x_lr)
        xout = F.interpolate(x_lr, scale_factor=2, mode='bilinear') # Resize and Normalize SNR map
        return xout


class SG(nn.Module):

    # --- Lifecycle & Initialization ---
    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SGE(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, channels: int):
        super().__init__()
        self.dwc = nn.Conv2d(
            in_channels=channels // 2,
            out_channels=channels // 2,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=channels // 2,
            bias=True,
        )

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwc(x1)
        return x1 * x2


# --- Containers ---

class MySequential(nn.Sequential):
    """My sequential container to handle multiple inputs."""

    # --- Callable & Context Manager ---
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

# endregion


# ==============================================================================
# region UTILITIES
# ==============================================================================

def make_layer(basic_block, num_basic_block, **kwarg) -> nn.Sequential:
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
