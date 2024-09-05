#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""UNet.

This module implements the paper: U-Net: Convolutional Networks for Biomedical
Image Segmentation.

References:
    https://github.com/milesial/Pytorch-UNet
"""

from __future__ import annotations

__all__ = [
    "UNet",
]

from typing import Any

import torch

from mon import core, nn
from mon.globals import MODELS, Scheme
from mon.nn import functional as F
from mon.vision.segment import base

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Module

class DoubleConvBlock(nn.Module):
    """(Convolution => BN => ReLU) * 2"""
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = self.con1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        return x


class DownConvBlock(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConvBlock(in_channels, out_channels)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpConvBlock(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        # If bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvBlock(in_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1     = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1     = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])
        # If you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)
    
# endregion


# region Model

@MODELS.register(name="unet", arch="unet")
class UNet(base.SegmentationModel):
    """U-Net: Convolutional Networks for Biomedical Image Segmentation
    
    References:
        https://github.com/milesial/Pytorch-UNet
    """
    
    model_dir: core.Path    = current_dir
    arch     : str          = "unet"
    schemes  : list[Scheme] = [Scheme.SUPERVISED]
    zoo      : dict         = {}
    
    def __init__(
        self,
        in_channels : int = 3,
        out_channels: int = 1,
        weights     : Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name         = "unet",
            in_channels  = in_channels,
            out_channels = out_channels,
            weights      = weights,
            *args, **kwargs
        )
        
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            in_channels  = self.weights.get("in_channels",  in_channels)
            out_channels = self.weights.get("out_channels", out_channels)
        self.in_channels  = in_channels  or self.in_channels
        self.out_channels = out_channels or self.out_channels
        
        # Construct model
        nb_filter  = [64, 128, 256, 512, 1024]
        bilinear   = True
        factor     = 2 if bilinear else 1
        multiclass = self.out_channels > 1
        #
        self.inc   = DoubleConvBlock(self.in_channels, nb_filter[0])
        self.down1 = DownConvBlock(nb_filter[0],       nb_filter[1])
        self.down2 = DownConvBlock(nb_filter[1],       nb_filter[2])
        self.down3 = DownConvBlock(nb_filter[2],       nb_filter[3])
        self.down4 = DownConvBlock(nb_filter[3], nb_filter[4] // factor)
        self.up1   =   UpConvBlock(nb_filter[4], nb_filter[3] // factor, bilinear)
        self.up2   =   UpConvBlock(nb_filter[3], nb_filter[2] // factor, bilinear)
        self.up3   =   UpConvBlock(nb_filter[2], nb_filter[1] // factor, bilinear)
        self.up4   =   UpConvBlock(nb_filter[1], nb_filter[0],           bilinear)
        self.outc  = OutConv(nb_filter[0], self.out_channels)
        
        # Loss
        self.loss = nn.DiceLoss(multiclass=multiclass)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
    
    def init_weights(self, m: nn.Module):
        pass
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        image = datapoint.get("image")
        # Encoder
        x1 = self.inc(image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # Decoder
        y1 = self.up1(x5, x4)
        y2 = self.up2(y1, x3)
        y3 = self.up3(y2, x2)
        y4 = self.up4(y3, x1)
        y  = self.outc(y4)
        #
        return {
            "semantic": y,
        }
    
# endregion
