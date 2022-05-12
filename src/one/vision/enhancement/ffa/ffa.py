#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FFA-Net: Feature Fusion Attention Network for Single Image Dehazing.

References:
    https://github.com/zhilin007/FFA-Net
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch import Tensor

from one.core import Indexes
from one.core import Int2T
from one.core import MODELS
from one.core import Pretrained
from one.nn import CAL
from one.nn import PAL
from one.nn import PerceptualL1Loss
from one.vision.classification import VGG16
from one.vision.enhancement.image_enhancer import ImageEnhancer

__all__ = [
    "FFA",
]


# MARK: - Modules

def default_conv(
    in_channels: int, out_channels: int, kernel_size: Int2T, bias: bool = True
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2),
        bias=bias
    )


class Block(nn.Module):
    
    def __init__(self, conv: nn.Module, channels: int, kernel_size: Int2T):
        super().__init__()
        self.conv1    = conv(channels, channels, kernel_size, bias=True)
        self.act1     = nn.ReLU(inplace=True)
        self.conv2    = conv(channels, channels, kernel_size, bias=True)
        self.ca_layer = CAL(channels, reduction=8, bias=True)
        self.pa_layer = PAL(channels, reduction=8, bias=True)
    
    def forward(self, x: Tensor) -> Tensor:
        out  = self.act1(self.conv1(x))
        out += x
        out  = self.conv2(out)
        out  = self.ca_layer(out)
        out  = self.pa_layer(out)
        out += x
        return out


class Group(nn.Module):
    
    def __init__(
        self, conv: nn.Module, channels: int, kernel_size: Int2T, blocks: int
    ):
        super().__init__()
        modules = [Block(conv, channels, kernel_size) for _ in range(blocks)]
        modules.append(conv(channels, channels, kernel_size))
        self.gp = nn.Sequential(*modules)
    
    def forward(self, x: Tensor) -> Tensor:
        out  = self.gp(x)
        out += x
        return out


# MARK: - FFANet

@MODELS.register(name="ffa")
class FFA(ImageEnhancer):
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # Hyperparameters
        in_channels : int           = 3,
        out_channels: int           = 64,
        kernel_size : Int2T         = 3,
        groups      : int           = 3,
        blocks      : int           = 20,
        conv        : nn.Conv2d     = default_conv,
        # BaseModel's args
        basename    : Optional[str] = "ffa",
        name        : Optional[str] = "ffa",
        num_classes : Optional[int] = None,
        out_indexes : Indexes       = -1,
        pretrained  : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs["loss"] = PerceptualL1Loss(
            vgg=VGG16(out_indexes=list(range(3, 8, 15)), pretrained=True),
            weights=(1.0, 0.04)
        )
        super().__init__(
            basename    = basename,
            name        = name,
            num_classes = num_classes,
            out_indexes = out_indexes,
            pretrained  = pretrained,
            *args, **kwargs
        )
        # NOTE: Get Hyperparameters
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.groups       = groups
        self.blocks       = blocks
        
        if self.groups != 3:
            raise ValueError()
        
        self.pre = nn.Sequential(conv(self.in_channels, self.out_channels, self.kernel_size))
        self.g1  = Group(conv, self.out_channels, self.kernel_size, blocks=self.blocks)
        self.g2  = Group(conv, self.out_channels, self.kernel_size, blocks=self.blocks)
        self.g3  = Group(conv, self.out_channels, self.kernel_size, blocks=self.blocks)
        self.ca  = nn.Sequential(
            *[
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(
                    self.out_channels * self.groups, self.out_channels // 16,
                    (1, 1), padding=0
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    self.out_channels // 16, self.out_channels * self.groups,
                    (1, 1), padding=0, bias=True
                ),
                nn.Sigmoid()
            ]
        )
        self.pa_layer = PAL(self.out_channels, reduction=8, bias=True)
        self.post     = nn.Sequential(
            conv(self.out_channels, self.out_channels, self.kernel_size),
            conv(self.out_channels, 3, self.kernel_size)
        )
 
    # MARK: Forward Pass
    
    def forward_once(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Forward pass once. Implement the logic for a single forward pass.

        Args:
            x (Tensor):
                Input of shape [B, C, H, W].

        Returns:
            yhat (Tensor):
                Predictions.
        """
        out  = self.pre(x)
        res1 = self.g1(out)
        res2 = self.g2(res1)
        res3 = self.g3(res2)
        w    = self.ca(torch.cat([res1, res2, res3], dim=1))
        w    = w.view(-1, self.groups, self.out_channels)[:, :, :, None, None]
        out  = w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3
        out  = self.pa_layer(out)
        out  = self.post(out)
        return out + x
