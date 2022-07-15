#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trident: Trident Dehazing Network

References:
    https://github.com/lj1995-computer-vision/Trident-Dehazing-Network
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torchvision.ops import DeformConv2d

from one.core import Indexes
from one.core import MODELS
from one.core import Pretrained
from one.nn import DUB
from one.nn import InversePixelShuffle
from one.nn import RWAB
from one.vision.classification import DPN92
from one.vision.enhancement.image_enhancer import ImageEnhancer

__all__ = [
    "Trident",
]


# MARK: - Modules

class UNet(nn.Module):
    
    # MARK: Magic Functions

    def __init__(
        self, in_channels: int = 3, out_channels: int = 3, num_features: int = 16
    ):
        super().__init__()
        
        # NOTE: Encoder
        # 256 x 256
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, num_features, (4, 4), (2, 2), 1, bias=False),
        )
        # 128 x 128
        self.layer2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 1, num_features * 2, (4, 4), (2, 2), 1, bias=False),
            nn.BatchNorm2d(num_features * 2),
        )
        # 64 x 64
        self.layer3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 2, num_features * 4, (4, 4), (2, 2), 1, bias=False),
            nn.BatchNorm2d(num_features * 4),
        )
        # 32 x 32
        self.layer4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 4, num_features * 8, (4, 4), (2, 2), 1, bias=False),
            nn.BatchNorm2d(num_features * 8),
        )
        # 16 x 16
        self.layer5 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 8, num_features * 8, (4, 4), (2, 2), 1, bias=False),
            nn.BatchNorm2d(num_features * 8),
            
        )
        # 8 x 8
        self.layer6 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 8, num_features * 8, (4, 4), (2, 2), 1, bias=False),
            
        )
        # NOTE: Decoder
        # 4 x 4
        self.dlayer6 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_features * 8, num_features * 8, (4, 4), (2, 2), 1, bias=False),
            nn.BatchNorm2d(num_features * 8),
        )
        # 8 x 8
        self.dlayer5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_features * 16, num_features * 8, (4, 4), (2, 2), 1, bias=False),
            nn.BatchNorm2d(num_features * 8),
        )
        # 16 x 16
        self.dlayer4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_features * 16, num_features * 4, (4, 4), (2, 2), 1, bias=False),
            nn.BatchNorm2d(num_features * 4),
        )
        # 32 x 32
        self.dlayer3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_features * 8, num_features * 2, (4, 4), (2, 2), 1, bias=False),
            nn.BatchNorm2d(num_features * 2),
        )
        # 64 x 64
        self.dlayer2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_features * 4, num_features, (4, 4), (2, 2), 1, bias=False),
            nn.BatchNorm2d(num_features),
        )
        # 128 x 128
        self.dlayer1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_features * 2, num_features * 2, (4, 4), (2, 2), 1, bias=False),
            nn.BatchNorm2d(num_features * 2),
        )
        # NOTE: Head
        self.last_conv = nn.Conv2d(num_features * 2, out_channels, (3, 3), padding=1, bias=True)
        
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        out1       = self.layer1(x)
        out2       = self.layer2(out1)
        out3       = self.layer3(out2)
        out4       = self.layer4(out3)
        out5       = self.layer5(out4)
        out6       = self.layer6(out5)
        dout6      = self.dlayer6(out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5      = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4      = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3      = self.dlayer3(dout4_out3)
        dout3_out2 = torch.cat([dout3, out2], 1)
        dout2      = self.dlayer2(dout3_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1      = self.dlayer1(dout2_out1)
        dout1      = self.tail_conv(dout1)
        return dout1


# MARK: - Trident

@MODELS.register(name="trident")
class Trident(ImageEnhancer):
    """
 
    Args:
        name (str, optional):
            Name of the backbone. Default: `rdn`.
        out_indexes (Indexes):
            List of output tensors taken from specific layers' indexes.
            If `>= 0`, return the ith layer's output.
            If `-1`, return the final layer's output. Default: `-1`.
        pretrained (Pretrained):
            Use pretrained weights. If `True`, returns a model pre-trained on
            ImageNet. If `str`, load weights from saved file. Default: `True`.
            - If `True`, returns a model pre-trained on ImageNet.
            - If `str` and is a weight file(path), then load weights from
              saved file.
            - In each inherited model, `pretrained` can be a dictionary's
              key to get the corresponding local file or url of the weight.
    """

    model_zoo = {}
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # Hyperparameters
        in_channels : int = 3,
        out_channels: int = 3,
        num_features: int = 8,
        # BaseModel's args
        basename    : Optional[str] = "trident",
        name        : Optional[str] = "trident",
        num_classes : Optional[int] = None,
        out_indexes : Indexes       = -1,
        pretrained  : Pretrained    = False,
        *args, **kwargs
    ):
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
        self.num_features = num_features
        
        if self.pretrained:
            dpn92 = DPN92(num_classes=1000, pretrained="imagenet+5k").features
        else:
            dpn92 = DPN92(num_classes=1000, pretrained=False).features

        # NOTE: Haze Density Map Generate sub-Net
        self.d64u1 = UNet(self.in_channels, self.out_channels, self.num_features)

        # NOTE: Encoder Decoder sub-Net
        self.d8   = dpn92[:5]   # out608
        self.d16  = dpn92[5:9]  # out1096
        self.d32  = dpn92[9:29]

        self.u16  = DUB(in_channels=2432, mid_channels=512, out_channels=256)
        self.u8   = DUB(in_channels=1352, mid_channels=256, out_channels=128)
        self.u4   = DUB(in_channels=736,  mid_channels=128, out_channels=256)
        self.u2   = DUB(in_channels=256,  mid_channels=64,  out_channels=128)
        self.u1   = DUB(in_channels=128,  mid_channels=32,  out_channels=16)

        self.in16 = nn.InstanceNorm2d(1096, affine=False)
        self.in8  = nn.InstanceNorm2d(608,  affine=False)
        
        # NOTE: Details Refinement sub-Net
        self.d4u1 = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), (1, 1), 1, bias=True),
            nn.BatchNorm2d(16),
            InversePixelShuffle(scale_factor=4),
            nn.Conv2d(256, 16, (3, 3), (1, 1), 1, bias=True),
            nn.BatchNorm2d(16),
            nn.Sequential(*[RWAB(in_channels=16) for _ in range(3)]),
            nn.Conv2d(16, 256, (3, 3), (1, 1), 1, bias=True),
            nn.PixelShuffle(4),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 13, (3, 3), (1, 1), 1, bias=True)
        )
        
        # NOTE: Head
        self.tail = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DeformConv2d(32, 3, 3, 1, 1)
        )
        
        # NOTE: Load Pretrained
        if self.pretrained:
            self.load_pretrained()

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
        b, c, h, w = x.shape
        mod1       = h % 64
        mod2       = w % 64
        if mod1:
            down1 = 64 - mod1
            x = F.pad(x, (0, 0, 0, down1), "reflect")
        if mod2:
            down2 = 64 - mod2
            x = F.pad(x, (0, down2, 0, 0), "reflect")

        d8  = self.d8(x)
        d16 = self.d16(d8)
        d32 = self.d32(d16)
        d16 = torch.cat(d16, 1)
        d8  = torch.cat(d8, 1)
        d16 = self.in16(d16)
        d8  = self.in8(d8)

        u16 = self.u16(torch.cat(d32, 1))
        u8  = self.u8(torch.cat([u16, d16], 1))
        u4  = self.u4(torch.cat([u8, d8], 1))
        u2  = self.u2(u4)
        u1  = self.u1(u2)

        d64u1 = self.d64u1(x)
        d4u1  = self.d4u1(x)
        x     = torch.cat([u1, d64u1, d4u1], 1)
        x     = self.tail(x)

        if mod1:
            x = x[:, :, :-down1, :]
        if mod2:
            x = x[:, :, :, :-down2]
        return x.clamp(0, 1)
