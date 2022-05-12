#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""HINMPR: Half Detection Normalization for Multistage Progressive Image
Restoration Network.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch import Tensor

from one.core import Indexes
from one.core import MODELS
from one.core import Pretrained
from one.core import Tensors
from one.nn import Conv3x3
from one.nn import SAM
from one.vision.enhancement.image_enhancer import ImageEnhancer

__all__ = [
    "HINet3",
    "HINet3DeBlur",
    "HINet3DeBlur_x0_5",
    "HINet3DeNoise",
    "HINet3DeRain",
]


# MARK: - Modules

class UNetConvBlock(nn.Module):
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        downsample  : bool,
        relu_slope  : float,
        use_csff    : bool = False,
        use_hin     : bool = False
    ):
        super().__init__()
        self.downsample = downsample
        self.identity   = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), 0)
        self.use_csff   = use_csff

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, (3, 3), padding=(1, 1), bias=True
        )
        self.relu1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, (3, 3), padding=(1, 1), bias=True
        )
        self.relu2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(
                out_channels, out_channels, (3, 3), (1, 1), (1, 1)
            )
            self.csff_dec = nn.Conv2d(
                out_channels, out_channels, (3, 3), (1, 1), (1, 1)
            )

        self.use_hin = use_hin
        if self.use_hin:
            self.norm = nn.InstanceNorm2d(out_channels//2, affine=True)

        if downsample:
            self.downsample = nn.Conv2d(
                in_channels, out_channels, kernel_size=(4, 4), stride=(2, 2),
                padding=1, bias=False
            )
    
    # MARK: Forward Pass
    
    def forward(
        self,
        x  : Tensor,
        enc: Optional[Tensor] = None,
        dec: Optional[Tensor] = None
    ) -> Tensors:
        out = self.conv1(x)

        if self.use_hin:
            out1, out2 = torch.chunk(out, 2, dim=1)
            out        = torch.cat([self.norm(out1), out2], dim=1)
        out = self.relu1(out)
        out = self.relu2(self.conv2(out))

        out += self.identity(x)
        if enc is not None and dec is not None:
            if not self.use_csff:
                raise ValueError()
            out = out + self.csff_enc(enc) + self.csff_dec(dec)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


class UNetUpBlock(nn.Module):
    
    # MARK: Magic Functions
    
    def __init__(self, in_channels: int, out_channels: int, relu_slope: float):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2),
            bias=True
        )
        self.conv_block = UNetConvBlock(
            in_channels, out_channels, False, relu_slope
        )
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor, bridge: Tensor) -> Tensor:
        up  = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out
    

class SkipBlocks(nn.Module):
    
    # MARK: Magic Functions
    
    def __init__(
        self, in_channels: int, out_channels: int, repeat_num: int = 1
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.re_num = repeat_num
        mid_c       = 128
        
        self.blocks.append(UNetConvBlock(in_channels, mid_c, False, 0.2))
        for i in range(self.re_num - 2):
            self.blocks.append(UNetConvBlock(mid_c, mid_c, False, 0.2))
        self.blocks.append(UNetConvBlock(mid_c, out_channels, False, 0.2))
        
        self.shortcut = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 1), bias=True
        )

    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        sc = self.shortcut(x)
        for m in self.blocks:
            x = m(x)
        return x + sc


# MARK: - HINet3

cfgs = {
    # De-blur
    "hinet3_deblur": {
        "in_channels": 3, "out_channels": 64, "depth": 5, "relu_slope": 0.2,
        "hin_position_left": 0, "hin_position_right": 4
    },
    "hinet3_deblur_x0.5": {
        "in_channels": 3, "out_channels": 32, "depth": 5, "relu_slope": 0.2,
        "hin_position_left": 0, "hin_position_right": 4
    },
    # De-noise
    "hinet3_denoise": {
        "in_channels": 3, "out_channels": 64, "depth": 5, "relu_slope": 0.2,
        "hin_position_left": 3, "hin_position_right": 4
    },
    # De-rain
    "hinet3_derain": {
        "in_channels": 3, "out_channels": 64, "depth": 5, "relu_slope": 0.2,
        "hin_position_left": 0, "hin_position_right": 4
    },
}


@MODELS.register(name="hinet3")
class HINet3(ImageEnhancer):

    # MARK: Magic Functions
    
    def __init__(
        self,
        # Hyperparameters
        in_channels       : int           = 3,
        out_channels      : int           = 64,
        depth             : int           = 5,
        relu_slope        : float         = 0.2,
        hin_position_left : int           = 0,
        hin_position_right: int           = 4,
        # BaseModel's args
        basename          : Optional[str] = "hinet3",
        name              : Optional[str] = "hinet3",
        num_classes       : Optional[int] = None,
        out_indexes       : Indexes       = -1,
        pretrained        : Pretrained    = False,
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
        self.in_channels        = in_channels
        self.out_channels       = out_channels
        self.depth              = depth
        self.relu_slope         = relu_slope
        self.hin_position_left  = hin_position_left
        self.hin_position_right = hin_position_right
        
        # UNet Down-paths
        self.down_path1 = nn.ModuleList()  # 1st UNet
        self.down_path2 = nn.ModuleList()  # 2nd Unet
        self.down_path3 = nn.ModuleList()  # 3rd Unet
        self.conv1      = nn.Conv2d(
            self.in_channels, self.out_channels, (3, 3), (1, 1), (1, 1)
        )
        self.conv2      = nn.Conv2d(
            self.in_channels, self.out_channels, (3, 3), (1, 1), (1, 1)
        )
        self.conv3      = nn.Conv2d(
            self.in_channels, self.out_channels, (3, 3), (1, 1), (1, 1)
        )

        prev_channels = self.get_input_channels(self.out_channels)
        for i in range(self.depth):  # 0, 1, 2, 3, 4
            use_hin    = (True if self.hin_position_left <= i <= self.hin_position_right
                          else False)
            downsample = True if (i + 1) < self.depth else False
            self.down_path1.append(UNetConvBlock(
                prev_channels, (2 ** i) * self.out_channels, downsample,
                self.relu_slope, use_hin=use_hin
            ))
            self.down_path2.append(UNetConvBlock(
                prev_channels, (2 ** i) * self.out_channels, downsample,
                self.relu_slope, use_csff=downsample, use_hin=use_hin
            ))
            self.down_path3.append(UNetConvBlock(
                prev_channels, (2 ** i) * self.out_channels, downsample,
                self.relu_slope, use_csff=downsample, use_hin=use_hin
            ))
            prev_channels = (2 ** i) * self.out_channels
        
        # UNet Up-paths
        self.up_path1   = nn.ModuleList()
        self.up_path2   = nn.ModuleList()
        self.up_path3   = nn.ModuleList()
        self.skip_conv1 = nn.ModuleList()
        self.skip_conv2 = nn.ModuleList()
        self.skip_conv3 = nn.ModuleList()
        for i in reversed(range(self.depth - 1)):
            self.up_path1.append(UNetUpBlock(
                prev_channels, (2 ** i) * self.out_channels, self.relu_slope
            ))
            self.up_path2.append(UNetUpBlock(
                prev_channels, (2 ** i) * self.out_channels, self.relu_slope
            ))
            self.up_path3.append(UNetUpBlock(
                prev_channels, (2 ** i) * self.out_channels, self.relu_slope
            ))
            self.skip_conv1.append(nn.Conv2d(
                (2 ** i) * self.out_channels, (2 ** i) * self.out_channels,
                (3, 3), (1, 1), (1, 1)
            ))
            self.skip_conv2.append(nn.Conv2d(
                (2 ** i) * self.out_channels, (2 ** i) * self.out_channels,
                (3, 3), (1, 1), (1, 1)
            ))
            self.skip_conv3.append(nn.Conv2d(
                (2**i) * self.out_channels, (2**i) * self.out_channels, (3, 3),
                (1, 1), (1, 1)
            ))
            prev_channels = (2**i) * self.out_channels
        
        # SAM and CSFF
        self.sam12 = SAM(prev_channels, kernel_size=3, bias=True)
        self.sam23 = SAM(prev_channels, kernel_size=3, bias=True)
        self.cat12 = nn.Conv2d(
            prev_channels * 2, prev_channels, (1, 1), (1, 1), 0
        )
        self.cat23 = nn.Conv2d(
            prev_channels * 2, prev_channels, (1, 1), (1, 1), 0
        )
        self.last  = Conv3x3(prev_channels, self.in_channels, padding=1, bias=True)
     
    # MARK: Configure

    # noinspection PyMethodMayBeStatic
    def get_input_channels(self, input_channels: int):
        return input_channels

    def _initialize(self):
        gain = nn.init.calculate_gain("leaky_relu", 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    # MARK: Forward Pass

    def forward_once(self, x: Tensor, *args, **kwargs) -> Tensors:
        """Forward pass once. Implement the logic for a single forward pass.

		Args:
			x (Tensor):
				Input of shape [B, C, H, W].

		Returns:
			yhat (Tensors):
				Predictions.
		"""
        image = x
        
        # NOTE: Stage 1
        x1    = self.conv1(image)
        encs1 = []
        decs1 = []
        # UNet1 down-path
        for i, down in enumerate(self.down_path1):
            if (i+1) < self.depth:
                x1, x1_up = down(x1)
                encs1.append(x1_up)
            else:
                x1 = down(x1)
        # Unet1 up-path
        for i, up in enumerate(self.up_path1):
            x1 = up(x1, self.skip_conv1[i](encs1[-i - 1]))
            decs1.append(x1)
        # SAM1
        sam1, out1 = self.sam12(x1, image)
        
        # NOTE: Stage 2
        x2     = self.conv2(image)
        x2     = self.cat12(torch.cat([x2, sam1], dim=1))
        encs2  = []
        decs2  = []
        # Unet2 down-path
        for i, down in enumerate(self.down_path2):
            if (i+1) < self.depth:
                x2, x2_up = down(x2, encs1[i], decs1[-i-1])
                encs2.append(x2_up)
            else:
                x2 = down(x2)
        # Unet2 up-path
        for i, up in enumerate(self.up_path2):
            x2 = up(x2, self.skip_conv2[i](encs2[-i - 1]))
            decs2.append(x2)
        # SAM2
        sam2, out2 = self.sam23(x2, image)

        # NOTE: Stage 3
        x3     = self.conv3(image)
        x3     = self.cat23(torch.cat([x3, sam2], dim=1))
        blocks = []
        # Unet3 down-path
        for i, down in enumerate(self.down_path3):
            if (i + 1) < self.depth:
                x3, x3_up = down(x3, encs2[i], decs2[-i - 1])
                blocks.append(x3_up)
            else:
                x3 = down(x3)
        # Unet3 up-path
        for i, up in enumerate(self.up_path3):
            x3 = up(x3, self.skip_conv3[i](blocks[-i - 1]))

        # NOTE: Last layer
        out3 = self.last(x3)
        out3 += image

        return [out1, out2, out3]
        
    # MARK: Training
    
    def on_fit_start(self):
        """Called at the very beginning of fit."""
        super().on_fit_start()
        if self.shape:
            h, w, c = self.shape
            if h != w:
                raise ValueError("Image height and width must be equal.")
            # assert h == 256 and w == 256, \
            #     (f"HINet model requires image's shape to be [256, 256, :]. "
            #      f"Got {self.shape}.")


@MODELS.register(name="hinet3_deblur")
class HINet3DeBlur(HINet3):
    
    models_zoo = {}
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "hinet3_deblur",
        num_classes: Optional[int] = None,
        out_indexes: Indexes       = -1,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["hinet3_deblur"] | kwargs
        super().__init__(
            name        = name,
            num_classes = num_classes,
            out_indexes = out_indexes,
            pretrained  = pretrained,
            *args, **kwargs
        )


@MODELS.register(name="hinet3_deblur_x0.5")
class HINet3DeBlur_x0_5(HINet3):
    
    models_zoo = {}
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "hinet3_deblur_x0.5",
        num_classes: Optional[int] = None,
        out_indexes: Indexes       = -1,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["hinet_deblur_x0.5"] | kwargs
        super().__init__(
            name        = name,
            num_classes = num_classes,
            out_indexes = out_indexes,
            pretrained  = pretrained,
            *args, **kwargs
        )


@MODELS.register(name="hinet3_denoise")
class HINet3DeNoise(HINet3):
    
    models_zoo = {}
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "hinet3_denoise",
        num_classes: Optional[int] = None,
        out_indexes: Indexes       = -1,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["hinet3_denoise"] | kwargs
        super().__init__(
            name        = name,
            num_classes = num_classes,
            out_indexes = out_indexes,
            pretrained  = pretrained,
            *args, **kwargs
        )


@MODELS.register(name="hinet3_derain")
class HINet3DeRain(HINet3):
    
    models_zoo = {}
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "hinet3_derain",
        num_classes: Optional[int] = None,
        out_indexes: Indexes       = -1,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["hinet3_derain"] | kwargs
        super().__init__(
            name        = name,
            num_classes = num_classes,
            out_indexes = out_indexes,
            pretrained  = pretrained,
            *args, **kwargs
        )
