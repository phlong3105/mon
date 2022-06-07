#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FINet: Fraction Detection Normalization Network for Image Restoration.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch import Tensor

from one.core import console
from one.core import Indexes
from one.core import MODELS
from one.core import Pretrained
from one.core import Tensors
from one.nn import Conv3x3
from one.nn import FractionInstanceNorm
from one.nn import SAM
from one.vision.image_enhancement.image_enhancer import ImageEnhancer

__all__ = [
    "FINet",
    "FINetDeBlur",
    "FINetDeBlur_x0_5",
    "FINetDeHaze",
    "FINetDeNoise",
    "FINetDeRain",
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
        use_csff    : bool  = False,
        use_fin     : bool  = False,
        alpha       : float = 0.5,
        selection   : str   = "linear",
    ):
        super().__init__()
        self.downsample = downsample
        self.identity   = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), 0)
        self.use_csff   = use_csff

        self.conv_1 = nn.Conv2d(
            in_channels, out_channels, (3, 3), padding=(1, 1), bias=True
        )
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(
            out_channels, out_channels, (3, 3), padding=(1, 1), bias=True
        )
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(
                out_channels, out_channels, (3, 3), (1, 1), (1, 1)
            )
            self.csff_dec = nn.Conv2d(
                out_channels, out_channels, (3, 3), (1, 1), (1, 1)
            )

        self.use_fin = use_fin
        self.alpha   = alpha
        if self.use_fin:
            self.norm = FractionInstanceNorm(
                alpha        = self.alpha,
                num_features = out_channels,
                selection    = selection,
            )
            console.log(f"Fractional Detection Normalization: "
                        f"num_features: {self.norm.num_features}")
       
        if downsample:
            self.downsample = nn.Conv2d(
                out_channels, out_channels, kernel_size=(4, 4), stride=(2, 2),
                padding=1, bias=False
            )
    
    # MARK: Forward Pass
    
    def forward(
        self,
        x  : Tensor,
        enc: Optional[Tensor] = None,
        dec: Optional[Tensor] = None
    ) -> Tensors:
        out = self.conv_1(x)

        if self.use_fin:
            out = self.norm(out)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

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


# MARK: - FINet

cfgs = {
    # De-blur
    "finet_deblur": {
        "in_channels": 3, "out_channels": 64, "depth": 5, "relu_slope": 0.2,
        "fin_position_left": 0, "fin_position_right": 4, "alpha": 0.5,
        "selection": "linear",
    },
    "finet_deblur_x0.5": {
        "in_channels": 3, "out_channels": 32, "depth": 5, "relu_slope": 0.2,
        "fin_position_left": 0, "fin_position_right": 4, "alpha": 0.5,
        "selection": "linear",
    },
    # De-haze
    "finet_dehaze": {
        "in_channels": 3, "out_channels": 64, "depth": 5, "relu_slope": 0.2,
        "fin_position_left": 0, "fin_position_right": 4, "alpha": 0.5,
        "selection": "linear",
    },
    # De-noise
    "finet_denoise": {
        "in_channels": 3, "out_channels": 64, "depth": 5, "relu_slope": 0.2,
        "fin_position_left": 3, "fin_position_right": 4, "alpha": 0.5,
        "selection": "linear",
    },
    # De-rain
    "finet_derain": {
        "in_channels": 3, "out_channels": 64, "depth": 5, "relu_slope": 0.2,
        "fin_position_left": 0, "fin_position_right": 4, "alpha": 0.0,
        "selection": "linear",
    },
}


@MODELS.register(name="finet")
class FINet(ImageEnhancer):

    # MARK: Magic Functions
    
    def __init__(
        self,
        # Hyperparameters
        in_channels       : int           = 3,
        out_channels      : int           = 64,
        depth             : int           = 5,
        relu_slope        : float         = 0.2,
        fin_position_left : int           = 0,
        fin_position_right: int           = 4,
        alpha             : float         = 0.5,
        selection         : str           = "linear",
        # BaseModel's args
        basename   : Optional[str] = "finet",
        name       : Optional[str] = "finet",
        num_classes: Optional[int] = None,
        out_indexes: Indexes       = -1,
        pretrained : Pretrained    = False,
        *args      , **kwargs
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
        self.fin_position_left  = fin_position_left
        self.fin_position_right = fin_position_right
        self.alpha              = alpha
        self.selection          = selection
        
        # UNet Down-paths
        self.down_path_1 = nn.ModuleList()  # 1st UNet
        self.down_path_2 = nn.ModuleList()  # 2nd Unet
        self.conv_01     = nn.Conv2d(
            self.in_channels, self.out_channels, (3, 3), (1, 1), (1, 1)
        )
        self.conv_02     = nn.Conv2d(
            self.in_channels, self.out_channels, (3, 3), (1, 1), (1, 1)
        )

        prev_channels = self.get_input_channels(self.out_channels)
        for i in range(self.depth):  # 0, 1, 2, 3, 4
            use_fin    = (True if self.fin_position_left <= i <= self.fin_position_right
                          else False)
            downsample = True if (i + 1) < self.depth else False
            self.down_path_1.append(UNetConvBlock(
                prev_channels, (2**i) * self.out_channels, downsample,
                self.relu_slope, use_fin=use_fin, alpha=self.alpha,
                selection=self.selection,
            ))
            self.down_path_2.append(UNetConvBlock(
                prev_channels, (2**i) * self.out_channels, downsample,
                self.relu_slope, use_csff=downsample, use_fin=use_fin,
                alpha=self.alpha, selection=self.selection,
            ))
            prev_channels = (2**i) * self.out_channels
        
        # UNet Up-paths
        self.up_path_1   = nn.ModuleList()
        self.up_path_2   = nn.ModuleList()
        self.skip_conv_1 = nn.ModuleList()
        self.skip_conv_2 = nn.ModuleList()
        for i in reversed(range(self.depth - 1)):
            self.up_path_1.append(UNetUpBlock(
                prev_channels, (2**i) * self.out_channels, self.relu_slope
            ))
            self.up_path_2.append(UNetUpBlock(
                prev_channels, (2**i) * self.out_channels, self.relu_slope
            ))
            self.skip_conv_1.append(nn.Conv2d(
                (2**i) * self.out_channels, (2**i) * self.out_channels, (3, 3),
                (1, 1), (1, 1)
            ))
            self.skip_conv_2.append(nn.Conv2d(
                (2**i) * self.out_channels, (2**i) * self.out_channels, (3, 3),
                (1, 1), (1, 1)
            ))
            prev_channels = (2**i) * self.out_channels
        
        # SAM and CSFF
        self.sam12 = SAM(prev_channels, kernel_size=3, bias=True)
        self.cat12 = nn.Conv2d(prev_channels * 2, prev_channels, (1, 1), (1, 1), 0)
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
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)
    
    # MARK: Forward Pass

    def forward_once(self, x: Tensor, *args, **kwargs) -> Tensors:
        """Forward pass once. Implement the logic for a single forward pass.

        Args:
            x (Tensor):
                Input of shape [B, C, H, W].

        Returns:
            yat (Tensors):
                Predictions.
        """
        image = x
        
        # NOTE: Stage 1
        x1   = self.conv_01(image)
        encs = []
        decs = []
        # UNet1 down-path
        for i, down in enumerate(self.down_path_1):
            if (i+1) < self.depth:
                x1, x1_up = down(x1)
                encs.append(x1_up)
            else:
                x1 = down(x1)
        # Unet1 up-path
        for i, up in enumerate(self.up_path_1):
            x1 = up(x1, self.skip_conv_1[i](encs[-i-1]))
            decs.append(x1)
        # SAM
        sam_feature, out_1 = self.sam12(x1, image)
        
        # NOTE: Stage 2
        x2     = self.conv_02(image)
        x2     = self.cat12(torch.cat([x2, sam_feature], dim=1))
        blocks = []
        # Unet2 down-path
        for i, down in enumerate(self.down_path_2):
            if (i+1) < self.depth:
                x2, x2_up = down(x2, encs[i], decs[-i-1])
                blocks.append(x2_up)
            else:
                x2 = down(x2)
        # Unet2 up-path
        for i, up in enumerate(self.up_path_2):
            x2 = up(x2, self.skip_conv_2[i](blocks[-i-1]))
        
        # NOTE: Last layer
        out_2 = self.last(x2)
        out_2 += image
        
        return [out_1, out_2]
        
    # MARK: Training
    
    def on_fit_start(self):
        """Called at the very beginning of fit."""
        super().on_fit_start()
        if self.shape:
            h, w, c = self.shape
            if h != w:
                raise ValueError("Image height and width must be equal.")
            # assert h == 256 and w == 256, \
            #     (f"FINet model requires image's shape to be [256, 256, :]. "
            #      f"Got {self.shape}.")


@MODELS.register(name="finet_deblur")
class FINetDeBlur(FINet):
    
    models_zoo = {}
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "finet_deblur",
        num_classes: Optional[int] = None,
        out_indexes: Indexes       = -1,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["finet_deblur"] | kwargs
        super().__init__(
            name        = name,
            num_classes = num_classes,
            out_indexes = out_indexes,
            pretrained  = pretrained,
            *args, **kwargs
        )


@MODELS.register(name="finet_deblur_x0.5")
class FINetDeBlur_x0_5(FINet):
    
    models_zoo = {}
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "finet_deblur_x0.5",
        num_classes: Optional[int] = None,
        out_indexes: Indexes       = -1,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["finet_deblur_x0.5"] | kwargs
        super().__init__(
            name        = name,
            num_classes = num_classes,
            out_indexes = out_indexes,
            pretrained  = pretrained,
            *args, **kwargs
        )


@MODELS.register(name="finet_dehaze")
class FINetDeHaze(FINet):
    
    models_zoo = {}
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "finet_dehaze",
        num_classes: Optional[int] = None,
        out_indexes: Indexes       = -1,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["finet_dehaze"] | kwargs
        super().__init__(
            name        = name,
            num_classes = num_classes,
            out_indexes = out_indexes,
            pretrained  = pretrained,
            *args, **kwargs
        )


@MODELS.register(name="finet_denoise")
class FINetDeNoise(FINet):
    
    models_zoo = {}
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "finet_denoise",
        num_classes: Optional[int] = None,
        out_indexes: Indexes       = -1,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["finet_denoise"] | kwargs
        super().__init__(
            name        = name,
            num_classes = num_classes,
            out_indexes = out_indexes,
            pretrained  = pretrained,
            *args, **kwargs
        )


@MODELS.register(name="finet_derain")
class FINetDeRain(FINet):
    
    models_zoo = {}
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "finet_derain",
        num_classes: Optional[int] = None,
        out_indexes: Indexes       = -1,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["finet_derain"] | kwargs
        super().__init__(
            name        = name,
            num_classes = num_classes,
            out_indexes = out_indexes,
            pretrained  = pretrained,
            *args, **kwargs
        )
