#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RDN: Residual Dense Network for Image Super-Resolution.

References:
    https://github.com/yjn870/RDN-pytorch
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch import Tensor

from one.core import Indexes
from one.core import MODELS
from one.core import Pretrained
from one.nn import ResidualDenseBlock
from one.vision.image_enhancement.image_enhancer import ImageEnhancer

__all__ = [
	"RDN",
    "RDNX2",
    "RDNX3",
    "RDNX4",
]


# MARK: - RDN

cfgs = {
    "rdn_x2": {
        "in_channels": 3, "mid_channels": 64, "growth_channels": 64,
        "num_blocks": 16, "num_layers": 8, "scale_factor": 2,
    },
    "rdn_x3": {
        "in_channels": 3, "mid_channels": 64, "growth_channels": 64,
        "num_blocks": 16, "num_layers": 8, "scale_factor": 3,
    },
    "rdn_x4": {
        "in_channels": 3, "mid_channels": 64, "growth_channels": 64,
        "num_blocks": 16, "num_layers": 8, "scale_factor": 4,
    },
}


@MODELS.register(name="rdn")
class RDN(ImageEnhancer):
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
        in_channels    : int = 3,
        mid_channels   : int = 64,
        growth_channels: int = 64,
        num_blocks     : int = 16,
        num_layers     : int = 8,
        scale_factor   : int = 4,
        # BaseModel's args
        basename       : Optional[str] = "rdn",
        name           : Optional[str] = "rdn",
        num_classes    : Optional[int] = None,
        out_indexes    : Indexes       = -1,
        pretrained     : Pretrained    = False,
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
        self.in_channels     = in_channels
        self.mid_channels    = mid_channels
        self.growth_channels = growth_channels
        self.num_blocks      = num_blocks
        self.num_layers      = num_layers
        self.scale_factor    = scale_factor
        
        # NOTE: Shallow Feature Extraction
        self.sfe1 = nn.Conv2d(self.in_channels,  self.mid_channels, kernel_size=(3, 3), padding=3 // 2)
        self.sfe2 = nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=(3, 3), padding=3 // 2)
        
        # NOTE: Residual Dense Blocks
        self.rdbs = nn.ModuleList([
            ResidualDenseBlock(
                num_layers      = self.num_layers,
                in_channels     = self.mid_channels,
                growth_channels = self.growth_channels,
                kernel_size     = 3,
                lff_kernel_size = 1,
                padding         = 3 // 2,
                lff_padding     = 0,
                apply_act       = True,
                act_layer       = nn.ReLU(),
                inplace         = True,
            )
        ])
        for _ in range(self.num_blocks - 1):
            self.rdbs.append(
                ResidualDenseBlock(
                    num_layers      = self.num_layers,
                    in_channels     = self.growth_channels,
                    growth_channels = self.growth_channels,
                    kernel_size     = 3,
                    lff_kernel_size = 1,
                    padding         = 3 // 2,
                    lff_padding     = 0,
                    apply_act       = True,
                    act_layer       = nn.ReLU(),
                    inplace         = True,
                )
            )

        # NOTE: Global Feature Fusion
        self.gff = nn.Sequential(
            nn.Conv2d(
                in_channels  = self.growth_channels * self.num_blocks,
                out_channels = self.mid_channels,
                kernel_size  = (1, 1),
            ),
            nn.Conv2d(
                in_channels  = self.mid_channels,
                out_channels = self.mid_channels,
                kernel_size  = (3, 3),
                padding      = 3 // 2,
            )
        )
        
        # NOTE: Up-sampling
        if not 2 <= self.scale_factor <= 4:
            raise ValueError(f"Scale factor must be in between [2, 4]. "
                             f"But got: {self.scale_factor}")
        if self.scale_factor == 2 or self.scale_factor == 4:
            self.upscale = []
            for _ in range(self.scale_factor // 2):
                self.upscale.extend([
                    nn.Conv2d(
                        in_channels  = self.mid_channels,
                        out_channels = self.mid_channels * (2 ** 2),
                        kernel_size  = (3, 3),
                        padding      = 3 // 2
                    ),
                    nn.PixelShuffle(2)
                ])
            self.upscale = nn.Sequential(*self.upscale)
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    in_channels  = self.mid_channels,
                    out_channels = self.mid_channels * (self.scale_factor ** 2),
                    kernel_size  = (3, 3),
                    padding      = 3 // 2
                ),
                nn.PixelShuffle(scale_factor)
            )
        
        self.output = nn.Conv2d(
            in_channels  = self.mid_channels,
            out_channels = self.in_channels,
            kernel_size  = (3, 3),
            padding      = 3 // 2
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
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.num_blocks):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        x = self.upscale(x)
        x = self.output(x)
        return x


@MODELS.register(name="rdn_x2")
class RDNX2(RDN):
    
    model_zoo = {
        "div2k": dict(path="", file_name="rdn_x2_div2k.pth", num_classes=None),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "rdn_x2",
        num_classes: Optional[int] = None,
        out_indexes: Indexes       = -1,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["rdn_x2"] | kwargs
        super().__init__(
            name        = name,
            num_classes = num_classes,
            out_indexes = out_indexes,
            pretrained  = pretrained,
            *args, **kwargs
        )


@MODELS.register(name="rdn_x3")
class RDNX3(RDN):
    
    model_zoo = {
        "div2k": dict(path="", file_name="rdn_x3_div2k.pth", num_classes=None),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "rdn_x3",
        num_classes: Optional[int] = None,
        out_indexes: Indexes       = -1,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["rdn_x3"] | kwargs
        super().__init__(
            name        = name,
            num_classes = num_classes,
            out_indexes = out_indexes,
            pretrained  = pretrained,
            *args, **kwargs
        )


@MODELS.register(name="rdn_x4")
class RDNX4(RDN):
    
    model_zoo = {
        "div2k": dict(path="", file_name="rdn_x4_div2k.pth", num_classes=None),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "rdn_x4",
        num_classes: Optional[int] = None,
        out_indexes: Indexes       = -1,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["rdn_x4"] | kwargs
        super().__init__(
            name        = name,
            num_classes = num_classes,
            out_indexes = out_indexes,
            pretrained  = pretrained,
            *args, **kwargs
        )
