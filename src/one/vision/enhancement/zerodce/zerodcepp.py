#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Zero-DCE++: Learning to Enhance Low-Light Image via Zero-Reference Deep Curve
Estimation

Zero-DCE++ has a fast inference speed (1000/11 FPS on single GPU/CPU for an
image with a size of 1200*900*3) while keeping the enhancement performance of
Zero-DCE.

References:
    https://github.com/Li-Chongyi/Zero-DCE_extension
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from one.core import ACT_LAYERS
from one.core import Indexes
from one.core import MODELS
from one.core import Pretrained
from one.core import Tensors
from one.nn import DepthwiseConv
from one.nn import PointwiseConv
from one.vision.enhancement.image_enhancer import ImageEnhancer
from one.vision.enhancement.zerodce.loss import CombinedLoss

__all__ = [
    "ZeroDCEPP",
]


# MARK: - Modules


class CSDNTem(nn.Module):
    
    # MARK: Magic Functions
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.dw_conv = DepthwiseConv(
            in_channels=in_channels, out_channels=in_channels, padding=1,
            groups=in_channels, bias=True,
        )
        self.pw_conv = PointwiseConv(
            in_channels=in_channels, out_channels=out_channels, padding=0,
            groups=1
        )

    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.dw_conv(x)
        out = self.pw_conv(out)
        return out


# MARK: - ZeroDCE++

@MODELS.register(name="zerodce++")
class ZeroDCEPP(ImageEnhancer):
    """
    
    References:
        https://github.com/Li-Chongyi/Zero-DCE
    
    Args:
        name (str, optional):
            Name of the backbone. Default: `zerodce++`.
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

    model_zoo = {
        "sice": dict(
            path="https://github.com/Li-Chongyi/Zero-DCE_extension/blob/main/Zero-DCE%2B%2B/snapshots_Zero_DCE%2B%2B/Epoch99.pth",
            file_name="zerodce++_sice.pth", num_classes=None,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # Hyperparameters
        scale_factor: int           = 1,
        channels    : int           = 32,
        act                         = nn.ReLU(inplace=True),
        # BaseModel's args
        basename    : Optional[str] = "zerodce++",
        name        : Optional[str] = "zerodce++",
        num_classes : Optional[int] = None,
        out_indexes : Indexes       = -1,
        pretrained  : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs["loss"] = CombinedLoss(
            spa_weight     = 1.0,
            exp_patch_size = 16,
            exp_mean_val   = 0.6,
            exp_weight     = 10.0,
            col_weight     = 5.0,
            tv_weight      = 1600.0,
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
        self.scale_factor = scale_factor
        self.channels     = channels
        
        # NOTE: Features
        # Zero-DCE DWC + p-shared
        self.e_conv1  = CSDNTem(3, self.channels)
        self.e_conv2  = CSDNTem(self.channels, self.channels)
        self.e_conv3  = CSDNTem(self.channels, self.channels)
        self.e_conv4  = CSDNTem(self.channels, self.channels)
        self.e_conv5  = CSDNTem(self.channels * 2, self.channels)
        self.e_conv6  = CSDNTem(self.channels * 2, self.channels)
        self.e_conv7  = CSDNTem(self.channels * 2, 3)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)

        self.act = act
        if isinstance(self.act, str):
            self.act = ACT_LAYERS.build(name=self.act)
      
        # NOTE: Load Pretrained
        if self.pretrained:
            self.load_pretrained()
        else:
            self.apply(self.weights_init)
        
    # MARK: Configure

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    
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
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(
                x, scale_factor=1.0 / self.scale_factor, mode="bilinear"
            )

        x1  = self.act(self.e_conv1(x_down))
        x2  = self.act(self.e_conv2(x1))
        x3  = self.act(self.e_conv3(x2))
        x4  = self.act(self.e_conv4(x3))
        x5  = self.act(self.e_conv5(torch.cat([x3, x4], 1)))
        x6  = self.act(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        
        if self.scale_factor == 1:
            x_r = x_r
        else:
            x_r = self.upsample(x_r)
        
        # NOTE: Enhance
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image_1 = x + x_r * (torch.pow(x, 2) - x)
        x = enhance_image_1 + x_r * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image = x + x_r * (torch.pow(x, 2) - x)

        return x_r, enhance_image
