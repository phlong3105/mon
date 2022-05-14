#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Zero-DCE: Zero-Reference Deep Curve Estimation for Low-Light Image
Enhancement:
- The first low-light enhancement network that is independent of paired and
  unpaired training data, thus avoiding the risk of overfitting. As a result,
  our method generalizes well to various lighting conditions.
- Design an image-specific curve that is able to approximate pixel-wise and
  higher-order curves by iteratively applying itself. Such image-specific curve
  can effectively perform mapping within a wide dynamic range.
- Show the potential of training a deep image enhancement model in the absence
  of reference images through task-specific non-reference loss functions that
  indirectly evaluate enhancement quality. It is capable of processing images
  in real-time (about 500 FPS for images of size 640*480*3 on GPU) and takes
  only 30 minutes for training.

References:
    https://github.com/Li-Chongyi/Zero-DCE
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
from one.core import to_2tuple
from one.vision.image_enhancement.image_enhancer import ImageEnhancer
from one.vision.image_enhancement.zerodce.loss import CombinedLoss

__all__ = [
    "ZeroDCE",
]


# MARK: - ZeroDCE

# noinspection PyDefaultArgument,PyMethodOverriding,PyMethodMayBeStatic
@MODELS.register(name="zerodce")
class ZeroDCE(ImageEnhancer):
    """
    
    References:
        https://github.com/Li-Chongyi/Zero-DCE
    
    Args:
        name (str, optional):
            Name of the backbone. Default: `zerodce`.
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
            path="https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/snapshots/Epoch99.pth",
            file_name="zerodce_sice.pth", num_classes=None,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # Hyperparameters
        channels   : int           = 32,
        kernel_size: int           = 3,
        act                        = nn.ReLU(inplace=True),
        # BaseModel's args
        basename   : Optional[str] = "zerodce",
        name       : Optional[str] = "zerodce",
        num_classes: Optional[int] = None,
        out_indexes: Indexes       = -1,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs["loss"] = CombinedLoss(
            spa_weight     = 1.0,
            exp_patch_size = 16,
            exp_mean_val   = 0.6,
            exp_weight     = 10.0,
            col_weight     = 5.0,
            tv_weight      = 200.0,
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
        self.channels    = channels
        self.kernel_size = to_2tuple(kernel_size)
        
        # NOTE: Features
        self.e_conv1 = nn.Conv2d(3, self.channels, self.kernel_size, (1, 1), 1,
                                 bias=True)
        self.e_conv2 = nn.Conv2d(self.channels, self.channels, self.kernel_size,
                                 (1, 1), 1, bias=True)
        self.e_conv3 = nn.Conv2d(self.channels, self.channels, self.kernel_size,
                                 (1, 1), 1, bias=True)
        self.e_conv4 = nn.Conv2d(self.channels, self.channels, self.kernel_size,
                                 (1, 1), 1, bias=True)
        self.e_conv5 = nn.Conv2d(self.channels * 2, self.channels,
                                 self.kernel_size, (1, 1), 1, bias=True)
        self.e_conv6 = nn.Conv2d(self.channels * 2, self.channels,
                                 self.kernel_size, (1, 1), 1, bias=True)
        self.e_conv7 = nn.Conv2d(self.channels * 2, 24, self.kernel_size,
                                 (1, 1), 1, bias=True)

        self.maxpool  = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

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
        x1 = self.act(self.e_conv1(x))
        # p1 = self.maxpool(x1)
        x2 = self.act(self.e_conv2(x1))
        # p2 = self.maxpool(x2)
        x3 = self.act(self.e_conv3(x2))
        # p3 = self.maxpool(x3)
        x4 = self.act(self.e_conv4(x3))

        x5 = self.act(self.e_conv5(torch.cat([x3, x4], 1)))
        # x5 = self.upsample(x5)
        x6 = self.act(self.e_conv6(torch.cat([x2, x5], 1)))

        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)

        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        return r, enhance_image_1, enhance_image
