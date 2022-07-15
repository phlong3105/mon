#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Xception models.

From paper: `Xception: Deep Learning with Depthwise Separable Convolutions`
- https://arxiv.org/pdf/1610.02357.pdf

This weights ported from the Keras implementation. Achieves the following
performance on the validation set:
Loss:0.9173 Prec@1:78.892 Prec@5:94.292

REMEMBER to set your image size to 3x299x299 for both test and validation.

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                  
Resize parameter of the validation transform should be 333, and make sure
to center crop at 299x299.
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from one.core import BACKBONES
from one.core import IMAGENET_INCEPTION_MEAN
from one.core import IMAGENET_INCEPTION_STD
from one.core import Indexes
from one.core import Int2T
from one.core import Int3T
from one.core import MODELS
from one.core import Padding2T
from one.core import Pretrained
from one.core import to_2tuple
from one.nn import create_classifier
from one.vision.classification.image_classifier import ImageClassifier

__all__ = [
    "Xception"
]


# MARK: - Custom Modules

class SeparableConv2d(nn.Module):
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Int2T 			  = 1,
        stride	    : Int2T 			  = 1,
        padding		: Optional[Padding2T] = 0,
        dilation	: Int2T 			  = 1
    ):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride 		= to_2tuple(stride)
        dilation    = to_2tuple(dilation)
        
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, dilation,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1),
                                   0, (1, 1), 1, bias=False)
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels    : int,
        out_channels   : int,
        reps,
        stride         : int  = 1,
        start_with_relu: bool = True,
        grow_first     : bool = True
    ):
        super().__init__()
        # stride = to_2tuple(stride)
        
        if out_channels != in_channels or stride != 1:
            self.skip = nn.Conv2d(
                in_channels, out_channels, (1, 1), stride=(stride, stride),
                bias=False
            )
            self.skip_bn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

        rep = []
        for i in range(reps):
            if grow_first:
                in_channels  = in_channels if i == 0 else out_channels
                out_channels = out_channels
            else:
                in_channels  = in_channels
                out_channels = in_channels if i < (reps - 1) else out_channels
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(
                in_channels, out_channels, 3, stride=1, padding=1)
            )
            rep.append(nn.BatchNorm2d(out_channels))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if stride != 1:
            rep.append(nn.MaxPool2d(3, stride, 1))
        self.rep = nn.Sequential(*rep)
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.rep(x)

        if self.skip is not None:
            skip = self.skip(x)
            skip = self.skip_bn(skip)
        else:
            skip = x

        x += skip
        return x


# MARK: - Xception

cfgs = {
    "xception": {
        "in_channels": 3, "drop_rate": 0.0, "global_pool": "avg",
        "input_size": (3, 299, 299), "pool_size": (10, 10), "crop_pct": 0.8975,
        "interpolation": "bicubic", "mean": IMAGENET_INCEPTION_MEAN,
        "std": IMAGENET_INCEPTION_STD, "first_conv": "conv1", "classifier": "fc"
    },
}


@MODELS.register(name="xception")
@BACKBONES.register(name="xception")
class Xception(ImageClassifier):
    """Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    
    Args:
        basename (str, optional):
            Model basename. Default: `xception`.
        name (str, optional):
            Name of the model. Default: `xception`.
        num_classes (int, optional):
            Number of classes for classification. Default: `None`.
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
        "imagenet": dict(
            path="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/xception-43020ad28.pth",
            file_name="xception_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # Hyperparameters
        in_channels     : int   = 3,
        drop_rate       : float = 0.0,
        global_pool     : str   = "avg",
        input_size      : Int3T  = (3, 299, 299),
        pool_size       : Int2T  = (10, 10),
        crop_pct        : float = 0.8975,
        interpolation   : str   = "bicubic",
		mean            : float = IMAGENET_INCEPTION_MEAN,
		std             : float = IMAGENET_INCEPTION_STD,
		first_conv      : str   = "conv1",
		classifier      : bool  = "fc",
        # BaseModel's args
        basename   : Optional[str] = "xception",
        name       : Optional[str] = "xception",
        num_classes: Optional[int] = None,
        out_indexes: Indexes       = -1,
        pretrained : Pretrained    = False,
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
        self.in_channels = in_channels
        self.drop_rate   = drop_rate
        self.global_pool = global_pool
        num_features     = 2048
        
        # NOTE: Features
        self.conv1   = nn.Conv2d(self.in_channels, 32, (3, 3), (2, 2), 0, bias=False)
        self.bn1     = nn.BatchNorm2d(32)
        self.act1    = nn.ReLU(inplace=True)

        self.conv2   = nn.Conv2d(32, 64, (3, 3), bias=False)
        self.bn2     = nn.BatchNorm2d(64)
        self.act2    = nn.ReLU(inplace=True)

        self.block1  = Block(64, 128, 2, 2, start_with_relu=False)
        self.block2  = Block(128, 256, 2, 2)
        self.block3  = Block(256, 728, 2, 2)

        self.block4  = Block(728, 728, 3, 1)
        self.block5  = Block(728, 728, 3, 1)
        self.block6  = Block(728, 728, 3, 1)
        self.block7  = Block(728, 728, 3, 1)

        self.block8  = Block(728, 728, 3, 1)
        self.block9  = Block(728, 728, 3, 1)
        self.block10 = Block(728, 728, 3, 1)
        self.block11 = Block(728, 728, 3, 1)

        self.block12 = Block(728, 1024, 2, 2, grow_first=False)

        self.conv3   = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 	 = nn.BatchNorm2d(1536)
        self.act3 	 = nn.ReLU(inplace=True)

        self.conv4   = SeparableConv2d(1536, num_features, 3, 1, 1)
        self.bn4 	 = nn.BatchNorm2d(num_features)
        self.act4 	 = nn.ReLU(inplace=True)
        self.feature_info = [
            dict(num_chs=64,   reduction=2,  module="act2"),
            dict(num_chs=128,  reduction=4,  module="block2.rep.0"),
            dict(num_chs=256,  reduction=8,  module="block3.rep.0"),
            dict(num_chs=728,  reduction=16, module="block12.rep.0"),
            dict(num_chs=2048, reduction=32, module="act4"),
        ]
        
        # NOTE: Classifier
        self.global_pool, self.fc = create_classifier(
            num_features, self.num_classes, pool_type=self.global_pool
        )
        
        # NOTE: Load Pretrained
        if self.pretrained:
            self.load_pretrained()
        else:
            self.init_weights()

        # NOTE: Alias
        self.features = nn.Sequential(
            self.conv1, self.bn1, self.act1, self.conv2, self.bn2, self.act2,
            self.block1, self.block2, self.block3, self.block4, self.block5,
            self.block6, self.block7, self.block8, self.block9, self.block10,
            self.block11, self.block12, self.conv3, self.bn3, self.act3,
            self.conv4, self.bn4, self.act4
        )
        self.classifier = self.fc
   
    # MARK: Configure
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
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
        x = self.features(x)
        x = self.global_pool(x)
        if self.drop_rate:
            F.dropout(x, self.drop_rate, training=self.training)
        x = self.classifier(x)
        return x
