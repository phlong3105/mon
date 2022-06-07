#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pytorch impl of Aligned Xception-41, -65, -71.

This is a correct, from scratch impl of Aligned Xception (Deeplab) models
compatible with TF weights at: https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md
"""

from __future__ import annotations

from functools import partial
from typing import Optional
from typing import Union

import torch.nn as nn
from torch import Tensor

from one.core import BACKBONES
from one.core import Callable
from one.core import IMAGENET_INCEPTION_MEAN
from one.core import IMAGENET_INCEPTION_STD
from one.core import Indexes
from one.core import Int2T
from one.core import Int3T
from one.core import MODELS
from one.core import Padding2T
from one.core import Pretrained
from one.core import to_2tuple
from one.core import to_3tuple
from one.nn import Classifier
from one.nn import ConvBnAct
from one.nn import create_conv2d
from one.vision.image_classification.image_classifier import ImageClassifier

__all__ = [
    "Xception41",
    "Xception65",
    "Xception71",
    "XceptionAligned",
    "XceptionModule",
]


# MARK: - Custom Modules

class SeparableConv2d(nn.Module):
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        inplanes   : int,
        planes     : int,
        kernel_size: Int2T              = 3,
        stride     : Int2T              = 1,
        padding    : Optional[Padding2T] = "",
        dilation   : Int2T              = 1,
        act_layer  : Callable            = nn.ReLU,
        norm_layer : Callable            = nn.BatchNorm2d,
    ):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride 		= to_2tuple(stride)
        dilation    = to_2tuple(dilation)

        # depthwise convolution
        self.conv_dw = create_conv2d(
            inplanes, inplanes, kernel_size, stride=stride, padding=padding,
            dilation=dilation, depthwise=True
        )
        self.bn_dw = norm_layer(inplanes)
        if act_layer is not None:
            self.act_dw = act_layer(inplace=True)
        else:
            self.act_dw = None

        # pointwise convolution
        self.conv_pw = create_conv2d(inplanes, planes, kernel_size=1)
        self.bn_pw   = norm_layer(planes)
        if act_layer is not None:
            self.act_pw = act_layer(inplace=True)
        else:
            self.act_pw = None
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_dw(x)
        x = self.bn_dw(x)
        if self.act_dw is not None:
            x = self.act_dw(x)
        x = self.conv_pw(x)
        x = self.bn_pw(x)
        if self.act_pw is not None:
            x = self.act_pw(x)
        return x


class XceptionModule(nn.Module):
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels    : int,
        out_channels   : int,
        stride         : int                      = 1,
        dilation       : int                      = 1,
        padding        : Union[str, Int2T, None] = "",
        start_with_relu: bool                     = True,
        no_skip        : bool                     = False,
        act_layer      : Callable                 = nn.ReLU,
        norm_layer     : Callable                 = None
    ):
        super().__init__()
        out_channels      = to_3tuple(out_channels)
        self.in_channels  = in_channels
        self.out_channels = out_channels[-1]
        self.no_skip      = no_skip
        
        if not no_skip and (self.out_channels != self.in_channels or
                            stride != 1):
            self.shortcut = ConvBnAct(
                in_channels, self.out_channels, 1, stride=stride,
                norm_layer=norm_layer, act_layer=None
            )
        else:
            self.shortcut = None

        separable_act_layer = None if start_with_relu else act_layer
        self.stack          = nn.Sequential()
        for i in range(3):
            if start_with_relu:
                self.stack.add_module(f"act{i + 1}", nn.ReLU(inplace=i > 0))
            self.stack.add_module(
                f"conv{i + 1}", SeparableConv2d(
                    in_channels, out_channels[i], 3,
                    stride     = stride if i == 2 else 1,
                    dilation   = dilation,
                    padding    = padding,
                    act_layer  = separable_act_layer,
                    norm_layer = norm_layer
                )
            )
            in_channels = out_channels[i]

    def forward(self, x: Tensor) -> Tensor:
        skip = x
        x    = self.stack(x)
        if self.shortcut is not None:
            skip = self.shortcut(skip)
        if not self.no_skip:
            x = x + skip
        return x


# MARK: - XceptionAligned

cfgs = {
    "xception41": {
        "block_cfg": [
            # Entry flow
            {"in_chs": 64, "out_chs": 128, "stride": 2},
            {"in_chs": 128, "out_chs": 256, "stride": 2},
            {"in_chs": 256, "out_chs": 728, "stride": 2},
            # Middle flow
            *([{"in_chs": 728, "out_chs": 728, "stride": 1}] * 8),
            # Exit flow
            {"in_chs": 728, "out_chs": (728, 1024, 1024), "stride": 2},
            {
                "in_chs": 1024, "out_chs": (1536, 1536, 2048), "stride": 1,
                "no_skip": True, "start_with_relu": False
            },
        ],
        "in_channels": 3,
        "output_stride": 32,
        "act_layer": nn.ReLU,
        "norm_layer": partial(nn.BatchNorm2d, eps=0.001, momentum=0.1),
        "drop_rate": 0.0,
        "global_pool": "avg",
        "input_size": (3, 299, 299),
        "pool_size": (10, 10),
        "crop_pct": 0.903,
        "interpolation": "bicubic",
        "mean": IMAGENET_INCEPTION_MEAN,
        "std": IMAGENET_INCEPTION_STD,
        "first_conv": "stem.0.conv",
        "classifier": "head.fc"
    },
    
    "xception65": {
        "block_cfg": [
            # Entry flow
            {"in_chs": 64, "out_chs": 128, "stride": 2},
            {"in_chs": 128, "out_chs": 256, "stride": 2},
            {"in_chs": 256, "out_chs": 728, "stride": 2},
            # Middle flow
            *([{"in_chs": 728, "out_chs": 728, "stride": 1}] * 16),
            # Exit flow
            {"in_chs": 728, "out_chs": (728, 1024, 1024), "stride": 2},
            {
                "in_chs": 1024, "out_chs": (1536, 1536, 2048), "stride": 1,
                "no_skip": True, "start_with_relu": False
            },
        ],
        "in_channels": 3,
        "output_stride": 32,
        "act_layer": nn.ReLU,
        "norm_layer": partial(nn.BatchNorm2d, eps=0.001, momentum=0.1),
        "drop_rate": 0.0,
        "global_pool": "avg",
        "input_size": (3, 299, 299),
        "pool_size": (10, 10),
        "crop_pct": 0.903,
        "interpolation": "bicubic",
        "mean": IMAGENET_INCEPTION_MEAN,
        "std": IMAGENET_INCEPTION_STD,
        "first_conv": "stem.0.conv",
        "classifier": "head.fc"
    },
    
    "xception71": {
        "block_cfg": [
            # Entry flow
            {"in_chs": 64, "out_chs": 128, "stride": 2},
            {"in_chs": 128, "out_chs": 256, "stride": 1},
            {"in_chs": 256, "out_chs": 256, "stride": 2},
            {"in_chs": 256, "out_chs": 728, "stride": 1},
            {"in_chs": 728, "out_chs": 728, "stride": 2},
            # Middle flow
            *([{"in_chs": 728, "out_chs": 728, "stride": 1}] * 16),
            # Exit flow
            {"in_chs": 728, "out_chs": (728, 1024, 1024), "stride": 2},
            {
                "in_chs": 1024, "out_chs": (1536, 1536, 2048), "stride": 1,
                "no_skip": True, "start_with_relu": False
            },
        ],
        "in_channels": 3,
        "output_stride": 32,
        "act_layer": nn.ReLU,
        "norm_layer": partial(nn.BatchNorm2d, eps=0.001, momentum=0.1),
        "drop_rate": 0.0,
        "global_pool": "avg",
        "input_size": (3, 299, 299),
        "pool_size": (10, 10),
        "crop_pct": 0.903,
        "interpolation": "bicubic",
        "mean": IMAGENET_INCEPTION_MEAN,
        "std": IMAGENET_INCEPTION_STD,
        "first_conv": "stem.0.conv",
        "classifier": "head.fc"
    }
}


@MODELS.register(name="xception_aligned")
@BACKBONES.register(name="xception_aligned")
class XceptionAligned(ImageClassifier):
    """Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    
    Args:
        basename (str, optional):
            Model basename. Default: `xception`.
        name (str, optional):
            Name of the model. Default: `xception_aligned`.
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
    
    Attributes:
        cfg (Config, optional):
            Config to build the model's layers.
            - If `str`, use the corresponding config from the predefined
              config dict. This is used to build the model dynamically.
            - If a file or filepath, it leads to the external config file that
              is used to build the model dynamically.
            - If `list`, then each element in the list is the corresponding
              config for each layer in the model. This is used to build the
              model dynamically.
            - If `dict`, it usually contains the hyperparameters used to
              build the model manually in the code.
            - If `None`, then you should manually define the model.
            Remark: You have 5 ways to build the model, so choose the style
            that you like.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # Hyperparameters
        block_cfg       : list,
        in_channels     : int       = 3,
        output_stride   : int       = 32,
        act_layer       : nn.Module = nn.ReLU,
        norm_layer                  = partial(nn.BatchNorm2d, eps=0.001, momentum=0.1),
        drop_rate       : float     = 0.0,
        global_pool     : str       = "avg",
        input_size      : Int3T      = (3, 299, 299),
        pool_size       : Int2T      = (10, 10),
        crop_pct        : float     = 0.903,
        interpolation   : str       = "bicubic",
		mean            : float     = IMAGENET_INCEPTION_MEAN,
		std             : float     = IMAGENET_INCEPTION_STD,
		first_conv      : str       = "stem.0.conv",
		classifier      : bool      = "head.fc",
        # BaseModel's args
        basename   : Optional[str] = "xception",
        name       : Optional[str] = "xception_aligned",
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
        self.block_cfg     = block_cfg
        self.in_channels   = in_channels
        self.output_stride = output_stride
        self.act_layer     = act_layer
        self.norm_layer    = norm_layer
        self.drop_rate     = drop_rate
        self.global_pool   = global_pool
        if self.output_stride not in (8, 16, 32):
            raise ValueError()
        
        # NOTE: Features
        layer_args = dict(act_layer=self.act_layer, norm_layer=self.norm_layer)
        self.stem  = nn.Sequential(*[
            ConvBnAct(self.in_channels, 32, kernel_size=3, stride=2, **layer_args),
            ConvBnAct(32, 64, kernel_size=3, stride=1, **layer_args)
        ])
        
        curr_dilation     = 1
        curr_stride       = 2
        self.feature_info = []
        self.blocks       = nn.Sequential()
        for i, b in enumerate(self.block_cfg):
            b["dilation"] = curr_dilation
            if b["stride"] > 1:
                self.feature_info += [dict(
                    num_chs=to_3tuple(b["out_chs"])[-2], reduction=curr_stride,
                    module=f"blocks.{i}.stack.act3"
                )]
                next_stride = curr_stride * b["stride"]
                if next_stride > self.output_stride:
                    curr_dilation *= b["stride"]
                    b["stride"]    = 1
                else:
                    curr_stride = next_stride
            self.blocks.add_module(str(i), XceptionModule(**b, **layer_args))
            self.num_features = self.blocks[-1].out_channels

        self.feature_info += [dict(
            num_chs=self.num_features, reduction=curr_stride,
            module="blocks." + str(len(self.blocks) - 1)
        )]

        # NOTE: Classifier
        self.head = Classifier(
            in_channels=self.num_features, num_classes=num_classes,
            pool_type=self.global_pool, drop_rate=self.drop_rate
        )
 
        # NOTE: Load Pretrained
        if self.pretrained:
            self.load_pretrained()

        # NOTE: Alias
        self.features   = nn.Sequential(*self.stem, *self.blocks)
        self.classifier = self.head.fc
  
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
        # x = self.stem(x)
        # x = self.blocks(x)
        x = self.features(x)
        x = self.head(x)
        return x


# MARK: - Xception41

@MODELS.register(name="xception41")
@BACKBONES.register(name="xception41")
class Xception41(XceptionAligned):
    """Modified Aligned Xception-41.
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_xception_41-e6439c97.pth",
            file_name="xception41_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "xception41",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["xception41"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - Xception65

@MODELS.register(name="xception65")
@BACKBONES.register(name="xception65")
class Xception65(XceptionAligned):
    """Modified Aligned Xception-65.
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_xception_65-c9ae96e8.pth",
            file_name="xception65_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "xception65",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["xception65"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - Xception71

@MODELS.register(name="xception71")
@BACKBONES.register(name="xception71")
class Xception71(XceptionAligned):
    """Modified Aligned Xception-71.
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_xception_71-8eec7df1.pth",
            file_name="xception71_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "xception71",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["xception71"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )
