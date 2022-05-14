#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ResNeSt: Split-Attention Network, A New ResNet Variant.
https://github.com/zhanghang1989/ResNeSt
"""

from __future__ import annotations

import math
from typing import Optional
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.utils import _pair

from one.core import BACKBONES
from one.core import Callable
from one.core import Indexes
from one.core import Int2T
from one.core import ListOrTupleAnyT
from one.core import MODELS
from one.core import Padding2T
from one.core import Pretrained
from one.vision.image_classification.image_classifier import ImageClassifier

__all__ = [
    "ResNest",
    "ResNest50",
    "ResNest101",
    "ResNest200",
    "ResNest269",
]


# MARK: - Modules

class IBN(nn.Module):
    
    # MARK: Magic Functions
    
    def __init__(self, planes: int):
        super().__init__()
        half1 	  = int(planes / 2)
        self.half = half1
        half2 	  = planes - half1
        self.IN   = nn.InstanceNorm2d(half1, affine=True)
        self.BN   = nn.BatchNorm2d(half2)

    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        split = torch.split(x, self.half, 1)
        out1  = self.IN(split[0].contiguous())
        out2  = self.BN(split[1].contiguous())
        out   = torch.cat((out1, out2), 1)
        return out


class DropBlock2D(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
    
    
class SplAtConv2d(nn.Module):
    """Split-Attention Conv2d."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels 	: int,
        out_channels	: int,
        kernel_size 	: Int2T,
        stride			: Int2T 			  = (1, 1),
        padding			: Optional[Padding2T] = (0, 0),
        dilation		: Int2T 			  = (1, 1),
        groups			: int 				  = 1,
        bias			: bool 				  = True,
        radix			: int 			      = 2,
        reduction_factor: int 			      = 4,
        rectify			: bool 			      = False,
        rectify_avg		: bool 			      = False,
        norm_layer		: Callable 		      = None,
        dropblock_prob	: float			      = 0.0,
        *args, **kwargs
    ):
        super().__init__()
        padding			    = _pair(padding)
        self.rectify 	    = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg    = rectify_avg
        inter_channels 	    = max(in_channels*radix//reduction_factor, 32)
        self.radix 		    = radix
        self.cardinality    = groups
        self.out_channels   = out_channels
        self.dropblock_prob = dropblock_prob
     
        if self.rectify:
            """
            from rfconv import RFConv2d
            self.conv = RFConv2d(
                in_channels, out_channels*radix, kernel_size, stride, padding,
                dilation, groups=groups*radix, bias=bias,
                average_mode=rectify_avg, **kwargs
            )
            """
            raise NotImplementedError
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels * radix, kernel_size, stride,
                padding, dilation, groups=groups * radix, bias=bias, **kwargs
            )
            
        self.use_bn = norm_layer is not None
        self.bn0    = norm_layer(out_channels*radix)
        self.relu   = nn.ReLU(inplace=True)
        self.fc1 	= nn.Conv2d(out_channels, inter_channels, (1, 1),
                                groups=self.cardinality)
        self.bn1 	= norm_layer(inter_channels)
        self.fc2 	= nn.Conv2d(inter_channels, out_channels * radix, (1, 1),
                                groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch, channel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, channel // self.radix, dim=1)
            gap	    = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap).view((batch, self.radix, self.channels))
        if self.radix > 1:
            atten = F.softmax(atten, dim=1).view(batch, -1, 1, 1)
        else:
            atten = F.sigmoid(atten, dim=1).view(batch, -1, 1, 1)

        if self.radix > 1:
            atten = torch.split(atten, channel//self.radix, dim=1)
            out   = sum([att*split for (att, split) in zip(atten, splited)])
        else:
            out = atten * x
        return out.contiguous()


# noinspection PyMethodMayBeStatic
class GlobalAvgPool2d(nn.Module):
    """Global average pooling over the input's spatial dimensions."""
    
    # MARK: Magic Functions
    
    def __init__(self):
        super().__init__()
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.adaptive_avg_pool2d(x, 1).view(x.size()[0], -1)


class Bottleneck(nn.Module):
    """ResNet Bottleneck."""
    
    expansion = 4
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        inplanes        : int,
        planes          : int,
        stride          : Int2T   = 1,
        downsample       		   = None,
        radix           : int      = 1,
        cardinality     : int      = 1,
        bottleneck_width: int      = 64,
        avd				: bool     = False,
        avd_first		: bool     = False,
        dilation		: int 	   = 1,
        is_first		: bool     = False,
        rectified_conv	: bool     = False,
        rectify_avg		: bool     = False,
        norm_layer		: Callable = None,
        dropblock_prob	: float    = 0.0,
        last_gamma		: bool     = False,
        ibn			    : bool     = False
    ):
        super().__init__()

        group_width = int(planes * (bottleneck_width / 64.0)) * cardinality
        self.conv1  = nn.Conv2d(inplanes, group_width, kernel_size=(1, 1),
                                bias=False)
        if ibn:
            self.bn1 = IBN(group_width)
        else:
            self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix 	 		= radix
        self.avd 	   		= avd and (stride > 1 or is_first)
        self.avd_first 		= avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock2D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock2D(dropblock_prob, 3)
            self.dropblock3 = DropBlock2D(dropblock_prob, 3)

        if radix > 1:
            self.conv2 = SplAtConv2d(
                group_width, group_width,
                kernel_size    = 3,
                stride         = stride,
                padding        = dilation,
                dilation       = dilation,
                groups         = cardinality,
                bias           = False,
                radix          = radix,
                rectify        = rectified_conv,
                rectify_avg    = rectify_avg,
                norm_layer     = norm_layer,
                dropblock_prob = dropblock_prob
            )
        elif rectified_conv:
            """
            from rfconv import RFConv2d
            self.conv2 = RFConv2d(
                group_width, group_width,
                kernel_size  = 3,
                stride       = stride,
                padding 	 = dilation,
                dilation     = dilation,
                groups       = cardinality,
                bias         = False,
                average_mode = rectify_avg
            )
            self.bn2 = norm_layer(group_width)
            """
            raise NotImplementedError
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width,
                kernel_size = (3, 3),
                stride		= (stride, stride),
                padding		= dilation,
                dilation	= (dilation, dilation),
                groups		= cardinality,
                bias		= False
            )
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(
            group_width, planes * 4, kernel_size=(1, 1), bias=False)
        self.bn3 = norm_layer(planes*4)

        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu 		= nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation 	= dilation
        self.stride 	= stride

    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out 	 = self.conv1(x)
        out 	 = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 1:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


# MARK: - ResNest

cfgs = {
    "resnest50": {
        "block": Bottleneck, "layers": [3, 4, 6, 3], "radix": 2, "groups": 1,
        "bottleneck_width": 64, "dilated": False, "dilation": 1,
        "deep_stem": True, "stem_width": 32, "avg_down": True,
        "rectified_conv": False, "rectify_avg": False, "avd": True,
        "avd_first": False, "final_drop": 0.0, "dropblock_prob": 0,
        "last_gamma": False, "norm_layer": nn.BatchNorm2d
    },
    "resnest101": {
        "block": Bottleneck, "layers": [3, 4, 23, 3], "radix": 2, "groups": 1,
        "bottleneck_width": 64, "dilated": False, "dilation": 1,
        "deep_stem": True, "stem_width": 64, "avg_down": True,
        "rectified_conv": False, "rectify_avg": False, "avd": True,
        "avd_first": False, "final_drop": 0.0, "dropblock_prob": 0,
        "last_gamma": False, "norm_layer": nn.BatchNorm2d
    },
    "resnest200": {
        "block": Bottleneck, "layers": [3, 24, 36, 3], "radix": 2, "groups": 1,
        "bottleneck_width": 64, "dilated": False, "dilation": 1,
        "deep_stem": True, "stem_width": 64, "avg_down": True,
        "rectified_conv": False, "rectify_avg": False, "avd": True,
        "avd_first": False, "final_drop": 0.0, "dropblock_prob": 0,
        "last_gamma": False, "norm_layer": nn.BatchNorm2d
    },
    "resnest269": {
        "block": Bottleneck, "layers": [3, 30, 48, 8], "radix": 2, "groups": 1,
        "bottleneck_width": 64, "dilated": False, "dilation": 1,
        "deep_stem": True, "stem_width": 64, "avg_down": True,
        "rectified_conv": False, "rectify_avg": False, "avd": True,
        "avd_first": False, "final_drop": 0.0, "dropblock_prob": 0,
        "last_gamma": False, "norm_layer": nn.BatchNorm2d
    },
}


@MODELS.register(name="resnest")
@BACKBONES.register(name="resnest")
class ResNest(ImageClassifier):
    """ResNeSt: Split-Attention Network, A New ResNet Variant.
    
    Args:
        basename (str, optional):
            Model basename. Default: `resnest`.
        name (str, optional):
            Name of the model. Default: `resnest`.
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
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # Hyperparameters
        last_stride     : int,
        block           : nn.Module            = Bottleneck,
        layers          : ListOrTupleAnyT[int] = (3, 4, 6, 3),
        radix           : int                  = 2,
        groups          : int                  = 1,
        bottleneck_width: int                  = 64,
        dilated         : bool                 = False,
        dilation        : int                  = 1,
        deep_stem       : bool                 = True,
        stem_width      : int                  = 32,
        avg_down        : bool                 = True,
        rectified_conv  : bool                 = False,
        rectify_avg     : bool                 = False,
        avd             : bool                 = True,
        avd_first       : bool                 = False,
        final_drop      : float                = 0.0,
        dropblock_prob  : int                  = 0,
        last_gamma      : bool                 = False,
        norm_layer      : nn.Module            = nn.BatchNorm2d,
        # BaseModel's args
        basename   : Optional[str] = "resnest",
        name       : Optional[str] = "resnest",
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
        self.block			  = block
        # Class for the residual block. Options are BasicBlockV1, BottleneckV1.
        self.layers			  = layers
        # Numbers of layers in each block
        self.radix            = radix
        self.groups           = groups
        self.cardinality      = self.groups
        self.bottleneck_width = bottleneck_width
        self.dilated          = dilated
        # Applying dilation strategy to pretrained ResNet yielding a stride-8
        # model, typically used in Semantic Segmentation.
        self.dilation         = dilation
        self.deep_stem        = deep_stem
        self.stem_width       = stem_width
        self.inplanes         = self.stem_width * 2 if self.deep_stem else 64
        self.avg_down         = avg_down
        self.rectified_conv   = rectified_conv
        self.rectify_avg      = rectify_avg
        self.avd              = avd
        self.avd_first        = avd_first
        self.final_drop       = final_drop
        self.dropblock_prob   = dropblock_prob
        self.last_gamma       = last_gamma
        self.norm_layer       = norm_layer
        # Normalization layer used in backbone network
        self.frozen_stages    = 4
        
        if last_stride == 1:
            self.dilation = 2
        if self.rectified_conv:
            # from rfconv import RFConv2d
            # conv_layer = RFConv2d
            raise NotImplementedError
        else:
            conv_layer = nn.Conv2d
        conv_kwargs = ({"average_mode": self.rectify_avg}
                       if self.rectified_conv else {})
        
        # NOTE: Features
        if self.deep_stem:
            self.conv1 = nn.Sequential(
                conv_layer(3, self.stem_width, kernel_size=(3, 3),
                           stride=(2, 2), padding=1, bias=False, **conv_kwargs),
                self.norm_layer(self.stem_width),
                nn.ReLU(inplace=True),
                conv_layer(self.stem_width, self.stem_width, kernel_size=(3, 3),
                           stride=(1, 1), padding=1, bias=False, **conv_kwargs),
                self.norm_layer(self.stem_width),
                nn.ReLU(inplace=True),
                conv_layer(self.stem_width, self.stem_width * 2,
                           kernel_size=(3, 3), stride=(1, 1), padding=1,
                           bias=False, **conv_kwargs),
            )
        else:
            self.conv1 = conv_layer(
                3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3,
                bias=False, **conv_kwargs
            )
        
        self.bn1     = self.norm_layer(self.inplanes)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1  = self._make_layer(
            self.block, 64, self.layers[0], norm_layer=self.norm_layer,
            is_first=False
        )
        self.layer2  = self._make_layer(
            self.block, 128, self.layers[1], stride=2,
            norm_layer=self.norm_layer
        )
        if self.dilated or self.dilation == 4:
            self.layer3 = self._make_layer(
                self.block, 256, self.layers[2], stride=1, dilation=2,
                norm_layer=self.norm_layer, dropblock_prob=self.dropblock_prob
            )
            self.layer4 = self._make_layer(
                self.block, 512, self.layers[3], stride=last_stride, dilation=4,
                norm_layer=self.norm_layer, dropblock_prob=self.dropblock_prob
            )
        elif self.dilation == 2:
            self.layer3 = self._make_layer(
                self.block, 256, self.layers[2], stride=2, dilation=1,
                norm_layer=self.norm_layer, dropblock_prob=self.dropblock_prob
            )
            self.layer4 = self._make_layer(
                self.block, 512, self.layers[3], stride=last_stride, dilation=2,
                norm_layer=self.norm_layer, dropblock_prob=self.dropblock_prob
            )
        else:
            self.layer3 = self._make_layer(
                self.block, 256, self.layers[2], stride=2,
                norm_layer=self.norm_layer, dropblock_prob=self.dropblock_prob
            )
            self.layer4 = self._make_layer(
                self.block, 512, self.layers[3], stride=last_stride,
                norm_layer=self.norm_layer, dropblock_prob=self.dropblock_prob
            )
        self.avgpool = GlobalAvgPool2d()
        self.drop    = nn.Dropout(self.final_drop) if self.final_drop > 0.0 else None
        
        # NOTE: Classifier
        num_features = 512 * self.block.expansion
        self.fc 	 = self.create_classifier(num_features, self.num_classes)
        
        # NOTE: Load Pretrained
        if self.pretrained:
            self.load_pretrained()
        else:
            self.init_weights(self.norm_layer)
        
        # NOTE: Alias
        self.features = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.maxpool, self.layer1,
            self.layer2, self.layer3, self.layer4
        )
        self.classifier = self.fc
        
    # MARK: Configure
    
    def make_layer(
        self,
        block   	  : Type[Bottleneck],
        planes 	      : int,
        blocks  	  : int,
        stride  	  : int  			   = 1,
        dilation	  : bool 			   = False,
        norm_layer	  : Optional[Callable] = None,
        dropblock_prob: float			   = 0.0,
        is_first	  : bool  			   = True
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(
                        kernel_size=stride, stride=stride, ceil_mode=True,
                        count_include_pad=False
                    ))
                else:
                    down_layers.append(nn.AvgPool2d(
                        kernel_size=1, stride=1, ceil_mode=True,
                        count_include_pad=False
                    ))
                down_layers.append(nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=(1, 1), stride=(1, 1), bias=False
                ))
            else:
                down_layers.append(nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=(1, 1), stride=(stride, stride), bias=False
                ))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(
                self.inplanes, planes, stride,
                downsample       = downsample,
                radix            = self.radix,
                cardinality      = self.cardinality,
                bottleneck_width = self.bottleneck_width,
                avd              = self.avd,
                avd_first        = self.avd_first,
                dilation         = 1,
                is_first         = is_first,
                rectified_conv   = self.rectified_conv,
                rectify_avg      = self.rectify_avg,
                norm_layer       = norm_layer,
                dropblock_prob   = dropblock_prob,
                last_gamma       = self.last_gamma
            ))
        elif dilation == 4:
            layers.append(block(
                self.inplanes, planes, stride,
                downsample       = downsample,
                radix            = self.radix,
                cardinality      = self.cardinality,
                bottleneck_width = self.bottleneck_width,
                avd              = self.avd,
                avd_first        = self.avd_first,
                dilation         = 2,
                is_first         = is_first,
                rectified_conv   = self.rectified_conv,
                rectify_avg      = self.rectify_avg,
                norm_layer       = norm_layer,
                dropblock_prob   = dropblock_prob,
                last_gamma       = self.last_gamma
            ))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.inplanes, planes,
                radix            = self.radix,
                cardinality      = self.cardinality,
                bottleneck_width = self.bottleneck_width,
                avd              = self.avd,
                avd_first        = self.avd_first,
                dilation         = dilation,
                rectified_conv   = self.rectified_conv,
                rectify_avg      = self.rectify_avg,
                norm_layer       = norm_layer,
                dropblock_prob   = dropblock_prob,
                last_gamma       = self.last_gamma
            ))

        return nn.Sequential(*layers)

    @staticmethod
    def create_classifier(
        num_features: int, num_classes: Optional[int]
    ) -> nn.Module:
        if num_classes and num_classes > 0:
            classifier = nn.Linear(num_features, num_classes)
        else:
            classifier = nn.Identity()
        return classifier
    
    def init_weights(self, norm_layer):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
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
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# MARK: - ResNest50

@MODELS.register(name="resnest50")
@BACKBONES.register(name="resnest50")
class ResNest50(ResNest):
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        last_stride: int,
        name       : Optional[str] = "resnest50",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["resnest50"] | kwargs
        super().__init__(
            last_stride = last_stride,
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )
    

# MARK: - ResNest101

@MODELS.register(name="resnest101")
@BACKBONES.register(name="resnest101")
class ResNest101(ResNest):

    # MARK: Magic Functions
    
    def __init__(
        self,
        last_stride: int,
        name       : Optional[str] = "resnest101",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["resnest101"] | kwargs
        super().__init__(
            last_stride = last_stride,
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )
        

# MARK: - ResNest200

@MODELS.register(name="resnest200")
@BACKBONES.register(name="resnest200")
class ResNest200(ResNest):

    # MARK: Magic Functions
    
    def __init__(
        self,
        last_stride: int,
        name       : Optional[str] = "resnest200",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["resnest200"] | kwargs
        super().__init__(
            last_stride = last_stride,
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - ResNest269

@MODELS.register(name="resnest269")
@BACKBONES.register(name="resnest269")
class ResNest269(ResNest):
 
    # MARK: Magic Functions
    
    def __init__(
        self,
        last_stride: int,
        name       : Optional[str] = "resnest269",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["resnest269"] | kwargs
        super().__init__(
            last_stride = last_stride,
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )
