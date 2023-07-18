#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements ResNet models."""

from __future__ import annotations

__all__ = [
    "ResNet101", "ResNet152", "ResNet18", "ResNet34", "ResNet50",
    "ResNetBasicBlock", "ResNetBlock", "ResNetBottleneck",
]

from abc import ABC
from typing import Callable, Type

import torch
from torch import nn

from mon.coreml import layer, model
from mon.foundation import pathlib
from mon.globals import LAYERS, MODELS
from mon.vision.classify import base

_current_dir = pathlib.Path(__file__).absolute().parent


# region Module
@LAYERS.register()
class ResNetBasicBlock(layer.ConvLayerParsingMixin, nn.Module):
    
    expansion: int = 1
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : int              = 1,
        groups      : int              = 1,
        dilation    : int              = 1,
        base_width  : int              = 64,
        downsample  : nn.Module | None = None,
        norm        : Callable         = None,
        *args, **kwargs
    ):
        super().__init__()
        if norm is None:
            norm = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "'BasicBlock' only supports 'groups=1' and 'base_width=64'"
            )
        if dilation > 1:
            raise NotImplementedError(
                "dilation > 1 not supported in 'BasicBlock'"
            )
        # Both self.conv1 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = layer.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 3,
            stride       = stride,
            padding      = dilation,
            groups       = groups,
            bias         = False,
            dilation     = dilation,
        )
        self.bn1   = norm(out_channels)
        self.relu  = layer.ReLU(inplace=True)
        self.conv2 = layer.Conv2d(
            in_channels  = out_channels,
            out_channels = out_channels,
            kernel_size  = 3,
            stride       = stride,
            padding      = dilation,
            groups       = groups,
            bias         = False,
            dilation     = dilation,
        )
        self.bn2        = norm(out_channels)
        self.downsample = downsample
        self.stride     = stride
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.downsample is not None:
            x = self.downsample(x)
        y += x
        y  = self.relu(y)
        return y


@LAYERS.register()
class ResNetBottleneck(layer.ConvLayerParsingMixin, nn.Module):
    """Bottleneck in torchvision places the stride for down-sampling at 3x3
    convolution(self.conv2) while original implementation places the stride at
    the first 1x1 convolution(self.conv1) according to "Deep residual learning
    for image recognition" https://arxiv.org/abs/1512.03385. This variant is
    also known as ResNet V1.5 and improves accuracy according to
    https://ngc.nvidia.com/catalog/model-scripts/nvidia
    :resnet_50_v1_5_for_pytorch.
    """
    
    expansion: int = 4
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : int              = 1,
        groups      : int              = 1,
        dilation    : int              = 1,
        base_width  : int              = 64,
        downsample  : nn.Module | None = None,
        norm        : Callable         = None,
    ):
        super().__init__()
        if norm is None:
            norm = layer.BatchNorm2d
        width = int(out_channels * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = layer.Conv2d(
            in_channels  = in_channels,
            out_channels = width,
            kernel_size  = 1,
            stride       = stride,
            padding      = 0,
            dilation     = 1,
            groups       = 1,
            bias         = False
        )
        self.bn1 = norm(width)
        self.conv2 = layer.Conv2d(
            in_channels  = width,
            out_channels = width,
            kernel_size  = 3,
            stride       = stride,
            padding      = dilation,
            groups       = groups,
            bias         = False,
            dilation     = dilation,
        )
        self.bn2   = norm(width)
        self.conv3 = layer.Conv2d(
            in_channels  = width,
            out_channels = out_channels * self.expansion,
            kernel_size  = 1,
            stride       = stride,
            padding      = 0,
            dilation     = 1,
            groups       = 1,
            bias         = False
        )
        self.bn3        = norm(out_channels * self.expansion)
        self.relu       = layer.ReLU(inplace=True)
        self.downsample = downsample
        self.stride     = stride
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.bn3(y)
        if self.downsample is not None:
            x = self.downsample(x)
        y += x
        y  = self.relu(y)
        return y


@LAYERS.register()
class ResNetBlock(layer.LayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        block       : Type[ResNetBasicBlock | ResNetBottleneck],
        num_blocks  : int,
        in_channels : int,
        out_channels: int,
        stride      : int      = 1,
        groups      : int      = 1,
        dilation    : int      = 1,
        base_width  : int      = 64,
        dilate      : bool     = False,
        norm        : Callable = layer.BatchNorm2d,
    ):
        super().__init__()
        downsample    = None
        prev_dilation = dilation
        if dilate:
            dilation *= stride
            stride    = 1
        
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = torch.nn.Sequential(
                layer.Conv2d(
                    in_channels  = in_channels,
                    out_channels = out_channels * block.expansion,
                    kernel_size  = 1,
                    stride       = stride,
                    bias         = False,
                ),
                norm(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(
            block(
                in_channels  = in_channels,
                out_channels = out_channels,
                stride       = stride,
                groups       = groups,
                dilation     = prev_dilation,
                base_width   = base_width,
                downsample   = downsample,
                norm         = norm,
            )
        )
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    in_channels  = out_channels * block.expansion,
                    out_channels = out_channels,
                    stride       = 1,
                    groups       = groups,
                    dilation     = dilation,
                    base_width   = base_width,
                    downsample   = None,
                    norm         = norm,
                )
            )
        self.convs = torch.nn.Sequential(*layers)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.convs(x)
        return y
    
    @classmethod
    def parse_layer_args(cls, f: int, args: list, ch: list) -> tuple[list, list]:
        c1 = args[2]
        c2 = args[3]
        ch.append(c2)
        return args, ch

# endregion


# region Model

class ResNet(base.ImageClassificationModel, ABC):
    """ResNet.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {}
    map_weights = {}
    
    def init_weights(self, m: nn.Module):
        """Initialize model's weights."""
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                torch.nn.init.kaiming_normal_(m.conv.weight, mode="fan_out", nonlinearity="relu")
            else:
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif classname.find("GroupNorm") != -1:
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)

        zero_init_residual = self.cfg["zero_init_residual"]
        if zero_init_residual:
            if isinstance(m, ResNetBottleneck) and m.bn3.weight is not None:
                torch.nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, ResNetBottleneck) and m.bn2.weight is not None:
                torch.nn.init.constant_(m.bn2.weight, 0)
    
    def load_weights(self):
        """Load weights. It only loads the intersection layers of matching keys
        and shapes between the current model and weights.
        """
        if isinstance(self.weights, dict) \
            and self.weights["name"] in ["imagenet"]:
            state_dict = model.load_state_dict_from_path(
                model_dir=self.zoo_dir, **self.weights
            )
            model_state_dict = self.model.state_dict()
            """
            for k in self.model.state_dict().keys():
                print(f"\"{k}\": ")
            for k in state_dict.keys():
                print(f"\"{k}\"")
            """
            for k, v in state_dict.items():
                if "layer1" in k:
                    k = k.replace("layer1", "4.convs")
                elif "layer2" in k:
                    k = k.replace("layer2", "5.convs")
                elif "layer3" in k:
                    k = k.replace("layer3", "6.convs")
                elif "layer4" in k:
                    k = k.replace("layer4", "7.convs")
                else:
                    continue
                model_state_dict[k] = v
            model_state_dict["0.weight"]       = state_dict["conv1.weight"]
            model_state_dict["1.weight"]       = state_dict["bn1.weight"]
            model_state_dict["1.bias"]         = state_dict["bn1.bias"]
            model_state_dict["1.running_mean"] = state_dict["bn1.running_mean"]
            model_state_dict["1.running_var"]  = state_dict["bn1.running_var"]
            if self.weights["num_classes"] == self.num_classes:
                model_state_dict["9.linear.weight"] = state_dict["fc.weight"]
                model_state_dict["9.linear.bias"]   = state_dict["fc.bias"]
            self.model.load_state_dict(model_state_dict)
        else:
            super().load_weights()


@MODELS.register(name="resnet18")
class ResNet18(ResNet):
    """ResNet18.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/resnet18-f37072fd.pth",
            "file_name"  : "resnet18-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "resnet18.yaml",
            "name"   : "resnet",
            "variant": "resnet18"
        }
        super().__init__(*args, **kwargs)


@MODELS.register(name="resnet34")
class ResNet34(ResNet):
    """ResNet34.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/resnet34-b627a593.pth",
            "file_name"  : "resnet34-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "resnet34.yaml",
            "name"   : "resnet",
            "variant": "resnet34"
        }
        super().__init__(*args, **kwargs)


@MODELS.register(name="resnet50")
class ResNet50(ResNet):
    """ResNet50.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
            "file_name"  : "resnet50-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "resnet50.yaml",
            "name"   : "resnet",
            "variant": "resnet50"
        }
        super().__init__(*args, **kwargs)


@MODELS.register(name="resnet101")
class ResNet101(ResNet):
    """ResNet101.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/resnet101-cd907fc2.pth",
            "file_name"  : "resnet101-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "resnet101.yaml",
            "name"   : "resnet",
            "variant": "resnet101"
        }
        super().__init__(*args, **kwargs)


@MODELS.register(name="resnet152")
class ResNet152(ResNet):
    """ResNet152.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/resnet152-f82ba261.pth",
            "file_name"  : "resnet152-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "resnet152.yaml",
            "name"   : "resnet",
            "variant": "resnet152"
        }
        super().__init__(*args, **kwargs)
        
# endregion
