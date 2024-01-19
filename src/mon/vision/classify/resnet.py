#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements ResNet models."""

from __future__ import annotations

__all__ = [
    "ResNeXt101_32X8D",
    "ResNeXt101_64X4D",
    "ResNeXt50_32X4D",
    "ResNet",
    "ResNet101",
    "ResNet152",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "WideResNet101",
    "WideResNet50",
]

from abc import ABC
from typing import Type, Any

import torch

from mon.globals import MODELS
from mon.vision import core, nn
from mon.vision.classify import base
from mon.vision.nn import _callable

console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region Module

class Conv3x3(nn.Conv2d):
    """3x3 convolution with padding"""
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : int = 1,
        groups      : int = 1,
        dilation    : int = 1,
    ):
        super().__init__(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 3,
            stride       = stride,
            padding      = dilation,
            groups       = groups,
            bias         = False,
            dilation     = dilation,
        )


class Conv1x1(nn.Conv2d):
    """1x1 convolution"""
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : int = 1,
    ):
        super().__init__(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = stride,
            bias         = False,
        )


class BasicBlock(nn.Module):
    
    expansion: int = 1

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : int = 1,
        downsample  : nn.Module | None = None,
        groups      : int = 1,
        base_width  : int = 64,
        dilation    : int = 1,
        norm_layer  : _callable = None,
        *args, **kwargs
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(":class:`BasicBlock` only supports :math:`groups=1` and :math:`base_width=64`")
        if dilation > 1:
            raise NotImplementedError(":math:`dilation > 1` not supported in :class:`BasicBlock`")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1      = Conv3x3(in_channels, out_channels, stride)
        self.bn1        = norm_layer(out_channels)
        self.relu       = nn.ReLU(inplace=True)
        self.conv2      = Conv3x3(out_channels, out_channels)
        self.bn2        = norm_layer(out_channels)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        x += identity
        y  = self.relu(x)
        return y
    

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : int = 1,
        downsample  : nn.Module | None = None,
        groups      : int = 1,
        base_width  : int = 64,
        dilation    : int = 1,
        norm_layer  : _callable = None,
        *args, **kwargs
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(out_channels * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1      = Conv1x1(in_channels, width)
        self.bn1        = norm_layer(width)
        self.conv2      = Conv3x3(width, width, stride, groups, dilation)
        self.bn2        = norm_layer(width)
        self.conv3      = Conv1x1(width, out_channels * self.expansion)
        self.bn3        = norm_layer(out_channels * self.expansion)
        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        x += identity
        y  = self.relu(x)
        return y

# endregion


# region ResNet

class ResNet(base.ImageClassificationModel, ABC):
    """ResNet.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {}
    
    def __init__(
        self,
        block                       : Type[BasicBlock | Bottleneck],
        layers                      : list[int],
        channels                    : int  = 3,
        num_classes                 : int  = 1000,
        zero_init_residual          : bool = False,
        groups                      : int  = 1,
        width_per_group             : int  = 64,
        replace_stride_with_dilation: list[bool] | None = None,
        norm_layer                  : _callable = None,
        weights                     : Any       = None,
        name                        : str       = "resnet",
        *args, **kwargs
    ):
        super().__init__(
            channels    = channels,
            num_classes = num_classes,
            weights     = weights,
            name        = name,
            *args, **kwargs
        )
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.num_channels = 64
        self.dilation     = 1
        if replace_stride_with_dilation is None:
            # Each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                f":param:`replace_stride_with_dilation` should be ``None`` "
                f"or a 3-element :class:`tuple`, but got {replace_stride_with_dilation}"
            )
        self.groups     = groups
        self.base_width = width_per_group
        self.conv1      = nn.Conv2d(self.channels, self.num_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1        = norm_layer(self.num_channels)
        self.relu       = nn.ReLU(inplace=True)
        self.maxpool    = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1     = self._make_layer(block, 64,  layers[0])
        self.layer2     = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3     = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4     = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool    = nn.AdaptiveAvgPool2d((1, 1))
        self.fc         = nn.Linear(512 * block.expansion, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch, so that the
        # residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    torch.nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    torch.nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        
    def _make_layer(
        self,
        block : Type[BasicBlock | Bottleneck],
        planes: int,
        blocks: int,
        stride: int  = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer        = self._norm_layer
        downsample        = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.num_channels != planes * block.expansion:
            downsample = nn.Sequential(
                Conv1x1(self.num_channels, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                in_channels  = self.num_channels,
                out_channels = planes,
                stride       = stride,
                downsample   = downsample,
                groups       = self.groups,
                base_width   = self.base_width,
                dilation     = previous_dilation,
                norm_layer   = norm_layer,
            )
        )
        self.num_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    in_channels  = self.num_channels,
                    out_channels = planes,
                    groups       = self.groups,
                    base_width   = self.base_width,
                    dilation     = self.dilation,
                    norm_layer   = norm_layer,
                )
            )

        return nn.Sequential(*layers)
    
    def forward_once(
        self,
        input    : torch.Tensor,
        profile  : bool = False,
        out_index: int  = -1,
        *args, **kwargs
    ) -> torch.Tensor:
        x = input
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.fc(x)
        return y
    

@MODELS.register(name="resnet18")
class ResNet18(ResNet):
    """ResNet-18 from `Deep Residual Learning for Image Recognition
    <https://arxiv.org/abs/1512.03385>`__.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/resnet18-f37072fd.pth",
            "path"       : "resnet18-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "resnet",
        variant: str = "resnet18",
        *args, **kwargs
    ):
        super().__init__(
            block   = BasicBlock,
            layers  = [2, 2, 2, 2],
            name    = name,
            variant = variant,
            *args, **kwargs
        )


@MODELS.register(name="resnet34")
class ResNet34(ResNet):
    """ResNet-34 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/resnet34-b627a593.pth",
            "path"       : "resnet34-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }

    def __init__(
        self,
        name   : str = "resnet",
        variant: str = "resnet34",
        *args, **kwargs
    ):
        super().__init__(
            block   = BasicBlock,
            layers  = [3, 4, 6, 3],
            name    = name,
            variant = variant,
            *args, **kwargs
        )


@MODELS.register(name="resnet50")
class ResNet50(ResNet):
    """ResNet-50 from `Deep Residual Learning for Image Recognition
    <https://arxiv.org/abs/1512.03385>`__.

    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
            "path"       : "resnet50-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-v2": {
            "url"        : "https://download.pytorch.org/models/resnet50-0676ba61.pth",
            "path"       : "resnet50-imagenet1k-v2.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "resnet",
        variant: str = "resnet50",
        *args, **kwargs
    ):
        super().__init__(
            block   = Bottleneck,
            layers  = [3, 4, 6, 3],
            name    = name,
            variant = variant,
            *args, **kwargs
        )


@MODELS.register(name="resnet101")
class ResNet101(ResNet):
    """ResNet-101 from `Deep Residual Learning for Image Recognition
    <https://arxiv.org/abs/1512.03385>`__.

    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/resnet101-63fe2227.pth",
            "path"       : "resnet101-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-v2": {
            "url"        : "https://download.pytorch.org/models/resnet101-cd907fc2.pth",
            "path"       : "resnet101-imagenet1k-v2.pth",
            "num_classes": 1000,
            "map": {},
        },
    }

    def __init__(
        self,
        name   : str = "resnet",
        variant: str = "resnet101",
        *args, **kwargs
    ):
        super().__init__(
            block   = Bottleneck,
            layers  = [3, 4, 23, 3],
            name    = name,
            variant = variant,
            *args, **kwargs
        )


@MODELS.register(name="resnet152")
class ResNet152(ResNet):
    """ResNet-152 from `Deep Residual Learning for Image Recognition
    <https://arxiv.org/abs/1512.03385>`__.

    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/resnet152-394f9c45.pth",
            "path"       : "resnet152-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-v2": {
            "url"        : "https://download.pytorch.org/models/resnet152-f82ba261.pth",
            "path"       : "resnet152-imagenet1k-v2.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "resnet",
        variant: str = "resnet152",
        *args, **kwargs
    ):
        super().__init__(
            block   = Bottleneck,
            layers  = [3, 8, 36, 3],
            name    = name,
            variant = variant,
            *args, **kwargs
        )
        
# endregion


# region ResNeXt

@MODELS.register(name="resnext50_32x4d")
class ResNeXt50_32X4D(ResNet):
    """ResNeXt-50 32x4d model from `Aggregated Residual Transformation for Deep
    Neural Networks <https://arxiv.org/abs/1611.05431>`__.

    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
            "path"       : "resnext50_32x4d-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-v2": {
            "url"        : "https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth",
            "path"       : "resnext50_32x4d-imagenet1k-v2.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "resnet",
        variant: str = "resnext50_32x4d",
        *args, **kwargs
    ):
        super().__init__(
            block           = Bottleneck,
            layers          = [3, 4, 6, 3],
            groups          = 32,
            width_per_group = 4,
            name            = name,
            variant         = variant,
            *args, **kwargs
        )


@MODELS.register(name="resnext101_32x8d")
class ResNeXt101_32X8D(ResNet):
    """ResNeXt-101 32x8d model from `Aggregated Residual Transformation for
    Deep Neural Networks <https://arxiv.org/abs/1611.05431>`__.

    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
            "path"       : "resnext101_32x8d-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-v2": {
            "url"        : "https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth",
            "path"       : "resnext101_32x8d-imagenet1k-v2.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "resnet",
        variant: str = "resnext101_32x8d",
        *args, **kwargs
    ):
        super().__init__(
            block           = Bottleneck,
            layers          = [3, 4, 23, 3],
            groups          = 32,
            width_per_group = 8,
            name            = name,
            variant         = variant,
            *args, **kwargs
        )


@MODELS.register(name="resnext101_64x4d")
class ResNeXt101_64X4D(ResNet):
    """ResNeXt-101 32x8d model from `Aggregated Residual Transformation for
    Deep Neural Networks <https://arxiv.org/abs/1611.05431>`__.

    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth",
            "path"       : "resnext101_64x4d-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "resnet",
        variant: str = "resnext101_64x4d",
        *args, **kwargs
    ):
        super().__init__(
            block           = Bottleneck,
            layers          = [3, 4, 23, 3],
            groups          = 64,
            width_per_group = 4,
            name            = name,
            variant         = variant,
            *args, **kwargs
        )
        
# endregion


# region WideResNet

@MODELS.register(name="wide_resnet50")
class WideResNet50(ResNet):
    """Wide ResNet-50-2 model from `Wide Residual Networks
    <https://arxiv.org/abs/1605.07146>`__.

    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
            "path"       : "wide_resnet50-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-v2": {
            "url"        : "https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth",
            "path"       : "wide_resnet50-imagenet1k-v2.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "resnet",
        variant: str = "wide_resnet50",
        *args, **kwargs
    ):
        super().__init__(
            block           = Bottleneck,
            layers          = [3, 4, 6, 3],
            width_per_group = 64 * 2,
            name            = name,
            variant         = variant,
            *args, **kwargs
        )


@MODELS.register(name="wide_resnet101")
class WideResNet101(ResNet):
    """Wide ResNet-101-2 model from `Wide Residual Networks
    <https://arxiv.org/abs/1605.07146>`__.

    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
            "path"       : "wide_resnet101-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-v2": {
            "url"        : "https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth",
            "path"       : "wide_resnet101-imagenet1k-v2.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "resnet",
        variant: str = "wide_resnet101",
        *args, **kwargs
    ):
        super().__init__(
            block           = Bottleneck,
            layers          = [3, 4, 23, 3],
            width_per_group = 64 * 2,
            name            = name,
            variant         = variant,
            *args, **kwargs
        )
        
# endregion
