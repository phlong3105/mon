#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements ShuffleNetV2 models."""

from __future__ import annotations

__all__ = [
    "ShuffleNetV2",
    "ShuffleNetV2_X1_0",
    "ShuffleNetV2_X1_5",
    "ShuffleNetV2_X2_0",
    "ShuffleNetV2_x0_5",
]

from abc import ABC
from typing import Any

import torch

from mon import core, nn
from mon.core import _callable
from mon.globals import MODELS, Scheme
from mon.vision.classify import base

console = core.console


# region Module

def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    # Reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # Flatten
    x = x.view(batch_size, num_channels, height, width)
    return x


class InvertedResidual(nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : int,
        *args, **kwargs
    ):
        super().__init__()
        if not (1 <= stride <= 3):
            raise ValueError("Illegal stride value.")
        self.stride = stride
        
        branch_features = out_channels // 2
        if (self.stride == 1) and (in_channels != branch_features << 1):
            raise ValueError(
                f"Invalid combination of :param:`stride` {stride}, "
                f":param:`in_channels` {in_channels} and "
                f":param:`out_channels` {out_channels} values. "
                f"If :math:`stride == 1` then :param:`in_channels` should be "
                f"equal to :math:`out_channels // 2 << 1`."
            )

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(in_channels, in_channels, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels  = in_channels if (self.stride > 1) else branch_features,
                out_channels = branch_features,
                kernel_size  = 1,
                stride       = 1,
                padding      = 0,
                bias         = False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(
        in_channels : int,
        out_channels: int,
        kernel_size : int,
        stride      : int  = 1,
        padding     : int  = 0,
        bias        : bool = False
    ) -> nn.Conv2d:
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=in_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            y      = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            y = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        y = channel_shuffle(y, 2)
        return y

# endregion


# region Model

class ShuffleNetV2(base.ImageClassificationModel, ABC):
    """ShuffleNetV2.
    
    See Also: :class:`base.ImageClassificationModel`
    """
    
    _scheme: list[Scheme] = [Scheme.SUPERVISED]
    _zoo   : dict = {}

    def __init__(
        self,
        stages_repeats     : list[int],
        stages_out_channels: list[int],
        channels           : int       = 3,
        num_classes        : int       = 1000,
        inverted_residual  : _callable = InvertedResidual,
        weights            : Any       = None,
        *args, **kwargs
    ):
        super().__init__(
            channels    = channels,
            num_classes = num_classes,
            weights     = weights,
            *args, **kwargs
        )
        if len(stages_repeats) != 3:
            raise ValueError("Expected :param:`stages_repeats` as :class:`list` of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("Expected :param:`stages_out_channels` as :class:`list` of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        input_channels  = self.channels
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels
        self.maxpool   = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        stage_names = [f"stage{i}" for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(output_channels, self.num_classes)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self._init_weights)
    
    def _init_weights(self, model: nn.Module):
        pass
    
    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> torch.Tensor:
        x = input
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        y = self.fc(x)
        return y


@MODELS.register(name="shufflenetv2_x0_5")
class ShuffleNetV2_x0_5(ShuffleNetV2):
    """ShuffleNetV2 architecture with 0.5x output channels, as described in
    `ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    <https://arxiv.org/abs/1807.11164>`__.
    
    See Also: :class:`ShuffleNetV2`
    """
    
    _zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",
            "path"       : "shufflenetv2/shufflenetv2_x0_5_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(self, args, **kwargs):
        super().__init__(
            name                = "shufflenetv2_x0_5",
            stages_repeats      = [4, 8, 4],
            stages_out_channels = [24, 48, 96, 192, 1024],
            *args, **kwargs
        )


@MODELS.register(name="shufflenetv2_x1_0")
class ShuffleNetV2_X1_0(ShuffleNetV2):
    """ShuffleNetV2 architecture with 1.0x output channels, as described in
    `ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    <https://arxiv.org/abs/1807.11164>`__.
    
    See Also: :class:`ShuffleNetV2`
    """
    
    _zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",
            "path"       : "shufflenetv2/shufflenetv2_x1_0_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(self, args, **kwargs):
        super().__init__(
            name                = "shufflenetv2_x1",
            stages_repeats      = [4, 8, 4],
            stages_out_channels = [24, 116, 232, 464, 1024],
            *args, **kwargs
        )


@MODELS.register(name="shufflenetv2_x1_5")
class ShuffleNetV2_X1_5(ShuffleNetV2):
    """ShuffleNetV2 architecture with 1.5x output channels, as described in
    `ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    <https://arxiv.org/abs/1807.11164>`__.
    
    See Also: :class:`ShuffleNetV2`
    """
    
    _zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/shufflenetv2_x1_5-3c479a10.pth",
            "path"       : "shufflenetv2/shufflenetv2_x1_5_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(self, args, **kwargs):
        super().__init__(
            name                = "shufflenetv2_x1_5",
            stages_repeats      = [4, 8, 4],
            stages_out_channels = [24, 176, 352, 704, 1024],
            *args, **kwargs
        )


@MODELS.register(name="shufflenetv2_x2_0")
class ShuffleNetV2_X2_0(ShuffleNetV2):
    """ShuffleNetV2 architecture with 2.0x output channels, as described in
    `ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    <https://arxiv.org/abs/1807.11164>`__.
    
    See Also: :class:`ShuffleNetV2`
    """
    
    _zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/shufflenetv2_x2_0-8be3c8ee.pth",
            "path"       : "shufflenetv2/shufflenetv2_x2_0_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(self, args, **kwargs):
        super().__init__(
            name                = "shufflenetv2_x2_0",
            stages_repeats      = [4, 8, 4],
            stages_out_channels = [24, 244, 488, 976, 2048],
            *args, **kwargs
        )
        
# endregion
