#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements MobileNetV2 models."""

from __future__ import annotations

__all__ = [
    "MobileNetV2",
]

from typing import Callable, Any

import torch
from torchvision.models import _utils

from mon.globals import MODELS
from mon.vision import core, nn
from mon.vision.classify import base

console      = core.console
math         = core.math
_current_dir = core.Path(__file__).absolute().parent


# region Module

class InvertedResidual(nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : int,
        expand_ratio: int,
        norm_layer  : Callable[..., nn.Module] | None = None,
        *args, **kwargs,
    ):
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f":param:`stride` should be ``1`` or ``2``, but got {stride}.")
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        
        layers: list[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                nn.Conv2dNormAct(
                    in_channels      = in_channels,
                    out_channels     = hidden_dim,
                    kernel_size      = 1,
                    norm_layer       = norm_layer,
                    activation_layer = nn.ReLU6,
                )
            )
        layers.extend(
            [
                # DW
                nn.Conv2dNormAct(
                    in_channels      = hidden_dim,
                    out_channels     = hidden_dim,
                    stride           = stride,
                    groups           = hidden_dim,
                    norm_layer       = norm_layer,
                    activation_layer = nn.ReLU6,
                ),
                # PW-linear
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                norm_layer(out_channels),
            ]
        )
        self.conv         = nn.Sequential(*layers)
        self.out_channels = out_channels
        self._is_cn = stride > 1
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# endregion


# region Model

@MODELS.register(name="mobilenet_v2")
class MobileNetV2(base.ImageClassificationModel):
    """MobileNetV2 architecture from the `MobileNetV2: Inverted Residuals and
    Linear Bottlenecks <https://arxiv.org/abs/1801.04381>`_ paper.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
            "path"       : "mobilenet_v2-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-v2": {
            "url"        : "https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth",
            "path"       : "mobilenet_v2-imagenet1k-v2.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        num_classes              : int   = 1000,
        width_mult               : float = 1.0,
        inverted_residual_setting: list[list[int]] | None = None,
        round_nearest            : int   = 8,
        block                    : Callable[..., nn.Module] | list = None,
        norm_layer               : Callable[..., nn.Module] | list = None,
        dropout                  : float = 0.2,
        weights                  : Any   = None,
        name                     : str   = "mobilenetv2",
        *args, **kwargs,
    ):
        super().__init__(
            num_classes = num_classes,
            weights     = weights,
            name        = name,
            *args, **kwargs
        )
        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.width_mult    = width_mult
        self.round_nearest = round_nearest
        self.dropout       = dropout
        input_channel      = 32
        last_channel       = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1,  16, 1, 1],
                [6,  24, 2, 2],
                [6,  32, 3, 2],
                [6,  64, 4, 2],
                [6,  96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # Only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f":param:`inverted_residual_setting` should be non-empty or a "
                f"4-element :class:`list`, but got {inverted_residual_setting}."
            )

        # building first layer
        input_channel     = _utils._make_divisible(input_channel * self.width_mult, self.round_nearest)
        self.last_channel = _utils._make_divisible(last_channel * max(1.0, self.width_mult), self.round_nearest)
        features: list[nn.Module] = [
            nn.Conv2dNormAct(
                in_channels      = 3,
                out_channels     = input_channel,
                stride           = 2,
                norm_layer       = norm_layer,
                activation_layer = nn.ReLU6,
            )
        ]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _utils._make_divisible(c * self.width_mult, self.round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(
                        in_channels  = input_channel,
                        out_channels = output_channel,
                        stride       = stride,
                        expand_ratio = t,
                        norm_layer   = norm_layer,
                    )
                )
                input_channel = output_channel
        # Building last several layers
        features.append(
            nn.Conv2dNormAct(
                in_channels      = input_channel,
                out_channels     = self.last_channel,
                kernel_size      = 1,
                norm_layer       = norm_layer,
                activation_layer = nn.ReLU6,
            )
        )
        # Make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.last_channel, self.num_classes),
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        """Initialize model's weights."""
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, 0, 0.01)
            torch.nn.init.zeros_(m.bias)
    
    def forward_once(
        self,
        input    : torch.Tensor,
        profile  : bool = False,
        out_index: int  = -1,
        *args, **kwargs
    ) -> torch.Tensor:
        x = input
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        y = self.classifier(x)
        return y

# endregion
