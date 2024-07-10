#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements MobileNetV3 models."""

from __future__ import annotations

__all__ = [
    "MobileNetV3",
    "MobileNetV3_Large",
    "MobileNetV3_Small",
]

import functools
from abc import ABC
from typing import Any, Sequence

import torch
from torchvision.models import _utils

from mon import core, nn
from mon.core import _callable
from mon.globals import MODELS, Scheme
from mon.vision.classify import base

console = core.console


# region Module

class InvertedResidualConfig:
    """Stores information listed at Tables 1 and 2 of the MobileNetV3 paper."""
    
    def __init__(
        self,
        in_channels      : int,
        kernel           : int,
        expanded_channels: int,
        out_channels     : int,
        use_se           : bool,
        activation       : str,
        stride           : int,
        dilation         : int,
        width_mult       : float,
    ):
        self.in_channels       = self.adjust_channels(in_channels, width_mult)
        self.kernel            = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels      = self.adjust_channels(out_channels, width_mult)
        self.use_se            = use_se
        self.use_hs            = activation == "HS"
        self.stride            = stride
        self.dilation          = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _utils._make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    """Implemented as described at section 5 of MobileNetV3 paper."""
    
    def __init__(
        self,
        cnf       : InvertedResidualConfig,
        norm_layer: _callable,
        se_layer  : _callable = functools.partial(nn.SqueezeExcitation, scale_activation=nn.Hardsigmoid),
        *args, **kwargs
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError(f"Illegal stride value.")

        self.use_res_connect = cnf.stride == 1 and cnf.in_channels == cnf.out_channels

        layers: list[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # Expand
        if cnf.expanded_channels != cnf.in_channels:
            layers.append(
                nn.Conv2dNormAct(
                    in_channels      = cnf.in_channels,
                    out_channels     = cnf.expanded_channels,
                    kernel_size      = 1,
                    norm_layer       = norm_layer,
                    activation_layer = activation_layer,
                )
            )

        # Depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            nn.Conv2dNormAct(
                in_channels      = cnf.expanded_channels,
                out_channels     = cnf.expanded_channels,
                kernel_size      = cnf.kernel,
                stride           = stride,
                dilation         = cnf.dilation,
                groups           = cnf.expanded_channels,
                norm_layer       = norm_layer,
                activation_layer = activation_layer,
            )
        )
        if cnf.use_se:
            squeeze_channels = _utils._make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels))

        # Project
        layers.append(
            nn.Conv2dNormAct(
                in_channels      = cnf.expanded_channels,
                out_channels     = cnf.out_channels,
                kernel_size      = 1,
                norm_layer       = norm_layer,
                activation_layer = None,
            )
        )

        self.block        = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn       = cnf.stride > 1

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.block(x)
        if self.use_res_connect:
            y += x
        return y

# endregion


# region Model

def _mobilenetv3_conf(
    arch        : str,
    width_mult  : float = 1.0,
    reduced_tail: bool  = False,
    dilated     : bool  = False,
    *args, **kwargs
):
    reduce_divider  = 2 if reduced_tail else 1
    dilation        = 2 if dilated      else 1
    bneck_conf      = functools.partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = functools.partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)
    
    if arch == "mobilenetv3_large":
        inverted_residual_setting = [
            bneck_conf(16,  3,  16,  16, False, "RE", 1, 1),
            bneck_conf(16,  3,  64,  24, False, "RE", 2, 1),  # C1
            bneck_conf(24,  3,  72,  24, False, "RE", 1, 1),
            bneck_conf(24,  5,  72,  40, True,  "RE", 2, 1),  # C2
            bneck_conf(40,  5, 120,  40, True,  "RE", 1, 1),
            bneck_conf(40,  5, 120,  40, True,  "RE", 1, 1),
            bneck_conf(40,  3, 240,  80, False, "HS", 2, 1),  # C3
            bneck_conf(80,  3, 200,  80, False, "HS", 1, 1),
            bneck_conf(80,  3, 184,  80, False, "HS", 1, 1),
            bneck_conf(80,  3, 184,  80, False, "HS", 1, 1),
            bneck_conf(80,  3, 480, 112, True,  "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True,  "HS", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)  # C5
    elif arch == "mobilenetv3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3,  16, 16, True,  "RE", 2, 1),  # C1
            bneck_conf(16, 3,  72, 24, False, "RE", 2, 1),  # C2
            bneck_conf(24, 3,  88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5,  96, 40, True,  "HS", 2, 1),  # C3
            bneck_conf(40, 5, 240, 40, True,  "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True,  "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True,  "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True,  "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1024 // reduce_divider)  # C5
    else:
        raise ValueError(f"Unsupported model type {arch}")
    
    return inverted_residual_setting, last_channel


class MobileNetV3(base.ImageClassificationModel, ABC):
    """MobileNetV3.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    _scheme: list[Scheme] = [Scheme.SUPERVISED]
    _zoo   : dict = {}
    
    def __init__(
        self,
        inverted_residual_setting: list[InvertedResidualConfig],
        last_channel             : int,
        in_channels              : int       = 3,
        num_classes              : int       = 1000,
        block                    : _callable = None,
        norm_layer               : _callable = None,
        dropout                  : float     = 0.2,
        weights                  : Any       = None,
        *args, **kwargs,
    ):
        super().__init__(
            in_channels = in_channels,
            num_classes = num_classes,
            weights     = weights,
            *args, **kwargs
        )
        if not inverted_residual_setting:
            raise ValueError("The :param:`inverted_residual_setting` should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The :param:`inverted_residual_setting` should be :class:`list[InvertedResidualConfig]`")

        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = functools.partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        
        self.dropout = dropout
        
        layers: list[nn.Module] = []

        # Building first layer
        firstconv_output_channels = inverted_residual_setting[0].in_channels
        layers.append(
            nn.Conv2dNormAct(
                in_channels      = self.in_channels,
                out_channels     = firstconv_output_channels,
                kernel_size      = 3,
                stride           = 2,
                norm_layer       = norm_layer,
                activation_layer = nn.Hardswish,
            )
        )

        # Building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # Building last several layers
        lastconv_input_channels  = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            nn.Conv2dNormAct(
                in_channels      = lastconv_input_channels,
                out_channels     = lastconv_output_channels,
                kernel_size      = 1,
                norm_layer       = norm_layer,
                activation_layer = nn.Hardswish,
            )
        )

        self.features   = nn.Sequential(*layers)
        self.avgpool    = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=self.dropout, inplace=True),
            nn.Linear(last_channel, self.num_classes),
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
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
    
    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> torch.Tensor:
        x = input
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.classifier(x)
        return y
    

@MODELS.register(name="mobilenetv3_large")
class MobileNetV3_Large(MobileNetV3):
    """MobileNetV3 architecture from `Searching for MobileNetV3
    <https://arxiv.org/abs/1905.02244>`__.
    
    See Also: :class:`MobileNetV3`
    """
    
    _zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
            "path"       : "mobilenetv3/mobilenetv3_large/imagenet1k_v1/mobilenetv3_large_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth",
            "path"       : "mobilenetv3/mobilenetv3_large/imagenet1k_v2/mobilenetv3_large_imagenet1k_v2.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(self, args, **kwargs):
        inverted_residual_setting, last_channel = _mobilenetv3_conf(arch="mobilenetv3_large", **kwargs)
        super().__init__(
            name                      = "mobilenetv3_large",
            inverted_residual_setting = inverted_residual_setting,
            last_channel              = last_channel,
            *args, **kwargs
        )


@MODELS.register(name="mobilenetv3_small")
class MobileNetV3_Small(MobileNetV3):
    """MobileNetV3 architecture from `Searching for MobileNetV3
    <https://arxiv.org/abs/1905.02244>`__.
    
    See Also: :class:`MobileNetV3`
    """
    
    _zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
            "path"       : "mobilenetv3/mobilenetv3_small/imagenet1k_v1/mobilenetv3_small_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(self, args, **kwargs):
        inverted_residual_setting, last_channel = _mobilenetv3_conf(arch="mobilenetv3_small", **kwargs)
        super().__init__(
            name                      = "mobilenetv3_small",
            inverted_residual_setting = inverted_residual_setting,
            last_channel              = last_channel,
            *args, **kwargs
        )

# endregion
