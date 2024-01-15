#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements EfficientNet models."""

from __future__ import annotations

__all__ = [
    "EfficientNet",
    "EfficientNet_B0",
    "EfficientNet_B1",
    "EfficientNet_B2",
    "EfficientNet_B3",
    "EfficientNet_B4",
    "EfficientNet_B5",
    "EfficientNet_B6",
    "EfficientNet_B7",
    "EfficientNet_V2_L",
    "EfficientNet_V2_M",
    "EfficientNet_V2_S",
]

import copy
import functools
from abc import ABC
from dataclasses import dataclass
from typing import Any, Sequence, Callable

import torch
from torchvision import ops
from torchvision.models import _utils

from mon.globals import MODELS
from mon.vision import core, nn
from mon.vision.classify import base

console      = core.console
math         = core.math
_current_dir = core.Path(__file__).absolute().parent


# region Module

@dataclass
class _MBConvConfig:
    expand_ratio: float
    kernel      : int
    stride      : int
    in_channels : int
    out_channels: int
    num_layers  : int
    block       : Callable[..., nn.Module]

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: int | None = None) -> int:
        return _utils._make_divisible(channels * width_mult, 8, min_value)


class MBConvConfig(_MBConvConfig):
    """Stores information listed at Table 1 of the EfficientNet paper & Table 4
    of the EfficientNetV2 paper.
    """

    def __init__(
        self,
        expand_ratio: float,
        kernel      : int,
        stride      : int,
        in_channels : int,
        out_channels: int,
        num_layers  : int,
        width_mult  : float = 1.0,
        depth_mult  : float = 1.0,
        block       : Callable[..., nn.Module] | None = None,
        *args, **kwargs
    ):
        in_channels  = self.adjust_channels(in_channels, width_mult)
        out_channels = self.adjust_channels(out_channels, width_mult)
        num_layers   = self.adjust_depth(num_layers, depth_mult)
        if block is None:
            block = MBConv
        super().__init__(expand_ratio, kernel, stride, in_channels, out_channels, num_layers, block)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class FusedMBConvConfig(_MBConvConfig):
    """Stores information listed at Table 4 of the EfficientNetV2 paper."""
    
    def __init__(
        self,
        expand_ratio: float,
        kernel      : int,
        stride      : int,
        in_channels : int,
        out_channels: int,
        num_layers  : int,
        block       : Callable[..., nn.Module] | None = None,
        *args, **kwargs
    ):
        if block is None:
            block = FusedMBConv
        super().__init__(expand_ratio, kernel, stride, in_channels, out_channels, num_layers, block)


class MBConv(nn.Module):
    
    def __init__(
        self,
        cnf                  : MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer           : Callable[..., nn.Module],
        se_layer             : Callable[..., nn.Module] = nn.SqueezeExcitation,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        if not (1 <= cnf.stride <= 2):
            raise ValueError("Illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.in_channels == cnf.out_channels
        
        layers: list[nn.Module] = []
        activation_layer        = nn.SiLU

        # Expand
        expanded_channels = cnf.adjust_channels(cnf.in_channels, cnf.expand_ratio)
        if expanded_channels != cnf.in_channels:
            layers.append(
                nn.Conv2dNormAct(
                    in_channels      = cnf.in_channels,
                    out_channels     = expanded_channels,
                    kernel_size      = 1,
                    norm_layer       = norm_layer,
                    activation_layer = activation_layer,
                )
            )

        # Depthwise
        layers.append(
            nn.Conv2dNormAct(
                in_channels      = expanded_channels,
                out_channels     = expanded_channels,
                kernel_size      = cnf.kernel,
                stride           = cnf.stride,
                groups           = expanded_channels,
                norm_layer       = norm_layer,
                activation_layer = activation_layer,
            )
        )

        # Squeeze and excitation
        squeeze_channels = max(1, cnf.in_channels // 4)
        layers.append(
            se_layer(
                expanded_channels,
                squeeze_channels,
                activation=functools.partial(nn.SiLU, inplace=True)
            )
        )

        # Project
        layers.append(
            nn.Conv2dNormAct(
                in_channels      = expanded_channels,
                out_channels     = cnf.out_channels,
                kernel_size      = 1,
                norm_layer       = norm_layer,
                activation_layer = None,
            )
        )

        self.block            = nn.Sequential(*layers)
        self.stochastic_depth = ops.StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels     = cnf.out_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.block(x)
        if self.use_res_connect:
            y = self.stochastic_depth(y)
            y += x
        return y


class FusedMBConv(nn.Module):
    
    def __init__(
        self,
        cnf                  : FusedMBConvConfig,
        stochastic_depth_prob: float,
        norm_layer           : Callable[..., nn.Module],
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        if not (1 <= cnf.stride <= 2):
            raise ValueError(f"Illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.in_channels == cnf.out_channels

        layers: list[nn.Module] = []
        activation_layer        = nn.SiLU

        expanded_channels = cnf.adjust_channels(cnf.in_channels, cnf.expand_ratio)
        if expanded_channels != cnf.in_channels:
            # Fused expand
            layers.append(
                nn.Conv2dNormAct(
                    in_channels      = cnf.in_channels,
                    out_channels     = expanded_channels,
                    kernel_size      = cnf.kernel,
                    stride           = cnf.stride,
                    norm_layer       = norm_layer,
                    activation_layer = activation_layer,
                )
            )

            # Project
            layers.append(
                nn.Conv2dNormAct(
                    in_channels      = expanded_channels,
                    out_channels     = cnf.out_channels,
                    kernel_size      = 1,
                    norm_layer       = norm_layer,
                    activation_layer = None,
                )
            )
        else:
            layers.append(
                nn.Conv2dNormAct(
                    in_channels      = cnf.in_channels,
                    out_channels     = cnf.out_channels,
                    kernel_size      = cnf.kernel,
                    stride           = cnf.stride,
                    norm_layer       = norm_layer,
                    activation_layer = activation_layer,
                )
            )

        self.block            = nn.Sequential(*layers)
        self.stochastic_depth = ops.StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels     = cnf.out_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.block(x)
        if self.use_res_connect:
            y = self.stochastic_depth(y)
            y += x
        return y

# endregion


# region Model

def _efficientnet_conf(
    arch: str, **kwargs: Any,
) -> tuple[Sequence[MBConvConfig | FusedMBConvConfig], int | None]:
    inverted_residual_setting: Sequence[MBConvConfig | FusedMBConvConfig]
    if arch.startswith("efficientnet_b"):
        bneck_conf = functools.partial(
            MBConvConfig,
            width_mult=kwargs.pop("width_mult"),
            depth_mult=kwargs.pop("depth_mult")
        )
        inverted_residual_setting = [
            bneck_conf(1, 3, 1,  32,  16, 1),
            bneck_conf(6, 3, 2,  16,  24, 2),
            bneck_conf(6, 5, 2,  24,  40, 2),
            bneck_conf(6, 3, 2,  40,  80, 3),
            bneck_conf(6, 5, 1,  80, 112, 3),
            bneck_conf(6, 5, 2, 112, 192, 4),
            bneck_conf(6, 3, 1, 192, 320, 1),
        ]
        last_channel = None
    elif arch.startswith("efficientnet_v2_s"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 2),
            FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 2, 48, 64, 4),
            MBConvConfig(4, 3, 2, 64, 128, 6),
            MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 160, 256, 15),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_m"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 3),
            FusedMBConvConfig(4, 3, 2, 24, 48, 5),
            FusedMBConvConfig(4, 3, 2, 48, 80, 5),
            MBConvConfig(4, 3, 2, 80, 160, 7),
            MBConvConfig(6, 3, 1, 160, 176, 14),
            MBConvConfig(6, 3, 2, 176, 304, 18),
            MBConvConfig(6, 3, 1, 304, 512, 5),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_l"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 32, 32, 4),
            FusedMBConvConfig(4, 3, 2, 32, 64, 7),
            FusedMBConvConfig(4, 3, 2, 64, 96, 7),
            MBConvConfig(4, 3, 2, 96, 192, 10),
            MBConvConfig(6, 3, 1, 192, 224, 19),
            MBConvConfig(6, 3, 2, 224, 384, 25),
            MBConvConfig(6, 3, 1, 384, 640, 7),
        ]
        last_channel = 1280
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


class EfficientNet(base.ImageClassificationModel, ABC):
    """EfficientNet.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {}
    
    def __init__(
        self,
        inverted_residual_setting: Sequence[MBConvConfig | FusedMBConvConfig],
        dropout                  : float,
        stochastic_depth_prob    : float      = 0.2,
        num_classes              : int        = 1000,
        norm_layer               : Callable[..., nn.Module] | None = None,
        last_channel             : int | None = None,
        name                     : str        = "efficientnet",
        *args, **kwargs,
    ):
        super().__init__(
            num_classes = num_classes,
            name        = name,
            *args, **kwargs
        )
        if not inverted_residual_setting:
            raise ValueError(f"The :param:`inverted_residual_setting` should not be empty.")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The :param:`inverted_residual_setting` should be :class:`list[MBConvConfig]`")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: list[nn.Module] = []
        
        # Building first layer
        firstconv_output_channels = inverted_residual_setting[0].in_channels
        layers.append(
            nn.Conv2dNormAct(
                in_channels      = 3,
                out_channels     = firstconv_output_channels,
                kernel_size      = 3,
                stride           = 2,
                norm_layer       = norm_layer,
                activation_layer = nn.SiLU,
            )
        )

        # Building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id     = 0
        for cnf in inverted_residual_setting:
            stage: list[nn.Module] = []
            for _ in range(cnf.num_layers):
                # Copy to avoid modifications. Shallow copy is enough
                block_cnf = copy.copy(cnf)

                # Overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.in_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # Adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1
            
            layers.append(nn.Sequential(*stage))

        # Building last several layers
        lastconv_input_channels  = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        layers.append(
            nn.Conv2dNormAct(
                in_channels      = lastconv_input_channels,
                out_channels     = lastconv_output_channels,
                kernel_size      = 1,
                norm_layer       = norm_layer,
                activation_layer = nn.SiLU,
            )
        )

        self.features   = nn.Sequential(*layers)
        self.avgpool    = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )
        
        self.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        """Initialize model's weights."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            init_range = 1.0 / math.sqrt(m.out_features)
            nn.init.uniform_(m.weight, -init_range, init_range)
            nn.init.zeros_(m.bias)
    
    def forward_once(
        self,
        input    : torch.Tensor,
        profile  : bool = False,
        out_index: int  = -1,
        *args, **kwargs
    ) -> torch.Tensor:
        x = input
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.classifier(x)
        return y
    

@MODELS.register(name="efficientnet_b0")
class EfficientNet_B0(EfficientNet):
    """EfficientNet B0 model architecture from the `EfficientNet: Rethinking
    Model Scaling for Convolutional Neural Networks
    <https://arxiv.org/abs/1905.11946>`_ paper.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth",
            "path"       : "efficientnet_b0-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "efficientnet",
        variant: str = "efficientnet_b0",
        *args, **kwargs
    ):
        inverted_residual_setting, last_channel = _efficientnet_conf(
            arch       = "efficientnet_b0",
            width_mult = 1.0,
            depth_mult = 1.0,
        )
        super().__init__(
            inverted_residual_setting = inverted_residual_setting,
            dropout                   = kwargs.pop("dropout", 0.2),
            last_channel              = last_channel,
            name                      = name,
            variant                   = variant,
            *args, **kwargs
        )


@MODELS.register(name="efficientnet_b1")
class EfficientNet_B1(EfficientNet):
    """EfficientNet B1 model architecture from the `EfficientNet: Rethinking
    Model Scaling for Convolutional Neural Networks
    <https://arxiv.org/abs/1905.11946>`_ paper.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b1_rwightman-bac287d4.pth",
            "path"       : "efficientnet_b1-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-v2": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b1-c27df63c.pth",
            "path"       : "efficientnet_b1-imagenet1k-v2.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "efficientnet",
        variant: str = "efficientnet_b1",
        *args, **kwargs
    ):
        inverted_residual_setting, last_channel = _efficientnet_conf(
            arch       = "efficientnet_b1",
            width_mult = 1.0,
            depth_mult = 1.1,
        )
        super().__init__(
            inverted_residual_setting = inverted_residual_setting,
            dropout                   = kwargs.pop("dropout", 0.2),
            last_channel              = last_channel,
            name                      = name,
            variant                   = variant,
            *args, **kwargs
        )


@MODELS.register(name="efficientnet_b2")
class EfficientNet_B2(EfficientNet):
    """EfficientNet B2 model architecture from the `EfficientNet: Rethinking
    Model Scaling for Convolutional Neural Networks
    <https://arxiv.org/abs/1905.11946>`_ paper.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b2_rwightman-c35c1473.pth",
            "path"       : "efficientnet_b2-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "efficientnet",
        variant: str = "efficientnet_b2",
        *args, **kwargs
    ):
        inverted_residual_setting, last_channel = _efficientnet_conf(
            arch       = "efficientnet_b2",
            width_mult = 1.1,
            depth_mult = 1.2,
        )
        super().__init__(
            inverted_residual_setting = inverted_residual_setting,
            dropout                   = kwargs.pop("dropout", 0.3),
            last_channel              = last_channel,
            name                      = name,
            variant                   = variant,
            *args, **kwargs
        )


@MODELS.register(name="efficientnet_b3")
class EfficientNet_B3(EfficientNet):
    """EfficientNet B3 model architecture from the `EfficientNet: Rethinking
    Model Scaling for Convolutional Neural Networks
    <https://arxiv.org/abs/1905.11946>`_ paper.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b3_rwightman-b3899882.pth",
            "path"       : "efficientnet_b3-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "efficientnet",
        variant: str = "efficientnet_b3",
        *args, **kwargs
    ):
        inverted_residual_setting, last_channel = _efficientnet_conf(
            arch       = "efficientnet_b3",
            width_mult = 1.2,
            depth_mult = 1.4,
        )
        super().__init__(
            inverted_residual_setting = inverted_residual_setting,
            dropout                   = kwargs.pop("dropout", 0.3),
            last_channel              = last_channel,
            name                      = name,
            variant                   = variant,
            *args, **kwargs
        )


@MODELS.register(name="efficientnet_b4")
class EfficientNet_B4(EfficientNet):
    """EfficientNet B4 model architecture from the `EfficientNet: Rethinking
    Model Scaling for Convolutional Neural Networks
    <https://arxiv.org/abs/1905.11946>`_ paper.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b4_rwightman-23ab8bcd.pth",
            "path"       : "efficientnet_b4-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "efficientnet",
        variant: str = "efficientnet_b4",
        *args, **kwargs
    ):
        inverted_residual_setting, last_channel = _efficientnet_conf(
            arch       = "efficientnet_b4",
            width_mult = 1.4,
            depth_mult = 1.8,
        )
        super().__init__(
            inverted_residual_setting = inverted_residual_setting,
            dropout                   = kwargs.pop("dropout", 0.4),
            last_channel              = last_channel,
            name                      = name,
            variant                   = variant,
            *args, **kwargs
        )


@MODELS.register(name="efficientnet_b5")
class EfficientNet_B5(EfficientNet):
    """EfficientNet B5 model architecture from the `EfficientNet: Rethinking
    Model Scaling for Convolutional Neural Networks
    <https://arxiv.org/abs/1905.11946>`_ paper.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b5_lukemelas-1a07897c.pth",
            "path"       : "efficientnet_b5-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "efficientnet",
        variant: str = "efficientnet_b5",
        *args, **kwargs
    ):
        inverted_residual_setting, last_channel = _efficientnet_conf(
            arch       = "efficientnet_b5",
            width_mult = 1.6,
            depth_mult = 2.2,
        )
        super().__init__(
            inverted_residual_setting = inverted_residual_setting,
            dropout                   = kwargs.pop("dropout", 0.4),
            last_channel              = last_channel,
            norm_layer                = functools.partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
            name                      = name,
            variant                   = variant,
            *args, **kwargs
        )


@MODELS.register(name="efficientnet_b6")
class EfficientNet_B6(EfficientNet):
    """EfficientNet B6 model architecture from the `EfficientNet: Rethinking
    Model Scaling for Convolutional Neural Networks
    <https://arxiv.org/abs/1905.11946>`_ paper.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b6_lukemelas-24a108a5.pth",
            "path"       : "efficientnet_b6-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "efficientnet",
        variant: str = "efficientnet_b6",
        *args, **kwargs
    ):
        inverted_residual_setting, last_channel = _efficientnet_conf(
            arch       = "efficientnet_b6",
            width_mult = 1.8,
            depth_mult = 2.6,
        )
        super().__init__(
            inverted_residual_setting = inverted_residual_setting,
            dropout                   = kwargs.pop("dropout", 0.5),
            last_channel              = last_channel,
            norm_layer                = functools.partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
            name                      = name,
            variant                   = variant,
            *args, **kwargs
        )
   
   
@MODELS.register(name="efficientnet_b7")
class EfficientNet_B7(EfficientNet):
    """EfficientNet B7 model architecture from the `EfficientNet: Rethinking
    Model Scaling for Convolutional Neural Networks
    <https://arxiv.org/abs/1905.11946>`_ paper.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b7_lukemelas-c5b4e57e.pth",
            "path"       : "efficientnet_b7-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "efficientnet",
        variant: str = "efficientnet_b7",
        *args, **kwargs
    ):
        inverted_residual_setting, last_channel = _efficientnet_conf(
            arch       = "efficientnet_b7",
            width_mult = 2.0,
            depth_mult = 3.1,
        )
        super().__init__(
            inverted_residual_setting = inverted_residual_setting,
            dropout                   = kwargs.pop("dropout", 0.5),
            last_channel              = last_channel,
            norm_layer                = functools.partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
            name                      = name,
            variant                   = variant,
            *args, **kwargs
        )


@MODELS.register(name="efficientnet_v2_s")
class EfficientNet_V2_S(EfficientNet):
    """EfficientNetV2-S architecture from `EfficientNetV2: Smaller Models and
    Faster Training <https://arxiv.org/abs/2104.00298>`__.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth",
            "path"       : "efficientnet_v2_s-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "efficientnet",
        variant: str = "efficientnet_v2_s",
        *args, **kwargs
    ):
        inverted_residual_setting, last_channel = _efficientnet_conf(
            arch = "efficientnet_v2_s",
        )
        super().__init__(
            inverted_residual_setting = inverted_residual_setting,
            dropout                   = kwargs.pop("dropout", 0.2),
            last_channel              = last_channel,
            norm_layer                = functools.partial(nn.BatchNorm2d, eps=1e-03),
            name                      = name,
            variant                   = variant,
            *args, **kwargs
        )


@MODELS.register(name="efficientnet_v2_m")
class EfficientNet_V2_M(EfficientNet):
    """EfficientNetV2-M architecture from `EfficientNetV2: Smaller Models and
    Faster Training <https://arxiv.org/abs/2104.00298>`__.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_v2_m-dc08266a.pth",
            "path"       : "efficientnet_v2_m-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "efficientnet",
        variant: str = "efficientnet_v2_m",
        *args, **kwargs
    ):
        inverted_residual_setting, last_channel = _efficientnet_conf(
            arch = "efficientnet_v2_m",
        )
        super().__init__(
            inverted_residual_setting = inverted_residual_setting,
            dropout                   = kwargs.pop("dropout", 0.3),
            last_channel              = last_channel,
            norm_layer                = functools.partial(nn.BatchNorm2d, eps=1e-03),
            name                      = name,
            variant                   = variant,
            *args, **kwargs
        )


@MODELS.register(name="efficientnet_v2_l")
class EfficientNet_V2_L(EfficientNet):
    """EfficientNetV2-L architecture from `EfficientNetV2: Smaller Models and
    Faster Training <https://arxiv.org/abs/2104.00298>`__.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_v2_l-59c71312.pth",
            "path"       : "efficientnet_v2_l-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "efficientnet",
        variant: str = "efficientnet_v2_l",
        *args, **kwargs
    ):
        inverted_residual_setting, last_channel = _efficientnet_conf(
            arch = "efficientnet_v2_l",
        )
        super().__init__(
            inverted_residual_setting = inverted_residual_setting,
            dropout                   = kwargs.pop("dropout", 0.4),
            last_channel              = last_channel,
            norm_layer                = functools.partial(nn.BatchNorm2d, eps=1e-03),
            name                      = name,
            variant                   = variant,
            *args, **kwargs
        )
        
# endregion
