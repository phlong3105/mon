#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements RegNet models."""

from __future__ import annotations

__all__ = [
    "RegNet",
    "RegNetX_32GF",
    "RegNet_X_16GF",
    "RegNet_X_1_6GF",
    "RegNet_X_3_2GF",
    "RegNet_X_400MF",
    "RegNet_X_800MF",
    "RegNet_X_8GF",
    "RegNet_Y_128GF",
    "RegNet_Y_16GF",
    "RegNet_Y_1_6GF",
    "RegNet_Y_32GF",
    "RegNet_Y_3_2GF",
    "RegNet_Y_400MF",
    "RegNet_Y_800MF",
    "RegNet_Y_8GF",
]

from abc import ABC
from collections import OrderedDict
from functools import partial
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

class SimpleStemIN(nn.Conv2dNormAct):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(
        self,
        width_in        : int,
        width_out       : int,
        norm_layer      : Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        *args, **kwargs
    ):
        super().__init__(
            in_channels      = width_in,
            out_channels     = width_out,
            kernel_size      = 3,
            stride           = 2,
            norm_layer       = norm_layer,
            activation_layer = activation_layer,
            *args, **kwargs
        )


class BottleneckTransform(nn.Sequential):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(
        self,
        width_in             : int,
        width_out            : int,
        stride               : int,
        norm_layer           : Callable[..., nn.Module],
        activation_layer     : Callable[..., nn.Module],
        group_width          : int,
        bottleneck_multiplier: float,
        se_ratio             : float | None,
    ):
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        w_b = int(round(width_out * bottleneck_multiplier))
        g   = w_b // group_width

        layers["a"] = nn.Conv2dNormAct(
            in_channels      = width_in,
            out_channels     = w_b,
            kernel_size      = 1,
            stride           = 1,
            norm_layer       = norm_layer,
            activation_layer = activation_layer,
        )
        layers["b"] = nn.Conv2dNormAct(
            in_channels      = w_b,
            out_channels     = w_b,
            kernel_size      = 3,
            stride           = stride,
            groups           = g,
            norm_layer       = norm_layer,
            activation_layer = activation_layer,
        )

        if se_ratio:
            # The SE reduction ratio is defined with respect to the beginning of
            # the block
            width_se_out = int(round(se_ratio * width_in))
            layers["se"] = nn.SqueezeExcitation(
                input_channels   = w_b,
                squeeze_channels = width_se_out,
                activation       = activation_layer,
            )

        layers["c"] = nn.Conv2dNormAct(
            in_channels      = w_b,
            out_channels     = width_out,
            kernel_size      = 1,
            stride           = 1,
            norm_layer       = norm_layer,
            activation_layer = None,
        )
        super().__init__(layers)


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""
    
    def __init__(
        self,
        width_in             : int,
        width_out            : int,
        stride               : int,
        norm_layer           : Callable[..., nn.Module],
        activation_layer     : Callable[..., nn.Module],
        group_width          : int          = 1,
        bottleneck_multiplier: float        = 1.0,
        se_ratio             : float | None = None,
        *args, **kwargs
    ):
        super().__init__()

        # Use skip connection with projection if shape changes
        self.proj   = None
        should_proj = (width_in != width_out) or (stride != 1)
        if should_proj:
            self.proj = nn.Conv2dNormAct(
                in_channels      = width_in,
                out_channels     = width_out,
                kernel_size      = 1,
                stride           = stride,
                norm_layer       = norm_layer,
                activation_layer = None,
            )
        self.f = BottleneckTransform(
            width_in              = width_in,
            width_out             = width_out,
            stride                = stride,
            norm_layer            = norm_layer,
            activation_layer      = activation_layer,
            group_width           = group_width,
            bottleneck_multiplier = bottleneck_multiplier,
            se_ratio              = se_ratio,
        )
        self.activation = activation_layer(inplace=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.proj is not None:
            x = self.proj(x) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)


class AnyStage(nn.Sequential):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""
    
    def __init__(
        self,
        width_in             : int,
        width_out            : int,
        stride               : int,
        depth                : int,
        block_constructor    : Callable[..., nn.Module],
        norm_layer           : Callable[..., nn.Module],
        activation_layer     : Callable[..., nn.Module],
        group_width          : int,
        bottleneck_multiplier: float,
        se_ratio             : float | None = None,
        stage_index          : int          = 0,
        *args, **kwargs
    ):
        super().__init__()
        for i in range(depth):
            block = block_constructor(
                width_in              = width_in if i == 0 else width_out,
                width_out             = width_out,
                stride                = stride if i == 0 else 1,
                norm_layer            = norm_layer,
                activation_layer      = activation_layer,
                group_width           = group_width,
                bottleneck_multiplier = bottleneck_multiplier,
                se_ratio              = se_ratio,
            )
            self.add_module(f"block{stage_index}-{i}", block)


class BlockParams:
    
    def __init__(
        self,
        depths                : list[int],
        widths                : list[int],
        group_widths          : list[int],
        bottleneck_multipliers: list[float],
        strides               : list[int],
        se_ratio              : float | None = None,
    ):
        self.depths                 = depths
        self.widths                 = widths
        self.group_widths           = group_widths
        self.bottleneck_multipliers = bottleneck_multipliers
        self.strides                = strides
        self.se_ratio               = se_ratio

    @classmethod
    def from_init_params(
        cls,
        depth                : int,
        w_0                  : int,
        w_a                  : float,
        w_m                  : float,
        group_width          : int,
        bottleneck_multiplier: float        = 1.0,
        se_ratio             : float | None = None,
        *args, **kwarg,
    ) -> "BlockParams":
        """Programmatically compute all the per-block settings, given the RegNet
        parameters.
        
        The first step is to compute the quantized linear block parameters,
        in log space. Key parameters are:
        - `w_a` is the width progression slope
        - `w_0` is the initial width
        - `w_m` is the width stepping in the log space

        In other terms
        `log(block_width) = log(w_0) + w_m * block_capacity`,
        with `bock_capacity` ramping up following the w_0 and w_a params.
        This block width is finally quantized to multiples of 8.

        The second step is to compute the parameters per stage,
        taking into account the skip connection and the final 1x1 convolutions.
        We use the fact that the output width is constant within a stage.
        """

        QUANT  = 8
        STRIDE = 2

        if w_a < 0 or w_0 <= 0 or w_m <= 1 or w_0 % 8 != 0:
            raise ValueError("Invalid RegNet settings")
        # Compute the block widths. Each stage has one unique block width
        widths_cont    = torch.arange(depth) * w_a + w_0
        block_capacity = torch.round(torch.log(widths_cont / w_0) / math.log(w_m))
        block_widths   = (torch.round(torch.divide(w_0 * torch.pow(w_m, block_capacity), QUANT)) * QUANT).int().tolist()
        num_stages     = len(set(block_widths))

        # Convert to per stage parameters
        split_helper = zip(
            block_widths + [0],
            [0] + block_widths,
            block_widths + [0],
            [0] + block_widths,
        )
        splits       = [w != wp or r != rp for w, wp, r, rp in split_helper]
        stage_widths = [w for w, t in zip(block_widths, splits[:-1]) if t]
        stage_depths = torch.diff(torch.tensor([d for d, t in enumerate(splits) if t])).int().tolist()

        strides = [STRIDE] * num_stages
        bottleneck_multipliers = [bottleneck_multiplier] * num_stages
        group_widths = [group_width] * num_stages

        # Adjust the compatibility of stage widths and group widths
        stage_widths, group_widths = cls._adjust_widths_groups_compatibility(
            stage_widths, bottleneck_multipliers, group_widths
        )

        return cls(
            depths                 = stage_depths,
            widths                 = stage_widths,
            group_widths           = group_widths,
            bottleneck_multipliers = bottleneck_multipliers,
            strides                = strides,
            se_ratio               = se_ratio,
        )

    def _get_expanded_params(self):
        return zip(self.widths, self.strides, self.depths, self.group_widths, self.bottleneck_multipliers)

    @staticmethod
    def _adjust_widths_groups_compatibility(
        stage_widths     : list[int],
        bottleneck_ratios: list[float],
        group_widths     : list[int]
    ) -> tuple[list[int], list[int]]:
        """Adjusts the compatibility of widths and groups, depending on the
        bottleneck ratio.
        """
        # Compute all widths for the current settings
        widths           = [int(w * b) for w, b in zip(stage_widths, bottleneck_ratios)]
        group_widths_min = [min(g, w_bot) for g, w_bot in zip(group_widths, widths)]
        # Compute the adjusted widths so that stage and group widths fit
        ws_bot           = [_utils._make_divisible(w_bot, g) for w_bot, g in zip(widths, group_widths_min)]
        stage_widths     = [int(w_bot / b) for w_bot, b in zip(ws_bot, bottleneck_ratios)]
        return stage_widths, group_widths_min

# endregion


# region ResNet

class RegNet(base.ImageClassificationModel, ABC):
    """RegNet.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {}
    
    def __init__(
        self,
        block_params: BlockParams,
        num_classes : int = 1000,
        stem_width  : int = 32,
        stem_type   : Callable[..., nn.Module] | None = None,
        block_type  : Callable[..., nn.Module] | None = None,
        norm_layer  : Callable[..., nn.Module] | None = None,
        activation  : Callable[..., nn.Module] | None = None,
        weights     : Any = None,
        name        : str = "regnet",
        *args, **kwargs
    ):
        super().__init__(
            num_classes = num_classes,
            weights     = weights,
            name        = name,
            *args, **kwargs
        )
        if stem_type is None:
            stem_type = SimpleStemIN
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if block_type is None:
            block_type = ResBottleneckBlock
        if activation is None:
            activation = nn.ReLU

        # Ad hoc stem
        self.stem = stem_type(
            3,  # width_in
            stem_width,
            norm_layer,
            activation,
        )

        current_width = stem_width
        blocks = []
        for i, (width_out, stride, depth, group_width, bottleneck_multiplier) in enumerate(block_params._get_expanded_params()):
            blocks.append(
                (
                    f"block{i + 1}",
                    AnyStage(
                         width_in             = current_width,
                         width_out            = width_out,
                         stride               = stride,
                         depth                = depth,
                         block_constructor    = block_type,
                         norm_layer           = norm_layer,
                         activation_layer     = activation,
                         group_width          = group_width,
                         bottleneck_multiplier= bottleneck_multiplier,
                         se_ratio             = block_params.se_ratio,
                        stage_index           = i + 1,
                    ),
                )
            )
            current_width = width_out

        self.trunk_output = nn.Sequential(OrderedDict(blocks))
        self.avgpool      = nn.AdaptiveAvgPool2d((1, 1))
        self.fc           = nn.Linear(in_features=current_width, out_features=num_classes)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
    
    def init_weights(self, m: nn.Module):
        """Initialize model's weights."""
        if isinstance(m, nn.Conv2d):
            # Note that there is no bias due to BN
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            torch.nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            torch.nn.init.zeros_(m.bias)
        
    def forward_once(
        self,
        input    : torch.Tensor,
        profile  : bool = False,
        out_index: int  = -1,
        *args, **kwargs
    ) -> torch.Tensor:
        x = input
        x = self.stem(x)
        x = self.trunk_output(x)
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        y = self.fc(x)
        return y
    

@MODELS.register(name="regnet_y_400mf")
class RegNet_Y_400MF(RegNet):
    """RegNetY_400MF architecture from `Designing Network Design Spaces
    <https://arxiv.org/abs/2003.13678>`__.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/regnet_y_400mf-c65dace8.pth",
            "path"       : "regnet_y_400mf-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-v2": {
            "url"        : "https://download.pytorch.org/models/regnet_y_400mf-e6988f5f.pth",
            "path"       : "regnet_y_400mf-imagenet1k-v2.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "regnet",
        variant: str = "regnet_y_400mf",
        *args, **kwargs
    ):
        params     = BlockParams.from_init_params(depth=16, w_0=48, w_a=27.89, w_m=2.09, group_width=8, se_ratio=0.25, **kwargs)
        norm_layer = kwargs.pop("norm_layer", partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
        super().__init__(
            block_params = params,
            norm_layer   = norm_layer,
            name         = name,
            variant      = variant,
            *args, **kwargs
        )


@MODELS.register(name="regnet_y_800mf")
class RegNet_Y_800MF(RegNet):
    """RegNetY_800MF architecture from `Designing Network Design Spaces
    <https://arxiv.org/abs/2003.13678>`__.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/regnet_y_800mf-1b27b58c.pth",
            "path"       : "regnet_y_800mf-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-v2": {
            "url"        : "https://download.pytorch.org/models/regnet_y_800mf-58fc7688.pth",
            "path"       : "regnet_y_800mf-imagenet1k-v2.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "regnet",
        variant: str = "regnet_y_800mf",
        *args, **kwargs
    ):
        params     = BlockParams.from_init_params(depth=14, w_0=56, w_a=38.84, w_m=2.4, group_width=16, se_ratio=0.25, **kwargs)
        norm_layer = kwargs.pop("norm_layer", partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
        super().__init__(
            block_params = params,
            norm_layer   = norm_layer,
            name         = name,
            variant      = variant,
            *args, **kwargs
        )
    

@MODELS.register(name="regnet_y_1_6gf")
class RegNet_Y_1_6GF(RegNet):
    """RegNetY_1.6GF architecture from `Designing Network Design Spaces
    <https://arxiv.org/abs/2003.13678>`__.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/regnet_y_1_6gf-b11a554e.pth",
            "path"       : "regnet_y_1_6gf-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-v2": {
            "url"        : "https://download.pytorch.org/models/regnet_y_1_6gf-0d7bc02a.pth",
            "path"       : "regnet_y_1_6gf-imagenet1k-v2.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "regnet",
        variant: str = "regnet_y_1_6gf",
        *args, **kwargs
    ):
        params     = BlockParams.from_init_params(depth=27, w_0=48, w_a=20.71, w_m=2.65, group_width=24, se_ratio=0.25, **kwargs)
        norm_layer = kwargs.pop("norm_layer", partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
        super().__init__(
            block_params = params,
            norm_layer   = norm_layer,
            name         = name,
            variant      = variant,
            *args, **kwargs
        )


@MODELS.register(name="regnet_y_3_2gf")
class RegNet_Y_3_2GF(RegNet):
    """RegNetY_3.2GF architecture from `Designing Network Design Spaces
    <https://arxiv.org/abs/2003.13678>`__.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/regnet_y_3_2gf-b5a9779c.pth",
            "path"       : "regnet_y_3_2gf-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-v2": {
            "url"        : "https://download.pytorch.org/models/regnet_y_3_2gf-9180c971.pth",
            "path"       : "regnet_y_3_2gf-imagenet1k-v2.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "regnet",
        variant: str = "regnet_y_3_2gf",
        *args, **kwargs
    ):
        params     = BlockParams.from_init_params(depth=21, w_0=80, w_a=42.63, w_m=2.66, group_width=24, se_ratio=0.25, **kwargs)
        norm_layer = kwargs.pop("norm_layer", partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
        super().__init__(
            block_params = params,
            norm_layer   = norm_layer,
            name         = name,
            variant      = variant,
            *args, **kwargs
        )


@MODELS.register(name="regnet_y_8gf")
class RegNet_Y_8GF(RegNet):
    """RegNetY_8GF architecture from `Designing Network Design Spaces
    <https://arxiv.org/abs/2003.13678>`__.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/regnet_y_8gf-d0d0e4a8.pth",
            "path"       : "regnet_y_8gf-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-v2": {
            "url"        : "https://download.pytorch.org/models/regnet_y_8gf-dc2b1b54.pth",
            "path"       : "regnet_y_8gf-imagenet1k-v2.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "regnet",
        variant: str = "regnet_y_8gf",
        *args, **kwargs
    ):
        params     = BlockParams.from_init_params(depth=17, w_0=192, w_a=76.82, w_m=2.19, group_width=56, se_ratio=0.25, **kwargs)
        norm_layer = kwargs.pop("norm_layer", partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
        super().__init__(
            block_params = params,
            norm_layer   = norm_layer,
            name         = name,
            variant      = variant,
            *args, **kwargs
        )
        

@MODELS.register(name="regnet_y_16gf")
class RegNet_Y_16GF(RegNet):
    """RegNetY_16GF architecture from `Designing Network Design Spaces
    <https://arxiv.org/abs/2003.13678>`__.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/regnet_y_16gf-9e6ed7dd.pth",
            "path"       : "regnet_y_16gf-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-v2": {
            "url"        : "https://download.pytorch.org/models/regnet_y_16gf-3e4a00f9.pth",
            "path"       : "regnet_y_16gf-imagenet1k-v2.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-swag-e2e-v1": {
            "url"        : "https://download.pytorch.org/models/regnet_y_16gf_swag-43afe44d.pth",
            "path"       : "regnet_y_16gf-swag-e2e-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-swag-lc-v1": {
            "url"        : "https://download.pytorch.org/models/regnet_y_16gf_lc_swag-f3ec0043.pth",
            "path"       : "regnet_y_16gf-lc-swag-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "regnet",
        variant: str = "regnet_y_16gf",
        *args, **kwargs
    ):
        params     = BlockParams.from_init_params(depth=18, w_0=200, w_a=106.23, w_m=2.48, group_width=112, se_ratio=0.25, **kwargs)
        norm_layer = kwargs.pop("norm_layer", partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
        super().__init__(
            block_params = params,
            norm_layer   = norm_layer,
            name         = name,
            variant      = variant,
            *args, **kwargs
        )
     

@MODELS.register(name="regnet_y_32gf")
class RegNet_Y_32GF(RegNet):
    """RegNetY_32GF architecture from `Designing Network Design Spaces
    <https://arxiv.org/abs/2003.13678>`__.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/regnet_y_32gf-4dee3f7a.pth",
            "path"       : "regnet_y_32gf-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-v2": {
            "url"        : "https://download.pytorch.org/models/regnet_y_32gf-8db6d4b5.pth",
            "path"       : "regnet_y_32gf-imagenet1k-v2.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-swag-e2e-v1": {
            "url"        : "https://download.pytorch.org/models/regnet_y_32gf_swag-04fdfa75.pth",
            "path"       : "regnet_y_32gf_swag-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-swag-lc-v1": {
            "url"        : "https://download.pytorch.org/models/regnet_y_32gf_lc_swag-e1583746.pth",
            "path"       : "regnet_y_32gf_lc_swag-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "regnet",
        variant: str = "regnet_y_32gf",
        *args, **kwargs
    ):
        params     = BlockParams.from_init_params(depth=20, w_0=232, w_a=115.89, w_m=2.53, group_width=232, se_ratio=0.25, **kwargs)
        norm_layer = kwargs.pop("norm_layer", partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
        super().__init__(
            block_params = params,
            norm_layer   = norm_layer,
            name         = name,
            variant      = variant,
            *args, **kwargs
        )
    

@MODELS.register(name="regnet_y_128gf")
class RegNet_Y_128GF(RegNet):
    """RegNetY_128GF architecture from `Designing Network Design Spaces
    <https://arxiv.org/abs/2003.13678>`__.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-swag-e2e-v1": {
            "url"        : "https://download.pytorch.org/models/regnet_y_128gf_swag-c8ce3e52.pth",
            "path"       : "regnet_y_128gf_swag-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-swag-lc-v1": {
            "url"        : "https://download.pytorch.org/models/regnet_y_128gf_lc_swag-cbe8ce12.pth",
            "path"       : "regnet_y_128gf_lc_swag-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "regnet",
        variant: str = "regnet_y_128gf",
        *args, **kwargs
    ):
        params     = BlockParams.from_init_params(depth=27, w_0=456, w_a=160.83, w_m=2.52, group_width=264, se_ratio=0.25, **kwargs)
        norm_layer = kwargs.pop("norm_layer", partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
        super().__init__(
            block_params = params,
            norm_layer   = norm_layer,
            name         = name,
            variant      = variant,
            *args, **kwargs
        )
        
 
@MODELS.register(name="regnet_x_400mf")
class RegNet_X_400MF(RegNet):
    """RegNetY_400MF architecture from `Designing Network Design Spaces
    <https://arxiv.org/abs/2003.13678>`__.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/regnet_x_400mf-adf1edd5.pth",
            "path"       : "regnet_x_400mf-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-v2": {
            "url"        : "https://download.pytorch.org/models/regnet_x_400mf-62229a5f.pth",
            "path"       : "regnet_x_400mf-imagenet1k-v2.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "regnet",
        variant: str = "regnet_x_400mf",
        *args, **kwargs
    ):
        params     = BlockParams.from_init_params(depth=22, w_0=24, w_a=24.48, w_m=2.54, group_width=16, **kwargs)
        norm_layer = kwargs.pop("norm_layer", partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
        super().__init__(
            block_params = params,
            norm_layer   = norm_layer,
            name         = name,
            variant      = variant,
            *args, **kwargs
        )
   

@MODELS.register(name="regnet_x_800mf")
class RegNet_X_800MF(RegNet):
    """RegNetY_800MF architecture from `Designing Network Design Spaces
    <https://arxiv.org/abs/2003.13678>`__.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/regnet_x_800mf-ad17e45c.pth",
            "path"       : "regnet_x_800mf-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-v2": {
            "url"        : "https://download.pytorch.org/models/regnet_x_800mf-94a99ebd.pth",
            "path"       : "regnet_x_800mf-imagenet1k-v2.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "regnet",
        variant: str = "regnet_x_800mf",
        *args, **kwargs
    ):
        params     = BlockParams.from_init_params(depth=16, w_0=56, w_a=35.73, w_m=2.28, group_width=16, **kwargs)
        norm_layer = kwargs.pop("norm_layer", partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
        super().__init__(
            block_params = params,
            norm_layer   = norm_layer,
            name         = name,
            variant      = variant,
            *args, **kwargs
        )
  

@MODELS.register(name="regnet_x_1_6gf")
class RegNet_X_1_6GF(RegNet):
    """RegNetX_1.6GF architecture from `Designing Network Design Spaces
    <https://arxiv.org/abs/2003.13678>`__.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/regnet_x_1_6gf-e3633e7f.pth",
            "path"       : "regnet_x_1_6gf-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-v2": {
            "url"        : "https://download.pytorch.org/models/regnet_x_1_6gf-a12f2b72.pth",
            "path"       : "regnet_x_1_6gf-imagenet1k-v2.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "regnet",
        variant: str = "regnet_x_1_6gf",
        *args, **kwargs
    ):
        params     = BlockParams.from_init_params(depth=18, w_0=80, w_a=34.01, w_m=2.25, group_width=24, **kwargs)
        norm_layer = kwargs.pop("norm_layer", partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
        super().__init__(
            block_params = params,
            norm_layer   = norm_layer,
            name         = name,
            variant      = variant,
            *args, **kwargs
        )
      
      
@MODELS.register(name="regnet_x_3_2gf")
class RegNet_X_3_2GF(RegNet):
    """RegNetX_3.2GF architecture from `Designing Network Design Spaces
    <https://arxiv.org/abs/2003.13678>`__.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/regnet_x_3_2gf-f342aeae.pth",
            "path"       : "regnet_x_3_2gf-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-v2": {
            "url"        : "https://download.pytorch.org/models/regnet_x_3_2gf-7071aa85.pth",
            "path"       : "regnet_x_3_2gf-imagenet1k-v2.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "regnet",
        variant: str = "regnet_x_3_2gf",
        *args, **kwargs
    ):
        params     = BlockParams.from_init_params(depth=25, w_0=88, w_a=26.31, w_m=2.25, group_width=48, **kwargs)
        norm_layer = kwargs.pop("norm_layer", partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
        super().__init__(
            block_params = params,
            norm_layer   = norm_layer,
            name         = name,
            variant      = variant,
            *args, **kwargs
        )


@MODELS.register(name="regnet_x_8gf")
class RegNet_X_8GF(RegNet):
    """RegNetX_8GF architecture from `Designing Network Design Spaces
    <https://arxiv.org/abs/2003.13678>`__.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/regnet_x_8gf-03ceed89.pth",
            "path"       : "regnet_x_8gf-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-v2": {
            "url"        : "https://download.pytorch.org/models/regnet_x_8gf-2b70d774.pth",
            "path"       : "regnet_x_8gf-imagenet1k-v2.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "regnet",
        variant: str = "regnet_x_8gf",
        *args, **kwargs
    ):
        params     = BlockParams.from_init_params(depth=23, w_0=80, w_a=49.56, w_m=2.88, group_width=120, **kwargs)
        norm_layer = kwargs.pop("norm_layer", partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
        super().__init__(
            block_params = params,
            norm_layer   = norm_layer,
            name         = name,
            variant      = variant,
            *args, **kwargs
        )


@MODELS.register(name="regnet_x_16gf")
class RegNet_X_16GF(RegNet):
    """RegNetX_16GF architecture from `Designing Network Design Spaces
    <https://arxiv.org/abs/2003.13678>`__.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/regnet_x_16gf-2007eb11.pth",
            "path"       : "regnet_x_16gf-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-v2": {
            "url"        : "https://download.pytorch.org/models/regnet_x_16gf-ba3796d7.pth",
            "path"       : "regnet_x_16gf-imagenet1k-v2.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "regnet",
        variant: str = "regnet_x_16gf",
        *args, **kwargs
    ):
        params     = BlockParams.from_init_params(depth=22, w_0=216, w_a=55.59, w_m=2.1, group_width=128, **kwargs)
        norm_layer = kwargs.pop("norm_layer", partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
        super().__init__(
            block_params = params,
            norm_layer   = norm_layer,
            name         = name,
            variant      = variant,
            *args, **kwargs
        )


@MODELS.register(name="regnet_x32gf")
class RegNetX_32GF(RegNet):
    """RegNetX_32GF architecture from `Designing Network Design Spaces
    <https://arxiv.org/abs/2003.13678>`__.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/regnet_x_32gf-9d47f8d0.pth",
            "path"       : "regnet_x32gf-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-v2": {
            "url"        : "https://download.pytorch.org/models/regnet_x_32gf-6eb8fdc6.pth",
            "path"       : "regnet_x32gf-imagenet1k-v2.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "regnet",
        variant: str = "regnet_x32gf",
        *args, **kwargs
    ):
        params     = BlockParams.from_init_params(depth=23, w_0=320, w_a=69.86, w_m=2.0, group_width=168, **kwargs)
        norm_layer = kwargs.pop("norm_layer", partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
        super().__init__(
            block_params = params,
            norm_layer   = norm_layer,
            name         = name,
            variant      = variant,
            *args, **kwargs
        )
        
# endregion