#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DualPathNetworks (version 2).
Ported by pretrainedmodels

Based on original MXNet implementation https://github.com/cypw/DPNs with
many ideas from another PyTorch implementation https://github.com/oyam/pytorch-DPNs.

This implementation is compatible with the pretrained weights from cypw's MXNet
implementation.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from one.core import BACKBONES
from one.core import console
from one.core import Indexes
from one.core import Int2T
from one.core import ListOrTupleAnyT
from one.core import MODELS
from one.core import Padding4T
from one.core import Pretrained
from one.core import to_2tuple
from one.vision.image_classification.image_classifier import ImageClassifier

__all__ = [
    "DPN",
    "DPN68",
    "DPN68b",
    "DPN92",
    "DPN98",
    "DPN107",
    "DPN131",
]


# MARK: - Modules

class CatBnAct(nn.Module):
    
    # MARK: Magic Functions
    
    def __init__(self, in_channels: int, activation_fn=nn.ReLU(inplace=True)):
        super().__init__()
        self.bn  = nn.BatchNorm2d(in_channels, eps=0.001)
        self.act = activation_fn
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        x = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        return self.act(self.bn(x))


class BnActConv2d(nn.Module):
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Int2T,
        stride      : Int2T,
        padding     : Padding4T = 0,
        groups      : int       = 1,
        activation_fn           = nn.ReLU(inplace=True)
    ):
        super().__init__()
        self.bn     = nn.BatchNorm2d(in_channels, eps=0.001)
        self.act    = activation_fn
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        self.conv   = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            groups=groups, bias=False
        )

    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.act(self.bn(x)))


class InputBlock(nn.Module):
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        num_init_features: int,
        kernel_size      : Int2T    = 7,
        padding          : Padding4T = 3,
        activation_fn                = nn.ReLU(inplace=True)
    ):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        self.conv   = nn.Conv2d(
            3, num_init_features, kernel_size=kernel_size, stride=(2, 2),
            padding=padding, bias=False
        )
        self.bn     = nn.BatchNorm2d(num_init_features, eps=0.001)
        self.act    = activation_fn
        self.pool   = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class DualPathBlock(nn.Module):
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels: int,
        num_1x1_a  : int,
        num_3x3_b  : int,
        num_1x1_c  : int,
        inc        : int,
        groups     : int,
        block_type : str  = "normal",
        b          : bool = False
    ):
        super().__init__()
        self.num_1x1_c = num_1x1_c
        self.inc       = inc
        self.b         = b
        if block_type == "proj":
            self.key_stride = 1
            self.has_proj   = True
        elif block_type == "down":
            self.key_stride = 2
            self.has_proj   = True
        else:
            assert block_type == "normal"
            self.key_stride = 1
            self.has_proj   = False

        if self.has_proj:
            # Using different member names here to allow easier parameter key
            # matching for conversion
            if self.key_stride == 2:
                self.c1x1_w_s2 = BnActConv2d(
                    in_channels=in_channels, out_channels=num_1x1_c + 2 * inc,
                    kernel_size=1, stride=2
                )
            else:
                self.c1x1_w_s1 = BnActConv2d(
                    in_channels=in_channels, out_channels=num_1x1_c + 2 * inc,
                    kernel_size=1, stride=1
                )
        self.c1x1_a = BnActConv2d(
            in_channels=in_channels, out_channels=num_1x1_a, kernel_size=1,
            stride=1
        )
        self.c3x3_b = BnActConv2d(
            in_channels=num_1x1_a, out_channels=num_3x3_b, kernel_size=3,
            stride=self.key_stride, padding=1, groups=groups)
        if b:
            self.c1x1_c  = CatBnAct(in_channels=num_3x3_b)
            self.c1x1_c1 = nn.Conv2d(
                num_3x3_b, num_1x1_c, kernel_size=(1, 1), bias=False
            )
            self.c1x1_c2 = nn.Conv2d(
                num_3x3_b, inc, kernel_size=(1, 1), bias=False
            )
        else:
            self.c1x1_c  = BnActConv2d(
                in_channels=num_3x3_b, out_channels=num_1x1_c + inc,
                kernel_size=1, stride=1
            )
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        if self.has_proj:
            if self.key_stride == 2:
                x_s = self.c1x1_w_s2(x_in)
            else:
                x_s = self.c1x1_w_s1(x_in)
            x_s1 = x_s[:, :self.num_1x1_c,  :, :]
            x_s2 = x_s[:,  self.num_1x1_c:, :, :]
        else:
            x_s1 = x[0]
            x_s2 = x[1]
        x_in = self.c1x1_a(x_in)
        x_in = self.c3x3_b(x_in)
        if self.b:
            x_in = self.c1x1_c(x_in)
            out1 = self.c1x1_c1(x_in)
            out2 = self.c1x1_c2(x_in)
        else:
            x_in = self.c1x1_c(x_in)
            out1 = x_in[:, :self.num_1x1_c,  :, :]
            out2 = x_in[:,  self.num_1x1_c:, :, :]
        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)
        return resid, dense


def pooling_factor(pool_type: str = "avg") -> int:
    return 2 if pool_type == "avgmaxc" else 1


def adaptive_avgmax_pool2d(
    x            : Tensor,
    pool_type        : str       = "avg",
    padding          : Padding4T = 0,
    count_include_pad: bool      = False
) -> Tensor:
    """Selectable global pooling function with dynamic input kernel size.
    """
    if pool_type == "avgmaxc":
        x = torch.cat([
            F.avg_pool2d(
                x,
                kernel_size       = (x.size(2), x.size(3)),
                padding           = padding,
                count_include_pad = count_include_pad),
            F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=padding)
        ], dim=1)
    elif pool_type == "avgmax":
        x_avg = F.avg_pool2d(
            x,
            kernel_size       = (x.size(2), x.size(3)),
            padding           = padding,
            count_include_pad = count_include_pad
        )
        x_max = F.max_pool2d(
            x, kernel_size=(x.size(2), x.size(3)), padding=padding
        )
        x     = 0.5 * (x_avg + x_max)
    elif pool_type == "max":
        x = F.max_pool2d(
            x, kernel_size=(x.size(2), x.size(3)), padding=padding
        )
    else:
        if pool_type != "avg":
            console.log("Invalid pool type %s specified. Defaulting to "
                        "average pooling." % pool_type)
        x = F.avg_pool2d(
            x,
            kernel_size       = (x.size(2), x.size(3)),
            padding           = padding,
            count_include_pad = count_include_pad
        )
    return x


class AdaptiveAvgMaxPool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size.
    """
    
    # MARK: Magic Functions
    
    def __init__(self, output_size: int = 1, pool_type: str = "avg"):
        super().__init__()
        self.output_size = output_size
        self.pool_type   = pool_type
        if pool_type == "avgmaxc" or pool_type == "avgmax":
            self.pool = nn.ModuleList([
                nn.AdaptiveAvgPool2d(output_size),
                nn.AdaptiveMaxPool2d(output_size)
            ])
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            if pool_type != "avg":
                console.log("Invalid pool type %s specified. Defaulting to "
                            "average pooling." % pool_type)
            self.pool = nn.AdaptiveAvgPool2d(output_size)

    def __repr__(self):
        return self.__class__.__name__ + " (" \
               + "output_size=" + str(self.output_size) \
               + ", pool_type=" + self.pool_type + ")"

    # MARK: Configure
    
    def factor(self):
        return pooling_factor(self.pool_type)

    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        if self.pool_type == "avgmaxc":
            x = torch.cat([p(x) for p in self.pool], dim=1)
        elif self.pool_type == "avgmax":
            x = 0.5 * torch.sum(
                torch.stack([p(x) for p in self.pool]), 0
            ).squeeze(dim=0)
        else:
            x = self.pool(x)
        return x


# MARK: - DPN

cfgs = {
    "dpn68": {
        "small": True, "num_init_features": 10, "k_r": 128, "groups": 32,
        "k_sec": (3, 4, 12, 3), "inc_sec": (16, 32, 32, 64),
    },
    "dpn68b": {
        "small": True, "num_init_features": 10, "k_r": 128, "groups": 32,
        "b": True, "k_sec": (3, 4, 12, 3), "inc_sec": (16, 32, 32, 64),
    },
    "dpn92": {
        "num_init_features": 64, "k_r": 96, "groups": 32,
        "k_sec": (3, 4, 20, 3), "inc_sec": (16, 32, 24, 128),
    },
    "dpn98": {
        "num_init_features": 96, "k_r": 160, "groups": 40,
        "k_sec": (3, 6, 20, 3), "inc_sec": (16, 32, 32, 128),
    },
    "dpn107": {
        "num_init_features": 128, "k_r": 200, "groups": 50,
        "k_sec": (4, 8, 20, 3), "inc_sec": (20, 64, 64, 128),
    },
    "dpn131": {
        "num_init_features": 128, "k_r": 160, "groups": 40,
        "k_sec": (4, 8, 28, 3), "inc_sec": (16, 32, 32, 128),
    },
}


@MODELS.register(name="dpn")
@BACKBONES.register(name="dpn")
class DPN(ImageClassifier):
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # Hyperparameters
        small             : bool                 = False,
        num_init_features : int                  = 64,
        k_r               : int                  = 96,
        groups            : int                  = 32,
        b                 : bool                 = False,
        k_sec             : ListOrTupleAnyT[int] = (3, 4 , 20, 3),
        inc_sec           : ListOrTupleAnyT[int] = (16, 32, 24, 128),
        test_time_pool    : bool                 = False,
        # BaseModel's args
        basename          : Optional[str] = "dpn",
        name              : Optional[str] = "dpn",
        num_classes       : Optional[int] = None,
        out_indexes       : Indexes       = -1,
        pretrained        : Pretrained    = False,
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
        self.test_time_pool = test_time_pool
        self.b              = b
        bw_factor           = 1 if small else 4

        blocks = OrderedDict()

        # conv1
        if small:
            blocks["conv1_1"] = InputBlock(
                num_init_features, kernel_size=3, padding=1
            )
        else:
            blocks["conv1_1"] = InputBlock(
                num_init_features, kernel_size=7, padding=3
            )

        # conv2
        bw  = 64 * bw_factor
        inc = inc_sec[0]
        r   = (k_r * bw) // (64 * bw_factor)
        blocks["conv2_1"] = DualPathBlock(
            num_init_features, r, r, bw, inc, groups, "proj", b
        )
        in_channels = bw + 3 * inc
        for i in range(2, k_sec[0] + 1):
            blocks["conv2_" + str(i)] = DualPathBlock(
                in_channels, r, r, bw, inc, groups, "normal", b
            )
            in_channels += inc

        # conv3
        bw  = 128 * bw_factor
        inc = inc_sec[1]
        r   = (k_r * bw) // (64 * bw_factor)
        blocks["conv3_1"] = DualPathBlock(
            in_channels, r, r, bw, inc, groups, "down", b
        )
        in_channels = bw + 3 * inc
        for i in range(2, k_sec[1] + 1):
            blocks["conv3_" + str(i)] = DualPathBlock(
                in_channels, r, r, bw, inc, groups, "normal", b
            )
            in_channels += inc

        # conv4
        bw  = 256 * bw_factor
        inc = inc_sec[2]
        r   = (k_r * bw) // (64 * bw_factor)
        blocks["conv4_1"] = DualPathBlock(
            in_channels, r, r, bw, inc, groups, "down", b
        )
        in_channels = bw + 3 * inc
        for i in range(2, k_sec[2] + 1):
            blocks["conv4_" + str(i)] = DualPathBlock(
                in_channels, r, r, bw, inc, groups, "normal", b
            )
            in_channels += inc

        # conv5
        bw  = 512 * bw_factor
        inc = inc_sec[3]
        r   = (k_r * bw) // (64 * bw_factor)
        blocks["conv5_1"] = DualPathBlock(
            in_channels, r, r, bw, inc, groups, "down", b
        )
        in_channels = bw + 3 * inc
        for i in range(2, k_sec[3] + 1):
            blocks["conv5_" + str(i)] = DualPathBlock(
                in_channels, r, r, bw, inc, groups, "normal", b
            )
            in_channels += inc
        blocks["conv5_bn_ac"] = CatBnAct(in_channels)

        self.features = nn.Sequential(blocks)

        # Using 1x1 conv for the FC layer to allow the extra pooling scheme
        self.last_linear = self.create_classifier(in_channels, self.num_classes)
        self.classifier  = self.last_linear

        # NOTE: Load Pretrained
        if self.pretrained:
            self.load_pretrained()
        
    # MARK: Configure
    
    @staticmethod
    def create_classifier(
        num_features: int, num_classes: Optional[int]
    ) -> nn.Module:
        if num_classes and num_classes > 0:
            classifier = nn.Conv2d(
                num_features, num_classes, kernel_size=(1, 1), bias=True
            )
        else:
            classifier = nn.Identity()
        return classifier
    
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
        if not self.training and self.test_time_pool:
            x   = F.avg_pool2d(x, kernel_size=7, stride=1)
            out = self.last_linear(x)
            # The extra test time pool should be pooling an img_size//32 - 6
            # size patch
            out = adaptive_avgmax_pool2d(out, pool_type="avgmax")
        else:
            x   = adaptive_avgmax_pool2d(x, pool_type="avg")
            out = self.last_linear(x)
        return out.view(out.size(0), -1)


# MARK: - DPN68

@MODELS.register(name="dpn68")
@BACKBONES.register(name="dpn68")
class DPN68(DPN):
    
    model_zoo = {
        "imagenet": dict(
            path="http://data.lip6.fr/cadene/pretrainedmodels/dpn68-4af7d88d2.pth",
            file_name="dpn68_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "dpn68",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["dpn68"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - DPN68b

@MODELS.register(name="dpn68b")
@BACKBONES.register(name="dpn68b")
class DPN68b(DPN):
    
    model_zoo = {
        "imagenet+5k": dict(
            path="http://data.lip6.fr/cadene/pretrainedmodels/dpn68b_extra-363ab9c19.pth",
            file_name="dpn68b_imagenet5k.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "dpn68b",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["dpn68b"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - DPN92

@MODELS.register(name="dpn92")
@BACKBONES.register(name="dpn92")
class DPN92(DPN):
    
    model_zoo = {
        "imagenet+5k": dict(
            path="http://data.lip6.fr/cadene/pretrainedmodels/dpn92_extra-fda993c95.pth",
            file_name="dpn92_imagenet5k.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "dpn92",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["dpn92"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - DPN98

@MODELS.register(name="dpn98")
@BACKBONES.register(name="dpn98")
class DPN98(DPN):
    
    model_zoo = {
        "imagenet": dict(
            path="http://data.lip6.fr/cadene/pretrainedmodels/dpn98-722954780.pth",
            file_name="dpn98_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "dpn98",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["dpn98"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - DPN107

@MODELS.register(name="dpn107")
@BACKBONES.register(name="dpn107")
class DPN107(DPN):
    
    model_zoo = {
        "imagenet+5k": dict(
            path="http://data.lip6.fr/cadene/pretrainedmodels/dpn107_extra-b7f9f4cc9.pth",
            file_name="dpn107_imagenet5k.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "dpn107",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["dpn107"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - DPN131

@MODELS.register(name="dpn131")
@BACKBONES.register(name="dpn131")
class DPN131(DPN):
    
    model_zoo = {
        "imagenet": dict(
            path="http://data.lip6.fr/cadene/pretrainedmodels/dpn131-7af84be88.pth",
            file_name="dpn131_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "dpn131",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["dpn131"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )
