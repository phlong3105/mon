#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DualPathNetworks (version 1).
Ported by timm (https://github.com/rwightman/pytorch-dpn-pretrained)

Based on original MXNet implementation https://github.com/cypw/DPNs with
many ideas from another PyTorch implementation https://github.com/oyam/pytorch-DPNs.

This implementation is compatible with the pretrained weights from cypw's MXNet
implementation.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from one.core import BACKBONES
from one.core import Indexes
from one.core import Int2T
from one.core import ListOrTupleAnyT
from one.core import MODELS
from one.core import Pretrained
from one.core import Tensors
from one.nn import BatchNormAct2d
from one.nn import ConvBnAct
from one.nn import create_classifier
from one.nn import create_conv2d
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
    
    def __init__(self, in_channels: int, norm_layer=BatchNormAct2d):
        super().__init__()
        self.bn = norm_layer(in_channels, eps=0.001)
    
    # MARK: Forward pass
    
    @torch.jit._overload_method  # noqa: F811
    def forward(self, x: Tensors) -> Tensor:
        # type: (tuple[torch.Tensor, torch.Tensor]) -> (torch.Tensor)
        pass
    
    @torch.jit._overload_method  # noqa: F811
    def forward(self, x: Tensors) -> Tensor:
        # type: (torch.Tensor) -> (torch.Tensor)
        pass
    
    def forward(self, x: Tensors) -> Tensor:
        if isinstance(x, (tuple, list)):
            x = torch.cat(x, dim=1)
        return self.bn(x)


class BnActConv2d(nn.Module):
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Int2T,
        stride      : Int2T,
        groups      : int       = 1,
        norm_layer  : nn.Module = BatchNormAct2d
    ):
        super().__init__()
        self.bn   = norm_layer(in_channels, eps=0.001)
        self.conv = create_conv2d(
            in_channels, out_channels, kernel_size, stride=stride, groups=groups
        )
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.bn(x))


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
            self.has_proj = True
        elif block_type == "down":
            self.key_stride = 2
            self.has_proj   = True
        else:
            assert block_type == "normal"
            self.key_stride = 1
            self.has_proj   = False

        self.c1x1_w_s1 = None
        self.c1x1_w_s2 = None
        if self.has_proj:
            # Using different member names here to allow easier parameter key
            # matching for conversion
            if self.key_stride == 2:
                self.c1x1_w_s2 = BnActConv2d(
                    in_channels  = in_channels,
                    out_channels = num_1x1_c + 2 * inc,
                    kernel_size  = 1,
                    stride       = 2
                )
            else:
                self.c1x1_w_s1 = BnActConv2d(
                    in_channels  = in_channels,
                    out_channels = num_1x1_c + 2 * inc,
                    kernel_size  = 1,
                    stride       = 1
                )

        self.c1x1_a = BnActConv2d(
            in_channels  = in_channels,
            out_channels = num_1x1_a,
            kernel_size  = 1,
            stride       = 1
        )
        self.c3x3_b = BnActConv2d(
            in_channels  = num_1x1_a,
            out_channels = num_3x3_b,
            kernel_size  = 3,
            stride       = self.key_stride,
            groups       = groups
        )
        if b:
            self.c1x1_c  = CatBnAct(in_channels=num_3x3_b)
            self.c1x1_c1 = create_conv2d(num_3x3_b, num_1x1_c, kernel_size=1)
            self.c1x1_c2 = create_conv2d(num_3x3_b, inc,       kernel_size=1)
        else:
            self.c1x1_c  = BnActConv2d(
                in_channels  = num_3x3_b,
                out_channels = num_1x1_c + inc,
                kernel_size  = 1,
                stride       = 1
            )
            self.c1x1_c1 = None
            self.c1x1_c2 = None

    @torch.jit._overload_method  # noqa: F811
    def forward(self, x: Tensors) -> Tensors:
        # type: (tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, x: Tensors) -> Tensors:
        # type: (torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        pass

    def forward(self, x: Tensors) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, tuple):
            x_in = torch.cat(x, dim=1)
        else:
            x_in = x
        if self.c1x1_w_s1 is None and self.c1x1_w_s2 is None:
            # self.has_proj == False, torchscript requires condition on module == None
            x_s1 = x[0]
            x_s2 = x[1]
        else:
            # self.has_proj == True
            if self.c1x1_w_s1 is not None:
                # self.key_stride = 1
                x_s = self.c1x1_w_s1(x_in)
            else:
                # self.key_stride = 2
                x_s = self.c1x1_w_s2(x_in)
            x_s1 = x_s[:, :self.num_1x1_c,  :, :]
            x_s2 = x_s[:,  self.num_1x1_c:, :, :]
       
        x_in = self.c1x1_a(x_in)
        x_in = self.c3x3_b(x_in)
        x_in = self.c1x1_c(x_in)
        if self.c1x1_c1 is not None:
            # self.b == True, using None check for torchscript compat
            out1 = self.c1x1_c1(x_in)
            out2 = self.c1x1_c2(x_in)
        else:
            out1 = x_in[:, :self.num_1x1_c,  :, :]
            out2 = x_in[:,  self.num_1x1_c:, :, :]
        
        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)
        return resid, dense
    

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
        k_sec             : ListOrTupleAnyT[int] = (3, 4, 20, 3),
        inc_sec           : ListOrTupleAnyT[int] = (16, 32, 24, 128),
        output_stride     : int                  = 32,
        in_channels       : int                  = 3,
        drop_rate         : float                = 0.0,
        global_pool       : str                  = "avg",
        fc_act                                   = nn.ELU,
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
        self.drop_rate = drop_rate
        self.b         = b
        if output_stride != 32:  # FIXME look into dilation support
            raise ValueError()
        
        norm_layer    = partial(BatchNormAct2d, eps=0.001)
        fc_norm_layer = partial(BatchNormAct2d, eps=0.001, act_layer=fc_act,
                                inplace=False)
        bw_factor     = 1 if small else 4
        
        # NOTE: Features
        blocks = OrderedDict()

        # conv1
        blocks["conv1_1"] = ConvBnAct(
            in_channels, num_init_features,
            kernel_size = 3 if small else 7,
            stride      = 2,
            norm_layer  = norm_layer
        )
        blocks["conv1_pool"] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.feature_info    = [dict(
            num_chs=num_init_features, reduction=2, module="features.conv1_1"
        )]

        # conv2
        bw  = 64 * bw_factor
        inc = inc_sec[0]
        r   = (k_r * bw) // (64 * bw_factor)
        blocks["conv2_1"] = DualPathBlock(
            num_init_features, r, r, bw, inc, groups, "proj", b
        )
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0] + 1):
            blocks["conv2_" + str(i)] = DualPathBlock(
                in_channels, r, r, bw, inc, groups, "normal", b
            )
            in_chs += inc
        self.feature_info += [dict(
            num_chs=in_channels, reduction=4, module=f"features.conv2_{k_sec[0]}"
        )]

        # conv3
        bw  = 128 * bw_factor
        inc = inc_sec[1]
        r   = (k_r * bw) // (64 * bw_factor)
        blocks["conv3_1"] = DualPathBlock(
            in_channels, r, r, bw, inc, groups, "down", b
        )
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1] + 1):
            blocks["conv3_" + str(i)] = DualPathBlock(
                in_channels, r, r, bw, inc, groups, "normal", b
            )
            in_chs += inc
        self.feature_info += [dict(
            num_chs=in_channels, reduction=8, module=f"features.conv3_{k_sec[1]}"
        )]

        # conv4
        bw  = 256 * bw_factor
        inc = inc_sec[2]
        r   = (k_r * bw) // (64 * bw_factor)
        blocks["conv4_1"] = DualPathBlock(
            in_channels, r, r, bw, inc, groups, "down", b
        )
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2] + 1):
            blocks["conv4_" + str(i)] = DualPathBlock(
                in_channels, r, r, bw, inc, groups, "normal", b
            )
            in_chs += inc
        self.feature_info += [dict(
            num_chs=in_channels, reduction=16, module=f"features.conv4_{k_sec[2]}"
        )]

        # conv5
        bw  = 512 * bw_factor
        inc = inc_sec[3]
        r   = (k_r * bw) // (64 * bw_factor)
        blocks["conv5_1"] = DualPathBlock(
            in_channels, r, r, bw, inc, groups, "down", b
        )
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3] + 1):
            blocks["conv5_" + str(i)] = DualPathBlock(
                in_channels, r, r, bw, inc, groups, "normal", b
            )
            in_chs += inc
        self.feature_info += [dict(
            num_chs=in_channels, reduction=32, module=f"features.conv5_{k_sec[3]}"
        )]

        blocks["conv5_bn_ac"] = CatBnAct(in_channels, norm_layer=fc_norm_layer)

        self.num_features = in_channels
        self.features     = nn.Sequential(blocks)
        
        # NOTE: Classifier
        # Using 1x1 conv for the FC layer to allow the extra pooling scheme
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool,
            use_conv=True
        )
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()
    
    # MARK: Configure

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool="avg"):
        self.num_classes = num_classes
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool, use_conv=True)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()
    
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
        x = self.global_pool(x)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        x = self.flatten(x)
        return x


# MARK: - DPN68

@MODELS.register(name="dpn68")
@BACKBONES.register(name="dpn68")
class DPN68(DPN):
    
    model_zoo = {
        "imagenet": dict(
            path="https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn68-66bebafa7.pth",
            file_name="dpn68_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
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
        "imagenet": dict(
            path="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/dpn68b_ra-a31ca160.pth",
            file_name="dpn68b_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
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
        "imagenet": dict(
            path="https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn92_extra-b040e4a9b.pth",
            file_name="dpn92_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
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
            path="https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn98-5b90dec4d.pth",
            file_name="dpn98_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
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
        "imagenet": dict(
            path="https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn107_extra-1ac7121e2.pth",
            file_name="dpn107_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
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
            path="https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn131-71dfe43e0.pth",
            file_name="dpn131_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
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
