#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements EfficientNet model for image classification.

References:
    - Paper: https://arxiv.org/abs/1905.11946
"""

__all__ = [
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

import abc
from functools import partial
from typing import Any

import box
from torchvision import models as tvm
from torchvision.models.efficientnet import _efficientnet_conf

from mon.constants import MODELS, ROOT_DIR
from mon.core import MLType, ModelMixin, nn, Path, Task

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


class EfficientNet(tvm.EfficientNet, ModelMixin, abc.ABC):
    """EfficientNet model for image classification.

    References:
        - https://arxiv.org/abs/1905.11946
    """
    
    arch     : str          = "efficientnet"
    name     : str          = "efficientnet",
    tasks    : list[Task]   = [Task.CLASSIFY]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        weights, path, num_classes = self.parse_weights(weights, num_classes)
        super().__init__(num_classes=num_classes, *args, **kwargs)
        if weights:
            self.load_state_dict(weights)
    

@MODELS.register(name="efficientnet_b0", arch="efficientnet")
class EfficientNet_B0(EfficientNet):
    
    name: str  = "efficientnet_b0"
    zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/efficientnet/efficientnet_b0/imagenet1k_v1/efficientnet_b0_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)
        dropout = kwargs.pop("dropout", 0.2)
        super().__init__(
            inverted_residual_setting = inverted_residual_setting,
            dropout                   = dropout,
            last_channel              = last_channel,
            weights                   = weights,
            num_classes               = num_classes,
            *args, **kwargs
        )


@MODELS.register(name="efficientnet_b1", arch="efficientnet")
class EfficientNet_B1(EfficientNet):
    
    name: str  = "efficientnet_b1"
    zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b1_rwightman-bac287d4.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/efficientnet/efficientnet_b1/imagenet1k_v1/efficientnet_b1_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b1-c27df63c.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/efficientnet/efficientnet_b1/imagenet1k_v2/efficientnet_b1_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b1", width_mult=1.0, depth_mult=1.1)
        dropout = kwargs.pop("dropout", 0.2)
        super().__init__(
            inverted_residual_setting = inverted_residual_setting,
            dropout                   = dropout,
            last_channel              = last_channel,
            weights                   = weights,
            num_classes               = num_classes,
            *args, **kwargs
        )


@MODELS.register(name="efficientnet_b2", arch="efficientnet")
class EfficientNet_B2(EfficientNet):

    name: str  = "efficientnet_b2"
    zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b2_rwightman-c35c1473.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/efficientnet/efficientnet_b2/imagenet1k_v1/efficientnet_b2_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b2", width_mult=1.1, depth_mult=1.2)
        dropout = kwargs.pop("dropout", 0.3)
        super().__init__(
            inverted_residual_setting = inverted_residual_setting,
            dropout                   = dropout,
            last_channel              = last_channel,
            weights                   = weights,
            num_classes               = num_classes,
            *args, **kwargs
        )


@MODELS.register(name="efficientnet_b3", arch="efficientnet")
class EfficientNet_B3(EfficientNet):

    name: str  = "efficientnet_b3"
    zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b3_rwightman-b3899882.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/efficientnet/efficientnet_b3/imagenet1k_v1/efficientnet_b3_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b3", width_mult=1.2, depth_mult=1.4)
        dropout = kwargs.pop("dropout", 0.3)
        super().__init__(
            inverted_residual_setting = inverted_residual_setting,
            dropout                   = dropout,
            last_channel              = last_channel,
            weights                   = weights,
            num_classes               = num_classes,
            *args, **kwargs
        )


@MODELS.register(name="efficientnet_b4", arch="efficientnet")
class EfficientNet_B4(EfficientNet):
    
    name: str  = "efficientnet_b4"
    zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b4_rwightman-23ab8bcd.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/efficientnet/efficientnet_b4/imagenet1k_v1/efficientnet_b4_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b4", width_mult=1.4, depth_mult=1.8)
        dropout = kwargs.pop("dropout", 0.4)
        super().__init__(
            inverted_residual_setting = inverted_residual_setting,
            dropout                   = dropout,
            last_channel              = last_channel,
            weights                   = weights,
            num_classes               = num_classes,
            *args, **kwargs
        )


@MODELS.register(name="efficientnet_b5", arch="efficientnet")
class EfficientNet_B5(EfficientNet):

    name: str  = "efficientnet_b5"
    zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b5_lukemelas-1a07897c.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/efficientnet/efficientnet_b5/imagenet1k_v1/efficientnet_b5_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b5", width_mult=1.6, depth_mult=2.2)
        dropout    = kwargs.pop("dropout", 0.4)
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        super().__init__(
            inverted_residual_setting = inverted_residual_setting,
            dropout                   = dropout,
            last_channel              = last_channel,
            norm_layer                = norm_layer,
            weights                   = weights,
            num_classes               = num_classes,
            *args, **kwargs
        )


@MODELS.register(name="efficientnet_b6", arch="efficientnet")
class EfficientNet_B6(EfficientNet):

    name: str  = "efficientnet_b6"
    zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b6_lukemelas-24a108a5.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/efficientnet/efficientnet_b6/imagenet1k_v1/efficientnet_b6_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b6", width_mult=1.8, depth_mult=2.6)
        dropout    = kwargs.pop("dropout", 0.5)
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        super().__init__(
            inverted_residual_setting = inverted_residual_setting,
            dropout                   = dropout,
            last_channel              = last_channel,
            norm_layer                = norm_layer,
            weights                   = weights,
            num_classes               = num_classes,
            *args, **kwargs
        )
   
   
@MODELS.register(name="efficientnet_b7", arch="efficientnet")
class EfficientNet_B7(EfficientNet):

    name: str  = "efficientnet_b7"
    zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b7_lukemelas-c5b4e57e.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/efficientnet/efficientnet_b7/imagenet1k_v1/efficientnet_b7_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b7", width_mult=2.0, depth_mult=3.1)
        dropout    = kwargs.pop("dropout", 0.5)
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        super().__init__(
            inverted_residual_setting = inverted_residual_setting,
            dropout                   = dropout,
            last_channel              = last_channel,
            norm_layer                = norm_layer,
            weights                   = weights,
            num_classes               = num_classes,
            *args, **kwargs
        )


@MODELS.register(name="efficientnet_v2_s", arch="efficientnet")
class EfficientNet_V2_S(EfficientNet):

    name: str  = "efficientnet_v2_s"
    zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/efficientnet/efficientnet_v2_s/imagenet1k_v1/efficientnet_v2_s_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_s")
        dropout    = kwargs.pop("dropout", 0.2)
        norm_layer = partial(nn.BatchNorm2d, eps=1e-03)
        super().__init__(
            inverted_residual_setting = inverted_residual_setting,
            dropout                   = dropout,
            last_channel              = last_channel,
            norm_layer                = norm_layer,
            weights                   = weights,
            num_classes               = num_classes,
            *args, **kwargs
        )


@MODELS.register(name="efficientnet_v2_m", arch="efficientnet")
class EfficientNet_V2_M(EfficientNet):

    name: str  = "efficientnet_v2_m"
    zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_v2_m-dc08266a.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/efficientnet/efficientnet_v2_m/imagenet1k_v1/efficientnet_v2_m_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_m")
        dropout    = kwargs.pop("dropout", 0.3)
        norm_layer = partial(nn.BatchNorm2d, eps=1e-03)
        super().__init__(
            inverted_residual_setting = inverted_residual_setting,
            dropout                   = dropout,
            last_channel              = last_channel,
            norm_layer                = norm_layer,
            weights                   = weights,
            num_classes               = num_classes,
            *args, **kwargs
        )


@MODELS.register(name="efficientnet_v2_l", arch="efficientnet")
class EfficientNet_V2_L(EfficientNet):

    name: str  = "efficientnet_v2_l"
    zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_v2_l-59c71312.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/efficientnet/efficientnet_v2_l/imagenet1k_v1/efficientnet_v2_l_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_l")
        dropout    = kwargs.pop("dropout", 0.4)
        norm_layer = partial(nn.BatchNorm2d, eps=1e-03)
        super().__init__(
            inverted_residual_setting = inverted_residual_setting,
            dropout                   = dropout,
            last_channel              = last_channel,
            norm_layer                = norm_layer,
            weights                   = weights,
            num_classes               = num_classes,
            *args, **kwargs
        )
