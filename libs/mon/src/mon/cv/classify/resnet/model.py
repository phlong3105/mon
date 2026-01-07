#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ResNet model for image classification.

References:
    - Paper: https://arxiv.org/abs/1512.03385
"""

__all__ = [
    "ResNeXt101_32X8D",
    "ResNeXt101_64X4D",
    "ResNeXt50_32X4D",
    "ResNet101",
    "ResNet152",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "WideResNet101_2",
    "WideResNet50_2",
]

import abc
from typing import Any

import box
from torchvision import models as tvm
from torchvision.models.resnet import (
    _ovewrite_named_param,
    BasicBlock,
    Bottleneck,
)

from mon import nn
from mon.core import MLType, MODELS, Path, ROOT_DIR, Task

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


class ResNet(tvm.ResNet, nn.ModelMetadataMixin, abc.ABC):
    """ResNet model for image classification.

    References:
        - Paper: https://arxiv.org/abs/1512.03385
    """
    
    _arch     : str          = "resnet"
    _name     : str          = "resnet"
    _tasks    : list[Task]   = [Task.CLASSIFY]
    _mltypes  : list[MLType] = [MLType.SUPERVISED]
    _model_dir: Path         = root_dir
    _zoo      : dict         = box.Box()
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        weights, path, num_classes = self.parse_weights(weights, num_classes)
        super().__init__(num_classes=num_classes, *args, **kwargs)
        if weights:
            self.load_state_dict(weights)
            

@MODELS.register(variant="resnet18", name="resnet")
class ResNet18(ResNet):
    """ResNet-18 model for image classification.

    Args:
        num_classes: Number of output classes. Default: ``1000``.
    """
    
    _name: str  = "resnet18"
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/resnet18-f37072fd.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/resnet/resnet18/imagenet1k_v1/resnet18_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        super().__init__(
            block       = BasicBlock,
            layers      = [2, 2, 2, 2],
            weights     = weights,
            num_classes = num_classes,
            *args, **kwargs
        )
        

@MODELS.register(variant="resnet34", name="resnet")
class ResNet34(ResNet):
    """ResNet-34 model for image classification.

    Args:
        num_classes: Number of output classes. Default: ``1000``.
    """
    
    _name: str  = "resnet34"
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/resnet34-b627a593.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/resnet/resnet34/imagenet1k_v1/resnet34_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        super().__init__(
            block       = BasicBlock,
            layers      = [3, 4, 6, 3],
            weights     = weights,
            num_classes = num_classes,
            *args, **kwargs
        )


@MODELS.register(variant="resnet50", name="resnet")
class ResNet50(ResNet):
    """ResNet-50 model for image classification.

    Args:
        num_classes: Number of output classes. Default: ``1000``.
    """
    
    _name: str  = "resnet50"
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/resnet/resnet50/imagenet1k_v1/resnet50_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/resnet50-0676ba61.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/resnet/resnet50/imagenet1k_v2/resnet50_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        super().__init__(
            block       = Bottleneck,
            layers      = [3, 4, 6, 3],
            weights     = weights,
            num_classes = num_classes,
            *args, **kwargs
        )


@MODELS.register(variant="resnet101", name="resnet")
class ResNet101(ResNet):
    """ResNet-101 model for image classification.

    Args:
        num_classes: Number of output classes. Default: ``1000``.
    """
    
    _name: str  = "resnet101"
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/resnet101-63fe2227.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/resnet/resnet101/imagenet1k_v1/resnet101_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/resnet101-cd907fc2.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/resnet/resnet101/imagenet1k_v2/resnet101_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        super().__init__(
            block       = Bottleneck,
            layers      = [3, 4, 23, 3],
            weights     = weights,
            num_classes = num_classes,
            *args, **kwargs
        )


@MODELS.register(variant="resnet152", name="resnet")
class ResNet152(ResNet):
    """ResNet-152 model for image classification.

    Args:
        num_classes: Number of output classes. Default: ``1000``.
    """
    
    _name: str  = "resnet152"
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/resnet152-394f9c45.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/resnet/resnet152/imagenet1k_v1/resnet152_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/resnet152-f82ba261.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/resnet/resnet152/imagenet1k_v2/resnet152_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        super().__init__(
            block       = Bottleneck,
            layers      = [3, 8, 36, 3],
            weights     = weights,
            num_classes = num_classes,
            *args, **kwargs
        )
        

# --- ResNeXt ---
@MODELS.register(variant="resnext50_32x4d", name="resnet")
class ResNeXt50_32X4D(ResNet):
    """ResNeXt-50-32x4d model for image classification.

    Args:
        num_classes: Number of output classes. Default: ``1000``.
    """
    
    _name: str  = "resnext50_32x4d"
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/resnet/resnext50_32x4d/imagenet1k_v1/resnext50_32x4d_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/resnet/resnext50_32x4d/imagenet1k_v2/resnext50_32x4d_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        _ovewrite_named_param(kwargs, "groups", 32)
        _ovewrite_named_param(kwargs, "width_per_group", 4)
        super().__init__(
            block       = Bottleneck,
            layers      = [3, 4, 6, 3],
            weights     = weights,
            num_classes = num_classes,
            *args, **kwargs
        )


@MODELS.register(variant="resnext101_32x8d", name="resnet")
class ResNeXt101_32X8D(ResNet):
    """ResNeXt-101-32x8d model for image classification.

    Args:
        num_classes: Number of output classes. Default: ``1000``.
    """
    
    _name: str  = "resnext101_32x8d"
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/resnet/resnext101_32x8d/imagenet1k_v1/resnext101_32x8d_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/resnet/resnext101_32x8d/imagenet1k_v2/resnext101_32x8d_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        _ovewrite_named_param(kwargs, "groups", 32)
        _ovewrite_named_param(kwargs, "width_per_group", 8)
        super().__init__(
            block       = Bottleneck,
            layers      = [3, 4, 23, 3],
            weights     = weights,
            num_classes = num_classes,
            *args, **kwargs
        )


@MODELS.register(variant="resnext101_64x4d", name="resnet")
class ResNeXt101_64X4D(ResNet):
    """ResNeXt-101-64x4d model for image classification.

    Args:
        num_classes: Number of output classes. Default: ``1000``.
    """
    
    _name: str  = "resnext101_64x4d"
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/resnet/resnext101_64x4d/imagenet1k_v1/resnext101_64x4d_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        _ovewrite_named_param(kwargs, "groups", 64)
        _ovewrite_named_param(kwargs, "width_per_group", 4)
        super().__init__(
            block       = Bottleneck,
            layers      = [3, 4, 23, 3],
            weights     = weights,
            num_classes = num_classes,
            *args, **kwargs
        )
        

# --- WideResNet ---
@MODELS.register(variant="wide_resnet50_2", name="resnet")
class WideResNet50_2(ResNet):
    """WideResNet-50-2 model for image classification.

    Args:
        num_classes: Number of output classes. Default: ``1000``.
    """
    
    _name: str  = "wide_resnet50_2"
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/resnet/wide_resnet50/imagenet1k_v1/wide_resnet50_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/resnet/wide_resnet50/imagenet1k_v2/wide_resnet50_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
        super().__init__(
            block       = Bottleneck,
            layers      = [3, 4, 6, 3],
            weights     = weights,
            num_classes = num_classes,
            *args, **kwargs
        )


@MODELS.register(variant="wide_resnet101_2", name="resnet")
class WideResNet101_2(ResNet):
    """WideResNet-101-2 model for image classification.

    Args:
        num_classes: Number of output classes. Default: ``1000``.
    """
    
    _name: str  = "wide_resnet101_2"
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/resnet/wide_resnet101/imagenet1k_v1/wide_resnet101_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/resnet/wide_resnet101/imagenet1k_v2/wide_resnet101_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
        super().__init__(
            block       = Bottleneck,
            layers      = [3, 4, 23, 3],
            weights     = weights,
            num_classes = num_classes,
            *args, **kwargs
        )
