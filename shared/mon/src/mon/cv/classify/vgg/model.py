#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements VGG model for image classification.

References:
    - Paper: https://arxiv.org/abs/1409.1556
"""

__all__ = [
    "VGG11",
    "VGG11_BN",
    "VGG13",
    "VGG13_BN",
    "VGG16",
    "VGG16_BN",
    "VGG19",
    "VGG19_BN",
]

import abc
from typing import Any

import box
from torchvision import models as tvm
from torchvision.models.vgg import cfgs, make_layers

from mon import nn
from mon.core import MLType, MODELS, Path, ROOT_DIR, Task

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


class VGG(tvm.VGG, nn.ModelMetadataMixin, abc.ABC):
    """VGG model for image classification.

    References:
        - Paper: https://arxiv.org/abs/1409.1556
    """
    
    _arch     : str          = "vgg"
    _name     : str          = "vgg"
    _tasks    : list[Task]   = [Task.CLASSIFY]
    _mltypes  : list[MLType] = [MLType.SUPERVISED]
    _model_dir: Path         = root_dir
    _zoo      : dict         = box.Box()
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        weights, path, num_classes = self.parse_weights(weights, num_classes)
        super().__init__(num_classes=num_classes, *args, **kwargs)
        if weights:
            self.load_state_dict(weights)
    

@MODELS.register(name="vgg11", arch="vgg")
class VGG11(VGG):

    _name: str  = "vgg11"
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vgg11-8a719046.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/vgg/vgg11/imagenet1k_v1/vgg11_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        super().__init__(
            features    = make_layers(cfgs["A"], batch_norm=False),
            weights     = weights,
            num_classes = num_classes,
            *args, **kwargs
        )
        

@MODELS.register(name="vgg13", arch="vgg")
class VGG13(VGG):
    
    _name: str  = "vgg13"
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vgg13-19584684.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/vgg/vgg13/imagenet1k_v1/vgg13_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        super().__init__(
            features    = make_layers(cfgs["B"], batch_norm=False),
            weights     = weights,
            num_classes = num_classes,
            *args, **kwargs
        )
    

@MODELS.register(name="vgg16", arch="vgg")
class VGG16(VGG):

    _name: str  = "vgg16"
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vgg16-397923af.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/vgg/vgg16/imagenet1k_v1/vgg16_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        super().__init__(
            features    = make_layers(cfgs["D"], batch_norm=False),
            weights     = weights,
            num_classes = num_classes,
            *args, **kwargs
        )
    

@MODELS.register(name="vgg19", arch="vgg")
class VGG19(VGG):
    
    _name: str  = "vgg19"
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/vgg/vgg19/imagenet1k_v1/vgg19_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        super().__init__(
            features    = make_layers(cfgs["E"], batch_norm=False),
            weights     = weights,
            num_classes = num_classes,
            *args, **kwargs
        )
    

@MODELS.register(name="vgg11_bn", arch="vgg")
class VGG11_BN(VGG):
    
    _name: str  = "vgg11_bn"
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/vgg/vgg11_bn/imagenet1k_v1/vgg11_bn_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        super().__init__(
            features    = make_layers(cfgs["A"], batch_norm=True),
            weights     = weights,
            num_classes = num_classes,
            *args, **kwargs
        )
        
            
@MODELS.register(name="vgg13_bn", arch="vgg")
class VGG13_BN(VGG):
    
    _name: str  = "vgg13_bn"
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/vgg/vgg13_bn/imagenet1k_v1/vgg13_bn_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        super().__init__(
            features    = make_layers(cfgs["B"], batch_norm=True),
            weights     = weights,
            num_classes = num_classes,
            *args, **kwargs
        )
        
            
@MODELS.register(name="vgg16_bn", arch="vgg")
class VGG16_BN(VGG):
  
    _name: str  = "vgg16_bn"
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/vgg/vgg16_bn/imagenet1k_v1/vgg16_bn_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        super().__init__(
            features    = make_layers(cfgs["D"], batch_norm=True),
            weights     = weights,
            num_classes = num_classes,
            *args, **kwargs
        )
    

@MODELS.register(name="vgg19_bn", arch="vgg")
class VGG19_BN(VGG):
  
    _name: str  = "vgg19_bn"
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/vgg/vgg19_bn/imagenet1k_v1/vgg19_bn_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        super().__init__(
            features    = make_layers(cfgs["E"], batch_norm=True),
            weights     = weights,
            num_classes = num_classes,
            *args, **kwargs
        )
