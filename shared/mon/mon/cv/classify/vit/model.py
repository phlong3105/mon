#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Vision Transformer model for image classification.

References:
    - Paper: https://arxiv.org/abs/2010.11929
"""

__all__ = [
    "ViT_B_16",
    "ViT_B_32",
    "ViT_H_14",
    "ViT_L_16",
    "ViT_L_32",
]

import abc
from typing import Any

import box
from torchvision import models as tvm

import mon.nn as nn
import mon.nn as nn
from mon.constants import MODELS, ROOT_DIR
from mon.core import MLType, Path, Task

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Model -----
class ViT(tvm.VisionTransformer, nn.ModelMixin, abc.ABC):
    """Vision Transformer model for image classification.

    References:
        - Paper: https://arxiv.org/abs/2010.11929
    """
    
    arch     : str          = "vit"
    name     : str          = "vit"
    tasks    : list[Task]   = [Task.CLASSIFY]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        weights, path, num_classes = self.parse_weights(weights, num_classes)
        super().__init__(num_classes=num_classes, *args, **kwargs)
        if weights:
            self.load_state_dict(weights)
            

@MODELS.register(name="vit_b_16", arch="vit")
class ViT_B_16(ViT):
    
    name: str  = "vit_b_16"
    zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vit_b_16-c867db91.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/vit/vit_b_16/imagenet1k_v1/vit_b_16_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_swag_e2e_v1": {
            "url"        : "https://download.pytorch.org/models/vit_b_16_swag-9ac1b537.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/vit/vit_b_16_swag/imagenet1k_v1/vit_b_16_swag_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_swag_linear_v1": {
            "url"        : "https://download.pytorch.org/models/vit_b_16_lc_swag-4e70ced5.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/vit/vit_b_16_lc_swag/imagenet1k_v1/vit_b_16_lc_swag_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        image_size = kwargs.pop("image_size", 224)
        super().__init__(
            image_size  = image_size,
            patch_size  = 16,
            num_layers  = 12,
            num_heads   = 12,
            hidden_dim  = 768,
            mlp_dim     = 3072,
            weights     = weights,
            num_classes = num_classes,
            *args, **kwargs
        )
        

@MODELS.register(name="vit_b_32", arch="vit")
class ViT_B_32(ViT):
    
    name: str  = "vit_b_32"
    zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vit_b_32-d86f8d99.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/vit/vit_b_32/imagenet1k_v1/vit_b_32_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        image_size = kwargs.pop("image_size", 224)
        super().__init__(
            image_size  = image_size,
            patch_size  = 32,
            num_layers  = 12,
            num_heads   = 12,
            hidden_dim  = 768,
            mlp_dim     = 3072,
            weights     = weights,
            num_classes = num_classes,
            *args, **kwargs
        )
    

@MODELS.register(name="vit_l_16", arch="vit")
class ViT_L_16(ViT):
    
    name: str  = "vit_l_16"
    zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vit_l_16-852ce7e3.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/vit/vit_l_16/imagenet1k_v1/vit_l_16_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_swag_e2e_v1": {
            "url"        : "https://download.pytorch.org/models/vit_l_16_swag-4f3808c9.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/vit/vit_l_16_swag/imagenet1k_v1/vit_l_16_swag_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_swag_linear_v1": {
            "url"        : "https://download.pytorch.org/models/vit_l_16_lc_swag-4d563306.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/vit/vit_l_16_lc_swag/imagenet1k_v1/vit_l_16_lc_swag_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        image_size = kwargs.pop("image_size", 224)
        super().__init__(
            image_size  = image_size,
            patch_size  = 16,
            num_layers  = 24,
            num_heads   = 16,
            hidden_dim  = 1024,
            mlp_dim     = 4096,
            weights     = weights,
            num_classes = num_classes,
            *args, **kwargs
        )
        

@MODELS.register(name="vit_l_32", arch="vit")
class ViT_L_32(ViT):

    name: str  = "vit_l_32"
    zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vit_l_32-c7638314.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/vit/vit_l_32/imagenet1k_v1/vit_l_32_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        image_size = kwargs.pop("image_size", 224)
        super().__init__(
            image_size  = image_size,
            patch_size  = 32,
            num_layers  = 24,
            num_heads   = 16,
            hidden_dim  = 1024,
            mlp_dim     = 4096,
            weights     = weights,
            num_classes = num_classes,
            *args, **kwargs
        )
    

@MODELS.register(name="vit_h_14", arch="vit")
class ViT_H_14(ViT):
    
    name: str  = "vit_h_14"
    zoo : dict = box.Box({
        "imagenet1k_swag_e2e_v1": {
            "url"        : "https://download.pytorch.org/models/vit_h_14_swag-80465313.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/vit/vit_h_14_swag/imagenet1k_v1/vit_h_14_swag_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_swag_linear_v1": {
            "url"        : "https://download.pytorch.org/models/vit_h_14_lc_swag-c1eb923e.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/vit/vit_h_14_lc_swag/imagenet1k_v1/vit_h_14_lc_swag_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        image_size = kwargs.pop("image_size", 224)
        super().__init__(
            image_size  = image_size,
            patch_size  = 14,
            num_layers  = 32,
            num_heads   = 16,
            hidden_dim  = 1280,
            mlp_dim     = 5120,
            weights     = weights,
            num_classes = num_classes,
            *args, **kwargs
        )
