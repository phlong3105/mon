#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Swin Transformer model for image classification.

References:
    - Paper: https://arxiv.org/pdf/2103.14030
"""

__all__ = [
    "Swin_B",
    "Swin_S",
    "Swin_T",
    "Swin_V2_B",
    "Swin_V2_S",
    "Swin_V2_T",
]

import abc
from typing import Any

import box
from torchvision import models as tvm
from torchvision.models.swin_transformer import (
    PatchMergingV2,
    SwinTransformerBlockV2,
)

from mon import nn
from mon.core import MLType, MODELS, Path, ROOT_DIR, Task

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


class SwinTransformer(tvm.SwinTransformer, nn.ModelMetadataMixin, abc.ABC):
    """Swin Transformer model for image classification.

    References:
        - Paper: https://arxiv.org/pdf/2103.14030
    """
    
    _arch     : str          = "swin"
    _name     : str          = "swin"
    _tasks    : list[Task]   = [Task.CLASSIFY]
    _mltypes  : list[MLType] = [MLType.SUPERVISED]
    _model_dir: Path         = root_dir
    _zoo      : dict         = box.Box()
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        weights, path, num_classes = self.parse_weights(weights, num_classes)
        super().__init__(num_classes=num_classes, *args, **kwargs)
        if weights:
            self.load_state_dict(weights)
    

@MODELS.register(name="swin_t", arch="swin")
class Swin_T(SwinTransformer):
    
    _name: str  = "swin_t"
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/swin_t-704ceda3.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/swin/swin_t/imagenet1k_v1/swin_t_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        super().__init__(
            patch_size            = [4, 4],
            embed_dim             = 96,
            depths                = [2, 2,  6,  2],
            num_heads             = [3, 6, 12, 24],
            window_size           = [7, 7],
            stochastic_depth_prob = 0.2,
            weights               = weights,
            num_classes           = num_classes,
            *args, **kwargs
        )
        

@MODELS.register(name="swin_s", arch="swin")
class Swin_S(SwinTransformer):

    _name: str  = "swin_s"
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/swin_s-5e29d889.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/swin/swin_s/imagenet1k_v1/swin_s_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        super().__init__(
            patch_size            = [4, 4],
            embed_dim             = 96,
            depths                = [2, 2, 18,  2],
            num_heads             = [3, 6, 12, 24],
            window_size           = [7, 7],
            stochastic_depth_prob = 0.3,
            weights               = weights,
            num_classes           = num_classes,
            *args, **kwargs
        )


@MODELS.register(name="swin_b", arch="swin")
class Swin_B(SwinTransformer):

    _name: str  = "swin_b"
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/swin_b-68c6b09e.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/swin/swin_b/imagenet1k_v1/swin_b_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        super().__init__(
            patch_size            = [4, 4],
            embed_dim             = 128,
            depths                = [2, 2, 18,  2],
            num_heads             = [4, 8, 16, 32],
            window_size           = [7, 7],
            stochastic_depth_prob = 0.5,
            weights               = weights,
            num_classes           = num_classes,
            *args, **kwargs
        )
        

@MODELS.register(name="swin_v2_t", arch="swin")
class Swin_V2_T(SwinTransformer):

    _name: str  = "swin_v2_t"
    _zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/swin_v2_t-b137f0e2.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/swin/swin_v2_t/imagenet1k_v1/swin_v2_t_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        super().__init__(
            patch_size            = [4, 4],
            embed_dim             = 96,
            depths                = [2, 2,  6,  2],
            num_heads             = [3, 6, 12, 24],
            window_size           = [8, 8],
            stochastic_depth_prob = 0.2,
            block                 = SwinTransformerBlockV2,
            downsample_layer      = PatchMergingV2,
            weights               = weights,
            num_classes           = num_classes,
            *args, **kwargs
        )
        

@MODELS.register(name="swin_v2_s", arch="swin")
class Swin_V2_S(SwinTransformer):
    
    _name: str  = "swin_v2_s"
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/swin_v2_s-637d8ceb.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/swin/swin_v2_s/imagenet1k_v1/swin_v2_s_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        super().__init__(
            patch_size            = [4, 4],
            embed_dim             = 96,
            depths                = [2, 2, 18,  2],
            num_heads             = [3, 6, 12, 24],
            window_size           = [8, 8],
            stochastic_depth_prob = 0.3,
            block                 = SwinTransformerBlockV2,
            downsample_layer      = PatchMergingV2,
            weights               = weights,
            num_classes           = num_classes,
            *args, **kwargs
        )
        

@MODELS.register(name="swin_v2_b", arch="swin")
class Swin_V2_B(SwinTransformer):

    _name: str  = "swin_v2_b"
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/swin_v2_b-781e5279.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/swin/swin_v2_b/imagenet1k_v1/swin_v2_b_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        super().__init__(
            patch_size            = [4, 4],
            embed_dim             = 128,
            depths                = [2, 2, 18,  2],
            num_heads             = [4, 8, 16, 32],
            window_size           = [8, 8],
            stochastic_depth_prob = 0.5,
            block                 = SwinTransformerBlockV2,
            downsample_layer      = PatchMergingV2,
            weights               = weights,
            num_classes           = num_classes,
            *args, **kwargs
        )
