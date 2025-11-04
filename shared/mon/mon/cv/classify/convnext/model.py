#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ConvNeXt models for image classification.

References:
    - Paper: https://arxiv.org/abs/2201.03545
"""

__all__ = [
    "ConvNeXtBase",
    "ConvNeXtLarge",
    "ConvNeXtSmall",
    "ConvNeXtTiny",
]

import abc
from typing import Any

import box
from torchvision import models as tvm
from torchvision.models.convnext import CNBlockConfig

from mon.constants import MODELS, ROOT_DIR
from mon.core import MLType, ModelMixin, Path, Task

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


class ConvNeXt(tvm.ConvNeXt, ModelMixin, abc.ABC):
    """ConvNeXt models for image classification.
    
    References:
        - Paper: https://arxiv.org/abs/2201.03545
    """
    
    arch     : str          = "convnext"
    name     : str          = "convnext"
    tasks    : list[Task]   = [Task.CLASSIFY]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        weights, path, num_classes = self.parse_weights(weights, num_classes)
        super().__init__(num_classes=num_classes, *args, **kwargs)
        if weights:
            self.load_state_dict(weights)
    
    
@MODELS.register(name="convnext_base", arch="convnext")
class ConvNeXtBase(ConvNeXt):

    name: str  = "convnext_base"
    zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/convnext_base-6075fbad.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/convnext/convnext_base/imagenet1k_v1/convnext_base_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        block_setting = [
            CNBlockConfig( 128,  256,  3),
            CNBlockConfig( 256,  512,  3),
            CNBlockConfig( 512, 1024, 27),
            CNBlockConfig(1024, None,  3),
        ]
        stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
        super().__init__(
            block_setting         = block_setting,
            stochastic_depth_prob = stochastic_depth_prob,
            weights               = weights,
            num_classes           = num_classes,
            *args, **kwargs
        )
        

@MODELS.register(name="convnext_tiny", arch="convnext")
class ConvNeXtTiny(ConvNeXt):
    
    name: str  = "convnext_tiny"
    zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/convnext/convnext_tiny/imagenet1k_v1/convnext_tiny_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        block_setting = [
            CNBlockConfig( 96,  192, 3),
            CNBlockConfig(192,  384, 3),
            CNBlockConfig(384,  768, 9),
            CNBlockConfig(768, None, 3),
        ]
        stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
        super().__init__(
            block_setting         = block_setting,
            stochastic_depth_prob = stochastic_depth_prob,
            weights               = weights,
            num_classes           = num_classes,
            *args, **kwargs
        )


@MODELS.register(name="convnext_small", arch="convnext")
class ConvNeXtSmall(ConvNeXt):
    """ConvNeXt Small model for image classification.

    Args:
        num_classes: Number of output classes. Default: ``1000``.
    """
    
    name: str  = "convnext_small"
    zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/convnext_small-0c510722.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/convnext/convnext_small/imagenet1k_v1/convnext_small_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        block_setting = [
            CNBlockConfig( 96,  192,  3),
            CNBlockConfig(192,  384,  3),
            CNBlockConfig(384,  768, 27),
            CNBlockConfig(768, None,  3),
        ]
        stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.4)
        super().__init__(
            block_setting         = block_setting,
            stochastic_depth_prob = stochastic_depth_prob,
            weights               = weights,
            num_classes           = num_classes,
            *args, **kwargs
        )


@MODELS.register(name="convnext_large", arch="convnext")
class ConvNeXtLarge(ConvNeXt):
    """ConvNeXt Large model for image classification.

    Args:
        num_classes: Number of output classes. Default: ``1000``.
    """
    
    name: str  = "convnext_large"
    zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/convnext_large-ea097f82.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/convnext/convnext_large/imagenet1k_v1/convnext_large_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        block_setting = [
            CNBlockConfig( 192,  384,  3),
            CNBlockConfig( 384,  768,  3),
            CNBlockConfig( 768, 1536, 27),
            CNBlockConfig(1536, None,  3),
        ]
        stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
        super().__init__(
            block_setting         = block_setting,
            stochastic_depth_prob = stochastic_depth_prob,
            weights               = weights,
            num_classes           = num_classes,
            *args, **kwargs
        )
