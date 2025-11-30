#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements MobileOne models for image classification.

References:
    - Paper: "MobileOne: An Improved One millisecond Mobile Backbone," CVPR 2023.
    - Code: https://github.com/apple/ml-mobileone/tree/main
"""

__all__ = [
    "MobileOneS0",
    "MobileOneS1",
    "MobileOneS2",
    "MobileOneS3",
    "MobileOneS4",
    "reparameterize_model",
]

import abc
import copy
from typing import Any

import box
import torch

from mon import nn
from mon.core import MLType, MODELS, Path, ROOT_DIR, Task

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Model -----
class MobileOne(nn.Module, nn.ModelMixin, abc.ABC):
    """MobileOne models for image classification.
    
    Args:
        num_classes: Number of output classes. Default: ``1000``.
        num_blocks_per_stage: List of number of blocks per stage.
        width_multipliers: List of width multiplier for blocks in a stage.
        inference: If True, instantiates model in inference mode.
        use_se: Whether to use SE-ReLU activations.
        num_conv_branches: Number of linear conv branches.
    
    References:
        - Paper: "MobileOne: An Improved One millisecond Mobile Backbone," CVPR 2023.
        - Code: https://github.com/apple/ml-mobileone/tree/main
    """
    
    _arch     : str          = "mobileone"
    _name     : str          = "mobileone",
    _tasks    : list[Task]   = [Task.CLASSIFY]
    _mltypes  : list[MLType] = [MLType.SUPERVISED]
    _model_dir: Path         = root_dir
    _zoo      : dict         = box.Box()
    
    def __init__(
        self,
        num_classes         : int   = 1000,
        num_blocks_per_stage: tuple = (2, 8, 10, 1),
        width_multipliers   : tuple = None,
        inference           : bool  = False,
        use_se              : bool  = False,
        num_conv_branches   : int   = 1
    ):
        super().__init__()
        assert len(width_multipliers) == 4
        self.inference         = inference
        self.in_planes         = min(64, int(64 * width_multipliers[0]))
        self.use_se            = use_se
        self.num_conv_branches = num_conv_branches
        
        # Build stages
        self.stage0 = nn.MobileOneBlock(3, self.in_planes, 3, 2, 1, inference=self.inference)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64  * width_multipliers[0]), num_blocks_per_stage[0], num_se_blocks=0)
        self.stage2 = self._make_stage(int(128 * width_multipliers[1]), num_blocks_per_stage[1], num_se_blocks=0)
        self.stage3 = self._make_stage(int(256 * width_multipliers[2]), num_blocks_per_stage[2], num_se_blocks=int(num_blocks_per_stage[2] // 2) if use_se else 0)
        self.stage4 = self._make_stage(int(512 * width_multipliers[3]), num_blocks_per_stage[3], num_se_blocks=num_blocks_per_stage[3] if use_se else 0)
        self.gap    = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multipliers[3]), num_classes)

    def _make_stage(self, planes: int, num_blocks: int, num_se_blocks: int) -> nn.Sequential:
        # Get strides for all layers
        strides = [2] + [1] * (num_blocks-1)
        blocks  = []
        for ix, stride in enumerate(strides):
            use_se = False
            if num_se_blocks > num_blocks:
                raise ValueError("Number of SE blocks cannot exceed number of layers.")
            if ix >= (num_blocks - num_se_blocks):
                use_se = True

            # Depthwise conv
            blocks.append(nn.MobileOneBlock(
                in_channels       = self.in_planes,
                out_channels      = self.in_planes,
                kernel_size       = 3,
                stride            = stride,
                padding           = 1,
                groups            = self.in_planes,
                inference= self.inference,
                use_se            = use_se,
                num_conv_branches = self.num_conv_branches
            ))
            # Pointwise conv
            blocks.append(nn.MobileOneBlock(
                in_channels       = self.in_planes,
                out_channels      = planes,
                kernel_size       = 1,
                stride            = 1,
                padding           = 0,
                groups            = 1,
                inference= self.inference,
                use_se            = use_se,
                num_conv_branches = self.num_conv_branches
            ))
            self.in_planes      = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
    
        
@MODELS.register(name="mobileone_s0", arch="mobileone")
class MobileOneS0(MobileOne):
    
    _name: str  = "mobileone_s0",
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/classify/mobileone/mobileone_s0/imagenet1k_v1/mobileone_s0_imagenet1k_v1.pth.tar",
            "num_classes": 1000,
        },
    })
    
    def __init__(
        self,
        weights    : str = "imagenet1k_v1",
        num_classes: int = 1000,
        *args, **kwargs
    ):
        weights, path, num_classes = self.parse_weights(weights, num_classes)
        super().__init__(
            num_classes       = num_classes,
            width_multipliers = (0.75, 1.0, 1.0, 2.0),
            num_conv_branches = 4,
            *args, **kwargs
        )
        if weights:
            self.load_state_dict(weights)


@MODELS.register(name="mobileone_s1", arch="mobileone")
class MobileOneS1(MobileOne):
    
    _name: str  = "mobileone_s1",
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/classify/mobileone/mobileone_s1/imagenet1k_v1/mobileone_s1_imagenet1k_v1.pth.tar",
            "num_classes": 1000,
        },
    })
    
    def __init__(
        self,
        weights    : Any = "imagenet1k_v1",
        num_classes: int = 1000,
        *args, **kwargs
    ):
        weights, path, num_classes = self.parse_weights(weights, num_classes)
        super().__init__(
            num_classes       = num_classes,
            width_multipliers = (1.5, 1.5, 2.0, 2.5),
            *args, **kwargs
        )
        if weights:
            self.load_state_dict(weights)


@MODELS.register(name="mobileone_s2", arch="mobileone")
class MobileOneS2(MobileOne):
    
    _name: str  = "mobileone_s2",
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/classify/mobileone/mobileone_s2/imagenet1k_v1/mobileone_s2_imagenet1k_v1.pth.tar",
            "num_classes": 1000,
        },
    })
    
    def __init__(
        self,
        weights    : Any = "imagenet1k_v1",
        num_classes: int = 1000,
        *args, **kwargs
    ):
        weights, path, num_classes = self.parse_weights(weights, num_classes)
        super().__init__(
            num_classes       = num_classes,
            width_multipliers = (1.5, 2.0, 2.5, 4.0),
            *args, **kwargs
        )
        if weights:
            self.load_state_dict(weights)


@MODELS.register(name="mobileone_s3", arch="mobileone")
class MobileOneS3(MobileOne):
    
    _name: str  = "mobileone_s3",
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/classify/mobileone/mobileone_s3/imagenet1k_v1/mobileone_s3_imagenet1k_v1.pth.tar",
            "num_classes": 1000,
        },
    })
    
    def __init__(
        self,
        weights    : Any = "imagenet1k_v1",
        num_classes: int = 1000,
        *args, **kwargs
    ):
        weights, path, num_classes = self.parse_weights(weights, num_classes)
        super().__init__(
            num_classes       = num_classes,
            width_multipliers = (2.0, 2.5, 3.0, 4.0),
            *args, **kwargs
        )
        if weights:
            self.load_state_dict(weights)


@MODELS.register(name="mobileone_s4", arch="mobileone")
class MobileOneS4(MobileOne):
    
    _name: str  = "mobileone_s4",
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/classify/mobileone/mobileone_s4/imagenet1k_v1/mobileone_s4_imagenet1k_v1.pth.tar",
            "num_classes": 1000,
        },
    })
    
    def __init__(
        self,
        weights    : Any = "imagenet1k_v1",
        num_classes: int = 1000,
        *args, **kwargs
    ):
        weights, path, num_classes = self.parse_weights(weights, num_classes)
        super().__init__(
            num_classes       = num_classes,
            width_multipliers = (3.0, 3.5, 3.5, 4.0),
            use_se            = True,
            *args, **kwargs
        )
        if weights:
            self.load_state_dict(weights)


#----- Re-parameterization -----
def reparameterize_model(model: nn.Module) -> nn.Module:
    """Method returns a model where a multi-branched structure used in training
    is re-parameterized into a single branch for inference.

    Args:
        model: Model to re-parameterize.
    
    Returns:
        Re-parameterized model.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()
    return model
