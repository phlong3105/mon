#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements HINet models."""

from __future__ import annotations

__all__ = [
    "HINet",
]

import munch
from mon.vision.typing import (
    ClassLabelsType, ConfigType, DictType, LossesType,
    MetricsType, ModelPhaseType, OptimizersType, PathType, WeightsType,
)
from torch import nn

from mon import core, coreml
from mon.vision import constant
from mon.vision.model.enhancement import base


# region Model

@constant.MODELS.register(name="hinet")
class HINet(base.ImageEnhancementModel):
    """Half-Instance Normalization Network
    
    See Also: :class:`base.ImageEnhancementModel`
    """
    
    cfgs = {
        "hinet"        : {
            "name"    : "hinet",
            "channels": 3,
            "backbone": [
                # [from,       number, module,
                # args(out_channels, ...)]
                [-1, 1, coreml.Identity, []],  # 0  (x)
                # UNet 01 Down
                [-1, 1, coreml.Conv2d, [64, 3, 1, 1]],  # 1  (x1)
                [[-1], 1, coreml.HINetConvBlock, [64, True, 0.2, False, True]],
                # 2  (x1_1_down, x1_1)
                [-1, 1, coreml.ExtractItem, [0]],  # 3  (x1_1_down)
                [[-1], 1, coreml.HINetConvBlock, [128, True, 0.2, False, True]],
                # 4  (x1_2_down, x1_2)
                [-1, 1, coreml.ExtractItem, [0]],  # 5  (x1_2_down)
                [[-1], 1, coreml.HINetConvBlock, [256, True, 0.2, False, True]],
                # 6  (x1_3_down, x1_3)
                [-1, 1, coreml.ExtractItem, [0]],  # 7  (x1_3_down)
                [[-1], 1, coreml.HINetConvBlock, [512, True, 0.2, False, True]],
                # 8  (x1_4_down, x1_4)
                [-1, 1, coreml.ExtractItem, [0]],  # 9  (x1_4_down)
                [[-1], 1, coreml.HINetConvBlock,
                 [1024, False, 0.2, False, True]],  # 10 (None,      x1_5)
                # UNet 01 Skip
                [2, 1, coreml.ExtractItem, [1]],  # 11 (x1_1)
                [-1, 1, coreml.Conv2d, [64, 3, 1, 1]],  # 12 (x1_1_skip)
                [4, 1, coreml.ExtractItem, [1]],  # 13 (x1_2)
                [-1, 1, coreml.Conv2d, [128, 3, 1, 1]],  # 14 (x1_2_skip)
                [6, 1, coreml.ExtractItem, [1]],  # 15 (x1_3)
                [-1, 1, coreml.Conv2d, [256, 3, 1, 1]],  # 16 (x1_3_skip)
                [8, 1, coreml.ExtractItem, [1]],  # 17 (x1_4)
                [-1, 1, coreml.Conv2d, [512, 3, 1, 1]],  # 18 (x1_4_skip)
                [10, 1, coreml.ExtractItem, [1]],  # 19 (x1_5)
                # UNet 01 Up
                [[-1, 18], 1, coreml.HINetUpBlock, [512, 0.2]],
                # 20 (x1_4_up = x1_5    + x1_4_skip)
                [[-1, 16], 1, coreml.HINetUpBlock, [256, 0.2]],
                # 21 (x1_3_up = x1_4_up + x1_3_skip)
                [[-1, 14], 1, coreml.HINetUpBlock, [128, 0.2]],
                # 22 (x1_2_up = x1_3_up + x1_2_skip)
                [[-1, 12], 1, coreml.HINetUpBlock, [64, 0.2]],
                # 23 (x1_1_up = x1_2_up + x1_1_skip)
                # SAM
                [[-1, 0], 1, coreml.SAM, [3]],  # 24 (sam_features, y1)
                [-1, 1, coreml.ExtractItem, [0]],  # 25 (sam_features)
                [-2, 1, coreml.ExtractItem, [1]],  # 26 (y1)
                # UNet 02 Down
                [0, 1, coreml.Conv2d, [64, 3, 1, 1]],  # 27 (x2)
                [[-1, 25], 1, coreml.Concat, []],  # 28 (x2 + sam_features)
                [-1, 1, coreml.Conv2d, [64, 1, 1, 0]],  # 29 (x2)
                [[-1, 11, 23], 1, coreml.HINetConvBlock,
                 [64, True, 0.2, True, True]],  # 30 (x2_1_down, x2_1)
                [-1, 1, coreml.ExtractItem, [0]],  # 31 (x2_1_down)
                [[-1, 13, 22], 1, coreml.HINetConvBlock,
                 [128, True, 0.2, True, True]],  # 32 (x2_2_down, x2_2)
                [-1, 1, coreml.ExtractItem, [0]],  # 33 (x2_2_down)
                [[-1, 15, 21], 1, coreml.HINetConvBlock,
                 [256, True, 0.2, True, True]],  # 34 (x2_3_down, x2_3)
                [-1, 1, coreml.ExtractItem, [0]],  # 35 (x2_3_down)
                [[-1, 17, 20], 1, coreml.HINetConvBlock,
                 [512, True, 0.2, True, True]],  # 36 (x2_4_down, x2_4)
                [-1, 1, coreml.ExtractItem, [0]],  # 37 (x2_4_down)
                [[-1], 1, coreml.HINetConvBlock,
                 [1024, False, 0.2, False, True]],  # 38 (None,      x2_5)
                # UNet 02 Skip
                [30, 1, coreml.ExtractItem, [1]],  # 39 (x2_1)
                [-1, 1, coreml.Conv2d, [64, 3, 1, 1]],  # 40 (x2_1_skip)
                [32, 1, coreml.ExtractItem, [1]],  # 41 (x2_2)
                [-1, 1, coreml.Conv2d, [128, 3, 1, 1]],  # 42 (x2_2_skip)
                [34, 1, coreml.ExtractItem, [1]],  # 43 (x2_3)
                [-1, 1, coreml.Conv2d, [256, 3, 1, 1]],  # 44 (x2_3_skip)
                [36, 1, coreml.ExtractItem, [1]],  # 45 (x2_4)
                [-1, 1, coreml.Conv2d, [512, 3, 1, 1]],  # 46 (x2_4_skip)
                [38, 1, coreml.ExtractItem, [1]],  # 47 (x2_5)
                # UNet 02 Up
                [[-1, 46], 1, coreml.HINetUpBlock, [512, 0.2]],
                # 48 (x2_4_up = x2_5    + x2_4_skip)
                [[-1, 44], 1, coreml.HINetUpBlock, [256, 0.2]],
                # 49 (x2_3_up = x2_4_up + x2_3_skip)
                [[-1, 42], 1, coreml.HINetUpBlock, [128, 0.2]],
                # 50 (x2_2_up = x2_3_up + x2_2_skip)
                [[-1, 40], 1, coreml.HINetUpBlock, [64, 0.2]],
                # 51 (x2_1_up = x2_2_up + x2_1_skip)
            ],
            "head"    : [
                [-1, 1, coreml.Conv2d, [3, 3, 1, 1]],  # 52
                [[-1, 0], 1, coreml.Sum, []],  # 53
                [[26, -1], 1, coreml.Join, []],  # 54
            ]
        },
        "hinet-x0.5"   : {
            "name"    : "hinet-x0.5",
            "channels": 3,
            "backbone": [
                # [from,       number, module,
                # args(out_channels, ...)]
                [-1, 1, coreml.Identity, []],  # 0  (x)
                # UNet 01 Down
                [-1, 1, coreml.Conv2d, [32, 3, 1, 1]],  # 1  (x1)
                [[-1], 1, coreml.HINetConvBlock, [32, True, 0.2, False, True]],
                # 2  (x1_1_down, x1_1)
                [-1, 1, coreml.ExtractItem, [0]],  # 3  (x1_1_down)
                [[-1], 1, coreml.HINetConvBlock, [64, True, 0.2, False, True]],
                # 4  (x1_2_down, x1_2)
                [-1, 1, coreml.ExtractItem, [0]],  # 5  (x1_2_down)
                [[-1], 1, coreml.HINetConvBlock, [128, True, 0.2, False, True]],
                # 6  (x1_3_down, x1_3)
                [-1, 1, coreml.ExtractItem, [0]],  # 7  (x1_3_down)
                [[-1], 1, coreml.HINetConvBlock, [256, True, 0.2, False, True]],
                # 8  (x1_4_down, x1_4)
                [-1, 1, coreml.ExtractItem, [0]],  # 9  (x1_4_down)
                [[-1], 1, coreml.HINetConvBlock,
                 [512, False, 0.2, False, True]],  # 10 (None,      x1_5)
                # UNet 01 Skip
                [2, 1, coreml.ExtractItem, [1]],  # 11 (x1_1)
                [-1, 1, coreml.Conv2d, [32, 3, 1, 1]],  # 12 (x1_1_skip)
                [4, 1, coreml.ExtractItem, [1]],  # 13 (x1_2)
                [-1, 1, coreml.Conv2d, [64, 3, 1, 1]],  # 14 (x1_2_skip)
                [6, 1, coreml.ExtractItem, [1]],  # 15 (x1_3)
                [-1, 1, coreml.Conv2d, [128, 3, 1, 1]],  # 16 (x1_3_skip)
                [8, 1, coreml.ExtractItem, [1]],  # 17 (x1_4)
                [-1, 1, coreml.Conv2d, [256, 3, 1, 1]],  # 18 (x1_4_skip)
                [10, 1, coreml.ExtractItem, [1]],  # 19 (x1_5)
                # UNet 01 Up
                [[-1, 18], 1, coreml.HINetUpBlock, [256, 0.2]],
                # 20 (x1_4_up = x1_5    + x1_4_skip)
                [[-1, 16], 1, coreml.HINetUpBlock, [128, 0.2]],
                # 21 (x1_3_up = x1_4_up + x1_3_skip)
                [[-1, 14], 1, coreml.HINetUpBlock, [64, 0.2]],
                # 22 (x1_2_up = x1_3_up + x1_2_skip)
                [[-1, 12], 1, coreml.HINetUpBlock, [32, 0.2]],
                # 23 (x1_1_up = x1_2_up + x1_1_skip)
                # SAM
                [[-1, 0], 1, coreml.SAM, [3]],  # 24 (sam_features, y1)
                [-1, 1, coreml.ExtractItem, [0]],  # 25 (sam_features)
                [-2, 1, coreml.ExtractItem, [1]],  # 26 (y1)
                # UNet 02 Down
                [0, 1, coreml.Conv2d, [32, 3, 1, 1]],  # 27 (x2)
                [[-1, 25], 1, coreml.Concat, []],  # 28 (x2 + sam_features)
                [-1, 1, coreml.Conv2d, [32, 1, 1, 0]],  # 29 (x2)
                [[-1, 11, 23], 1, coreml.HINetConvBlock,
                 [32, True, 0.2, True, True]],  # 30 (x2_1_down, x2_1)
                [-1, 1, coreml.ExtractItem, [0]],  # 31 (x2_1_down)
                [[-1, 13, 22], 1, coreml.HINetConvBlock,
                 [64, True, 0.2, True, True]],  # 32 (x2_2_down, x2_2)
                [-1, 1, coreml.ExtractItem, [0]],  # 33 (x2_2_down)
                [[-1, 15, 21], 1, coreml.HINetConvBlock,
                 [128, True, 0.2, True, True]],  # 34 (x2_3_down, x2_3)
                [-1, 1, coreml.ExtractItem, [0]],  # 35 (x2_3_down)
                [[-1, 17, 20], 1, coreml.HINetConvBlock,
                 [256, True, 0.2, True, True]],  # 36 (x2_4_down, x2_4)
                [-1, 1, coreml.ExtractItem, [0]],  # 37 (x2_4_down)
                [[-1], 1, coreml.HINetConvBlock,
                 [512, False, 0.2, False, True]],  # 38 (None,      x2_5)
                # UNet 02 Skip
                [30, 1, coreml.ExtractItem, [1]],  # 39 (x2_1)
                [-1, 1, coreml.Conv2d, [32, 3, 1, 1]],  # 40 (x2_1_skip)
                [32, 1, coreml.ExtractItem, [1]],  # 41 (x2_2)
                [-1, 1, coreml.Conv2d, [64, 3, 1, 1]],  # 42 (x2_2_skip)
                [34, 1, coreml.ExtractItem, [1]],  # 43 (x2_3)
                [-1, 1, coreml.Conv2d, [128, 3, 1, 1]],  # 44 (x2_3_skip)
                [36, 1, coreml.ExtractItem, [1]],  # 45 (x2_4)
                [-1, 1, coreml.Conv2d, [256, 3, 1, 1]],  # 46 (x2_4_skip)
                [38, 1, coreml.ExtractItem, [1]],  # 47 (x2_5)
                # UNet 02 Up
                [[-1, 46], 1, coreml.HINetUpBlock, [256, 0.2]],
                # 48 (x2_4_up = x2_5    + x2_4_skip)
                [[-1, 44], 1, coreml.HINetUpBlock, [128, 0.2]],
                # 49 (x2_3_up = x2_4_up + x2_3_skip)
                [[-1, 42], 1, coreml.HINetUpBlock, [64, 0.2]],
                # 50 (x2_2_up = x2_3_up + x2_2_skip)
                [[-1, 40], 1, coreml.HINetUpBlock, [32, 0.2]],
                # 51 (x2_1_up = x2_2_up + x2_1_skip)
            ],
            "head"    : [
                [-1, 1, coreml.Conv2d, [3, 3, 1, 1]],  # 52
                [[-1, 0], 1, coreml.Sum, []],  # 53
                [[26, -1], 1, coreml.Join, []],  # 54
            ]
        },
        "hinet-denoise": {
            "name"    : "hinet-denoise",
            "channels": 3,
            "backbone": [
                # [from,       number, module,
                # args(out_channels, ...)]
                [-1, 1, coreml.Identity, []],  # 0  (x)
                # UNet 01 Down
                [-1, 1, coreml.Conv2d, [64, 3, 1, 1]],  # 1  (x1)
                [[-1], 1, coreml.HINetConvBlock, [64, True, 0.2, False, False]],
                # 2  (x1_1_down, x1_1)
                [-1, 1, coreml.ExtractItem, [0]],  # 3  (x1_1_down)
                [[-1], 1, coreml.HINetConvBlock,
                 [128, True, 0.2, False, False]],  # 4  (x1_2_down, x1_2)
                [-1, 1, coreml.ExtractItem, [0]],  # 5  (x1_2_down)
                [[-1], 1, coreml.HINetConvBlock,
                 [256, True, 0.2, False, False]],  # 6  (x1_3_down, x1_3)
                [-1, 1, coreml.ExtractItem, [0]],  # 7  (x1_3_down)
                [[-1], 1, coreml.HINetConvBlock, [512, True, 0.2, False, True]],
                # 8  (x1_4_down, x1_4)
                [-1, 1, coreml.ExtractItem, [0]],  # 9  (x1_4_down)
                [[-1], 1, coreml.HINetConvBlock,
                 [1024, False, 0.2, False, True]],  # 10 (None,      x1_5)
                # UNet 01 Skip
                [2, 1, coreml.ExtractItem, [1]],  # 11 (x1_1)
                [-1, 1, coreml.Conv2d, [64, 3, 1, 1]],  # 12 (x1_1_skip)
                [4, 1, coreml.ExtractItem, [1]],  # 13 (x1_2)
                [-1, 1, coreml.Conv2d, [128, 3, 1, 1]],  # 14 (x1_2_skip)
                [6, 1, coreml.ExtractItem, [1]],  # 15 (x1_3)
                [-1, 1, coreml.Conv2d, [256, 3, 1, 1]],  # 16 (x1_3_skip)
                [8, 1, coreml.ExtractItem, [1]],  # 17 (x1_4)
                [-1, 1, coreml.Conv2d, [512, 3, 1, 1]],  # 18 (x1_4_skip)
                [10, 1, coreml.ExtractItem, [1]],  # 19 (x1_5)
                # UNet 01 Up
                [[-1, 18], 1, coreml.HINetUpBlock, [512, 0.2]],
                # 20 (x1_4_up = x1_5    + x1_4_skip)
                [[-1, 16], 1, coreml.HINetUpBlock, [256, 0.2]],
                # 21 (x1_3_up = x1_4_up + x1_3_skip)
                [[-1, 14], 1, coreml.HINetUpBlock, [128, 0.2]],
                # 22 (x1_2_up = x1_3_up + x1_2_skip)
                [[-1, 12], 1, coreml.HINetUpBlock, [64, 0.2]],
                # 23 (x1_1_up = x1_2_up + x1_1_skip)
                # SAM
                [[-1, 0], 1, coreml.SAM, [3]],  # 24 (sam_features, y1)
                [-1, 1, coreml.ExtractItem, [0]],  # 25 (sam_features)
                [-2, 1, coreml.ExtractItem, [1]],  # 26 (y1)
                # UNet 02 Down
                [0, 1, coreml.Conv2d, [64, 3, 1, 1]],  # 27 (x2)
                [[-1, 25], 1, coreml.Concat, []],  # 28 (x2 + sam_features)
                [-1, 1, coreml.Conv2d, [64, 1, 1, 0]],  # 29 (x2)
                [[-1, 11, 23], 1, coreml.HINetConvBlock,
                 [64, True, 0.2, True, False]],  # 30 (x2_1_down, x2_1)
                [-1, 1, coreml.ExtractItem, [0]],  # 31 (x2_1_down)
                [[-1, 13, 22], 1, coreml.HINetConvBlock,
                 [128, True, 0.2, True, False]],  # 32 (x2_2_down, x2_2)
                [-1, 1, coreml.ExtractItem, [0]],  # 33 (x2_2_down)
                [[-1, 15, 21], 1, coreml.HINetConvBlock,
                 [256, True, 0.2, True, False]],  # 34 (x2_3_down, x2_3)
                [-1, 1, coreml.ExtractItem, [0]],  # 35 (x2_3_down)
                [[-1, 17, 20], 1, coreml.HINetConvBlock,
                 [512, True, 0.2, True, True]],  # 36 (x2_4_down, x2_4)
                [-1, 1, coreml.ExtractItem, [0]],  # 37 (x2_4_down)
                [[-1], 1, coreml.HINetConvBlock,
                 [1024, False, 0.2, False, True]],  # 38 (None,      x2_5)
                # UNet 02 Skip
                [30, 1, coreml.ExtractItem, [1]],  # 39 (x2_1)
                [-1, 1, coreml.Conv2d, [64, 3, 1, 1]],  # 40 (x2_1_skip)
                [32, 1, coreml.ExtractItem, [1]],  # 41 (x2_2)
                [-1, 1, coreml.Conv2d, [128, 3, 1, 1]],  # 42 (x2_2_skip)
                [34, 1, coreml.ExtractItem, [1]],  # 43 (x2_3)
                [-1, 1, coreml.Conv2d, [256, 3, 1, 1]],  # 44 (x2_3_skip)
                [36, 1, coreml.ExtractItem, [1]],  # 45 (x2_4)
                [-1, 1, coreml.Conv2d, [512, 3, 1, 1]],  # 46 (x2_4_skip)
                [38, 1, coreml.ExtractItem, [1]],  # 47 (x2_5)
                # UNet 02 Up
                [[-1, 46], 1, coreml.HINetUpBlock, [512, 0.2]],
                # 48 (x2_4_up = x2_5    + x2_4_skip)
                [[-1, 44], 1, coreml.HINetUpBlock, [256, 0.2]],
                # 49 (x2_3_up = x2_4_up + x2_3_skip)
                [[-1, 42], 1, coreml.HINetUpBlock, [128, 0.2]],
                # 50 (x2_2_up = x2_3_up + x2_2_skip)
                [[-1, 40], 1, coreml.HINetUpBlock, [64, 0.2]],
                # 51 (x2_1_up = x2_2_up + x2_1_skip)
            ],
            "head"    : [
                [-1, 1, coreml.Conv2d, [3, 3, 1, 1]],  # 52
                [[-1, 0], 1, coreml.Sum, []],  # 53
                [[26, -1], 1, coreml.Join, []],  # 54
            ]
        },
    }
    pretrained_weights = {
        "hinet-deblur-gopro"     : dict(
            name="gopro",
            path="",
            filename="hinet-deblur-gopro.pth",
            num_classes=None,
        ),
        "hinet-deblur-reds"      : dict(
            name="reds",
            path="",
            filename="hinet-deblur-reds.pth",
            num_classes=None,
        ),
        "hinet-denoise-sidd-x1.0": dict(
            name="sidd",
            path="",
            filename="hinet-denoise-sidd-x1.0.pth",
            num_classes=None,
        ),
        "hinet-denoise-sidd-x0.5": dict(
            name="sidd",
            path="",
            filename="hinet-denoise-sidd-x0.5.pth",
            num_classes=None,
        ),
        "hinet-derain-rain13k"   : dict(
            name="rain13k",
            path="",
            filename="hinet-derain-rain13k.pth",
            num_classes=None,
        ),
    }
    
    def __init__(
        self,
        cfg: ConfigType | None = "hinet.yaml",
        hparams: DictType | None = None,
        channels: int = 3,
        num_classes: int | None = None,
        classlabels: ClassLabelsType | None = None,
        weights: WeightsType = False,
        # For management
        name: str | None = "hinet",
        variant: str | None = None,
        fullname: str | None = "hinet",
        root: PathType = constant.RUN_DIR,
        project: str | None = None,
        # For training
        phase: ModelPhaseType = "training",
        loss: LossesType | None = None,
        metrics: MetricsType | None = None,
        optimizers: OptimizersType | None = None,
        debug: DictType | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        super().__init__(
            cfg=cfg,
            hparams=hparams,
            channels=channels,
            num_classes=num_classes,
            classlabels=classlabels,
            weights=weights,
            name=name,
            variant=variant,
            fullname=fullname,
            root=root,
            project=project,
            phase=phase,
            loss=loss,
            metrics=metrics,
            optimizers=optimizers,
            debug=debug,
            verbose=verbose,
            *args, **kwargs
        )
    
    @property
    def cfg_dir(self) -> PathType:
        return core.Path(__file__).resolve().parent / "cfg"
    
    def init_weights(self, m: nn.Module):
        """Initialize model's weights."""
        pass
    
    def load_weights(self):
        """Load weights. It only loads the intersection layers of matching keys
        and shapes between the current model and weights.
        """
        if isinstance(self.pretrained, munch.Munch | dict) \
            and self.pretrained["name"] in ["gopro", "reds", "sidd", "rain13k"]:
            state_dict = coreml.load_state_dict_from_path(
                model_dir=self.zoo_dir, **self.pretrained
            )
            state_dict = state_dict["params"]
            """
            for k in sorted(self.model.state_dict().keys()):
                print(f"model_state_dict[\"{k}\"] = ")
            for k in sorted(state_dict.keys()):
                print(f"state_dict[\"{k}\"]")
            """
            model_state_dict = self.model.state_dict()
            model_state_dict["1.weight"] = state_dict["conv_01.weight"]
            model_state_dict["1.bias"] = state_dict["conv_01.bias"]
            model_state_dict["2.conv1.weight"] = state_dict[
                "down_path_1.0.conv_1.weight"]
            model_state_dict["2.conv1.bias"] = state_dict[
                "down_path_1.0.conv_1.bias"]
            model_state_dict["2.conv2.weight"] = state_dict[
                "down_path_1.0.conv_2.weight"]
            model_state_dict["2.conv2.bias"] = state_dict[
                "down_path_1.0.conv_2.bias"]
            model_state_dict["2.identity.weight"] = state_dict[
                "down_path_1.0.identity.weight"]
            model_state_dict["2.identity.bias"] = state_dict[
                "down_path_1.0.identity.bias"]
            model_state_dict["2.downsample.weight"] = state_dict[
                "down_path_1.0.downsample.weight"]
            model_state_dict["4.conv1.weight"] = state_dict[
                "down_path_1.1.conv_1.weight"]
            model_state_dict["4.conv1.bias"] = state_dict[
                "down_path_1.1.conv_1.bias"]
            model_state_dict["4.conv2.weight"] = state_dict[
                "down_path_1.1.conv_2.weight"]
            model_state_dict["4.conv2.bias"] = state_dict[
                "down_path_1.1.conv_2.bias"]
            model_state_dict["4.identity.weight"] = state_dict[
                "down_path_1.1.identity.weight"]
            model_state_dict["4.identity.bias"] = state_dict[
                "down_path_1.1.identity.bias"]
            model_state_dict["4.downsample.weight"] = state_dict[
                "down_path_1.1.downsample.weight"]
            model_state_dict["6.conv1.weight"] = state_dict[
                "down_path_1.2.conv_1.weight"]
            model_state_dict["6.conv1.bias"] = state_dict[
                "down_path_1.2.conv_1.bias"]
            model_state_dict["6.conv2.weight"] = state_dict[
                "down_path_1.2.conv_2.weight"]
            model_state_dict["6.conv2.bias"] = state_dict[
                "down_path_1.2.conv_2.bias"]
            model_state_dict["6.identity.weight"] = state_dict[
                "down_path_1.2.identity.weight"]
            model_state_dict["6.identity.bias"] = state_dict[
                "down_path_1.2.identity.bias"]
            model_state_dict["6.downsample.weight"] = state_dict[
                "down_path_1.2.downsample.weight"]
            model_state_dict["8.conv1.weight"] = state_dict[
                "down_path_1.3.conv_1.weight"]
            model_state_dict["8.conv1.bias"] = state_dict[
                "down_path_1.3.conv_1.bias"]
            model_state_dict["8.conv2.weight"] = state_dict[
                "down_path_1.3.conv_2.weight"]
            model_state_dict["8.conv2.bias"] = state_dict[
                "down_path_1.3.conv_2.bias"]
            model_state_dict["8.identity.weight"] = state_dict[
                "down_path_1.3.identity.weight"]
            model_state_dict["8.identity.bias"] = state_dict[
                "down_path_1.3.identity.bias"]
            model_state_dict["8.norm.weight"] = state_dict[
                "down_path_1.3.norm.weight"]
            model_state_dict["8.norm.bias"] = state_dict[
                "down_path_1.3.norm.bias"]
            model_state_dict["8.downsample.weight"] = state_dict[
                "down_path_1.3.downsample.weight"]
            model_state_dict["10.conv1.weight"] = state_dict[
                "down_path_1.4.conv_1.weight"]
            model_state_dict["10.conv1.bias"] = state_dict[
                "down_path_1.4.conv_1.bias"]
            model_state_dict["10.conv2.weight"] = state_dict[
                "down_path_1.4.conv_2.weight"]
            model_state_dict["10.conv2.bias"] = state_dict[
                "down_path_1.4.conv_2.bias"]
            model_state_dict["10.identity.weight"] = state_dict[
                "down_path_1.4.identity.weight"]
            model_state_dict["10.identity.bias"] = state_dict[
                "down_path_1.4.identity.bias"]
            model_state_dict["10.norm.weight"] = state_dict[
                "down_path_1.4.norm.weight"]
            model_state_dict["10.norm.bias"] = state_dict[
                "down_path_1.4.norm.bias"]
            model_state_dict["12.weight"] = state_dict["skip_conv_1.3.weight"]
            model_state_dict["12.bias"] = state_dict["skip_conv_1.3.bias"]
            model_state_dict["14.weight"] = state_dict["skip_conv_1.2.weight"]
            model_state_dict["14.bias"] = state_dict["skip_conv_1.2.bias"]
            model_state_dict["16.weight"] = state_dict["skip_conv_1.1.weight"]
            model_state_dict["16.bias"] = state_dict["skip_conv_1.1.bias"]
            model_state_dict["18.weight"] = state_dict["skip_conv_1.0.weight"]
            model_state_dict["18.bias"] = state_dict["skip_conv_1.0.bias"]
            model_state_dict["20.up.weight"] = state_dict[
                "up_path_1.0.up.weight"]
            model_state_dict["20.up.bias"] = state_dict["up_path_1.0.up.bias"]
            model_state_dict["20.conv.conv1.weight"] = state_dict[
                "up_path_1.0.conv_block.conv_1.weight"]
            model_state_dict["20.conv.conv1.bias"] = state_dict[
                "up_path_1.0.conv_block.conv_1.bias"]
            model_state_dict["20.conv.conv2.weight"] = state_dict[
                "up_path_1.0.conv_block.conv_2.weight"]
            model_state_dict["20.conv.conv2.bias"] = state_dict[
                "up_path_1.0.conv_block.conv_2.bias"]
            model_state_dict["20.conv.identity.weight"] = state_dict[
                "up_path_1.0.conv_block.identity.weight"]
            model_state_dict["20.conv.identity.bias"] = state_dict[
                "up_path_1.0.conv_block.identity.bias"]
            model_state_dict["21.up.weight"] = state_dict[
                "up_path_1.1.up.weight"]
            model_state_dict["21.up.bias"] = state_dict["up_path_1.1.up.bias"]
            model_state_dict["21.conv.conv1.weight"] = state_dict[
                "up_path_1.1.conv_block.conv_1.weight"]
            model_state_dict["21.conv.conv1.bias"] = state_dict[
                "up_path_1.1.conv_block.conv_1.bias"]
            model_state_dict["21.conv.conv2.weight"] = state_dict[
                "up_path_1.1.conv_block.conv_2.weight"]
            model_state_dict["21.conv.conv2.bias"] = state_dict[
                "up_path_1.1.conv_block.conv_2.bias"]
            model_state_dict["21.conv.identity.weight"] = state_dict[
                "up_path_1.1.conv_block.identity.weight"]
            model_state_dict["21.conv.identity.bias"] = state_dict[
                "up_path_1.1.conv_block.identity.bias"]
            model_state_dict["22.up.weight"] = state_dict[
                "up_path_1.2.up.weight"]
            model_state_dict["22.up.bias"] = state_dict["up_path_1.2.up.bias"]
            model_state_dict["22.conv.conv1.weight"] = state_dict[
                "up_path_1.2.conv_block.conv_1.weight"]
            model_state_dict["22.conv.conv1.bias"] = state_dict[
                "up_path_1.2.conv_block.conv_1.bias"]
            model_state_dict["22.conv.conv2.weight"] = state_dict[
                "up_path_1.2.conv_block.conv_2.weight"]
            model_state_dict["22.conv.conv2.bias"] = state_dict[
                "up_path_1.2.conv_block.conv_2.bias"]
            model_state_dict["22.conv.identity.weight"] = state_dict[
                "up_path_1.2.conv_block.identity.weight"]
            model_state_dict["22.conv.identity.bias"] = state_dict[
                "up_path_1.2.conv_block.identity.bias"]
            model_state_dict["23.up.weight"] = state_dict[
                "up_path_1.3.up.weight"]
            model_state_dict["23.up.bias"] = state_dict["up_path_1.3.up.bias"]
            model_state_dict["23.conv.conv1.weight"] = state_dict[
                "up_path_1.3.conv_block.conv_1.weight"]
            model_state_dict["23.conv.conv1.bias"] = state_dict[
                "up_path_1.3.conv_block.conv_1.bias"]
            model_state_dict["23.conv.conv2.weight"] = state_dict[
                "up_path_1.3.conv_block.conv_2.weight"]
            model_state_dict["23.conv.conv2.bias"] = state_dict[
                "up_path_1.3.conv_block.conv_2.bias"]
            model_state_dict["23.conv.identity.weight"] = state_dict[
                "up_path_1.3.conv_block.identity.weight"]
            model_state_dict["23.conv.identity.bias"] = state_dict[
                "up_path_1.3.conv_block.identity.bias"]
            model_state_dict["24.conv1.weight"] = state_dict[
                "sam12.conv1.weight"]
            model_state_dict["24.conv1.bias"] = state_dict["sam12.conv1.bias"]
            model_state_dict["24.conv2.weight"] = state_dict[
                "sam12.conv2.weight"]
            model_state_dict["24.conv2.bias"] = state_dict["sam12.conv2.bias"]
            model_state_dict["24.conv3.weight"] = state_dict[
                "sam12.conv3.weight"]
            model_state_dict["24.conv3.bias"] = state_dict["sam12.conv3.bias"]
            model_state_dict["27.weight"] = state_dict["conv_02.weight"]
            model_state_dict["27.bias"] = state_dict["conv_02.bias"]
            model_state_dict["29.weight"] = state_dict["cat12.weight"]
            model_state_dict["29.bias"] = state_dict["cat12.bias"]
            model_state_dict["30.conv1.weight"] = state_dict[
                "down_path_2.0.conv_1.weight"]
            model_state_dict["30.conv1.bias"] = state_dict[
                "down_path_2.0.conv_1.bias"]
            model_state_dict["30.conv2.weight"] = state_dict[
                "down_path_2.0.conv_2.weight"]
            model_state_dict["30.conv2.bias"] = state_dict[
                "down_path_2.0.conv_2.bias"]
            model_state_dict["30.identity.weight"] = state_dict[
                "down_path_2.0.identity.weight"]
            model_state_dict["30.identity.bias"] = state_dict[
                "down_path_2.0.identity.bias"]
            model_state_dict["30.csff_enc.weight"] = state_dict[
                "down_path_2.0.csff_enc.weight"]
            model_state_dict["30.csff_enc.bias"] = state_dict[
                "down_path_2.0.csff_enc.bias"]
            model_state_dict["30.csff_dec.weight"] = state_dict[
                "down_path_2.0.csff_dec.weight"]
            model_state_dict["30.csff_dec.bias"] = state_dict[
                "down_path_2.0.csff_dec.bias"]
            model_state_dict["30.downsample.weight"] = state_dict[
                "down_path_2.0.downsample.weight"]
            model_state_dict["32.conv1.weight"] = state_dict[
                "down_path_2.1.conv_1.weight"]
            model_state_dict["32.conv1.bias"] = state_dict[
                "down_path_2.1.conv_1.bias"]
            model_state_dict["32.conv2.weight"] = state_dict[
                "down_path_2.1.conv_2.weight"]
            model_state_dict["32.conv2.bias"] = state_dict[
                "down_path_2.1.conv_2.bias"]
            model_state_dict["32.identity.weight"] = state_dict[
                "down_path_2.1.identity.weight"]
            model_state_dict["32.identity.bias"] = state_dict[
                "down_path_2.1.identity.bias"]
            model_state_dict["32.csff_enc.weight"] = state_dict[
                "down_path_2.1.csff_enc.weight"]
            model_state_dict["32.csff_enc.bias"] = state_dict[
                "down_path_2.1.csff_enc.bias"]
            model_state_dict["32.csff_dec.weight"] = state_dict[
                "down_path_2.1.csff_dec.weight"]
            model_state_dict["32.csff_dec.bias"] = state_dict[
                "down_path_2.1.csff_dec.bias"]
            model_state_dict["32.downsample.weight"] = state_dict[
                "down_path_2.1.downsample.weight"]
            model_state_dict["34.conv1.weight"] = state_dict[
                "down_path_2.2.conv_1.weight"]
            model_state_dict["34.conv1.bias"] = state_dict[
                "down_path_2.2.conv_1.bias"]
            model_state_dict["34.conv2.weight"] = state_dict[
                "down_path_2.2.conv_2.weight"]
            model_state_dict["34.conv2.bias"] = state_dict[
                "down_path_2.2.conv_2.bias"]
            model_state_dict["34.identity.weight"] = state_dict[
                "down_path_2.2.identity.weight"]
            model_state_dict["34.identity.bias"] = state_dict[
                "down_path_2.2.identity.bias"]
            model_state_dict["34.csff_enc.weight"] = state_dict[
                "down_path_2.2.csff_enc.weight"]
            model_state_dict["34.csff_enc.bias"] = state_dict[
                "down_path_2.2.csff_enc.bias"]
            model_state_dict["34.csff_dec.weight"] = state_dict[
                "down_path_2.2.csff_dec.weight"]
            model_state_dict["34.csff_dec.bias"] = state_dict[
                "down_path_2.2.csff_dec.bias"]
            model_state_dict["34.downsample.weight"] = state_dict[
                "down_path_2.2.downsample.weight"]
            model_state_dict["36.conv1.weight"] = state_dict[
                "down_path_2.3.conv_1.weight"]
            model_state_dict["36.conv1.bias"] = state_dict[
                "down_path_2.3.conv_1.bias"]
            model_state_dict["36.conv2.weight"] = state_dict[
                "down_path_2.3.conv_2.weight"]
            model_state_dict["36.conv2.bias"] = state_dict[
                "down_path_2.3.conv_2.bias"]
            model_state_dict["36.identity.weight"] = state_dict[
                "down_path_2.3.identity.weight"]
            model_state_dict["36.identity.bias"] = state_dict[
                "down_path_2.3.identity.bias"]
            model_state_dict["36.csff_enc.weight"] = state_dict[
                "down_path_2.3.csff_enc.weight"]
            model_state_dict["36.csff_enc.bias"] = state_dict[
                "down_path_2.3.csff_enc.bias"]
            model_state_dict["36.csff_dec.weight"] = state_dict[
                "down_path_2.3.csff_dec.weight"]
            model_state_dict["36.csff_dec.bias"] = state_dict[
                "down_path_2.3.csff_dec.bias"]
            model_state_dict["36.norm.weight"] = state_dict[
                "down_path_2.3.norm.weight"]
            model_state_dict["36.norm.bias"] = state_dict[
                "down_path_2.3.norm.bias"]
            model_state_dict["36.downsample.weight"] = state_dict[
                "down_path_2.3.downsample.weight"]
            model_state_dict["38.conv1.weight"] = state_dict[
                "down_path_2.4.conv_1.weight"]
            model_state_dict["38.conv1.bias"] = state_dict[
                "down_path_2.4.conv_1.bias"]
            model_state_dict["38.conv2.weight"] = state_dict[
                "down_path_2.4.conv_2.weight"]
            model_state_dict["38.conv2.bias"] = state_dict[
                "down_path_2.4.conv_2.bias"]
            model_state_dict["38.identity.weight"] = state_dict[
                "down_path_2.4.identity.weight"]
            model_state_dict["38.identity.bias"] = state_dict[
                "down_path_2.4.identity.bias"]
            model_state_dict["38.norm.weight"] = state_dict[
                "down_path_2.4.norm.weight"]
            model_state_dict["38.norm.bias"] = state_dict[
                "down_path_2.4.norm.bias"]
            model_state_dict["40.weight"] = state_dict["skip_conv_2.3.weight"]
            model_state_dict["40.bias"] = state_dict["skip_conv_2.3.bias"]
            model_state_dict["42.weight"] = state_dict["skip_conv_2.2.weight"]
            model_state_dict["42.bias"] = state_dict["skip_conv_2.2.bias"]
            model_state_dict["44.weight"] = state_dict["skip_conv_2.1.weight"]
            model_state_dict["44.bias"] = state_dict["skip_conv_2.1.bias"]
            model_state_dict["46.weight"] = state_dict["skip_conv_2.0.weight"]
            model_state_dict["46.bias"] = state_dict["skip_conv_2.0.bias"]
            model_state_dict["48.up.weight"] = state_dict[
                "up_path_2.0.up.weight"]
            model_state_dict["48.up.bias"] = state_dict["up_path_2.0.up.bias"]
            model_state_dict["48.conv.conv1.weight"] = state_dict[
                "up_path_2.0.conv_block.conv_1.weight"]
            model_state_dict["48.conv.conv1.bias"] = state_dict[
                "up_path_2.0.conv_block.conv_1.bias"]
            model_state_dict["48.conv.conv2.weight"] = state_dict[
                "up_path_2.0.conv_block.conv_2.weight"]
            model_state_dict["48.conv.conv2.bias"] = state_dict[
                "up_path_2.0.conv_block.conv_2.bias"]
            model_state_dict["48.conv.identity.weight"] = state_dict[
                "up_path_2.0.conv_block.identity.weight"]
            model_state_dict["48.conv.identity.bias"] = state_dict[
                "up_path_2.0.conv_block.identity.bias"]
            model_state_dict["49.up.weight"] = state_dict[
                "up_path_2.1.up.weight"]
            model_state_dict["49.up.bias"] = state_dict["up_path_2.1.up.bias"]
            model_state_dict["49.conv.conv1.weight"] = state_dict[
                "up_path_2.1.conv_block.conv_1.weight"]
            model_state_dict["49.conv.conv1.bias"] = state_dict[
                "up_path_2.1.conv_block.conv_1.bias"]
            model_state_dict["49.conv.conv2.weight"] = state_dict[
                "up_path_2.1.conv_block.conv_2.weight"]
            model_state_dict["49.conv.conv2.bias"] = state_dict[
                "up_path_2.1.conv_block.conv_2.bias"]
            model_state_dict["49.conv.identity.weight"] = state_dict[
                "up_path_2.1.conv_block.identity.weight"]
            model_state_dict["49.conv.identity.bias"] = state_dict[
                "up_path_2.1.conv_block.identity.bias"]
            model_state_dict["50.up.weight"] = state_dict[
                "up_path_2.2.up.weight"]
            model_state_dict["50.up.bias"] = state_dict["up_path_2.2.up.bias"]
            model_state_dict["50.conv.conv1.weight"] = state_dict[
                "up_path_2.2.conv_block.conv_1.weight"]
            model_state_dict["50.conv.conv1.bias"] = state_dict[
                "up_path_2.2.conv_block.conv_1.bias"]
            model_state_dict["50.conv.conv2.weight"] = state_dict[
                "up_path_2.2.conv_block.conv_2.weight"]
            model_state_dict["50.conv.conv2.bias"] = state_dict[
                "up_path_2.2.conv_block.conv_2.bias"]
            model_state_dict["50.conv.identity.weight"] = state_dict[
                "up_path_2.2.conv_block.identity.weight"]
            model_state_dict["50.conv.identity.bias"] = state_dict[
                "up_path_2.2.conv_block.identity.bias"]
            model_state_dict["51.up.weight"] = state_dict[
                "up_path_2.3.up.weight"]
            model_state_dict["51.up.bias"] = state_dict["up_path_2.3.up.bias"]
            model_state_dict["51.conv.conv1.weight"] = state_dict[
                "up_path_2.3.conv_block.conv_1.weight"]
            model_state_dict["51.conv.conv1.bias"] = state_dict[
                "up_path_2.3.conv_block.conv_1.bias"]
            model_state_dict["51.conv.conv2.weight"] = state_dict[
                "up_path_2.3.conv_block.conv_2.weight"]
            model_state_dict["51.conv.conv2.bias"] = state_dict[
                "up_path_2.3.conv_block.conv_2.bias"]
            model_state_dict["51.conv.identity.weight"] = state_dict[
                "up_path_2.3.conv_block.identity.weight"]
            model_state_dict["51.conv.identity.bias"] = state_dict[
                "up_path_2.3.conv_block.identity.bias"]
            model_state_dict["52.weight"] = state_dict["last.weight"]
            model_state_dict["52.bias"] = state_dict["last.bias"]
            self.model.load_state_dict(state_dict=model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_weights()

# endregion
