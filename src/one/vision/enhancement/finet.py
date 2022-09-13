#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from one.nn import *

CURRENT_DIR = Path(__file__).resolve().parent
CFG_DIR     = CURRENT_DIR / "cfg"


# H1: - Model ------------------------------------------------------------------

cfgs = {
    "finet": {
        "channels": 3,
        "backbone": [
            # [from,       number, module,              args(out_channels, ...)]
            [-1,           1,      Identity,            []],                                              # 0  (x)
            # UNet 01 Down
            [-1,           1,      Conv2d,              [64, 3, 1, 1]],                                   # 1  (x1)
            [[-1],         1,      HINetConvBlock,      [64,   True,  0.2, False, True, 0.5, "linear"]],  # 2  (x1_1_down, x1_1)
            [-1,           1,      ExtractItem,         [0]],                                             # 3  (x1_1_down)
            [[-1],         1,      HINetConvBlock,      [128,  True,  0.2, False, True, 0.5, "linear"]],  # 4  (x1_2_down, x1_2)
            [-1,           1,      ExtractItem,         [0]],                                             # 5  (x1_2_down)
            [[-1],         1,      HINetConvBlock,      [256,  True,  0.2, False, True, 0.5, "linear"]],  # 6  (x1_3_down, x1_3)
            [-1,           1,      ExtractItem,         [0]],                                             # 7  (x1_3_down)
            [[-1],         1,      HINetConvBlock,      [512,  True,  0.2, False, True, 0.5, "linear"]],  # 8  (x1_4_down, x1_4)
            [-1,           1,      ExtractItem,         [0]],                                             # 9  (x1_4_down)
            [[-1],         1,      HINetConvBlock,      [1024, False, 0.2, False, True, 0.5, "linear"]],  # 10 (None,      x1_5)
            # UNet 01 Skip
            [2,            1,      ExtractItem,         [1]],                                             # 11 (x1_1)
            [-1,           1,      Conv2d,              [64,  3, 1, 1]],                                  # 12 (x1_1_skip)
            [4,            1,      ExtractItem,         [1]],                                             # 13 (x1_2)
            [-1,           1,      Conv2d,              [128, 3, 1, 1]],                                  # 14 (x1_2_skip)
            [6,            1,      ExtractItem,         [1]],                                             # 15 (x1_3)
            [-1,           1,      Conv2d,              [256, 3, 1, 1]],                                  # 16 (x1_3_skip)
            [8,            1,      ExtractItem,         [1]],                                             # 17 (x1_4)
            [-1,           1,      Conv2d,              [512, 3, 1, 1]],                                  # 18 (x1_4_skip)
            [10,           1,      ExtractItem,         [1]],                                             # 19 (x1_5)
            # UNet 01 Up
            [[-1, 18],     1,      HINetUpBlock,        [512, 0.2]],                                      # 20 (x1_4_up = x1_5    + x1_4_skip)
            [[-1, 16],     1,      HINetUpBlock,        [256, 0.2]],                                      # 21 (x1_3_up = x1_4_up + x1_3_skip)
            [[-1, 14],     1,      HINetUpBlock,        [128, 0.2]],                                      # 22 (x1_2_up = x1_3_up + x1_2_skip)
            [[-1, 12],     1,      HINetUpBlock,        [64,  0.2]],                                      # 23 (x1_1_up = x1_2_up + x1_1_skip)
            # SAM
            [[-1, 0],      1,      SupervisedAttentionModule, [3]],                                       # 24 (sam_features, y1)
            [-1,           1,      ExtractItem,         [0]],                                             # 25 (sam_features)
            [-2,           1,      ExtractItem,         [1]],                                             # 26 (y1)
            # UNet 02 Down
            [0,            1,      Conv2d,              [64, 3, 1, 1]],                                   # 27 (x2)
            [[-1, 25],     1,      Concat,              []],                                              # 28 (x2 + sam_features)
            [-1,           1,      Conv2d,              [64, 1, 1, 0]],                                   # 29 (x2)
            [[-1, 11, 23], 1,      HINetConvBlock,      [64,   True,  0.2, True,  True, 0.5, "linear"]],  # 30 (x2_1_down, x2_1)
            [-1,           1,      ExtractItem,         [0]],                                             # 31 (x2_1_down)
            [[-1, 13, 22], 1,      HINetConvBlock,      [128,  True,  0.2, True,  True, 0.5, "linear"]],  # 32 (x2_2_down, x2_2)
            [-1,           1,      ExtractItem,         [0]],                                             # 33 (x2_2_down)
            [[-1, 15, 21], 1,      HINetConvBlock,      [256,  True,  0.2, True,  True, 0.5, "linear"]],  # 34 (x2_3_down, x2_3)
            [-1,           1,      ExtractItem,         [0]],                                             # 35 (x2_3_down)
            [[-1, 17, 20], 1,      HINetConvBlock,      [512,  True,  0.2, True,  True, 0.5, "linear"]],  # 36 (x2_4_down, x2_4)
            [-1,           1,      ExtractItem,         [0]],                                             # 37 (x2_4_down)
            [[-1],         1,      HINetConvBlock,      [1024, False, 0.2, False, True, 0.5, "linear"]],  # 38 (None,      x2_5)
            # UNet 02 Skip
            [30,           1,      ExtractItem,         [1]],                                             # 39 (x2_1)
            [-1,           1,      Conv2d,              [64,  3, 1, 1]],                                  # 40 (x2_1_skip)
            [32,           1,      ExtractItem,         [1]],                                             # 41 (x2_2)
            [-1,           1,      Conv2d,              [128, 3, 1, 1]],                                  # 42 (x2_2_skip)
            [34,           1,      ExtractItem,         [1]],                                             # 43 (x2_3)
            [-1,           1,      Conv2d,              [256, 3, 1, 1]],                                  # 44 (x2_3_skip)
            [36,           1,      ExtractItem,         [1]],                                             # 45 (x2_4)
            [-1,           1,      Conv2d,              [512, 3, 1, 1]],                                  # 46 (x2_4_skip)
            [38,           1,      ExtractItem,         [1]],                                             # 47 (x2_5)
            # UNet 02 Up                                                                                  
            [[-1, 46],     1,      HINetUpBlock,        [512, 0.2]],                                      # 48 (x2_4_up = x2_5    + x2_4_skip)
            [[-1, 44],     1,      HINetUpBlock,        [256, 0.2]],                                      # 49 (x2_3_up = x2_4_up + x2_3_skip)
            [[-1, 42],     1,      HINetUpBlock,        [128, 0.2]],                                      # 50 (x2_2_up = x2_3_up + x2_2_skip)
            [[-1, 40],     1,      HINetUpBlock,        [64,  0.2]],                                      # 51 (x2_1_up = x2_2_up + x2_1_skip)
        ],
        "head": [
            [-1,           1,      Conv2d,              [3, 3, 1, 1]],                     # 52
            [[-1,  0],     1,      Sum,                 []],                               # 53
            [[26, -1],     1,      Join,                []],                               # 54
        ]
    },
    "finet-x0.5": {
        "channels": 3,
        "backbone": [
            # [from,       number, module,              args(out_channels, ...)]
            [-1,           1,      Identity,            []],                                              # 0  (x)
            # UNet 01 Down
            [-1,           1,      Conv2d,              [32, 3, 1, 1]],                                   # 1  (x1)
            [[-1],         1,      HINetConvBlock,      [32,  True,  0.2, False, True, 0.5, "linear"]],   # 2  (x1_1_down, x1_1)
            [-1,           1,      ExtractItem,         [0]],                                             # 3  (x1_1_down)
            [[-1],         1,      HINetConvBlock,      [64,  True,  0.2, False, True, 0.5, "linear"]],   # 4  (x1_2_down, x1_2)
            [-1,           1,      ExtractItem,         [0]],                                             # 5  (x1_2_down)
            [[-1],         1,      HINetConvBlock,      [128, True,  0.2, False, True, 0.5, "linear"]],   # 6  (x1_3_down, x1_3)
            [-1,           1,      ExtractItem,         [0]],                                             # 7  (x1_3_down)
            [[-1],         1,      HINetConvBlock,      [256, True,  0.2, False, True, 0.5, "linear"]],   # 8  (x1_4_down, x1_4)
            [-1,           1,      ExtractItem,         [0]],                                             # 9  (x1_4_down)
            [[-1],         1,      HINetConvBlock,      [512, False, 0.2, False, True, 0.5, "linear"]],   # 10 (None,      x1_5)
            # UNet 01 Skip
            [2,            1,      ExtractItem,         [1]],                                             # 11 (x1_1)
            [-1,           1,      Conv2d,              [32,  3, 1, 1]],                                  # 12 (x1_1_skip)
            [4,            1,      ExtractItem,         [1]],                                             # 13 (x1_2)
            [-1,           1,      Conv2d,              [64,  3, 1, 1]],                                  # 14 (x1_2_skip)
            [6,            1,      ExtractItem,         [1]],                                             # 15 (x1_3)
            [-1,           1,      Conv2d,              [128, 3, 1, 1]],                                  # 16 (x1_3_skip)
            [8,            1,      ExtractItem,         [1]],                                             # 17 (x1_4)
            [-1,           1,      Conv2d,              [256, 3, 1, 1]],                                  # 18 (x1_4_skip)
            [10,           1,      ExtractItem,         [1]],                                             # 19 (x1_5)
            # UNet 01 Up
            [[-1, 18],     1,      HINetUpBlock,        [256, 0.2]],                                      # 20 (x1_4_up = x1_5    + x1_4_skip)
            [[-1, 16],     1,      HINetUpBlock,        [128, 0.2]],                                      # 21 (x1_3_up = x1_4_up + x1_3_skip)
            [[-1, 14],     1,      HINetUpBlock,        [64,  0.2]],                                      # 22 (x1_2_up = x1_3_up + x1_2_skip)
            [[-1, 12],     1,      HINetUpBlock,        [32,  0.2]],                                      # 23 (x1_1_up = x1_2_up + x1_1_skip)
            # SAM
            [[-1, 0],      1,      SupervisedAttentionModule, [3]],                                       # 24 (sam_features, y1)
            [-1,           1,      ExtractItem,         [0]],                                             # 25 (sam_features)
            [-2,           1,      ExtractItem,         [1]],                                             # 26 (y1)
            # UNet 02 Down
            [0,            1,      Conv2d,              [32, 3, 1, 1]],                                   # 27 (x2)
            [[-1, 25],     1,      Concat,              []],                                              # 28 (x2 + sam_features)
            [-1,           1,      Conv2d,              [32, 1, 1, 0]],                                   # 29 (x2)
            [[-1, 11, 23], 1,      HINetConvBlock,      [32,  True,  0.2, True,  True, 0.5, "linear"]],   # 30 (x2_1_down, x2_1)
            [-1,           1,      ExtractItem,         [0]],                                             # 31 (x2_1_down)
            [[-1, 13, 22], 1,      HINetConvBlock,      [64,  True,  0.2, True,  True, 0.5, "linear"]],   # 32 (x2_2_down, x2_2)
            [-1,           1,      ExtractItem,         [0]],                                             # 33 (x2_2_down)
            [[-1, 15, 21], 1,      HINetConvBlock,      [128, True,  0.2, True,  True, 0.5, "linear"]],   # 34 (x2_3_down, x2_3)
            [-1,           1,      ExtractItem,         [0]],                                             # 35 (x2_3_down)
            [[-1, 17, 20], 1,      HINetConvBlock,      [256, True,  0.2, True,  True, 0.5, "linear"]],   # 36 (x2_4_down, x2_4)
            [-1,           1,      ExtractItem,         [0]],                                             # 37 (x2_4_down)
            [[-1],         1,      HINetConvBlock,      [512, False, 0.2, False, True, 0.5, "linear"]],   # 38 (None,      x2_5)
            # UNet 02 Skip
            [30,           1,      ExtractItem,         [1]],                                             # 39 (x2_1)
            [-1,           1,      Conv2d,              [32,  3, 1, 1]],                                  # 40 (x2_1_skip)
            [32,           1,      ExtractItem,         [1]],                                             # 41 (x2_2)
            [-1,           1,      Conv2d,              [64,  3, 1, 1]],                                  # 42 (x2_2_skip)
            [34,           1,      ExtractItem,         [1]],                                             # 43 (x2_3)
            [-1,           1,      Conv2d,              [128, 3, 1, 1]],                                  # 44 (x2_3_skip)
            [36,           1,      ExtractItem,         [1]],                                             # 45 (x2_4)
            [-1,           1,      Conv2d,              [256, 3, 1, 1]],                                  # 46 (x2_4_skip)
            [38,           1,      ExtractItem,         [1]],                                             # 47 (x2_5)
            # UNet 02 Up                                                                                  
            [[-1, 46],     1,      HINetUpBlock,        [256, 0.2]],                                      # 48 (x2_4_up = x2_5    + x2_4_skip)
            [[-1, 44],     1,      HINetUpBlock,        [128, 0.2]],                                      # 49 (x2_3_up = x2_4_up + x2_3_skip)
            [[-1, 42],     1,      HINetUpBlock,        [64,  0.2]],                                      # 50 (x2_2_up = x2_3_up + x2_2_skip)
            [[-1, 40],     1,      HINetUpBlock,        [32,  0.2]],                                      # 51 (x2_1_up = x2_2_up + x2_1_skip)
        ],
        "head": [
            [-1,           1,      Conv2d,              [3, 3, 1, 1]],                     # 52
            [[-1,  0],     1,      Sum,                 []],                               # 53
            [[26, -1],     1,      Join,                []],                               # 54
        ]
    },
    "finet-denoise": {
        "channels": 3,
        "backbone": [
            # [from,       number, module,              args(out_channels, ...)]
            [-1,           1,      Identity,            []],                                              # 0  (x)
            # UNet 01 Down
            [-1,           1,      Conv2d,              [32, 3, 1, 1]],                                   # 1  (x1)
            [[-1],         1,      HINetConvBlock,      [32,  True,  0.2, False, False, 0.5, "linear"]],  # 2  (x1_1_down, x1_1)
            [-1,           1,      ExtractItem,         [0]],                                             # 3  (x1_1_down)
            [[-1],         1,      HINetConvBlock,      [64,  True,  0.2, False, False, 0.5, "linear"]],  # 4  (x1_2_down, x1_2)
            [-1,           1,      ExtractItem,         [0]],                                             # 5  (x1_2_down)
            [[-1],         1,      HINetConvBlock,      [128, True,  0.2, False, False, 0.5, "linear"]],  # 6  (x1_3_down, x1_3)
            [-1,           1,      ExtractItem,         [0]],                                             # 7  (x1_3_down)
            [[-1],         1,      HINetConvBlock,      [256, True,  0.2, False, True,  0.5, "linear"]],  # 8  (x1_4_down, x1_4)
            [-1,           1,      ExtractItem,         [0]],                                             # 9  (x1_4_down)
            [[-1],         1,      HINetConvBlock,      [512, False, 0.2, False, True,  0.5, "linear"]],  # 10 (None,      x1_5)
            # UNet 01 Skip
            [2,            1,      ExtractItem,         [1]],                                             # 11 (x1_1)
            [-1,           1,      Conv2d,              [32,  3, 1, 1]],                                  # 12 (x1_1_skip)
            [4,            1,      ExtractItem,         [1]],                                             # 13 (x1_2)
            [-1,           1,      Conv2d,              [64,  3, 1, 1]],                                  # 14 (x1_2_skip)
            [6,            1,      ExtractItem,         [1]],                                             # 15 (x1_3)
            [-1,           1,      Conv2d,              [128, 3, 1, 1]],                                  # 16 (x1_3_skip)
            [8,            1,      ExtractItem,         [1]],                                             # 17 (x1_4)
            [-1,           1,      Conv2d,              [256, 3, 1, 1]],                                  # 18 (x1_4_skip)
            [10,           1,      ExtractItem,         [1]],                                             # 19 (x1_5)
            # UNet 01 Up
            [[-1, 18],     1,      HINetUpBlock,        [256, 0.2]],                                      # 20 (x1_4_up = x1_5    + x1_4_skip)
            [[-1, 16],     1,      HINetUpBlock,        [128, 0.2]],                                      # 21 (x1_3_up = x1_4_up + x1_3_skip)
            [[-1, 14],     1,      HINetUpBlock,        [64,  0.2]],                                      # 22 (x1_2_up = x1_3_up + x1_2_skip)
            [[-1, 12],     1,      HINetUpBlock,        [32,  0.2]],                                      # 23 (x1_1_up = x1_2_up + x1_1_skip)
            # SAM
            [[-1, 0],      1,      SupervisedAttentionModule, [3]],                                       # 24 (sam_features, y1)
            [-1,           1,      ExtractItem,         [0]],                                             # 25 (sam_features)
            [-2,           1,      ExtractItem,         [1]],                                             # 26 (y1)
            # UNet 02 Down
            [0,            1,      Conv2d,              [32, 3, 1, 1]],                                   # 27 (x2)
            [[-1, 25],     1,      Concat,              []],                                              # 28 (x2 + sam_features)
            [-1,           1,      Conv2d,              [32, 1, 1, 0]],                                   # 29 (x2)
            [[-1, 11, 23], 1,      HINetConvBlock,      [32,  True,  0.2, True,  False, 0.5, "linear"]],  # 30 (x2_1_down, x2_1)
            [-1,           1,      ExtractItem,         [0]],                                             # 31 (x2_1_down)
            [[-1, 13, 22], 1,      HINetConvBlock,      [64,  True,  0.2, True,  False, 0.5, "linear"]],  # 32 (x2_2_down, x2_2)
            [-1,           1,      ExtractItem,         [0]],                                             # 33 (x2_2_down)
            [[-1, 15, 21], 1,      HINetConvBlock,      [128, True,  0.2, True,  False, 0.5, "linear"]],  # 34 (x2_3_down, x2_3)
            [-1,           1,      ExtractItem,         [0]],                                             # 35 (x2_3_down)
            [[-1, 17, 20], 1,      HINetConvBlock,      [256, True,  0.2, True,  True,  0.5, "linear"]],  # 36 (x2_4_down, x2_4)
            [-1,           1,      ExtractItem,         [0]],                                             # 37 (x2_4_down)
            [[-1],         1,      HINetConvBlock,      [512, False, 0.2, False, True,  0.5, "linear"]],  # 38 (None,      x2_5)
            # UNet 02 Skip
            [30,           1,      ExtractItem,         [1]],                                             # 39 (x2_1)
            [-1,           1,      Conv2d,              [32,  3, 1, 1]],                                  # 40 (x2_1_skip)
            [32,           1,      ExtractItem,         [1]],                                             # 41 (x2_2)
            [-1,           1,      Conv2d,              [64,  3, 1, 1]],                                  # 42 (x2_2_skip)
            [34,           1,      ExtractItem,         [1]],                                             # 43 (x2_3)
            [-1,           1,      Conv2d,              [128, 3, 1, 1]],                                  # 44 (x2_3_skip)
            [36,           1,      ExtractItem,         [1]],                                             # 45 (x2_4)
            [-1,           1,      Conv2d,              [256, 3, 1, 1]],                                  # 46 (x2_4_skip)
            [38,           1,      ExtractItem,         [1]],                                             # 47 (x2_5)
            # UNet 02 Up
            [[-1, 46],     1,      HINetUpBlock,        [256, 0.2]],                                      # 48 (x2_4_up = x2_5    + x2_4_skip)
            [[-1, 44],     1,      HINetUpBlock,        [128, 0.2]],                                      # 49 (x2_3_up = x2_4_up + x2_3_skip)
            [[-1, 42],     1,      HINetUpBlock,        [64,  0.2]],                                      # 50 (x2_2_up = x2_3_up + x2_2_skip)
            [[-1, 40],     1,      HINetUpBlock,        [32,  0.2]],                                      # 51 (x2_1_up = x2_2_up + x2_1_skip)
        ],
        "head": [
            [-1,           1,      Conv2d,              [3, 3, 1, 1]],                                    # 52
            [[-1,  0],     1,      Sum,                 []],                                              # 53
            [[26, -1],     1,      Join,                []],                                              # 54
        ]
    },
}


@MODELS.register(name="finet")
class FINet(ImageEnhancementModel):
    """
    """
    
    def __init__(
        self,
        root       : Path_               = RUNS_DIR,
        name       : str          | None = "finet",
        fullname   : str          | None = "finet",
        cfg        : dict | Path_ | None = "finet",
        channels   : int                 = 3,
        num_classes: int          | None = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = None,
        metrics	   : Metrics_     | None = None,
        optimizers : Optimizers_  | None = None,
        debug      : dict | Munch | None = None,
        verbose    : bool                = False,
        *args, **kwargs
    ):
        cfg = cfg or "finet"
        if isinstance(cfg, str) and cfg in cfgs:
            cfg = cfgs[cfg]
        elif isinstance(cfg, (str, Path)) and not is_yaml_file(cfg):
            cfg = CFG_DIR / cfg

        super().__init__(
            root        = root,
            name        = name,
            fullname    = fullname,
            cfg         = cfg,
            channels    = channels,
            num_classes = num_classes,
            classlabels = classlabels,
            pretrained  = pretrained,
            phase       = phase,
            loss        = loss,
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )
   
    def init_weights(self, m: Module):
        pass
    
    
@MODELS.register(name="finet-deblur")
class FINetDeblur(FINet):
    """
    """
    
    model_zoo = {}
    
    def __init__(
        self,
        root       : Path_               = RUNS_DIR,
        name       : str          | None = "finet",
        fullname   : str          | None = "finet-deblur",
        cfg        : dict | Path_ | None = "finet",
        channels   : int                 = 3,
        num_classes: int          | None = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = None,
        metrics	   : Metrics_     | None = None,
        optimizers : Optimizers_  | None = None,
        debug      : dict | Munch | None = None,
        verbose    : bool                = False,
        *args, **kwargs
    ):
        cfg = cfg or "finet"
        if isinstance(cfg, str) and cfg in cfgs:
            cfg = cfgs[cfg]
        elif isinstance(cfg, (str, Path)) and not is_yaml_file(cfg):
            cfg = CFG_DIR / cfg

        super().__init__(
            root        = root,
            name        = name,
            fullname    = fullname,
            cfg         = cfg,
            channels    = channels,
            num_classes = num_classes,
            classlabels = classlabels,
            pretrained  = FINetDeblur.init_pretrained(pretrained),
            phase       = phase,
            loss        = loss,
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )


@MODELS.register(name="finet-dehaze")
class FINetDehaze(FINet):
    """
    """
    
    model_zoo = {}
    
    def __init__(
        self,
        root       : Path_               = RUNS_DIR,
        name       : str          | None = "finet",
        fullname   : str          | None = "finet-dehaze",
        cfg        : dict | Path_ | None = "finet",
        channels   : int                 = 3,
        num_classes: int          | None = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = None,
        metrics	   : Metrics_     | None = None,
        optimizers : Optimizers_  | None = None,
        debug      : dict | Munch | None = None,
        verbose    : bool                = False,
        *args, **kwargs
    ):
        cfg = cfg or "finet"
        if isinstance(cfg, str) and cfg in cfgs:
            cfg = cfgs[cfg]
        elif isinstance(cfg, (str, Path)) and not is_yaml_file(cfg):
            cfg = CFG_DIR / cfg
    
        super().__init__(
            root        = root,
            name        = name,
            fullname    = fullname,
            cfg         = cfg,
            channels    = channels,
            num_classes = num_classes,
            classlabels = classlabels,
            pretrained  = FINetDehaze.init_pretrained(pretrained),
            phase       = phase,
            loss        = loss,
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )


@MODELS.register(name="FINet_denoise")
class FINetDenoise(FINet):
    """
    """
    
    model_zoo = {}
    
    def __init__(
        self,
        root       : Path_               = RUNS_DIR,
        name       : str          | None = "finet",
        fullname   : str          | None = "finet-denoise",
        cfg        : dict | Path_ | None = "finet-denoise",
        channels   : int                 = 3,
        num_classes: int          | None = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = None,
        metrics	   : Metrics_     | None = None,
        optimizers : Optimizers_  | None = None,
        debug      : dict | Munch | None = None,
        verbose    : bool                = False,
        *args, **kwargs
    ):
        cfg = cfg or "finet-denoise"
        if isinstance(cfg, str) and cfg in cfgs:
            cfg = cfgs[cfg]
        elif isinstance(cfg, (str, Path)) and not is_yaml_file(cfg):
            cfg = CFG_DIR / cfg

        super().__init__(
            root        = root,
            name        = name,
            fullname    = fullname,
            cfg         = cfg,
            channels    = channels,
            num_classes = num_classes,
            classlabels = classlabels,
            pretrained  = FINetDeblur.init_pretrained(pretrained),
            phase       = phase,
            loss        = loss,
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )
   
   
@MODELS.register(name="finet-denoise-x0.5")
class FINetDenoise_x0_5(FINet):
    """
    """
    
    model_zoo = {}
    
    def __init__(
        self,
        root       : Path_               = RUNS_DIR,
        name       : str          | None = "finet",
        fullname   : str          | None = "finet-denoise-x0.5",
        cfg        : dict | Path_ | None = "finet-x0.5",
        channels   : int                 = 3,
        num_classes: int          | None = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = None,
        metrics	   : Metrics_     | None = None,
        optimizers : Optimizers_  | None = None,
        debug      : dict | Munch | None = None,
        verbose    : bool                = False,
        *args, **kwargs
    ):
        cfg = cfg or "finet-x0.5"
        if isinstance(cfg, str) and cfg in cfgs:
            cfg = cfgs[cfg]
        elif isinstance(cfg, (str, Path)) and not is_yaml_file(cfg):
            cfg = CFG_DIR / cfg

        super().__init__(
            root        = root,
            name        = name,
            fullname    = fullname,
            cfg         = cfg,
            channels    = channels,
            num_classes = num_classes,
            classlabels = classlabels,
            pretrained  = FINetDenoise_x0_5.init_pretrained(pretrained),
            phase       = phase,
            loss        = loss,
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )


@MODELS.register(name="finet-derain")
class FINetDerain(FINet):
    """
    """
    
    model_zoo = {}
    
    def __init__(
        self,
        root       : Path_               = RUNS_DIR,
        name       : str          | None = "finet",
        fullname   : str          | None = "finet-derain",
        cfg        : dict | Path_ | None = "finet",
        channels   : int                 = 3,
        num_classes: int          | None = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = None,
        metrics	   : Metrics_     | None = None,
        optimizers : Optimizers_  | None = None,
        debug      : dict | Munch | None = None,
        verbose    : bool                = False,
        *args, **kwargs
    ):
        cfg = cfg or "finet"
        if isinstance(cfg, str) and cfg in cfgs:
            cfg = cfgs[cfg]
        elif isinstance(cfg, (str, Path)) and not is_yaml_file(cfg):
            cfg = CFG_DIR / cfg

        super().__init__(
            root        = root,
            name        = name,
            fullname    = fullname,
            cfg         = cfg,
            channels    = channels,
            num_classes = num_classes,
            classlabels = classlabels,
            pretrained  = FINetDeblur.init_pretrained(pretrained),
            phase       = phase,
            loss        = loss,
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )
