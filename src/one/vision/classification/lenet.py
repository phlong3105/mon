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
    "lenet": {
        "channels": 1,
        "backbone": [
            # [from, number, module,          args(out_channels, ...)]
            [-1,     1,      Conv2d,          [6, 5, 1]],    # 0
            [-1,     1,      Tanh,            []],           # 1
            [-1,     1,      AvgPool2d,       [2]],          # 2
            [-1,     1,      Conv2d,          [16, 5, 1]],   # 3
            [-1,     1,      Tanh,            []],           # 4
            [-1,     1,      AvgPool2d,       [2]],          # 5
            [-1,     1,      Conv2d,          [120, 5, 1]],  # 6
            [-1,     1,      Tanh,            []],           # 7
        ],
        "head": [                                           
            [-1,     1,      LeNetClassifier, []],           # 8
        ]
    },
}


@MODELS.register(name="lenet")
class LeNet(ImageClassificationModel):
    
    def __init__(
        self,
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "lenet",
        fullname   : str  | None         = "lenet",
        cfg        : dict | Path_ | None = "lenet.yaml",
        channels   : int                 = 1,
        num_classes: int  | None 		 = None,
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
        cfg = cfg or "lenet"
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
