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
    "resnet18": {
        "channels": 3,
        "backbone": [
            # [from, number, module,            args(out_channels, ...)]
            [-1,     1,      Conv2d,            [64, 7, 2, 3, 1, 1, False]],                 # 0
            [-1,     1,      BatchNorm2d,       []],                                         # 1
            [-1,     1,      ReLU,              [True]],                                     # 2
            [-1,     1,      MaxPool2d,         [3, 2, 1]],                                  # 3
            
        ],
        "head": [
            [-1,     1,      VGGClassifier,     []],                   # 22
        ]
    },
}


@MODELS.register(name="resnet")
class ResNet(ImageClassificationModel):
    
    def __init__(
        self,
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "resnet",
        fullname   : str  | None         = "resnet50",
        cfg        : dict | Path_ | None = "resnet50",
        channels   : int                 = 3,
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
        cfg = cfg or "vgg11"
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
        if isinstance(m, Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
