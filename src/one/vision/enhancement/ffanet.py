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
    "ffanet": {
        "channels": 3,
        "backbone": [
            # [from,    number, module,              args(out_channels, ...)]
            [-1,        1,      Identity,            []],         # 0  (x)
            [-1,        1,      FFAPreProcess,       [64, 3]],    # 1  (x1)
            [-1,        1,      FFAGroup,            [3, 20]],    # 2  (g1)
            [-1,        1,      FFAGroup,            [3, 20]],    # 3  (g2)
            [-1,        1,      FFAGroup,            [3, 20]],    # 4  (g3)
            [[2, 3, 4], 1,      FFA,                 [3]],        # 5
            [-1,        1,      FFAPostProcess,      [3, 3]],     # 6
        ],
        "head": [
            [[0, 6],    1,      Sum,                 [8]],        # 8
        ]
    },
}


@MODELS.register(name="ffanet")
class FFANet(ImageEnhancementModel):
    """
    """
    
    def __init__(
        self,
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "ffanet",
        fullname   : str  | None         = "ffanet",
        cfg        : dict | Path_ | None = "ffanet.yaml",
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
        cfg = cfg or "ffanet"
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
   
    def init_weights(self):
        pass
