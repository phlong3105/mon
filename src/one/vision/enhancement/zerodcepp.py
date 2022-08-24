#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from one.nn import *
from one.vision.enhancement.zerodce import CombinedLoss

CURRENT_DIR = Path(__file__).resolve().parent
CFG_DIR     = CURRENT_DIR / "cfg"


# H1: - Model ------------------------------------------------------------------

cfgs = {
    "zerodce++": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,                       args(out_channels, ...)]
            [-1,      1,      Downsample,                   [None, 1, "bilinear"]],   # 0  (x)
            [-1,      1,      DepthwiseSeparableConvReLU2d, [32, 3, 1, 1, 1, 1, 0]],  # 1  (x1)
            [-1,      1,      DepthwiseSeparableConvReLU2d, [32, 3, 1, 1, 1, 1, 0]],  # 2  (x2)
            [-1,      1,      DepthwiseSeparableConvReLU2d, [32, 3, 1, 1, 1, 1, 0]],  # 3  (x3)
            [-1,      1,      DepthwiseSeparableConvReLU2d, [32, 3, 1, 1, 1, 1, 0]],  # 4  (x4)
            [[3, 4],  1,      Concat,                       []],                      # 5
            [-1,      1,      DepthwiseSeparableConvReLU2d, [32, 3, 1, 1, 1, 1, 0]],  # 6  (x5)
            [[2, 6],  1,      Concat,                       []],                      # 7
            [-1,      1,      DepthwiseSeparableConvReLU2d, [32, 3, 1, 1, 1, 1, 0]],  # 8  (x6)
            [[1, 8],  1,      Concat,                       []],                      # 9
            [-1,      1,      DepthwiseSeparableConvReLU2d, [3,  3, 1, 1, 1, 1, 0]],  # 10 (x_r)
            [-1,      1,      Tanh,                         []],                      # 11
            [-1,      1,      UpsamplingBilinear2d,         [None, 1]],               # 12
        ],
        "head": [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve,  [8]],                     # 13
        ]
    }
}


@MODELS.register(name="zerodce++")
class ZeroDCEPP(ImageEnhancementModel):
    """
    
    References:
        https://github.com/Li-Chongyi/Zero-DCE_extension
    """
    
    def __init__(
        self,
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "zerodce++",
        fullname   : str  | None         = "zerodce++",
        cfg        : dict | Path_ | None = "zerodce++.yaml",
        channels   : int                 = 3,
        num_classes: int  | None 		 = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = CombinedLoss(tv_weight=Tensor([1600.0])),
        metrics	   : Metrics_     | None = None,
        optimizers : Optimizers_  | None = None,
        debug      : dict | Munch | None = None,
        verbose    : bool                = False,
        *args, **kwargs
    ):
        cfg = cfg or "zerodce++"
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
            loss        = loss or CombinedLoss(tv_weight=Tensor([1600.0])),
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )
   
    def init_weights(self, m: Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                m.conv.weight.data.normal_(0.0, 0.02)
            elif hasattr(m, "dw_conv"):
                m.dw_conv.weight.data.normal_(0.0, 0.02)
            elif hasattr(m, "pw_conv"):
                m.pw_conv.weight.data.normal_(0.0, 0.02)
            else:
                m.weight.data.normal_(0.0, 0.02)
