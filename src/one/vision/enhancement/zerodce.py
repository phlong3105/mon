#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from one.nn import *


# H1: - Loss -------------------------------------------------------------------

class CombinedLoss(BaseLoss):
    """
    Loss = loss_spa + loss_exp + loss_col + loss_tv.
    """
    
    def __init__(
        self,
        spa_weight    : Tensor = Tensor([1.0]),
	    exp_patch_size: int    = 16,
	    exp_mean_val  : float  = 0.6,
        exp_weight    : Tensor = Tensor([10.0]),
        col_weight    : Tensor = Tensor([5.0]),
        tv_weight     : Tensor = Tensor([200.0]),
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.name     = "combined_loss"
        self.loss_spa = SpatialConsistencyLoss(weight=spa_weight)
        self.loss_exp = ExposureControlLoss(
            patch_size = exp_patch_size,
            mean_val   = exp_mean_val,
            weight     = exp_weight
        )
        self.loss_col = ColorConstancyLoss(weight=col_weight)
        self.loss_tv  = IlluminationSmoothnessLoss(weight=tv_weight)
     
    def forward(self, input: Tensors, target: Tensor, **_) -> Tensor:
        r, enhance = target[0], target[-1]
        return self.loss_spa(input=input, target=enhance) \
               + self.loss_exp(input=enhance) \
               + self.loss_col(input=enhance) \
               + self.loss_tv(input=r)
    

# H1: - Model ------------------------------------------------------------------

cfgs = {
    "zerodce": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,     args(out_channels, ...)]
            [-1,      1,      Identity,   []],             # 0  (x)
            [-1,      1,      ConvReLU2d, [32, 3, 1, 1]],  # 1  (x1)
            [-1,      1,      ConvReLU2d, [32, 3, 1, 1]],  # 2  (x2)
            [-1,      1,      ConvReLU2d, [32, 3, 1, 1]],  # 3  (x3)
            [-1,      1,      ConvReLU2d, [32, 3, 1, 1]],  # 4  (x4)
            [[3, 4],  1,      Concat,     []],             # 5
            [-1,      1,      ConvReLU2d, [32, 3, 1, 1]],  # 6  (x5)
            [[2, 6],  1,      Concat,     []],             # 7
            [-1,      1,      ConvReLU2d, [32, 3, 1, 1]],  # 8  (x6)
            [[1, 8],  1,      Concat,     []],             # 9
            [-1,      1,      ConvReLU2d, [24, 3, 1, 1]],  # 10 (x_r)
            [-1,      1,      Tanh,       []],             # 11
        ],                                
        "head": [                         
            [[0, 11], 1, PixelwiseCurve8, []],             # 12
        ]
    }
}


class ZeroDCE(ImageEnhancementModel):
    """
    
    References:
        https://github.com/Li-Chongyi/Zero-DCE
    """
    
    def __init__(
        self,
        root       : Path_               = RUNS_DIR,
        basename   : str  | None         = "zerodce",
        name       : str  | None         = "zerodce",
        cfg        : dict | Path_ | None = None,
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
        cfg = cfg or "zerodce"
        if isinstance(cfg, str) and cfg in cfgs:
            cfg = cfgs[cfg]
        super().__init__(
            root        = root,
            basename    = basename,
            name        = name,
            cfg         = cfg,
            channels    = channels,
            num_classes = num_classes,
            pretrained  = pretrained,
            phase       = phase,
            loss        = loss or CombinedLoss(),
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )
   
    def init_weights(self):
        for m in self.model.children():
            classname = m.__class__.__name__
            if classname.find("Conv") != -1:
                if hasattr(m, "conv"):
                    m.conv.weight.data.normal_(0.0, 0.02)
                else:
                    m.weight.data.normal_(0.0, 0.02)
