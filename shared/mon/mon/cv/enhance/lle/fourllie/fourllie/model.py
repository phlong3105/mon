#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements FourLLIE model for low-light image enhancement.

References:
    - Paper: "FourLLIE: Boosting Low-Light Image Enhancement by Fourier Frequency
      Information," ACMMM 2023.
    - Code: https://github.com/wangchx67/FourLLIE
"""

__all__ = [
    "FourLLIE",
    "option",
    "read_img",
    "tensor2img",
]

import box
import torch
from thop import profile

from mon import nn
from mon.core import image as I, MLType, MODELS, Path, ROOT_DIR, Task
from .data.util import read_img
from .models.enhancement_model import enhancement_model
from .option import options as option
from .utils.util import tensor2img

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="fourllie", arch="fourllie")
class FourLLIE(enhancement_model, nn.ModelMixin):
    """FourLLIE model for low-light image enhancement.
    
    References:
        - Paper: "FourLLIE: Boosting Low-Light Image Enhancement by Fourier Frequency
          Information," ACMMM 2023.
        - Code: https://github.com/wangchx67/FourLLIE
    """
    
    _arch     : str          = "fourllie"
    _name     : str          = "fourllie"
    _tasks    : list[Task]   = [Task.LLE]
    _mltypes  : list[MLType] = [MLType.SUPERVISED]
    _model_dir: Path         = root_dir
    _zoo      : dict         = box.Box({
        "lolv2real" : {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/fourllie/fourllie/lolv2real/fourllie_lolv2real.pth",
            "num_classes": None,
        },
        "lolv2syn"  : {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/fourllie/fourllie/lolv2syn/fourllie_lolv2syn.pth",
            "num_classes": None,
        },
        "lsrwhuawei": {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/fourllie/fourllie/lsrwhuawei/fourllie_lsrwhuawei.pth",
            "num_classes": None,
        },
        "lsrwnikon" : {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/fourllie/fourllie/lsrwnikon/fourllie_lsrwnikon.pth",
            "num_classes": None,
        },
    })
    
    def __init__(self, cfgs: dict, weights: any = None):
        _, path, _ = self.parse_weights(weights, None)
        if path:
            cfgs["path"]["pretrain_model_G"] = str(path)
        super().__init__(cfgs)
    
    def forward(self):
        self.test()
        
    def compute_complexity(self, imgsz: int = 512, channels: int = 3) -> tuple[float, float]:
        h, w  = I.imgsz(imgsz)
        input = torch.rand(1, channels, h, w).to(self.device)
        data  = {
            "idx": 0,
            "LQs": input,
            "nf" : input,
        }
        self.feed_data(data, need_GT=False)
        flops, params = profile(self, inputs=(), verbose=False)
        return flops, params
