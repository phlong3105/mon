#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements DCC-Net model for low-light image enhancement.

References:
    - Paper: "Deep Color Consistent Network for Low Light-Image Enhancement," CVPR 2022.
    - Code: https://github.com/Ian0926/DCC-Net
"""

__all__ = [
    "DCCNet",
]

from typing import Any

import box
import torch

from mon import nn
from mon.core import MLType, MODELS, Path, ROOT_DIR, Task
from .module import c_net, g_net, r_net

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


# --- Model ---
@MODELS.register(variant="dccnet", name="dccnet")
class DCCNet(nn.Module, nn.ModelMetadataMixin):
    """DCC-Net model for low-light image enhancement.

    References:
        - Paper: "Deep Color Consistent Network for Low Light-Image Enhancement," CVPR 2022.
        - Code: https://github.com/Ian0926/DCC-Net
    """
    
    _arch     : str          = "dccnet"
    _name     : str          = "dccnet"
    _tasks    : list[Task]   = [Task.LLE]
    _mltypes  : list[MLType] = [MLType.SUPERVISED]
    _model_dir: Path         = root_dir
    _zoo      : dict         = box.Box({
        "lolv1": {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/dccnet/dccnet/lolv1/dccnet_lolv1.pth",
            "num_classes": None,
        },
    })
    
    def __init__(self, d_hist: int = 64, weights: Any = None):
        super().__init__()
        self.g_net = g_net()
        self.c_net = c_net(d_hist)
        self.r_net = r_net()
        
        # Load weights
        self.load_weights(weights)
        
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gray     = self.g_net(image)
        color_hist, color_feature = self.c_net(image)
        enhanced = self.r_net(image, gray, color_feature)
        return gray, color_hist, enhanced
