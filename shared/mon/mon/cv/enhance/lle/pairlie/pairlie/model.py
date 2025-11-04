#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements PairLIE model for low-light image enhancement.

References:
    - Paper: "Learning a Simple Low-light Image Enhancer from Paired Low-light
      Instances," CVPR 2023.
    - Code: https://github.com/zhenqifu/PairLIE
"""

__all__ = [
    "PairLIE",
]

from typing import Any

import box

import mon.nn as nn
from mon.constants import MODELS, ROOT_DIR
from mon.core import MLType, Path, Task
from .net.net import net

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="pairlie", arch="pairlie")
class PairLIE(net, nn.ModelMixin):
    """PairLIE model for low-light image enhancement.
    
    References:
        - Paper: "Learning a Simple Low-light Image Enhancer from Paired Low-light
          Instances," CVPR 2023.
        - Code: https://github.com/zhenqifu/PairLIE
    """
    
    arch     : str          = "pairlie"
    name     : str          = "pairlie"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box({
        "sice": {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/pairlie/pairlie/sice/pairlie_sice.pth",
            "num_classes": None,
        },
    })
    
    def __init__(self, weights: Any = None):
        super().__init__()
        # Load weights
        self.load_weights(weights)
