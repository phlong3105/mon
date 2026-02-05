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

from mon import nn
from mon.core import MLType, MODELS, Path, ROOT_DIR, Task
from .net.net import net

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(variant="pairlie", name="pairlie")
class PairLIE(net, nn.ModelMetadataMixin):
    """PairLIE model for low-light image enhancement.
    
    References:
        - Paper: "Learning a Simple Low-light Image Enhancer from Paired Low-light
          Instances," CVPR 2023.
        - Code: https://github.com/zhenqifu/PairLIE
    """
    
    _arch     : str          = "pairlie"
    _name     : str          = "pairlie"
    _tasks    : list[Task]   = [Task.LLE]
    _mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    _model_dir: Path         = root_dir
    _zoo      : dict         = box.Box({
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
