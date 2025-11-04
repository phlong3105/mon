#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements SGZ model for low-light image enhancement.

References:
    - Paper: "Semantic-Guided Zero-Shot Learning for Low-Light Image/Video
      Enhancement," WACV 2022.
    - Code: https://github.com/ShenZheng2000/Semantic-Guided-Low-Light-Image-Enhancement
"""

__all__ = [
    "SGZ",
]

from typing import Any

import box

import mon.nn as nn
from mon.constants import MODELS, ROOT_DIR
from mon.core import MLType, Path, Task
from .modeling.model import enhance_net_nopool

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="sgz", arch="sgz")
class SGZ(enhance_net_nopool, nn.ModelMixin):
    """SGZ model for low-light image enhancement.
    
    References:
        - Paper: "Semantic-Guided Zero-Shot Learning for Low-Light Image/Video
          Enhancement," WACV 2022.
        - Code: https://github.com/ShenZheng2000/Semantic-Guided-Low-Light-Image-Enhancement
    """
    
    arch     : str          = "sgz"
    name     : str          = "sgz"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box({
        "lolv1": {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/sgz/sgz/lolv1/sgz_lolv1.pt",
            "num_classes": None,
        },
    })
    
    def __init__(self, scale_factor, conv_type="dsc", weights: Any = None):
        super().__init__(scale_factor=scale_factor, conv_type=conv_type)
        # Load weights
        self.load_weights(weights)
