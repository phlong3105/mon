#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements PSENet model for low-light image enhancement.

References:
    - Paper: "PSENet: Progressive Self-Enhancement Network for Unsupervised
      Extreme-Light Image Enhancement," WACV 2023.
    - Code: https://github.com/VinAIResearch/PSENet-Image-Enhancement
"""

__all__ = [
    "PSENet",
]

from typing import Any

import box

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, Path, Task
from .module import UnetTMO

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


def read_pytorch_lightning_state_dict(ckpt):
    new_state_dict = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("model."):
            new_state_dict[k[len("model.") :]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


@MODELS.register(name="psenet", arch="psenet")
class PSENet(UnetTMO, ModelMixin):
    """PSENet model for low-light image enhancement.
    
    References:
        - Paper: "PSENet: Progressive Self-Enhancement Network for Unsupervised
          Extreme-Light Image Enhancement," WACV 2023.
        - Code: https://github.com/VinAIResearch/PSENet-Image-Enhancement
    """
    
    arch     : str          = "psenet"
    name     : str          = "psenet"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, weights: Any = None):
        super().__init__()
        # Load weights
        weights, _, _ = self.parse_weights(weights)
        weights = read_pytorch_lightning_state_dict(weights)
        self.load_weights(weights)
