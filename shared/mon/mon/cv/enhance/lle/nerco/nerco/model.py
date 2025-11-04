#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements NeRCo model for low-light image enhancement.

References:
    - Paper: "Implicit Neural Representation for Cooperative Low-light
      Image Enhancement," ICCV 2023.
    - Code: https://github.com/Ysz2022/NeRCo
"""

__all__ = [
    "NeRCo",
]

import argparse
from typing import Any

import box

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, Path, Task
from .models.nerco_model import NeRComodel

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="nerco", arch="nerco")
class NeRCo(NeRComodel, ModelMixin):
    """NeRCo model for low-light image enhancement.
    
    References:
        - Paper: "Implicit Neural Representation for Cooperative Low-light
          Image Enhancement," ICCV 2023.
        - Code: https://github.com/Ysz2022/NeRCo
    """
    
    arch     : str          = "nerco"
    name     : str          = "nerco"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, opt: argparse.Namespace, weights: Any = None):
        super().__init__(opt)
        # Load weights
        _, path, _ = self.parse_weights(weights)
        self.setup(path, opt)
