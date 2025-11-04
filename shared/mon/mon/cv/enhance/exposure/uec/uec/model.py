#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements UEC model for unsupervised exposure correction.

References:
    - Paper: "Unsupervised Exposure Correction," ECCV 2024.
    - Code: https://github.com/BeyondHeaven/uec_code
"""

__all__ = [
    "UEC",
]

import argparse
from typing import Any

import box

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, Path, Task
from .models.uec_model import UECModel

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="uec", arch="uec")
class UEC(UECModel, ModelMixin):
    """UEC model for unsupervised exposure correction.
    
    References:
        - Paper: "Unsupervised Exposure Correction," ECCV 2024.
        - Code: https://github.com/BeyondHeaven/uec_code
    """
    
    arch     : str          = "uec"
    name     : str          = "uec"
    tasks    : list[Task]   = [Task.EXPOSURE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, opt: argparse.Namespace, weights: Any = None):
        super().__init__(opt)
        # Load weights
        _, path, _ = self.parse_weights(weights)
        self.setup(path, opt)
