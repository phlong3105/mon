#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements MEFODE model for multi-exposure fusion.

References:
    - Paper: "Continuous Exposure Learning for Multi-Exposure Fusion using
      Neural ODEs," arXiv 2025.
    - Code:
"""

__all__ = [
    "MEFODE",
]

from typing import Any

import box

import mon.nn as nn
from mon.constants import MODELS
from mon.core import MLType, Path, Task
from .network import NODE

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="mefode", arch="mefode")
class MEFODE(NODE, nn.ModelMixin):
    """MEFODE model for multi-exposure fusion.

    References:
        - Paper: "Continuous Exposure Learning for Multi-Exposure Fusion using
          Neural ODEs," arXiv 2025.
        - Code:
    """
    
    arch     : str          = "mefode"
    name     : str          = "mefode"
    tasks    : list[Task]   = [Task.MEF, Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, weights: Any = None, *args, **kwarg):
        super().__init__(*args, **kwarg)
        # Load weights
        self.load_weights(weights, strict=False)
