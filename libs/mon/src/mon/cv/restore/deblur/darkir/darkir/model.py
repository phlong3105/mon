#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements DarkIR model for low-light deblurring.

References:
    - Paper: "DarkIR: Robust Low-Light Image Restoration," CVPR 2025.
    - Code: https://github.com/cidautai/DarkIR
"""

__all__ = [
    "DarkIR",
]

import box

from mon import nn
from mon.core import MLType, MODELS, Path, Task
from . import archs

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


class DarkIR(archs.DarkIR, nn.ModelMetadataMixin):
    """DarkIR model for low-light deblurring.
    
    References:
        - Paper: "DarkIR: Robust Low-Light Image Restoration," CVPR 2025.
        - Code: https://github.com/cidautai/DarkIR
    """
    
    _arch     : str          = "darkir"
    _name     : str          = "darkir"
    _tasks    : list[Task]   = [Task.LLE, Task.DEBLUR]
    _mltypes  : list[MLType] = [MLType.SUPERVISED]
    _model_dir: Path         = root_dir
    _zoo      : dict         = box.Box()


MODELS.register(variant="darkir_m", name="darkir", module=DarkIR)
MODELS.register(variant="darkir_l", name="darkir", module=DarkIR)
