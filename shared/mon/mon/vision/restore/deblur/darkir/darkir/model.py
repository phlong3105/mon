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

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, Path, Task
from . import archs

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


class DarkIR(archs.DarkIR, ModelMixin):
    """DarkIR model for low-light deblurring.
    
    References:
        - Paper: "DarkIR: Robust Low-Light Image Restoration," CVPR 2025.
        - Code: https://github.com/cidautai/DarkIR
    """
    
    arch     : str          = "darkir"
    name     : str          = "darkir"
    tasks    : list[Task]   = [Task.LLE, Task.DEBLUR]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()


MODELS.register(name="darkir_m", arch="darkir", module=DarkIR)
MODELS.register(name="darkir_l", arch="darkir", module=DarkIR)
