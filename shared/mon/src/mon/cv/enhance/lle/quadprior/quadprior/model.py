#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements QuadPrior model for low-light image enhancement.

References:
    - Paper: "Zero-Reference Low-Light Enhancement via Physical Quadruple
      Priors," CVPR 2024.
    - Code: https://github.com/daooshee/QuadPrior
"""

__all__ = [
    "QuadPrior",
]

import box

from mon import nn
from mon.core import MLType, MODELS, Path, Task
from .cldm.cldm import ControlLDM

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="quadprior", arch="quadprior")
class QuadPrior(ControlLDM, nn.ModelMetadataMixin):
    """QuadPrior model for low-light image enhancement.
    
    References:
        - Paper: "Zero-Reference Low-Light Enhancement via Physical Quadruple
          Priors," CVPR 2024.
        - Code: https://github.com/daooshee/QuadPrior
    """
    
    _arch     : str          = "quadprior"
    _name     : str          = "quadprior"
    _tasks    : list[Task]   = [Task.LLE]
    _mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    _model_dir: Path         = root_dir
    _zoo      : dict         = box.Box()
