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

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, Path, Task
from .cldm.cldm import ControlLDM

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="quadprior", arch="quadprior")
class QuadPrior(ControlLDM, ModelMixin):
    """QuadPrior model for low-light image enhancement.
    
    References:
        - Paper: "Zero-Reference Low-Light Enhancement via Physical Quadruple
          Priors," CVPR 2024.
        - Code: https://github.com/daooshee/QuadPrior
    """
    
    arch     : str          = "quadprior"
    name     : str          = "quadprior"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
