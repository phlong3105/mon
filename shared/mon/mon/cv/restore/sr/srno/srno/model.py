#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements SRNO model for super-resolution.

References:
    - Paper: "Super-Resolution Neural Operator," CVPR 2023.
    - Code: https://github.com/2y7c3/Super-Resolution-Neural-Operator
"""

__all__ = [
    "SRNO",
]

import box

from mon import nn
from mon.core import MLType, MODELS, Path, Task
from .models import sronet

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="srno", arch="srno")
class SRNO(sronet.SRNO, nn.ModelMixin):
    """SRNO model for super-resolution.
    
    References:
        - Paper: "Super-Resolution Neural Operator," CVPR 2023.
        - Code: https://github.com/2y7c3/Super-Resolution-Neural-Operator
    """
    
    _arch     : str          = "srno"
    _name     : str          = "srno"
    _tasks    : list[Task]   = [Task.SR]
    _mltypes  : list[MLType] = [MLType.SUPERVISED]
    _model_dir: Path         = root_dir
    _zoo      : dict         = box.Box()
