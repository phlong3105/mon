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

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, Path, Task
from .models import sronet

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="srno", arch="srno")
class SRNO(sronet.SRNO, ModelMixin):
    """SRNO model for super-resolution.
    
    References:
        - Paper: "Super-Resolution Neural Operator," CVPR 2023.
        - Code: https://github.com/2y7c3/Super-Resolution-Neural-Operator
    """
    
    arch     : str          = "srno"
    name     : str          = "srno"
    tasks    : list[Task]   = [Task.SR]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
