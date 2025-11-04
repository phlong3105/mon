#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements NeurOP model for image retouching.

References:
    - Paper: "Neural Color Operators for Sequential Image Retouching," ECCV 2022.
    - Code: https://github.com/amberwangyili/neurop
"""

__all__ = [
    "NeurOP",
    "NeurOPInit",
]

import box

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, Path, Task
from .models.model import FinetuneModel, InitModel

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="neurop", arch="neurop")
class NeurOP(FinetuneModel, ModelMixin):
    """NeurOP model for image retouching.
    
    References:
        - Paper: "Neural Color Operators for Sequential Image Retouching," ECCV 2022.
        - Code: https://github.com/amberwangyili/neurop
    """
    
    arch     : str          = "neurop"
    name     : str          = "neurop"
    tasks    : list[Task]   = [Task.RETOUCH]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()


class NeurOPInit(InitModel):
    """NeurOP model for image retouching.
    
    References:
        - Paper: "Neural Color Operators for Sequential Image Retouching," ECCV 2022.
        - Code: https://github.com/amberwangyili/neurop
    """
    pass
