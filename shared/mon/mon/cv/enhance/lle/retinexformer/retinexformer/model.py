#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Retinexformer model for low-light image enhancement.

References:
    - Paper: "Retinexformer: One-stage Retinex-based Transformer for Low-light
      Image Enhancement," ICCV 2023.
    - Code: https://github.com/caiyuanhao1998/Retinexformer
"""

__all__ = [
    "Retinexformer",
]

import box

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, Path, Task
from .basicsr.models.image_restoration_model import ImageCleanModel

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


@MODELS.register(name="retinexformer", arch="retinexformer")
class Retinexformer(ImageCleanModel, ModelMixin):
    """Retinexformer model for low-light image enhancement.
    
    References:
        - Paper: "Retinexformer: One-stage Retinex-based Transformer for Low-light
          Image Enhancement," ICCV 2023.
        - Code: https://github.com/caiyuanhao1998/Retinexformer
    """
    
    arch     : str          = "retinexformer"
    name     : str          = "retinexformer"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
