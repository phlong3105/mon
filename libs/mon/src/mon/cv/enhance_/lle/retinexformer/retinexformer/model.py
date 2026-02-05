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

from mon import nn
from mon.core import MLType, MODELS, Path, Task
from .basicsr.models.image_restoration_model import ImageCleanModel

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


@MODELS.register(variant="retinexformer", name="retinexformer")
class Retinexformer(ImageCleanModel, nn.ModelMetadataMixin):
    """Retinexformer model for low-light image enhancement.
    
    References:
        - Paper: "Retinexformer: One-stage Retinex-based Transformer for Low-light
          Image Enhancement," ICCV 2023.
        - Code: https://github.com/caiyuanhao1998/Retinexformer
    """
    
    _arch     : str          = "retinexformer"
    _name     : str          = "retinexformer"
    _tasks    : list[Task]   = [Task.LLE]
    _mltypes  : list[MLType] = [MLType.SUPERVISED]
    _model_dir: Path         = root_dir
    _zoo      : dict         = box.Box()
