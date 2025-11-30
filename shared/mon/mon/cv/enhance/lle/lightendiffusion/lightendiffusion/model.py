#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements LightenDiffusion model for low-light image enhancement.

References:
    - Paper: "LightenDiffusion: Unsupervised Low-Light Image Enhancement with
      Latent-Retinex Diffusion Models," ECCV 2024.
    - Code: https://github.com/JianghaiSCU/LightenDiffusion
"""

__all__ = [
    "LightenDiffusion",
]

import box

from mon import nn
from mon.core import MLType, MODELS, Path, Task
from .models import DenoisingDiffusion

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="lightendiffusion", arch="lightendiffusion")
class LightenDiffusion(DenoisingDiffusion, nn.ModelMixin):
    """LightenDiffusion model for low-light image enhancement.
    
    References:
        - Paper: "LightenDiffusion: Unsupervised Low-Light Image Enhancement with
          Latent-Retinex Diffusion Models," ECCV 2024.
        - Code: https://github.com/JianghaiSCU/LightenDiffusion
    """
    
    _arch     : str          = "lightendiffusion"
    _name     : str          = "lightendiffusion"
    _tasks    : list[Task]   = [Task.LLE]
    _mltypes  : list[MLType] = [MLType.SUPERVISED]
    _model_dir: Path         = root_dir
    _zoo      : dict         = box.Box()
