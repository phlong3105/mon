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

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, Path, Task
from .models import DenoisingDiffusion

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="lightendiffusion", arch="lightendiffusion")
class LightenDiffusion(DenoisingDiffusion, ModelMixin):
    """LightenDiffusion model for low-light image enhancement.
    
    References:
        - Paper: "LightenDiffusion: Unsupervised Low-Light Image Enhancement with
          Latent-Retinex Diffusion Models," ECCV 2024.
        - Code: https://github.com/JianghaiSCU/LightenDiffusion
    """
    
    arch     : str          = "lightendiffusion"
    name     : str          = "lightendiffusion"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
