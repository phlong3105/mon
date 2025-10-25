#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements FourierDiff model for zero-shot joint low-light enhancement and deblurring.

References:
    - Paper: "Fourier Priors-Guided Diffusion for Zero-Shot Joint Low-Light
      Enhancement and Deblurring," CVPR 2024.
    - Code: https://github.com/aipixel/FourierDiff
"""

__all__ = [
    "FourierDiff",
]

import box

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, Path, Task
from .guided_diffusion.diffusion_llie_modified import Diffusion

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="fourierdiff", arch="fourierdiff")
class FourierDiff(Diffusion, ModelMixin):
    """FourierDiff model for zero-shot joint low-light enhancement and deblurring.
    
    References:
        - Paper: "Fourier Priors-Guided Diffusion for Zero-Shot Joint Low-Light
          Enhancement and Deblurring," CVPR 2024.
        - Code: https://github.com/aipixel/FourierDiff
    """
    
    arch     : str          = "fourierdiff"
    name     : str          = "fourierdiff"
    tasks    : list[Task]   = [Task.LLE, Task.DEBLUR]
    mltypes  : list[MLType] = [MLType.ZERO_SHOT]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
