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

from mon import nn
from mon.core import MLType, MODELS, Path, Task
from .guided_diffusion.diffusion_llie_modified import Diffusion

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="fourierdiff", arch="fourierdiff")
class FourierDiff(Diffusion, nn.ModelMixin):
    """FourierDiff model for zero-shot joint low-light enhancement and deblurring.
    
    References:
        - Paper: "Fourier Priors-Guided Diffusion for Zero-Shot Joint Low-Light
          Enhancement and Deblurring," CVPR 2024.
        - Code: https://github.com/aipixel/FourierDiff
    """
    
    _arch     : str          = "fourierdiff"
    _name     : str          = "fourierdiff"
    _tasks    : list[Task]   = [Task.LLE, Task.DEBLUR]
    _mltypes  : list[MLType] = [MLType.ZERO_SHOT]
    _model_dir: Path         = root_dir
    _zoo      : dict         = box.Box()
