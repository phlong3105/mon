#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ZERO-IG model for low-light image enhancement.

References:
    - Paper: "Zero-Shot Illumination-Guided Joint Denoising and Adaptive
      Enhancement for Low-Light Images," CVPR 2024.
    - Code: https://github.com/Doyle59217/ZeroIG
"""

__all__ = [
    "ZERO_IG",
    "ZERO_IG_Finetune",
]

import box

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, Path, Task
from .network import Finetunemodel, Network

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="zeroig", arch="zeroig")
class ZERO_IG(Network, ModelMixin):
    """ZERO-IG model for low-light image enhancement.
    
    References:
        - Paper: "Zero-Shot Illumination-Guided Joint Denoising and Adaptive
          Enhancement for Low-Light Images," CVPR 2024.
        - Code: https://github.com/Doyle59217/ZeroIG
    """
    
    arch     : str          = "zeroig"
    name     : str          = "zeroig"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()


class ZERO_IG_Finetune(Finetunemodel):
    """ZERO-IG model for low-light image enhancement.
    
    References:
        - Paper: "Zero-Shot Illumination-Guided Joint Denoising and Adaptive
          Enhancement for Low-Light Images," CVPR 2024.
        - Code: https://github.com/Doyle59217/ZeroIG
    """
    pass
