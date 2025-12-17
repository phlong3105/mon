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

from mon import nn
from mon.core import MLType, MODELS, Path, Task
from .network import Finetunemodel, Network

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="zeroig", arch="zeroig")
class ZERO_IG(Network, nn.ModelMetadataMixin):
    """ZERO-IG model for low-light image enhancement.
    
    References:
        - Paper: "Zero-Shot Illumination-Guided Joint Denoising and Adaptive
          Enhancement for Low-Light Images," CVPR 2024.
        - Code: https://github.com/Doyle59217/ZeroIG
    """
    
    _arch     : str          = "zeroig"
    _name     : str          = "zeroig"
    _tasks    : list[Task]   = [Task.LLE]
    _mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    _model_dir: Path         = root_dir
    _zoo      : dict         = box.Box()


class ZERO_IG_Finetune(Finetunemodel):
    """ZERO-IG model for low-light image enhancement.
    
    References:
        - Paper: "Zero-Shot Illumination-Guided Joint Denoising and Adaptive
          Enhancement for Low-Light Images," CVPR 2024.
        - Code: https://github.com/Doyle59217/ZeroIG
    """
    pass
