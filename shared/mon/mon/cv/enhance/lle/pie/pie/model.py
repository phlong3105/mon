#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements PIE model for low-light image enhancement.

References:
    - Paper: "A Probabilistic Method for Image Enhancement With Simultaneous
      Illumination and Reflectance Estimation," IEEE TIP 2015.
    - Code: https://github.com/DavidQiuChao/PIE
"""

__all__ = [
    "PIE",
]

import box
import cv2
import numpy as np

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, nn, Path, Task
from .module import *

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


'''
def PIE(src):
    alpha = 1000
    beta  = 0.01
    lam   = 10
    gama  = 0.1
    eta1  = 0.1
    eta2  = 0.1
    im    = src[:, :, ::-1]
    hsv   = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    v     = hsv[:, :, 2].astype(np.float32)
    v     = optimizAlgo(v, alpha, beta, lam, gama, eta1, eta2)
    hsv[:, :, 2] = v
    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    im = np.minimum(255, np.maximum(0, im))
    im = im[:, :, ::-1]
    return im
'''


@MODELS.register(name="pie", arch="pie")
class PIE(nn.Module, ModelMixin):
    """PIE model for low-light image enhancement.
    
    References:
        - Paper: "A Probabilistic Method for Image Enhancement With Simultaneous
          Illumination and Reflectance Estimation," IEEE TIP 2015.
        - Code: https://github.com/DavidQiuChao/PIE
    """
    
    arch     : str          = "pie"
    name     : str          = "pie"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.TRADITIONAL]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()

    def __init__(self):
        super().__init__()
        self.alpha = 1000
        self.beta  = 0.01
        self.lam   = 10
        self.gama  = 0.1
        self.eta1  = 0.1
        self.eta2  = 0.1
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
        v   = hsv[:, :, 2].astype(np.float32)
        v   = optimizAlgo(v, self.alpha, self.beta, self.lam, self.gama, self.eta1, self.eta2)
        hsv[:, :, 2] = v
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        rgb = np.minimum(255, np.maximum(0, rgb))
        return rgb
