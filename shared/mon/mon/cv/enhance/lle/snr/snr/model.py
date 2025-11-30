#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements SNR model for low-light image enhancement.

References:
    - Paper: "SNR-aware Low-Light Image Enhancement," CVPR 2022.
    - Code: https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance
"""

__all__ = [
    "SNR",
]

import box

from mon import nn
from mon.core import MLType, MODELS, Path, Task
from .models.Video_base_model4_m import VideoBaseModel

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="snr", arch="snr")
class SNR(VideoBaseModel, nn.ModelMixin):
    """SNR model for low-light image enhancement.
    
    References:
        - Paper: "SNR-aware Low-Light Image Enhancement," CVPR 2022.
        - Code: https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance
    """
    
    _arch     : str          = "snr"
    _name     : str          = "snr"
    _tasks    : list[Task]   = [Task.LLE]
    _mltypes  : list[MLType] = [MLType.SUPERVISED]
    _model_dir: Path         = root_dir
    _zoo      : dict         = box.Box()
