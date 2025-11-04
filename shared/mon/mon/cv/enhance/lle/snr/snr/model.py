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

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, Path, Task
from .models.Video_base_model4_m import VideoBaseModel

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="snr", arch="snr")
class SNR(VideoBaseModel, ModelMixin):
    """SNR model for low-light image enhancement.
    
    References:
        - Paper: "SNR-aware Low-Light Image Enhancement," CVPR 2022.
        - Code: https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance
    """
    
    arch     : str          = "snr"
    name     : str          = "snr"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
