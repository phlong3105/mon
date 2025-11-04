#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements LFormer model for low-light image enhancement.

References:
    - Paper: "Interpretable Unsupervised Joint Denoising and Enhancement for
      Real-World low-light Scenarios," ICLR 2025.
    - Code: https://github.com/huaqlili/unsupervised-light-enhance-ICLR2025
"""

__all__ = [
    "LFormer",
]

from typing import Any

import box

import mon.nn as nn
from mon.constants import MODELS, ROOT_DIR
from mon.core import MLType, Path, Task
from .net.lformer import net

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="lformer", arch="lformer")
class LFormer(net, nn.ModelMixin):
    """LFormer model for low-light image enhancement.
    
    References:
        - Paper: "Interpretable Unsupervised Joint Denoising and Enhancement for
          Real-World low-light Scenarios," ICLR 2025.
        - Code: https://github.com/huaqlili/unsupervised-light-enhance-ICLR2025
    """
    
    arch     : str          = "lformer"
    name     : str          = "lformer"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box({
        "lolv1"    : {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/lformer/lformer/lolv1/lformer_lolv1.pth",
            "num_classes": None,
        },
        "lolv2real": {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/lformer/lformer/lolv2real/lformer_lolv2real.pth",
            "num_classes": None,
        },
        "sice"     : {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/lformer/lformer/sice/lformer_sice.pth",
            "num_classes": None,
        },
    })
    
    def __init__(self, weights: Any = None):
        super().__init__()
        # Load weights
        self.load_weights(weights)
