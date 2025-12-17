#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements RUAS model for low-light image enhancement.

References:
    - Paper: "Retinex-inspired Unrolling with Cooperative Prior Architecture
      Search for Low-light Image Enhancement," 2021.
    - Code: https://github.com/KarelZhang/RUAS
"""

__all__ = [
    "RUAS",
]

from typing import Any

import box

from mon import nn
from mon.core import MLType, MODELS, Path, ROOT_DIR, Task
from .module import Network

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="ruas", arch="ruas")
class RUAS(Network, nn.ModelMetadataMixin):
    """RUAS model for low-light image enhancement.
    
    References:
        - Paper: "Retinex-inspired Unrolling with Cooperative Prior Architecture
          Search for Low-light Image Enhancement," 2021.
        - Code: https://github.com/KarelZhang/RUAS
    """
    
    _arch     : str          = "ruas"
    _name     : str          = "ruas"
    _tasks    : list[Task]   = [Task.LLE]
    _mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    _model_dir: Path         = root_dir
    _zoo      : dict         = box.Box({
        "darkface": {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/ruas/ruas/darkface/ruas_darkface.pt",
            "num_classes": None,
        },
        "lolv1"   : {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/ruas/ruas/lolv1/ruas_lolv1.pt",
            "num_classes": None,
        },
        "upe"     : {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/ruas/ruas/upe/ruas_upe.pt",
            "num_classes": None,
        },
    })
    
    def __init__(self, weights: Any = None):
        super().__init__()
        # Load weights
        self.load_weights(weights)
