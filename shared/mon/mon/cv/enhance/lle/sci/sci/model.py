#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements SCI model for low-light image enhancement.

References:
    - Paper: "Toward Fast, Flexible, and Robust Low-Light Image Enhancement,"
      CVPR 2022.
    - Code: https://github.com/vis-opt-group/SCI
"""

__all__ = [
    "SCI",
]

from typing import Any

import box

from mon import nn
from mon.core import MLType, MODELS, Path, ROOT_DIR, Task
from .module import Finetunemodel

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


@MODELS.register(name="sci", arch="sci")
class SCI(Finetunemodel, nn.ModelMixin):
    """SCI model for low-light image enhancement.
    
    References:
        - Paper: "Toward Fast, Flexible, and Robust Low-Light Image Enhancement,"
          CVPR 2022.
        - Code: https://github.com/vis-opt-group/SCI
    """
    
    arch     : str          = "sci"
    name     : str          = "sci"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box({
        "darkface": {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/sci/sci/darkface/sci_darkface.pt",
            "num_classes": None,
        },
        "fivek"  : {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/sci/sci/fiveke/sci_fiveke.pt",
            "num_classes": None,
        },
        "lolv1"  : {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/sci/sci/lolv1/sci_lolv1.pt",
            "num_classes": None,
        },
    })
    
    def __init__(self, weights: Any = None):
        _, path, _ = self.parse_weights(weights)
        super().__init__(weights=path)
