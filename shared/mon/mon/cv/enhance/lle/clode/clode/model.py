#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements CLODE model for low-light image enhancement.

References:
    - Paper: "Continuous Exposure Learning for Low-light Image Enhancement using
      Neural ODEs," ICLR 2025.
    - Code: https://github.com/dgjung0220/CLODE
"""

__all__ = [
    "CLODE",
]

from typing import Any

import box

from mon.constants import MODELS, ROOT_DIR
from mon.core import MLType, ModelMixin, Path, Task
from .network import NODE

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="clode", arch="clode")
class CLODE(NODE, ModelMixin):
    """CLODE model for low-light image enhancement.

    References:
        - Paper: "Continuous Exposure Learning for Low-light Image Enhancement using
          Neural ODEs," ICLR 2025.
        - Code: https://github.com/dgjung0220/CLODE
    """
    
    arch     : str          = "clode"
    name     : str          = "clode"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box({
        "lolv1"    : {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/clode/clode/lolv1/clode_lolv1.pth",
            "num_classes": None,
        },
        "sice"     : {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/clode/clode/sice/clode_sice.pth",
            "num_classes": None,
        },
        "universal": {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/clode/clode/universal/clode_universal.pth",
            "num_classes": None,
        },
    })
    
    def __init__(self, weights: Any = None, *args, **kwarg):
        super().__init__(*args, **kwarg)
        # Load weights
        self.load_weights(weights, strict=False)
