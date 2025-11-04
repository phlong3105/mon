#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements LL-UNet++ model for low-light image enhancement.

References:
    - Paper: "LL-UNet++:UNet++ Based Nested Skip Connections Network for Low-Light
      Image Enhancement," TCI 2024.
    - Code: https://github.com/xiwang-online/LLUnetPlusPlus
"""

__all__ = [
    "LLUnetPP",
]

from typing import Any

import box

from mon.constants import MODELS, ROOT_DIR
from mon.core import MLType, ModelMixin, Path, Task
from .module import NestedUNet

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="llunet++", arch="llunet++")
class LLUnetPP(NestedUNet, ModelMixin):
    """LL-UNet++ model for low-light image enhancement.
    
    References:
        - Paper: "LL-UNet++:UNet++ Based Nested Skip Connections Network for
          Low-Light Image Enhancement," TCI 2024.
        - Code: https://github.com/xiwang-online/LLUnetPlusPlus
    """
    
    arch     : str          = "llunet++"
    name     : str          = "llunet++"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box({
        "lolv1"    : {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/llunet++/llunet++/lolv1/llunet++_lolv1.pt",
            "num_classes": None,
        },
        "lolv2real": {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/llunet++/llunet++/lolv2real/llunet++_lolv2real.pt",
            "num_classes": None,
        },
        "lolv2syn" : {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/llunet++/llunet++/lolv2syn/llunet++_lolv2syn.pt",
            "num_classes": None,
        },
    })
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, weights: Any = None):
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        # Load weights
        self.load_weights(weights)
