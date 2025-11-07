#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements HVI-CIDNet model for low-light image enhancement.

References:
    - Paper: "HVI: A New color space for Low-light Image Enhancement," CVPR 2025.
    - Code: https://github.com/Fediory/HVI-CIDNet
"""

__all__ = [
    "HVI_CIDNet",
]

import box

from mon import nn
from mon.core import MLType, MODELS, Path, ROOT_DIR, Task
from .net.CIDNet import CIDNet

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="hvi_cidnet", arch="hvi_cidnet")
class HVI_CIDNet(CIDNet, nn.ModelMixin):
    """HVI-CIDNet model for low-light image enhancement.
    
    References:
        - Paper: "HVI: A New color space for Low-light Image Enhancement," CVPR 2025.
        - Code: https://github.com/Fediory/HVI-CIDNet
    """
    
    arch     : str          = "hvi_cidnet"
    name     : str          = "hvi_cidnet"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box({
        "lolblur"  : {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/hvi_cidnet/hvi_cidnet/lolblur/hvi_cidnet_lolblur.pth",
            "num_classes": None,
        },
        "lolv1"    : {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/hvi_cidnet/hvi_cidnet/lolv1/hvi_cidnet_lolv1.pth",
            "num_classes": None,
        },
        "lolv2real": {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/hvi_cidnet/hvi_cidnet/lolv2real/hvi_cidnet_lolv2real.pth",
            "num_classes": None,
        },
        "lolv2syn" : {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/hvi_cidnet/hvi_cidnet/lolv2syn/hvi_cidnet_lolv2syn.pth",
            "num_classes": None,
        },
        "sice"     : {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/hvi_cidnet/hvi_cidnet/sice/hvi_cidnet_sice.pth",
            "num_classes": None,
        },
        "sidsony"  : {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/hvi_cidnet/hvi_cidnet/sidsony/hvi_cidnet_sidsony.pth",
            "num_classes": None,
        },
    })
    
    def __init__(self, weights: any = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load weights
        self.load_weights(weights)
