#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements DAAC model for depth estimation.

References:
    - Paper: "Depth Anything At Any Condition," arXiv 2025.
    - Code: https://github.com/HVision-NKU/DepthAnythingAC
"""

__all__ = [
    "DAAC",
    "DAV2_ViTS",
]

from typing import Any

import torch

from mon import nn
from mon.core import MLType, MODELS, Path, ROOT_DIR, Task
from .depth_anything.dpt import DepthAnything_AC

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


class DAAC(DepthAnything_AC, nn.ModelMixin):
    """DAAC model for depth estimation.
    
    References:
        - Paper: "Depth Anything At Any Condition," arXiv 2025.
        - Code: https://github.com/HVision-NKU/DepthAnythingAC
    """
    
    arch     : str          = "daac"
    name     : str          = "daac"
    tasks    : list[Task]   = [Task.MONODEPTH]
    mltypes  : list[MLType] = [MLType.ZERO_SHOT]
    model_dir: Path         = root_dir
    zoo      : dict         = {}


@MODELS.register(name="daac_vits", arch="daac")
class DAV2_ViTS(DAAC):
    
    name: str  = "daac_vits"
    zoo : dict = {
        "pretrained": {
            "path": ROOT_DIR / "zoo/cv/monodepth/daac/daac_vits/pretrained/daac_vits.pth",
        },
    }
    
    def __init__(self, weights: Any = "pretrained"):
        super().__init__(
            config = {
                "encoder"        : "vits",
                "features"       : 64,
                "out_channels"   : [48, 96, 192, 384],
                "dino_pretrained": ROOT_DIR / "zoo/cv/monodepth/daac/daac_vits/pretrained/dinov2_vits14_pretrain.pth",
                "version"        : "v2",
            }
        )
        self.load_state_dict(torch.load(str(weights), weights_only=True))
