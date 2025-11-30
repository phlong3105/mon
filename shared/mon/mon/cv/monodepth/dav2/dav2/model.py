#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements DAV2 model for monocular depth estimation.

References:
    - Paper: "Depth Anything V2. A More Capable Foundation Model for Monocular
      Depth Estimation," NeurIPS 2024.
    - Code: https://github.com/DepthAnything/Depth-Anything-V2
"""

__all__ = [
    "DAV2",
    "DAV2_ViTS",
    "DAV2_ViTB",
    "DAV2_ViTL",
]

from typing import Any

import torch

from mon import nn
from mon.core import MLType, MODELS, Path, ROOT_DIR, Task
from .dpt import DepthAnythingV2

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


class DAV2(DepthAnythingV2, nn.ModelMixin):
    """DAV2 model for monocular depth estimation.

    References:
        - Paper: "Depth Anything V2. A More Capable Foundation Model for Monocular
          Depth Estimation," NeurIPS 2024.
        - https://github.com/DepthAnything/Depth-Anything-V2
    """
    
    _arch     : str          = "dav2"
    _name     : str          = "dav2"
    _tasks    : list[Task]   = [Task.MONODEPTH]
    _mltypes  : list[MLType] = [MLType.SUPERVISED]
    _model_dir: Path         = root_dir
    _zoo      : dict         = {}
    

@MODELS.register(name="dav2_vits", arch="dav2")
class DAV2_ViTS(DAV2):
    
    _name: str  = "dav2_vits"
    _zoo : dict = {
        "pretrained": {
            "path": ROOT_DIR / "zoo/cv/monodepth/dav2/dav2_vits/pretrained/dav2_vits.pth",
        },
    }
    
    def __init__(self, weights: Any = "pretrained"):
        super().__init__(
            encoder      = "vits",
            features     = 64,
            out_channels = [48, 96, 192, 384],
        )
        self.load_state_dict(torch.load(str(weights), weights_only=True))


@MODELS.register(name="dav2_vitb", arch="dav2")
class DAV2_ViTB(DAV2):
    
    _name: str = "dav2_vitb"
    _zoo : dict = {
        "pretrained": {
            "path": ROOT_DIR / "zoo/cv/monodepth/dav2/dav2_vitb/pretrained/dav2_vitb.pth",
        },
    }

    def __init__(self, weights: Any = "pretrained"):
        super().__init__(
            encoder      = "vitb",
            features     = 128,
            out_channels = [96, 192, 384, 768],
        )
        self.load_state_dict(torch.load(str(weights), weights_only=True))
        

@MODELS.register(name="dav2_vitl", arch="dav2")
class DAV2_ViTL(DAV2):
    
    _name: str = "dav2_vitl"
    _zoo : dict = {
        "pretrained": {
            "path": ROOT_DIR / "zoo/cv/monodepth/dav2/dav2_vitl/pretrained/dav2_vitl.pth",
        },
    }

    def __init__(self, weights: Any = "pretrained"):
        super().__init__(
            encoder      = "vitl",
            features     = 256,
            out_channels = [256, 512, 1024, 1024],
        )
        self.load_state_dict(torch.load(str(weights), weights_only=True))
