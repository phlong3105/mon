#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements DEIM model for object detection.

References:
    - Paper: "DEIM: DETR with Improved Matching for Fast convergence," CVPR 2025.
    - Code: https://github.com/ShihuaHuang95/DEIM
"""

__all__ = [
    "DEIM",
]

from typing import Any

import box
import torch

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, nn, Path, Task
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from .core import YAMLConfig

# import os
# import sys

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


class DEIM(nn.Module, ModelMixin):
    """DEIM model for object detection.
    
    References:
        - Paper: "DEIM: DETR with Improved Matching for Fast convergence," CVPR 2025.
        - Code: https://github.com/ShihuaHuang95/DEIM
    """
    
    arch     : str          = "deim"
    name     : str          = "deim"
    tasks    : list[Task]   = [Task.DETECT]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
    
    def __init__(
        self,
        cfg        : str,
        weights    : Any,
        root       : Path,
        device     : torch.device  = torch.device("cpu"),
        seed       : int           = 0,
        updated_cfg: dict          = None,
        export_postprocessor: bool = True
    ):
        super().__init__()
        cfg_path     = root_dir / "option" / cfg
        updated_cfg  = updated_cfg or {}
        updated_cfg |= {"resume": str(weights)} if weights else {}
        updated_cfg |= {
            "device": device,
            "seed"  : seed,
        }
        cfg = YAMLConfig(cfg_path=str(cfg_path), root=str(root), **updated_cfg)
    
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    
        if weights:
            checkpoint = torch.load(weights, map_location="cpu")
            if "ema" in checkpoint:
                state = checkpoint["ema"]["module"]
            else:
                state = checkpoint["model"]
        else:
            raise AttributeError("Only support resume to load model.state_dict by now.")
        cfg.model.load_state_dict(state)
        
        self.model = cfg.model.deploy()
        if export_postprocessor:
            self.postprocessor = cfg.postprocessor.deploy()
        else:
            self.postprocessor = None
    
    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        if self.postprocessor is not None:
            outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs
    

MODELS.register(name="deim_dfine_n", arch="deim", module=DEIM)
MODELS.register(name="deim_dfine_s", arch="deim", module=DEIM)
MODELS.register(name="deim_dfine_m", arch="deim", module=DEIM)
MODELS.register(name="deim_dfine_l", arch="deim", module=DEIM)
MODELS.register(name="deim_dfine_x", arch="deim", module=DEIM)
