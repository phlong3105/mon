#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements DEIMv2 model for object detection.

References:
    - Paper: "Real-Time Object Detection Meets DINOv3," arXiv 2025.
    - Code: https://github.com/Intellindust-AI-Lab/DEIMv2
"""

__all__ = [
    "DEIMv2",
]

from typing import Any

import box
import torch

from mon import nn
from mon.core import MLType, MODELS, Path, Task
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from .core import YAMLConfig

# import os
# import sys

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


class DEIMv2(nn.Module, nn.ModelMetadataMixin):
    """DEIMv2 model for object detection.
    
    References:
        - Paper: "Real-Time Object Detection Meets DINOv3," arXiv 2025.
        - Code: https://github.com/Intellindust-AI-Lab/DEIMv2
    """
    
    _arch     : str          = "deimv2"
    _name     : str          = "deimv2"
    _tasks    : list[Task]   = [Task.DETECT]
    _mltypes  : list[MLType] = [MLType.SUPERVISED]
    _model_dir: Path         = root_dir
    _zoo      : dict         = box.Box()
    
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
    

MODELS.register(variant="deimv2_dinov3_s", name="deimv2", module=DEIMv2)
MODELS.register(variant="deimv2_dinov3_m", name="deimv2", module=DEIMv2)
MODELS.register(variant="deimv2_dinov3_l", name="deimv2", module=DEIMv2)
MODELS.register(variant="deimv2_dinov3_x", name="deimv2", module=DEIMv2)
