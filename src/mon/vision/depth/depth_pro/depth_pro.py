#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DepthPro

This module implements a wrapper for DepthPro models. It provides a simple
interface for loading pre-trained models and performing inference.
"""

from __future__ import annotations

__all__ = [
    "DepthPro",
]

import sys
from typing import Any

import torch

from mon import core, nn
from mon.globals import MODELS, Scheme, ZOO_DIR
from mon.vision.depth import base

console       = core.console
error_console = core.error_console
current_file  = core.Path(__file__).absolute()
current_dir   = current_file.parents[0]

try:
    import depth_pro
    depth_pro_available = True
except ImportError:
    depth_pro_available = False
    error_console.log("The package 'depth_pro' has not been installed.")
    # sys.exit(1)  # Exit and raise error
    sys.exit(0)  # Exit without error


# region Model

@MODELS.register(name="depth_pro", arch="depth_pro")
class DepthPro(nn.ExtraModel, base.DepthEstimationModel):
    """This class implements a wrapper for :obj:`DepthAnythingV2` models
    defined in :obj:`mon_extra.vision.depth.depth_anything_v2`.
    """
    
    model_dir: core.Path    = current_dir
    arch     : str          = "depth_pro"
    schemes  : list[Scheme] = [Scheme.INFERENCE]
    zoo      : dict         = {
        "pretrained": {
            "url"        : None,
            "path"       : ZOO_DIR / "vision/depth/depth_pro/depth_pro/pretrained/depth_pro.pt",
            "num_classes": None,
        },
    }
    
    def __init__(
        self,
        name                : str  = "depth_pro",
        in_channels         : int  = 3,
        patch_encoder_preset: str  = "dinov2l16_384",
        image_encoder_preset: str  = "dinov2l16_384",
        decoder_features    : int  = 256,
        use_fov_head        : bool = True,
        fov_encoder_preset  : str  = "dinov2l16_384",
        weights             : Any  = "pretrained",
        *args, **kwargs
    ):
        super().__init__(
            name        = name,
            in_channels = in_channels,
            weights     = weights,
            *args, **kwargs
        )
        self.in_channels                 = in_channels or self.in_channels
        self.config                      = depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT
        self.config.patch_encoder_preset = patch_encoder_preset
        self.config.image_encoder_preset = image_encoder_preset
        self.config.decoder_features     = decoder_features
        self.config.use_fov_head         = use_fov_head
        self.config.fov_encoder_preset   = fov_encoder_preset
        self.config.checkpoint_uri       = self.weights
        
        # Load model and preprocessing transform
        self.model, self.transform = depth_pro.create_model_and_transforms(config=self.config)
        self.model.eval()
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
    
    def init_weights(self, m: nn.Module):
        pass
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        x       = datapoint.get("image")
        f_px    = datapoint.get("f_px")
        outputs = self.model.infer(x, f_px=f_px)
        return {
            "focallength_px": outputs["focallength_px"],
            "depth"         : outputs["depth"],
        }
    
    def infer(self, datapoint : dict, *args, **kwargs) -> dict:
        # Pre-processing
        self.assert_datapoint(datapoint)
        meta               = datapoint.get("meta")
        image_path         = core.Path(meta["path"])
        image, _, f_px     = depth_pro.load_rgb(str(image_path))
        image              = self.transform(image)
        datapoint["image"] = image
        datapoint["f_px"]  = f_px
        for k, v in datapoint.items():
            if isinstance(v, torch.Tensor):
                datapoint[k] = v.to(self.device)
        
        # Forward
        timer = core.Timer()
        timer.tick()
        outputs = self.forward(datapoint, *args, **kwargs)
        timer.tock()
        self.assert_outputs(outputs)
        
        # Return
        outputs["time"] = timer.avg_time
        return outputs
    
# endregion
