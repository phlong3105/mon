#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DepthAnythingV2.

This module implements a wrapper for DepthAnythingV2 models. It provides a
simple interface for loading pre-trained models and performing inference.

Notice:
This is the first example of using a third-party package in the `mon` package.
Why? Because reimplementing all of :obj:`depth_anything_v2` is a lot of work and
is not a smart idea.
"""

from __future__ import annotations

__all__ = [
    "DepthAnythingV2_ViTB",
    "DepthAnythingV2_ViTL",
    "DepthAnythingV2_ViTS",
    "build_depth_anything_v2",
]

import sys
from abc import ABC
from typing import Any, Literal

from mon import core, nn
from mon.globals import MODELS, Scheme
from mon.vision.depth import base
from mon_ss.globals import ZOO_DIR

console       = core.console
error_console = core.error_console

try:
    import depth_anything_v2
    from depth_anything_v2 import dpt
    depth_anything_v2s_available = True
except ImportError:
    depth_anything_v2_available = False
    error_console.log("The package 'depth_anything_v2' has not been installed.")
    # sys.exit(1)  # Exit and raise error
    sys.exit(0)  # Exit without error


# region Model

class DepthAnythingV2(nn.ExtraModel, base.DepthEstimationModel, ABC):
    """This class implements a wrapper for :obj:`DepthAnythingV2` models
    defined in :obj:`mon_extra.vision.depth.depth_anything_v2`.
    """
    
    arch   : str          = "depth_anything_v2"
    schemes: list[Scheme] = [Scheme.INFERENCE_ONLY]
    zoo    : dict         = {}
    
    def init_weights(self, m: nn.Module):
        pass
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        x = datapoint.get("image")
        y = self.model(x)
        y = (y - y.min()) / (y.max() - y.min())  # Normalize the depth map in the range [0, 1].
        y = y.unsqueeze(1)
        return {"depth": y}


@MODELS.register(name="depth_anything_v2_vits", arch="depth_anything_v2")
class DepthAnythingV2_ViTS(DepthAnythingV2):
    
    zoo: dict = {
        "da_2k": {
            "url"        : None,
            "path"       : ZOO_DIR / "vision/depth/depth_anything_v2/depth_anything_v2_vits/da_2k/depth_anything_v2_vits_da_2k.pth",
            "num_classes": None,
        },
    }
    
    def __init__(
        self,
        name       : str = "depth_anything_v2_vits",
        in_channels: int = 3,
        weights    : Any = "da_2k",
        *args, **kwargs
    ):
        super().__init__(
            name        = name,
            in_channels = in_channels,
            weights     = weights,
            *args, **kwargs
        )
        if isinstance(self.weights, dict):
            in_channels = self.weights.get("in_channels", in_channels)
        self.in_channels = in_channels or self.in_channels
        
        self.model = dpt.DepthAnythingV2(
            encoder      = "vits",
            features     = 64,
            out_channels = [48, 96, 192, 384],
        )
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="depth_anything_v2_vitb", arch="depth_anything_v2")
class DepthAnythingV2_ViTB(DepthAnythingV2):
    
    zoo: dict = {
        "da_2k": {
            "url"        : None,
            "path"       : ZOO_DIR / "vision/depth/depth_anything_v2/depth_anything_v2_vitb/da_2k/depth_anything_v2_vitb_da_2k.pth",
            "num_classes": None,
        },
    }

    def __init__(
        self,
        name       : str = "depth_anything_v2_vits",
        in_channels: int = 3,
        weights    : Any = "da_2k",
        *args, **kwargs
    ):
        super().__init__(
            name        = name,
            in_channels = in_channels,
            weights     = weights,
            *args, **kwargs
        )
        if isinstance(self.weights, dict):
            in_channels = self.weights.get("in_channels", in_channels)
        self.in_channels = in_channels or self.in_channels
        
        self.model = dpt.DepthAnythingV2(
            encoder      = "vitb",
            features     = 128,
            out_channels = [96, 192, 384, 768],
        )
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
            

@MODELS.register(name="depth_anything_v2_vitl", arch="depth_anything_v2")
class DepthAnythingV2_ViTL(DepthAnythingV2):
    
    zoo: dict = {
        "da_2k": {
            "url"        : None,
            "path"       : ZOO_DIR / "vision/depth/depth_anything_v2/depth_anything_v2_vitl/da_2k/depth_anything_v2_vitl_da_2k.pth",
            "num_classes": None,
        },
    }

    def __init__(
        self,
        name       : str = "depth_anything_v2_vits",
        in_channels: int = 3,
        weights    : Any = "da_2k",
        *args, **kwargs
    ):
        super().__init__(
            name        = name,
            in_channels = in_channels,
            weights     = weights,
            *args, **kwargs
        )
        if isinstance(self.weights, dict):
            in_channels = self.weights.get("in_channels", in_channels)
        self.in_channels = in_channels or self.in_channels
        
        self.model = dpt.DepthAnythingV2(
            encoder      = "vitl",
            features     = 256,
            out_channels = [256, 512, 1024, 1024],
        )
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


def build_depth_anything_v2(
    encoder     : Literal["vits", "vitb", "vitl", "vitg"] = "vits",
    in_channels : int = 3,
    weights     : Any = "da_2k",
    *args, **kwargs
) -> DepthAnythingV2:
    if encoder not in ["vits", "vitb", "vitl", "vitg"]:
        raise ValueError(f"`encoder` must be one of ['vits', 'vitb', 'vitl', 'vitg'], but got {encoder}.")
    if encoder == "vits":
        return DepthAnythingV2_ViTS(in_channels=in_channels, weights=weights, *args, **kwargs)
    elif encoder == "vitb":
        return DepthAnythingV2_ViTB(in_channels=in_channels, weights=weights, *args, **kwargs)
    elif encoder == "vitl":
        return DepthAnythingV2_ViTL(in_channels=in_channels, weights=weights, *args, **kwargs)
    
# endregion
