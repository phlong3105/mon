#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements a wrapper for DepthAnythingV2 models. It provides a
simple interface for loading pre-trained models and performing inference.

Notice:
This is the first example of using a third-party package in the `mon` package.
Why? Because reimplementing all of :mod:`depth_anything_v2` is a lot of work and
is not a smart idea.
"""

from __future__ import annotations

__all__ = [
    "DepthAnythingV2",
    "DepthAnythingV2_ViTB",
    "DepthAnythingV2_ViTL",
    "DepthAnythingV2_ViTS",
    "build_depth_anything_v2",
]

import sys
from abc import ABC
from typing import Any, Literal

import torch
from mon import core, nn
from mon.core import _callable
from mon.globals import MODELS, Scheme, Task, ZOO_DIR

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

class DepthAnythingV2(nn.ExtraModel, ABC):
    """This class implements a wrapper for :class:`DepthAnythingV2` models
    defined in :mod:`mon_extra.vision.depth.depth_anything_v2`.
    
    See Also: :class:`mon.nn.model.ExtraModel`
    """
    
    arch   : str          = "depth_anything_v2"
    tasks  : list[Task]   = [Task.DEPTH]
    schemes: list[Scheme] = [Scheme.INFERENCE_ONLY]
    zoo    : dict = {}
    zoo_dir: core.Path    = ZOO_DIR / "vision" / "depth"
    
    def __init__(
        self,
        name        : str       = "depth_anything_v2",
        in_channels : int       = 3,
        encoder     : Literal["vits", "vitb", "vitl", "vitg"] = "vits",
        features    : int       = 64,
        out_channels: list[int] = [48, 96, 192, 384],
        weights     : Any       = "da_2k",
        *args, **kwargs
    ):
        super().__init__(
            name        = name,
            in_channels = in_channels,
            weights     = weights,
            *args, **kwargs
        )
        self.encoder      = encoder
        self.features     = features
        self.out_channels = out_channels
        self.model = dpt.DepthAnythingV2(
            encoder      = self.encoder,
            features     = self.features,
            out_channels = self.out_channels,
        )
        
        # Load weights
        if self.weights:
            # print(self.weights)
            self.load_weights()
        else:
            self.apply(self.init_weights)
        
    def init_weights(self, model: nn.Module):
        pass
    
    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> torch.Tensor:
        x = input
        y = self.model(x)
        y = y.unsqueeze(1)
        # Normalize the depth map in the range [0, 1].
        y = (y - y.min()) / (y.max() - y.min())
        return y


@MODELS.register(name="depth_anything_v2_vits", arch="depth_anything_v2")
class DepthAnythingV2_ViTS(DepthAnythingV2):
    """
    
    See Also: :class:`DepthAnythingV2`
    """
    
    zoo: dict = {
        "da_2k": {
            "url"        : None,
            "path"       : "depth_anything_v2/depth_anything_v2_vits/da_2k/depth_anything_v2_vits_da_2k.pth",
            "num_classes": None,
            "map": {},
        },
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            name         = "depth_anything_v2_vits",
            encoder      = "vits",
            features     = 64,
            out_channels = [48, 96, 192, 384],
            *args, **kwargs
        )


@MODELS.register(name="depth_anything_v2_vitb", arch="depth_anything_v2")
class DepthAnythingV2_ViTB(DepthAnythingV2):
    """
    
    See Also: :class:`DepthAnythingV2`
    """
    
    zoo: dict = {
        "da_2k": {
            "url"        : None,
            "path"       : "depth_anything_v2/depth_anything_v2_vitb/da_2k/depth_anything_v2_vitb_da_2k.pth",
            "num_classes": None,
            "map": {},
        },
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            name         = "depth_anything_v2_vitb",
            encoder      = "vitb",
            features     = 128,
            out_channels = [96, 192, 384, 768],
            *args, **kwargs
        )


@MODELS.register(name="depth_anything_v2_vitl", arch="depth_anything_v2")
class DepthAnythingV2_ViTL(DepthAnythingV2):
    """
    
    See Also: :class:`DepthAnythingV2`
    """
    
    zoo: dict = {
        "da_2k": {
            "url"        : None,
            "path"       : "depth_anything_v2/depth_anything_v2_vitl/da_2k/depth_anything_v2_vitl_da_2k.pth",
            "num_classes": None,
            "map": {},
        },
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            name         = "depth_anything_v2_vitl",
            encoder      = "vitl",
            features     = 256,
            out_channels = [256, 512, 1024, 1024],
            *args, **kwargs
        )


def build_depth_anything_v2(
    encoder     : Literal["vits", "vitb", "vitl", "vitg"] = "vits",
    in_channels : int = 3,
    weights     : Any = "da_2k",
    *args, **kwargs
) -> DepthAnythingV2:
    if encoder not in ["vits", "vitb", "vitl", "vitg"]:
        raise ValueError(f":param:`encoder` must be one of ['vits', 'vitb', 'vitl', 'vitg'], but got {encoder}.")
    if encoder == "vits":
        return DepthAnythingV2_ViTS(in_channels = in_channels, weights = weights, *args, **kwargs)
    elif encoder == "vitb":
        return DepthAnythingV2_ViTB(in_channels = in_channels, weights = weights, *args, **kwargs)
    elif encoder == "vitl":
        return DepthAnythingV2_ViTL(in_channels = in_channels, weights = weights, *args, **kwargs)
    
# endregion
