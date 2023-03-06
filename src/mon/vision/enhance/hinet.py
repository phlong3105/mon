#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements HINet (Half-Instance Normalization Network) models."""

from __future__ import annotations

__all__ = [
    "HINet",
]

from typing import Any

from mon.coreml import model as mmodel
from mon.foundation import pathlib
from mon.globals import MODELS
from mon.vision.enhance import base

_current_dir = pathlib.Path(__file__).absolute().parent


# region Model

@MODELS.register(name="hinet")
class HINet(base.ImageEnhancementModel):
    """Half-Instance Normalization Network.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "hinet-deblur-gopro"     : {
            "name"       : "gopro",
            "path"       : "",
            "file_name"  : "hinet-deblur-gopro.pth",
            "num_classes": None,
        },
        "hinet-deblur-reds"      : {
            "name"       : "reds",
            "path"       : "",
            "file_name"  : "hinet-deblur-reds.pth",
            "num_classes": None,
        },
        "hinet-denoise-sidd-x1.0": {
            "name"       : "sidd",
            "path"       : "",
            "file_name"  : "hinet-denoise-sidd-x1.0.pth",
            "num_classes": None,
        },
        "hinet-denoise-sidd-x0.5": {
            "name"       : "sidd",
            "path"       : "",
            "file_name"  : "hinet-denoise-sidd-x0.5.pth",
            "num_classes": None,
        },
        "hinet-derain-rain13k"   : {
            "name"       : "rain13k",
            "path"       : "",
            "file_name"  : "hinet-derain-rain13k.pth",
            "num_classes": None,
        },
    }
    map_weights = {
        "backbone": {
            "1.weight"               : "conv_01.weight",
            "1.bias"                 : "conv_01.bias",
            "2.conv1.weight"         : "down_path_1.0.conv_1.weight",
            "2.conv1.bias"           : "down_path_1.0.conv_1.bias",
            "2.conv2.weight"         : "down_path_1.0.conv_2.weight",
            "2.conv2.bias"           : "down_path_1.0.conv_2.bias",
            "2.identity.weight"      : "down_path_1.0.identity.weight",
            "2.identity.bias"        : "down_path_1.0.identity.bias",
            "2.downsample.weight"    : "down_path_1.0.downsample.weight",
            "4.conv1.weight"         : "down_path_1.1.conv_1.weight",
            "4.conv1.bias"           : "down_path_1.1.conv_1.bias",
            "4.conv2.weight"         : "down_path_1.1.conv_2.weight",
            "4.conv2.bias"           : "down_path_1.1.conv_2.bias",
            "4.identity.weight"      : "down_path_1.1.identity.weight",
            "4.identity.bias"        : "down_path_1.1.identity.bias",
            "4.downsample.weight"    : "down_path_1.1.downsample.weight",
            "6.conv1.weight"         : "down_path_1.2.conv_1.weight",
            "6.conv1.bias"           : "down_path_1.2.conv_1.bias",
            "6.conv2.weight"         : "down_path_1.2.conv_2.weight",
            "6.conv2.bias"           : "down_path_1.2.conv_2.bias",
            "6.identity.weight"      : "down_path_1.2.identity.weight",
            "6.identity.bias"        : "down_path_1.2.identity.bias",
            "6.downsample.weight"    : "down_path_1.2.downsample.weight",
            "8.conv1.weight"         : "down_path_1.3.conv_1.weight",
            "8.conv1.bias"           : "down_path_1.3.conv_1.bias",
            "8.conv2.weight"         : "down_path_1.3.conv_2.weight",
            "8.conv2.bias"           : "down_path_1.3.conv_2.bias",
            "8.identity.weight"      : "down_path_1.3.identity.weight",
            "8.identity.bias"        : "down_path_1.3.identity.bias",
            "8.norm.weight"          : "down_path_1.3.norm.weight",
            "8.norm.bias"            : "down_path_1.3.norm.bias",
            "8.downsample.weight"    : "down_path_1.3.downsample.weight",
            "10.conv1.weight"        : "down_path_1.4.conv_1.weight",
            "10.conv1.bias"          : "down_path_1.4.conv_1.bias",
            "10.conv2.weight"        : "down_path_1.4.conv_2.weight",
            "10.conv2.bias"          : "down_path_1.4.conv_2.bias",
            "10.identity.weight"     : "down_path_1.4.identity.weight",
            "10.identity.bias"       : "down_path_1.4.identity.bias",
            "10.norm.weight"         : "down_path_1.4.norm.weight",
            "10.norm.bias"           : "down_path_1.4.norm.bias",
            "12.weight"              : "skip_conv_1.3.weight",
            "12.bias"                : "skip_conv_1.3.bias",
            "14.weight"              : "skip_conv_1.2.weight",
            "14.bias"                : "skip_conv_1.2.bias",
            "16.weight"              : "skip_conv_1.1.weight",
            "16.bias"                : "skip_conv_1.1.bias",
            "18.weight"              : "skip_conv_1.0.weight",
            "18.bias"                : "skip_conv_1.0.bias",
            "20.up.weight"           : "up_path_1.0.up.weight",
            "20.up.bias"             : "up_path_1.0.up.bias",
            "20.conv.conv1.weight"   : "up_path_1.0.conv_block.conv_1.weight",
            "20.conv.conv1.bias"     : "up_path_1.0.conv_block.conv_1.bias",
            "20.conv.conv2.weight"   : "up_path_1.0.conv_block.conv_2.weight",
            "20.conv.conv2.bias"     : "up_path_1.0.conv_block.conv_2.bias",
            "20.conv.identity.weight": "up_path_1.0.conv_block.identity.weight",
            "20.conv.identity.bias"  : "up_path_1.0.conv_block.identity.bias",
            "21.up.weight"           : "up_path_1.1.up.weight",
            "21.up.bias"             : "up_path_1.1.up.bias",
            "21.conv.conv1.weight"   : "up_path_1.1.conv_block.conv_1.weight",
            "21.conv.conv1.bias"     : "up_path_1.1.conv_block.conv_1.bias",
            "21.conv.conv2.weight"   : "up_path_1.1.conv_block.conv_2.weight",
            "21.conv.conv2.bias"     : "up_path_1.1.conv_block.conv_2.bias",
            "21.conv.identity.weight": "up_path_1.1.conv_block.identity.weight",
            "21.conv.identity.bias"  : "up_path_1.1.conv_block.identity.bias",
            "22.up.weight"           : "up_path_1.2.up.weight",
            "22.up.bias"             : "up_path_1.2.up.bias",
            "22.conv.conv1.weight"   : "up_path_1.2.conv_block.conv_1.weight",
            "22.conv.conv1.bias"     : "up_path_1.2.conv_block.conv_1.bias",
            "22.conv.conv2.weight"   : "up_path_1.2.conv_block.conv_2.weight",
            "22.conv.conv2.bias"     : "up_path_1.2.conv_block.conv_2.bias",
            "22.conv.identity.weight": "up_path_1.2.conv_block.identity.weight",
            "22.conv.identity.bias"  : "up_path_1.2.conv_block.identity.bias",
            "23.up.weight"           : "up_path_1.3.up.weight",
            "23.up.bias"             : "up_path_1.3.up.bias",
            "23.conv.conv1.weight"   : "up_path_1.3.conv_block.conv_1.weight",
            "23.conv.conv1.bias"     : "up_path_1.3.conv_block.conv_1.bias",
            "23.conv.conv2.weight"   : "up_path_1.3.conv_block.conv_2.weight",
            "23.conv.conv2.bias"     : "up_path_1.3.conv_block.conv_2.bias",
            "23.conv.identity.weight": "up_path_1.3.conv_block.identity.weight",
            "23.conv.identity.bias"  : "up_path_1.3.conv_block.identity.bias",
            "24.conv1.weight"        : "sam12.conv1.weight",
            "24.conv1.bias"          : "sam12.conv1.bias",
            "24.conv2.weight"        : "sam12.conv2.weight",
            "24.conv2.bias"          : "sam12.conv2.bias",
            "24.conv3.weight"        : "sam12.conv3.weight",
            "24.conv3.bias"          : "sam12.conv3.bias",
            "27.weight"              : "conv_02.weight",
            "27.bias"                : "conv_02.bias",
            "29.weight"              : "cat12.weight",
            "29.bias"                : "cat12.bias",
            "30.conv1.weight"        : "down_path_2.0.conv_1.weight",
            "30.conv1.bias"          : "down_path_2.0.conv_1.bias",
            "30.conv2.weight"        : "down_path_2.0.conv_2.weight",
            "30.conv2.bias"          : "down_path_2.0.conv_2.bias",
            "30.identity.weight"     : "down_path_2.0.identity.weight",
            "30.identity.bias"       : "down_path_2.0.identity.bias",
            "30.csff_enc.weight"     : "down_path_2.0.csff_enc.weight",
            "30.csff_enc.bias"       : "down_path_2.0.csff_enc.bias",
            "30.csff_dec.weight"     : "down_path_2.0.csff_dec.weight",
            "30.csff_dec.bias"       : "down_path_2.0.csff_dec.bias",
            "30.downsample.weight"   : "down_path_2.0.downsample.weight",
            "32.conv1.weight"        : "down_path_2.1.conv_1.weight",
            "32.conv1.bias"          : "down_path_2.1.conv_1.bias",
            "32.conv2.weight"        : "down_path_2.1.conv_2.weight",
            "32.conv2.bias"          : "down_path_2.1.conv_2.bias",
            "32.identity.weight"     : "down_path_2.1.identity.weight",
            "32.identity.bias"       : "down_path_2.1.identity.bias",
            "32.csff_enc.weight"     : "down_path_2.1.csff_enc.weight",
            "32.csff_enc.bias"       : "down_path_2.1.csff_enc.bias",
            "32.csff_dec.weight"     : "down_path_2.1.csff_dec.weight",
            "32.csff_dec.bias"       : "down_path_2.1.csff_dec.bias",
            "32.downsample.weight"   : "down_path_2.1.downsample.weight",
            "34.conv1.weight"        : "down_path_2.2.conv_1.weight",
            "34.conv1.bias"          : "down_path_2.2.conv_1.bias",
            "34.conv2.weight"        : "down_path_2.2.conv_2.weight",
            "34.conv2.bias"          : "down_path_2.2.conv_2.bias",
            "34.identity.weight"     : "down_path_2.2.identity.weight",
            "34.identity.bias"       : "down_path_2.2.identity.bias",
            "34.csff_enc.weight"     : "down_path_2.2.csff_enc.weight",
            "34.csff_enc.bias"       : "down_path_2.2.csff_enc.bias",
            "34.csff_dec.weight"     : "down_path_2.2.csff_dec.weight",
            "34.csff_dec.bias"       : "down_path_2.2.csff_dec.bias",
            "34.downsample.weight"   : "down_path_2.2.downsample.weight",
            "36.conv1.weight"        : "down_path_2.3.conv_1.weight",
            "36.conv1.bias"          : "down_path_2.3.conv_1.bias",
            "36.conv2.weight"        : "down_path_2.3.conv_2.weight",
            "36.conv2.bias"          : "down_path_2.3.conv_2.bias",
            "36.identity.weight"     : "down_path_2.3.identity.weight",
            "36.identity.bias"       : "down_path_2.3.identity.bias",
            "36.csff_enc.weight"     : "down_path_2.3.csff_enc.weight",
            "36.csff_enc.bias"       : "down_path_2.3.csff_enc.bias",
            "36.csff_dec.weight"     : "down_path_2.3.csff_dec.weight",
            "36.csff_dec.bias"       : "down_path_2.3.csff_dec.bias",
            "36.norm.weight"         : "down_path_2.3.norm.weight",
            "36.norm.bias"           : "down_path_2.3.norm.bias",
            "36.downsample.weight"   : "down_path_2.3.downsample.weight",
            "38.conv1.weight"        : "down_path_2.4.conv_1.weight",
            "38.conv1.bias"          : "down_path_2.4.conv_1.bias",
            "38.conv2.weight"        : "down_path_2.4.conv_2.weight",
            "38.conv2.bias"          : "down_path_2.4.conv_2.bias",
            "38.identity.weight"     : "down_path_2.4.identity.weight",
            "38.identity.bias"       : "down_path_2.4.identity.bias",
            "38.norm.weight"         : "down_path_2.4.norm.weight",
            "38.norm.bias"           : "down_path_2.4.norm.bias",
            "40.weight"              : "skip_conv_2.3.weight",
            "40.bias"                : "skip_conv_2.3.bias",
            "42.weight"              : "skip_conv_2.2.weight",
            "42.bias"                : "skip_conv_2.2.bias",
            "44.weight"              : "skip_conv_2.1.weight",
            "44.bias"                : "skip_conv_2.1.bias",
            "46.weight"              : "skip_conv_2.0.weight",
            "46.bias"                : "skip_conv_2.0.bias",
            "48.up.weight"           : "up_path_2.0.up.weight",
            "48.up.bias"             : "up_path_2.0.up.bias",
            "48.conv.conv1.weight"   : "up_path_2.0.conv_block.conv_1.weight",
            "48.conv.conv1.bias"     : "up_path_2.0.conv_block.conv_1.bias",
            "48.conv.conv2.weight"   : "up_path_2.0.conv_block.conv_2.weight",
            "48.conv.conv2.bias"     : "up_path_2.0.conv_block.conv_2.bias",
            "48.conv.identity.weight": "up_path_2.0.conv_block.identity.weight",
            "48.conv.identity.bias"  : "up_path_2.0.conv_block.identity.bias",
            "49.up.weight"           : "up_path_2.1.up.weight",
            "49.up.bias"             : "up_path_2.1.up.bias",
            "49.conv.conv1.weight"   : "up_path_2.1.conv_block.conv_1.weight",
            "49.conv.conv1.bias"     : "up_path_2.1.conv_block.conv_1.bias",
            "49.conv.conv2.weight"   : "up_path_2.1.conv_block.conv_2.weight",
            "49.conv.conv2.bias"     : "up_path_2.1.conv_block.conv_2.bias",
            "49.conv.identity.weight": "up_path_2.1.conv_block.identity.weight",
            "49.conv.identity.bias"  : "up_path_2.1.conv_block.identity.bias",
            "50.up.weight"           : "up_path_2.2.up.weight",
            "50.up.bias"             : "up_path_2.2.up.bias",
            "50.conv.conv1.weight"   : "up_path_2.2.conv_block.conv_1.weight",
            "50.conv.conv1.bias"     : "up_path_2.2.conv_block.conv_1.bias",
            "50.conv.conv2.weight"   : "up_path_2.2.conv_block.conv_2.weight",
            "50.conv.conv2.bias"     : "up_path_2.2.conv_block.conv_2.bias",
            "50.conv.identity.weight": "up_path_2.2.conv_block.identity.weight",
            "50.conv.identity.bias"  : "up_path_2.2.conv_block.identity.bias",
            "51.up.weight"           : "up_path_2.3.up.weight",
            "51.up.bias"             : "up_path_2.3.up.bias",
            "51.conv.conv1.weight"   : "up_path_2.3.conv_block.conv_1.weight",
            "51.conv.conv1.bias"     : "up_path_2.3.conv_block.conv_1.bias",
            "51.conv.conv2.weight"   : "up_path_2.3.conv_block.conv_2.weight",
            "51.conv.conv2.bias"     : "up_path_2.3.conv_block.conv_2.bias",
            "51.conv.identity.weight": "up_path_2.3.conv_block.identity.weight",
            "51.conv.identity.bias"  : "up_path_2.3.conv_block.identity.bias",
            "52.weight"              : "last.weight",
            "52.bias"                : "last.bias",
        },
        "head"    : {},
    }

    def __init__(self, config: Any = "hinet.yaml", *args, **kwargs):
        super().__init__(config=config, *args, **kwargs)
    
    def load_weights(self):
        """Load weights. It only loads the intersection layers of matching keys
        and shapes between the current model and weights.
        """
        if isinstance(self.weights, dict) \
            and self.weights["name"] in ["gopro", "reds", "sidd", "rain13k"]:
            state_dict = mmodel.load_state_dict_from_path(
                model_dir=self.zoo_dir, **self.weights
            )
            state_dict       = state_dict["params"]
            model_state_dict = self.model.state_dict()
            """
            for k in self.model.state_dict().keys():
                print(f"\"{k}\": ")
            for k in state_dict.keys():
                print(f"\"{k}\"")
            """
            for k, v in self.map_weights["backbone"].items():
                model_state_dict[k] = state_dict[v]
            if self.weights["num_classes"] == self.num_classes:
                for k, v in self.map_weights["head"].items():
                    model_state_dict[k] = state_dict[v]
            self.model.load_state_dict(model_state_dict)
        else:
            super().load_weights()
    
# endregion
