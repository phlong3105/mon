#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements MobileIE model for real-time image enhancement on mobile devices.

References:
    - Paper: "MobileIE: An Extremely Lightweight and Effective ConvNet for
      Real-Time Image Enhancement on Mobile Devices," ICCV 2025.
    - Code: https://github.com/AVC2-UESTC/MobileIE
"""

__all__ = [
    "MobileIELLE",
]

from typing import Any

import box
import torch

from mon import nn
from mon.core import MLType, MODELS, Path, ROOT_DIR, Task
from .network.utils import DropBlock, FST, FSTS, MBRConv1, MBRConv3, MBRConv5

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="mobileie_lle", arch="mobileie")
class MobileIELLE(nn.Module, nn.ModelMetadataMixin):
    """MobileIE model for real-time low-light image enhancement.
    
    References:
        - Paper: "MobileIE: An Extremely Lightweight and Effective ConvNet for
          Real-Time Image Enhancement on Mobile Devices," ICCV 2025.
        - Code: https://github.com/AVC2-UESTC/MobileIE
    """
    
    _arch     : str          = "mobileie"
    _name     : str          = "mobileie_lle"
    _tasks    : list[Task]   = [Task.LLE]
    _mltypes  : list[MLType] = [MLType.SUPERVISED]
    _model_dir: Path         = root_dir
    _zoo      : dict         = box.Box({
        "lolv1"    : {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/multitask/mobileie/mobileie_lle/lolv1/mobileie_lle_lolv1_slim.pkl",
            "num_classes": None,
        },
        "lolv2real": {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/multitask/mobileie/mobileie_lle/lolv2real/mobileie_lle_lolv2real_slim.pkl",
            "num_classes": None,
        },
    })
    
    def __init__(
        self,
        channels : int,
        rep_scale: int  = 4,
        inference: bool = False,
        weights  : Any  = None,
    ):
        super().__init__()
        self.channels = channels
        
        if inference:
            (self.head,
             self.body,
             self.att, self.att1,
             self.tail) = self.build_network_slim(channels)
            self.load_weights(weights)
        else:
            (self.head,
             self.body,
             self.att,  self.att1,
             self.tail, self.tail_warm,
             self.drop) = self.build_network(channels, rep_scale)
    
    def build_network(self, channels: int, rep_scale: int = 4):
        head = FST(
            nn.Sequential(
                MBRConv5(3, channels, rep_scale=rep_scale),
                nn.PReLU(channels),
                MBRConv3(channels, channels, rep_scale=rep_scale)
            ),
            channels
        )
        body = FST(
            MBRConv3(channels, channels, rep_scale=rep_scale),
            channels
        )
        att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            MBRConv1(channels, channels, rep_scale=rep_scale),
            nn.Sigmoid()
        )
        att1 = nn.Sequential(
            MBRConv1(1, channels, rep_scale=rep_scale),
            nn.Sigmoid()
        )
        tail      = MBRConv3(channels, 3, rep_scale=rep_scale)
        tail_warm = MBRConv3(channels, 3, rep_scale=rep_scale)
        drop      = DropBlock(3)
        return head, body, att, att1, tail, tail_warm, drop
    
    def build_network_slim(self, channels: int):
        head = FSTS(
            nn.Sequential(
                nn.Conv2d(3, channels, 5, 1, 2),
                nn.PReLU(channels),
                nn.Conv2d(channels, channels, 3, 1, 1)
            ),
            channels
        )
        body = FSTS(
            nn.Conv2d(channels, channels, 3, 1, 1),
            channels
        )
        att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        att1 = nn.Sequential(
            nn.Conv2d(1, channels, 1, 1),
            nn.Sigmoid()
        )
        tail = nn.Conv2d(channels, 3, 3, 1, 1)
        return head, body, att, att1, tail
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.head(x)
        x1 = self.body(x0)
        x2 = self.att(x1)
        max_out, _ = torch.max(x2 * x1 , dim=1, keepdim=True)
        x3 = self.att1(max_out)
        x4 = torch.mul(x2, x3) * x1
        return self.tail(x4)

    def forward_warm(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.drop(x)
        x = self.head(x)
        x = self.body(x)
        return self.tail(x), self.tail_warm(x)

    def slim(self) -> nn.Module:
        net_slim    = MobileIELLE(self.channels, inference=True)
        weight_slim = net_slim.state_dict()
        for name, mod in self.named_modules():
            if isinstance(mod, MBRConv3) or isinstance(mod, MBRConv5) or isinstance(mod, MBRConv1):
               if "%s.weight" % name in weight_slim:
                    w, b = mod.slim()
                    weight_slim["%s.weight" % name] = w
                    weight_slim["%s.bias"   % name] = b
            elif isinstance(mod, FST):
                weight_slim["%s.bias"    % name] = mod.bias
                weight_slim["%s.weight1" % name] = mod.weight1
                weight_slim["%s.weight2" % name] = mod.weight2
            elif isinstance(mod, nn.PReLU):
                weight_slim["%s.weight"  % name] = mod.weight
        net_slim.load_state_dict(weight_slim)
        return net_slim
