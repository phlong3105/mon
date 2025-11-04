#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Zero-DCE++ model for low-light image enhancement.

References:
    - Paper: "Learning to Enhance Low-Light Image via Zero-Reference Deep Curve
      Estimation," IEEE TPAMI 2022.
    - Code: https://github.com/Li-Chongyi/Zero-DCE_extension
"""

__all__ = [
    "ZeroDCEpp",
]

from typing import Any

import box
import torch

from mon.constants import MODELS, ROOT_DIR
from mon.core import MLType, ModelMixin, nn, Path, Task
from mon.core.nn import functional as F

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


# ----- Modules -----
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
        
class DSConv(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels,
            kernel_size  = 3,
            stride       = 1,
            padding      = 1,
            groups       = in_channels
        )
        self.point_conv = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            groups       = 1
        )
        self.apply(weights_init)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        y = self.depth_conv(input)
        y = self.point_conv(y)
        return y


# ----- Model -----
@MODELS.register(name="zerodce++", arch="zerodce++")
class ZeroDCEpp(nn.Module, ModelMixin):
    """Zero-DCE++ model for low-light image enhancement.
    
    References:
        - Paper: "Learning to Enhance Low-Light Image via Zero-Reference Deep Curve
          Estimation," IEEE TPAMI 2022.
        - Code: https://github.com/Li-Chongyi/Zero-DCE_extension
    """
    
    arch     : str          = "zerodce++"
    name     : str          = "zerodce++"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box({
        "siceme": {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/zerodce++/zerodce++/siceme/zerodce++_siceme.pth",
            "num_classes": None,
        },
    })
    
    def __init__(self, scale_factor: float = 1.0, weights: Any = None):
        super().__init__()
        self.scale_factor = scale_factor
        
        in_channels   = 3
        hidden_dim    = 32
        out_channels  = 3
        self.e_conv1  = DSConv(in_channels,    hidden_dim)
        self.e_conv2  = DSConv(hidden_dim,     hidden_dim)
        self.e_conv3  = DSConv(hidden_dim,     hidden_dim)
        self.e_conv4  = DSConv(hidden_dim,     hidden_dim)
        self.e_conv5  = DSConv(hidden_dim * 2, hidden_dim)
        self.e_conv6  = DSConv(hidden_dim * 2, hidden_dim)
        self.e_conv7  = DSConv(hidden_dim * 2, out_channels)
        self.relu     = nn.ReLU(inplace=True)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        
        # Load weights
        self.load_weights(weights)
        
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.scale_factor == 1:
            x_down = image
        else:
            x_down = F.interpolate(image, scale_factor=1 / self.scale_factor, mode="bilinear")

        r  = self.learn_curve(x_down)
        
        if self.scale_factor == 1:
            r = r
        else:
            r = self.upsample(r)
            
        y1, y2, y3, y4, y5, y6, y7, y8 = self.enhance(image, r)
        return r, y1, y2, y3, y4, y5, y6, y7, y8
    
    def learn_curve(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        r  =    F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        return r
    
    def enhance(self, image: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, ...]:
        y0 = image
        y1 = y0 + r * (torch.pow(y0, 2) - y0)
        y2 = y1 + r * (torch.pow(y1, 2) - y1)
        y3 = y2 + r * (torch.pow(y2, 2) - y2)
        y4 = y3 + r * (torch.pow(y3, 2) - y3)
        y5 = y4 + r * (torch.pow(y4, 2) - y4)
        y6 = y5 + r * (torch.pow(y5, 2) - y5)
        y7 = y6 + r * (torch.pow(y6, 2) - y6)
        y8 = y7 + r * (torch.pow(y7, 2) - y7)
        return y1, y2, y3, y4, y5, y6, y7, y8
