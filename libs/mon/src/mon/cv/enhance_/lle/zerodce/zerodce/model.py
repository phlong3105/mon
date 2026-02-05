#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Zero-DCE model for low-light image enhancement.

References:
    - Paper: "Zero-Reference Deep Curve Estimation for Low-Light Image
      Enhancement," CVPR 2020.
    - Code: https://github.com/Li-Chongyi/Zero-DCE
"""

__all__ = [
    "ZeroDCE",
]

from typing import Any

import box
import torch

from mon import nn
from mon.core import MLType, MODELS, Path, ROOT_DIR, Task
from mon.nn import functional as F

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


# --- Modules ---
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        

# --- Model ---
@MODELS.register(variant="zerodce", name="zerodce")
class ZeroDCE(nn.Module, nn.ModelMetadataMixin):
    """Zero-DCE model for low-light image enhancement.
    
    References:
        - Paper: "Zero-Reference Deep Curve Estimation for Low-Light Image
          Enhancement," CVPR 2020.
        - Code: https://github.com/Li-Chongyi/Zero-DCE
    """
    
    _arch     : str          = "zerodce"
    _name     : str          = "zerodce"
    _tasks    : list[Task]   = [Task.LLE]
    _mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    _model_dir: Path         = root_dir
    _zoo      : dict         = box.Box({
        "siceme": {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/zerodce/zerodce/siceme/zerodce_siceme.pth",
            "num_classes": None,
        },
    })

    def __init__(self, weights: Any = None):
        super().__init__()
        in_channels   = 3
        hidden_dim    = 32
        out_channels  = 24
        self.e_conv1  = nn.Conv2d(in_channels,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv2  = nn.Conv2d(hidden_dim,     hidden_dim,   3, 1, 1, bias=True)
        self.e_conv3  = nn.Conv2d(hidden_dim,     hidden_dim,   3, 1, 1, bias=True)
        self.e_conv4  = nn.Conv2d(hidden_dim,     hidden_dim,   3, 1, 1, bias=True)
        self.e_conv5  = nn.Conv2d(hidden_dim * 2, hidden_dim,   3, 1, 1, bias=True)
        self.e_conv6  = nn.Conv2d(hidden_dim * 2, hidden_dim,   3, 1, 1, bias=True)
        self.e_conv7  = nn.Conv2d(hidden_dim * 2, out_channels, 3, 1, 1, bias=True)
        self.relu     = nn.ReLU(inplace=True)
        self.maxpool  = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.apply(weights_init)
        
        # Load weights
        self.load_weights(weights)
        
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x1 = self.relu(self.e_conv1(image))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        r  =    F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(r, 3, dim=1)
        y0 = image
        y1 = y0 + r1 * (torch.pow(y0, 2) - y0)
        y2 = y1 + r2 * (torch.pow(y1, 2) - y1)
        y3 = y2 + r3 * (torch.pow(y2, 2) - y2)
        y4 = y3 + r4 * (torch.pow(y3, 2) - y3)
        y5 = y4 + r5 * (torch.pow(y4, 2) - y4)
        y6 = y5 + r6 * (torch.pow(y5, 2) - y5)
        y7 = y6 + r7 * (torch.pow(y6, 2) - y6)
        y8 = y7 + r8 * (torch.pow(y7, 2) - y7)
        
        return r, y1, y2, y3, y4, y5, y6, y7, y8
