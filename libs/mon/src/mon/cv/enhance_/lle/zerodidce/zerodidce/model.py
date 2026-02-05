#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Zero-DiDCE model for low-light image enhancement.

References:
    - Paper: "Rethinking Zero-DCE for Low-Light Image Enhancement,"
      Neural Processing Letters 2024.
    - Code: https://github.com/Wenhui-Luo/Zero-DiDCE
"""

from typing import Any

import box
import torch

from mon import nn
from mon.core import MLType, MODELS, Path, ROOT_DIR, Task
from mon.nn import functional as F

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(variant="zerodidce", name="zerodidce")
class ZeroDiDCE(nn.Module, nn.ModelMetadataMixin):
    """Zero-DiDCE model for low-light image enhancement.
    
    References:
        - Paper: "Rethinking Zero-DCE for Low-Light Image Enhancement,"
          Neural Processing Letters 2024.
        - Code: https://github.com/Wenhui-Luo/Zero-DiDCE
    """
    
    _arch     : str          = "zerodidce"
    _name     : str          = "zerodidce"
    _tasks    : list[Task]   = [Task.LLE]
    _mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    _model_dir: Path         = root_dir
    _zoo      : dict         = box.Box({
        "siceme": {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/cv/enhance/lle/zerodidce/zerodidce/siceme/zerodidce_siceme.pth",
            "num_classes": None,
        },
    })
    
    def __init__(self, weights: Any = None):
        super().__init__()
        in_channels   = 3
        hidden_dim    = 32
        hidden_dim_x2 = hidden_dim * 2
        out_channels  = 3
        self.e_conv1  = nn.Conv2d(in_channels,   hidden_dim,   3, 1, 1, bias=True)
        self.e_conv2  = nn.Conv2d(hidden_dim,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv3  = nn.Conv2d(hidden_dim,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv7  = nn.Conv2d(hidden_dim_x2, out_channels, 3, 1, 1, bias=True)
        self.relu     = nn.ReLU(inplace=True)
        self.maxpool  = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        
        # Load weights
        self.load_weights(weights)
        
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Branch 1
        x    = image
        x1   = self.relu(self.e_conv1(x))
        x2   = self.relu(self.e_conv2(x1))
        x3   = self.relu(self.e_conv3(x2))
        r    =    F.tanh(self.e_conv7(torch.cat([x1, x3], 1)))
        # Branch 2
        x_i  = 1 - x
        x_i1 = self.relu(self.e_conv1(x_i))
        x_i2 = self.relu(self.e_conv2(x_i1))
        x_i3 = self.relu(self.e_conv3(x_i2))
        r1   =    F.tanh(self.e_conv7(torch.cat([x_i1, x_i3], 1)))
        #
        r   = (r + r1) / 2

        x_m = torch.mean(x).item()
        n1  = 0.63
        s   = x_m * x_m
        n3  = -0.79 * s + 0.81 * x_m + 1.4
        
        if x_m < 0.1:
            b = -25 * x_m + 10
        elif x_m < 0.45:
            b = 17.14 * s - 15.14 * x_m + 10
        else:
            b =  5.66 * s -  2.93 * x_m + 7.2

        b = int(b)
        for i in range(b):
            x = x + r * (torch.pow(x, 2) - x) * ((n1 - torch.mean(x).item()) / (n3 - torch.mean(x).item()))  # + (n1-x)*0.01

        # xxx0 = 1 - x
        # xxx0 = xxx0 - 0.22
        # x = x + xxx0 * 0.15

        y = x
        return r, y
