#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "SIREN",
]

from typing import Any

import numpy as np
import torch

from mon import nn


# ----- SIREN -----
class SIRENLayer(nn.Module):
    
    def __init__(
        self,
        in_features : int,
        out_features: int,
        w0          : float = 30,
        is_first    : bool  = False,
        is_last     : bool  = False
    ):
        super().__init__()
        self.in_features = in_features
        self.w0          = w0
        self.linear      = nn.Linear(in_features, out_features)
        self.is_first    = is_first
        self.is_last     = is_last
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1.0 / self.in_features,
                                             1.0 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6.0 / self.in_features) / self.w0,
                                             np.sqrt(6.0 / self.in_features) / self.w0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return nn.Sigmoid()(x) if self.is_last else torch.sin(self.w0 * x)


class SIREN(nn.Module):
    
    def __init__(
        self,
        patch_dim   : int,
        hidden_dim  : int,
        num_layers  : int,
        add_layer   : int,
        weight_decay: Any = None
    ):
        super().__init__()

        patch_layers   = [SIRENLayer(patch_dim,   hidden_dim, is_first=True)]
        spatial_layers = [SIRENLayer(2, hidden_dim, is_first=True)]
        output_layers  = []
        
        for _ in range(1, add_layer - 2):
            patch_layers.append(SIRENLayer(hidden_dim, hidden_dim))
            spatial_layers.append(SIRENLayer(hidden_dim, hidden_dim))
        patch_layers.append(SIRENLayer(hidden_dim, hidden_dim // 2))
        spatial_layers.append(SIRENLayer(hidden_dim, hidden_dim // 2))
        
        for _ in range(add_layer, num_layers - 1):
            output_layers.append(SIRENLayer(hidden_dim, hidden_dim))
        output_layers.append(SIRENLayer(hidden_dim, 1, is_last=True))

        self.patch_net   = nn.Sequential(*patch_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net  = nn.Sequential(*output_layers)
        
        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]
            
        self.params = []
        self.params += [{"params": self.spatial_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patch_net.parameters(),   "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),  "weight_decay": weight_decay[2]}]

    def forward(self, patch: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        return self.output_net(torch.cat((self.patch_net(patch), self.spatial_net(spatial)), -1))
