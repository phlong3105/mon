#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules.

This module provides various layers, blocks, and modules for the CoLIE model.
"""

from __future__ import annotations

__all__ = [
    "ResidualINR",
]

import torch
from torch import nn, Tensor

from mon.nn import SineLinear


# ==============================================================================
# region MODULES
# ==============================================================================

class ResidualINR(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        patch_dim: int,
        hidden_dim: int,
        num_layers: int,
        add_layer: int,
        weight_decay: list[float] | None = None
    ):
        super().__init__()
        '''
        `add_layer` should be in range of  [1, num_layers-2]
        '''
        patch_layers = [SineLinear(patch_dim, hidden_dim, is_first=True)]
        spatial_layers = [SineLinear(2, hidden_dim, is_first=True)]
        output_layers = []

        for _ in range(1, add_layer - 2):
            patch_layers.append(SineLinear(hidden_dim, hidden_dim))
            spatial_layers.append(SineLinear(hidden_dim, hidden_dim))
        patch_layers.append(SineLinear(hidden_dim, hidden_dim // 2))
        spatial_layers.append(SineLinear(hidden_dim, hidden_dim // 2))

        for _ in range(add_layer, num_layers - 1):
            output_layers.append(SineLinear(hidden_dim, hidden_dim))
        output_layers.append(SineLinear(hidden_dim, 1, is_last=True))
        output_layers.append(nn.Sigmoid())

        self.patch_net = nn.Sequential(*patch_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net = nn.Sequential(*output_layers)

        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]

        self.params = []
        self.params += [{"params": self.spatial_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patch_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),"weight_decay": weight_decay[2]}]

    # --- Callable & Context Manager ---
    def forward(self, patch: Tensor, spatial: Tensor) -> Tensor:
        patch = self.patch_net(patch)
        spatial = self.spatial_net(spatial)
        concat = torch.cat((patch, spatial), dim=-1)
        return self.output_net(concat)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
