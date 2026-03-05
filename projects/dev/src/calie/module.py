#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules.

This module provides various layers, blocks, and modules.
"""

from __future__ import annotations

__all__ = [
    "ResidualINR",
]

from typing import Any

import torch
from torch import nn, Tensor

from mon.nn import FourierPE, SineLinear


# ==============================================================================
# region MODULES
# ==============================================================================

# --- INR Networks ---

class ResidualINR(nn.Module):
    """SIREN network for residual mapping.

    A conditional INR using SIREN layers. It doesn't just memorize coordinates
    ``(x, y)``; it looks at the local neighborhood of the input image to decide
    how to enhance the pixel.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        patch_dim: int,
        out_features: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 4,
        add_layers: int = 2,
        pos_encode: bool = False,
        mapping_size: int = 256,
        B: float = 20.0,
        weight_decay: Any = None,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            patch_dim (int): Input dimension of the patch branch.
            out_features (int, optional): Number of output features.
                Defaults to 1.
            hidden_dim (int, optional): Hidden dimension of the networks.
                Defaults to 256.
            num_layers (int, optional): Number of layers in each branch.
                Defaults to 4.
            add_layers (int, optional): Number of layers to add between the two
                branches. Defaults to 2.
            pos_encode (bool, optional): Whether to use positional encoding.
                Defaults to False.
            mapping_size (int, optional): Size of Fourier feature mapping.
                Defaults to 256.
            B (float, optional): Fourier feature scaling factor. Defaults to 20.0.
            weight_decay (Any): Weight decay parameters for each branch.
                Defaults to None.
        """
        super().__init__()
        # Assign attributes
        self.patch_dim = patch_dim
        self.hidden_dim = hidden_dim

        # Define network
        if pos_encode:
            self.ff = FourierPE(mapping_size=mapping_size, B=B)
            self.coords_dim = self.ff.out_features
        else:
            self.ff = None
            self.coords_dim = 2

        coord_layers = [SineLinear(self.coords_dim, hidden_dim, is_first=True)]
        patch_layers = [SineLinear(self.patch_dim, hidden_dim, is_first=True)]
        for _ in range(1, add_layers - 2):
            coord_layers.append(SineLinear(hidden_dim, hidden_dim))
            patch_layers.append(SineLinear(hidden_dim, hidden_dim))
        coord_layers.append(SineLinear(hidden_dim, hidden_dim // 2))
        patch_layers.append(SineLinear(hidden_dim, hidden_dim // 2))

        output_layers = []
        for _ in range(add_layers, num_layers - 1):
            output_layers.append(SineLinear(hidden_dim, hidden_dim))
        output_layers.append(SineLinear(hidden_dim, out_features, is_last=True))
        output_layers.append(nn.Sigmoid())

        self.coord_net = nn.Sequential(*coord_layers)
        self.patch_net = nn.Sequential(*patch_layers)
        self.output_net = nn.Sequential(*output_layers)

        # Weight decay params
        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]
        self.params = []
        self.params += [{"params": self.coord_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patch_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(), "weight_decay": weight_decay[2]}]

    # --- Callable & Context Manager ---
    def forward(self, coords : Tensor, patches: Tensor) -> Tensor:
        """Forward the input through the network.

        Args:
            coords (Tensor): Input coordinates tensor of shape (... , 2) and
                values ranging from -1.0 to 1.0.
            patches (Tensor): Input patches tensor of shape (... , C) and
                values ranging from 0.0 to 1.0.
        """
        # 1. Process Spatial Branch
        coords_e = self.ff(coords) if self.ff is not None else coords
        coords_f = self.coord_net(coords_e)

        # 2. Process Photometric Branch
        patches_f = self.patch_net(patches)

        # 3. Fuse and output
        concat_f = torch.cat((coords_f, patches_f), dim=-1)
        output = self.output_net(concat_f)

        return output

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
