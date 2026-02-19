#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SALEO modules.

This module provides various layers, blocks, and modules.
"""

from __future__ import annotations

__all__ = [
    "InDi",
    "ReflectanceINR",
    "ResidualINR",
]

from typing import Any, Optional

import torch

from mon import nn


# ==============================================================================
# region MODULES
# ==============================================================================

# --- INR Networks ---

class ResidualINR(nn.Module):
    """SIREN network for residual mapping.

    A conditional INR using SIREN layers. It doesn't just memorize coordinates
    ``(x, y)``; it looks at the local neighborhood of the input image to decide
    how to enhance the pixel.

    Attributes:
        patch_dim (int): Input dimension of the patch branch.
        hidden_dim (int): Hidden dimension of the networks.
        coords_dim (int): Output dimension of the coordinate branch.
        ff (nn.FourierPE): Fourier feature mapping for coordinates.
        coord_net (nn.Sequential): SIREN network for processing coordinates.
        patch_net (nn.Sequential): SIREN network for processing patches.
        output_net (nn.Sequential): Output network for combining features.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        patch_dim   : int,
        hidden_dim  : int   = 256,
        num_layers  : int   = 4,
        add_layers  : int   = 2,
        pos_encode  : bool  = False,
        mapping_size: int   = 256,
        B           : float = 20.0,
        weight_decay: Any   = None,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            patch_dim: Input dimension of the patch branch.
            hidden_dim: Hidden dimension of the networks. Defaults to 256.
            num_layers: Number of layers in each branch. Defaults to 4.
            add_layers: Number of layers to add between the two branches. Defaults to 2.
            pos_encode: Whether to use positional encoding. Defaults to False.
            mapping_size: Size of Fourier feature mapping. Defaults to 256.
            B: Fourier feature scaling factor. Defaults to 20.0.
            weight_decay: Weight decay parameters for each branch. Defaults to None.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        # Attribute assignments
        self.patch_dim  = patch_dim
        self.hidden_dim = hidden_dim

        # Define networks
        if pos_encode:
            self.ff         = nn.FourierPE(mapping_size=mapping_size, B=B)
            self.coords_dim = self.ff.out_features
        else:
            self.ff         = None
            self.coords_dim = 2

        coord_layers = [nn.SineLinear(self.coords_dim, hidden_dim, is_first=True)]
        patch_layers = [nn.SineLinear(self.patch_dim,  hidden_dim, is_first=True)]
        for _ in range(1, add_layers - 2):
            coord_layers.append(nn.SineLinear(hidden_dim, hidden_dim))
            patch_layers.append(nn.SineLinear(hidden_dim, hidden_dim))
        coord_layers.append(nn.SineLinear(hidden_dim, hidden_dim // 2))
        patch_layers.append(nn.SineLinear(hidden_dim, hidden_dim // 2))

        output_layers = []
        for _ in range(add_layers, num_layers - 1):
            output_layers.append(nn.SineLinear(hidden_dim, hidden_dim))
        output_layers.append(nn.SineLinear(hidden_dim, 1, is_last=True))
        output_layers.append(nn.Sigmoid())

        self.coord_net  = nn.Sequential(*coord_layers)
        self.patch_net  = nn.Sequential(*patch_layers)
        self.output_net = nn.Sequential(*output_layers)

        # Weight decay params
        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{"params": self.coord_net.parameters(),  "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patch_net.parameters(),  "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(), "weight_decay": weight_decay[2]}]

    # --- Callable & Context Manager ---
    def forward(
        self,
        coords : torch.Tensor,
        patches: torch.Tensor,
    ) -> torch.Tensor:
        """Forward the input through the network.

        Args:
            coords: Input coordinates, formatted as a torch.Tensor of shape
                (... , 2) and values ranging from -1.0 to 1.0.
            patches: Input patches, formatted as a torch.Tensor of shape
                (... , C) and values ranging from 0.0 to 1.0.
        """
        # Process each branch
        coords_e  = self.ff(coords) if self.ff is not None else coords
        coords_f  = self.coord_net(coords_e)
        patches_f = self.patch_net(patches)
        concat_f  = torch.cat((coords_f, patches_f), dim=-1)
        # Final output
        output    = self.output_net(concat_f)
        return output


class ReflectanceINR(nn.Module):
    """SIREN network for reflectance mapping.

    Attributes:
        hidden_dim (int): Hidden dimension of the networks.
        coords_dim (int): Output dimension of the coordinate branch.
        ff (nn.FourierPE): Fourier feature mapping for coordinates.
        coord_net (nn.Sequential): SIREN network for processing coordinates.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        hidden_dim  : int   = 256,
        num_layers  : int   = 4,
        pos_encode  : bool  = False,
        mapping_size: int   = 256,
        B           : float = 20.0,
        weight_decay: Any   = None,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            hidden_dim: Hidden dimension of the networks. Defaults to 256.
            num_layers: Number of layers in each branch. Defaults to 4.
            pos_encode: Whether to use positional encoding. Defaults to False.
            mapping_size: Size of Fourier feature mapping. Defaults to 256.
            B: Fourier feature scaling factor. Defaults to 20.0.
            weight_decay: Weight decay parameters for each branch. Defaults to None.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        # Attribute assignments
        self.hidden_dim = hidden_dim

        # Define networks
        if pos_encode:
            self.ff         = nn.FourierPE(mapping_size=mapping_size, B=B)
            self.coords_dim = self.ff.out_features
        else:
            self.ff         = None
            self.coords_dim = 2

        coord_layers = []
        coord_layers.append(nn.SineLinear(self.coords_dim, hidden_dim, is_first=True))
        for _ in range(1, num_layers - 1):
            coord_layers.append(nn.SineLinear(hidden_dim, hidden_dim))
        coord_layers.append(nn.SineLinear(hidden_dim, 1, is_last=True))
        coord_layers.append(nn.Sigmoid())

        self.coord_net = nn.Sequential(*coord_layers)

        # Weight decay params
        if not weight_decay:
            weight_decay = [0.1]
        self.params  = []
        self.params += [{"params": self.coord_net.parameters(),  "weight_decay": weight_decay[0]}]

    # --- Callable & Context Manager ---
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward the input through the network.

        Args:
            coords: Input coordinates, formatted as a torch.Tensor of shape
                (... , 2) and values ranging from -1.0 to 1.0.
        """
        # Process each branch
        coords_e = self.ff(coords) if self.ff is not None else coords
        return self.coord_net(coords_e)


# --- InDi Networks ---

class InDi(nn.Module):
    """InDi network.

    Dual-branch network that combines a backbone (e.g., SIREN) with an auxiliary
    network to iteratively refine predictions based on previous outputs and time.

    Attributes:
        backbone (nn.Module): Backbone network (e.g., SIREN).
        aux_net (nn.Sequential): Auxiliary network for iterative feedback.
        proj (nn.Sequential): Projection layer for concatenated features.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, backbone: nn.Module, weight_decay: Any = None, *args, **kwargs):
        """Initialize a new instance.

        Args:
            backbone: Backbone network (e.g., SIREN).
            weight_decay: Weight decay parameters for aux and proj networks. Defaults to None.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        # Attribute assignments
        self.backbone = backbone
        patch_dim     = int(backbone.patch_dim)
        hidden_dim    = int(backbone.hidden_dim)

        # Auxiliary network for iterative feedback (small MLP with sin activations)
        aux_in_dim    = 2 + patch_dim + 1 + 1  # spatial (2) + patch + prev_g (1) + t (1)
        self.aux_net  = nn.Sequential(
            nn.SineLinear(aux_in_dim, hidden_dim),
            nn.SineLinear(hidden_dim, hidden_dim)
        )
        # Projection layer for concatenated features
        proj_in_dim   = hidden_dim + hidden_dim  # backbone_feat + aux_out
        self.proj     = nn.Sequential(
            nn.Linear(proj_in_dim, 1),
            nn.Sigmoid(),
        )

        # Weight decay params
        if not weight_decay:
            weight_decay = [0.001, 0.001]  # Added for aux and proj
        self.params  = []
        self.params += [{"params": self.aux_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.proj.parameters(),    "weight_decay": weight_decay[1]}]

    # --- Callable & Context Manager ---
    def forward(
        self,
        coords : torch.Tensor,
        patches: torch.Tensor,
        prev   : torch.Tensor,
        t      : torch.Tensor,
        *args, **kwargs
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward the input through the network."""
        # Backbone branch
        backbone_out, backbone_f = self.backbone(coords, patches)
        # Auxiliary branch
        t_tensor    = torch.ones_like(prev) * t
        aux_in      = torch.cat((coords, patches, prev, t_tensor), dim=-1)
        aux_out     = self.aux_net(aux_in)
        # Concatenate and project
        concat_feat = torch.cat((backbone_f, aux_out), -1)
        proj_out    = self.proj(concat_feat)
        # Final output: element-wise multiplication (dot product for scalars)
        output      = proj_out * backbone_out
        return output, None

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
