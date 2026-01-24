#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SALEO modules.

This module provides various layers, blocks, and modules.
"""

from __future__ import annotations

__all__ = [
    "InDi",
    "SIREN",
    "SIREN_DAM",
]

from typing import Any, Optional

import torch

from mon import nn


# ==============================================================================
# region UTILS
# ==============================================================================

def depth_aware_modulation(
    patch    : torch.Tensor,
    depth    : torch.Tensor,
    patch_dim: int,
    alpha    : float = 8.3
) -> torch.Tensor:
    """Modulate intensity features based on depth similarity.

    Adjust the intensity features of image patches according to the depth
    similarity between the center pixel and its neighbors. This helps to
    emphasize features from geometrically similar regions while attenuating
    those from dissimilar regions.
    """
    # 1. Calculate absolute depth difference: |D(p_i) - depth_center|
    center  = patch_dim // 2
    depth_c = depth[:, :, center:center + 1]
    depth_d = torch.abs(depth_c - depth)
    # 2. Calculate Depth Similarity Weights: f_d = exp(-alpha * depth_diff)
    # These weights define how much each neighbor contributes to the output feature.
    f_d     = torch.exp(-alpha * depth_d)
    # 3. Modulate the Intensity Features: N(y'_V) = N(y_V) * f_d
    # Element-wise multiplication ensures that features from geometrically
    # dissimilar pixels (high depth_diff, low f_d) are attenuated.
    patch   = patch * f_d
    return patch

# endregion


# ==============================================================================
# region MODULES
# ==============================================================================

# --- INR Networks ---

class SIREN(nn.Module):
    """SIREN network (similar to CoLIE model).

    Dual-path SIREN network that processes coordinate and image patch inputs
    separately before combining them for final output prediction.

    Attributes:
        patch_dim (int): Input dimension of the patch branch.
        hidden_dim (int): Hidden dimension of the networks.
        coords_dim (int): Output dimension of the coordinate branch. Defaults to 2.
        pos_encode (nn.PosEncodingNeRF): Positional encoding module for coordinates.
        coord_net (nn.Sequential): SIREN network for processing coordinates.
        patch_net (nn.Sequential): SIREN network for processing patches.
        output_net (nn.Sequential): Output network for combining features.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        patch_dim   : int,
        hidden_dim  : int  = 256,
        num_layers  : int  = 4,
        add_layers  : int  = 2,
        pos_encode  : bool = False,
        weight_decay: Any  = None,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            patch_dim: Input dimension of the patch branch.
            hidden_dim: Hidden dimension of the networks. Defaults to 256.
            num_layers: Number of layers in each branch. Defaults to 4.
            add_layers: Number of layers to add between the two branches. Defaults to 2.
            pos_encode: Whether to use positional encoding. Defaults to False.
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
            self.pos_encode = nn.PosEncodingNeRF(in_features=2, sidelength=hidden_dim)
            self.coords_dim = self.pos_encode.out_features
        else:
            self.pos_encode = None
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
        output_layers.append(nn.Linear(hidden_dim, 1))
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
        coord: torch.Tensor,
        patch: torch.Tensor,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward the input through the network."""
        # Process each branch
        coord_e  = self.pos_encode(coord) if self.pos_encode else coord
        coord_f  = self.coords_net(coord_e)
        patch_f  = self.patch_net(patch)
        concat_f = torch.cat((coord_f, patch_f), dim=-1)
        # Final output
        output   = self.output_net(concat_f)
        return output, concat_f


# noinspection PyMethodOverriding
class SIREN_DAM(SIREN):
    """SIREN network with Depth-Aware Modulation (DAM)."""

    # --- Callable & Context Manager ---
    def forward(
        self,
        coord: torch.Tensor,
        patch: torch.Tensor,
        depth: torch.Tensor,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward the input through the network."""
        # Modulate intensity features based on depth similarity
        patch = depth_aware_modulation(patch, depth, self.patch_dim)
        # Call parent forward method
        return super().forward(coord, patch, *args, **kwargs)


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
        coord: torch.Tensor,
        patch: torch.Tensor,
        prev : torch.Tensor,
        t    : torch.Tensor,
        *args, **kwargs
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward the input through the network."""
        # Backbone branch
        backbone_out, backbone_f = self.backbone(coord, patch)
        # Auxiliary branch
        t_tensor    = torch.ones_like(prev) * t
        aux_in      = torch.cat((coord, patch, prev, t_tensor), dim=-1)
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
