#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "DAM_SIREN",
    "DAM_iSIREN",
    "D_SIREN",
    "SIREN",
    "iSIREN",
    "peSIREN",
]

from typing import Any

import torch

from mon.core import nn


# ----- Utils -----
def positional_encoding(delta_z: torch.Tensor, num_freq: int = 6) -> torch.Tensor:
    """Positional encoding for depth deviation as in DINER."""
    freq_bands = torch.pow(2, torch.linspace(0, num_freq - 1, num_freq, device=delta_z.device))  # DINER uses exponential with base 2
    encoded    = [torch.sin(delta_z * freq) for freq in freq_bands] + [torch.cos(delta_z * freq) for freq in freq_bands]
    return torch.cat(encoded + [delta_z], dim=-1)  # (..., 2 * num_freq + 1), e.g., 13D for num_freq=6


# ----- SIREN -----
class SIREN(nn.Module):
    """The original SIREN network, similar to CoLIE model."""
    
    def __init__(
        self,
        patch_dim   : int,
        hidden_dim  : int = 256,
        num_layers  : int = 4,
        add_layers  : int = 2,
        weight_decay: Any = None,
    ):
        super().__init__()
        
        # Backbone networks
        coords_layers  = [nn.SineLayer(2, hidden_dim, is_first=True)]
        patches_layers = [nn.SineLayer(patch_dim, hidden_dim, is_first=True)]
        for _ in range(1, add_layers - 2):
            coords_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
            patches_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        coords_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        patches_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        
        output_layers = []
        for _ in range(add_layers, num_layers - 1):
            output_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        output_layers.append(nn.Linear(hidden_dim, 1))
        output_layers.append(nn.Sigmoid())
        
        self.coords_net  = nn.Sequential(*coords_layers)
        self.patches_net = nn.Sequential(*patches_layers)
        self.output_net  = nn.Sequential(*output_layers)
        
        # Weight decay params
        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{"params": self.coords_net.parameters(),  "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patches_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),  "weight_decay": weight_decay[2]}]
        
    def forward(self, coords: torch.Tensor, patches: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        coords_feat   = self.coords_net(coords)
        patches_feat  = self.patches_net(patches)
        backbone_feat = torch.cat((coords_feat, patches_feat), -1)
        backbone_out  = self.output_net(backbone_feat)
        return backbone_out


class peSIREN(nn.Module):
    """SIREN network with positional encoding for coordinates."""
    
    def __init__(
        self,
        patch_dim   : int,
        hidden_dim  : int = 256,
        num_layers  : int = 4,
        add_layers  : int = 2,
        weight_decay: Any = None
    ):
        super().__init__()
        self.encoding  = nn.PosEncodingNeRF(in_features=2, sidelength=hidden_dim)
        spatial_dim    = self.encoding.out_features
        
        coords_layers  = [nn.SineLayer(spatial_dim, hidden_dim, is_first=True)]
        patches_layers = [nn.SineLayer(patch_dim,   hidden_dim, is_first=True)]
        for _ in range(1, add_layers - 2):
            coords_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
            patches_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        coords_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        patches_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        
        output_layers = []
        for _ in range(add_layers, num_layers - 1):
            output_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        output_layers.append(nn.Linear(hidden_dim, 1))
        output_layers.append(nn.Sigmoid())
        
        self.coords_net  = nn.Sequential(*coords_layers)
        self.patches_net = nn.Sequential(*patches_layers)
        self.output_net  = nn.Sequential(*output_layers)
        
        # Weight decay params
        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{"params": self.coords_net.parameters(),  "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patches_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),  "weight_decay": weight_decay[2]}]
    
    def forward(self, coords: torch.Tensor, patches: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        coords_feat   = self.coords_net(self.encoding(coords))
        patches_feat  = self.patches_net(patches)
        backbone_feat = torch.cat((coords_feat, patches_feat), -1)
        backbone_out  = self.output_net(backbone_feat)
        return backbone_out


class iSIREN(nn.Module):
    
    def __init__(
        self,
        patch_dim   : int,
        hidden_dim  : int = 256,
        num_layers  : int = 4,
        add_layers  : int = 2,
        weight_decay: Any = None,
    ):
        super().__init__()
        # Coordinate & context branches
        coords_layers  = [nn.SineLayer(2, hidden_dim, is_first=True)]
        patches_layers = [nn.SineLayer(patch_dim,   hidden_dim, is_first=True)]
        for _ in range(1, add_layers - 2):
            coords_layers.append(nn.SineLayer(hidden_dim,  hidden_dim))
            patches_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        coords_layers.append(nn.SineLayer(hidden_dim,  hidden_dim // 2))
        patches_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        
        # Output branch
        output_layers = []
        for _ in range(add_layers, num_layers - 1):
            output_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        output_layers.append(nn.Linear(hidden_dim, 1))
        output_layers.append(nn.Sigmoid())
        
        self.coords_net  = nn.Sequential(*coords_layers)
        self.patches_net = nn.Sequential(*patches_layers)
        self.output_net  = nn.Sequential(*output_layers)
        
        # Auxiliary network for iterative feedback (small MLP with sin activations)
        aux_in_dim   = 2 + patch_dim + 1 + 1  # spatial (2) + patch + prev_g (1) + t (1)
        self.aux_net = nn.Sequential(
            nn.SineLayer(aux_in_dim, hidden_dim),
            nn.SineLayer(hidden_dim, hidden_dim)
        )
        # Projection layer for concatenated features
        proj_in_dim = hidden_dim + hidden_dim  # backbone_feat + aux_out
        self.proj   = nn.Sequential(
            nn.Linear(proj_in_dim, 1),
            nn.Sigmoid(),
        )
        
        # Weight decay params
        if weight_decay is None:
            weight_decay = [0.1, 0.0001, 0.001, 0.001, 0.001]  # Added for aux and proj
        self.params  = []
        self.params += [{"params": self.coords_net.parameters(),  "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patches_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),  "weight_decay": weight_decay[2]}]
        self.params += [{"params": self.aux_net.parameters(),     "weight_decay": weight_decay[3]}]
        self.params += [{"params": self.proj.parameters(),        "weight_decay": weight_decay[4]}]
    
    def forward(
        self,
        coords : torch.Tensor,
        patches: torch.Tensor,
        prev_g : torch.Tensor,
        t      : torch.Tensor,
        *args, **kwargs
    ) -> torch.Tensor:
        # Backbone processing
        coords_feat   = self.coords_net(coords)
        patches_feat  = self.patches_net(patches)
        backbone_feat = torch.cat((coords_feat, patches_feat), -1)
        backbone_out  = self.output_net(backbone_feat)

        # Auxiliary input: cat(spatial, patch, prev_g, t)
        t_tensor = torch.ones_like(prev_g) * t
        aux_in   = torch.cat((coords, patches, prev_g, t_tensor), dim=-1)
        aux_out  = self.aux_net(aux_in)
        
        # Concatenate and project
        concat_feat = torch.cat((backbone_feat, aux_out), -1)
        proj_out    = self.proj(concat_feat)
        
        # Final output: element-wise multiplication (dot product for scalars)
        return proj_out * backbone_out
    

# ----- DAM-SIREN -----
class DAMLayer(nn.Module):
    """Depth-Aware feature Modulation (DAM) based on the D-CNN principle. It weights
    the intensity features of the local context window by their depth similarity
    to the center pixel.
    
    Args:
        patch_dim: The window size of the local context window.
        alpha: Sensitivity parameter for depth similarity. Default: ``8.3``.
    """
    
    def __init__(self, patch_dim: int, alpha: float = 8.3):
        super().__init__()
        # Total elements in the context window.
        self.context_elements = patch_dim
        # Register alpha as a non-learnable buffer/parameter [3]
        self.register_buffer("alpha", torch.tensor(alpha))

    def forward(self, patches: torch.Tensor, depth: torch.Tensor, depth_center: torch.Tensor) -> torch.Tensor:
        """Performs depth modulation of the context features.

        Args:
            patches: Local intensity context features (Value component).
            depth: Local depth context features.
            depth_center: Depth value at the central pixel.
            
        Returns:
            Depth-modulated intensity context.
        """
        # 1. Calculate absolute depth difference |D(p_i) - D(p_j)|
        # D_center (B, 1) is broadcasted across the W*W context dimension of N_D.
        depth_diff = torch.abs(depth_center - depth)

        # 2. Calculate Depth Similarity Weights F_D = exp(-alpha * depth_diff)
        # These weights define how much each neighbor contributes to the output feature.
        F_D = torch.exp(-self.alpha * depth_diff)
        
        # 3. Modulate the Intensity Features: N(y'_V) = N(y_V) * F_D
        # Element-wise multiplication ensures that features from geometrically dissimilar
        # pixels (high depth_diff, low F_D) are attenuated.
        patches = patches * F_D
        return patches


class DAM_SIREN(nn.Module):
    """Depth-Aware Modulated SIREN."""
    
    def __init__(
        self,
        patch_dim   : int,
        hidden_dim  : int   = 256,
        num_layers  : int   = 4,
        add_layers  : int   = 2,
        alpha       : float = 8.3,
        weight_decay: Any   = None,
    ):
        super().__init__()
        self.patch_dim = patch_dim
        self.alpha     = alpha
        
        # Depth-aware modulation
        self.dam_layer = DAMLayer(patch_dim=patch_dim, alpha=alpha)
        
        # Positional encoding
        self.pos_encode = nn.PosEncodingNeRF(in_features=2, sidelength=hidden_dim)
        spatial_dim     = self.pos_encode.out_features
        
        # Coordinate & context branches
        coords_layers  = [nn.SineLayer(spatial_dim, hidden_dim, is_first=True)]
        patches_layers = [nn.SineLayer(patch_dim,   hidden_dim, is_first=True)]
        for _ in range(1, add_layers - 2):
            coords_layers.append(nn.SineLayer(hidden_dim,  hidden_dim))
            patches_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        coords_layers.append(nn.SineLayer(hidden_dim,  hidden_dim // 2))
        patches_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        
        # Output branch
        output_layers = []
        for _ in range(add_layers, num_layers - 1):
            output_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        output_layers.append(nn.SineLayer(hidden_dim, 1))
        output_layers.append(nn.Sigmoid())
        
        self.coords_net  = nn.Sequential(*coords_layers)
        self.patches_net = nn.Sequential(*patches_layers)
        self.output_net  = nn.Sequential(*output_layers)
        
        # Weight decay params
        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{"params": self.coords_net.parameters(),  "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patches_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),  "weight_decay": weight_decay[2]}]
        
    def forward(
        self,
        coords : torch.Tensor,
        patches: torch.Tensor,
        depth  : torch.Tensor = None,
    ) -> torch.Tensor:
        center_idx        = self.patch_dim // 2
        depth_center      = depth[:, :, center_idx:center_idx + 1]
        patches_modulated = self.dam_layer(patches, depth, depth_center)
        
        coords_feat       = self.coords_net(self.pos_encode(coords))
        patches_feat      = self.patches_net(patches_modulated)
        backbone_feat     = torch.cat((coords_feat, patches_feat), -1)
        backbone_out      = self.output_net(backbone_feat)
        return backbone_out


class DAM_iSIREN(nn.Module):
    
    def __init__(
        self,
        patch_dim   : int,
        hidden_dim  : int   = 256,
        num_layers  : int   = 4,
        add_layers  : int   = 2,
        alpha       : float = 8.3,
        weight_decay: Any   = None,
    ):
        super().__init__()
        self.patch_dim = patch_dim
        self.alpha     = alpha
        
        # Depth-aware modulation
        self.dam_layer = DAMLayer(patch_dim=patch_dim, alpha=alpha)
        
        # Positional encoding
        self.pos_encode = nn.PosEncodingNeRF(in_features=2, sidelength=hidden_dim)
        spatial_dim     = self.pos_encode.out_features
        
        # Coordinate & context branches
        coords_layers  = [nn.SineLayer(spatial_dim, hidden_dim, is_first=True)]
        patches_layers = [nn.SineLayer(patch_dim,   hidden_dim, is_first=True)]
        for _ in range(1, add_layers - 2):
            coords_layers.append(nn.SineLayer(hidden_dim,  hidden_dim))
            patches_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        coords_layers.append(nn.SineLayer(hidden_dim,  hidden_dim // 2))
        patches_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        
        # Output branch
        output_layers = []
        for _ in range(add_layers, num_layers - 1):
            output_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        output_layers.append(nn.Linear(hidden_dim, 1))
        output_layers.append(nn.Sigmoid())
        
        self.coords_net  = nn.Sequential(*coords_layers)
        self.patches_net = nn.Sequential(*patches_layers)
        self.output_net  = nn.Sequential(*output_layers)
        
        # Auxiliary network for iterative feedback (small MLP with sin activations)
        aux_in_dim   = 2 + patch_dim + 1 + 1 # spatial (2) + patch + prev_g (1) + t (1)
        self.aux_net = nn.Sequential(
            nn.SineLayer(aux_in_dim, hidden_dim),
            nn.SineLayer(hidden_dim, hidden_dim)
        )
        # Projection layer for concatenated features
        proj_in_dim = hidden_dim + hidden_dim  # backbone_feat + aux_out
        # self.proj   = nn.Linear(proj_in_dim, 1)  # Project to output dim for multiplication
        self.proj   = nn.Sequential(
            nn.Linear(proj_in_dim, 1),
            nn.Sigmoid(),
        )
        
        # Weight decay params
        if weight_decay is None:
            weight_decay = [0.1, 0.0001, 0.001, 0.001, 0.001]  # Added for aux and proj
        self.params  = []
        self.params += [{"params": self.coords_net.parameters(),  "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patches_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),  "weight_decay": weight_decay[2]}]
        self.params += [{"params": self.aux_net.parameters(),     "weight_decay": weight_decay[3]}]
        self.params += [{"params": self.proj.parameters(),        "weight_decay": weight_decay[4]}]
    
    def forward(
        self,
        coords : torch.Tensor,
        patches: torch.Tensor,
        prev_g : torch.Tensor,
        t      : torch.Tensor,
        depth  : torch.Tensor = None,
    ) -> torch.Tensor:
        center_idx   = self.patch_dim // 2
        depth_center = depth[:, :, center_idx:center_idx + 1]
        
        # Backbone processing
        patches_modulated = self.dam_layer(patches, depth, depth_center)
        coords_feat       = self.coords_net(self.pos_encode(coords))
        patches_feat      = self.patches_net(patches_modulated)
        backbone_feat     = torch.cat((coords_feat, patches_feat), -1)
        backbone_out      = self.output_net(backbone_feat)

        # Auxiliary input: cat(spatial, patch, prev_g, t)
        t_tensor = torch.ones_like(prev_g) * t
        aux_in   = torch.cat((coords, patches, prev_g, t_tensor), dim=-1)
        aux_out  = self.aux_net(aux_in)
        
        # Concatenate and project
        concat_feat = torch.cat((backbone_feat, aux_out), -1)
        proj_out    = self.proj(concat_feat)
        
        # Final output: element-wise multiplication (dot product for scalars)
        return proj_out * backbone_out


class D_SIREN(nn.Module):
    """Adopt the Depth-Aware technique from DINER into SIREN."""
    
    def __init__(
        self,
        patch_dim   : int,
        hidden_dim  : int = 256,
        num_layers  : int = 4,
        add_layers  : int = 2,
        weight_decay: Any = None,
    ):
        super().__init__()
        
        # Backbone networks
        coords_layers  = [nn.SineLayer(2 + 13, hidden_dim, is_first=True)]
        patches_layers = [nn.SineLayer(patch_dim, hidden_dim, is_first=True)]
        for _ in range(1, add_layers - 2):
            coords_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
            patches_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        coords_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        patches_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        
        output_layers = []
        for _ in range(add_layers, num_layers - 1):
            output_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        output_layers.append(nn.Linear(hidden_dim, 1))
        output_layers.append(nn.Sigmoid())
        
        self.coords_net  = nn.Sequential(*coords_layers)
        self.patches_net = nn.Sequential(*patches_layers)
        self.output_net  = nn.Sequential(*output_layers)
        
        # Weight decay params
        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{"params": self.coords_net.parameters(),  "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patches_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),  "weight_decay": weight_decay[2]}]
        
    def forward(
        self,
        coords : torch.Tensor,
        patches: torch.Tensor,
        depth  : torch.Tensor = None,
    ) -> torch.Tensor:
        depth    = depth.squeeze(0).squeeze(0)              # Shape: (H, W)
        # Delta_z: Use coords[:,:,1] as proxy for 'z' (y coord, normalized [0,1])
        delta_z  = depth.unsqueeze(-1) - coords[:, :, 1:2]  # Shape: (H, W, 1)
        # delta_z  = depth.unsqueeze(-1) - torch.mean(depth)  # Shape: (H, W, 1)
        pe_delta = positional_encoding(delta_z)             # Shape: (H, W, 13)
        
        # Concat to spatial
        coords_in = torch.cat((coords, pe_delta), dim=-1)  # Shape: (H, W, 15)
        
        coords_feat   = self.coords_net(coords_in)
        patches_feat  = self.patches_net(patches)  # (H, W, hidden_dim // 2)
        backbone_feat = torch.cat((coords_feat, patches_feat), -1)
        backbone_out  = self.output_net(backbone_feat)
        return backbone_out
