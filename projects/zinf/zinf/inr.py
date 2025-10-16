#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "Denoise",
    "IndiSIREN",
    "PE_SIREN",
    "SIREN",
]

from typing import Any

import torch

from mon.core import nn


# ----- Utils -----
def fd_2d(patch: torch.Tensor, depth: torch.Tensor = None, alpha: float = 8.3) -> torch.Tensor:
    if depth is None:
        return 1
    
    center_idx   = patch.shape[-1] // 2
    depth_center = depth[center_idx:center_idx + 1]  # Shape: [..., 1]
    depth_diff   = torch.abs(depth - depth_center)
    fd           = torch.exp(-alpha * depth_diff)
    return fd


def fd_3d(patch: torch.Tensor, depth: torch.Tensor = None, alpha: float = 8.3) -> torch.Tensor:
    if depth is None:
        return 1
    
    center_idx   = patch.shape[-1] // 2
    depth_center = depth[..., center_idx:center_idx + 1]  # Shape: [..., 1]
    depth_diff   = torch.abs(depth - depth_center)
    fd           = torch.exp(-alpha * depth_diff)
    return fd


# ----- SIREN -----
class SIREN(nn.Module):
    """The original SIREN backbone network for ZINF, similar to CoLIE model."""
    
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
        
    def forward(self, coords: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        coords_feat   = self.coords_net(coords)
        patches_feat  = self.patches_net(patches)
        backbone_feat = torch.cat((coords_feat, patches_feat), -1)
        backbone_out  = self.output_net(backbone_feat)
        return backbone_out


class PE_SIREN(nn.Module):
    """SIREN with positional encoding for coordinates."""
    
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
    
    def forward(self, coords: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        coords_feat   = self.coords_net(self.encoding(coords))
        patches_feat  = self.patches_net(patches)
        backbone_feat = torch.cat((coords_feat, patches_feat), -1)
        backbone_out  = self.output_net(backbone_feat)
        return backbone_out


class IndiSIREN(nn.Module):
    
    def __init__(
        self,
        patch_dim   : int,
        hidden_dim  : int = 256,
        num_layers  : int = 4,
        add_layers  : int = 2,
        weight_decay: Any = None,
    ):
        super().__init__()
        # Original backbone networks
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
    
    def forward(self, coords: torch.Tensor, patches: torch.Tensor, prev_g: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
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


class DepthSIREN(nn.Module):

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
        self.alpha = alpha
        
        coords_layers  = [nn.SineLayer(2, hidden_dim, is_first=True)]
        patches_layers = [nn.DepthAwareSineLayer(patch_dim, hidden_dim, 1, kernel_size=3, alpha=alpha, is_first=True)]
        for _ in range(1, add_layers - 2):
            coords_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
            patches_layers.append(nn.DepthAwareSineLayer(hidden_dim, hidden_dim,  1, kernel_size=3, alpha=alpha))
        coords_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        patches_layers.append(nn.DepthAwareSineLayer(hidden_dim, hidden_dim // 2, 1, kernel_size=3, alpha=alpha))
        
        output_layers = []
        for _ in range(add_layers, num_layers - 1):
            output_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        output_layers.append(nn.SineLayer(hidden_dim, 1))
        output_layers.append(nn.Sigmoid())
        
        self.coords_net  = nn.Sequential(*coords_layers)
        self.patches_net = nn.Sequential(*patches_layers)
        self.output_net  = nn.Sequential(*output_layers)
        
        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]
            
        self.params  = []
        self.params += [{"params": self.coords_net.parameters(),  "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patches_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),  "weight_decay": weight_decay[2]}]
        
    def forward(self, coords: torch.Tensor, patches: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        coords_feat  = self.coords_net(coords)
        patches_feat = patches
        for p_layer in self.patches_net:
            patches_feat = p_layer(patches_feat, depth)
        backbone_feat = torch.cat((coords_feat, patches_feat), -1)
        backbone_out  = self.output_net(backbone_feat)
        return backbone_out


# ----- Denoise -----
class Denoise(nn.Module):
    
    def __init__(self, in_channels: int, embed_channels: int = 48):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,    embed_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(embed_channels, embed_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(embed_channels, in_channels,    1)
        self.act   = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x
