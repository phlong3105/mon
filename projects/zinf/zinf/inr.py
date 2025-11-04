#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "InDi_SIREN",
    "InDi_SIREN_D",
    "InDi_SIREN_DA",
    "SIREN",
    "SIREN_D",
    "SIREN_DA",
]

from typing import Any

import torch

import mon.nn as nn
from mon.nn.inr.utils import *


# ----- Utils -----
class DaM(nn.Module):
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


# ----- SIREN -----
class SIREN(nn.Module):
    """The original SIREN network, similar to CoLIE model."""
    
    def __init__(
        self,
        window_size : int,
        patch_dim   : int,
        hidden_dim  : int = 256,
        num_layers  : int = 4,
        add_layers  : int = 2,
        weight_decay: Any = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.imgsz       = hidden_dim
        
        # Backbone networks
        spatial_layers = [nn.SineLayer(2, hidden_dim, is_first=True)]
        patch_layers   = [nn.SineLayer(patch_dim,   hidden_dim, is_first=True)]
        for _ in range(1, add_layers - 2):
            spatial_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
            patch_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        spatial_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        patch_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        
        output_layers = []
        for _ in range(add_layers, num_layers - 1):
            output_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        output_layers.append(nn.Linear(hidden_dim, 1))
        output_layers.append(nn.Sigmoid())
        
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.patch_net   = nn.Sequential(*patch_layers)
        self.output_net  = nn.Sequential(*output_layers)
        
        # Weight decay params
        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{"params": self.spatial_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patch_net.parameters(),   "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),  "weight_decay": weight_decay[2]}]
        
    def forward(self, coords: torch.Tensor, I: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        patch_I = create_patches(I, self.window_size)
        
        spatial_feat  = self.spatial_net(coords)
        patch_feat    = self.patch_net(patch_I)
        backbone_feat = torch.cat((spatial_feat, patch_feat), -1)
        backbone_out  = self.output_net(backbone_feat)
        backbone_out  = backbone_out.view(1, 1, self.imgsz, self.imgsz)  # Reshape to image size
        return backbone_out, None


class SIREN_D(nn.Module):
    """SIREN + Depth-Aware Modulation."""
    
    def __init__(
        self,
        window_size : int,
        patch_dim   : int,
        hidden_dim  : int   = 256,
        num_layers  : int   = 4,
        add_layers  : int   = 2,
        alpha       : float = 8.3,
        weight_decay: Any   = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.patch_dim   = patch_dim
        self.imgsz       = hidden_dim
        self.alpha       = alpha
        
        # Depth-aware modulation
        self.dam = DaM(patch_dim=patch_dim, alpha=alpha)
        
        # Positional encoding
        self.pos_encode = nn.PosEncodingNeRF(in_features=2, sidelength=hidden_dim)
        spatial_dim     = self.pos_encode.out_features
        
        # Coordinate & context branches
        spatial_layers = [nn.SineLayer(spatial_dim, hidden_dim, is_first=True)]
        patch_layers   = [nn.SineLayer(patch_dim,   hidden_dim, is_first=True)]
        for _ in range(1, add_layers - 2):
            spatial_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
            patch_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        spatial_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        patch_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        
        # Output branch
        output_layers = []
        for _ in range(add_layers, num_layers - 1):
            output_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        output_layers.append(nn.Linear(hidden_dim, 1))
        output_layers.append(nn.Sigmoid())
        
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.patch_net   = nn.Sequential(*patch_layers)
        self.output_net  = nn.Sequential(*output_layers)
        
        # Weight decay params
        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{"params": self.spatial_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patch_net.parameters(),   "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),  "weight_decay": weight_decay[2]}]
        
    def forward(self, coords : torch.Tensor, I: torch.Tensor, D: torch.Tensor = None) -> torch.Tensor:
        patch_I = create_patches(I, self.window_size)
        patch_D = create_patches(D, self.window_size) if D is not None else None
        
        center_idx      = self.patch_dim // 2
        D_center        = patch_D[:, :, center_idx:center_idx + 1]
        patch_modulated = self.dam(patch_I, patch_D, D_center)
        
        spatial_feat    = self.spatial_net(self.pos_encode(coords))
        patch_feat      = self.patch_net(patch_modulated)
        backbone_feat   = torch.cat((spatial_feat, patch_feat), -1)
        backbone_out    = self.output_net(backbone_feat)
        backbone_out    = backbone_out.view(1, 1, self.imgsz, self.imgsz)  # Reshape to image size
        return backbone_out, None


class SIREN_DA(nn.Module):
    """SIREN + Depth-Aware Convolution."""
    
    def __init__(
        self,
        window_size : int,
        patch_dim   : int,
        hidden_dim  : int   = 256,
        num_layers  : int   = 4,
        add_layers  : int   = 2,
        alpha       : float = 8.3,
        weight_decay: Any   = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.patch_dim   = patch_dim
        self.imgsz       = hidden_dim
        self.alpha       = alpha
        
        # Depth-aware convolution
        self.d_conv1 = nn.DepthAwareConv2d(1, patch_dim, kernel_size=3, padding=1, alpha=alpha)
        self.d_conv2 = nn.DepthAwareConv2d(patch_dim, patch_dim, kernel_size=3, padding=1, alpha=alpha)
        self.d_conv3 = nn.DepthAwareConv2d(patch_dim, patch_dim, kernel_size=3, padding=1, alpha=alpha)
        self.d_relu  = nn.ReLU()
        
        # Positional encoding
        self.pos_encode = nn.PosEncodingNeRF(in_features=2, sidelength=hidden_dim)
        spatial_dim     = self.pos_encode.out_features
        
        # Coordinate & context branches
        spatial_layers = [nn.SineLayer(spatial_dim, hidden_dim, is_first=True)]
        patch_layers   = [nn.SineLayer(patch_dim,   hidden_dim, is_first=True)]
        for _ in range(1, add_layers - 2):
            spatial_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
            patch_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        spatial_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        patch_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        
        # Output branch
        output_layers = []
        for _ in range(add_layers, num_layers - 1):
            output_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        output_layers.append(nn.Linear(hidden_dim, 1))
        output_layers.append(nn.Sigmoid())
        
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.patch_net   = nn.Sequential(*patch_layers)
        self.output_net  = nn.Sequential(*output_layers)
        
        # Weight decay params
        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{"params": self.spatial_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patch_net.parameters(),   "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),  "weight_decay": weight_decay[2]}]
    
    def forward(self, coords: torch.Tensor, I: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        IDA     = self.d_relu(self.d_conv1(I, D))
        IDA     = self.d_relu(self.d_conv2(IDA, D))
        IDA     = self.d_relu(self.d_conv3(IDA, D))
        patch_I = create_patches(IDA, self.window_size)
        
        # patch_I       = create_patches(I, self.window_size)
        spatial_feat  = self.spatial_net(self.pos_encode(coords))
        patch_feat    = self.patch_net(patch_I)
        backbone_feat = torch.cat((spatial_feat, patch_feat), -1)
        backbone_out  = self.output_net(backbone_feat)
        backbone_out  = backbone_out.view(1, 1, self.imgsz, self.imgsz)  # Reshape to image size
        return backbone_out, IDA


# ----- InDi -----
class InDi_SIREN(nn.Module):
    
    def __init__(
        self,
        window_size : int,
        patch_dim   : int,
        hidden_dim  : int = 256,
        num_layers  : int = 4,
        add_layers  : int = 2,
        weight_decay: Any = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.imgsz       = hidden_dim
        
        # Positional encoding
        self.pos_encode = nn.PosEncodingNeRF(in_features=2, sidelength=hidden_dim)
        spatial_dim     = self.pos_encode.out_features
        
        # Coordinate & context branches
        spatial_layers  = [nn.SineLayer(spatial_dim, hidden_dim, is_first=True)]
        patch_layers    = [nn.SineLayer(patch_dim,   hidden_dim, is_first=True)]
        for _ in range(1, add_layers - 2):
            spatial_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
            patch_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        spatial_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        patch_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        
        # Output branch
        output_layers = []
        for _ in range(add_layers, num_layers - 1):
            output_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        output_layers.append(nn.Linear(hidden_dim, 1))
        output_layers.append(nn.Sigmoid())
        
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.patch_net   = nn.Sequential(*patch_layers)
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
        self.params += [{"params": self.spatial_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patch_net.parameters(),   "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),  "weight_decay": weight_decay[2]}]
        self.params += [{"params": self.aux_net.parameters(),     "weight_decay": weight_decay[3]}]
        self.params += [{"params": self.proj.parameters(),        "weight_decay": weight_decay[4]}]
    
    def forward(
        self,
        coords: torch.Tensor,
        I     : torch.Tensor,
        prev_g: torch.Tensor,
        t     : torch.Tensor,
        *args, **kwargs
    ) -> torch.Tensor:
        patch_I = create_patches(I, self.window_size)
        
        # Backbone processing
        spatial_feat  = self.spatial_net(self.pos_encode(coords))
        patch_feat    = self.patch_net(patch_I)
        backbone_feat = torch.cat((spatial_feat, patch_feat), -1)
        backbone_out  = self.output_net(backbone_feat)

        # Auxiliary input: cat(spatial, patch, prev_g, t)
        t_tensor = torch.ones_like(prev_g) * t
        aux_in   = torch.cat((coords, patch_I, prev_g, t_tensor), dim=-1)
        aux_out  = self.aux_net(aux_in)
        
        # Concatenate and project
        concat_feat  = torch.cat((backbone_feat, aux_out), -1)
        proj_out     = self.proj(concat_feat)
        
        # Final output: element-wise multiplication (dot product for scalars)
        backbone_out = proj_out * backbone_out
        backbone_out = backbone_out.view(1, 1, self.imgsz, self.imgsz)  # Reshape to image size
        return backbone_out, None


class InDi_SIREN_D(nn.Module):
    
    def __init__(
        self,
        window_size : int,
        patch_dim   : int,
        hidden_dim  : int   = 256,
        num_layers  : int   = 4,
        add_layers  : int   = 2,
        alpha       : float = 8.3,
        weight_decay: Any   = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.patch_dim   = patch_dim
        self.imgsz       = hidden_dim
        self.alpha       = alpha
        
        # Depth-aware modulation
        self.dam = DaM(patch_dim=patch_dim, alpha=alpha)
        
        # Positional encoding
        self.pos_encode = nn.PosEncodingNeRF(in_features=2, sidelength=hidden_dim)
        spatial_dim     = self.pos_encode.out_features
        
        # Coordinate & context branches
        spatial_layers  = [nn.SineLayer(spatial_dim, hidden_dim, is_first=True)]
        patch_layers    = [nn.SineLayer(patch_dim,   hidden_dim, is_first=True)]
        for _ in range(1, add_layers - 2):
            spatial_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
            patch_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        spatial_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        patch_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        
        # Output branch
        output_layers = []
        for _ in range(add_layers, num_layers - 1):
            output_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        output_layers.append(nn.Linear(hidden_dim, 1))
        output_layers.append(nn.Sigmoid())
        
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.patch_net   = nn.Sequential(*patch_layers)
        self.output_net  = nn.Sequential(*output_layers)
        
        # Auxiliary network for iterative feedback (small MLP with sin activations)
        aux_in_dim   = 2 + patch_dim + 1 + 1 # spatial (2) + patch + prev_g (1) + t (1)
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
        self.params += [{"params": self.spatial_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patch_net.parameters(),   "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),  "weight_decay": weight_decay[2]}]
        self.params += [{"params": self.aux_net.parameters(),     "weight_decay": weight_decay[3]}]
        self.params += [{"params": self.proj.parameters(),        "weight_decay": weight_decay[4]}]
    
    def forward(
        self,
        coords: torch.Tensor,
        I     : torch.Tensor,
        D     : torch.Tensor,
        prev_g: torch.Tensor,
        t     : torch.Tensor,
    ) -> torch.Tensor:
        patch_I = create_patches(I, self.window_size)
        patch_D = create_patches(D, self.window_size) if D is not None else None
        
        center_idx      = self.patch_dim // 2
        D_center        = patch_D[:, :, center_idx:center_idx + 1]
        patch_modulated = self.dam(patch_I, patch_D, D_center)
        
        # Backbone processing
        coords_feat   = self.spatial_net(self.pos_encode(coords))
        patch_feat    = self.patch_net(patch_modulated)
        backbone_feat = torch.cat((coords_feat, patch_feat), -1)
        backbone_out  = self.output_net(backbone_feat)
        
        # Auxiliary input: cat(spatial, patch, prev_g, t)
        t_tensor = torch.ones_like(prev_g) * t
        aux_in   = torch.cat((coords, patch_I, prev_g, t_tensor), dim=-1)
        aux_out  = self.aux_net(aux_in)
        
        # Concatenate and project
        concat_feat = torch.cat((backbone_feat, aux_out), -1)
        proj_out    = self.proj(concat_feat)
        
        # Final output: element-wise multiplication (dot product for scalars)
        backbone_out = proj_out * backbone_out
        backbone_out = backbone_out.view(1, 1, self.imgsz, self.imgsz)  # Reshape to image size
        return backbone_out, None


class InDi_SIREN_DA(nn.Module):
    
    def __init__(
        self,
        window_size : int,
        patch_dim   : int,
        hidden_dim  : int   = 256,
        num_layers  : int   = 4,
        add_layers  : int   = 2,
        alpha       : float = 8.3,
        weight_decay: Any   = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.patch_dim   = patch_dim
        self.imgsz       = hidden_dim
        self.alpha       = alpha
        
        # Depth-aware convolution
        self.d_conv1 = nn.DepthAwareConv2d(1, patch_dim, kernel_size=3, padding=1, alpha=alpha)
        self.d_conv2 = nn.DepthAwareConv2d(patch_dim, patch_dim, kernel_size=3, padding=1, alpha=alpha)
        self.d_conv3 = nn.DepthAwareConv2d(patch_dim, patch_dim, kernel_size=3, padding=1, alpha=alpha)
        self.d_relu  = nn.ReLU()
        
        # Positional encoding
        self.pos_encode = nn.PosEncodingNeRF(in_features=2, sidelength=hidden_dim)
        spatial_dim     = self.pos_encode.out_features
        
        # Coordinate & context branches
        spatial_layers  = [nn.SineLayer(spatial_dim, hidden_dim, is_first=True)]
        patch_layers    = [nn.SineLayer(patch_dim,   hidden_dim, is_first=True)]
        for _ in range(1, add_layers - 2):
            spatial_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
            patch_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        spatial_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        patch_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        
        # Output branch
        output_layers = []
        for _ in range(add_layers, num_layers - 1):
            output_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        output_layers.append(nn.Linear(hidden_dim, 1))
        output_layers.append(nn.Sigmoid())
        
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.patch_net   = nn.Sequential(*patch_layers)
        self.output_net  = nn.Sequential(*output_layers)
        
        # Auxiliary network for iterative feedback (small MLP with sin activations)
        aux_in_dim   = 2 + patch_dim + 1 + 1 # spatial (2) + patch + prev_g (1) + t (1)
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
        self.params += [{"params": self.spatial_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patch_net.parameters(),   "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),  "weight_decay": weight_decay[2]}]
        self.params += [{"params": self.aux_net.parameters(),     "weight_decay": weight_decay[3]}]
        self.params += [{"params": self.proj.parameters(),        "weight_decay": weight_decay[4]}]
    
    def forward(
        self,
        coords: torch.Tensor,
        I     : torch.Tensor,
        D     : torch.Tensor,
        prev_g: torch.Tensor,
        t     : torch.Tensor,
    ) -> torch.Tensor:
        IDA     = self.d_relu(self.d_conv1(I, D))
        IDA     = self.d_relu(self.d_conv2(IDA, D))
        IDA     = self.d_relu(self.d_conv3(IDA, D))
        patch_I = create_patches(IDA, self.window_size)

        # Backbone processing
        coords_feat   = self.spatial_net(self.pos_encode(coords))
        patch_feat    = self.patch_net(patch_I)
        backbone_feat = torch.cat((coords_feat, patch_feat), -1)
        backbone_out  = self.output_net(backbone_feat)
        
        # Auxiliary input: cat(spatial, patch, prev_g, t)
        t_tensor = torch.ones_like(prev_g) * t
        aux_in   = torch.cat((coords, patch_I, prev_g, t_tensor), dim=-1)
        aux_out  = self.aux_net(aux_in)
        
        # Concatenate and project
        concat_feat = torch.cat((backbone_feat, aux_out), -1)
        proj_out    = self.proj(concat_feat)
        
        # Final output: element-wise multiplication (dot product for scalars)
        backbone_out = proj_out * backbone_out
        backbone_out = backbone_out.view(1, 1, self.imgsz, self.imgsz)  # Reshape to image size
        return backbone_out, IDA
