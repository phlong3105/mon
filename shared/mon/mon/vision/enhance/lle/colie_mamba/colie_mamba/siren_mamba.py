# This file provides a robust implementation of the MambaBlock for 2D vision tasks.
# To use this, you will need to install the official Mamba implementation:
#   pip install mamba-ssm causal-conv1d

__all__ = [
    "CoLIEMambaNet",
]

import numpy as np
import torch
import torch.nn as nn
from mamba_ssm import Mamba


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
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.w0, np.sqrt(6 / self.in_features) / self.w0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)


# ----- Mamba -----
class MambaBlock(nn.Module):
    """A Mamba block adapted for 2D vision tasks.
    
    The input is expected to be in the shape (B, C, H, W). The block flattens
    the spatial dimensions, processes the sequence with Mamba, and then reshapes
    it back to the original 2D format.
    
    Args:
        d_model: The feature dimension.
        d_state: The state dimension of the SSM. Default: ``16``.
        d_conv: The kernel size of the 1D convolution. Default: ``3``.
        expand: The expansion factor for the hidden dimension. Default: ``2``.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 3, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.norm    = nn.LayerNorm(d_model)

        # Mamba requires the input to be (B, L, D) where L is sequence length and D is dimension.
        self.mambas = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(4)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != self.d_model:
            raise ValueError(f"``x`` channel dimension {c} does not match Mamba ``d_model`` {self.d_model}.")
        
        # Pre-normalization
        x_norm = self.norm(x.flatten(2).transpose(1, 2)).transpose(1, 2).view(b, c, h, w)

        # --- Multi-Directional Scanning ---
        # 1. Forward scan (top-left to bottom-right)
        x_f  = x_norm.flatten(2).transpose(1, 2)  # (b, h*w, c)
        y_f  = self.mambas[0](x_f)

        # 2. Reverse scan (bottom-right to top-left)
        x_r  = torch.flip(x_f, dims=[1])
        y_r  = torch.flip(self.mambas[1](x_r), dims=[1])

        # 3. Transposed-Forward scan (column-wise)
        x_t  = x_norm.permute(0, 1, 3, 2).contiguous()  # (b, c, w, h)
        x_tf = x_t.flatten(2).transpose(1, 2)           # (b, w*h, c)
        y_tf = self.mambas[2](x_tf)
        y_t  = y_tf.transpose(1, 2).view(b, c, w, h).permute(0, 1, 3, 2).contiguous()  # Reshape back

        # 4. Transposed-Reverse Scan
        x_tr = torch.flip(x_tf, dims=[1])
        y_tr = torch.flip(self.mambas[3](x_tr), dims=[1])
        y_tr = y_tr.transpose(1, 2).view(b, c, w, h).permute(0, 1, 3, 2).contiguous()

        # Combine the outputs from all directions
        y_f  = y_f.transpose(1, 2).view(b, c, h, w)
        y_r  = y_r.transpose(1, 2).view(b, c, h, w)

        y    = y_f + y_r + y_t + y_tr
        return y


class MambaEncoderBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, mamba_args):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.mamba = MambaBlock(d_model=out_channels, **mamba_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x       = self.downsample(x)
        x_mamba = self.mamba(x)
        return x + x_mamba


class MambaDecoderBlock(nn.Module):
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, mamba_args):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.mamba    = MambaBlock(d_model=out_channels + skip_channels, **mamba_args)
        self.conv     = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x_mamba = self.mamba(x)
        x = x + x_mamba
        x = self.conv(x)
        return x


# --- Final Three-Branch Network ---
class CoLIEMambaNet(nn.Module):
    
    def __init__(
        self,
        hidden_dim   : int = 256,
        mamba_d_model: int = 64,
        mamba_d_state: int = 16,
        mamba_d_conv : int = 4,
        mamba_expand : int = 2,
    ):
        super().__init__()
        # patch_feat_dim = hidden_dim // 4
        spatial_feat_dim = hidden_dim // 4
        mamba_feat_dim   = hidden_dim // 2
        
        # Branch 1: CNN for local features with added BatchNorm
        # self.patch_net = nn.Sequential(
        #     nn.Conv2d(1, patch_feat_dim // 2, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(patch_feat_dim // 2),  # <-- Added Normalization
        #     nn.GELU(),
        #     nn.Conv2d(patch_feat_dim // 2, patch_feat_dim, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(patch_feat_dim),  # <-- Added Normalization
        #     nn.GELU()
        # )

        # Branch 2: SirenLayer-based MLP for spatial coordinates
        self.spatial_net = nn.Sequential(
            SIRENLayer(2, hidden_dim, is_first=True),
            SIRENLayer(hidden_dim, hidden_dim),
            SIRENLayer(hidden_dim, spatial_feat_dim, is_last=True)
        )
        
        # Branch 3: Hierarchical Mamba U-Net for global context
        mamba_args = {
            "d_state": mamba_d_state,
            "d_conv" : mamba_d_conv,
            "expand" : mamba_expand,
        }
        self.mamba_entry = nn.Sequential(
            nn.Conv2d(1, mamba_d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(mamba_d_model), nn.GELU()
        )
        self.encoder1   = MambaEncoderBlock(mamba_d_model, mamba_d_model * 2, mamba_args)
        self.encoder2   = MambaEncoderBlock(mamba_d_model * 2, mamba_d_model * 4, mamba_args)
        self.bottleneck = MambaBlock(d_model=mamba_d_model * 4, **mamba_args)
        self.decoder1   = MambaDecoderBlock(mamba_d_model * 4, mamba_d_model * 2, mamba_d_model * 2, mamba_args)
        self.decoder2   = MambaDecoderBlock(mamba_d_model * 2, mamba_d_model, mamba_d_model, mamba_args)
        self.mamba_exit = nn.Conv2d(mamba_d_model, mamba_feat_dim, kernel_size=1)
        
        # Final Fusion Network with added BatchNorm
        self.fusion_net = nn.Sequential(
            # nn.Conv2d(patch_feat_dim + spatial_feat_dim + mamba_feat_dim, hidden_dim, kernel_size=1),
            nn.Conv2d(spatial_feat_dim + mamba_feat_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),  # <-- Added Normalization
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),  # <-- Added Normalization
            nn.GELU()
        )
        
        self.output_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, image: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        # Flatten coordinates for the spatial_net (MLP)
        coords_flat = coords.view(-1, 2)
        H, W        = image.shape[-2], image.shape[-1]
        
        # Branch 1: Local Features
        # patch_feat_map = self.patch_net(image)
        
        # Branch 2: Spatial Features
        spatial_feat_flat = self.spatial_net(coords_flat)
        spatial_feat_map  = spatial_feat_flat.view(H, W, -1).permute(2, 0, 1).unsqueeze(0)
        
        # Branch 3: Global Features
        s0 = self.mamba_entry(image)
        s1 = self.encoder1(s0)
        s2 = self.encoder2(s1)
        bottleneck_out  = self.bottleneck(s2) + s2
        d1 = self.decoder1(bottleneck_out, s1)
        d2 = self.decoder2(d1, s0)
        global_feat_map = self.mamba_exit(d2)
        
        # Fusion
        # combined_feat_map = torch.cat((patch_feat_map, spatial_feat_map, global_feat_map), dim=1)
        combined_feat_map = torch.cat((spatial_feat_map, global_feat_map), dim=1)
        fused_map         = self.fusion_net(combined_feat_map)
        
        output = self.output_head(fused_map)
        return output
