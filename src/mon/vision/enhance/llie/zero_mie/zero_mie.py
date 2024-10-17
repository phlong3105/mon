#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Zero-MIE.

This module implement our idea: Zero-shot Multimodal Illumination Estimation for
Low-light Image Enhancement via Neural Implicit Representations.
"""

from __future__ import annotations

__all__ = [
    "ZeroMIE",
]

from abc import ABC
from typing import Any, Literal

import kornia
import torch
from fvcore.nn import parameter_count
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t

from mon import core, nn
from mon.globals import MODELS, Scheme, Task
from mon.vision import filtering
from mon.vision.enhance import base, utils

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]
LDA          = nn.LayeredFeatureAggregation

bilateral_ksize = (3, 3)
bilateral_color = 0.1
bilateral_space = (1.5, 1.5)


# region Loss

class Loss(nn.Loss):
    
    def __init__(
        self,
        exp_mean    : float = 0.9,
        weight_spa  : float = 1,
        weight_exp  : float = 10,
        weight_color: float = 5,
        weight_tv   : float = 1600,
        weight_depth: float = 1,
        weight_edge : float = 1,
        reduction   : Literal["none", "mean", "sum"] = "mean",
        *args, **kwargs
    ):
        super().__init__(reduction=reduction, *args, **kwargs)
        self.weight_spa   = weight_spa
        self.weight_exp   = weight_exp
        self.weight_color = weight_color
        self.weight_tv    = weight_tv
        self.weight_depth = weight_depth
        self.weight_edge  = weight_edge
        
        self.loss_spa   = nn.SpatialConsistencyLoss(8, reduction=reduction)
        self.loss_exp   = nn.ExposureControlLoss(16, exp_mean, reduction=reduction)
        self.loss_color = nn.ColorConstancyLoss(reduction=reduction)
        self.loss_tv    = nn.TotalVariationLoss(reduction=reduction)
        self.loss_depth = nn.MultiscaleDepthConsistencyLoss(reduction=reduction)
        self.loss_edge  = nn.EdgeAwareDepthConsistencyLoss(reduction=reduction)
        
    def forward(
        self,
        image          : torch.Tensor,
        image_lr       : torch.Tensor,
        illumination_lr: torch.Tensor,
        enhanced       : torch.Tensor,
        depth_lr       : torch.Tensor = None,
    ) -> torch.Tensor:
        loss_spa   = self.loss_spa(input=enhanced, target=image)
        loss_exp   = self.loss_exp(input=enhanced)
        loss_color = self.loss_color(input=enhanced)
        loss_tv    = self.loss_tv(input=illumination_lr)
        if depth_lr is not None:
            loss_depth = self.loss_depth(image_lr, depth_lr)
            loss_edge  = self.loss_edge(image_lr, depth_lr)
        else:
            loss_depth = 0
            loss_edge  = 0
        loss = (
              self.weight_spa   * loss_spa
            + self.weight_exp   * loss_exp
            + self.weight_color * loss_color
            + self.weight_tv    * loss_tv
            + self.weight_depth * loss_depth
            + self.weight_edge  * loss_edge
        )
        return loss


class LossHSV(nn.Loss):
    
    def __init__(
        self,
        exp_mean    : float = 0.1,
        weight_spa  : float = 1,
        weight_tv   : float = 20,
        weight_exp  : float = 8,
        weight_spar : float = 5,
        weight_depth: float = 1,
        weight_edge : float = 1,
        weight_color: float = 1,
        reduction: Literal["none", "mean", "sum"] = "mean",
        *args, **kwargs
    ):
        super().__init__(reduction=reduction, *args, **kwargs)
        self.weight_spa   = weight_spa
        self.weight_tv    = weight_tv
        self.weight_exp   = weight_exp
        self.weight_spar  = weight_spar
        self.weight_depth = weight_depth
        self.weight_edge  = weight_edge
        self.weight_color = weight_color
        
        self.loss_exp   = nn.ExposureValueControlLoss(16, exp_mean, reduction=reduction)
        self.loss_tv    = nn.TotalVariationLoss(reduction=reduction)
        self.loss_depth = nn.MultiscaleDepthConsistencyLoss(reduction=reduction)
        self.loss_edge  = nn.EdgeAwareDepthConsistencyLoss(reduction=reduction)
        self.loss_color = nn.ColorConstancyLoss(reduction=reduction)
        
    def forward(
        self,
        image          : torch.Tensor,
        image_lr       : torch.Tensor,
        illumination_lr: torch.Tensor,
        enhanced       : torch.Tensor,
        depth_lr       : torch.Tensor = None,
    ) -> torch.Tensor:
        loss_spa   = torch.mean(torch.abs(torch.pow(illumination_lr - image_lr, 2)))
        loss_tv    = self.loss_tv(illumination_lr)
        loss_exp   = torch.mean(self.loss_exp(illumination_lr))
        loss_spar  = torch.mean(enhanced)
        loss_color = self.loss_color(enhanced)
        if depth_lr is not None:
            loss_depth = self.loss_depth(image_lr, depth_lr)
            loss_edge  = self.loss_edge(image_lr, depth_lr)
        else:
            loss_depth = 0
            loss_edge  = 0
        loss = (
              self.weight_spa   * loss_spa
            + self.weight_tv    * loss_tv
            + self.weight_exp   * loss_exp
            + self.weight_spar  * loss_spar
            + self.weight_depth * loss_depth
            + self.weight_edge  * loss_edge
            + self.weight_color * loss_color
        )
        return loss

# endregion


# region Modules

class FiLM(nn.Module):
    
    def __init__(self, in_channels):
        super().__init__()
        # Two linear layers to generate scale and shift from depth information
        self.fc_scale = nn.Linear(1, in_channels)  # Scale from depth
        self.fc_shift = nn.Linear(1, in_channels)  # Shift from depth

    def forward(self, x, depth_map):
        batch_size = depth_map.size(0)  # This should be 1 for your case
        height, width, _ = x.size()  # height=256, width=256, in_channels=64

        # Flatten depth_map to [B, H*W, 1]
        depth_map_flat = depth_map.view(batch_size, -1, 1)  # Shape: [1, 256*256, 1]

        # Calculate scale and shift based on depth
        scale = self.fc_scale(depth_map_flat)  # Shape: [1, H*W, in_channels]
        shift = self.fc_shift(depth_map_flat)  # Shape: [1, H*W, in_channels]

        # Reshape x to [B, H*W, in_channels]
        x = x.view(-1, x.size(2))  # Flatten x to shape [H*W, in_channels] => [65536, 64]

        # Apply FiLM to input features
        x = (x * scale.view(-1, x.size(1))) + shift.view(-1, x.size(1))  # Apply modulation

        # Reshape back to original dimensions [H, W, C]
        return x.view(height, width, -1)  # Shape: [H, W, in_channels]


class CrossAttentionLayer(nn.Module):
    
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        query   = query.permute(1, 0, 2)  # [seq_len, batch_size, dim]
        key     = key.permute(1, 0, 2)
        value   = value.permute(1, 0, 2)
        attn, _ = self.attn(query, key, value)
        return attn.permute(1, 0, 2)  # Back to [batch_size, seq_len, dim]


class INF(nn.Module, ABC):
    
    def interpolate_image(self, image: torch.Tensor) -> torch.Tensor:
        """Reshapes the image based on new resolution."""
        return F.interpolate(image, size=(self.down_size, self.down_size), mode="bicubic")
    
    @staticmethod
    def filter_up(
        x_lr  : torch.Tensor,
        y_lr  : torch.Tensor,
        x_hr  : torch.Tensor,
        radius: int = 1
    ):
        """Applies the guided filter to upscale the predicted image. """
        gf   = filtering.FastGuidedFilter(radius=radius)
        y_hr = gf(x_lr, y_lr, x_hr)
        y_hr = torch.clip(y_hr, 0, 1)
        return y_hr
    
    @staticmethod
    def replace_v_component(image_hsv: torch.Tensor, v_new: torch.Tensor) -> torch.Tensor:
        """Replaces the `V` component of an HSV image `[1, 3, H, W]`."""
        image_hsv[:, -1, :, :] = v_new
        return image_hsv
    
    @staticmethod
    def replace_i_component(image_hvi: torch.Tensor, i_new: torch.Tensor) -> torch.Tensor:
        """Replaces the `I` component of an HVI image `[1, 3, H, W]`."""
        image_hvi[:, 2, :, :] = i_new
        return image_hvi


class INF_RGB(INF):
    
    def __init__(
        self,
        window_size : int  = 1,
        num_layers  : int  = 2,
        add_layers  : int  = 1,
        down_size   : int  = 256,
        hidden_dim  : int  = 256,
        weight_decay: list[float] = [0.1, 0.0001, 0.001],
        use_denoise : bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.down_size   = down_size
        self.use_denoise = use_denoise
        
        self.patch_net   = nn.PatchINF(window_size, hidden_dim // 2, down_size, num_layers, 30, 6, weight_decay[1])
        self.film        = FiLM(hidden_dim // 2)
        self.patch_d_net = nn.PatchINF(window_size, hidden_dim // 2, down_size, num_layers, 30, 6, weight_decay[1])
        self.patch_e_net = nn.PatchINF(window_size, hidden_dim // 2, down_size, num_layers, 30, 6, weight_decay[1])
        self.spatial_net = nn.SpatialINF(hidden_dim // 2, down_size, num_layers, 30, 6, weight_decay[0])
        self.output_net  = nn.OutputINF(hidden_dim,   3, add_layers, 30, 6, weight_decay[2])
        self.cross_atten = CrossAttentionLayer(dim=hidden_dim // 2, num_heads=4)
        self.dba         = nn.BoundaryAwarePrior(eps=0.05, normalized=False)
        
    def forward(self, image: torch.Tensor,  depth: torch.Tensor = None) -> torch.Tensor:
        # Prepare input
        if depth is None:
            depth = core.rgb_to_grayscale(image)
        edge = self.dba(depth)
        # Mapping
        image_lr, patch_rgb = self.patch_net(image)
        depth_lr,   patch_d = self.patch_d_net(depth)
        edge_lr,    patch_e = self.patch_e_net(edge)
        patch_rgb   = self.film(patch_rgb, depth_lr)
        spatial     = self.spatial_net(image)
        atten       = self.cross_atten(patch_rgb, patch_d, patch_e)
        illu_res_lr = self.output_net(torch.cat([atten, spatial], -1))
        illu_res_lr = illu_res_lr.view(1, 3, self.down_size, self.down_size)
        # Enhancement
        illu_lr        = illu_res_lr + image_lr
        image_fixed_lr = image_lr / (illu_lr + 1e-8)
        if self.use_denoise:
            image_fixed_lr = kornia.filters.bilateral_blur(image_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
        enhanced = self.filter_up(image_lr, image_fixed_lr, image)
        enhanced = enhanced / torch.max(enhanced)
        # Return
        return {
            "image"      : image,
            "image_lr"   : image_lr,
            "depth"      : depth,
            "depth_lr"   : depth_lr,
            "edge"       : edge,
            "edge_lr"    : edge_lr,
            "illu_res_lr": illu_res_lr,
            "illu_lr"    : illu_lr,
            "enhanced_lr": image_fixed_lr,
            "enhanced"   : enhanced,
        }


class INF_RGB_D(INF):
    
    def __init__(
        self,
        window_size : int  = 1,
        num_layers  : int  = 2,
        add_layers  : int  = 1,
        down_size   : int  = 256,
        hidden_dim  : int  = 256,
        weight_decay: list[float] = [0.1, 0.0001, 0.001],
        use_denoise : bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.down_size   = down_size
        self.use_denoise = use_denoise
        
        self.patch_net   = nn.PatchINF(window_size, hidden_dim // 4, down_size, num_layers, 30, 6, weight_decay[1])
        self.film        = FiLM(hidden_dim // 4)
        self.patch_d_net = nn.PatchINF(window_size, hidden_dim // 4, down_size, num_layers, 30, 6, weight_decay[1])
        self.patch_e_net = nn.PatchINF(window_size, hidden_dim // 4, down_size, num_layers, 30, 6, weight_decay[1])
        self.spatial_net = nn.SpatialINF(hidden_dim // 4, down_size, num_layers, 30, 6, weight_decay[0])
        self.output_net  = nn.OutputINF(hidden_dim,   3, add_layers, 30, 6, weight_decay[2])
        self.cross_atten = CrossAttentionLayer(dim=hidden_dim // 4, num_heads=4)
        self.dba         = nn.BoundaryAwarePrior(eps=0.05, normalized=False)
        
    def forward(self, image: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        # Prepare input
        if depth is None:
            depth = core.rgb_to_grayscale(image)
        edge = self.dba(depth)
        # Mapping
        image_lr, patch_rgb = self.patch_net(image)
        depth_lr,   patch_d = self.patch_d_net(depth)
        edge_lr,    patch_e = self.patch_e_net(edge)
        patch_rgb   = self.film(patch_rgb, depth_lr)
        spatial     = self.spatial_net(image)
        atten       = self.cross_atten(patch_rgb, patch_d, patch_e)
        illu_res_lr = self.output_net(torch.cat([atten, patch_e, patch_d, spatial], -1))
        illu_res_lr = illu_res_lr.view(1, 3, self.down_size, self.down_size)
        # Enhancement
        illu_lr        = illu_res_lr + image_lr
        image_fixed_lr = image_lr / (illu_lr + 1e-8)
        if self.use_denoise:
            image_fixed_lr = kornia.filters.bilateral_blur(image_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
        enhanced = self.filter_up(image_lr, image_fixed_lr, image)
        enhanced = enhanced / torch.max(enhanced)
        # Return
        return {
            "image"      : image,
            "image_lr"   : image_lr,
            "depth"      : depth,
            "depth_lr"   : depth_lr,
            "edge"       : edge,
            "edge_lr"    : edge_lr,
            "illu_res_lr": illu_res_lr,
            "illu_lr"    : illu_lr,
            "enhanced_lr": image_fixed_lr,
            "enhanced"   : enhanced,
        }


class INF_R_G_B_D(INF):
    
    def __init__(
        self,
        window_size : int  = 1,
        num_layers  : int  = 2,
        add_layers  : int  = 1,
        down_size   : int  = 256,
        hidden_dim  : int  = 256,
        weight_decay: list[float] = [0.1, 0.0001, 0.001],
        use_denoise : bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.down_size   = down_size
        self.use_denoise = use_denoise
        
        self.patch_r_net = nn.PatchINF(window_size, hidden_dim // 4, down_size, num_layers, 30, 6, weight_decay[1])
        self.film_r      = FiLM(hidden_dim // 4)
        self.atten_r     = CrossAttentionLayer(dim=hidden_dim // 4, num_heads=4)
        self.patch_g_net = nn.PatchINF(window_size, hidden_dim // 4, down_size, num_layers, 30, 6, weight_decay[1])
        self.film_g      = FiLM(hidden_dim // 4)
        self.atten_g     = CrossAttentionLayer(dim=hidden_dim // 4, num_heads=4)
        self.patch_b_net = nn.PatchINF(window_size, hidden_dim // 4, down_size, num_layers, 30, 6, weight_decay[1])
        self.film_b      = FiLM(hidden_dim // 4)
        self.atten_b     = CrossAttentionLayer(dim=hidden_dim // 4, num_heads=4)
        self.patch_d_net = nn.PatchINF(window_size, hidden_dim // 4, down_size, num_layers, 30, 6, weight_decay[1])
        self.patch_e_net = nn.PatchINF(window_size, hidden_dim // 4, down_size, num_layers, 30, 6, weight_decay[1])
        self.spatial_net = nn.SpatialINF(hidden_dim // 4, down_size, num_layers, 30, 6, weight_decay[0])
        self.output_net  = nn.OutputINF(hidden_dim,   3, add_layers, 30, 6, weight_decay[2])
        self.dba         = nn.BoundaryAwarePrior(eps=0.05, normalized=False)
        
    def forward(self, image: torch.Tensor,  depth: torch.Tensor = None) -> torch.Tensor:
        # Prepare input
        if depth is None:
            depth = core.rgb_to_grayscale(image)
        edge = self.dba(depth)
        # Mapping
        image_r   = image[:, 0:1, :, :]
        image_g   = image[:, 1:2, :, :]
        image_b   = image[:, 2:3, :, :]
        image_lr  = self.interpolate_image(image)
        image_r_lr, patch_r = self.patch_r_net(image_r)
        image_g_lr, patch_g = self.patch_g_net(image_g)
        image_b_lr, patch_b = self.patch_b_net(image_b)
        depth_lr,   patch_d = self.patch_d_net(depth)
        edge_lr,    patch_e = self.patch_e_net(edge)
        patch_r     = self.film_r(patch_r, depth_lr)
        patch_g     = self.film_g(patch_g, depth_lr)
        patch_b     = self.film_b(patch_b, depth_lr)
        spatial     = self.spatial_net(image)
        atten_r     = self.atten_r(patch_r, patch_d, patch_e)
        atten_g     = self.atten_g(patch_g, patch_d, patch_e)
        atten_b     = self.atten_b(patch_b, patch_d, patch_e)
        illu_res_lr = self.output_net(torch.cat([atten_r, atten_g, atten_b, spatial], -1))
        illu_res_lr = illu_res_lr.view(1, 3, self.down_size, self.down_size)
        # Enhancement
        illu_lr        = illu_res_lr + image_lr
        image_fixed_lr = image_lr / (illu_lr + 1e-8)
        if self.use_denoise:
            image_fixed_lr = kornia.filters.bilateral_blur(image_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
        enhanced = self.filter_up(image_lr, image_fixed_lr, image)
        enhanced = enhanced / torch.max(enhanced)
        # Return
        return {
            "image"      : image,
            "image_lr"   : image_lr,
            "depth"      : depth,
            "depth_lr"   : depth_lr,
            "edge"       : edge,
            "edge_lr"    : edge_lr,
            "illu_res_lr": illu_res_lr,
            "illu_lr"    : illu_lr,
            "enhanced_lr": image_fixed_lr,
            "enhanced"   : enhanced,
        }


class INF_HSV_V(INF):
    
    def __init__(
        self,
        window_size : int  = 1,
        num_layers  : int  = 2,
        add_layers  : int  = 1,
        down_size   : int  = 256,
        hidden_dim  : int  = 256,
        weight_decay: list[float] = [0.1, 0.0001, 0.001],
        use_denoise : bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.down_size   = down_size
        self.use_denoise = use_denoise
        
        self.patch_v_net = nn.PatchINF(window_size, hidden_dim // 2, down_size, num_layers, 30, 6, weight_decay[1])
        self.film        = FiLM(hidden_dim // 2)
        self.patch_d_net = nn.PatchINF(window_size, hidden_dim // 2, down_size, num_layers, 30, 6, weight_decay[1])
        self.patch_e_net = nn.PatchINF(window_size, hidden_dim // 2, down_size, num_layers, 30, 6, weight_decay[1])
        self.spatial_net = nn.SpatialINF(hidden_dim // 2, down_size, num_layers, 30, 6, weight_decay[0])
        self.output_net  = nn.OutputINF(hidden_dim,   1, add_layers, 30, 6, weight_decay[2])
        self.cross_atten = CrossAttentionLayer(dim=hidden_dim // 2, num_heads=4)
        self.dba         = nn.BoundaryAwarePrior(eps=0.05, normalized=False)
        
    def forward(self, image: torch.Tensor,  depth: torch.Tensor = None) -> torch.Tensor:
        # Prepare input
        if depth is None:
            depth = core.rgb_to_grayscale(image)
        edge = self.dba(depth)
        # Mapping
        image_hsv = core.rgb_to_hsv(image)
        image_v   = core.rgb_to_v(image)
        image_v_lr, patch_v = self.patch_v_net(image_v)
        depth_lr,   patch_d = self.patch_d_net(depth)
        edge_lr,    patch_e = self.patch_e_net(edge)
        patch_v       = self.film(patch_v, depth_lr)
        spatial       = self.spatial_net(image)
        atten_v       = self.cross_atten(patch_v, patch_d, patch_e)
        illu_res_v_lr = self.output_net(torch.cat([atten_v, spatial], -1))
        illu_res_v_lr = illu_res_v_lr.view(1, 1, self.down_size, self.down_size)
        # Enhancement
        illu_v_lr        = illu_res_v_lr + image_v_lr
        image_v_fixed_lr = image_v_lr / (illu_v_lr + 1e-8)
        if self.use_denoise:
            image_v_fixed_lr = kornia.filters.bilateral_blur(image_v_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
        image_v_fixed    = self.filter_up(image_v_lr, image_v_fixed_lr, image_v)
        image_hsv_fixed  = self.replace_v_component(image_hsv, image_v_fixed)
        enhanced         = core.hsv_to_rgb(image_hsv_fixed)
        enhanced         = enhanced / torch.max(enhanced)
        # Return
        return {
            "image"      : image,
            "image_lr"   : image_v_lr,
            "depth"      : depth,
            "depth_lr"   : depth_lr,
            "edge"       : edge,
            "edge_lr"    : edge_lr,
            "illu_res_lr": illu_res_v_lr,
            "illu_lr"    : illu_v_lr,
            "enhance_lr" : image_v_fixed_lr,
            "enhanced"   : enhanced,
        }


class INF_HSV_V_D(INF):
    
    def __init__(
        self,
        window_size : int  = 1,
        num_layers  : int  = 2,
        add_layers  : int  = 1,
        down_size   : int  = 256,
        hidden_dim  : int  = 256,
        weight_decay: list[float] = [0.1, 0.0001, 0.001],
        use_denoise : bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.down_size   = down_size
        self.use_denoise = use_denoise
        
        self.patch_v_net = nn.PatchINF(window_size, hidden_dim // 4, down_size, num_layers, 30, 6, weight_decay[1])
        self.film        = FiLM(hidden_dim // 4)
        self.patch_d_net = nn.PatchINF(window_size, hidden_dim // 4, down_size, num_layers, 30, 6, weight_decay[1])
        self.patch_e_net = nn.PatchINF(window_size, hidden_dim // 4, down_size, num_layers, 30, 6, weight_decay[1])
        self.spatial_net = nn.SpatialINF(hidden_dim // 4, down_size, num_layers, 30, 6, weight_decay[0])
        self.output_net  = nn.OutputINF(hidden_dim,   1, add_layers, 30, 6, weight_decay[2])
        self.cross_atten = CrossAttentionLayer(dim=hidden_dim // 4, num_heads=4)
        self.dba         = nn.BoundaryAwarePrior(eps=0.05, normalized=False)
        
    def forward(self, image: torch.Tensor,  depth: torch.Tensor = None) -> torch.Tensor:
        # Prepare input
        if depth is None:
            depth = core.rgb_to_grayscale(image)
        edge = self.dba(depth)
        # Mapping
        image_hsv = core.rgb_to_hsv(image)
        image_v   = core.rgb_to_v(image)
        image_v_lr, patch_v = self.patch_v_net(image_v)
        depth_lr,   patch_d = self.patch_d_net(depth)
        edge_lr,    patch_e = self.patch_e_net(edge)
        patch_v       = self.film(patch_v, depth_lr)
        spatial       = self.spatial_net(image)
        atten_v       = self.cross_atten(patch_v, patch_d, patch_e)
        illu_res_v_lr = self.output_net(torch.cat([atten_v, patch_d, patch_e, spatial], -1))
        illu_res_v_lr = illu_res_v_lr.view(1, 1, self.down_size, self.down_size)
        # Enhancement
        illu_v_lr        = illu_res_v_lr + image_v_lr
        image_v_fixed_lr = image_v_lr / (illu_v_lr + 1e-8)
        if self.use_denoise:
            image_v_fixed_lr = kornia.filters.bilateral_blur(image_v_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
        image_v_fixed    = self.filter_up(image_v_lr, image_v_fixed_lr, image_v)
        image_hsv_fixed  = self.replace_v_component(image_hsv, image_v_fixed)
        enhanced         = core.hsv_to_rgb(image_hsv_fixed)
        enhanced         = enhanced / torch.max(enhanced)
        # Return
        return {
            "image"      : image,
            "image_lr"   : image_v_lr,
            "depth"      : depth,
            "depth_lr"   : depth_lr,
            "edge"       : edge,
            "edge_lr"    : edge_lr,
            "illu_res_lr": illu_res_v_lr,
            "illu_lr"    : illu_v_lr,
            "enhance_lr" : image_v_fixed_lr,
            "enhanced"   : enhanced,
        }

# endregion


# region Model

@MODELS.register(name="zero_mie", arch="zero_mie")
class ZeroMIE(base.ImageEnhancementModel):
    
    model_dir: core.Path    = current_dir
    arch     : str          = "zero_mie"
    tasks    : list[Task]   = [Task.LLIE]
    schemes  : list[Scheme] = [Scheme.ZERO_REFERENCE, Scheme.INSTANCE]
    zoo      : dict         = {}
    
    def __init__(
        self,
        name        : str   = "zero_mie",
        window_size : int   = 1,
        num_layers  : int   = 2,
        add_layers  : int   = 1,
        down_size   : int   = 256,
        hidden_dim  : int   = 256,
        weight_decay: list[float] = [0.1, 0.0001, 0.001],
        color_space : Literal["rgb", "rgb_d", "r_g_b", "r_g_b_d", "hsv_v", "hsv_v_d"] = "rgb",
        use_denoise : bool  = False,
        use_pse     : bool  = False,
        number_refs : int   = 2,
        weight_enh  : float = 5,
        exp_mean    : float = 0.6,
        weight_spa  : float = 1,
        weight_exp  : float = 10,
        weight_color: float = 5,
        weight_tv   : float = 1600,
        weight_depth: float = 1,
        weight_edge : float = 1,
        weights     : Any   = None,
        *args, **kwargs
    ):
        super().__init__(name=name, weights=weights, *args, **kwargs)
        self.color_space = color_space
        self.use_pse     = use_pse
        self.number_refs = number_refs
        self.weight_enh  = weight_enh
        
        if color_space == "rgb":
            self.model = INF_RGB(
                window_size  = window_size,
                num_layers   = num_layers,
                add_layers   = add_layers,
                down_size    = down_size,
                hidden_dim   = hidden_dim,
                weight_decay = weight_decay,
                use_denoise  = use_denoise,
            )
        elif color_space == "rgb_d":
            self.model = INF_RGB_D(
                window_size  = window_size,
                num_layers   = num_layers,
                add_layers   = add_layers,
                down_size    = down_size,
                hidden_dim   = hidden_dim,
                weight_decay = weight_decay,
                use_denoise  = use_denoise,
            )
        elif color_space == "r_g_b_d":
            self.model = INF_R_G_B_D(
                window_size  = window_size,
                num_layers   = num_layers,
                add_layers   = add_layers,
                down_size    = down_size,
                hidden_dim   = hidden_dim,
                weight_decay = weight_decay,
                use_denoise  = use_denoise,
            )
        elif color_space == "hsv_v":
            self.model = INF_HSV_V(
                window_size  = window_size,
                num_layers   = num_layers,
                add_layers   = add_layers,
                down_size    = down_size,
                hidden_dim   = hidden_dim,
                weight_decay = weight_decay,
                use_denoise  = use_denoise,
            )
        elif color_space == "hsv_v_d":
            self.model = INF_HSV_V_D(
                window_size  = window_size,
                num_layers   = num_layers,
                add_layers   = add_layers,
                down_size    = down_size,
                hidden_dim   = hidden_dim,
                weight_decay = weight_decay,
                use_denoise  = use_denoise,
            )
        
        self.pseudo_gt_generator = utils.PseudoGTGenerator(
            number_refs   = self.number_refs,
            gamma_upper   = -2,
            gamma_lower   =  3,
            exposed_level =  0.5,
            pool_size     =  25,
        )
        self.saved_input     = None
        self.saved_pseudo_gt = None
        
        # Loss
        if color_space in ["hsv_v", "hsv_v_d"]:
            self.loss = LossHSV(exp_mean=exp_mean)
        else:
            self.loss = Loss(
                exp_mean     = exp_mean,
                weight_spa   = weight_spa,
                weight_exp   = weight_exp,
                weight_color = weight_color,
                weight_tv    = weight_tv,
                weight_depth = weight_depth,
                weight_edge  = weight_edge,
            )
        self.loss_recon = nn.MSELoss()
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        self.initial_state_dict = self.state_dict()
    
    def init_weights(self, m: nn.Module):
        pass
    
    def compute_efficiency_score(
        self,
        image_size: _size_2_t = 512,
        channels  : int       = 3,
        runs      : int       = 1000,
        verbose   : bool      = False,
    ) -> tuple[float, float, float]:
        """Compute the efficiency score of the model, including FLOPs, number
        of parameters, and runtime.
        """
        # Define input tensor
        h, w      = core.get_image_size(image_size)
        datapoint = {
            "image": torch.rand(1, channels, h, w).to(self.device),
            "depth": torch.rand(1,        1, h, w).to(self.device)
        }
        
        # Get FLOPs and Params
        flops, params = core.custom_profile(self, inputs=datapoint, verbose=verbose)
        # flops         = FlopCountAnalysis(self, datapoint).total() if flops == 0 else flops
        params        = self.params                if hasattr(self, "params") and params == 0 else params
        params        = parameter_count(self)      if hasattr(self, "params")  else params
        params        = sum(list(params.values())) if isinstance(params, dict) else params
        
        # Get time
        timer = core.Timer()
        for i in range(runs):
            timer.tick()
            _ = self(datapoint)
            timer.tock()
        avg_time = timer.avg_time
        
        # Print
        if verbose:
            console.log(f"FLOPs (G) : {flops:.4f}")
            console.log(f"Params (M): {params:.4f}")
            console.log(f"Time (s)  : {avg_time:.17f}")
        
        return flops, params, avg_time
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        self.assert_datapoint(datapoint)
        if self.use_pse:
            # Saving n-th input and n-th pseudo gt
            nth_input     = datapoint
            nth_output    = self.forward(datapoint=datapoint, *args, **kwargs)
            nth_image     = nth_output["image"]
            nth_enhanced  = nth_output["enhanced"].clone().detach()
            nth_pseudo_gt = self.pseudo_gt_generator(nth_image, nth_enhanced)
            if self.saved_input is not None:
                # Getting (n - 1)th input and (n - 1)-th pseudo gt -> calculate loss -> update model weight (handled automatically by pytorch lightning)
                outputs         = self.forward(datapoint=self.saved_input, *args, **kwargs)
                image           = outputs["image"]
                image_lr        = outputs["image_lr"]
                illu_lr         = outputs["illu_lr"]
                enhanced        = outputs["enhanced"]
                depth_lr        = outputs["depth_lr"]
                pseudo_gt       = self.saved_pseudo_gt
                loss_recon      = self.loss_recon(enhanced, pseudo_gt)
                loss_enh        = self.loss(image, image_lr, illu_lr, enhanced, depth_lr)
                loss            = loss_recon + loss_enh * self.weight_enh
                outputs["loss"] = loss
            else:  # Skip updating model's weight at the first batch
                outputs = {"loss": None}
            # Saving n-th input and n-th pseudo gt
            self.saved_input     = nth_input
            self.saved_pseudo_gt = nth_pseudo_gt
        else:
            outputs         = self.forward(datapoint=datapoint, *args, **kwargs)
            image           = outputs["image"]
            image_lr        = outputs["image_lr"]
            illu_lr         = outputs["illu_lr"]
            enhanced        = outputs["enhanced"]
            depth_lr        = outputs["depth_lr"]
            outputs["loss"] = self.loss(image, image_lr, illu_lr, enhanced, depth_lr)
        return outputs
        
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        # Prepare input
        self.assert_datapoint(datapoint)
        image   = datapoint.get("image")
        depth   = datapoint.get("depth")
        outputs = self.model(image, depth)
        # Return
        return outputs
       
    def infer(
        self,
        datapoint    : dict,
        epochs       : int   = 500,
        lr           : float = 1e-5,
        weight_decay : float = 3e-4,
        reset_weights: bool  = True,
        *args, **kwargs
    ) -> dict:
        # Initialize training components
        self.train()
        if reset_weights:
            self.load_state_dict(self.initial_state_dict)
        if isinstance(self.optims, dict):
            optimizer = self.optims.get("optimizer", None)
        else:
            optimizer = nn.Adam(
                self.parameters(),
                lr           = lr,
                betas        = (0.9, 0.999),
                weight_decay = weight_decay
            )
        
        # Pre-processing
        self.saved_input     = None
        self.saved_pseudo_gt = None
        self.assert_datapoint(datapoint)
        for k, v in datapoint.items():
            if isinstance(v, torch.Tensor):
                datapoint[k] = v.to(self.device)
        
        # Training
        for _ in range(epochs):
            outputs = self.forward_loss(datapoint=datapoint)
            optimizer.zero_grad()
            loss = outputs["loss"]
            if loss is not None:
                loss.backward(retain_graph=True)
                optimizer.step()
            # if self.verbose:
            #    console.log(f"Loss: {loss.item()}")
            
        # Forward
        self.eval()
        timer = core.Timer()
        timer.tick()
        outputs = self.forward(datapoint=datapoint)
        timer.tock()
        self.assert_outputs(outputs)
        
        # Return
        outputs["time"] = timer.avg_time
        return outputs

# endregion
