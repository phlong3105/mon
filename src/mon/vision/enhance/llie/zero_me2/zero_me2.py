#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Zero-ME2.

This module implement our paper: Zero-shot Multimodal Multiple Exposure
Enhancement via Neural Implicit Representations.
"""

from __future__ import annotations

__all__ = [
    "ZeroME2",
    "ZeroME2_01_RGB",
    "ZeroME2_02_RGBD",
    "ZeroME2_03_RGBD_ZSN2N",
    "ZeroME2_04_HSV",
    "ZeroME2_05_HSVD",
    "ZeroME2_06_HSVD_ZSN2N",
    "ZeroME2_07_HVI",
    "ZeroME2_08_HVID",
    "ZeroME2_09_HVID_ZSN2N",
    "ZeroME2_10_HVI3",
    "ZeroME2_11_HVID3",
    "ZeroME2_12_HVID3_ZSN2N",
    "ZeroME2_13_RGBD_HSVD",
    "ZeroME2_14_RGBD_HSVD_ZSN2N",
]

from typing import Any, Literal

import kornia
import numpy as np
import torch
from torch.nn import functional as F

from mon import core, nn
from mon.globals import MODELS, Scheme, Task
from mon.vision import filtering
from mon.vision.enhance import base

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]

LDA = nn.LayeredFeatureAggregation


# region Loss

class Loss(nn.Loss):
    
    def __init__(
        self,
        L     : float = 0.5,
        alpha : float = 1,
        beta  : float = 20,
        gamma : float = 8,
        delta : float = 5,
        loss_weight: float = 1.0,
        reduction  : Literal["none", "mean", "sum"] = "mean",
        *args, **kwargs
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        self.l_exp = nn.ExposureValueControlLoss(patch_size=16, mean_val=L)
        self.l_tv  = nn.TotalVariationLoss()
        
    def forward(
        self,
        illu_lr         : torch.Tensor,
        image_v_lr      : torch.Tensor,
        image_v_fixed_lr: torch.Tensor,
    ) -> torch.Tensor:
        loss_spa      = torch.mean(torch.abs(torch.pow(illu_lr - image_v_lr, 2)))
        loss_tv       = self.l_tv(illu_lr)
        loss_exp      = torch.mean(self.l_exp(illu_lr))
        loss_sparsity = torch.mean(image_v_fixed_lr)
        loss = (
                   loss_spa * self.alpha
                  + loss_tv * self.beta
                 + loss_exp * self.gamma
            + loss_sparsity * self.delta
        )
        return loss

# endregion


# region Module

class Denoise1(nn.Module):
    
    def __init__(self, embed_channels: int = 48):
        super().__init__()
        self.act   = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(3,   embed_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(embed_channels, embed_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(embed_channels, 3,  1)
    
    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x


class Denoise2(nn.Module):
    
    def __init__(self, embed_channels: int = 96):
        super().__init__()
        self.act   = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(6,   embed_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(embed_channels, embed_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(embed_channels, 6,  1)
    
    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x

# endregion


# region Model

@MODELS.register(name="zero_me2", arch="zero_me2")
class ZeroME2(base.ImageEnhancementModel):

    model_dir: core.Path    = current_dir
    arch     : str          = "zero_me2"
    tasks    : list[Task]   = [Task.LLIE]
    schemes  : list[Scheme] = [Scheme.ZERO_REFERENCE, Scheme.INSTANCE]
    zoo      : dict         = {}
    
    def __init__(
        self,
        name          : str   = "zero_me2",
        embed_channels: int   = 48,
        window_size   : int   = 1,
        down_size     : int   = 256,
        num_layers    : int   = 4,
        hidden_dim    : int   = 256,
        add_layer     : int   = 2,
        weight_decay  : list[float] = [0.1, 0.0001, 0.001],
        L             : float = 0.3,
        alpha         : float = 1,
        beta          : float = 20,
        gamma         : float = 8,
        delta         : float = 5,
        weights       : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            name    = name,
            weights = weights,
            *args, **kwargs
        )
        self.embed_channels = embed_channels
        self.window_size    = window_size
        self.patch_dim      = window_size ** 2
        self.down_size      = down_size
        self.omega_0        = 30.0
        self.siren_C        = 6.0
        
        patch_layers   = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        spatial_layers = [nn.SIREN(2,   hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        for _ in range(1, add_layer - 2):
            patch_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            spatial_layers.append(nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
        patch_layers.append(nn.SIREN(hidden_dim, hidden_dim // 2, self.omega_0, self.siren_C))
        spatial_layers.append(nn.SIREN(hidden_dim, hidden_dim // 2, self.omega_0, self.siren_C))
        
        output_layers = []
        for _ in range(add_layer, num_layers - 1):
            output_layers.append(nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
        output_layers.append(nn.SIREN(hidden_dim, 3, self.omega_0, self.siren_C, is_last=True))
        
        self.patch_net   = nn.Sequential(*patch_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net  = nn.Sequential(*output_layers)

        # Loss
        self.loss = Loss(L, alpha, beta, gamma, delta)
        
        weight_decay = weight_decay or [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{
	        "params"      : self.spatial_net.parameters(),
	        "weight_decay": weight_decay[0]
        }]
        self.params += [{
	        "params"      : self.patch_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.output_net.parameters(),
	        "weight_decay": weight_decay[2]
        }]
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        self.initial_state_dict = self.state_dict()
    
    def init_weights(self, m: nn.Module):
        pass
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        self.assert_datapoint(datapoint)
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        self.assert_outputs(outputs)
        # Loss
        illu_lr            = outputs["illu_lr"]
        image_rgb_lr       = outputs["image_rgb_lr"]
        image_rgb_fixed_lr = outputs["image_rgb_fixed_lr"]
        outputs["loss"]    = self.loss(illu_lr, image_rgb_lr, image_rgb_fixed_lr)
        # Return
        return outputs
        
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        # Prepare input
        image_rgb          = datapoint.get("image")
        # Enhance
        image_rgb_lr       = self.interpolate_image(image_rgb)
        patch              = self.patch_net(self.get_patches(image_rgb_lr))
        spatial            = self.spatial_net(self.get_coords())
        illu_res_lr        = self.output_net(torch.cat([patch, spatial], -1))
        illu_res_lr        = illu_res_lr.view(1, 3, self.down_size, self.down_size)
        illu_lr            = illu_res_lr + image_rgb_lr
        image_rgb_fixed_lr = image_rgb_lr / (illu_lr + 1e-4)
        image_rgb_fixed_lr = kornia.filters.bilateral_blur(image_rgb_fixed_lr, (3, 3), 0.1, (1.5, 1.5))
        image_rgb_fixed    = self.filter_up(image_rgb_lr, image_rgb_fixed_lr, image_rgb)
        image_rgb_fixed    = image_rgb_fixed / torch.max(image_rgb_fixed)
        # Return
        if self.debug:
            return {
                "illu_lr"           : illu_lr,
                "image_rgb_lr"      : image_rgb_lr,
                "image_rgb_fixed_lr": image_rgb_fixed_lr,
                "enhanced"          : image_rgb_fixed,
            }
        else:
            return {
                "enhanced": image_rgb_fixed,
            }
        
    def interpolate_image(self, image: torch.Tensor) -> torch.Tensor:
        """Reshapes the image based on new resolution."""
        return F.interpolate(image, size=(self.down_size, self.down_size), mode="bicubic")
    
    def get_patches(self, image: torch.Tensor) -> torch.Tensor:
        """Creates a tensor where the channel contains patch information."""
        num_channels = core.get_image_num_channels(image)
        kernel       = torch.zeros((self.window_size ** 2, num_channels, self.window_size, self.window_size)).cuda()
        for i in range(self.window_size):
            for j in range(self.window_size):
                kernel[int(torch.sum(kernel).item()), 0, i, j] = 1
        
        pad       = nn.ReflectionPad2d(self.window_size // 2)
        im_padded = pad(image)
        extracted = F.conv2d(im_padded, kernel, padding=0).squeeze(0)
        return torch.movedim(extracted, 0, -1)
    
    def get_coords(self) -> torch.Tensor:
        """Creates a coordinates grid."""
        coords = np.dstack(
            np.meshgrid(
                np.linspace(0, 1, self.down_size),
                np.linspace(0, 1, self.down_size)
            )
        )
        coords = torch.from_numpy(coords).float().cuda()
        return coords
    
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
    
    def infer(
        self,
        datapoint    : dict,
        epochs       : int   = 200,
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
                weight_decay = weight_decay,
            )
        
        # Pre-processing
        self.assert_datapoint(datapoint)
        for k, v in datapoint.items():
            if isinstance(v, torch.Tensor):
                datapoint[k] = v.to(self.device)
        
        # Training
        for _ in range(epochs):
            outputs = self.forward_loss(datapoint=datapoint)
            optimizer.zero_grad()
            loss = outputs["loss"]
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


# region Ablation

@MODELS.register(name="zero_me2_01_rgb", arch="zero_me2")
class ZeroME2_01_RGB(ZeroME2):
    
    def __init__(
        self,
        name          : str   = "zero_me2_01_rgb",
        embed_channels: int   = 48,
        window_size   : int   = 1,
        down_size     : int   = 256,
        num_layers    : int   = 4,
        hidden_dim    : int   = 256,
        add_layer     : int   = 2,
        weight_decay  : list[float] = [0.1, 0.0001, 0.001],
        L             : float = 0.3,
        alpha         : float = 1,
        beta          : float = 20,
        gamma         : float = 8,
        delta         : float = 5,
        weights       : Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name    = name,
            weights = weights,
            *args, **kwargs
        )
        self.embed_channels = embed_channels
        self.window_size    = window_size
        self.patch_dim      = window_size ** 2
        self.down_size      = down_size
        self.omega_0        = 30.0
        self.siren_C        = 6.0
        
        patch_r_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_g_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_b_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        spatial_layers = [nn.SIREN(2,   hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        for _ in range(1, add_layer - 2):
            patch_r_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_g_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_b_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            spatial_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
        patch_r_layers.append(nn.SIREN(hidden_dim, hidden_dim // 2, self.omega_0, self.siren_C))
        patch_g_layers.append(nn.SIREN(hidden_dim, hidden_dim // 2, self.omega_0, self.siren_C))
        patch_b_layers.append(nn.SIREN(hidden_dim, hidden_dim // 2, self.omega_0, self.siren_C))
        spatial_layers.append(nn.SIREN(hidden_dim, hidden_dim // 2, self.omega_0, self.siren_C))
        
        output_layers = []
        for _ in range(add_layer, num_layers - 1):
            output_layers.append(nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
        output_layers.append(nn.SIREN(hidden_dim, 1, self.omega_0, self.siren_C, is_last=True))
        
        self.patch_r_net = nn.Sequential(*patch_r_layers)
        self.patch_g_net = nn.Sequential(*patch_g_layers)
        self.patch_b_net = nn.Sequential(*patch_b_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net  = nn.Sequential(*output_layers)
        
        # Loss
        self.loss = Loss(L, alpha, beta, gamma, delta)
        
        weight_decay = weight_decay or [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{
	        "params"      : self.spatial_net.parameters(),
	        "weight_decay": weight_decay[0]
        }]
        self.params += [{
	        "params"      : self.patch_r_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.patch_g_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.patch_b_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.output_net.parameters(),
	        "weight_decay": weight_decay[2]
        }]
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        self.initial_state_dict = self.state_dict()
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        self.assert_datapoint(datapoint)
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        self.assert_outputs(outputs)
        # Loss
        illu_lr            = outputs["illu_lr"]
        image_rgb_lr       = outputs["image_rgb_lr"]
        image_rgb_fixed_lr = outputs["image_rgb_fixed_lr"]
        outputs["loss"]    = self.loss(illu_lr, image_rgb_lr, image_rgb_fixed_lr)
        # Return
        return outputs
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        # Prepare input
        image_rgb       = datapoint.get("image")
        image_r         = image_rgb[:, 0:1, :, :]
        image_g         = image_rgb[:, 1:2, :, :]
        image_b         = image_rgb[:, 2:3, :, :]
        # Enhance
        image_rgb_lr    = self.interpolate_image(image_rgb)
        image_r_lr      = image_rgb_lr[:, 0:1, :, :]
        image_g_lr      = image_rgb_lr[:, 1:2, :, :]
        image_b_lr      = image_rgb_lr[:, 2:3, :, :]
        patch_r         = self.patch_r_net(self.get_patches(image_r_lr))
        patch_g         = self.patch_g_net(self.get_patches(image_g_lr))
        patch_b         = self.patch_b_net(self.get_patches(image_b_lr))
        spatial         = self.spatial_net(self.get_coords())
        illu_res_r_lr   = self.output_net(torch.cat([patch_r, spatial], -1))
        illu_res_g_lr   = self.output_net(torch.cat([patch_g, spatial], -1))
        illu_res_b_lr   = self.output_net(torch.cat([patch_b, spatial], -1))
        illu_res_r_lr   = illu_res_r_lr.view(1, 1, self.down_size, self.down_size)
        illu_res_g_lr   = illu_res_g_lr.view(1, 1, self.down_size, self.down_size)
        illu_res_b_lr   = illu_res_b_lr.view(1, 1, self.down_size, self.down_size)
        illu_res_lr     = torch.cat([illu_res_r_lr, illu_res_g_lr, illu_res_b_lr], 1)
        illu_r_lr       = illu_res_r_lr + image_r_lr
        illu_g_lr       = illu_res_g_lr + image_g_lr
        illu_b_lr       = illu_res_b_lr + image_b_lr
        illu_lr         = torch.cat([illu_r_lr, illu_g_lr, illu_b_lr], 1)
        illu_r_fixed_lr = image_r_lr / (illu_r_lr + 1e-4)
        illu_g_fixed_lr = image_g_lr / (illu_g_lr + 1e-4)
        illu_b_fixed_lr = image_b_lr / (illu_b_lr + 1e-4)
        illu_r_fixed_lr = torch.clamp(illu_r_fixed_lr, 1e-4, 1)
        illu_g_fixed_lr = torch.clamp(illu_g_fixed_lr, 1e-4, 1)
        illu_b_fixed_lr = torch.clamp(illu_b_fixed_lr, 1e-4, 1)
        illu_r_fixed_lr = kornia.filters.bilateral_blur(illu_r_fixed_lr, (3, 3), 0.1, (1.5, 1.5))
        illu_g_fixed_lr = kornia.filters.bilateral_blur(illu_g_fixed_lr, (3, 3), 0.1, (1.5, 1.5))
        illu_b_fixed_lr = kornia.filters.bilateral_blur(illu_b_fixed_lr, (3, 3), 0.1, (1.5, 1.5))
        image_rgb_fixed_lr = torch.cat([illu_r_fixed_lr, illu_g_fixed_lr, illu_b_fixed_lr], 1)
        image_r_fixed   = self.filter_up(image_r_lr, illu_r_fixed_lr, image_r)
        image_g_fixed   = self.filter_up(image_g_lr, illu_g_fixed_lr, image_g)
        image_b_fixed   = self.filter_up(image_b_lr, illu_b_fixed_lr, image_b)
        image_rgb_fixed = torch.cat([image_r_fixed, image_g_fixed, image_b_fixed], 1)
        image_rgb_fixed = image_rgb_fixed / torch.max(image_rgb_fixed)
        # Return
        if self.debug:
            return {
                "illu_lr"           : illu_lr,
                "image_rgb_lr"      : image_rgb_lr,
                "image_rgb_fixed_lr": image_rgb_fixed_lr,
                "enhanced"          : image_rgb_fixed,
            }
        else:
            return {
                "enhanced": image_rgb_fixed,
            }


@MODELS.register(name="zero_me2_02_rgbd", arch="zero_me2")
class ZeroME2_02_RGBD(ZeroME2):
    
    def __init__(
        self,
        name          : str   = "zero_me2_02_rgbd",
        embed_channels: int   = 48,
        window_size   : int   = 1,
        down_size     : int   = 256,
        num_layers    : int   = 4,
        hidden_dim    : int   = 256,
        add_layer     : int   = 2,
        weight_decay  : list[float] = [0.1, 0.0001, 0.001],
        L             : float = 0.3,
        alpha         : float = 1,
        beta          : float = 20,
        gamma         : float = 8,
        delta         : float = 5,
        weights       : Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name    = name,
            weights = weights,
            *args, **kwargs
        )
        self.embed_channels = embed_channels
        self.window_size    = window_size
        self.patch_dim      = window_size ** 2
        self.down_size      = down_size
        self.omega_0        = 30.0
        self.siren_C        = 6.0
        
        patch_r_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_g_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_b_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_d_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_e_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        spatial_layers = [nn.SIREN(2,   hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        for _ in range(1, add_layer - 2):
            patch_r_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_g_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_b_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_d_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_e_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            spatial_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
        patch_r_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        patch_g_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        patch_b_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        patch_d_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        patch_e_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        spatial_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        
        output_layers = []
        for _ in range(add_layer, num_layers - 1):
            output_layers.append(nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
        output_layers.append(nn.SIREN(hidden_dim, 1, self.omega_0, self.siren_C, is_last=True))
        
        self.patch_r_net = nn.Sequential(*patch_r_layers)
        self.patch_g_net = nn.Sequential(*patch_g_layers)
        self.patch_b_net = nn.Sequential(*patch_b_layers)
        self.patch_d_net = nn.Sequential(*patch_d_layers)
        self.patch_e_net = nn.Sequential(*patch_e_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net  = nn.Sequential(*output_layers)
        self.dba         = nn.BoundaryAwarePrior(eps=0.05, normalized=False)
        
        # Loss
        self.loss = Loss(L, alpha, beta, gamma, delta)
        
        weight_decay = weight_decay or [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{
	        "params"      : self.spatial_net.parameters(),
	        "weight_decay": weight_decay[0]
        }]
        self.params += [{
	        "params"      : self.patch_r_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.patch_g_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.patch_b_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.patch_d_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.patch_e_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.output_net.parameters(),
	        "weight_decay": weight_decay[2]
        }]
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        self.initial_state_dict = self.state_dict()
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        self.assert_datapoint(datapoint)
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        self.assert_outputs(outputs)
        # Loss
        illu_lr            = outputs["illu_lr"]
        image_rgb_lr       = outputs["image_rgb_lr"]
        image_rgb_fixed_lr = outputs["image_rgb_fixed_lr"]
        outputs["loss"]    = self.loss(illu_lr, image_rgb_lr, image_rgb_fixed_lr)
        # Return
        return outputs
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        # Prepare input
        image_rgb = datapoint.get("image")
        depth     = datapoint.get("depth")
        if depth is None:
            depth = core.rgb_to_grayscale(image_rgb)
        edge      = self.dba(depth)
        # Enhance
        image_r         = image_rgb[:, 0:1, :, :]
        image_g         = image_rgb[:, 1:2, :, :]
        image_b         = image_rgb[:, 2:3, :, :]
        image_rgb_lr    = self.interpolate_image(image_rgb)
        depth_lr        = self.interpolate_image(depth)
        edge_lr         = self.interpolate_image(edge)
        image_r_lr      = image_rgb_lr[:, 0:1, :, :]
        image_g_lr      = image_rgb_lr[:, 1:2, :, :]
        image_b_lr      = image_rgb_lr[:, 2:3, :, :]
        patch_r         = self.patch_r_net(self.get_patches(image_r_lr))
        patch_g         = self.patch_g_net(self.get_patches(image_g_lr))
        patch_b         = self.patch_b_net(self.get_patches(image_b_lr))
        patch_d         = self.patch_d_net(self.get_patches(depth_lr))
        patch_e         = self.patch_e_net(self.get_patches(edge_lr))
        spatial         = self.spatial_net(self.get_coords())
        illu_res_r_lr   = self.output_net(torch.cat([patch_r, patch_e, patch_d, spatial], -1))
        illu_res_g_lr   = self.output_net(torch.cat([patch_g, patch_e, patch_d, spatial], -1))
        illu_res_b_lr   = self.output_net(torch.cat([patch_b, patch_e, patch_d, spatial], -1))
        illu_res_r_lr   = illu_res_r_lr.view(1, 1, self.down_size, self.down_size)
        illu_res_g_lr   = illu_res_g_lr.view(1, 1, self.down_size, self.down_size)
        illu_res_b_lr   = illu_res_b_lr.view(1, 1, self.down_size, self.down_size)
        illu_res_lr     = torch.cat([illu_res_r_lr, illu_res_g_lr, illu_res_b_lr], 1)
        illu_r_lr       = illu_res_r_lr + image_r_lr
        illu_g_lr       = illu_res_g_lr + image_g_lr
        illu_b_lr       = illu_res_b_lr + image_b_lr
        illu_lr         = torch.cat([illu_r_lr, illu_g_lr, illu_b_lr], 1)
        illu_r_fixed_lr = image_r_lr / (illu_r_lr + 1e-4)
        illu_g_fixed_lr = image_g_lr / (illu_g_lr + 1e-4)
        illu_b_fixed_lr = image_b_lr / (illu_b_lr + 1e-4)
        illu_r_fixed_lr = torch.clamp(illu_r_fixed_lr, 1e-4, 1)
        illu_g_fixed_lr = torch.clamp(illu_g_fixed_lr, 1e-4, 1)
        illu_b_fixed_lr = torch.clamp(illu_b_fixed_lr, 1e-4, 1)
        illu_r_fixed_lr = kornia.filters.bilateral_blur(illu_r_fixed_lr, (3, 3), 0.1, (1.5, 1.5))
        illu_g_fixed_lr = kornia.filters.bilateral_blur(illu_g_fixed_lr, (3, 3), 0.1, (1.5, 1.5))
        illu_b_fixed_lr = kornia.filters.bilateral_blur(illu_b_fixed_lr, (3, 3), 0.1, (1.5, 1.5))
        image_rgb_fixed_lr = torch.cat([illu_r_fixed_lr, illu_g_fixed_lr, illu_b_fixed_lr], 1)
        image_r_fixed   = self.filter_up(image_r_lr, illu_r_fixed_lr, image_r)
        image_g_fixed   = self.filter_up(image_g_lr, illu_g_fixed_lr, image_g)
        image_b_fixed   = self.filter_up(image_b_lr, illu_b_fixed_lr, image_b)
        image_rgb_fixed = torch.cat([image_r_fixed, image_g_fixed, image_b_fixed], 1)
        image_rgb_fixed = image_rgb_fixed / torch.max(image_rgb_fixed)
        # Return
        if self.debug:
            return {
                "depth"             : depth,
                "edge"              : edge,
                "illu_lr"           : illu_lr,
                "image_rgb_lr"      : image_rgb_lr,
                "image_rgb_fixed_lr": image_rgb_fixed_lr,
                "enhanced"          : image_rgb_fixed,
            }
        else:
            return {
                "enhanced": image_rgb_fixed,
            }


@MODELS.register(name="zero_me2_03_rgbd_zsn2n", arch="zero_me2")
class ZeroME2_03_RGBD_ZSN2N(ZeroME2_02_RGBD):
    
    def __init__(
        self,
        name   : str = "zero_me2_03_rgbd_zsn2n",
        weights: Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name    = name,
            weights = weights,
            *args, **kwargs
        )
        
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        self.assert_datapoint(datapoint)
        image          = datapoint.get("image")
        image1, image2 = core.pair_downsample(image)
        datapoint1     = datapoint | {"image": image1}
        datapoint2     = datapoint | {"image": image2}
        outputs1       = self.forward(datapoint=datapoint1, *args, **kwargs)
        outputs2       = self.forward(datapoint=datapoint2, *args, **kwargs)
        outputs        = self.forward(datapoint=datapoint,  *args, **kwargs)
        self.assert_outputs(outputs)
        # Symmetric Loss
        illu_lr1            = outputs1["illu_lr"]
        image_rgb_lr1       = outputs1["image_rgb_lr"]
        image_rgb_fixed_lr1 = outputs1["image_rgb_fixed_lr"]
        image_rgb_fixed1    = outputs1["enhanced"]
        illu_lr2            = outputs2["illu_lr"]
        image_rgb_lr2       = outputs2["image_rgb_lr"]
        image_rgb_fixed_lr2 = outputs2["image_rgb_fixed_lr"]
        image_rgb_fixed2    = outputs2["enhanced"]
        illu_lr             = outputs["illu_lr"]
        image_rgb_lr        = outputs["image_rgb_lr"]
        image_rgb_fixed_lr  = outputs["image_rgb_fixed_lr"]
        image_rgb_fixed     = outputs["enhanced"]
        image_rgb_fixed_1, image_rgb_fixed_2 = core.pair_downsample(image_rgb_fixed)
        mse_loss = nn.MSELoss()
        loss_res = 0.5 * (mse_loss(image1,           image_rgb_fixed_2) + mse_loss(image2,           image_rgb_fixed_1))
        loss_con = 0.5 * (mse_loss(image_rgb_fixed1, image_rgb_fixed_1) + mse_loss(image_rgb_fixed2, image_rgb_fixed_2))
        loss_enh = self.loss(illu_lr, image_rgb_lr, image_rgb_fixed_lr)
        loss     = 0.15 * loss_res + 0.15 * loss_con + 0.7 * loss_enh
        outputs["loss"] = loss
        # Return
        return outputs


@MODELS.register(name="zero_me2_04_hsv", arch="zero_me2")
class ZeroME2_04_HSV(ZeroME2):
    
    def __init__(
        self,
        name          : str   = "zero_me2_04_hsv",
        embed_channels: int   = 48,
        window_size   : int   = 1,
        down_size     : int   = 256,
        num_layers    : int   = 4,
        hidden_dim    : int   = 256,
        add_layer     : int   = 2,
        weight_decay  : list[float] = [0.1, 0.0001, 0.001],
        L             : float = 0.3,
        alpha         : float = 1,
        beta          : float = 20,
        gamma         : float = 8,
        delta         : float = 5,
        weights       : Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name    = name,
            weights = weights,
            *args, **kwargs
        )
        self.embed_channels = embed_channels
        self.window_size    = window_size
        self.patch_dim      = window_size ** 2
        self.down_size      = down_size
        self.omega_0        = 30.0
        self.siren_C        = 6.0
        
        patch_v_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        spatial_layers = [nn.SIREN(2,   hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        for _ in range(1, add_layer - 2):
            patch_v_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            spatial_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
        patch_v_layers.append(nn.SIREN(hidden_dim, hidden_dim // 2, self.omega_0, self.siren_C))
        spatial_layers.append(nn.SIREN(hidden_dim, hidden_dim // 2, self.omega_0, self.siren_C))
        
        output_layers = []
        for _ in range(add_layer, num_layers - 1):
            output_layers.append(nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
        output_layers.append(nn.SIREN(hidden_dim, 1, self.omega_0, self.siren_C, is_last=True))
        
        self.patch_v_net = nn.Sequential(*patch_v_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net  = nn.Sequential(*output_layers)
        
        # Loss
        self.loss = Loss(L, alpha, beta, gamma, delta)
        
        weight_decay = weight_decay or [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{
	        "params"      : self.spatial_net.parameters(),
	        "weight_decay": weight_decay[0]
        }]
        self.params += [{
	        "params"      : self.patch_v_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.output_net.parameters(),
	        "weight_decay": weight_decay[2]
        }]
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        self.initial_state_dict = self.state_dict()
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        self.assert_datapoint(datapoint)
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        self.assert_outputs(outputs)
        # Loss
        illu_v_lr        = outputs["illu_v_lr"]
        image_v_lr       = outputs["image_v_lr"]
        image_v_fixed_lr = outputs["image_v_fixed_lr"]
        outputs["loss"]  = self.loss(illu_v_lr, image_v_lr, image_v_fixed_lr)
        # Return
        return outputs
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        # Prepare inputs
        image_rgb        = datapoint.get("image")
        # Enhance
        image_hsv        = core.rgb_to_hsv(image_rgb)
        image_v          = core.rgb_to_v(image_rgb)
        image_v_lr       = self.interpolate_image(image_v)
        patch_v          = self.patch_v_net(self.get_patches(image_v_lr))
        spatial          = self.spatial_net(self.get_coords())
        illu_res_v_lr    = self.output_net(torch.cat([patch_v, spatial], -1))
        illu_res_v_lr    = illu_res_v_lr.view(1, 1, self.down_size, self.down_size)
        illu_v_lr        = illu_res_v_lr + image_v_lr
        image_v_fixed_lr = image_v_lr / (illu_v_lr + 1e-4)
        image_v_fixed_lr = torch.clamp(image_v_fixed_lr, 1e-4, 1)
        image_v_fixed_lr = kornia.filters.bilateral_blur(image_v_fixed_lr, (3, 3), 0.1, (1.5, 1.5))
        image_v_fixed    = self.filter_up(image_v_lr, image_v_fixed_lr, image_v)
        image_hsv_fixed  = self.replace_v_component(image_hsv, image_v_fixed)
        image_rgb_fixed  = core.hsv_to_rgb(image_hsv_fixed)
        image_rgb_fixed  = image_rgb_fixed / torch.max(image_rgb_fixed)
        # Return
        if self.debug:
            return {
                "illu_v_lr"       : illu_v_lr,
                "image_v_lr"      : image_v_lr,
                "image_v_fixed_lr": image_v_fixed_lr,
                "enhanced"        : image_rgb_fixed,
            }
        else:
            return {
                "enhanced": image_rgb_fixed,
            }


@MODELS.register(name="zero_me2_05_hsvd", arch="zero_me2")
class ZeroME2_05_HSVD(ZeroME2):
    
    def __init__(
        self,
        name          : str   = "zero_me2_05_hsvd",
        embed_channels: int   = 48,
        window_size   : int   = 1,
        down_size     : int   = 256,
        num_layers    : int   = 4,
        hidden_dim    : int   = 256,
        add_layer     : int   = 2,
        weight_decay  : list[float] = [0.1, 0.0001, 0.001],
        L             : float = 0.3,
        alpha         : float = 1,
        beta          : float = 20,
        gamma         : float = 8,
        delta         : float = 5,
        weights       : Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name    = name,
            weights = weights,
            *args, **kwargs
        )
        self.embed_channels = embed_channels
        self.window_size    = window_size
        self.patch_dim      = window_size ** 2
        self.down_size      = down_size
        self.omega_0        = 30.0
        self.siren_C        = 6.0
        
        patch_v_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_d_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_e_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        spatial_layers = [nn.SIREN(2,   hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        for _ in range(1, add_layer - 2):
            patch_v_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_d_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_e_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            spatial_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
        patch_v_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        patch_d_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        patch_e_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        spatial_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        
        output_layers = []
        for _ in range(add_layer, num_layers - 1):
            output_layers.append(nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
        output_layers.append(nn.SIREN(hidden_dim, 1, self.omega_0, self.siren_C, is_last=True))
        
        self.patch_v_net = nn.Sequential(*patch_v_layers)
        self.patch_d_net = nn.Sequential(*patch_d_layers)
        self.patch_e_net = nn.Sequential(*patch_e_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net  = nn.Sequential(*output_layers)
        self.dba         = nn.BoundaryAwarePrior(eps=0.05, normalized=False)
        
        # Loss
        self.loss = Loss(L, alpha, beta, gamma, delta)
        
        weight_decay = weight_decay or [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{
	        "params"      : self.spatial_net.parameters(),
	        "weight_decay": weight_decay[0]
        }]
        self.params += [{
	        "params"      : self.patch_v_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.patch_d_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.patch_e_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.output_net.parameters(),
	        "weight_decay": weight_decay[2]
        }]
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        self.initial_state_dict = self.state_dict()
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        self.assert_datapoint(datapoint)
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        self.assert_outputs(outputs)
        # Loss
        illu_v_lr        = outputs["illu_v_lr"]
        image_v_lr       = outputs["image_v_lr"]
        image_v_fixed_lr = outputs["image_v_fixed_lr"]
        outputs["loss"]  = self.loss(illu_v_lr, image_v_lr, image_v_fixed_lr)
        # Return
        return outputs
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        # Prepare inputs
        image_rgb = datapoint.get("image")
        depth     = datapoint.get("depth")
        if depth is None:
            depth = core.rgb_to_grayscale(image_rgb)
        edge      = self.dba(depth)
        # Enhance
        image_hsv        = core.rgb_to_hsv(image_rgb)
        image_v          = core.rgb_to_v(image_rgb)
        image_v_lr       = self.interpolate_image(image_v)
        depth_lr         = self.interpolate_image(depth)
        edge_lr          = self.interpolate_image(edge)
        patch_v          = self.patch_v_net(self.get_patches(image_v_lr))
        patch_d          = self.patch_d_net(self.get_patches(depth_lr))
        patch_e          = self.patch_e_net(self.get_patches(edge_lr))
        spatial          = self.spatial_net(self.get_coords())
        illu_res_v_lr    = self.output_net(torch.cat([patch_v, patch_e, patch_d, spatial], -1))
        illu_res_v_lr    = illu_res_v_lr.view(1, 1, self.down_size, self.down_size)
        illu_v_lr        = illu_res_v_lr + image_v_lr
        image_v_fixed_lr = image_v_lr / (illu_v_lr + 1e-4)
        image_v_fixed_lr = torch.clamp(image_v_fixed_lr, 1e-4, 1)
        image_v_fixed_lr = kornia.filters.bilateral_blur(image_v_fixed_lr, (3, 3), 0.1, (1.5, 1.5))
        image_v_fixed    = self.filter_up(image_v_lr, image_v_fixed_lr, image_v)
        image_hsv_fixed  = self.replace_v_component(image_hsv, image_v_fixed)
        image_rgb_fixed  = core.hsv_to_rgb(image_hsv_fixed)
        image_rgb_fixed  = image_rgb_fixed / torch.max(image_rgb_fixed)
        # Return
        if self.debug:
            return {
                "depth"           : depth,
                "edge"            : edge,
                "illu_v_lr"       : illu_v_lr,
                "image_v_lr"      : image_v_lr,
                "image_v_fixed_lr": image_v_fixed_lr,
                "enhanced"        : image_rgb_fixed,
            }
        else:
            return {
                "enhanced": image_rgb_fixed,
            }


@MODELS.register(name="zero_me2_06_hsvd_zsn2n", arch="zero_me2")
class ZeroME2_06_HSVD_ZSN2N(ZeroME2_05_HSVD):
    
    def __init__(
        self,
        name   : str = "zero_me2_05_hsvd_zsn2n",
        weights: Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name    = name,
            weights = weights,
            *args, **kwargs
        )
        
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        self.assert_datapoint(datapoint)
        image          = datapoint.get("image")
        image1, image2 = core.pair_downsample(image)
        datapoint1     = datapoint | {"image": image1}
        datapoint2     = datapoint | {"image": image2}
        outputs1       = self.forward(datapoint=datapoint1, *args, **kwargs)
        outputs2       = self.forward(datapoint=datapoint2, *args, **kwargs)
        outputs        = self.forward(datapoint=datapoint,  *args, **kwargs)
        self.assert_outputs(outputs)
        # Symmetric Loss
        illu_v_lr1        = outputs1["illu_v_lr"]
        image_v_lr1       = outputs1["image_v_lr"]
        image_v_fixed_lr1 = outputs1["image_v_fixed_lr"]
        image_rgb_fixed1  = outputs1["enhanced"]
        illu_v_lr2        = outputs2["illu_v_lr"]
        image_v_lr2       = outputs2["image_v_lr"]
        image_v_fixed_lr2 = outputs2["image_v_fixed_lr"]
        image_rgb_fixed2  = outputs2["enhanced"]
        illu_v_lr         = outputs["illu_v_lr"]
        image_v_lr        = outputs["image_v_lr"]
        image_v_fixed_lr  = outputs["image_v_fixed_lr"]
        image_rgb_fixed   = outputs["enhanced"]
        image_rgb_fixed_1, image_rgb_fixed_2 = core.pair_downsample(image_rgb_fixed)
        mse_loss = nn.MSELoss()
        loss_res = 0.5 * (mse_loss(image1,           image_rgb_fixed_2) + mse_loss(image2,           image_rgb_fixed_1))
        loss_con = 0.5 * (mse_loss(image_rgb_fixed1, image_rgb_fixed_1) + mse_loss(image_rgb_fixed2, image_rgb_fixed_2))
        loss_enh = self.loss(illu_v_lr, image_v_lr, image_v_fixed_lr)
        loss     = 0.15 * loss_res + 0.15 * loss_con + 0.7 * loss_enh
        outputs["loss"] = loss
        # Return
        return outputs


@MODELS.register(name="zero_me2_07_hvi", arch="zero_me2")
class ZeroME2_07_HVI(ZeroME2):
    
    def __init__(
        self,
        name          : str   = "zero_me2_07_hvi",
        embed_channels: int   = 48,
        window_size   : int   = 1,
        down_size     : int   = 256,
        num_layers    : int   = 4,
        hidden_dim    : int   = 256,
        add_layer     : int   = 2,
        weight_decay  : list[float] = [0.1, 0.0001, 0.001],
        L             : float = 0.3,
        alpha         : float = 1,
        beta          : float = 20,
        gamma         : float = 8,
        delta         : float = 5,
        weights       : Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name    = name,
            weights = weights,
            *args, **kwargs
        )
        self.embed_channels = embed_channels
        self.window_size    = window_size
        self.patch_dim      = window_size ** 2
        self.down_size      = down_size
        self.omega_0        = 30.0
        self.siren_C        = 6.0
        
        patch_i_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        spatial_layers = [nn.SIREN(2,   hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        for _ in range(1, add_layer - 2):
            patch_i_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            spatial_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
        patch_i_layers.append(nn.SIREN(hidden_dim, hidden_dim // 2, self.omega_0, self.siren_C))
        spatial_layers.append(nn.SIREN(hidden_dim, hidden_dim // 2, self.omega_0, self.siren_C))
        
        output_layers = []
        for _ in range(add_layer, num_layers - 1):
            output_layers.append(nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
        output_layers.append(nn.SIREN(hidden_dim, 1, self.omega_0, self.siren_C, is_last=True))
        
        self.patch_i_net = nn.Sequential(*patch_i_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net  = nn.Sequential(*output_layers)
        self.trans       = core.RGBToHVI()
        
        # Loss
        self.loss = Loss(L, alpha, beta, gamma, delta)
        
        weight_decay = weight_decay or [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{
	        "params"      : self.spatial_net.parameters(),
	        "weight_decay": weight_decay[0]
        }]
        self.params += [{
	        "params"      : self.patch_i_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.output_net.parameters(),
	        "weight_decay": weight_decay[2]
        }]
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        self.initial_state_dict = self.state_dict()
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        self.assert_datapoint(datapoint)
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        self.assert_outputs(outputs)
        # Loss
        illu_i_lr        = outputs["illu_i_lr"]
        image_i_lr       = outputs["image_i_lr"]
        image_i_fixed_lr = outputs["image_i_fixed_lr"]
        outputs["loss"]  = self.loss(illu_i_lr, image_i_lr, image_i_fixed_lr)
        # Return
        return outputs
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        # Prepare inputs
        image_rgb        = datapoint.get("image")
        # Enhance
        image_hvi        = self.trans.rgb_to_hvi(image_rgb)
        image_hvi_clone  = image_hvi.clone().detach()
        image_h          = image_hvi_clone[:, 0:1, :, :]
        image_v          = image_hvi_clone[:, 1:2, :, :]
        image_i          = image_hvi_clone[:, 2:3, :, :]
        image_i_lr       = self.interpolate_image(image_i)
        patch_i          = self.patch_i_net(self.get_patches(image_i_lr))
        spatial          = self.spatial_net(self.get_coords())
        illu_res_i_lr    = self.output_net(torch.cat([patch_i, spatial], -1))
        illu_res_i_lr    = illu_res_i_lr.view(1, 1, self.down_size, self.down_size)
        illu_i_lr        = illu_res_i_lr + image_i_lr
        image_i_fixed_lr = image_i_lr / (illu_i_lr + 1e-4)
        image_i_fixed_lr = torch.clamp(image_i_fixed_lr, 1e-4, 1)
        image_i_fixed_lr = kornia.filters.bilateral_blur(image_i_fixed_lr, (3, 3), 0.1, (1.5, 1.5))
        image_i_fixed    = self.filter_up(image_i_lr, image_i_fixed_lr, image_i)
        image_hvi_fixed  = self.replace_i_component(image_hvi, image_i_fixed)
        image_rgb_fixed  = self.trans.hvi_to_rgb(image_hvi_fixed)
        image_rgb_fixed  = image_rgb_fixed.clone().detach()
        image_rgb_fixed  = image_rgb_fixed / torch.max(image_rgb_fixed)
        # Return
        if self.debug:
            return {
                "image_h"         : image_h,
                "image_v"         : image_v,
                "image_i"         : image_i,
                "illu_i_lr"       : illu_i_lr,
                "image_i_lr"      : image_i_lr,
                "image_i_fixed_lr": image_i_fixed_lr,
                "enhanced"        : image_rgb_fixed,
            }
        else:
            return {
                "enhanced": image_rgb_fixed,
            }


@MODELS.register(name="zero_me2_08_hvid", arch="zero_me2")
class ZeroME2_08_HVID(ZeroME2):
    
    def __init__(
        self,
        name          : str   = "zero_me2_08_hvid",
        embed_channels: int   = 48,
        window_size   : int   = 1,
        down_size     : int   = 256,
        num_layers    : int   = 4,
        hidden_dim    : int   = 256,
        add_layer     : int   = 2,
        weight_decay  : list[float] = [0.1, 0.0001, 0.001],
        L             : float = 0.3,
        alpha         : float = 1,
        beta          : float = 20,
        gamma         : float = 8,
        delta         : float = 5,
        weights       : Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name    = name,
            weights = weights,
            *args, **kwargs
        )
        self.embed_channels = embed_channels
        self.window_size    = window_size
        self.patch_dim      = window_size ** 2
        self.down_size      = down_size
        self.omega_0        = 30.0
        self.siren_C        = 6.0
        
        patch_i_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_d_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_e_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        spatial_layers = [nn.SIREN(2,   hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        for _ in range(1, add_layer - 2):
            patch_i_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_d_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_e_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            spatial_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
        patch_i_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        patch_d_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        patch_e_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        spatial_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        
        output_layers = []
        for _ in range(add_layer, num_layers - 1):
            output_layers.append(nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
        output_layers.append(nn.SIREN(hidden_dim, 1, self.omega_0, self.siren_C, is_last=True))
        
        self.patch_i_net = nn.Sequential(*patch_i_layers)
        self.patch_d_net = nn.Sequential(*patch_d_layers)
        self.patch_e_net = nn.Sequential(*patch_e_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net  = nn.Sequential(*output_layers)
        self.dba         = nn.BoundaryAwarePrior(eps=0.05, normalized=False)
        self.trans       = core.RGBToHVI()
        
        # Loss
        self.loss = Loss(L, alpha, beta, gamma, delta)
        
        weight_decay = weight_decay or [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{
	        "params"      : self.spatial_net.parameters(),
	        "weight_decay": weight_decay[0]
        }]
        self.params += [{
	        "params"      : self.patch_i_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.patch_d_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.patch_e_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.output_net.parameters(),
	        "weight_decay": weight_decay[2]
        }]
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        self.initial_state_dict = self.state_dict()
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        self.assert_datapoint(datapoint)
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        self.assert_outputs(outputs)
        # Loss
        illu_i_lr        = outputs["illu_i_lr"]
        image_i_lr       = outputs["image_i_lr"]
        image_i_fixed_lr = outputs["image_i_fixed_lr"]
        outputs["loss"]  = self.loss(illu_i_lr, image_i_lr, image_i_fixed_lr)
        # Return
        return outputs
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        # Prepare inputs
        image_rgb = datapoint.get("image")
        depth     = datapoint.get("depth")
        if depth is None:
            depth = core.rgb_to_grayscale(image_rgb)
        edge      = self.dba(depth)
        # Enhance
        image_hvi        = self.trans.rgb_to_hvi(image_rgb)
        image_hvi_clone  = image_hvi.clone().detach()
        image_h          = image_hvi_clone[:, 0:1, :, :]
        image_v          = image_hvi_clone[:, 1:2, :, :]
        image_i          = image_hvi_clone[:, 2:3, :, :]
        image_i_lr       = self.interpolate_image(image_i)
        depth_lr         = self.interpolate_image(depth)
        edge_lr          = self.interpolate_image(edge)
        patch_i          = self.patch_i_net(self.get_patches(image_i_lr))
        patch_d          = self.patch_d_net(self.get_patches(depth_lr))
        patch_e          = self.patch_e_net(self.get_patches(edge_lr))
        spatial          = self.spatial_net(self.get_coords())
        illu_res_i_lr    = self.output_net(torch.cat([patch_i, patch_e, patch_d, spatial], -1))
        illu_res_i_lr    = illu_res_i_lr.view(1, 1, self.down_size, self.down_size)
        illu_i_lr        = illu_res_i_lr + image_i_lr
        image_i_fixed_lr = image_i_lr / (illu_i_lr + 1e-4)
        image_i_fixed_lr = torch.clamp(image_i_fixed_lr, 1e-4, 1)
        image_i_fixed_lr = kornia.filters.bilateral_blur(image_i_fixed_lr, (3, 3), 0.1, (1.5, 1.5))
        image_i_fixed    = self.filter_up(image_i_lr, image_i_fixed_lr, image_i)
        image_hvi_fixed  = self.replace_i_component(image_hvi, image_i_fixed)
        image_rgb_fixed  = self.trans.hvi_to_rgb(image_hvi_fixed)
        image_rgb_fixed  = image_rgb_fixed.clone().detach()
        image_rgb_fixed  = image_rgb_fixed / torch.max(image_rgb_fixed)
        # Return
        if self.debug:
            return {
                "image_h"         : image_h,
                "image_v"         : image_v,
                "image_i"         : image_i,
                "depth"           : depth,
                "edge"            : edge,
                "illu_i_lr"       : illu_i_lr,
                "image_i_lr"      : image_i_lr,
                "image_i_fixed_lr": image_i_fixed_lr,
                "enhanced"        : image_rgb_fixed,
            }
        else:
            return {
                "enhanced": image_rgb_fixed,
            }


@MODELS.register(name="zero_me2_09_hvid_zsn2n", arch="zero_me2")
class ZeroME2_09_HVID_ZSN2N(ZeroME2_08_HVID):
    
    def __init__(
        self,
        name   : str = "zero_me2_09_hvid_zsn2n",
        weights: Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name    = name,
            weights = weights,
            *args, **kwargs
        )
        
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        self.assert_datapoint(datapoint)
        image          = datapoint.get("image")
        image1, image2 = core.pair_downsample(image)
        datapoint1     = datapoint | {"image": image1}
        datapoint2     = datapoint | {"image": image2}
        outputs1       = self.forward(datapoint=datapoint1, *args, **kwargs)
        outputs2       = self.forward(datapoint=datapoint2, *args, **kwargs)
        outputs        = self.forward(datapoint=datapoint,  *args, **kwargs)
        self.assert_outputs(outputs)
        # Symmetric Loss
        illu_i_lr1        = outputs1["illu_i_lr"]
        image_i_lr1       = outputs1["image_i_lr"]
        image_i_fixed_lr1 = outputs1["image_i_fixed_lr"]
        image_rgb_fixed1  = outputs1["enhanced"]
        illu_i_lr2        = outputs2["illu_i_lr"]
        image_i_lr2       = outputs2["image_i_lr"]
        image_i_fixed_lr2 = outputs2["image_i_fixed_lr"]
        image_rgb_fixed2  = outputs2["enhanced"]
        illu_i_lr         = outputs["illu_i_lr"]
        image_i_lr        = outputs["image_i_lr"]
        image_i_fixed_lr  = outputs["image_i_fixed_lr"]
        image_rgb_fixed   = outputs["enhanced"]
        image_rgb_fixed_1, image_rgb_fixed_2 = core.pair_downsample(image_rgb_fixed)
        mse_loss = nn.MSELoss()
        loss_res = 0.5 * (mse_loss(image1,           image_rgb_fixed_2) + mse_loss(image2,           image_rgb_fixed_1))
        loss_con = 0.5 * (mse_loss(image_rgb_fixed1, image_rgb_fixed_1) + mse_loss(image_rgb_fixed2, image_rgb_fixed_2))
        loss_enh = self.loss(illu_i_lr, image_i_lr, image_i_fixed_lr)
        loss     = 0.25 * loss_res + 0.25 * loss_con + 0.5 * loss_enh
        outputs["loss"] = loss
        # Return
        return outputs


@MODELS.register(name="zero_me2_10_hvi3", arch="zero_me2")
class ZeroME2_10_HVI3(ZeroME2):
    
    def __init__(
        self,
        name          : str   = "zero_me2_10_hvi3",
        embed_channels: int   = 48,
        window_size   : int   = 1,
        down_size     : int   = 256,
        num_layers    : int   = 4,
        hidden_dim    : int   = 256,
        add_layer     : int   = 2,
        weight_decay  : list[float] = [0.1, 0.0001, 0.001],
        L             : float = 0.3,
        alpha         : float = 1,
        beta          : float = 20,
        gamma         : float = 8,
        delta         : float = 5,
        weights       : Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name    = name,
            weights = weights,
            *args, **kwargs
        )
        self.embed_channels = embed_channels
        self.window_size    = window_size
        self.patch_dim      = window_size ** 2
        self.down_size      = down_size
        self.omega_0        = 30.0
        self.siren_C        = 6.0
        
        patch_h_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_v_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_i_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        spatial_layers = [nn.SIREN(2,   hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        for _ in range(1, add_layer - 2):
            patch_h_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_v_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_i_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            spatial_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
        patch_h_layers.append(nn.SIREN(hidden_dim, hidden_dim // 2, self.omega_0, self.siren_C))
        patch_v_layers.append(nn.SIREN(hidden_dim, hidden_dim // 2, self.omega_0, self.siren_C))
        patch_i_layers.append(nn.SIREN(hidden_dim, hidden_dim // 2, self.omega_0, self.siren_C))
        spatial_layers.append(nn.SIREN(hidden_dim, hidden_dim // 2, self.omega_0, self.siren_C))
        
        output_layers = []
        for _ in range(add_layer, num_layers - 1):
            output_layers.append(nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
        output_layers.append(nn.SIREN(hidden_dim, 1, self.omega_0, self.siren_C, is_last=True))
        
        self.patch_h_net = nn.Sequential(*patch_h_layers)
        self.patch_v_net = nn.Sequential(*patch_v_layers)
        self.patch_i_net = nn.Sequential(*patch_i_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net  = nn.Sequential(*output_layers)
        self.trans       = core.RGBToHVI()
        
        # Loss
        self.loss = Loss(L, alpha, beta, gamma, delta)
        
        weight_decay = weight_decay or [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{
	        "params"      : self.spatial_net.parameters(),
	        "weight_decay": weight_decay[0]
        }]
        self.params += [{
	        "params"      : self.patch_h_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.patch_v_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.patch_i_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.output_net.parameters(),
	        "weight_decay": weight_decay[2]
        }]
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        self.initial_state_dict = self.state_dict()
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        self.assert_datapoint(datapoint)
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        self.assert_outputs(outputs)
        # Loss
        illu_h_lr        = outputs["illu_h_lr"]
        illu_v_lr        = outputs["illu_v_lr"]
        illu_i_lr        = outputs["illu_i_lr"]
        image_h_lr       = outputs["image_h_lr"]
        image_v_lr       = outputs["image_v_lr"]
        image_i_lr       = outputs["image_i_lr"]
        image_h_fixed_lr = outputs["image_h_fixed_lr"]
        image_v_fixed_lr = outputs["image_v_fixed_lr"]
        image_i_fixed_lr = outputs["image_i_fixed_lr"]
        loss_h           = self.loss(illu_h_lr, image_h_lr, image_h_fixed_lr)
        loss_v           = self.loss(illu_v_lr, image_v_lr, image_v_fixed_lr)
        loss_i           = self.loss(illu_i_lr, image_i_lr, image_i_fixed_lr)
        outputs["loss"]  = (loss_h + loss_v + loss_i) / 3
        # Return
        return outputs
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        # Prepare inputs
        image_rgb        = datapoint.get("image")
        # Enhance
        image_hvi        = self.trans.rgb_to_hvi(image_rgb)
        image_hvi_clone  = image_hvi.clone().detach()
        image_h          = image_hvi_clone[:, 0:1, :, :]
        image_v          = image_hvi_clone[:, 1:2, :, :]
        image_i          = image_hvi_clone[:, 2:3, :, :]
        image_h_lr       = self.interpolate_image(image_h)
        image_v_lr       = self.interpolate_image(image_v)
        image_i_lr       = self.interpolate_image(image_i)
        patch_h          = self.patch_h_net(self.get_patches(image_h_lr))
        patch_v          = self.patch_v_net(self.get_patches(image_v_lr))
        patch_i          = self.patch_i_net(self.get_patches(image_i_lr))
        spatial          = self.spatial_net(self.get_coords())
        illu_res_h_lr    = self.output_net(torch.cat([patch_h, spatial], -1))
        illu_res_v_lr    = self.output_net(torch.cat([patch_v, spatial], -1))
        illu_res_i_lr    = self.output_net(torch.cat([patch_i, spatial], -1))
        illu_res_h_lr    = illu_res_h_lr.view(1, 1, self.down_size, self.down_size)
        illu_res_v_lr    = illu_res_v_lr.view(1, 1, self.down_size, self.down_size)
        illu_res_i_lr    = illu_res_i_lr.view(1, 1, self.down_size, self.down_size)
        illu_h_lr        = illu_res_h_lr + image_h_lr
        illu_v_lr        = illu_res_v_lr + image_v_lr
        illu_i_lr        = illu_res_i_lr + image_i_lr
        image_h_fixed_lr = image_h_lr / (illu_h_lr + 1e-4)
        image_v_fixed_lr = image_v_lr / (illu_v_lr + 1e-4)
        image_i_fixed_lr = image_i_lr / (illu_i_lr + 1e-4)
        image_h_fixed_lr = torch.clamp(image_h_fixed_lr, 1e-4, 1)
        image_v_fixed_lr = torch.clamp(image_v_fixed_lr, 1e-4, 1)
        image_i_fixed_lr = torch.clamp(image_i_fixed_lr, 1e-4, 1)
        image_h_fixed_lr = kornia.filters.bilateral_blur(image_h_fixed_lr, (3, 3), 0.1, (1.5, 1.5))
        image_v_fixed_lr = kornia.filters.bilateral_blur(image_v_fixed_lr, (3, 3), 0.1, (1.5, 1.5))
        image_i_fixed_lr = kornia.filters.bilateral_blur(image_i_fixed_lr, (3, 3), 0.1, (1.5, 1.5))
        image_h_fixed    = self.filter_up(image_h_lr, image_h_fixed_lr, image_h)
        image_v_fixed    = self.filter_up(image_v_lr, image_v_fixed_lr, image_v)
        image_i_fixed    = self.filter_up(image_i_lr, image_i_fixed_lr, image_i)
        image_hvi_fixed  = torch.cat([image_h_fixed, image_v_fixed, image_i_fixed], 1)
        image_rgb_fixed  = self.trans.hvi_to_rgb(image_hvi_fixed)
        image_rgb_fixed  = image_rgb_fixed.clone().detach()
        image_rgb_fixed  = image_rgb_fixed / torch.max(image_rgb_fixed)
        # Return
        if self.debug:
            return {
                "image_h"         : image_h,
                "image_v"         : image_v,
                "image_i"         : image_i,
                "illu_h_lr"       : illu_h_lr,
                "illu_v_lr"       : illu_v_lr,
                "illu_i_lr"       : illu_i_lr,
                "image_h_lr"      : image_h_lr,
                "image_v_lr"      : image_v_lr,
                "image_i_lr"      : image_i_lr,
                "image_h_fixed_lr": image_h_fixed_lr,
                "image_v_fixed_lr": image_v_fixed_lr,
                "image_i_fixed_lr": image_i_fixed_lr,
                "enhanced"        : image_rgb_fixed,
            }
        else:
            return {
                "enhanced": image_rgb_fixed,
            }


@MODELS.register(name="zero_me2_11_hvid3", arch="zero_me2")
class ZeroME2_11_HVID3(ZeroME2):
    
    def __init__(
        self,
        name          : str   = "zero_me2_11_hvid3",
        embed_channels: int   = 48,
        window_size   : int   = 1,
        down_size     : int   = 256,
        num_layers    : int   = 4,
        hidden_dim    : int   = 256,
        add_layer     : int   = 2,
        weight_decay  : list[float] = [0.1, 0.0001, 0.001],
        L             : float = 0.3,
        alpha         : float = 1,
        beta          : float = 20,
        gamma         : float = 8,
        delta         : float = 5,
        weights       : Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name    = name,
            weights = weights,
            *args, **kwargs
        )
        self.embed_channels = embed_channels
        self.window_size    = window_size
        self.patch_dim      = window_size ** 2
        self.down_size      = down_size
        self.omega_0        = 30.0
        self.siren_C        = 6.0
        
        patch_h_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_v_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_i_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_d_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_e_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        spatial_layers = [nn.SIREN(2,   hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        for _ in range(1, add_layer - 2):
            patch_h_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_v_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_i_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_d_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_e_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            spatial_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
        patch_h_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        patch_v_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        patch_i_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        patch_d_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        patch_e_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        spatial_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        
        output_layers = []
        for _ in range(add_layer, num_layers - 1):
            output_layers.append(nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
        output_layers.append(nn.SIREN(hidden_dim, 1, self.omega_0, self.siren_C, is_last=True))
        
        self.patch_h_net = nn.Sequential(*patch_h_layers)
        self.patch_v_net = nn.Sequential(*patch_v_layers)
        self.patch_i_net = nn.Sequential(*patch_i_layers)
        self.patch_d_net = nn.Sequential(*patch_d_layers)
        self.patch_e_net = nn.Sequential(*patch_e_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net  = nn.Sequential(*output_layers)
        self.trans       = core.RGBToHVI()
        
        # Loss
        self.loss = Loss(L, alpha, beta, gamma, delta)
        
        weight_decay = weight_decay or [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{
	        "params"      : self.spatial_net.parameters(),
	        "weight_decay": weight_decay[0]
        }]
        self.params += [{
	        "params"      : self.patch_h_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.patch_v_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.patch_i_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.patch_d_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.patch_e_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.output_net.parameters(),
	        "weight_decay": weight_decay[2]
        }]
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        self.initial_state_dict = self.state_dict()
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        self.assert_datapoint(datapoint)
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        self.assert_outputs(outputs)
        # Loss
        illu_h_lr        = outputs["illu_h_lr"]
        illu_v_lr        = outputs["illu_v_lr"]
        illu_i_lr        = outputs["illu_i_lr"]
        image_h_lr       = outputs["image_h_lr"]
        image_v_lr       = outputs["image_v_lr"]
        image_i_lr       = outputs["image_i_lr"]
        image_h_fixed_lr = outputs["image_h_fixed_lr"]
        image_v_fixed_lr = outputs["image_v_fixed_lr"]
        image_i_fixed_lr = outputs["image_i_fixed_lr"]
        loss_h           = self.loss(illu_h_lr, image_h_lr, image_h_fixed_lr)
        loss_v           = self.loss(illu_v_lr, image_v_lr, image_v_fixed_lr)
        loss_i           = self.loss(illu_i_lr, image_i_lr, image_i_fixed_lr)
        outputs["loss"]  = (loss_h + loss_v + loss_i) / 3
        # Return
        return outputs
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        # Prepare inputs
        image_rgb = datapoint.get("image")
        depth     = datapoint.get("depth")
        if depth is None:
            depth = core.rgb_to_grayscale(image_rgb)
        edge      = self.dba(depth)
        # Enhance
        image_hvi        = self.trans.rgb_to_hvi(image_rgb)
        image_hvi_clone  = image_hvi.clone().detach()
        image_h          = image_hvi_clone[:, 0:1, :, :]
        image_v          = image_hvi_clone[:, 1:2, :, :]
        image_i          = image_hvi_clone[:, 2:3, :, :]
        image_h_lr       = self.interpolate_image(image_h)
        image_v_lr       = self.interpolate_image(image_v)
        image_i_lr       = self.interpolate_image(image_i)
        depth_lr         = self.interpolate_image(depth)
        edge_lr          = self.interpolate_image(edge)
        patch_h          = self.patch_h_net(self.get_patches(image_h_lr))
        patch_v          = self.patch_v_net(self.get_patches(image_v_lr))
        patch_i          = self.patch_i_net(self.get_patches(image_i_lr))
        patch_d          = self.patch_d_net(self.get_patches(depth_lr))
        patch_e          = self.patch_e_net(self.get_patches(edge_lr))
        spatial          = self.spatial_net(self.get_coords())
        illu_res_h_lr    = self.output_net(torch.cat([patch_h, patch_d, patch_e, spatial], -1))
        illu_res_v_lr    = self.output_net(torch.cat([patch_v, patch_d, patch_e, spatial], -1))
        illu_res_i_lr    = self.output_net(torch.cat([patch_i, patch_d, patch_e, spatial], -1))
        illu_res_h_lr    = illu_res_h_lr.view(1, 1, self.down_size, self.down_size)
        illu_res_v_lr    = illu_res_v_lr.view(1, 1, self.down_size, self.down_size)
        illu_res_i_lr    = illu_res_i_lr.view(1, 1, self.down_size, self.down_size)
        illu_h_lr        = illu_res_h_lr + image_h_lr
        illu_v_lr        = illu_res_v_lr + image_v_lr
        illu_i_lr        = illu_res_i_lr + image_i_lr
        image_h_fixed_lr = image_h_lr / (illu_h_lr + 1e-4)
        image_v_fixed_lr = image_v_lr / (illu_v_lr + 1e-4)
        image_i_fixed_lr = image_i_lr / (illu_i_lr + 1e-4)
        image_h_fixed_lr = torch.clamp(image_h_fixed_lr, 1e-4, 1)
        image_v_fixed_lr = torch.clamp(image_v_fixed_lr, 1e-4, 1)
        image_i_fixed_lr = torch.clamp(image_i_fixed_lr, 1e-4, 1)
        image_h_fixed_lr = kornia.filters.bilateral_blur(image_h_fixed_lr, (3, 3), 0.1, (1.5, 1.5))
        image_v_fixed_lr = kornia.filters.bilateral_blur(image_v_fixed_lr, (3, 3), 0.1, (1.5, 1.5))
        image_i_fixed_lr = kornia.filters.bilateral_blur(image_i_fixed_lr, (3, 3), 0.1, (1.5, 1.5))
        image_h_fixed    = self.filter_up(image_h_lr, image_h_fixed_lr, image_h)
        image_v_fixed    = self.filter_up(image_v_lr, image_v_fixed_lr, image_v)
        image_i_fixed    = self.filter_up(image_i_lr, image_i_fixed_lr, image_i)
        image_hvi_fixed  = torch.cat([image_h_fixed, image_v_fixed, image_i_fixed], 1)
        image_rgb_fixed  = self.trans.hvi_to_rgb(image_hvi_fixed)
        image_rgb_fixed  = image_rgb_fixed.clone().detach()
        image_rgb_fixed  = image_rgb_fixed / torch.max(image_rgb_fixed)
        # Return
        if self.debug:
            return {
                "image_h"         : image_h,
                "image_v"         : image_v,
                "image_i"         : image_i,
                "depth"           : depth,
                "edge"            : edge,
                "illu_h_lr"       : illu_h_lr,
                "illu_v_lr"       : illu_v_lr,
                "illu_i_lr"       : illu_i_lr,
                "image_h_lr"      : image_h_lr,
                "image_v_lr"      : image_v_lr,
                "image_i_lr"      : image_i_lr,
                "image_h_fixed_lr": image_h_fixed_lr,
                "image_v_fixed_lr": image_v_fixed_lr,
                "image_i_fixed_lr": image_i_fixed_lr,
                "enhanced"        : image_rgb_fixed,
            }
        else:
            return {
                "enhanced": image_rgb_fixed,
            }


@MODELS.register(name="zero_me2_12_hvid3_zsn2n", arch="zero_me2")
class ZeroME2_12_HVID3_ZSN2N(ZeroME2_11_HVID3):
    
    def __init__(
        self,
        name   : str = "zero_me2_12_hvid3_zsn2n",
        weights: Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name    = name,
            weights = weights,
            *args, **kwargs
        )
        
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        self.assert_datapoint(datapoint)
        image          = datapoint.get("image")
        image1, image2 = core.pair_downsample(image)
        datapoint1     = datapoint | {"image": image1}
        datapoint2     = datapoint | {"image": image2}
        outputs1       = self.forward(datapoint=datapoint1, *args, **kwargs)
        outputs2       = self.forward(datapoint=datapoint2, *args, **kwargs)
        outputs        = self.forward(datapoint=datapoint,  *args, **kwargs)
        self.assert_outputs(outputs)
        # Symmetric Loss
        illu_i_lr1        = outputs1["illu_i_lr"]
        image_i_lr1       = outputs1["image_i_lr"]
        image_i_fixed_lr1 = outputs1["image_i_fixed_lr"]
        image_rgb_fixed1  = outputs1["enhanced"]
        illu_i_lr2        = outputs2["illu_i_lr"]
        image_i_lr2       = outputs2["image_i_lr"]
        image_i_fixed_lr2 = outputs2["image_i_fixed_lr"]
        image_rgb_fixed2  = outputs2["enhanced"]
        illu_h_lr         = outputs["illu_h_lr"]
        illu_v_lr         = outputs["illu_v_lr"]
        illu_i_lr         = outputs["illu_i_lr"]
        image_h_lr        = outputs["image_h_lr"]
        image_v_lr        = outputs["image_v_lr"]
        image_i_lr        = outputs["image_i_lr"]
        image_h_fixed_lr  = outputs["image_h_fixed_lr"]
        image_v_fixed_lr  = outputs["image_v_fixed_lr"]
        image_i_fixed_lr  = outputs["image_i_fixed_lr"]
        image_rgb_fixed   = outputs["enhanced"]
        image_rgb_fixed_1, image_rgb_fixed_2 = core.pair_downsample(image_rgb_fixed)
        mse_loss = nn.MSELoss()
        loss_res = 0.5 * (mse_loss(image1,           image_rgb_fixed_2) + mse_loss(image2,           image_rgb_fixed_1))
        loss_con = 0.5 * (mse_loss(image_rgb_fixed1, image_rgb_fixed_1) + mse_loss(image_rgb_fixed2, image_rgb_fixed_2))
        loss_h   = self.loss(illu_h_lr, image_h_lr, image_h_fixed_lr)
        loss_v   = self.loss(illu_v_lr, image_v_lr, image_v_fixed_lr)
        loss_i   = self.loss(illu_i_lr, image_i_lr, image_i_fixed_lr)
        loss_enh = (loss_h + loss_v + loss_i) / 3
        loss     = 0.25 * loss_res + 0.25 * loss_con + 0.5 * loss_enh
        outputs["loss"] = loss
        # Return
        return outputs
    

@MODELS.register(name="zero_me2_13_rgbd_hsvd", arch="zero_me2")
class ZeroME2_13_RGBD_HSVD(ZeroME2):
    
    def __init__(
        self,
        name          : str   = "zero_me2_13_rgbd_hsvd",
        embed_channels: int   = 48,
        window_size   : int   = 1,
        down_size     : int   = 256,
        num_layers    : int   = 4,
        hidden_dim    : int   = 256,
        add_layer     : int   = 2,
        weight_decay  : list[float] = [0.1, 0.0001, 0.001],
        L             : float = 0.3,
        alpha         : float = 1,
        beta          : float = 20,
        gamma         : float = 8,
        delta         : float = 5,
        weights       : Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name    = name,
            weights = weights,
            *args, **kwargs
        )
        self.embed_channels = embed_channels
        self.window_size    = window_size
        self.patch_dim      = window_size ** 2
        self.down_size      = down_size
        self.omega_0        = 30.0
        self.siren_C        = 6.0
        
        patch_r_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_g_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_b_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_v_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_d_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_e_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        spatial_layers = [nn.SIREN(2,   hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        for _ in range(1, add_layer - 2):
            patch_r_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_g_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_b_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_v_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_d_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_e_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            spatial_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
        patch_r_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        patch_g_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        patch_b_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        patch_v_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        patch_d_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        patch_e_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        spatial_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        
        output_layers = []
        for _ in range(add_layer, num_layers - 1):
            output_layers.append(nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
        output_layers.append(nn.SIREN(hidden_dim, 1, self.omega_0, self.siren_C, is_last=True))
        
        self.patch_r_net = nn.Sequential(*patch_r_layers)
        self.patch_g_net = nn.Sequential(*patch_g_layers)
        self.patch_b_net = nn.Sequential(*patch_b_layers)
        self.patch_v_net = nn.Sequential(*patch_v_layers)
        self.patch_d_net = nn.Sequential(*patch_d_layers)
        self.patch_e_net = nn.Sequential(*patch_e_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net  = nn.Sequential(*output_layers)
        self.dba         = nn.BoundaryAwarePrior(eps=0.05, normalized=False)
        self.w_0         = nn.Parameter(torch.Tensor([0.5]))
        
        # Loss
        self.loss = Loss(L, alpha, beta, gamma, delta)
        
        weight_decay = weight_decay or [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{
	        "params"      : self.spatial_net.parameters(),
	        "weight_decay": weight_decay[0]
        }]
        self.params += [{
	        "params"      : self.patch_r_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.patch_g_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.patch_v_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.patch_b_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.patch_d_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.patch_e_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.output_net.parameters(),
	        "weight_decay": weight_decay[2]
        }]
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        self.initial_state_dict = self.state_dict()
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        # Prepare inputs
        image_rgb = datapoint.get("image")
        depth     = datapoint.get("depth")
        if depth is None:
            depth = core.rgb_to_grayscale(image_rgb)
        edge      = self.dba(depth)
        # Path 1: RGB
        image_r            = image_rgb[:, 0:1, :, :]
        image_g            = image_rgb[:, 1:2, :, :]
        image_b            = image_rgb[:, 2:3, :, :]
        image_rgb_lr       = self.interpolate_image(image_rgb)
        image_r_lr         = image_rgb_lr[:, 0:1, :, :]
        image_g_lr         = image_rgb_lr[:, 1:2, :, :]
        image_b_lr         = image_rgb_lr[:, 2:3, :, :]
        depth_lr           = self.interpolate_image(depth)
        edge_lr            = self.interpolate_image(edge)
        patch_r            = self.patch_r_net(self.get_patches(image_r_lr))
        patch_g            = self.patch_g_net(self.get_patches(image_g_lr))
        patch_b            = self.patch_b_net(self.get_patches(image_b_lr))
        patch_d            = self.patch_d_net(self.get_patches(depth_lr))
        patch_e            = self.patch_e_net(self.get_patches(edge_lr))
        spatial            = self.spatial_net(self.get_coords())
        illu_res_r_lr      = self.output_net(torch.cat([patch_r, patch_e, patch_d, spatial], -1))
        illu_res_g_lr      = self.output_net(torch.cat([patch_g, patch_e, patch_d, spatial], -1))
        illu_res_b_lr      = self.output_net(torch.cat([patch_b, patch_e, patch_d, spatial], -1))
        illu_res_r_lr      = illu_res_r_lr.view(1, 1, self.down_size, self.down_size)
        illu_res_g_lr      = illu_res_g_lr.view(1, 1, self.down_size, self.down_size)
        illu_res_b_lr      = illu_res_b_lr.view(1, 1, self.down_size, self.down_size)
        illu_res_lr        = torch.cat([illu_res_r_lr, illu_res_g_lr, illu_res_b_lr], 1)
        illu_r_lr          = illu_res_r_lr + image_r_lr
        illu_g_lr          = illu_res_g_lr + image_g_lr
        illu_b_lr          = illu_res_b_lr + image_b_lr
        illu_lr            = torch.cat([illu_r_lr, illu_g_lr, illu_b_lr], 1)
        image_r_fixed_lr   = image_r_lr / (illu_r_lr + 1e-4)
        image_g_fixed_lr   = image_g_lr / (illu_g_lr + 1e-4)
        image_b_fixed_lr   = image_b_lr / (illu_b_lr + 1e-4)
        image_r_fixed_lr   = torch.clamp(image_r_fixed_lr, 1e-4, 1)
        image_g_fixed_lr   = torch.clamp(image_g_fixed_lr, 1e-4, 1)
        image_b_fixed_lr   = torch.clamp(image_b_fixed_lr, 1e-4, 1)
        image_r_fixed_lr   = kornia.filters.bilateral_blur(image_r_fixed_lr, (3, 3), 0.1, (1.5, 1.5))
        image_g_fixed_lr   = kornia.filters.bilateral_blur(image_g_fixed_lr, (3, 3), 0.1, (1.5, 1.5))
        image_b_fixed_lr   = kornia.filters.bilateral_blur(image_b_fixed_lr, (3, 3), 0.1, (1.5, 1.5))
        image_rgb_fixed_lr = torch.cat([image_r_fixed_lr, image_g_fixed_lr, image_b_fixed_lr], 1)
        image_r_fixed      = self.filter_up(image_r_lr, image_r_fixed_lr, image_r)
        image_g_fixed      = self.filter_up(image_g_lr, image_g_fixed_lr, image_g)
        image_b_fixed      = self.filter_up(image_b_lr, image_b_fixed_lr, image_b)
        image_rgb_fixed1   = torch.cat([image_r_fixed, image_g_fixed, image_b_fixed], 1)
        image_rgb_fixed1   = image_rgb_fixed1 / torch.max(image_rgb_fixed1)
        # Path 2: HSV
        image_hsv          = core.rgb_to_hsv(image_rgb)
        image_v            = core.rgb_to_v(image_rgb)
        image_v_lr         = self.interpolate_image(image_v)
        patch_v            = self.patch_v_net(self.get_patches(image_v_lr))
        illu_res_v_lr      = self.output_net(torch.cat([patch_v, patch_e, patch_d, spatial], -1))
        illu_res_v_lr      = illu_res_v_lr.view(1, 1, self.down_size, self.down_size)
        illu_v_lr          = illu_res_v_lr + image_v_lr
        image_v_fixed_lr   = image_v_lr / (illu_v_lr + 1e-4)
        image_v_fixed_lr   = torch.clamp(image_v_fixed_lr, 1e-4, 1)
        # image_v_fixed_lr   = kornia.filters.bilateral_blur(image_v_fixed_lr, (3, 3), 0.1, (1.5, 1.5))
        image_v_fixed      = self.filter_up(image_v_lr, image_v_fixed_lr, image_v)
        image_hsv_fixed    = self.replace_v_component(image_hsv, image_v_fixed)
        image_rgb_fixed2   = core.hsv_to_rgb(image_hsv_fixed)
        image_rgb_fixed2   = image_rgb_fixed2 / torch.max(image_rgb_fixed2)
        # Combine
        # image_rgb_fixed    = self.lda([image_rgb_fixed1, image_rgb_fixed2])
        image_rgb_fixed    = image_rgb_fixed1 * self.w_0 + image_rgb_fixed2 * (1 - self.w_0)
        # Return
        if self.debug:
            return {
                "w_0"               : self.w_0,
                "illu_lr"           : illu_lr,
                "image_rgb_lr"      : image_rgb_lr,
                "image_rgb_fixed_lr": image_rgb_fixed_lr,
                "enhanced"          : image_rgb_fixed,
            }
        else:
            return {
                "enhanced": image_rgb_fixed,
            }


@MODELS.register(name="zero_me2_14_rgbd_hsvd_zsn2n", arch="zero_me2")
class ZeroME2_14_RGBD_HSVD_ZSN2N(ZeroME2):
    
    def __init__(
        self,
        name          : str   = "zero_me2_14_rgbd_hsvd_zsn2n",
        embed_channels: int   = 48,
        window_size   : int   = 1,
        down_size     : int   = 256,
        num_layers    : int   = 4,
        hidden_dim    : int   = 256,
        add_layer     : int   = 2,
        weight_decay  : list[float] = [0.1, 0.0001, 0.001],
        L             : float = 0.3,
        alpha         : float = 1,
        beta          : float = 20,
        gamma         : float = 8,
        delta         : float = 5,
        weights       : Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name    = name,
            weights = weights,
            *args, **kwargs
        )
        self.embed_channels = embed_channels
        self.window_size    = window_size
        self.patch_dim      = window_size ** 2
        self.down_size      = down_size
        self.omega_0        = 30.0
        self.siren_C        = 6.0
        
        patch_r_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_g_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_b_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_v_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_d_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_e_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        spatial_layers = [nn.SIREN(2,   hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        for _ in range(1, add_layer - 2):
            patch_r_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_g_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_b_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_v_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_d_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_e_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            spatial_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
        patch_r_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        patch_g_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        patch_b_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        patch_v_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        patch_d_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        patch_e_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        spatial_layers.append(nn.SIREN(hidden_dim, hidden_dim // 4, self.omega_0, self.siren_C))
        
        output_layers = []
        for _ in range(add_layer, num_layers - 1):
            output_layers.append(nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
        output_layers.append(nn.SIREN(hidden_dim, 1, self.omega_0, self.siren_C, is_last=True))
        
        self.patch_r_net = nn.Sequential(*patch_r_layers)
        self.patch_g_net = nn.Sequential(*patch_g_layers)
        self.patch_b_net = nn.Sequential(*patch_b_layers)
        self.patch_v_net = nn.Sequential(*patch_v_layers)
        self.patch_d_net = nn.Sequential(*patch_d_layers)
        self.patch_e_net = nn.Sequential(*patch_e_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net  = nn.Sequential(*output_layers)
        self.dba         = nn.BoundaryAwarePrior(eps=0.05, normalized=False)
        self.w_0         = nn.Parameter(torch.Tensor([0.5]))
        
        # Loss
        self.loss = Loss(L, alpha, beta, gamma, delta)
        
        weight_decay = weight_decay or [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{
	        "params"      : self.spatial_net.parameters(),
	        "weight_decay": weight_decay[0]
        }]
        self.params += [{
	        "params"      : self.patch_r_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.patch_g_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.patch_b_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.patch_v_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.patch_d_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.patch_e_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.output_net.parameters(),
	        "weight_decay": weight_decay[2]
        }]
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        self.initial_state_dict = self.state_dict()
        
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        self.assert_datapoint(datapoint)
        image          = datapoint.get("image")
        image1, image2 = core.pair_downsample(image)
        datapoint1     = datapoint | {"image": image1}
        datapoint2     = datapoint | {"image": image2}
        outputs1       = self.forward(datapoint=datapoint1, *args, **kwargs)
        outputs2       = self.forward(datapoint=datapoint2, *args, **kwargs)
        outputs        = self.forward(datapoint=datapoint,  *args, **kwargs)
        self.assert_outputs(outputs)
        # Symmetric Loss
        illu_lr1, image_rgb_lr1, image_rgb_fixed_lr1, image_rgb_fixed1 = outputs1.values()
        illu_lr2, image_rgb_lr2, image_rgb_fixed_lr2, image_rgb_fixed2 = outputs2.values()
        illu_lr , image_rgb_lr , image_rgb_fixed_lr ,  image_rgb_fixed = outputs.values()
        image_rgb_fixed_1, image_rgb_fixed_2 = core.pair_downsample(image_rgb_fixed)
        mse_loss = nn.MSELoss()
        loss_res = 0.5 * (mse_loss(image1,           image_rgb_fixed_2) + mse_loss(image2,           image_rgb_fixed_1))
        loss_con = 0.5 * (mse_loss(image_rgb_fixed1, image_rgb_fixed_1) + mse_loss(image_rgb_fixed2, image_rgb_fixed_2))
        loss_enh = self.loss(illu_lr, image_rgb_lr, image_rgb_fixed_lr)
        loss     = 0.15 * loss_res + 0.15 * loss_con + 0.7 * loss_enh
        outputs["loss"] = loss
        # Return
        return outputs
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        # Prepare inputs
        image_rgb = datapoint.get("image")
        depth     = datapoint.get("depth")
        if depth is None:
            depth = core.rgb_to_grayscale(image_rgb)
        edge      = self.dba(depth)
        # Path 1: RGB
        image_r            = image_rgb[:, 0:1, :, :]
        image_g            = image_rgb[:, 1:2, :, :]
        image_b            = image_rgb[:, 2:3, :, :]
        image_rgb_lr       = self.interpolate_image(image_rgb)
        image_r_lr         = image_rgb_lr[:, 0:1, :, :]
        image_g_lr         = image_rgb_lr[:, 1:2, :, :]
        image_b_lr         = image_rgb_lr[:, 2:3, :, :]
        depth_lr           = self.interpolate_image(depth)
        edge_lr            = self.interpolate_image(edge)
        patch_r            = self.patch_r_net(self.get_patches(image_r_lr))
        patch_g            = self.patch_g_net(self.get_patches(image_g_lr))
        patch_b            = self.patch_b_net(self.get_patches(image_b_lr))
        patch_d            = self.patch_d_net(self.get_patches(depth_lr))
        patch_e            = self.patch_e_net(self.get_patches(edge_lr))
        spatial            = self.spatial_net(self.get_coords())
        illu_res_r_lr      = self.output_net(torch.cat([patch_r, patch_e, patch_d, spatial], -1))
        illu_res_g_lr      = self.output_net(torch.cat([patch_g, patch_e, patch_d, spatial], -1))
        illu_res_b_lr      = self.output_net(torch.cat([patch_b, patch_e, patch_d, spatial], -1))
        illu_res_r_lr      = illu_res_r_lr.view(1, 1, self.down_size, self.down_size)
        illu_res_g_lr      = illu_res_g_lr.view(1, 1, self.down_size, self.down_size)
        illu_res_b_lr      = illu_res_b_lr.view(1, 1, self.down_size, self.down_size)
        illu_res_lr        = torch.cat([illu_res_r_lr, illu_res_g_lr, illu_res_b_lr], 1)
        illu_r_lr          = illu_res_r_lr + image_r_lr
        illu_g_lr          = illu_res_g_lr + image_g_lr
        illu_b_lr          = illu_res_b_lr + image_b_lr
        illu_lr            = torch.cat([illu_r_lr, illu_g_lr, illu_b_lr], 1)
        image_r_fixed_lr   = image_r_lr / (illu_r_lr + 1e-4)
        image_g_fixed_lr   = image_g_lr / (illu_g_lr + 1e-4)
        image_b_fixed_lr   = image_b_lr / (illu_b_lr + 1e-4)
        image_r_fixed_lr   = torch.clamp(image_r_fixed_lr, 1e-4, 1)
        image_g_fixed_lr   = torch.clamp(image_g_fixed_lr, 1e-4, 1)
        image_b_fixed_lr   = torch.clamp(image_b_fixed_lr, 1e-4, 1)
        image_r_fixed_lr   = kornia.filters.bilateral_blur(image_r_fixed_lr, (3, 3), 0.1, (1.5, 1.5))
        image_g_fixed_lr   = kornia.filters.bilateral_blur(image_g_fixed_lr, (3, 3), 0.1, (1.5, 1.5))
        image_b_fixed_lr   = kornia.filters.bilateral_blur(image_b_fixed_lr, (3, 3), 0.1, (1.5, 1.5))
        image_rgb_fixed_lr = torch.cat([image_r_fixed_lr, image_g_fixed_lr, image_b_fixed_lr], 1)
        image_r_fixed      = self.filter_up(image_r_lr, image_r_fixed_lr, image_r)
        image_g_fixed      = self.filter_up(image_g_lr, image_g_fixed_lr, image_g)
        image_b_fixed      = self.filter_up(image_b_lr, image_b_fixed_lr, image_b)
        image_rgb_fixed1   = torch.cat([image_r_fixed, image_g_fixed, image_b_fixed], 1)
        image_rgb_fixed1   = image_rgb_fixed1 / torch.max(image_rgb_fixed1)
        # Path 2: HSV
        image_hsv          = core.rgb_to_hsv(image_rgb)
        image_v            = core.rgb_to_v(image_rgb)
        image_v_lr         = self.interpolate_image(image_v)
        patch_v            = self.patch_v_net(self.get_patches(image_v_lr))
        illu_res_v_lr      = self.output_net(torch.cat([patch_v, patch_e, patch_d, spatial], -1))
        illu_res_v_lr      = illu_res_v_lr.view(1, 1, self.down_size, self.down_size)
        illu_v_lr          = illu_res_v_lr + image_v_lr
        image_v_fixed_lr   = image_v_lr / (illu_v_lr + 1e-4)
        image_v_fixed_lr   = torch.clamp(image_v_fixed_lr, 1e-4, 1)
        # image_v_fixed_lr   = kornia.filters.bilateral_blur(image_v_fixed_lr, (3, 3), 0.1, (1.5, 1.5))
        image_v_fixed      = self.filter_up(image_v_lr, image_v_fixed_lr, image_v)
        image_hsv_fixed    = self.replace_v_component(image_hsv, image_v_fixed)
        image_rgb_fixed2   = core.hsv_to_rgb(image_hsv_fixed)
        image_rgb_fixed2   = image_rgb_fixed2 / torch.max(image_rgb_fixed2)
        # Combine
        image_rgb_fixed    = image_rgb_fixed1 * self.w_0 + image_rgb_fixed2 * (1 - self.w_0)
        # Return
        if self.debug:
            return {
                "illu_lr"           : illu_lr,
                "image_rgb_lr"      : image_rgb_lr,
                "image_rgb_fixed_lr": image_rgb_fixed_lr,
                "enhanced"          : image_rgb_fixed,
            }
        else:
            return {
                "enhanced": image_rgb_fixed,
            }

# endregion
