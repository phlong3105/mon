#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CoLIE-RGBD.

Test new idea: starting with CoLIE, instead of using the HSV color space, we
use the RGB color space and depth.
"""

from __future__ import annotations

__all__ = [
    "CoLIE_RGBD",
]

from typing import Any

import kornia.filters
import numpy as np
import torch
from torch.nn import functional as F

from mon import core, nn
from mon.globals import MODELS, Scheme, Task
from mon.vision import filtering
from mon.vision.enhance import base
from mon.vision.enhance.llie.colie.colie import Loss

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Model

@MODELS.register(name="colie_rgbd", arch="colie")
class CoLIE_RGBD(base.ImageEnhancementModel):

    model_dir: core.Path    = current_dir
    arch     : str          = "colie"
    tasks    : list[Task]   = [Task.LLIE]
    schemes  : list[Scheme] = [Scheme.ZERO_REFERENCE, Scheme.INSTANCE]
    zoo      : dict         = {}
    
    def __init__(
        self,
        name        : str   = "colie_rgbd",
        window_size : int   = 1,
        down_size   : int   = 256,
        num_layers  : int   = 4,
        hidden_dim  : int   = 256,
        add_layer   : int   = 2,
        weight_decay: list[float] = [0.1, 0.0001, 0.001],
        L           : float = 0.3,
        alpha       : float = 1,
        beta        : float = 20,
        gamma       : float = 8,
        delta       : float = 5,
        weights     : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            name    = name,
            weights = weights,
            *args, **kwargs
        )
        self.window_size = window_size
        self.patch_dim   = window_size ** 2
        self.down_size   = down_size
        self.omega_0     = 30.0
        self.siren_C     = 6.0
        
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
        image_rgb       = datapoint.get("image")
        depth           = datapoint.get("depth")
        if depth is None:
            depth = core.rgb_to_grayscale(image_rgb)
        edge            = self.dba(depth)
        image_r         = image_rgb[:, 0:1, :, :]
        image_g         = image_rgb[:, 1:2, :, :]
        image_b         = image_rgb[:, 2:3, :, :]
        image_rgb_lr    = self.interpolate_image(image_rgb)
        depth_lr        = self.interpolate_image(depth)
        edge_lr         = self.interpolate_image(edge)
        image_r_lr      = image_rgb_lr[:, 0:1, :, :]
        image_g_lr      = image_rgb_lr[:, 1:2, :, :]
        image_b_lr      = image_rgb_lr[:, 2:3, :, :]
        patch_r         = self.get_patches(image_r_lr)
        patch_g         = self.get_patches(image_g_lr)
        patch_b         = self.get_patches(image_b_lr)
        patch_d         = self.get_patches(depth_lr)
        patch_e         = self.get_patches(edge_lr)
        spatial         = self.get_coords()
        illu_res_r_lr   = self.output_net(torch.cat([self.patch_r_net(patch_r), self.patch_e_net(patch_e), self.patch_d_net(patch_d), self.spatial_net(spatial)], -1))
        illu_res_g_lr   = self.output_net(torch.cat([self.patch_g_net(patch_g), self.patch_e_net(patch_e), self.patch_d_net(patch_d), self.spatial_net(spatial)], -1))
        illu_res_b_lr   = self.output_net(torch.cat([self.patch_b_net(patch_b), self.patch_e_net(patch_e), self.patch_d_net(patch_d), self.spatial_net(spatial)], -1))
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
        
    def interpolate_image(self, image: torch.Tensor) -> torch.Tensor:
        """Reshapes the image based on new resolution."""
        return F.interpolate(image, size=(self.down_size, self.down_size))
    
    def get_patches(self, image: torch.Tensor) -> torch.Tensor:
        """Creates a tensor where the channel contains patch information."""
        kernel = torch.zeros((self.window_size ** 2, 1, self.window_size, self.window_size)).cuda()
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
    def replace_v_component(
        image_hsv: torch.Tensor,
        v_new    : torch.Tensor
    ) -> torch.Tensor:
        """Replaces the `V` component of an HSV image `[1, 3, H, W]`."""
        image_hsv[:, -1, :, :] = v_new
        return image_hsv
    
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
