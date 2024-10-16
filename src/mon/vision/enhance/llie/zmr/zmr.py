#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ZMR.

This module implement our idea: Zero-shot Multimodal Retinex Model for Low-light
Image Enhancement via Neural Implicit Representations.
"""

from __future__ import annotations

__all__ = [
    "ZMR",
]

from abc import ABC
from typing import Any, Literal

import cv2
import numpy as np
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
        epsilon  : float = 1,
        zeta     : float = 1,
        eta      : float = 5000,
        reduction: Literal["none", "mean", "sum"] = "mean",
        *args, **kwargs
    ):
        super().__init__(reduction=reduction, *args, **kwargs)
        self.epsilon = epsilon
        self.zeta    = zeta
        self.eta     = eta
        
    def forward(
        self,
        image       : torch.Tensor,
        illumination: torch.Tensor,
        reflectance : torch.Tensor,
        noise       : torch.Tensor,
    ) -> torch.Tensor:
        loss_recons   =      self.reconstruction_loss(image, illumination, reflectance, noise)
        loss_illu     = self.illumination_smooth_loss(image, illumination, reflectance, noise)
        loss_reflect  =  self.reflectance_smooth_loss(image, illumination, reflectance, noise)
        loss_noise    =               self.noise_loss(image, illumination, reflectance, noise)
        loss          = (
              loss_recons
            + loss_illu     * self.epsilon
            + loss_reflect  * self.zeta
            + loss_noise    * self.eta
        )
        return loss
    
    def reconstruction_loss(
        self,
        image       : torch.Tensor,
        illumination: torch.Tensor,
        reflectance : torch.Tensor,
        noise       : torch.Tensor,
    ) -> torch.Tensor:
        reconstructed_image = illumination * reflectance + noise
        return torch.norm(image - reconstructed_image, 1)
    
    def illumination_smooth_loss(
        self,
        image       : torch.Tensor,
        illumination: torch.Tensor,
        reflectance : torch.Tensor,
        noise       : torch.Tensor,
    ) -> torch.Tensor:
        g_kernel_size   = 5
        g_padding       = 2
        sigma           = 3
        kx              = cv2.getGaussianKernel(g_kernel_size, sigma)
        ky              = cv2.getGaussianKernel(g_kernel_size, sigma)
        gaussian_kernel = np.multiply(kx, np.transpose(ky))
        gaussian_kernel = torch.FloatTensor(gaussian_kernel).unsqueeze(0).unsqueeze(0).to(image.device)
        gray_tensor     = 0.299 * image[0, 0, :, :] + 0.587 * image[0, 1, :, :] + 0.114 * image[0, 2, :, :]
        max_rgb, _      = torch.max(image, 1)
        max_rgb         = max_rgb.unsqueeze(1)
        gradient_gray_h, gradient_gray_w = self.gradient(gray_tensor.unsqueeze(0).unsqueeze(0))
        gradient_illu_h, gradient_illu_w = self.gradient(illumination)
        weight_h = 1 / (F.conv2d(gradient_gray_h, weight=gaussian_kernel, padding=g_padding) + 0.0001)
        weight_w = 1 / (F.conv2d(gradient_gray_w, weight=gaussian_kernel, padding=g_padding) + 0.0001)
        weight_h.detach()
        weight_w.detach()
        loss_h = weight_h * gradient_illu_h
        loss_w = weight_w * gradient_illu_w
        max_rgb.detach()
        return loss_h.sum() + loss_w.sum() + torch.norm(illumination-max_rgb, 1)
    
    def reflectance_smooth_loss(
        self,
        image       : torch.Tensor,
        illumination: torch.Tensor,
        reflectance : torch.Tensor,
        noise       : torch.Tensor,
    ) -> torch.Tensor:
        reffac      = 1
        gray_tensor = 0.299 * image[0, 0, :, :] + 0.587 * image[0, 1, :, :] + 0.114 * image[0, 2, :, :]
        gradient_gray_h,    gradient_gray_w    = self.gradient(gray_tensor.unsqueeze(0).unsqueeze(0))
        gradient_reflect_h, gradient_reflect_w = self.gradient(reflectance)
        weight = 1 / (illumination * gradient_gray_h * gradient_gray_w + 0.0001)
        weight = self.normalize01(weight)
        weight.detach()
        loss_h           = weight * gradient_reflect_h
        loss_w           = weight * gradient_reflect_w
        refrence_reflect = image / illumination
        refrence_reflect.detach()
        return loss_h.sum() + loss_w.sum() + reffac*torch.norm(refrence_reflect - reflectance, 1)
    
    def noise_loss(
        self,
        image       : torch.Tensor,
        illumination: torch.Tensor,
        reflectance : torch.Tensor,
        noise       : torch.Tensor,
    ) -> torch.Tensor:
        weight_illu = illumination
        weight_illu.detach()
        loss = weight_illu * noise
        return torch.norm(loss, 2)
    
    def gradient(self, img):
        height      = img.size(2)
        width       = img.size(3)
        gradient_h  = (img[:, :, 2:, :] - img[:, :, :height - 2, :]).abs()
        gradient_w  = (img[:, :, :, 2:] - img[:, :, :, :width-2]).abs()
        gradient_h  = F.pad(gradient_h, [0, 0, 1, 1], "replicate")
        gradient_w  = F.pad(gradient_w, [1, 1, 0, 0], "replicate")
        gradient2_h = (img[:, :, 4:, :] - img[:, :, :height - 4, :]).abs()
        gradient2_w = (img[:, :, :, 4:] - img[:, :, :, :width-4]).abs()
        gradient2_h = F.pad(gradient2_h, [0, 0, 2, 2], "replicate")
        gradient2_w = F.pad(gradient2_w, [2, 2, 0, 0], "replicate")
        return gradient_h * gradient2_h, gradient_w * gradient2_w
    
    def normalize01(self, img):
        minv = img.min()
        maxv = img.max()
        return (img-minv)/(maxv-minv)
    
    def gaussianblur3(self, input):
        g_kernel_size   = 5
        g_padding       = 2
        sigma           = 3
        kx              = cv2.getGaussianKernel(g_kernel_size, sigma)
        ky              = cv2.getGaussianKernel(g_kernel_size, sigma)
        gaussian_kernel = np.multiply(kx, np.transpose(ky))
        gaussian_kernel = torch.FloatTensor(gaussian_kernel).unsqueeze(0).unsqueeze(0).to(input.device)
        slice1 = F.conv2d(input[:, 0, :, :].unsqueeze(1), weight=gaussian_kernel, padding=g_padding)
        slice2 = F.conv2d(input[:, 1, :, :].unsqueeze(1), weight=gaussian_kernel, padding=g_padding)
        slice3 = F.conv2d(input[:, 2, :, :].unsqueeze(1), weight=gaussian_kernel, padding=g_padding)
        x      = torch.cat([slice1, slice2, slice3], dim=1)
        return x

# endregion


# region Modules

class RRDNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.illumination_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
        )
        self.reflectance_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1)
        )
        self.noise_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1)
        )
        
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        illumination = torch.sigmoid(self.illumination_net(image))
        reflectance  = torch.sigmoid(self.reflectance_net(image))
        noise        = torch.tanh(self.noise_net(image))
        return illumination, reflectance, noise


class INF(nn.Module, ABC):
    
    def interpolate_image(self, image: torch.Tensor) -> torch.Tensor:
        """Reshapes the image based on new resolution."""
        return F.interpolate(image, size=(self.down_size, self.down_size), mode="bicubic")
    
    def get_patches(self, image: torch.Tensor) -> torch.Tensor:
        """Creates a tensor where the channel contains patch information."""
        num_channels = core.get_image_num_channels(image)
        kernel       = torch.zeros((self.window_size ** 2, num_channels, self.window_size, self.window_size)).to(image.device)
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
        coords = torch.from_numpy(coords).float()
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


class INF_Retinex(INF):
    
    def __init__(
        self,
        window_size : int   = 1,
        down_size   : int   = 256,
        num_layers  : int   = 4,
        hidden_dim  : int   = 256,
        add_layer   : int   = 2,
        gamma       : float = 0.4,
        weight_decay: list[float] = [0.1, 0.0001, 0.001],
    ):
        super().__init__()
        self.window_size = window_size
        self.patch_dim   = window_size ** 2
        self.down_size   = down_size
        self.gamma       = gamma
        self.omega_0     = 30.0
        self.siren_C     = 6.0
        
        patch_layers       = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_depth_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_edge_layers  = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        spatial_layers     = [nn.SIREN(2,   hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        for _ in range(1, add_layer - 2):
            patch_layers.append(      nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
            patch_depth_layers.append(nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
            patch_edge_layers.append( nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
            spatial_layers.append(    nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
        patch_layers.append(      nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
        patch_depth_layers.append(nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
        patch_edge_layers.append( nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
        spatial_layers.append(    nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
        
        output_layers = []
        for _ in range(add_layer, num_layers - 1):
            output_layers.append(nn.SIREN(hidden_dim * 4, hidden_dim, self.omega_0, self.siren_C))
        output_layers.append(nn.SIREN(hidden_dim, 3, self.omega_0, self.siren_C, is_last=True))
        
        self.rrdnet          = RRDNet()
        self.dba             = nn.BoundaryAwarePrior(eps=0.05, normalized=False)
        self.patch_net       = nn.Sequential(*patch_layers)
        self.patch_depth_net = nn.Sequential(*patch_depth_layers)
        self.patch_edge_net  = nn.Sequential(*patch_edge_layers)
        self.spatial_net     = nn.Sequential(*spatial_layers)
        self.output_net      = nn.Sequential(*output_layers)
        
        weight_decay = weight_decay or [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{"params": self.spatial_net.parameters(),     "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patch_net.parameters(),       "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.patch_depth_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.patch_edge_net.parameters(),  "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),      "weight_decay": weight_decay[2]}]
        
    def forward(self, image: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        # Prepare input
        if depth is None:
            depth = core.rgb_to_grayscale(image)
        edge = self.dba(depth)
        # Retinex Decomposition
        image_lr = self.interpolate_image(image)
        depth_lr = self.interpolate_image(depth)
        edge_lr  = self.interpolate_image(edge)
        illu_lr, reflect_lr, noise_lr = self.rrdnet(image_lr)
        patch_illu  = self.patch_net(self.get_patches(illu_lr))
        patch_depth = self.patch_depth_net(self.get_patches(depth_lr))
        patch_edge  = self.patch_edge_net(self.get_patches(edge_lr))
        spatial     = self.spatial_net(self.get_coords().to(image.device))
        illu_res_lr = self.output_net(torch.cat([patch_illu, patch_depth, patch_edge, spatial], -1))
        illu_res_lr = illu_res_lr.view(1, 3, self.down_size, self.down_size)
        # Retinex model
        illu_fixed_lr = illu_res_lr + image_lr
        image_fixed_lr  = (image_lr - noise_lr) / (illu_fixed_lr + 1e-8)
        # illu_fixed_lr  = torch.pow(illu_res_lr, self.gamma)
        # image_fixed_lr = illu_fixed_lr * ((image_lr - noise_lr) / illu_res_lr)
        # image_fixed_lr = torch.clamp(image_fixed_lr, min=0, max=1)
        
        enhanced       = self.filter_up(image_lr, image_fixed_lr, image)
        # Return
        return {
            "image"         : image,
            "depth"         : depth,
            "edge"          : edge,
            "image_lr"      : image_lr,
            "depth_lr"      : depth_lr,
            "edge_lr"       : edge_lr,
            "illu_lr"       : illu_lr,
            "reflect_lr"    : reflect_lr,
            "noise_lr"      : noise_lr,
            "illu_res_lr"   : illu_res_lr,
            "illu_fixed_lr" : illu_fixed_lr,
            "image_fixed_lr": image_fixed_lr,
            "enhanced"      : enhanced,
        }

# endregion


# region Model

@MODELS.register(name="zmr", arch="zmr")
class ZMR(base.ImageEnhancementModel):
    
    model_dir: core.Path    = current_dir
    arch     : str          = "zmr"
    tasks    : list[Task]   = [Task.LLIE]
    schemes  : list[Scheme] = [Scheme.ZERO_REFERENCE, Scheme.INSTANCE]
    zoo      : dict         = {}
    
    def __init__(
        self,
        name        : str   = "zmr",
        window_size : int   = 1,
        down_size   : int   = 256,
        num_layers  : int   = 4,
        hidden_dim  : int   = 256,
        add_layer   : int   = 2,
        gamma       : float = 0.4,
        weight_decay: list[float] = [0.1, 0.0001, 0.001],
        use_pse     : bool  = False,
        number_refs : int   = 1,
        tv_weight   : float = 5,
        weights     : Any   = None,
        *args, **kwargs
    ):
        super().__init__(name=name, weights=weights, *args, **kwargs)
        self.gamma       = gamma
        self.use_pse     = use_pse
        self.number_refs = number_refs
        self.tv_weight   = tv_weight
    
        self.model = INF_Retinex(
            window_size  = window_size,
            down_size    = down_size,
            num_layers   = num_layers,
            hidden_dim   = hidden_dim,
            add_layer    = add_layer,
            weight_decay = weight_decay,
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
        self.loss = Loss()
        self.mse  = nn.MSELoss()
        
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
                image_lr        = outputs["image_lr"]
                illu_lr         = outputs["illu_lr"]
                reflect_lr      = outputs["reflect_lr"]
                noise_lr        = outputs["noise_lr"]
                enhanced        = outputs["enhanced"]
                pseudo_gt       = self.saved_pseudo_gt
                recon_loss      = self.mse(enhanced, pseudo_gt)
                loss            = self.loss(image_lr, illu_lr, reflect_lr, noise_lr)
                outputs["loss"] = recon_loss + loss  # * self.tv_weight
            else:  # Skip updating model's weight at the first batch
                outputs = {"loss": None}
            # Saving n-th input and n-th pseudo gt
            self.saved_input     = nth_input
            self.saved_pseudo_gt = nth_pseudo_gt
        else:
            outputs         = self.forward(datapoint=datapoint, *args, **kwargs)
            image_lr        = outputs["image_lr"]
            illu_lr         = outputs["illu_lr"]
            reflect_lr      = outputs["reflect_lr"]
            noise_lr        = outputs["noise_lr"]
            outputs["loss"] = self.loss(image_lr, illu_lr, reflect_lr, noise_lr)
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
        epochs       : int   = 1000,
        lr           : float = 0.001,
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
                lr    = lr,
                betas = (0.9, 0.999),
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
            self.zero_grad()
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
