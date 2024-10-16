#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Zero-MLIE.

This module implement our idea: Zero-shot Multimodal Low-light Image Enhancement
via Neural Implicit Representations.
"""

from __future__ import annotations

__all__ = [
    "ZeroMLIE",
    "ZeroMLIE_RGB",
    "ZeroMLIE_HVI",
]

from typing import Any, Literal

import kornia
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
        L          : float = 0.5,
        alpha      : float = 1,
        beta       : float = 20,
        gamma      : float = 8,
        delta      : float = 5,
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
        self.l_tv  = nn.TotalVariationLoss(reduction=reduction)
        
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
              self.alpha * loss_spa
            + self.beta  * loss_tv
            + self.gamma * loss_exp
            + self.delta * loss_sparsity
        )
        return loss


class TVLoss(nn.Loss):

    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction  : Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)
        
    def forward(self, input: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        x    = input
        x    = torch.log(x + 1e-3)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :-1, :]), 2)
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2)
        loss = torch.mean(h_tv) + torch.mean(w_tv)
        loss = loss * self.loss_weight
        return loss

# endregion


# region Model

@MODELS.register(name="zero_mlie", arch="zero_mlie")
class ZeroMLIE(base.ImageEnhancementModel):
    
    model_dir: core.Path    = current_dir
    arch     : str          = "zero_mlie"
    tasks    : list[Task]   = [Task.LLIE]
    schemes  : list[Scheme] = [Scheme.ZERO_REFERENCE, Scheme.INSTANCE]
    zoo      : dict         = {}
    
    def __init__(
        self,
        name          : str         = "zero_mlie",
        window_size   : int         = 1,
        down_size     : int         = 256,
        num_layers    : int         = 4,
        hidden_dim    : int         = 256,
        add_layer     : int         = 2,
        weight_decay  : list[float] = [0.1, 0.0001, 0.001],
        use_depth     : bool        = False,
        use_edge      : bool        = False,
        use_denoise   : bool        = False,
        use_pse       : bool        = False,
        number_refs   : int         = 1,
        tv_weight     : float       = 5,
        L             : float       = 0.1,
        alpha         : float       = 1,
        beta          : float       = 20,
        gamma         : float       = 8,
        delta         : float       = 5,
        weights       : Any         = None,
        *args, **kwargs
    ):
        super().__init__(name=name, weights=weights, *args, **kwargs)
        self.window_size = window_size
        self.patch_dim   = window_size ** 2
        self.down_size   = down_size
        self.omega_0     = 30.0
        self.siren_C     = 6.0
        self.use_depth   = use_depth
        self.use_edge    = use_edge
        self.use_denoise = use_denoise
        self.use_pse     = use_pse
        self.number_refs = number_refs
        self.tv_weight   = tv_weight
        
        patch_layers   = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        spatial_layers = [nn.SIREN(2,   hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        for _ in range(1, add_layer - 2):
            patch_layers.append(   nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
            spatial_layers.append( nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
        patch_layers.append(  nn.SIREN(hidden_dim, hidden_dim // 2, self.omega_0, self.siren_C))
        spatial_layers.append(nn.SIREN(hidden_dim, hidden_dim // 2, self.omega_0, self.siren_C))
        
        output_layers = []
        for _ in range(add_layer, num_layers - 1):
            output_layers.append(nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
        output_layers.append(nn.SIREN(hidden_dim, 1, self.omega_0, self.siren_C, is_last=True))
        
        self.patch_net   = nn.Sequential(*patch_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net  = nn.Sequential(*output_layers)
        
        self.dba = nn.BoundaryAwarePrior(eps=0.05, normalized=False)
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
        self.loss = Loss(L, alpha, beta, gamma, delta)
        # self.loss = TVLoss(reduction="mean")
        self.mse  = nn.MSELoss()
        
        weight_decay = weight_decay or [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{"params": self.spatial_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patch_net.parameters(),   "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),  "weight_decay": weight_decay[2]}]
        
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
                outputs          = self.forward(datapoint=self.saved_input, *args, **kwargs)
                enhanced         = outputs["enhanced"]
                illu_v_lr        = outputs["illu_v_lr"]
                image_v_lr       = outputs["image_v_lr"]
                image_v_fixed_lr = outputs["image_v_fixed_lr"]
                pseudo_gt        = self.saved_pseudo_gt
                recon_loss       = self.mse(enhanced, pseudo_gt)
                tv_loss          = self.loss(illu_v_lr, image_v_lr, image_v_fixed_lr)
                loss             = recon_loss + tv_loss  # * self.tv_weight
                outputs["loss"]  = loss
            else:  # Skip updating model's weight at the first batch
                outputs = {"loss": None}
            # Saving n-th input and n-th pseudo gt
            self.saved_input     = nth_input
            self.saved_pseudo_gt = nth_pseudo_gt
        else:
            outputs          = self.forward(datapoint=datapoint, *args, **kwargs)
            enhanced         = outputs["enhanced"]
            illu_v_lr        = outputs["illu_v_lr"]
            image_v_lr       = outputs["image_v_lr"]
            image_v_fixed_lr = outputs["image_v_fixed_lr"]
            outputs["loss"]  = self.loss(illu_v_lr, image_v_lr, image_v_fixed_lr)
        return outputs
        
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        # Prepare input
        self.assert_datapoint(datapoint)
        image = datapoint.get("image")
        depth = datapoint.get("depth")
        if depth is None:
            depth = core.rgb_to_grayscale(image)
        edge  = self.dba(depth)
        # Enhance
        image_hsv        = core.rgb_to_hsv(image)
        image_v          = core.rgb_to_v(image)
        image_v_lr       = self.interpolate_image(image_v)
        # image_lr         = self.interpolate_image(image)
        depth_lr         = self.interpolate_image(depth)
        edge_lr          = self.interpolate_image(edge)
        concat_lr        = image_v_lr
        if self.use_depth:
            concat_lr = torch.cat([concat_lr, depth_lr], 1)
        if self.use_edge:
            concat_lr = torch.cat([concat_lr, edge_lr], 1)
        patch            = self.patch_net( self.get_patches(concat_lr))
        spatial          = self.spatial_net(self.get_coords())
        illu_res_lr      = self.output_net(torch.cat([patch, spatial], -1))
        illu_res_lr      = illu_res_lr.view(1, 1, self.down_size, self.down_size)
        illu_v_lr        = illu_res_lr + image_v_lr
        image_v_fixed_lr = image_v_lr / (illu_v_lr + 1e-8)
        if self.use_denoise:
            image_v_fixed_lr = kornia.filters.bilateral_blur(image_v_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
        image_v_fixed    = self.filter_up(image_v_lr, image_v_fixed_lr, image_v)
        image_hsv_fixed  = self.replace_v_component(image_hsv, image_v_fixed)
        image_rgb_fixed  = core.hsv_to_rgb(image_hsv_fixed)
        enhanced         = image_rgb_fixed / torch.max(image_rgb_fixed)
        # enhanced         = kornia.filters.bilateral_blur(enhanced, bilateral_ksize, bilateral_color, bilateral_space)
        # adjust_lr   = adjust_lr.view(1, 3, self.down_size, self.down_size)
        # adjust_lr   = torch.abs(adjust_lr + 1)
        # enhanced_lr = 1 - (1 - image_lr) ** adjust_lr
        # enhanced_lr = kornia.filters.bilateral_blur(enhanced_lr, bilateral_ksize, bilateral_color, bilateral_space)
        # enhanced    = self.filter_up(image_lr, enhanced_lr, image)
        # Return
        if self.debug:
            return {
                "image"           : image,
                "depth"           : depth,
                "edge"            : edge,
                "pseudo_gt"       : self.saved_pseudo_gt,
                "illu_res_lr"     : illu_res_lr,
                "illu_v_lr"       : illu_v_lr,
                "image_v_lr"      : image_v_lr,
                "image_v_fixed_lr": image_v_fixed_lr,
                "enhanced"        : enhanced,
            }
        else:
            return {
                "pseudo_gt": self.saved_pseudo_gt,
                "enhanced" : enhanced,
            }
        
    def interpolate_image(self, image: torch.Tensor) -> torch.Tensor:
        """Reshapes the image based on new resolution."""
        return F.interpolate(image, size=(self.down_size, self.down_size), mode="bicubic")
    
    def get_patches(self, image: torch.Tensor) -> torch.Tensor:
        """Creates a tensor where the channel contains patch information."""
        num_channels = core.get_image_num_channels(image)
        kernel       = torch.zeros((self.window_size ** 2, num_channels, self.window_size, self.window_size)).to(self.device)
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
        coords = torch.from_numpy(coords).float().to(self.device)
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


@MODELS.register(name="zero_mlie_rgb", arch="zero_mlie")
class ZeroMLIE_RGB(ZeroMLIE):
    
    def __init__(
        self,
        name        : str         = "zero_mlie_rgb",
        window_size : int         = 1,
        down_size   : int         = 256,
        num_layers  : int         = 4,
        hidden_dim  : int         = 256,
        add_layer   : int         = 2,
        weight_decay: list[float] = [0.1, 0.0001, 0.001],
        use_depth   : bool        = False,
        use_edge    : bool        = False,
        use_denoise : bool        = False,
        use_pse     : bool        = False,
        number_refs : int         = 1,
        tv_weight   : float       = 5,
        L           : float       = 0.3,
        alpha       : float       = 1,
        beta        : float       = 20,
        gamma       : float       = 8,
        delta       : float       = 5,
        weights     : Any         = None,
        *args, **kwargs
    ):
        super().__init__(name=name, weights=weights, *args, **kwargs)
        self.window_size = window_size
        self.patch_dim   = window_size ** 2
        self.down_size   = down_size
        self.omega_0     = 30.0
        self.siren_C     = 6.0
        self.use_depth   = use_depth
        self.use_edge    = use_edge
        self.use_denoise = use_denoise
        self.use_pse     = use_pse
        self.number_refs = number_refs
        self.tv_weight   = tv_weight
        
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
        
        self.dba = nn.BoundaryAwarePrior(eps=0.05, normalized=False)
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
        self.loss = Loss(L, alpha, beta, gamma, delta)
        # self.loss = TVLoss(reduction="mean")
        self.mse  = nn.MSELoss()
        
        weight_decay = weight_decay or [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{"params": self.spatial_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patch_r_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.patch_g_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.patch_b_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.patch_d_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.patch_e_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),  "weight_decay": weight_decay[2]}]
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        self.initial_state_dict = self.state_dict()
    
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
                outputs            = self.forward(datapoint=self.saved_input, *args, **kwargs)
                enhanced           = outputs["enhanced"]
                illu_rgb_lr        = outputs["illu_rgb_lr"]
                image_rgb_lr       = outputs["image_rgb_lr"]
                image_rgb_fixed_lr = outputs["image_rgb_fixed_lr"]
                pseudo_gt          = self.saved_pseudo_gt
                recon_loss         = self.mse(enhanced, pseudo_gt)
                tv_loss            = self.loss(illu_rgb_lr, image_rgb_lr, image_rgb_fixed_lr)
                loss               = recon_loss + tv_loss  # * self.tv_weight
                outputs["loss"]    = loss
            else:  # Skip updating model's weight at the first batch
                outputs = {"loss": None}
            # Saving n-th input and n-th pseudo gt
            self.saved_input     = nth_input
            self.saved_pseudo_gt = nth_pseudo_gt
        else:
            outputs            = self.forward(datapoint=datapoint, *args, **kwargs)
            enhanced           = outputs["enhanced"]
            illu_rgb_lr        = outputs["illu_rgb_lr"]
            image_rgb_lr       = outputs["image_rgb_lr"]
            image_rgb_fixed_lr = outputs["image_rgb_fixed_lr"]
            outputs["loss"]    = self.loss(illu_rgb_lr, image_rgb_lr, image_rgb_fixed_lr)
        # Return
        return outputs
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        # Prepare input
        self.assert_datapoint(datapoint)
        image_rgb = datapoint.get("image")
        depth     = datapoint.get("depth")
        if depth is None:
            depth = core.rgb_to_grayscale(image_rgb)
        edge      = self.dba(depth)
        # Enhance
        image_r          = image_rgb[:, 0:1, :, :]
        image_g          = image_rgb[:, 1:2, :, :]
        image_b          = image_rgb[:, 2:3, :, :]
        image_rgb_lr     = self.interpolate_image(image_rgb)
        depth_lr         = self.interpolate_image(depth)
        edge_lr          = self.interpolate_image(edge)
        image_r_lr       = image_rgb_lr[:, 0:1, :, :]
        image_g_lr       = image_rgb_lr[:, 1:2, :, :]
        image_b_lr       = image_rgb_lr[:, 2:3, :, :]
        patch_r          = self.patch_r_net(self.get_patches(image_r_lr))
        patch_g          = self.patch_g_net(self.get_patches(image_g_lr))
        patch_b          = self.patch_b_net(self.get_patches(image_b_lr))
        patch_d          = self.patch_d_net(self.get_patches(depth_lr))
        patch_e          = self.patch_e_net(self.get_patches(edge_lr))
        spatial          = self.spatial_net(self.get_coords())
        illu_res_r_lr    = self.output_net(torch.cat([patch_r, patch_e, patch_d, spatial], -1))
        illu_res_g_lr    = self.output_net(torch.cat([patch_g, patch_e, patch_d, spatial], -1))
        illu_res_b_lr    = self.output_net(torch.cat([patch_b, patch_e, patch_d, spatial], -1))
        illu_res_r_lr    = illu_res_r_lr.view(1, 1, self.down_size, self.down_size)
        illu_res_g_lr    = illu_res_g_lr.view(1, 1, self.down_size, self.down_size)
        illu_res_b_lr    = illu_res_b_lr.view(1, 1, self.down_size, self.down_size)
        illu_res_rgb_lr  = torch.cat([illu_res_r_lr, illu_res_g_lr, illu_res_b_lr], 1)
        illu_r_lr        = illu_res_r_lr + image_r_lr
        illu_g_lr        = illu_res_g_lr + image_g_lr
        illu_b_lr        = illu_res_b_lr + image_b_lr
        illu_rgb_lr      = torch.cat([illu_r_lr, illu_g_lr, illu_b_lr], 1)
        image_r_fixed_lr = image_r_lr / (illu_r_lr + 1e-8)
        image_g_fixed_lr = image_g_lr / (illu_g_lr + 1e-8)
        image_b_fixed_lr = image_b_lr / (illu_b_lr + 1e-8)
        # image_r_fixed_lr = torch.clamp(illu_r_fixed_lr, 1e-4, 1)
        # image_g_fixed_lr = torch.clamp(illu_g_fixed_lr, 1e-4, 1)
        # image_b_fixed_lr = torch.clamp(illu_b_fixed_lr, 1e-4, 1)
        if self.use_denoise:
            image_r_fixed_lr = kornia.filters.bilateral_blur(image_r_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
            image_g_fixed_lr = kornia.filters.bilateral_blur(image_g_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
            image_b_fixed_lr = kornia.filters.bilateral_blur(image_b_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
        image_rgb_fixed_lr = torch.cat([image_r_fixed_lr, image_g_fixed_lr, image_b_fixed_lr], 1)
        image_r_fixed    = self.filter_up(image_r_lr, image_r_fixed_lr, image_r)
        image_g_fixed    = self.filter_up(image_g_lr, image_g_fixed_lr, image_g)
        image_b_fixed    = self.filter_up(image_b_lr, image_b_fixed_lr, image_b)
        enhanced         = torch.cat([image_r_fixed, image_g_fixed, image_b_fixed], 1)
        enhanced         = enhanced / torch.max(enhanced)
        # Return
        if self.debug:
            return {
                "image"             : image_rgb,
                "depth"             : depth,
                "edge"              : edge,
                "pseudo_gt"         : self.saved_pseudo_gt,
                "illu_rgb_lr"       : illu_rgb_lr,
                "image_rgb_lr"      : image_rgb_lr,
                "image_rgb_fixed_lr": image_rgb_fixed_lr,
                "enhanced"          : enhanced,
            }
        else:
            return {
                "pseudo_gt": self.saved_pseudo_gt,
                "enhanced" : enhanced,
            }


@MODELS.register(name="zero_mlie_hvi", arch="zero_mlie")
class ZeroMLIE_HVI(ZeroMLIE):
    
    def __init__(
        self,
        name        : str         = "zero_mlie_hvi",
        window_size : int         = 1,
        down_size   : int         = 256,
        num_layers  : int         = 4,
        hidden_dim  : int         = 256,
        add_layer   : int         = 2,
        weight_decay: list[float] = [0.1, 0.0001, 0.001],
        use_depth   : bool        = False,
        use_edge    : bool        = False,
        use_denoise : bool        = False,
        use_pse     : bool        = False,
        number_refs : int         = 1,
        tv_weight   : float       = 5,
        L           : float       = 0.3,
        alpha       : float       = 1,
        beta        : float       = 20,
        gamma       : float       = 8,
        delta       : float       = 5,
        weights     : Any         = None,
        *args, **kwargs
    ):
        super().__init__(name=name, weights=weights, *args, **kwargs)
        self.window_size = window_size
        self.patch_dim   = window_size ** 2
        self.down_size   = down_size
        self.omega_0     = 30.0
        self.siren_C     = 6.0
        self.use_depth   = use_depth
        self.use_edge    = use_edge
        self.use_denoise = use_denoise
        self.use_pse     = use_pse
        self.number_refs = number_refs
        self.tv_weight   = tv_weight
        
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
        
        self.dba   = nn.BoundaryAwarePrior(eps=0.05, normalized=False)
        self.trans = core.RGBToHVI()
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
        self.loss = Loss(L, alpha, beta, gamma, delta)
        # self.loss = TVLoss(reduction="mean")
        self.mse  = nn.MSELoss()
        
        weight_decay = weight_decay or [0.1, 0.0001, 0.001]
        self.params  = []
        self.params += [{"params": self.spatial_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patch_i_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.patch_d_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.patch_e_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),  "weight_decay": weight_decay[2]}]
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        self.initial_state_dict = self.state_dict()
    
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
                outputs          = self.forward(datapoint=self.saved_input, *args, **kwargs)
                enhanced         = outputs["enhanced"]
                illu_i_lr        = outputs["illu_i_lr"]
                image_i_lr       = outputs["image_i_lr"]
                image_i_fixed_lr = outputs["image_i_fixed_lr"]
                pseudo_gt        = self.saved_pseudo_gt
                recon_loss       = self.mse(enhanced, pseudo_gt)
                tv_loss          = self.loss(illu_i_lr, image_i_lr, image_i_fixed_lr)
                loss             = recon_loss + tv_loss  # * self.tv_weight
                outputs["loss"]  = loss
            else:  # Skip updating model's weight at the first batch
                outputs = {"loss": None}
            # Saving n-th input and n-th pseudo gt
            self.saved_input     = nth_input
            self.saved_pseudo_gt = nth_pseudo_gt
        else:
            outputs          = self.forward(datapoint=datapoint, *args, **kwargs)
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
        image_i_fixed_lr = image_i_lr / (illu_i_lr + 1e-8)
        # image_i_fixed_lr = torch.clamp(image_i_fixed_lr, 1e-8, 1)
        if self.use_denoise:
            image_i_fixed_lr = kornia.filters.bilateral_blur(image_i_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
        image_i_fixed    = self.filter_up(image_i_lr, image_i_fixed_lr, image_i)
        image_hvi_fixed  = self.replace_i_component(image_hvi, image_i_fixed)
        enhanced         = self.trans.hvi_to_rgb(image_hvi_fixed)
        enhanced         = enhanced.clone().detach()
        enhanced         = enhanced / torch.max(enhanced)
        # Return
        if self.debug:
            return {
                "image"           : image_rgb,
                "depth"           : depth,
                "edge"            : edge,
                "pseudo_gt"       : self.saved_pseudo_gt,
                "image_h"         : image_h,
                "image_v"         : image_v,
                "image_i"         : image_i,
                "illu_i_lr"       : illu_i_lr,
                "image_i_lr"      : image_i_lr,
                "image_i_fixed_lr": image_i_fixed_lr,
                "enhanced"        : enhanced,
            }
        else:
            return {
                "pseudo_gt": self.saved_pseudo_gt,
                "enhanced" : enhanced,
            }
        
# endregion


# region Ablation

@MODELS.register(name="zero_mlie_01_rgb", arch="zero_mlie")
class ZeroMLIE_01_RGB(ZeroMLIE):
    
    def __init__(
        self,
        name        : str         = "zero_mlie_01_rgb",
        window_size : int         = 1,
        down_size   : int         = 256,
        num_layers  : int         = 4,
        hidden_dim  : int         = 256,
        add_layer   : int         = 2,
        weight_decay: list[float] = [0.1, 0.0001, 0.001],
        L           : float       = 0.3,
        alpha       : float       = 1,
        beta        : float       = 20,
        gamma       : float       = 8,
        delta       : float       = 5,
        weights     : Any         = None,
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
        illu_r_fixed_lr = kornia.filters.bilateral_blur(illu_r_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
        illu_g_fixed_lr = kornia.filters.bilateral_blur(illu_g_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
        illu_b_fixed_lr = kornia.filters.bilateral_blur(illu_b_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
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


@MODELS.register(name="zero_mlie_02_rgbd", arch="zero_mlie")
class ZeroMLIE_02_RGBD(ZeroMLIE):
    
    def __init__(
        self,
        name        : str         = "zero_mlie_02_rgbd",
        window_size : int         = 1,
        down_size   : int         = 256,
        num_layers  : int         = 4,
        hidden_dim  : int         = 256,
        add_layer   : int         = 2,
        weight_decay: list[float] = [0.1, 0.0001, 0.001],
        L           : float       = 0.3,
        alpha       : float       = 1,
        beta        : float       = 20,
        gamma       : float       = 8,
        delta       : float       = 5,
        weights     : Any         = None,
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
        illu_r_fixed_lr = kornia.filters.bilateral_blur(illu_r_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
        illu_g_fixed_lr = kornia.filters.bilateral_blur(illu_g_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
        illu_b_fixed_lr = kornia.filters.bilateral_blur(illu_b_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
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


@MODELS.register(name="zero_mlie_03_rgbd_zsn2n", arch="zero_mlie")
class ZeroMLIE_03_RGBD_ZSN2N(ZeroMLIE_02_RGBD):
    
    def __init__(
        self,
        name   : str = "zero_mlie_03_rgbd_zsn2n",
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


@MODELS.register(name="zero_mlie_04_hsv", arch="zero_mlie")
class ZeroMLIE_04_HSV(ZeroMLIE):
    
    def __init__(
        self,
        name        : str         = "zero_mlie_04_hsv",
        window_size : int         = 1,
        down_size   : int         = 256,
        num_layers  : int         = 4,
        hidden_dim  : int         = 256,
        add_layer   : int         = 2,
        weight_decay: list[float] = [0.1, 0.0001, 0.001],
        L           : float       = 0.3,
        alpha       : float       = 1,
        beta        : float       = 20,
        gamma       : float       = 8,
        delta       : float       = 5,
        weights     : Any         = None,
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
        image_v_fixed_lr = kornia.filters.bilateral_blur(image_v_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
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


@MODELS.register(name="zero_mlie_05_hsvd", arch="zero_mlie")
class ZeroMLIE_05_HSVD(ZeroMLIE):
    
    def __init__(
        self,
        name        : str         = "zero_mlie_05_hsvd",
        window_size : int         = 1,
        down_size   : int         = 256,
        num_layers  : int         = 4,
        hidden_dim  : int         = 256,
        add_layer   : int         = 2,
        weight_decay: list[float] = [0.1, 0.0001, 0.001],
        L           : float       = 0.3,
        alpha       : float       = 1,
        beta        : float       = 20,
        gamma       : float       = 8,
        delta       : float       = 5,
        weights     : Any         = None,
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
        image_v_fixed_lr = kornia.filters.bilateral_blur(image_v_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
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


@MODELS.register(name="zero_mlie_06_hsvd_zsn2n", arch="zero_mlie")
class ZeroMLIE_06_HSVD_ZSN2N(ZeroMLIE_05_HSVD):
    
    def __init__(
        self,
        name   : str = "zero_mlie_05_hsvd_zsn2n",
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


@MODELS.register(name="zero_mlie_07_hvi", arch="zero_mlie")
class ZeroMLIE_07_HVI(ZeroMLIE):
    
    def __init__(
        self,
        name        : str         = "zero_mlie_07_hvi",
        window_size : int         = 1,
        down_size   : int         = 256,
        num_layers  : int         = 4,
        hidden_dim  : int         = 256,
        add_layer   : int         = 2,
        weight_decay: list[float] = [0.1, 0.0001, 0.001],
        L           : float       = 0.3,
        alpha       : float       = 1,
        beta        : float       = 20,
        gamma       : float       = 8,
        delta       : float       = 5,
        weights     : Any         = None,
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
        image_i_fixed_lr = kornia.filters.bilateral_blur(image_i_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
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


@MODELS.register(name="zero_mlie_08_hvid", arch="zero_mlie")
class ZeroMLIE_08_HVID(ZeroMLIE):
    
    def __init__(
        self,
        name        : str         = "zero_mlie_08_hvid",
        window_size : int         = 1,
        down_size   : int         = 256,
        num_layers  : int         = 4,
        hidden_dim  : int         = 256,
        add_layer   : int         = 2,
        weight_decay: list[float] = [0.1, 0.0001, 0.001],
        L           : float       = 0.3,
        alpha       : float       = 1,
        beta        : float       = 20,
        gamma       : float       = 8,
        delta       : float       = 5,
        weights     : Any         = None,
        *args, **kwargs
    ):
        super().__init__(name=name, weights=weights, *args, **kwargs)
        self.window_size = window_size
        self.patch_dim   = window_size ** 2
        self.down_size   = down_size
        self.omega_0     = 30.0
        self.siren_C     = 6.0
        
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
        image_i_fixed_lr = kornia.filters.bilateral_blur(image_i_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
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


@MODELS.register(name="zero_mlie_09_hvid_noblur", arch="zero_mlie")
class ZeroMLIE_09_HVID_NoBlur(ZeroMLIE):
    
    def __init__(
        self,
        name        : str         = "zero_mlie_09_hvid_noblur",
        window_size : int         = 1,
        down_size   : int         = 256,
        num_layers  : int         = 4,
        hidden_dim  : int         = 256,
        add_layer   : int         = 2,
        weight_decay: list[float] = [0.1, 0.0001, 0.001],
        L           : float       = 0.3,
        alpha       : float       = 1,
        beta        : float       = 20,
        gamma       : float       = 8,
        delta       : float       = 5,
        weights     : Any         = None,
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
        # image_i_fixed_lr = kornia.filters.bilateral_blur(image_i_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
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


@MODELS.register(name="zero_mlie_10_hvid_zsn2n", arch="zero_mlie")
class ZeroMLIE_10_HVID_ZSN2N(ZeroMLIE_08_HVID):
    
    def __init__(
        self,
        name   : str = "zero_mlie_10_hvid_zsn2n",
        weights: Any = None,
        *args, **kwargs
    ):
        super().__init__(name=name, weights=weights,*args, **kwargs)
        
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


@MODELS.register(name="zero_mlie_11_rgbd_hsvd", arch="zero_mlie")
class ZeroMLIE_11_RGBD_HSVD(ZeroMLIE):
    
    def __init__(
        self,
        name        : str         = "zero_mlie_11_rgbd_hsvd",
        window_size : int         = 1,
        down_size   : int         = 256,
        num_layers  : int         = 4,
        hidden_dim  : int         = 256,
        add_layer   : int         = 2,
        weight_decay: list[float] = [0.1, 0.0001, 0.001],
        L           : float       = 0.3,
        alpha       : float       = 1,
        beta        : float       = 20,
        gamma       : float       = 8,
        delta       : float       = 5,
        weights     : Any         = None,
        *args, **kwargs
    ):
        super().__init__(name=name, weights=weights, *args, **kwargs)
        self.window_size = window_size
        self.patch_dim   = window_size ** 2
        self.down_size   = down_size
        self.omega_0     = 30.0
        self.siren_C     = 6.0
        
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
        image_r_fixed_lr   = kornia.filters.bilateral_blur(image_r_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
        image_g_fixed_lr   = kornia.filters.bilateral_blur(image_g_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
        image_b_fixed_lr   = kornia.filters.bilateral_blur(image_b_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
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
        # image_v_fixed_lr   = kornia.filters.bilateral_blur(image_v_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
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


@MODELS.register(name="zero_mlie_12_rgbd_hsvd_zsn2n", arch="zero_mlie")
class ZeroMLIE_12_RGBD_HSVD_ZSN2N(ZeroMLIE):
    
    def __init__(
        self,
        name        : str         = "zero_mlie_12_rgbd_hsvd_zsn2n",
        window_size : int         = 1,
        down_size   : int         = 256,
        num_layers  : int         = 4,
        hidden_dim  : int         = 256,
        add_layer   : int         = 2,
        weight_decay: list[float] = [0.1, 0.0001, 0.001],
        L           : float       = 0.3,
        alpha       : float       = 1,
        beta        : float       = 20,
        gamma       : float       = 8,
        delta       : float       = 5,
        weights     : Any         = None,
        *args, **kwargs
    ):
        super().__init__(name=name, weights=weights, *args, **kwargs)
        self.window_size = window_size
        self.patch_dim   = window_size ** 2
        self.down_size   = down_size
        self.omega_0     = 30.0
        self.siren_C     = 6.0
        
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
        image_r_fixed_lr   = kornia.filters.bilateral_blur(image_r_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
        image_g_fixed_lr   = kornia.filters.bilateral_blur(image_g_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
        image_b_fixed_lr   = kornia.filters.bilateral_blur(image_b_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
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


@MODELS.register(name="zero_mlie_13_hvid_denoise", arch="zero_mlie")
class ZeroMLIE_13_HSVD_Denoise(ZeroMLIE):
    
    def __init__(
        self,
        name        : str         = "zero_mlie_13_hvid_denoise",
        window_size : int         = 1,
        down_size   : int         = 256,
        num_layers  : int         = 4,
        hidden_dim  : int         = 256,
        add_layer   : int         = 2,
        weight_decay: list[float] = [0.1, 0.0001, 0.001],
        L           : float       = 0.1,
        alpha       : float       = 1,
        beta        : float       = 20,
        gamma       : float       = 8,
        delta       : float       = 5,
        weights     : Any         = None,
        *args, **kwargs
    ):
        super().__init__(name=name, weights=weights, *args, **kwargs)
        self.window_size = window_size
        self.patch_dim   = window_size ** 2
        self.down_size   = down_size
        self.omega_0     = 30.0
        self.siren_C     = 6.0
        
        patch_i_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_d_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        patch_e_layers = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        spatial_layers = [nn.SIREN(2, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        for _ in range(1, add_layer - 2):
            patch_i_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_d_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            patch_e_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
            spatial_layers.append(nn.SIREN(hidden_dim, hidden_dim,  self.omega_0, self.siren_C))
        patch_i_layers.append(nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
        patch_d_layers.append(nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
        patch_e_layers.append(nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
        spatial_layers.append(nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
        
        output_layers = []
        for _ in range(add_layer, num_layers - 1):
            output_layers.append(nn.SIREN(hidden_dim * 4, hidden_dim, self.omega_0, self.siren_C))
        output_layers.append(nn.SIREN(hidden_dim, 1, self.omega_0, self.siren_C, is_last=True))
        
        self.patch_i_net = nn.Sequential(*patch_i_layers)
        self.patch_d_net = nn.Sequential(*patch_d_layers)
        self.patch_e_net = nn.Sequential(*patch_e_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net  = nn.Sequential(*output_layers)
        self.dba         = nn.BoundaryAwarePrior(eps=0.05, normalized=False)
        self.trans       = core.RGBToHVI()
        
        self.saved_input     = None
        self.saved_pseudo_gt = None
        
        # Loss
        self.loss = Loss(L, alpha, beta, gamma, delta)
        self.mse  = nn.MSELoss()
        # self.loss = TVLoss(reduction="mean")
        
        self.pseudo_gt_generator = utils.PseudoGTGenerator(
            number_refs   = 1,
            gamma_upper   = -2,
            gamma_lower   = 3,
            exposed_level = 0.5,
            pool_size     = 25,
        )
        
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
        image = datapoint.get("image")
        '''
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        self.assert_outputs(outputs)
        # Loss
        illu_i_lr        = outputs["illu_i_lr"]
        image_i_lr       = outputs["image_i_lr"]
        image_i_fixed_lr = outputs["image_i_fixed_lr"]
        outputs["loss"]  = self.loss(illu_i_lr, image_i_lr, image_i_fixed_lr)
        # Return
        return outputs
        '''
        # Saving n-th input and n-th pseudo gt
        nth_input     = datapoint
        nth_output    = self.forward(datapoint=datapoint, *args, **kwargs)
        nth_enhanced  = nth_output["enhanced"].clone().detach()
        nth_pseudo_gt = self.pseudo_gt_generator(image, nth_enhanced)
        if self.saved_input is not None:
            # Getting (n - 1)th input and (n - 1)-th pseudo gt -> calculate loss -> update model weight (handled automatically by pytorch lightning)
            x          = self.saved_input
            outputs    = self.forward(datapoint=x, *args, **kwargs)
            # y, r       = self.model(x)
            illu_i_lr        = outputs["illu_i_lr"]
            image_i_lr       = outputs["image_i_lr"]
            image_i_fixed_lr = outputs["image_i_fixed_lr"]
            y                = outputs["enhanced"]
            pseudo_gt  = self.saved_pseudo_gt
            recon_loss = self.mse(y, pseudo_gt)
            tv_loss    = self.loss(illu_i_lr, image_i_lr, image_i_fixed_lr)
            loss       = recon_loss + tv_loss * 5
            outputs["loss"] = loss
        else:  # Skip updating model's weight at the first batch
            outputs = {"loss": None}
        # Saving n-th input and n-th pseudo gt
        self.saved_input     = nth_input
        self.saved_pseudo_gt = nth_pseudo_gt
        # Return
        return outputs
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        # Prepare inputs
        image_rgb = datapoint.get("image")
        depth     = datapoint.get("depth")
        if depth is None:
            depth = core.rgb_to_grayscale(image_rgb)
        edge = self.dba(depth)
        # _, edge = kornia.filters.Canny(low_threshold=0.5, high_threshold=0.6)(depth)
        # image_rgb = kornia.filters.bilateral_blur(image_rgb, bilateral_ksize, bilateral_color, bilateral_space)
        
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
        illu_res_i_lr    = self.output_net(torch.cat([patch_i, patch_d, patch_e, spatial], -1))
        illu_res_i_lr    = illu_res_i_lr.view(1, 1, self.down_size, self.down_size)
        illu_i_lr        = illu_res_i_lr + image_i_lr
        image_i_fixed_lr = image_i_lr / (illu_i_lr + 1e-4)
        # image_i_fixed_lr = torch.clamp(image_i_fixed_lr, 1e-4, 1)
        image_i_fixed_lr = kornia.filters.bilateral_blur(image_i_fixed_lr, bilateral_ksize, bilateral_color, bilateral_space)
        image_i_fixed_lr_sharpen = kornia.filters.UnsharpMask((3, 3), (1.5, 1.5))(image_i_fixed_lr)
        difference       = (image_i_fixed_lr_sharpen - image_i_fixed_lr).abs()
        image_i_fixed    = self.filter_up(image_i_lr, image_i_fixed_lr_sharpen, image_i)
        image_hvi_fixed  = self.replace_i_component(image_hvi, image_i_fixed)
        image_rgb_fixed  = self.trans.hvi_to_rgb(image_hvi_fixed)
        image_rgb_fixed  = image_rgb_fixed.clone().detach()
        # image_rgb_fixed  = image_rgb_fixed / torch.max(image_rgb_fixed)
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
                "image_i_fixed_lr": image_i_fixed_lr_sharpen,
                "difference"      : difference,
                "pseudo_gt"       : self.saved_pseudo_gt,
                "enhanced"        : image_rgb_fixed,
            }
        else:
            return {
                "enhanced": image_rgb_fixed,
            }
    
# endregion
