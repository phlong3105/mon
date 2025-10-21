#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ZINF model for low-light image enhancement.

References:
    - Paper: "Zero-Shot Implicit Neural Fusion Network for Multimodal Low-Light
      Image Enhancement," arXiv 2025.
    - Code: https://github.com/phlong3105/mon
"""

__all__ = [
    "ZINF",
]

import box
import torch

from mon.constants import MODELS
from mon.core import image as I, MLType, ModelMixin, nn, Path, Task
from mon.core.nn.modules.inr.utils import *
from .inr import DAM_iSIREN, DAM_SIREN, iSIREN, peSIREN, SIREN

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]
INRS         = {
    "siren"     : SIREN,
    "pesiren"   : peSIREN,
    "isiren"    : iSIREN,
    "dam_siren" : DAM_SIREN,
    "dam_isiren": DAM_iSIREN,
}


@MODELS.register(name="zinf", arch="zinf")
class ZINF(nn.Module, ModelMixin):
    """ZINF model for low-light image enhancement."""
    
    arch     : str          = "zinf"
    name     : str          = "zinf"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.ZERO_SHOT]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
    
    def __init__(
        self,
        window_size: int   = 7,
        hidden_dim : int   = 256,
        num_layers : int   = 4,
        add_layers : int   = 2,
        inr        : str   = "siren",
        training   : str   = "default",   # "default", "lbfgs", "zsn2n"
        L          : float = 0.5,
        iters      : int   = 100,
    ):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim  = hidden_dim
        self.inr         = inr
        self.training    = training
        self.L           = L
        self.iters       = iters
        
        self.hvi_t = I.RGBToHVI(requires_grad=False)
        self.model = INRS[inr](
            patch_dim  = self.window_size ** 2,
            hidden_dim = hidden_dim,
            num_layers = num_layers,
            add_layers = add_layers,
        )
        self.state_dict = self.model.state_dict()
    
    def forward(self, image: torch.Tensor, depth: torch.Tensor = None, save_debug: bool = False):
        # Convert to HVI
        image_hvi = self.hvi_t.rgb_to_hvi(image)
        image_hv  = image_hvi[:, 0:2, :, :]
        image_i   = image_hvi[:, 2:3, :, :]
        
        # Optimize
        if self.inr in ["isiren", "dam_isiren"]:
            self.optimize_indi(y_I=image_i, depth=depth)
            f_lr, y_I_lr, x_I_lr = self.infer_illu_indi(y_I=image_i, depth=depth)
        else:
            if self.training == "lbfgs":
                self.optimize_lbfgs(y_I=image_i, depth=depth)
            elif self.training == "asym":
                self.optimize_asym(y_I=image_i, depth=depth)
            elif self.training == "sym":
                self.optimize_sym(y_I=image_i, depth=depth)
            else:
                self.optimize(y_I=image_i, depth=depth)
            f_lr, y_I_lr, x_I_lr = self.infer_illu(y_I=image_i, depth=depth)
            
        # Inference
        z_I_lr = y_I_lr / (x_I_lr + 1e-6)
        z_I    = filter_up(y_I_lr, z_I_lr, image_i, kernel_size=self.window_size)
        
        # Convert to RGB
        image_hvi_fixed = torch.cat((image_hv, z_I), dim=1).to(image.device)
        image_rgb_fixed = self.hvi_t.hvi_to_rgb(image_hvi_fixed)
        
        if save_debug:
            return {
                "residual": filter_up(y_I_lr, f_lr, image_i),
                "enhanced": image_rgb_fixed,
            }
        else:
            return {
                "enhanced": image_rgb_fixed,
            }
    
    # ----- Optimize: Default -----
    def optimize(self, y_I: torch.Tensor, depth: torch.Tensor = None):
        imgsz  = self.hidden_dim
        device = y_I.device
        
        # Preprocess
        y_I_lr    = interpolate_image(y_I, imgsz)
        D_lr      = interpolate_image(depth, imgsz) if depth is not None else None
        coords    = create_noisy_coords(imgsz).to(device)
        patches_I = create_patches(y_I_lr, self.window_size)
        patches_D = create_patches(D_lr,   self.window_size)
        
        # Optimize
        self.model.load_state_dict(self.state_dict)
        self.model.train()
        optimizer = nn.Adam(self.model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=3e-4)
        L_exp     = nn.ExposureControlLoss(16, self.L, channel_mean=True).to(device)
        L_tv      = nn.TotalVariationLoss().to(device)
        for i in range(self.iters):
            optimizer.zero_grad()  # Zero the gradients
            f_lr   = self.model(coords=coords, patches=patches_I, depth=patches_D)
            f_lr   = f_lr.view(1, 1, imgsz, imgsz)
            x_I_lr = f_lr + y_I_lr
            z_I_lr = y_I_lr / (x_I_lr + 1e-6)
            # Loss
            l_spa  = torch.mean(torch.abs(torch.pow(x_I_lr - y_I_lr, 2)))  # Spatial loss
            l_tv   = L_tv(x_I_lr)               # TV loss
            l_exp  = torch.mean(L_exp(x_I_lr))  # Exposure loss
            l_spar = torch.mean(z_I_lr)         # Sparsity loss
            loss   = 1 * l_spa + 20 * l_tv + 8 * l_exp + 5 * l_spar
            loss.backward()  # Compute gradients
            optimizer.step()
    
    def infer_illu(self, y_I: torch.Tensor, depth: torch.Tensor = None) -> tuple[torch.Tensor, ...]:
        imgsz  = self.hidden_dim
        device = y_I.device
        
        y_I_lr    = interpolate_image(y_I, imgsz)
        D_lr      = interpolate_image(depth, imgsz) if depth is not None else None
        coords    = create_coords(imgsz).to(device)
        patches_I = create_patches(y_I_lr, self.window_size)
        patches_D = create_patches(D_lr,   self.window_size)
        
        f_lr   = self.model(coords=coords, patches=patches_I, depth=patches_D)
        f_lr   = f_lr.view(1, 1, imgsz, imgsz)
        x_I_lr = f_lr + y_I_lr
        return f_lr, y_I_lr, x_I_lr
    
    # ----- Optimize: LBFGS -----
    def optimize_lbfgs(self, y_I: torch.Tensor, depth: torch.Tensor = None):
        imgsz  = self.hidden_dim
        device = y_I.device
        
        # Preprocess
        y_I_lr    = interpolate_image(y_I, imgsz)
        D_lr      = interpolate_image(depth, imgsz) if depth is not None else None  # Shape: (1, 1, 256, 256)
        coords    = create_noisy_coords(imgsz).to(device)     # Shape: (256, 256,  2)
        patches_I = create_patches(y_I_lr, self.window_size)  # Shape: (256, 256, 49)
        patches_D = create_patches(D_lr,   self.window_size)
        
        # Optimize
        self.model.load_state_dict(self.state_dict)
        self.model.train()
        optimizer = nn.LBFGS(self.model.parameters(), lr=1, max_iter=4, history_size=10, line_search_fn="strong_wolfe")
        L_exp     = nn.ExposureControlLoss(16, self.L, channel_mean=True).to(device)
        L_tv      = nn.TotalVariationLoss().to(device)
        for i in range(self.iters):
            
            def closure():
                optimizer.zero_grad()  # Zero the gradients
                f_lr   = self.model(coords=coords, patches=patches_I, depth=patches_D)
                f_lr   = f_lr.view(1, 1, imgsz, imgsz)
                x_I_lr = f_lr + y_I_lr
                z_I_lr = y_I_lr / (x_I_lr + 1e-6)
                #
                l_spa  = torch.mean(torch.abs(torch.pow(x_I_lr - y_I_lr, 2)))  # Spatial loss
                l_tv   = L_tv(x_I_lr)               # TV loss
                l_exp  = torch.mean(L_exp(x_I_lr))  # Exposure loss
                l_spar = torch.mean(z_I_lr)         # Sparsity loss
                loss   = 1 * l_spa + 20 * l_tv + 8 * l_exp + 5 * l_spar
                loss.backward()  # Compute gradients
                return loss
            
            optimizer.step(closure)
    
    # ----- Optimize: ZSN2N -----
    def optimize_asym(self, y_I: torch.Tensor, depth: torch.Tensor = None):
        imgsz  = self.hidden_dim
        device = y_I.device
        
        # Preprocess
        y_I_lr_noisy1, y_I_lr_noisy2 = pair_downsampler(interpolate_image(y_I, imgsz * 2))
        y_I_lr    = interpolate_image(y_I, imgsz)
        D_lr      = interpolate_image(depth, imgsz) if depth is not None else None
        coords    = create_noisy_coords(imgsz).to(device)
        patches_I = create_patches(y_I_lr, self.window_size)
        patches_D = create_patches(D_lr,   self.window_size)
        
        # Optimize
        self.model.load_state_dict(self.state_dict)
        self.model.train()
        optimizer = nn.LBFGS(self.model.parameters(), lr=1, max_iter=4, history_size=10, line_search_fn="strong_wolfe")
        L_exp     = nn.ExposureControlLoss(16, self.L, channel_mean=True).to(device)
        L_tv      = nn.TotalVariationLoss().to(device)
        for i in range(self.iters):
            
            def closure():
                optimizer.zero_grad()  # Zero the gradients
                f_lr   = self.model(coords=coords, patches=patches_I, depth=patches_D)
                f_lr   = f_lr.view(1, 1, imgsz, imgsz)
                x_I_lr = f_lr + y_I_lr_noisy1
                z_I_lr = y_I_lr_noisy1 / (x_I_lr + 1e-6)
                #
                l_spa  = torch.mean(torch.abs(torch.pow(x_I_lr - y_I_lr_noisy2, 2)))  # Spatial loss
                l_tv   = L_tv(x_I_lr)               # TV loss
                l_exp  = torch.mean(L_exp(x_I_lr))  # Exposure loss
                l_spar = torch.mean(z_I_lr)         # Sparsity loss
                loss   = 1 * l_spa + 20 * l_tv + 8 * l_exp + 5 * l_spar
                loss.backward()  # Compute gradients
                return loss
            
            optimizer.step(closure)
    
    def optimize_sym(self, y_I: torch.Tensor, depth: torch.Tensor = None):
        imgsz  = self.hidden_dim
        device = y_I.device
        
        # Preprocess
        y_I_lr_noisy1, y_I_lr_noisy2 = pair_downsampler(interpolate_image(y_I, imgsz * 2))
        y_I_lr    = interpolate_image(y_I, imgsz)
        D_lr      = interpolate_image(depth, imgsz) if depth is not None else None
        coords    = create_noisy_coords(imgsz).to(device)
        patches_I = create_patches(y_I_lr, self.window_size)
        patches_D = create_patches(D_lr,   self.window_size)
        
        # Optimize
        self.model.load_state_dict(self.state_dict)
        self.model.train()
        optimizer = nn.LBFGS(self.model.parameters(), lr=1, max_iter=4, history_size=10, line_search_fn="strong_wolfe")
        L_exp     = nn.ExposureControlLoss(16, self.L, channel_mean=True).to(device)
        L_tv      = nn.TotalVariationLoss().to(device)
        for i in range(self.iters):
            
            def closure():
                optimizer.zero_grad()  # Zero the gradients
                # Direction 1: y_I_lr_noisy1 → y_I_lr_noisy2
                f_lr1   = self.model(coords=coords, patches=patches_I, depth=patches_D)
                f_lr1   = f_lr1.view(1, 1, imgsz, imgsz)
                x_I_lr1 = f_lr1 + y_I_lr_noisy1
                z_I_lr1 = y_I_lr_noisy1 / (x_I_lr1 + 1e-6)
                #
                l_spa1  = torch.mean(torch.abs(torch.pow(x_I_lr1 - y_I_lr_noisy2, 2)))  # Spatial loss
                l_tv1   = L_tv(x_I_lr1)               # TV loss
                l_exp1  = torch.mean(L_exp(x_I_lr1))  # Exposure loss
                l_spar1 = torch.mean(z_I_lr1)         # Sparsity loss
                # Direction 1: y_I_lr_noisy2 → y_I_lr_noisy1
                f_lr2   = self.model(coords=coords, patches=patches_I)
                f_lr2   = f_lr2.view(1, 1, imgsz, imgsz)
                x_I_lr2 = f_lr2 + y_I_lr_noisy2
                z_I_lr2 = y_I_lr_noisy2 / (x_I_lr2 + 1e-6)
                #
                l_spa2  = torch.mean(torch.abs(torch.pow(x_I_lr2 - y_I_lr_noisy1, 2)))  # Spatial loss
                l_tv2   = L_tv(x_I_lr2)               # TV loss
                l_exp2  = torch.mean(L_exp(x_I_lr2))  # Exposure loss
                l_spar2 = torch.mean(z_I_lr2)         # Sparsity loss
                # Average losses
                l_spa   = (l_spa1  + l_spa2)  / 2
                l_tv    = (l_tv1   + l_tv2)   / 2
                l_exp   = (l_exp1  + l_exp2)  / 2
                l_spar  = (l_spar1 + l_spar2) / 2
                loss    = 1 * l_spa + 20 * l_tv + 8 * l_exp + 5 * l_spar
                loss.backward()  # Compute gradients
                return loss
            
            optimizer.step(closure)
    
    # ----- Optimize: I-INR -----
    def optimize_indi(self, y_I: torch.Tensor, depth: torch.Tensor = None):
        imgsz  = self.hidden_dim
        device = y_I.device
        
        # Preprocess
        y_I_lr_noisy1, y_I_lr_noisy2 = pair_downsampler(interpolate_image(y_I, imgsz * 2))
        y_I_lr    = interpolate_image(y_I, imgsz)
        D_lr      = interpolate_image(depth, imgsz) if depth is not None else None
        Z         = torch.randn_like(y_I_lr)
        coords    = create_noisy_coords(imgsz).to(device)
        patches_I = create_patches(y_I_lr, self.window_size)
        patches_D = create_patches(D_lr,   self.window_size)
        
        # Optimize
        self.model.load_state_dict(self.state_dict)
        self.model.train()
        # optimizer = nn.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=3e-4)
        optimizer = nn.LBFGS(self.model.parameters(), lr=1, max_iter=4, history_size=10, line_search_fn="strong_wolfe")
        L_exp     = nn.ExposureControlLoss(16, self.L, channel_mean=True).to(device)
        L_tv      = nn.TotalVariationLoss().to(device)
        for i in range(self.iters):
            
            def closure():
                optimizer.zero_grad()
                t      = torch.rand(1).item()      # Random t in [0, 1]
                n      = torch.randn_like(y_I_lr)  # Noise
                e      = 0.1                       # Small constant
                g_t    = y_I_lr * (1 - t) + Z * t + e * n * t
                prev_g = g_t.view(imgsz, imgsz, 1)
                #
                f_lr   = self.model(coords=coords, patches=patches_I, depth=patches_D, prev_g=prev_g, t=t)
                f_lr   = f_lr.view(1, 1, imgsz, imgsz)
                x_I_lr = f_lr + y_I_lr_noisy1
                z_I_lr = y_I_lr_noisy1 / (x_I_lr + 1e-6)
                #
                l_spa  = torch.mean(torch.abs(torch.pow(x_I_lr - y_I_lr_noisy2, 2)))  # Spatial loss
                l_tv   = L_tv(x_I_lr)               # TV loss
                l_exp  = torch.mean(L_exp(x_I_lr))  # Exposure loss
                l_spar = torch.mean(z_I_lr)         # Sparsity loss
                loss   = 1 * l_spa + 20 * l_tv + 8 * l_exp + 5 * l_spar
                loss.backward()
                return loss
            
            optimizer.step(closure)
    
    def infer_illu_indi(self, y_I: torch.Tensor, depth: torch.Tensor = None) -> tuple[torch.Tensor, ...]:
        imgsz  = self.hidden_dim
        device = y_I.device
        
        y_I_lr    = interpolate_image(y_I, imgsz)
        Z         = torch.randn_like(y_I_lr)
        D_lr      = interpolate_image(depth, imgsz) if depth is not None else None
        g_hat     = Z
        coords    = create_coords(imgsz).to(device)
        patches_I = create_patches(y_I_lr, self.window_size)
        patches_D = create_patches(D_lr,   self.window_size)
        
        delta = 0.05
        t     = delta
        while t <= 1.0:
            prev_g  = g_hat.view(imgsz, imgsz, 1)
            f_lr    = self.model(coords=coords, patches=patches_I, depth=patches_D, prev_g=prev_g, t=t)
            f_lr    = f_lr.view(1, 1, imgsz, imgsz)
            g_hat   = (delta / t) * f_lr + (1 - delta / t) * g_hat
            t      += delta
            
        f_lr   = g_hat
        x_I_lr = f_lr + y_I_lr
        return f_lr, y_I_lr, x_I_lr
