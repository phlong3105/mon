#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements CoLIE-Mamba model for low-light image enhancement.

References:
    - Paper:
    - Code: https://github.com/Lo9ite/colie_mamba
"""

__all__ = [
    "CoLIEMamba",
]

import box
import kornia
import torch

from mon import nn
from mon.core import MLType, MODELS, Path, Task
from mon.training import optims
from . import loss as L
from .siren_mamba import *
from .utils import *

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="colie_mamba", arch="colie_mamba")
class CoLIEMamba(nn.Module, nn.ModelMixin):
    """CoLIE-Mamba model for low-light image enhancement.

    References:
        - Paper:
        - Code: https://github.com/Lo9ite/colie_mamba
    """
    
    _arch     : str          = "colie_mamba"
    _name     : str          = "colie_mamba"
    _tasks    : list[Task]   = [Task.LLE]
    _mltypes  : list[MLType] = [MLType.ZERO_SHOT]
    _model_dir: Path         = root_dir
    _zoo      : dict         = box.Box()
    
    def __init__(
        self,
        hidden_dim   : int   = 256,
        mamba_d_model: int   = 64,
        mamba_d_state: int   = 16,
        mamba_d_conv : int   = 4,
        mamba_expand : int   = 2,
        L            : float = 0.5,
        iters        : int   = 100,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.L          = L
        self.iters      = iters
        
        self.model = CoLIEMambaNet(
            hidden_dim    = hidden_dim,
            mamba_d_model = mamba_d_model,
            mamba_d_state = mamba_d_state,
            mamba_d_conv  = mamba_d_conv,
            mamba_expand  = mamba_expand,
        )
        self.state_dict = self.model.state_dict()
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        imgsz  = self.hidden_dim
        device = image.device
        
        # Preprocess
        image_hsv  = kornia.color.rgb_to_hsv(image).to(device)
        image_v    = get_v_component(image_hsv).to(device)
        image_v_lr = interpolate_image(image_v, imgsz, imgsz).to(device)
        coords     = get_coords(imgsz, imgsz).to(device)
        
        # Optimize
        self.model.load_state_dict(self.state_dict)
        self.model.train()
        optimizer = optims.Adam(self.model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=3e-4)
        scheduler = optims.CosineAnnealingLR(optimizer, T_max=self.iters, eta_min=1e-6)
        L_exp     = L.L_exp(16, self.L).to(device)
        L_tv      = L.L_tv().to(device)
        L_tex     = L.L_texture().to(device)
        
        image_v_fixed_lr = None
        for i in range(self.iters):
            optimizer.zero_grad()
            illu_lr          = self.model(image_v_lr, coords)
            image_v_fixed_lr = image_v_lr / (illu_lr + 1e-4)  # This is the reflectance R = V / I
            # Loss
            l_spa      = torch.mean(torch.abs(torch.pow(illu_lr - image_v_lr, 2)))
            l_tv       = L_tv(illu_lr)
            l_exp      = torch.mean(L_exp(illu_lr))
            l_sparsity = torch.mean(image_v_fixed_lr)
            l_denoise  = L_tex(image_v_fixed_lr)
            loss       = 85 * l_spa + 25 * l_tv + 43 * l_exp + 18 * l_sparsity + 18 * l_denoise
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # Final Image Reconstruction
        self.model.eval()
        with torch.no_grad():
            final_illu_lr    = self.model(image_v_lr, coords)
            final_v_fixed_lr = image_v_lr / (final_illu_lr + 1e-4)
        
        # Postprocess
        image_v_fixed   = filter_up(image_v_lr, image_v_fixed_lr, image_v)
        image_hsv_fixed = replace_v_component(image_hsv, image_v_fixed)
        image_rgb_fixed = kornia.color.hsv_to_rgb(image_hsv_fixed)
        image_rgb_fixed = image_rgb_fixed / torch.max(image_rgb_fixed)
        # enhanced        = torch.movedim(image_rgb_fixed, 1, -1)[0].detach().cpu()
        enhanced        = image_rgb_fixed.detach().cpu()
        
        return enhanced
