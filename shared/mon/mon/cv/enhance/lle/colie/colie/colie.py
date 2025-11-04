#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements CoLIE model for low-light image enhancement.

References:
    - Paper: "Fast Context-Based Low-Light Image Enhancement via Neural Implicit
      Representations," ECCV 2024.
    - Code: https://github.com/ctom2/colie
"""

__all__ = [
    "CoLIE",
]

import box
import kornia
import torch

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, nn, Path, Task
from . import loss as L
from .siren import *
from .utils import *

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="colie", arch="colie")
class CoLIE(nn.Module, ModelMixin):
    """CoLIE model for low-light image enhancement.

    References:
        - Paper: "Fast Context-Based Low-Light Image Enhancement via Neural Implicit
          Representations," ECCV 2024.
        - Code: https://github.com/ctom2/colie
    """
    
    arch     : str          = "colie"
    name     : str          = "colie"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.ZERO_SHOT]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
    
    def __init__(
        self,
        window_size: int   = 7,
        hidden_dim : int   = 256,
        num_layers : int   = 4,
        add_layer  : int   = 2,
        L          : float = 0.5,
        iters      : int   = 100,
    ):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim  = hidden_dim
        self.L           = L
        self.iters       = iters
        
        self.model = SIREN(
            patch_dim  = self.window_size ** 2,
            hidden_dim = hidden_dim,
            num_layers = num_layers,
            add_layer  = add_layer
        )
        self.state_dict = self.model.state_dict()
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        window_size = self.window_size
        imgsz       = self.hidden_dim
        device      = image.device
        
        # Preprocess
        image_hsv  = kornia.color.rgb_to_hsv(image).to(device)
        image_v    = get_v_component(image_hsv).to(device)
        image_v_lr = interpolate_image(image_v, imgsz, imgsz).to(device)
        patches    = get_patches(image_v_lr, window_size).to(device)
        coords     = get_coords(imgsz, imgsz).to(device)
        
        # Optimize
        self.model.load_state_dict(self.state_dict)
        self.model.train()
        optimizer = nn.Adam(self.model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=3e-4)
        L_exp     = L.L_exp(16, self.L).to(device)
        L_tv      = L.L_tv().to(device)
        
        image_v_fixed_lr = None
        for i in range(self.iters):
            optimizer.zero_grad()
            illu_res_lr      = self.model(patches, coords)
            illu_res_lr      = illu_res_lr.view(1, 1, imgsz, imgsz)
            illu_lr          = illu_res_lr + image_v_lr
            image_v_fixed_lr = image_v_lr / (illu_lr + 1e-4)
            
            l_spa      = torch.mean(torch.abs(torch.pow(illu_lr - image_v_lr, 2)))
            l_tv       = L_tv(illu_lr)
            l_exp      = torch.mean(L_exp(illu_lr))
            l_sparsity = torch.mean(image_v_fixed_lr)
            loss       = l_spa + 20 * l_tv + 8 * l_exp + 5 * l_sparsity
            
            loss.backward()
            optimizer.step()
        
        # Postprocess
        image_v_fixed   = filter_up(image_v_lr, image_v_fixed_lr, image_v)
        image_hsv_fixed = replace_v_component(image_hsv, image_v_fixed)
        image_rgb_fixed = kornia.color.hsv_to_rgb(image_hsv_fixed)
        image_rgb_fixed = image_rgb_fixed / torch.max(image_rgb_fixed)
        # enhanced        = torch.movedim(image_rgb_fixed, 1, -1)[0].detach().cpu()
        enhanced        = image_rgb_fixed.detach().cpu()
        
        return enhanced
