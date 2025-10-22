#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Zero-Restore model for zero-shot single image restoration.

References:
    - Paper: "Zero-shot Single Image Restoration through Controlled Perturbation
      of Koschmieder's Model," CVPR 2021.
    - Code: https://github.com/aupendu/zero-restore
"""

__all__ = [
    "ZeroRestoreDehaze",
    "ZeroRestoreLLE",
    "ZeroRestoreUE",
]

import abc
import random

import box
import torch

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, nn, Path, Task
from .module import Estimation, EstimationLLIE

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


class ZeroRestore(nn.Module, abc.ABC):
    """Zero-Restore model for image dehazing.
    
    References:
        - Paper: "Zero-shot Single Image Restoration through Controlled Perturbation
          of Koschmieder's Model," CVPR 2021.
        - Code: https://github.com/aupendu/zero-restore
    """
    
    def forward_once(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trans, atm = self.model(image)
        atm        = torch.unsqueeze(torch.unsqueeze(atm, 2), 2)
        atm        = atm.expand_as(image)
        trans      = trans.expand_as(image)
        enhanced   = (image - (1 - trans.clone()) * atm) / trans
        return trans, atm, enhanced
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Optimize
        self.model.load_state_dict(self.state_dict)
        self.model.train()
        optimizer = nn.Adam(self.parameters(), lr=1e-3, weight_decay=1e-2)
        
        for i in range(self.iters):
            optimizer.zero_grad()
            image_  = self.augment(image)
            
            # Forward 1
            trans1, atm1, enhanced1 = self.forward_once(image_)
            
            # Forward 2
            p_x     = 0.9
            image_x = image * p_x + (1 - p_x) * atm1
            trans_x, atm_x, enhanced_x = self.forward_once(image_x)
            
            # Loss
            o_tensor = torch.ones(enhanced1.shape).to(self.device)
            z_tensor = torch.zeros(enhanced1.shape).to(self.device)
            loss_t   = torch.sum((trans_x - p_x * trans1) ** 2)
            loss_a   = torch.sum((atm1 - atm_x) ** 2)
            loss_mx  =   torch.sum(torch.max(enhanced1, o_tensor)) + torch.sum(torch.max(enhanced_x, o_tensor)) - 2 * torch.sum(o_tensor)
            loss_mn  = - torch.sum(torch.min(enhanced1, z_tensor)) - torch.sum(torch.min(enhanced_x, z_tensor))
            loss_col = nn.ColorConstancyLoss()(enhanced1)
            loss_tv  = nn.TotalVariationLoss()(enhanced1)
            loss     = 0.001 * loss_tv + loss_t + loss_a + 0.001 * loss_mx + 0.001 * loss_mn + 1000 * loss_col
            
            loss.backward()
            optimizer.step()
        
        self.model.eval()
        trans, atm, enhanced = self.forward_once(image)
        
        return trans, atm, enhanced
        
    def augment(self, image: torch.Tensor) -> torch.Tensor:
        it = random.randint(0, 7)
        if it == 1:
            image = image.rot90(1, [2, 3])
        if it == 2:
            image = image.rot90(2, [2, 3])
        if it == 3:
            image = image.rot90(3, [2, 3])
        if it == 4:
            image = image.flip(2).rot90(1, [2, 3])
        if it == 5:
            image = image.flip(3).rot90(1, [2, 3])
        if it == 6:
            image = image.flip(2)
        if it == 7:
            image = image.flip(3)
        return image
    

@MODELS.register(name="zerorestore_dehaze", arch="zerorestore")
class ZeroRestoreDehaze(ZeroRestore, ModelMixin):
    """Zero-Restore model for image dehazing.
    
    References:
        - Paper: "Zero-shot Single Image Restoration through Controlled Perturbation
          of Koschmieder's Model," CVPR 2021.
        - Code: https://github.com/aupendu/zero-restore
    """
    
    arch     : str          = "zerorestore"
    name     : str          = "zerorestore_dehaze"
    tasks    : list[Task]   = [Task.DEHAZE]
    mltypes  : list[MLType] = [MLType.ZERO_SHOT]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, num_channels: int = 64, iters: int = 10000):
        super().__init__()
        self.iters      = iters
        self.model      = Estimation(num_channels=num_channels)
        self.state_dict = self.model.state_dict()


@MODELS.register(name="zerorestore_lle", arch="zerorestore")
class ZeroRestoreLLE(ZeroRestore, ModelMixin):
    """Zero-Restore model for low-light image enhancement.
    
    References:
        - Paper: "Zero-shot Single Image Restoration through Controlled Perturbation
          of Koschmieder's Model," CVPR 2021.
        - Code: https://github.com/aupendu/zero-restore
    """
    
    arch     : str          = "zerorestore"
    name     : str          = "zerorestore_lle"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.ZERO_SHOT]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, num_channels: int = 64, iters: int = 10000):
        super().__init__()
        self.iters      = iters
        self.model      = EstimationLLIE(num_channels=num_channels)
        self.state_dict = self.model.state_dict()


@MODELS.register(name="zerorestore_uie", arch="zerorestore")
class ZeroRestoreUE(ZeroRestore, ModelMixin):
    """Zero-Restore model for underwater image enhancement.
    
    References:
        - Paper: "Zero-shot Single Image Restoration through Controlled Perturbation
          of Koschmieder's Model," CVPR 2021.
        - Code: https://github.com/aupendu/zero-restore
    """
    
    arch     : str          = "zerorestore"
    name     : str          = "zerorestore_uie"
    tasks    : list[Task]   = [Task.UNDERWATER]
    mltypes  : list[MLType] = [MLType.ZERO_SHOT]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, num_channels: int = 64, iters: int = 10000):
        super().__init__()
        self.iters      = iters
        self.model      = Estimation(num_channels=num_channels)
        self.state_dict = self.model.state_dict()
