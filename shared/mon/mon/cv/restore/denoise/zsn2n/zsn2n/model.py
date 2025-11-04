#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ZS-N2N model for zero-shot image denoising.

References:
    - Paper: "Zero-Shot Noise2Noise: Efficient Image Denoising without any Data," CVPR 2023.
    - Code: https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing#scrollTo=Srf0GQTYrkxA
"""

__all__ = [
    "ZSN2N",
]

import torch

from mon.constants import MODELS
from mon.core import image as I, MLType, ModelMixin, nn, Path, Task

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


class Network(nn.Module):
    
    def __init__(self, in_channels: int = 3):
        super().__init__()
        hidden_dim = 48
        self.conv1 = nn.Conv2d(in_channels, hidden_dim,  kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim,  hidden_dim,  kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim,  in_channels, kernel_size=1)
        self.act   = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x
    

@MODELS.register(name="zsn2n", arch="zsn2n")
class ZSN2N(nn.Module, ModelMixin):
    """ZS-N2N model for zero-shot image denoising.
    
    Args:
        in_channels: The first layer's input channel. Default: ``3`` for RGB image.
        iters: Number of optimization iterations. Default: ``3000``.
    
    References:
        - Paper: "Zero-Shot Noise2Noise: Efficient Image Denoising without any Data," CVPR 2023.
        - Code: https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing#scrollTo=Srf0GQTYrkxA
    """
    
    arch     : str          = "zsn2n"
    name     : str          = "zsn2n"
    tasks    : list[Task]   = [Task.DENOISE]
    mltypes  : list[MLType] = [MLType.ZERO_SHOT]
    model_dir: Path         = root_dir
    zoo      : dict         = {}
    
    def __init__(self, in_channels: int = 3, iters: int = 3000):
        super().__init__()
        self.iters      = iters
        self.model      = Network(in_channels=in_channels)
        self.state_dict = self.model.state_dict()
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        noisy  = image
        device = noisy.device
        
        # Optimize
        self.model.load_state_dict(self.state_dict)
        self.model.train()
        optimizer = nn.Adam(self.model.parameters(), lr=0.001)
        scheduler = nn.StepLR(optimizer, step_size=1000, gamma=0.5)
        mse       = nn.MSELoss().to(device)
        
        for i in range(self.iters):
            noisy1, noisy2       = I.pair_downsample(noisy)
            pred1                = noisy1 - self.model(noisy1)
            pred2                = noisy2 - self.model(noisy2)
            loss_res             = 0.5 * (mse(noisy1, pred2) + mse(noisy2, pred1))
            noisy_denoised       = noisy  - self.model(noisy)
            denoised1, denoised2 = I.pair_downsample(noisy_denoised)
            loss_cons            = 0.5 * (mse(pred1, denoised1) + mse(pred2, denoised2))
            loss                 = loss_res + loss_cons
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # Postprocess
        self.model.eval()
        with torch.no_grad():
            enhanced = torch.clamp(noisy - self.model(noisy), 0, 1)
            enhanced = enhanced.detach().cpu()
        
        return enhanced
