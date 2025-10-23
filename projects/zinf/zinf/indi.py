#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "indi_noise",
    "indi_transform",
    "sample",
]

import random

import torch
import torch.nn as nn
from tqdm import tqdm


def indi_noise(
    image        : torch.Tensor,
    steps        : int  = 10,
    deterministic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gets a random noise level for InDi transformation.
    
    Args:
        image: Clean input tensor of shape :math:`(B, C, H, W)`.
        steps: Number of discrete noise levels. Default: ``10``.
        deterministic: Whether to use deterministic noise levels. Default: ``True``.
    
    Returns:
        A tuple of :math:`(fct, t)` where ``fct`` is the noise factor tensor
        and ``t`` is the noise level tensor.
    """
    if deterministic:
        bins = torch.linspace(0, 1, steps + 1)
        noise_levels = []
        for x in range(0, image.size(0)):
            step = random.randint(0, steps)
            noise_levels.append(bins[step])
        t   = torch.tensor(noise_levels).float()
        fct = t[:, None, None, None]
    else:
        # Get random value between 0 and 1
        t   = torch.rand(size=(image.shape[0],))
        fct = t[:, None, None, None]
    
    device = image.device
    fct    = fct.to(device)
    t      = t.to(device)
    return fct, t


def indi_transform(fct: torch.Tensor, clean: torch.Tensor, dirty: torch.Tensor) -> torch.Tensor:
    if fct.dim() > clean.dim():
        fct = fct.squeeze(1)
    transformed = (1 - fct) * clean + fct * dirty
    return transformed


def sample(net: nn.Module, x: torch.Tensor, steps: int) -> torch.Tensor:
    net.eval()
    with torch.no_grad():
        for t in tqdm(torch.linspace(1,0, steps + 1, device=x.device)[:-1]):
            time = torch.tensor(t).unsqueeze(0)
            wav  = net(x, time)
            fct  = 1 / (steps * t)
            x    = fct * wav + (1 - fct) * x
    return x
