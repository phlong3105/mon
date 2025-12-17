# -*- coding: utf-8 -*-

__all__ = [
    "psnr",
]

import torch
import torch.nn as nn

mse = nn.MSELoss()


def psnr(input: torch.Tensor, target: torch.Tensor, max_val: float) -> torch.Tensor:
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"``input`` must be a torch.Tensor, got {type(input)}.")
    if not isinstance(target, torch.Tensor):
        raise TypeError(f"``target`` must be a torch.Tensor, got {type(target)}.")
    if input.shape != target.shape:
        raise TypeError(f"``input`` and ``target`` must have the same shape, got {input.shape} != {target.shape}.")
    
    return 10.0 * torch.log10(max_val ** 2 / mse(input, target))
