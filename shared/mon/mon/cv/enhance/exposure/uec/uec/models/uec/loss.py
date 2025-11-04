#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "CosineLoss",
    "TVLoss",
]

import torch
import torch.nn as nn


class TVLoss(nn.Module):
    
    def __init__(self, tvloss_weight: int = 1):
        super().__init__()
        self.tvloss_weight = tvloss_weight
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        b       = input.size()[0]
        h_x     = input.size()[2]
        w_x     = input.size()[3]
        count_h = self._tensor_size(input[:, :, 1:, :])
        count_w = self._tensor_size(input[:, :, :, 1:])
        h_tv    = torch.pow((input[:, :, 1:, :] - input[:, :, :h_x - 1, :]), 2).sum()
        w_tv    = torch.pow((input[:, :, :, 1:] - input[:, :, :, :w_x - 1]), 2).sum()
        return self.tvloss_weight * 2 * (h_tv / count_h + w_tv / count_w) / b
    
    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
    

class CosineLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, c, h, w = input.size()
        input  = input.permute(0, 2, 3, 1).view(-1, c)
        target = target.permute(0, 2, 3, 1).view(-1, c)
        loss   = 1.0 - self.cos(input, target).sum() / (1.0 * b * h * w)
        return loss
