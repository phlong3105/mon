#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16


class ColorLoss(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        b, c, h, w = x.shape
        mean_rgb   = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        d_rg       = torch.pow(mr - mg, 2)
        d_rb       = torch.pow(mr - mb, 2)
        d_gb       = torch.pow(mb - mg, 2)
        k          = torch.pow(torch.pow(d_rg, 2) + torch.pow(d_rb, 2) + torch.pow(d_gb, 2), 0.5)
        return k

			
class SpaLoss(nn.Module):

    def __init__(self):
        super().__init__()
        kernel_left  = torch.FloatTensor([[0,  0, 0], [-1, 1,  0], [0,  0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0,  0, 0], [ 0, 1, -1], [0,  0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up    = torch.FloatTensor([[0, -1, 0], [ 0, 1,  0], [0,  0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down  = torch.FloatTensor([[0,  0, 0], [ 0, 1,  0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left  = nn.Parameter(data=kernel_left,  requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up    = nn.Parameter(data=kernel_up,    requires_grad=False)
        self.weight_down  = nn.Parameter(data=kernel_down,  requires_grad=False)
        self.pool         = nn.AvgPool2d(4)
        
    def forward(self, org, enhance):
        org_mean     = torch.mean(org,     1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)
        org_pool     =  self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)	
 
        d_org_letf  = F.conv2d(org_pool, self.weight_left,  padding=1)
        d_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        d_org_up    = F.conv2d(org_pool, self.weight_up,    padding=1)
        d_org_down  = F.conv2d(org_pool, self.weight_down,  padding=1)

        d_enhance_left  = F.conv2d(enhance_pool, self.weight_left,  padding=1)
        d_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        d_enhance_up    = F.conv2d(enhance_pool, self.weight_up,    padding=1)
        d_enhance_down  = F.conv2d(enhance_pool, self.weight_down,  padding=1)

        d_left  = torch.pow(d_org_letf  - d_enhance_left,  2)
        d_right = torch.pow(d_org_right - d_enhance_right, 2)
        d_up    = torch.pow(d_org_up    - d_enhance_up,    2)
        d_down  = torch.pow(d_org_down  - d_enhance_down,  2)
        E       = (d_left + d_right + d_up + d_down)
        return E
    
    
class ExpLoss(nn.Module):
    
    def __init__(self, patch_size: int, mean_val: float):
        super().__init__()
        self.pool     = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x    = input
        x    = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        d    = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
        return d
      
        
class TVLoss(nn.Module):
    
    def __init__(self, weight: float = 1):
        super().__init__()
        self.weight = weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x       = input
        b       = x.size()[0]
        h_x     = x.size()[2]
        w_x     = x.size()[3]
        count_h = (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv    = torch.pow((x[: , : , 1: ,  : ] - x[: , : , : h_x-1, : ])     , 2).sum()
        w_tv    = torch.pow((x[: , : ,  : , 1: ] - x[: , : , :      , : w_x-1]), 2).sum()
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / b
    
    
class SaLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x          = input
        r, g, b    = torch.split(x, 1, dim=1)
        mean_rgb   = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        dr         = r - mr
        dg         = g - mg
        db         = b - mb
        k          = torch.pow(torch.pow(dr, 2) + torch.pow(db, 2) + torch.pow(dg, 2), 0.5)
        k          = torch.mean(k)
        return k


class PerceptionLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        features         = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        h = self.to_relu_1_2(x)
        h = self.to_relu_2_2(h)
        h = self.to_relu_3_3(h)
        h = self.to_relu_4_3(h)
        return h
