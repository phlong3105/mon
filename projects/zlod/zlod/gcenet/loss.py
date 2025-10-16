#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "L_col",
    "L_col_rate",
    "L_exp",
    "L_exp_value",
    "L_per",
    "L_sa",
    "L_spa",
    "L_tv",
]

import torch
from torchvision.models.vgg import vgg16

from mon.core import nn
from mon.core.nn import functional as F


class L_col(nn.BaseLoss):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_rgb   = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        d_rg       = torch.pow(mr - mg, 2)
        d_rb       = torch.pow(mr - mb, 2)
        d_gb       = torch.pow(mb - mg, 2)
        k          = torch.pow(torch.pow(d_rg, 2) + torch.pow(d_rb, 2) + torch.pow(d_gb, 2), 0.5)
        return k


class L_col_rate(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pre: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
        mr_pre, mg_pre, mb_pre = torch.split(pre * 255, 1, dim=1)
        mr_cur, mg_cur, mb_cur = torch.split(cur * 255, 1, dim=1)
        Drg = torch.pow(mr_pre.int() // mg_pre.int() - mr_cur.int() // mg_cur.int(), 2).sum() / 255.0 ** 2
        Drb = torch.pow(mr_pre.int() // mb_pre.int() - mr_cur.int() // mb_cur.int(), 2).sum() / 255.0 ** 2
        Dgb = torch.pow(mg_pre.int() // mb_pre.int() - mg_cur.int() // mb_cur.int(), 2).sum() / 255.0 ** 2
        k   = torch.pow(Drg + Drb + Dgb, 0.5)
        return k


class L_spa(nn.BaseLoss):

    def __init__(self):
        super().__init__()
        kernel_left       = torch.FloatTensor( [[0,  0, 0], [-1, 1,  0], [0,  0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right      = torch.FloatTensor( [[0,  0, 0], [ 0, 1, -1], [0,  0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up         = torch.FloatTensor( [[0, -1, 0], [ 0, 1,  0], [0,  0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down       = torch.FloatTensor( [[0,  0, 0], [ 0, 1,  0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)
        self.weight_left  = nn.Parameter(data=kernel_left,  requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up    = nn.Parameter(data=kernel_up,    requires_grad=False)
        self.weight_down  = nn.Parameter(data=kernel_down,  requires_grad=False)
        self.pool         = nn.AvgPool2d(4)
        
    def forward(self, input: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
        input_mean    = torch.mean(input,    1, keepdim=True)
        enhanced_mean = torch.mean(enhanced, 1, keepdim=True)

        input_pool    = self.pool(input_mean)
        enhanced_pool = self.pool(enhanced_mean)

        D_input_left  = F.conv2d(input_pool, self.weight_left,  padding=1)
        D_input_right = F.conv2d(input_pool, self.weight_right, padding=1)
        D_input_up    = F.conv2d(input_pool, self.weight_up,    padding=1)
        D_input_down  = F.conv2d(input_pool, self.weight_down,  padding=1)

        D_enhanced_left  = F.conv2d(enhanced_pool, self.weight_left,  padding=1)
        D_enhanced_right = F.conv2d(enhanced_pool, self.weight_right, padding=1)
        D_enhanced_up    = F.conv2d(enhanced_pool, self.weight_up,    padding=1)
        D_enhanced_down  = F.conv2d(enhanced_pool, self.weight_down,  padding=1)

        D_left  = torch.pow(D_input_left  - D_enhanced_left,  2)
        D_right = torch.pow(D_input_right - D_enhanced_right, 2)
        D_up    = torch.pow(D_input_up    - D_enhanced_up,    2)
        D_down  = torch.pow(D_input_down  - D_enhanced_down,  2)
        E       = (D_left + D_right + D_up + D_down)
        # E = 25 * (D_left + D_right + D_up + D_down)

        return E
    
    
class L_exp(nn.BaseLoss):

    def __init__(self, patch_size: int, mean_val: float):
        super().__init__()
        self.pool     = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x    = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        d    = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).to(x.device), 2))
        return d
      

class L_exp_value(nn.Module):

    def __init__(self, patch_size: int, mean_val: float = 0.6):
        super().__init__()
        self.pool     = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    
    def forward(self, x: torch.Tensor):
        x    = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        d    = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).to(x.device), 2))
        return d


class L_tv(nn.BaseLoss):
    
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        count_h    = (x.size()[2] - 1) * x.size()[3]
        count_w    = x.size()[2] * (x.size()[3] - 1)
        h_tv       = torch.pow((x[:, :, 1:, :] - x[:, :, :h - 1, :]), 2).sum()
        w_tv       = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / b
    
    
class L_sa(nn.BaseLoss):
    
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        b, c, h, w = x.shape
        # x_de = x.cpu().detach().numpy()
        r, g, b    = torch.split(x , 1, dim=1)
        mean_rgb   = torch.mean(x,[2,3],keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r - mr
        Dg = g - mg
        Db = b - mb
        k  = torch.pow( torch.pow(Dr,2) + torch.pow(Db,2) + torch.pow(Dg,2),0.5)
        k  = torch.mean(k)
        return k


class L_per(nn.BaseLoss):
    
    def __init__(self):
        super().__init__()
        features = vgg16(pretrained=True).features
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

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_4_3
