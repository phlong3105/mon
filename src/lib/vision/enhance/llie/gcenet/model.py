#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

# import pytorch_colors as colors
import torch
import torch.nn.functional as F

import mon
from mon import nn


class enhance_net_nopool(nn.Module):
    
    def __init__(
        self,
        variant      :         str | None = "00000",
        num_channels : int   | str        = 32,
        gamma        : float | str | None = None,
        unsharp_sigma: int   | str | None = None,
        *args, **kwargs
    ):
        super(enhance_net_nopool, self).__init__()
        self.variant       = variant or "00000"
        self.num_channels  = int(num_channels)     if isinstance(num_channels, int)    or (isinstance(num_channels, str)  and num_channels.isdigit())  else 32
        self.gamma         = float(gamma)          if isinstance(gamma, float)         or (isinstance(gamma, str)         and gamma.isdigit())         else None
        self.unsharp_sigma = float(unsharp_sigma)  if isinstance(unsharp_sigma, float) or (isinstance(unsharp_sigma, str) and unsharp_sigma.isdigit()) else None
        
        if self.variant[0:2] == "00":
            self.relu    = nn.ReLU(inplace=True)
            self.conv1   = nn.Conv2d(3,                     self.num_channels, 3, 1, 1, bias=True)
            self.conv2   = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
            self.conv3   = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
            self.conv4   = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
            self.conv5   = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
            self.conv6   = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
            self.conv7   = nn.Conv2d(self.num_channels * 2, 24,                3, 1, 1, bias=True)
            self.maxpool = nn.MaxPool2d(
                2,
                stride         = 2,
                return_indices = False,
                ceil_mode      = False
            )
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        elif self.variant[0:2] == "01":
            self.relu    = nn.LeakyReLU(inplace=True)
            self.conv1   = nn.Conv2d(3,                     self.num_channels, 3, 1, 1, bias=True)
            self.conv2   = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
            self.conv3   = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
            self.conv4   = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
            self.conv5   = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
            self.conv6   = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
            self.conv7   = nn.Conv2d(self.num_channels * 2, 24,                3, 1, 1, bias=True)
            self.attn    = nn.CBAM(channels=self.num_channels)
            self.maxpool = nn.MaxPool2d(
                2,
                stride         = 2,
                return_indices = False,
                ceil_mode      = False
            )
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.variant[0:2] in ["00"]:
            attn = mon.get_guided_brightness_enhancement_map_prior(x, gamma=2.5, denoise_ksize=9)
            x1   = attn * self.relu(self.conv1(x))
            x2   = attn * self.relu(self.conv2(x1))
            x3   = attn * self.relu(self.conv3(x2))
            x4   = attn * self.relu(self.conv4(x3))
            x5   = attn * self.relu(self.conv5(torch.cat([x3, x4], 1)))
            x6   = attn * self.relu(self.conv6(torch.cat([x2, x5], 1)))
            x_r  =           F.tanh(self.conv7(torch.cat([x1, x6], 1)))
        elif self.variant[0:2] in ["01"]:
            attn = mon.get_guided_brightness_enhancement_map_prior(x, gamma=2.5, denoise_ksize=9)
            x1   = attn * self.relu(self.conv1(x))
            x2   = attn * self.relu(self.conv2(x1))
            x3   = attn * self.relu(self.conv3(x2))
            x4   = attn * self.relu(self.conv4(x3))
            x4   = self.attn(x4)
            x5   = attn * self.relu(self.conv5(torch.cat([x3, x4], 1)))
            x6   = attn * self.relu(self.conv6(torch.cat([x2, x5], 1)))
            x_r  =           F.tanh(self.conv7(torch.cat([x1, x6], 1)))
        
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        return enhance_image_1, enhance_image, r
