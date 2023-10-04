#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Zero-DACE models."""

from __future__ import annotations

__all__ = ["ZeroDACE"]

import torch

from mon import nn
from mon.nn import functional as F


class ZeroDACE(nn.Module):

    def __init__(self, num_channels=32, scale_factor=1):
        super().__init__()
        self.scale_factor = scale_factor
        self.num_channels = num_channels
        self.act      = nn.LeakyReLU(inplace=True)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        self.conv1    = nn.ABSConv2dS(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
        self.conv2    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
        self.conv3    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
        self.conv4    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
        self.conv5    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
        self.conv6    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
        self.conv7    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
        
    def forward(self, input):
        x = input
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1/self.scale_factor, mode="bilinear")
        x1  = self.act(self.conv1(x_down))
        x2  = self.act(self.conv2(x1))
        x3  = self.act(self.conv3(x2))
        x4  = self.act(self.conv4(x3))
        x5  = self.act(self.conv5(torch.cat([x3, x4], 1)))
        x6  = self.act(self.conv6(torch.cat([x2, x5], 1)))
        x7  = F.tanh(self.conv7(torch.cat([x1, x6], 1)))
        if self.scale_factor == 1:
            x7 = x7
        else:
            x7 = self.upsample(x7)
        #
        y  = x  + x7 * (torch.pow(x, 2)  - x)
        y  = y  + x7 * (torch.pow(y, 2)  - y)
        y  = y  + x7 * (torch.pow(y, 2)  - y)
        y1 = y  + x7 * (torch.pow(y, 2)  - y)
        y  = y1 + x7 * (torch.pow(y1, 2) - y1)
        y  = y  + x7 * (torch.pow(y, 2)  - y)
        y  = y  + x7 * (torch.pow(y, 2)  - y)
        y  = y  + x7 * (torch.pow(y, 2)  - y)
        return y, x7
    
    def init_weights(self, m: nn.Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                m.conv.weight.data.normal_(0.0, 0.02)
            elif hasattr(m, "dw_conv"):
                m.dw_conv.weight.data.normal_(0.0, 0.02)
            elif hasattr(m, "pw_conv"):
                m.pw_conv.weight.data.normal_(0.0, 0.02)
            else:
                m.weight.data.normal_(0.0, 0.02)
                
    def regularization_loss(self, alpha: float = 0.1):
        loss = 0.0
        for sub_module in [
            self.conv1, self.conv2, self.conv3, self.conv4,
            self.conv5, self.conv6, self.conv7
        ]:
            if hasattr(sub_module, "regularization_loss"):
                loss += sub_module.regularization_loss()
        return alpha * loss
