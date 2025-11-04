#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements RetinexNet model for low-light image enhancement.

References:
    - Paper: "Deep Retinex Decomposition for Low-Light Enhancement," BMCV 2018.
    - Code: https://github.com/aasharma90/RetinexNet_PyTorch
"""

__all__ = [
    "DecomNet",
    "RelightNet",
    "calculate_efficiency_score_decomnet",
    "calculate_efficiency_score_enhancenet"
]

from copy import deepcopy

import thop
import torch

from mon.core import get_model_device, image as I, nn
from mon.core.nn import functional as F


class DecomNet(nn.Module):
    
    def __init__(self, channel=64, kernel_size=3):
        super().__init__()
        # Shallow feature extraction
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3, padding=4, padding_mode="replicate")
        # Activated layers!
        self.net1_convs = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode="replicate"),
            nn.ReLU()
        )
        # Final recon layer
        self.net1_recon = nn.Conv2d(channel, 4, kernel_size, padding=1, padding_mode="replicate")

    def forward(self, input_im):
        input_max = torch.max(input_im, dim=1, keepdim=True)[0]
        input_img = torch.cat((input_max, input_im), dim=1)
        feats0    = self.net1_conv0(input_img)
        featss    = self.net1_convs(feats0)
        outs      = self.net1_recon(featss)
        R         = torch.sigmoid(outs[:, 0:3, :, :])
        L         = torch.sigmoid(outs[:, 3:4, :, :])
        return R, L


class RelightNet(nn.Module):
    
    def __init__(self, channel=64, kernel_size=3):
        super().__init__()
        self.relu           = nn.ReLU()
        self.net2_conv0_1   = nn.Conv2d(4, channel, kernel_size, padding=1, padding_mode="replicate")
        self.net2_conv1_1   = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode="replicate")
        self.net2_conv1_2   = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode="replicate")
        self.net2_conv1_3   = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode="replicate")
        self.net2_deconv1_1 = nn.Conv2d(channel * 2, channel, kernel_size, padding=1, padding_mode="replicate")
        self.net2_deconv1_2 = nn.Conv2d(channel * 2, channel, kernel_size,  padding=1, padding_mode="replicate")
        self.net2_deconv1_3 = nn.Conv2d(channel * 2, channel, kernel_size, padding=1, padding_mode="replicate")
        self.net2_fusion    = nn.Conv2d(channel * 3, channel, kernel_size=1, padding=1, padding_mode="replicate")
        self.net2_output    = nn.Conv2d(channel, 1, kernel_size=3, padding=0)

    def forward(self, input_L=torch.rand(1, 1, 512, 512), input_R=torch.rand(1, 3, 512, 512)):
        device     = get_model_device(self)
        input_L    = input_L.to(device)
        input_R    = input_R.to(device)
        input_img  = torch.cat((input_R, input_L), dim=1)
        out0       = self.net2_conv0_1(input_img)
        out1       = self.relu(self.net2_conv1_1(out0))
        out2       = self.relu(self.net2_conv1_2(out1))
        out3       = self.relu(self.net2_conv1_3(out2))
                   
        out3_up    = F.interpolate(out3, size=(out2.size()[2], out2.size()[3]))
        deconv1    = self.relu(self.net2_deconv1_1(torch.cat((out3_up, out2), dim=1)))
        deconv1_up = F.interpolate(deconv1, size=(out1.size()[2], out1.size()[3]))
        deconv2    = self.relu(self.net2_deconv1_2(torch.cat((deconv1_up, out1), dim=1)))
        deconv2_up = F.interpolate(deconv2, size=(out0.size()[2], out0.size()[3]))
        deconv3    = self.relu(self.net2_deconv1_3(torch.cat((deconv2_up, out0), dim=1)))
                   
        deconv1_rs = F.interpolate(deconv1, size=(input_R.size()[2], input_R.size()[3]))
        deconv2_rs = F.interpolate(deconv2, size=(input_R.size()[2], input_R.size()[3]))
        feats_all  = torch.cat((deconv1_rs, deconv2_rs, deconv3), dim=1)
        feats_fus  = self.net2_fusion(feats_all)
        output     = self.net2_output(feats_fus)
        return output


def calculate_efficiency_score_decomnet(model, imgsz: int = 512):
    # Define input tensor
    h, w  = I.imgsz(imgsz)
    input = torch.rand(1, 3, h, w).to(get_model_device(model))
    # Get FLOPs and Params
    flops, params = thop.profile(deepcopy(model), inputs=(input, ), verbose=False)
    return flops, params


def calculate_efficiency_score_enhancenet(model, imgsz: int = 512):
    # Define input tensor
    h, w  = I.imgsz(imgsz)
    input = torch.rand(1, 1, h, w).to(get_model_device(model))
    mask  = torch.rand(1, 3, h, w).to(get_model_device(model))
    # Get FLOPs and Params
    flops, params = thop.profile(deepcopy(model), inputs=(input, mask, ), verbose=False)
    return flops, params
