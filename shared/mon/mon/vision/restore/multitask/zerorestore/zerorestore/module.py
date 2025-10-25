#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from mon.core import nn


class DoubleConv(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False, padding_mode="reflect"),
            nn.GroupNorm(num_channels=out_channels, num_groups=8, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False, padding_mode="reflect"),
            nn.GroupNorm(num_channels=out_channels, num_groups=8, affine=True),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class InDoubleConv(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 9, stride=4, padding=4, bias=False, padding_mode="reflect"),
            nn.GroupNorm(num_channels=out_channels, num_groups=8, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False, padding_mode="reflect"),
            nn.GroupNorm(num_channels=out_channels, num_groups=8, affine=True),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class InConv(nn.Module):
    
    def __init__(self, in_channels: int = 1, out_channels: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7, stride=4, padding=3, bias=False, padding_mode="reflect"),
            nn.GroupNorm(num_channels=out_channels, num_groups=8, affine=True),
            nn.ReLU(inplace=True)
        )
        self.convf = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False, padding_mode="reflect"),
            nn.GroupNorm(num_channels=out_channels, num_groups=8, affine=True),
            nn.ReLU(inplace=False)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x    = input
        R    = x[:, 0:1, :, :]
        G    = x[:, 1:2, :, :]
        B    = x[:, 2:3, :, :]
        xR   = torch.unsqueeze(self.conv(R), 1)
        xG   = torch.unsqueeze(self.conv(G), 1)
        xB   = torch.unsqueeze(self.conv(B), 1)
        x    = torch.cat([xR, xG, xB], 1)
        x, _ = torch.min(x, dim=1)
        return self.convf(x)


class SKConv(nn.Module):
    
    def __init__(self, in_channels: int = 1, out_channels: int = 64, M: int = 4, L: int = 32):
        super().__init__()
        self.M     = M
        self.convs = nn.ModuleList([])
        in_conv    = InConv(in_channels, out_channels)
        for i in range(M):
            if i == 0:
                self.convs.append(in_conv)
            else:
                self.convs.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=1 / (2 ** i), mode="bilinear", align_corners=True),
                        in_conv,
                        nn.Upsample(scale_factor=2 ** i, mode="bilinear", align_corners=True)
                    )
                )
        self.fc  = nn.Linear(out_channels, L)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(L, out_channels))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x    = input
        feas = None
        for i, conv in enumerate(self.convs):
            fea = conv(x)
            fea = torch.unsqueeze(fea, 1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_u = torch.sum(feas, dim=1)
        fea_s = fea_u.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        
        attention_vectors = None
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z)
            vector = torch.unsqueeze(vector, 1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = torch.unsqueeze(attention_vectors, -1)
        attention_vectors = torch.unsqueeze(attention_vectors, -1)
        
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


class Estimation(nn.Module):

    def __init__(self, num_channels: int = 64):
        super().__init__()
        self.num_channels = num_channels
        self.in_conv      = SKConv(1, num_channels, 3, 32)
        # Transmission Map
        self.conv_t1  = DoubleConv(num_channels, num_channels)
        self.conv_t2  = nn.Conv2d(num_channels, 1, 3, padding=1, stride=1, bias=False, padding_mode="reflect")
        self.up       = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        # Atmospheric Light
        self.conv_a1  = InDoubleConv(3, num_channels)
        self.conv_a2  = DoubleConv(num_channels, num_channels)
        self.maxpool  = nn.MaxPool2d(15, 7)
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.dense    = nn.Linear(num_channels, 3, bias=False)
        
        self.apply(self.init_weights)
        
    def init_weights(self, m: nn.Module):
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1:  # 0.02
            m.weight.data.normal_(0.0, 0.001)
        if classname.find("Linear") != -1:  # 0.02
            m.weight.data.normal_(0.0, 0.001)
            
    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x     = input
        x_min = self.in_conv(x)
        trans = self.conv_t2(self.up(self.conv_t1(x_min)))
        trans = torch.sigmoid(trans) + 1e-12
        atm   = self.conv_a1(x)
        atm   = torch.mul(atm, x_min)
        atm   = self.pool(self.conv_a2(self.maxpool(atm)))
        atm   = atm.view(-1, self.num_channels)
        atm   = torch.sigmoid(self.dense(atm))
        return trans, atm


class EstimationLLIE(nn.Module):

    def __init__(self, num_channels: int = 64):
        super().__init__()
        self.num_channels = num_channels
        self.in_conv      = SKConv(1, num_channels, 3, 32)
        # Transmission Map
        self.conv_t1 = DoubleConv(num_channels, num_channels)
        self.conv_t2 = nn.Conv2d(num_channels, 1, 3, padding=1, stride=1, bias=False, padding_mode="reflect")
        self.up      = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        # Atmospheric Light
        self.conv_a1 = InDoubleConv(3, num_channels)
        self.conv_a2 = DoubleConv(num_channels, num_channels)
        self.conv_a3 = nn.Conv2d(num_channels, 1, 3, padding=1, stride=1, bias=False, padding_mode="reflect")
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x     = input
        x_min = self.in_conv(x)
        trans = self.conv_t2(self.up(self.conv_t1(x_min)))
        trans = torch.sigmoid(trans) + 1e-12
        atm   = self.conv_a1(x)
        atm   = torch.mul(atm, self.up(x_min))
        atm   = self.conv_a3(self.conv_a2(atm))
        atm   = torch.sigmoid(atm)
        return trans, atm
