#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements DCC-Net (Deep Color Consistent Network for Low
Light-Image Enhancement) models.
"""

from __future__ import annotations

__all__ = [
    "DCCNet",
]

from typing import Any

import torch

from mon import core, nn
from mon.core import _callable
from mon.globals import MODELS, Scheme
from mon.vision.enhance.llie import base
import torchvision.transforms as transforms

console = core.console


# region Loss

class Loss(nn.Loss):
    """Loss = SpatialConsistencyLoss
              + ExposureControlLoss
              + ColorConstancyLoss
              + IlluminationSmoothnessLoss
    """
    
    def __init__(
        self,
        g_weight   : float = 1.0,
        c_weight   : float = 2.0,
        r_weight   : float = 2.0,
        ssim_weight: float = 2,
        tv_weight  : float = 0.1,
        reduction  : str   = "mean",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.g_weight    = g_weight
        self.c_weight    = c_weight
        self.r_weight    = r_weight
        self.ssim_weight = ssim_weight
        self.tv_weight   = tv_weight
        self.loss_ssim   = nn.SSIMLoss(reduction=reduction)
        self.loss_tv     = nn.TotalVariationALoss(reduction=reduction)
        self.gray_transform = transforms.Grayscale(1)
    
    def forward(
        self,
        input     : torch.Tensor,
        target    : torch.Tensor,
        gray      : torch.Tensor,
        color_hist: torch.Tensor,
        enhance   : torch.Tensor,
        **_
    ) -> torch.Tensor:
        b, c, h, w = input.shape
        #
        gray_target = self.gray_transform(target)
        loss_g      = nn.functional.l1_loss(gray, gray_target, reduction="mean") / (h * w)
        #
        # color_hist_target = torch.histc(target, bins=256)
        # print(color_hist.shape)
        # print(color_hist_target.shape)
        # loss_c            = nn.functional.l1_loss(color_hist, color_hist_target, reduction="mean") / (c * 256)
        #
        loss_r    = nn.functional.l1_loss(enhance, input, reduction="mean") / (h * w * c)
        loss_ssim = 1.0 - self.loss_ssim(enhance, target)
        #
        loss = (
              self.g_weight * loss_g
            # + self.c_weight * loss_c
            + self.r_weight * loss_r
            + self.ssim_weight * loss_ssim
        )
        return loss
        
    # endregion


# region Module

class Downscale(nn.Module):
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.main = BasicConv(in_channels, in_channels * 2, 3, 2)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.main(input)


class Upscale(nn.Module):
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.main = BasicConv(in_channels, in_channels // 2, kernel_size=4, activation=True, stride=2, transpose=True)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.main(input)


class BasicConv(nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : int,
        stride      : int,
        bias        : bool = True,
        norm        : bool = True,
        activation  : bool = True,
        transpose   : bool = False,
    ):
        super().__init__()
        if bias and norm:
            bias = False
        padding = kernel_size // 2
        layers  = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        if activation:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.main(input)


class GNet(nn.Module):
    
    def __init__(self, depth: list[int] = [2, 2, 2, 2]):
        super().__init__()
        base_channel = 32
        # encoder
        self.encoder = nn.ModuleList(
            [
                BasicConv(base_channel, base_channel, 3, 1),
                nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
                Downscale(base_channel),
                BasicConv(base_channel * 2, base_channel * 2, 3, 1),
                nn.Sequential(*[RB(base_channel * 2) for _ in range(depth[1])]),
                Downscale(base_channel * 2),
                BasicConv(base_channel * 4, base_channel * 4, 3, 1),
                nn.Sequential(*[RB(base_channel * 4) for _ in range(depth[2])]),
                Downscale(base_channel * 4),
            ]
        )
        # middle
        self.middle  = nn.Sequential(*[RB(base_channel * 8) for _ in range(depth[3])])
        # decoder
        self.decoder = nn.ModuleList(
            [
                Upscale(base_channel * 8),
                BasicConv(base_channel * 8, base_channel * 4, 3, 1),
                nn.Sequential(*[RB(base_channel * 4) for _ in range(depth[2])]),
                Upscale(base_channel * 4),
                BasicConv(base_channel * 4, base_channel * 2, 3, 1),
                nn.Sequential(*[RB(base_channel * 2) for _ in range(depth[1])]),
                Upscale(base_channel * 2),
                BasicConv(base_channel * 2, base_channel, 3, 1),
                nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
            ]
        )
        # conv
        self.conv_first = BasicConv(3, base_channel, 3, 1)
        self.conv_last  = nn.Conv2d(base_channel, 1, 3, 1, 1)
    
    def forward_encoder(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        shortcuts = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            if (i + 2) % 3 == 0:
                shortcuts.append(x)
        return x, shortcuts
    
    def forward_decoder(self, x: torch.Tensor, shortcuts: list[torch.Tensor]) -> torch.Tensor:
        for i in range(len(self.decoder)):
            if (i + 2) % 3 == 0:
                index = len(shortcuts) - (i//3 + 1)
                x = torch.cat([x, shortcuts[index]], 1)
            x = self.decoder[i](x)
        return x
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = self.conv_first(x)
        x, shortcuts = self.forward_encoder(x)
        x = self.middle(x)
        x = self.forward_decoder(x, shortcuts)
        x = self.conv_last(x)
        gray = (torch.tanh(x) + 1) / 2
        return gray


class CNet(nn.Module):
    
    def __init__(self, d_hist: int, depth: list[int] = [2, 2, 2]):
        super().__init__()
        base_channel = 32
        # encoder
        self.encoder    = nn.ModuleList(
            [
                BasicConv(base_channel, base_channel, 3, 1),
                nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
                Downscale(base_channel),
                BasicConv(base_channel * 2, base_channel * 2, 3, 1),
                nn.Sequential(*[RB(base_channel * 2) for _ in range(depth[1])]),
                Downscale(base_channel * 2),
                BasicConv(base_channel * 4, base_channel * 4, 3, 1),
                nn.Sequential(*[RB(base_channel * 4) for _ in range(depth[2])]),
            ]
        )
        self.conv_first = BasicConv(3, base_channel, 3, 1)
        # color hist
        self.conv_color = BasicConv(base_channel * 4, 256 * 3, 3, 1)
        self.pooling    = nn.AdaptiveAvgPool2d(1)
        self.fc         = nn.Linear(256, d_hist)
        self.softmax    = nn.Softmax(dim=2)
        self.d_hist     = d_hist
    
    def forward_encoder(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        shortcuts = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            if (i + 2) % 3 == 0:
                shortcuts.append(x)
        return x, shortcuts
    
    def forward_color(self, x: torch.Tensor) -> torch.Tensor:
        x          = self.conv_color(x)
        x          = self.pooling(x)
        x          = torch.reshape(x, (-1, 3, 256))
        color_hist = self.softmax(self.fc(x))
        return color_hist
    
    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x          = input
        x          = self.conv_first(x)
        x, _       = self.forward_encoder(x)
        color_hist = self.forward_color(x)
        return color_hist, x


class RNet(nn.Module):
    
    def __init__(self, depth: list[int] = [2, 2, 2, 2]):
        super().__init__()
        base_channel = 32
        # encoder
        self.encoder = nn.ModuleList(
            [
                BasicConv(base_channel, base_channel, 3, 1),
                nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
                Downscale(base_channel),
                BasicConv(base_channel * 2, base_channel * 2, 3, 1),
                nn.Sequential(*[RB(base_channel * 2) for _ in range(depth[1])]),
                Downscale(base_channel * 2),
                BasicConv(base_channel * 4, base_channel * 4, 3, 1),
                nn.Sequential(*[RB(base_channel * 4) for _ in range(depth[2])]),
                Downscale(base_channel * 4),
            ]
        )
        # middle
        self.middle  = nn.Sequential(*[RB(base_channel*8) for _ in range(depth[3])])
        # decoder
        self.decoder = nn.ModuleList(
            [
                Upscale(base_channel * 8),
                BasicConv(base_channel * 8, base_channel * 4, 3, 1),
                nn.Sequential(*[RB(base_channel * 4) for _ in range(depth[2])]),
                Upscale(base_channel * 4),
                BasicConv(base_channel * 4, base_channel * 2, 3, 1),
                nn.Sequential(*[RB(base_channel * 2) for _ in range(depth[1])]),
                Upscale(base_channel * 2),
                BasicConv(base_channel * 2, base_channel, 3, 1),
                nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
            ]
        )
        
        # conv
        self.conv_first = BasicConv(4, base_channel, 3, 1)
        self.conv_last  = nn.Conv2d(base_channel, 3, 3, 1, 1)
        self.pce        = PCE()
    
    def forward_encoder(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        shortcuts = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            if (i + 2) % 3 == 0:
                shortcuts.append(x)
        return x, shortcuts
    
    def forward_decoder(self, x: torch.Tensor, shortcuts: list[torch.Tensor]) -> torch.Tensor:
        for i in range(len(self.decoder)):
            if (i + 2) % 3 == 0:
                index = len(shortcuts) - (i // 3 + 1)
                x = torch.cat([x, shortcuts[index]], 1)
            x = self.decoder[i](x)
        return x
    
    def forward(self, img_low: torch.Tensor, gray: torch.Tensor, color_feature: torch.Tensor) -> torch.Tensor:
        x            = torch.cat([img_low, gray], 1)
        x            = self.conv_first(x)
        x, shortcuts = self.forward_encoder(x)
        x            = self.middle(x)
        shortcuts    = self.pce(color_feature, shortcuts)
        x            = self.forward_decoder(x, shortcuts)
        x            = self.conv_last(x)
        img_color    = (torch.tanh(x) + 1) / 2
        return img_color


class PCE(nn.Module):
    # Pyramid color embedding
    
    def __init__(self):
        super().__init__()
        self.cma_3 = CMA(128, 64)
        self.cma_2 = CMA(64,  32)
        self.cma_1 = CMA(32,  16)
    
    def forward(self, c: torch.Tensor, shortcuts: list[torch.Tensor]) -> list[torch.Tensor]:
        # Change channels
        x_3_color, c_2 = self.cma_3(c,   shortcuts[2])
        x_2_color, c_1 = self.cma_2(c_2, shortcuts[1])
        x_1_color, _   = self.cma_1(c_1, shortcuts[0])
        return [x_1_color, x_2_color, x_3_color]


class CMA(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest")
        )
    
    def forward(self, color_feat: torch.Tensor, gray_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: gray image features
        # c: color features
        
        # l1 distance
        channels   = color_feat.shape[1]
        sim_mat_l1 = -torch.abs(gray_feat - color_feat)          # <0  (b,c,h,w)
        sim_mat_l1 = torch.sum(sim_mat_l1, dim=1, keepdim=True)  # (b,1,h,w)
        sim_mat_l1 = torch.sigmoid(sim_mat_l1)                   # (0, 0.5) (b,1,h,w)
        sim_mat_l1 = sim_mat_l1.repeat(1, channels, 1, 1)
        sim_mat_l1 = 2 * sim_mat_l1                              # (0, 1)
        
        # cos distance
        sim_mat_cos = gray_feat * color_feat                       # >0 (b,c,h,w)
        sim_mat_cos = torch.sum(sim_mat_cos, dim=1, keepdim=True)  # (b,1,h,w)
        sim_mat_cos = torch.tanh(sim_mat_cos)                      # (0, 1) (b,1,h,w)
        sim_mat_cos = sim_mat_cos.repeat(1, channels, 1, 1)        # (0, 1)
        
        # similarity matrix
        sim_mat = sim_mat_l1 * sim_mat_cos  # (0, 1)
        
        # color embedding
        x_color = gray_feat + color_feat * sim_mat
        
        # color features upsample
        c_up = self.conv(color_feat)
        
        return x_color, c_up


class RB(nn.Module):
    
    def __init__(self, channels: int):
        super().__init__()
        self.layer_1 = BasicConv(channels, channels, 3, 1)
        self.layer_2 = BasicConv(channels, channels, 3, 1)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.layer_1(x)
        y = self.layer_2(y)
        return y + x

# endregion


# region Model

@MODELS.register(name="dccnet")
class DCCNet(base.LowLightImageEnhancementModel):
    """DCC-Net (Deep Color Consistent Network for Low Light-Image Enhancement)
    model.
    
    Args:
        channels: The first layer's input channel. Default: ``3`` for RGB image.
        num_channels: The number of input and output channels for subsequent
            layers. Default: ``32``.
        num_filters: The number of convolutional layers in the model.
            Default: ``8``.
        
    References:
        `<https://github.com/Ian0926/DCC-Net>`__

    See Also: :class:`base.LowLightImageEnhancementModel`
    """
    
    _scheme: list[Scheme] = [Scheme.SUPERVISED]
    _zoo   : dict = {
        "lol_v1": {
            "url"   : None,
            "path"  : "dccnet/dccnet_lol_v1.pth",
            "d_hist": 64,
            "map"   : {},
        },
    }

    def __init__(
        self,
        channels: int = 3,
        d_hist  : int = 64,
        weights : Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name     = "zerodce",
            channels = channels,
            weights  = weights,
            *args, **kwargs
        )
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            channels = self.weights.get("channels", channels)
            d_hist   = self.weights.get("d_hist",   d_hist)
            
        self._channels = channels
        self.d_hist    = d_hist
        
        # Construct model
        self.g_net = GNet()
        self.c_net = CNet(self.d_hist)
        self.r_net = RNet()
        
        # Loss
        self._loss = Loss()
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        pass

    def forward_loss(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pred = self.forward(input=input, *args, **kwargs)
        gray, color_hist, enhance = pred
        loss = self.loss(
            input      = input,
            target     = target,
            gray       = gray,
            color_hist = color_hist,
            enhance    = enhance,
        )
        return enhance, loss
    
    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gray    = self.g_net(input)
        color_hist, color_feat = self.c_net(input)
        enhance = self.r_net(input, gray, color_feat)
        return gray, color_hist, enhance

# endregion
