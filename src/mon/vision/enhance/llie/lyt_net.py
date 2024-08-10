#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements LYT-Net (LYT-Net: Lightweight YUV Transformer-based
Network for Low-Light Image Enhancement) models.

References:
    `<https://github.com/albrateanu/LYT-Net>`__
"""

from __future__ import annotations

__all__ = [
    "LYTNet",
]

from typing import Any, Literal

import torch
from torchvision.models import vgg19, VGG19_Weights

from mon import core, nn
from mon.core import _callable
from mon.globals import MODELS, Scheme
from mon.vision import color
from mon.vision.enhance.llie import base

console = core.console


# region Loss

class Loss(nn.Loss):
    
    def __init__(
        self,
        alpha1: float = 1.00,
        alpha2: float = 0.06,
        alpha3: float = 0.05,
        alpha4: float = 0.5,
        alpha5: float = 0.0083,
        alpha6: float = 0.25,
        reduction: Literal["none", "mean", "sum"] = "mean",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs, reduction=reduction)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.alpha4 = alpha4
        self.alpha5 = alpha5
        self.alpha6 = alpha6
        
        self.smooth_l1_loss  = nn.SmoothL1Loss(reduction=reduction)
        self.perceptual_loss = nn.PerceptualLoss(net=vgg19(weights=VGG19_Weights))
        self.histogram_loss  = nn.HistogramLoss(bins=256, reduction=reduction)
        self.psnr_loss       = nn.PSNRLoss(reduction=reduction)
        self.ssim_loss       = nn.SSIMLoss(reduction)
        
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor,
        **_
    ) -> torch.Tensor:
        smooth_l1_loss  = self.smooth_l1_loss(input, target)
        perceptual_loss = self.perceptual_loss(input, target)
        histogram_loss  = self.histogram_loss(input, target)
        ssim_loss       = self.ssim_loss(input, target)
        psnr_loss       = self.psnr_loss(input, target)
        color_loss      = self._color_loss(input, target)
        loss = (
              self.alpha1 * smooth_l1_loss
            + self.alpha2 * perceptual_loss
            + self.alpha3 * histogram_loss
            + self.alpha4 * ssim_loss
            + self.alpha5 * psnr_loss
            + self.alpha6 * color_loss
        )
        loss = nn.reduce_loss(loss=loss, reduction=self.reduction)
        return loss
    
    @staticmethod
    def _color_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mean_input  = torch.mean(input,  dim=[1, 2])
        mean_target = torch.mean(target, dim=[1, 2])
        abs_diff    = torch.abs(mean_input - mean_target)
        result      = torch.mean(abs_diff)
        return result
    
# endregion


# region Module

class SEBlock(nn.Module):
    
    def __init__(
        self,
        channels       : int,
        reduction_ratio: int  = 16,
        bias           : bool = False,
        device         : Any  = None,
        dtype          : Any  = None,
    ):
        super().__init__()
        self.avg_pool   = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(
                in_features  = channels,
				out_features = channels  // reduction_ratio,
				bias         = bias,
				device       = device,
				dtype        = dtype,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features  = channels  // reduction_ratio,
				out_features = channels,
				bias         = bias,
				device       = device,
				dtype        = dtype,
            ),
            nn.Tanh()
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        return y
    
    
class MSEFBlock(nn.Module):
    
    def __init__(self, channels: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm2d(channels)
        self.dwconv     = nn.DWConv2d(channels, 3)
        self.se_attn    = SEBlock(channels, 16)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x  = input
        x  = self.layer_norm(x)
        x1 = self.dwconv(x)
        x2 = self.se_attn(x)
        y  = x1 * x2
        y  = y + input
        return y


class Denoiser(nn.Module):
    
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1        = nn.Conv2dReLU(1, channels, kernel_size=kernel_size, stride=1)
        self.conv2        = nn.Conv2dReLU(channels, channels, kernel_size=kernel_size, stride=2)
        self.conv3        = nn.Conv2dReLU(channels, channels, kernel_size=kernel_size, stride=2)
        self.conv4        = nn.Conv2dReLU(channels, channels, kernel_size=kernel_size, stride=2)
        self.bottleneck   = nn.MultiHeadAttention(in_channels=channels, num_heads=4)
        self.up2          = nn.Upsample(2)
        self.up3          = nn.Upsample(2)
        self.up4          = nn.Upsample(2)
        self.output_layer = nn.Conv2dTanh(channels, 1, kernel_size=kernel_size, stride=1)
        self.res_layer    = nn.Conv2dTanh(channels, 1, kernel_size=kernel_size, stride=1)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x  = input
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        y = self.bottleneck(x4)
        y = self.up4(y)
        y = self.up3(x3 + y)
        y = self.up2(x2 + y)
        y = y + x1
        y = self.res_layer(y)
        y = self.output_layer(y + input)
        return y
    
# endregion


# region Model

@MODELS.register(name="lyt_net", arch="lyt_net")
class LYTNet(base.LowLightImageEnhancementModel):
    """LYT-Net (LYT-Net: Lightweight YUV Transformer-based Network for Low-Light
    Image Enhancement) model.
    
    References:
        `<https://github.com/albrateanu/LYT-Net>`__

    See Also: :class:`base.LowLightImageEnhancementModel`
    """
    
    arch   : str  = "lyt_net"
    schemes: list[Scheme] = [Scheme.SUPERVISED]
    zoo    : dict = {}

    def __init__(
        self,
        in_channels : int = 3,
        num_channels: int = 32,
        weights     : Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name        = "lyt_net",
            in_channels = in_channels,
            weights     = weights,
            *args, **kwargs
        )
        
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            in_channels  = self.weights.get("in_channels" , in_channels)
            num_channels = self.weights.get("num_channels", num_channels)
        self.in_channels  = in_channels
        self.num_channels = num_channels
        
        # Construct model
        self.process_y   = nn.Conv2dReLU(self.in_channels, self.num_channels, 3, 1)
        self.process_cb  = nn.Conv2dReLU(self.in_channels, self.num_channels, 3, 1)
        self.process_cr  = nn.Conv2dReLU(self.in_channels, self.num_channels, 3, 1)
        self.denoiser_cb = Denoiser(16)
        self.denoiser_cr = Denoiser(16)
        
        self.lum_pool    = nn.MaxPool2d(8)
        self.lum_mhsa    = nn.MultiHeadAttention(in_channels=self.num_channels, num_heads=4)
        self.lum_up      = nn.Upsample(8)
        self.lum_conv    = nn.Conv2d(self.num_channels, self.num_channels, 1)
        self.ref_conv    = nn.Conv2d(self.num_channels, self.num_channels, 1)
        self.msef        = MSEFBlock(self.num_channels)
        
        self.recombine         = nn.Conv2dReLU(self.num_channels, self.num_channels, 3)
        self.final_adjustments = nn.Conv2dTanh(self.num_channels, self.out_channels, 3)
        
        # Loss
        self._loss = Loss()
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        pass
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict | None:
        input  = datapoint.get("input",  None)
        target = datapoint.get("target", None)
        meta   = datapoint.get("meta",   None)
        pred   = self.forward(input=input, *args, **kwargs)
        loss   = self.loss(pred, target)
        return {
            "pred": pred,
            "loss": loss,
        }

    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ycbcr = color.rgb_to_ycbcr(input)
        y     = ycbcr[:, 0, :, :]
        cb    = ycbcr[:, 1, :, :]
        cr    = ycbcr[:, 2, :, :]
        
        cb    = self.denoiser_cb(cb) + cb
        cr    = self.denoiser_cr(cr) + cr
        
        y_processed  = self.process_y(y)
        cb_processed = self.process_cb(cb)
        cr_processed = self.process_cr(cr)
        
        ref = torch.concat([cb_processed, cr_processed], dim=1)
        
        lum        = y_processed
        lum_1      = self.lum_pool(lum)
        lum_1      = self.lum_mhsa(lum_1)
        lum_1      = self.lum_up(lum_1)
        lum        = lum + lum_1
        ref        = self.ref_conv(ref)
        shortcut   = ref
        ref        = ref + 0.2 * self.lum_conv(lum)
        ref        = self.msef(ref)
        ref        = ref + shortcut
        
        recombined = self.recombine(torch.concat([ref, lum], dim=-1))
        output     = self.final_adjustments(recombined)
        return output
    
# endregion
