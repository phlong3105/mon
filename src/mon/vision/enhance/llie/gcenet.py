#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements GCENet (Guidance Curve Estimation) models."""

from __future__ import annotations

__all__ = [
    "GCENet",
    "GCEUNetPP",
]

from typing import Any, Literal

import kornia
import torch

from mon import core, nn
from mon.core import _callable
from mon.globals import MODELS, Scheme
from mon.nn import functional as F
from mon.vision import prior
from mon.vision.enhance.llie import base

console = core.console


# region Loss

class TotalVariationLoss(nn.Loss):
    """Total Variation Loss on the Illumination (Illumination Smoothness Loss)
    :math:`\mathcal{L}_{tvA}` preserve the monotonicity relations between
    neighboring pixels. It is used to avoid aggressive and sharp changes between
    neighboring pixels.
    
    References:
        `<https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py>`__
    """
    
    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction  : Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None
    ) -> torch.Tensor:
        x       = input
        b       = x.size()[0]
        h_x     = x.size()[2]
        w_x     = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv    = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv    = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        loss    = self.loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / b
        # loss    = base.reduce_loss(loss=loss, reduction=self.reduction)
        return loss
    

class Loss(nn.Loss):
    
    def __init__(
        self,
        exp_patch_size : int   = 16,
        exp_mean_val   : float = 0.6,
        spa_num_regions: Literal[4, 8, 16, 24] = 4,
        spa_patch_size : int   = 4,
        weight_col     : float = 5,
        weight_exp     : float = 10,
        weight_spa     : float = 1,
        weight_tva     : float = 1600,
        reduction      : Literal["none", "mean", "sum"] = "mean",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.weight_col  = weight_col
        self.weight_exp  = weight_exp
        self.weight_spa  = weight_spa
        self.weight_tva  = weight_tva
        
        self.loss_col    = nn.ColorConstancyLoss(reduction=reduction)
        self.loss_exp    = nn.ExposureControlLoss(
            reduction  = reduction,
            patch_size = exp_patch_size,
            mean_val   = exp_mean_val,
        )
        self.loss_spa    = nn.SpatialConsistencyLoss(
            num_regions = spa_num_regions,
            patch_size  = spa_patch_size,
            reduction   = reduction,
        )
        self.loss_tva    = TotalVariationLoss(reduction=reduction)
    
    def forward(
        self,
        input   : torch.Tensor,
        adjust  : torch.Tensor,
        enhance : torch.Tensor,
        **_
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss_col = self.loss_col(input=enhance)               if self.weight_col  > 0 else 0
        loss_exp = self.loss_exp(input=enhance)               if self.weight_exp  > 0 else 0
        loss_spa = self.loss_spa(input=enhance, target=input) if self.weight_spa  > 0 else 0
        if adjust is not None:
            loss_tva = self.loss_tva(input=adjust)  if self.weight_tva > 0 else 0
        else:
            loss_tva = self.loss_tva(input=enhance) if self.weight_tva > 0 else 0
        loss = (
              self.weight_col * loss_col
            + self.weight_exp * loss_exp
            + self.weight_tva * loss_tva
            + self.weight_spa * loss_spa
        )
        return loss
        
# endregion


# region Module

class ConvBlock(nn.Module):
    
    def __init__(
        self,
        in_channels  : int,
        out_channels : int,
        relu_slope   : float = 0.2,
        is_last_layer: bool  = False,
        use_hin      : bool  = True,
    ):
        super().__init__()
        self.use_hin = use_hin
        
        self.conv1   = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        if use_hin:
            self.norm1 = nn.InstanceNorm2d(in_channels // 2, affine=True)
        else:
            self.norm1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.relu1   = nn.LeakyReLU(relu_slope, inplace=False)
        self.simam   = nn.SimAM()
        
        self.conv2   = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        
        self.conv3   = nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1, bias=True)
        self.relu3   = nn.LeakyReLU(relu_slope, inplace=False)
        
        self.conv4   = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        if is_last_layer:
            self.relu4 = nn.Tanh()
        else:
            self.relu4 = nn.LeakyReLU(relu_slope, inplace=False)
        
        self.conv1_3 = nn.Conv2d(in_channels,     in_channels,  1, 1, 0)
        # self.conv2_3 = nn.DSConv2d(in_channels * 2, out_channels, 1, 1, 0)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x    = input
        #
        x1   = self.conv1(x)
        if self.use_hin:
            x1_1, x1_2 = torch.chunk(x1, 2, dim=1)
            x1         = torch.cat([self.norm1(x1_1), x1_2], dim=1)
        else:
            x1 = self.norm1(x1)
        x1   = self.relu1(x1)
        x1   = self.simam(x1)
        #
        x2   = self.conv2(x1)
        #
        x3   = torch.cat([x2, self.conv1_3(x)], dim=1)
        # x2_3 = self.conv2_3(x2)
        #
        x3   = self.conv3(x3)
        x3   = self.relu3(x3)
        #
        x4   = self.conv4(x3)
        x4   = self.relu4(x4)
        #
        # x3   = x3 + x2_3
        #
        return x4

# endregion


# region Model

@MODELS.register(name="gcenet")
class GCENet(base.LowLightImageEnhancementModel):
    """GCENet (Guidance Curve Estimation Network) model.
    
    Args:
        in_channels: The first layer's input channel. Default: ``3`` for RGB image.
        out_channels: The output channel of the network. Default: ``3`` for RGB image.
        num_filters: Output channels for subsequent layers. Default: ``32``.
        num_iters: The number of convolutional layers in the model. Default: ``8``.
        scale_factor: Downsampling/upsampling ratio. Defaults: ``1``.
        gamma: Gamma value for dark channel prior. Default: ``2.8``.
        
    See Also: :class:`base.LowLightImageEnhancementModel`
    """
    
    _scheme: list[Scheme] = [Scheme.UNSUPERVISED, Scheme.ZEROSHOT]
    _zoo   : dict = {}
    
    def __init__(
        self,
        in_channels : int   = 3,
        out_channels: int   = 3,
        num_filters : int   = 32,
        num_iters   : int   = 8,
        scale_factor: int   = 1,
        gamma       : float = 2.8,
        weights     : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            name        = "gcenet",
            in_channels = in_channels,
            weights     = weights,
            *args, **kwargs
        )
        
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            in_channels  = self.weights.get("in_channels" , in_channels)
            out_channels = self.weights.get("out_channels", out_channels)
            num_filters  = self.weights.get("num_filters",  num_filters)
            num_iters    = self.weights.get("num_iters"   , num_iters)
            scale_factor = self.weights.get("scale_factor", scale_factor)
            gamma        = self.weights.get("gamma"       , gamma)
        self.in_channels  = in_channels  or self.in_channels
        self.out_channels = out_channels or self.out_channels
        self.num_filters  = num_filters
        self.num_iters    = num_iters
        self.scale_factor = scale_factor
        self.gamma        = gamma
        
        # Construct model
        self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
        self.enc1     = nn.DSConv2d(self.in_channels,     self.num_filters,  3, 1, 1, bias=True)
        self.enc2     = nn.DSConv2d(self.num_filters,     self.num_filters,  3, 1, 1, bias=True)
        self.enc3     = nn.DSConv2d(self.num_filters,     self.num_filters,  3, 1, 1, bias=True)
        self.mid      = nn.DSConv2d(self.num_filters,     self.num_filters,  3, 1, 1, bias=True)
        self.dec3     = nn.DSConv2d(self.num_filters * 2, self.num_filters,  3, 1, 1, bias=True)
        self.dec2     = nn.DSConv2d(self.num_filters * 2, self.num_filters,  3, 1, 1, bias=True)
        self.dec1     = nn.DSConv2d(self.num_filters * 2, self.out_channels, 3, 1, 1, bias=True)
        self.act      = nn.PReLU()
        
        # Loss
        self._loss = Loss(reduction="mean")
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                m.conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "dw_conv"):
                m.dw_conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "pw_conv"):
                m.pw_conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "weight"):
                m.weight.data.normal_(0.0, 0.02)  # 0.02
    
    def forward_loss(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pred = self.forward(input=input, *args, **kwargs)
        adjust, enhance = pred
        loss = self.loss(input, adjust, enhance)
        return enhance, loss
        
    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = input
        # Denoising
        x = kornia.filters.bilateral_blur(x, (3, 3), 0.1, (1.5, 1.5))
        # Downsampling
        x_down = x
        if self.scale_factor != 1:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode="bilinear")
        enc1   = self.act(self.enc1(x_down))
        enc2   = self.act(self.enc2(enc1))
        enc3   = self.act(self.enc3(enc2))
        mid    = self.act(self.mid(enc3))
        dec3   = self.act(self.dec3(torch.cat([mid,  enc3], dim=1)))
        dec2   = self.act(self.dec2(torch.cat([dec3, enc2], dim=1)))
        dec1   =   F.tanh(self.dec1(torch.cat([dec2, enc1], dim=1)))
        l      = dec1
        # Upsampling
        if self.scale_factor != 1:
            l = self.upsample(l)
        # Enhancement
        if not self.predicting:
            y = x
            for _ in range(self.num_iters):
                y = y + l * (torch.pow(y, 2) - y)
        else:
            y = x
            g = prior.get_guided_brightness_enhancement_map_prior(x, self.gamma, 9)
            for _ in range(self.num_iters):
                b = y * (1 - g)
                d = y * g
                y = b + d + l * (torch.pow(d, 2) - d)
        return l, y


@MODELS.register(name="gceunet++")
class GCEUNetPP(base.LowLightImageEnhancementModel):
    """GCEUNet++ (Guidance Curve Estimation UNet++) model.
    
    Args:
        in_channels: The first layer's input channel. Default: ``3`` for RGB image.
        out_channels: The output channel of the network. Default: ``3`` for RGB image.
        num_filters: Output channels for subsequent layers. Default: ``32``.
        num_iters: The number of convolutional layers in the model. Default: ``8``.
        gamma: Gamma value for dark channel prior. Default: ``2.8``.
        
    See Also: :class:`base.LowLightImageEnhancementModel`
    """
    
    _scheme: list[Scheme] = [Scheme.UNSUPERVISED, Scheme.ZEROSHOT]
    _zoo   : dict = {}
    
    def __init__(
        self,
        in_channels : int   = 3,
        out_channels: int   = 3,
        num_filters : int   = 32,
        num_iters   : int   = 8,
        gamma       : float = 2.8,
        weights     : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            name         = "gceunet++",
            in_channels  = in_channels,
            out_channels = out_channels,
            weights      = weights,
            *args, **kwargs
        )
        
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            in_channels  = self.weights.get("in_channels" , in_channels)
            out_channels = self.weights.get("out_channels", out_channels)
            num_filters  = self.weights.get("num_filters" , num_filters)
            num_iters    = self.weights.get("num_iters"   , num_iters)
            gamma        = self.weights.get("gamma"       , gamma)
        self.in_channels  = in_channels  or self.in_channels
        self.out_channels = out_channels or self.out_channels
        self.num_iters    = num_iters
        self.gamma        = gamma
        self.num_filters  = num_filters
        
        # Construct model
        self.conv1 = ConvBlock(self.in_channels,     self.num_filters, use_hin=False)
        self.conv2 = ConvBlock(self.num_filters,     self.num_filters)
        self.conv3 = ConvBlock(self.num_filters,     self.num_filters)
        self.conv4 = ConvBlock(self.num_filters,     self.num_filters)
        self.conv5 = ConvBlock(self.num_filters * 2, self.num_filters)
        self.conv6 = ConvBlock(self.num_filters * 2, self.num_filters)
        self.conv7 = ConvBlock(self.num_filters * 2, self.out_channels, is_last_layer=True)
        
        # Loss
        self._loss = Loss(reduction="mean")
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                m.conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "dw_conv"):
                m.dw_conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "pw_conv"):
                m.pw_conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "weight"):
                m.weight.data.normal_(0.0, 0.02)  # 0.02
    
    def forward_loss(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pred = self.forward(input=input, *args, **kwargs)
        adjust, enhance = pred
        loss = self.loss(input, adjust, enhance)
        return enhance, loss
    
    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = input
        
        # Denoising
        # x = kornia.filters.bilateral_blur(x, (3, 3), 0.1, (1.5, 1.5))
        
        # Forward Pass
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(torch.cat([x4, x3], dim=1))
        x6 = self.conv6(torch.cat([x5, x2], dim=1))
        x7 = self.conv7(torch.cat([x6, x1], dim=1))
        l  = x7
        
        # Enhancement
        if not self.predicting:
            y = x
            for _ in range(self.num_iters):
                y = y + l * (torch.pow(y, 2) - y)
        else:
            y = x
            g = prior.get_guided_brightness_enhancement_map_prior(x, self.gamma, 9)
            for _ in range(self.num_iters):
                b = y * (1 - g)
                d = y * g
                y = b + d + l * (torch.pow(d, 2) - d)
        
        return l, y


'''
@MODELS.register(name="gceunet++")
class GCEUNetPP(base.LowLightImageEnhancementModel):
    """GCEUNet++ (Guidance Curve Estimation UNet++) model.
    
    Args:
        in_channels: The first layer's input channel. Default: ``3`` for RGB image.
        out_channels: The output channel of the network. Default: ``3`` for RGB image.
        num_filters: Output channels for subsequent layers. Default: ``64``.
        num_iters: The number of convolutional layers in the model. Default: ``8``.
        gamma: Gamma value for dark channel prior. Default: ``2.8``.
        
    See Also: :class:`base.LowLightImageEnhancementModel`
    """
    
    _scheme: list[Scheme] = [Scheme.UNSUPERVISED, Scheme.ZEROSHOT]
    _zoo   : dict = {}
    
    def __init__(
        self,
        in_channels : int   = 3,
        out_channels: int   = 3,
        depth       : Literal[1, 2, 3, 4] = 4,
        num_iters   : int   = 8,
        gamma       : float = 2.8,
        weights     : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            name         = "gceunet++",
            in_channels  = in_channels,
            out_channels = out_channels,
            weights      = weights,
            *args, **kwargs
        )
        
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            in_channels  = self.weights.get("in_channels" , in_channels)
            out_channels = self.weights.get("out_channels", out_channels)
            depth        = self.weights.get("depth"       , depth)
            num_iters    = self.weights.get("num_iters"   , num_iters)
            gamma        = self.weights.get("gamma"       , gamma)
        self.in_channels  = in_channels  or self.in_channels
        self.out_channels = out_channels or self.out_channels
        self.depth        = depth
        self.num_iters    = num_iters
        self.gamma        = gamma
        self.num_filters  = [32, 64, 128, 256, 512]
        
        # Construct model
        #
        self.pool    = nn.MaxPool2d(2, 2)
        self.up      = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # Depth 0
        self.conv0_0 = UNetConvBlock(self.in_channels, self.num_filters[0])
        # Depth 1
        if self.depth in [1, 2, 3, 4]:
            self.conv1_0 = UNetConvBlock(self.num_filters[0],                       self.num_filters[1])
            self.conv0_1 = UNetConvBlock(self.num_filters[0] + self.num_filters[1], self.num_filters[0])
        # Depth 2
        if self.depth in [2, 3, 4]:
            self.conv2_0 = UNetConvBlock(self.num_filters[1],                           self.num_filters[2])
            self.conv1_1 = UNetConvBlock(self.num_filters[1]     + self.num_filters[2], self.num_filters[1])
            self.conv0_2 = UNetConvBlock(self.num_filters[0] * 2 + self.num_filters[1], self.num_filters[0])
        # Depth 3
        if self.depth in [3, 4]:
            self.conv3_0 = UNetConvBlock(self.num_filters[2],                           self.num_filters[3])
            self.conv2_1 = UNetConvBlock(self.num_filters[2]     + self.num_filters[3], self.num_filters[2])
            self.conv1_2 = UNetConvBlock(self.num_filters[1] * 2 + self.num_filters[2], self.num_filters[1])
            self.conv0_3 = UNetConvBlock(self.num_filters[0] * 3 + self.num_filters[1], self.num_filters[0])
        # Depth 4
        if self.depth in [4]:
            self.conv4_0 = UNetConvBlock(self.num_filters[3],                           self.num_filters[4])
            self.conv3_1 = UNetConvBlock(self.num_filters[3]     + self.num_filters[4], self.num_filters[3])
            self.conv2_2 = UNetConvBlock(self.num_filters[2] * 2 + self.num_filters[3], self.num_filters[2])
            self.conv1_3 = UNetConvBlock(self.num_filters[1] * 3 + self.num_filters[2], self.num_filters[1])
            self.conv0_4 = UNetConvBlock(self.num_filters[0] * 4 + self.num_filters[1], self.num_filters[0])
        # Final
        self.final   = nn.Conv2d(self.num_filters[0], self.out_channels, kernel_size=1)
        
        # self.conv0_0 = UNetConvBlock(self.in_channels,    self.num_filters[0])
        # self.conv1_0 = UNetConvBlock(self.num_filters[0], self.num_filters[1])
        # self.conv2_0 = UNetConvBlock(self.num_filters[1], self.num_filters[2])
        # self.conv3_0 = UNetConvBlock(self.num_filters[2], self.num_filters[3])
        # self.conv4_0 = UNetConvBlock(self.num_filters[3], self.num_filters[4])
        #
        # self.conv0_1 = UNetConvBlock(self.num_filters[0] + self.num_filters[1], self.num_filters[0])
        # self.conv1_1 = UNetConvBlock(self.num_filters[1] + self.num_filters[2], self.num_filters[1])
        # self.conv2_1 = UNetConvBlock(self.num_filters[2] + self.num_filters[3], self.num_filters[2])
        # self.conv3_1 = UNetConvBlock(self.num_filters[3] + self.num_filters[4], self.num_filters[3])
        #
        # self.conv0_2 = UNetConvBlock(self.num_filters[0] * 2 + self.num_filters[1], self.num_filters[0])
        # self.conv1_2 = UNetConvBlock(self.num_filters[1] * 2 + self.num_filters[2], self.num_filters[1])
        # self.conv2_2 = UNetConvBlock(self.num_filters[2] * 2 + self.num_filters[3], self.num_filters[2])
        #
        # self.conv0_3 = UNetConvBlock(self.num_filters[0] * 3 + self.num_filters[1], self.num_filters[0])
        # self.conv1_3 = UNetConvBlock(self.num_filters[1] * 3 + self.num_filters[2], self.num_filters[1])
        #
        # self.conv0_4 = UNetConvBlock(self.num_filters[0] * 4 + self.num_filters[1], self.num_filters[0])
        #
        # self.final   = nn.Conv2d(self.num_filters[0], self.out_channels, kernel_size=1)
        
        # self.enc1     = nn.DSConv2d(self.in_channels,     self.num_filters,  3, 1, 1, bias=True)
        # self.enc2     = nn.DSConv2d(self.num_filters,     self.num_filters,  3, 1, 1, bias=True)
        # self.enc3     = nn.DSConv2d(self.num_filters,     self.num_filters,  3, 1, 1, bias=True)
        # self.mid      = nn.DSConv2d(self.num_filters,     self.num_filters,  3, 1, 1, bias=True)
        # self.dec3     = nn.DSConv2d(self.num_filters * 2, self.num_filters,  3, 1, 1, bias=True)
        # self.dec2     = nn.DSConv2d(self.num_filters * 2, self.num_filters,  3, 1, 1, bias=True)
        # self.dec1     = nn.DSConv2d(self.num_filters * 2, self.out_channels, 3, 1, 1, bias=True)
        # self.act      = nn.PReLU()
        
        # Loss
        self._loss    = Loss(reduction="mean")
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                m.conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "dw_conv"):
                m.dw_conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "pw_conv"):
                m.pw_conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "weight"):
                m.weight.data.normal_(0.0, 0.02)  # 0.02
    
    def forward_loss(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pred = self.forward(input=input, *args, **kwargs)
        adjust, enhance = pred
        loss = self.loss(input, adjust, enhance)
        return enhance, loss
        
    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = input
        
        # Denoising
        # x = kornia.filters.bilateral_blur(x, (3, 3), 0.1, (1.5, 1.5))
        
        # Forward Pass
        x0_0 = self.conv0_0(x)
        if self.depth in [1, 2, 3, 4]:
            x1_0 = self.conv1_0(self.pool(x0_0))
            x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
            l    = x0_1
        if self.depth in [2, 3, 4]:
            x2_0 = self.conv2_0(self.pool(x1_0))
            x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
            x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
            l    = x0_2
        if self.depth in [3, 4]:
            x3_0 = self.conv3_0(self.pool(x2_0))
            x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
            x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
            x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
            l    = x0_3
        if self.depth in [4]:
            x4_0 = self.conv4_0(self.pool(x3_0))
            x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
            x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
            x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
            l    = x0_4
        y = self.final(l)
        y = torch.clamp(y, 0, 1)
        # Forward Pass
        # enc1   = self.act(self.enc1(x))
        # enc2   = self.act(self.enc2(enc1))
        # enc3   = self.act(self.enc3(enc2))
        # mid    = self.act(self.mid(enc3))
        # dec3   = self.act(self.dec3(torch.cat([mid,  enc3], dim=1)))
        # dec2   = self.act(self.dec2(torch.cat([dec3, enc2], dim=1)))
        # dec1   =   F.tanh(self.dec1(torch.cat([dec2, enc1], dim=1)))
        # l      = dec1
        
        # Enhancement
        # y = x
        # for _ in range(self.num_iters):
        #      y = y + l * (torch.pow(y, 2) - y)
        
        # Enhancement
        # if not self.predicting:
        #     y = x
        #     for _ in range(self.num_iters):
        #         y = y + l * (torch.pow(y, 2) - y)
        # else:
        #     y = x
        #     g = proc.get_guided_brightness_enhancement_map_prior(x, self.gamma, 9)
        #     for _ in range(self.num_iters):
        #         b = y * (1 - g)
        #         d = y * g
        #         y = b + d + l * (torch.pow(d, 2) - d)
      
        return None, y
'''

# endregion
