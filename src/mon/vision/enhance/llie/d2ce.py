#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements D2CE (Depth to Curve Estimation Network / Deep Depth
Curve Estimation Network) models.
"""

from __future__ import annotations

__all__ = [
    "D2CE",
]

from typing import Any, Literal

import torch

from mon import core, nn
from mon.core import _callable
from mon.globals import MODELS, Scheme
from mon.vision import filtering, geometry, prior
from mon.vision.depth import depth_anything_v2
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
        norm         : nn.Module | None = nn.AdaptiveBatchNorm2d,
    ):
        super().__init__()
        self.conv = nn.DSConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        #
        if norm is not None:
            self.norm = norm(out_channels)
        else:
            self.norm = nn.Identity()
        #
        if is_last_layer:
            self.relu = nn.Tanh()
        else:
            self.relu = nn.LeakyReLU(relu_slope, inplace=False)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class EnhanceNet(nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        num_channels: int,
        num_iters   : int,
        norm        : nn.Module | None = nn.AdaptiveBatchNorm2d,
    ):
        super().__init__()
        out_channels = 3  # in_channels * num_iters
        self.e_conv1 = ConvBlock(in_channels  + 1, num_channels, norm=norm)
        self.e_conv2 = ConvBlock(num_channels,     num_channels, norm=norm)
        self.e_conv3 = ConvBlock(num_channels,     num_channels, norm=norm)
        self.e_conv4 = ConvBlock(num_channels,     num_channels, norm=norm)
        self.e_conv5 = ConvBlock(num_channels * 2, num_channels, norm=norm)
        self.e_conv6 = ConvBlock(num_channels * 2, num_channels, norm=norm)
        self.e_conv7 = ConvBlock(num_channels * 2, out_channels, norm=norm, is_last_layer=True)
        self.apply(self.init_weights)
    
    def init_weights(self, m: nn.Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            # if hasattr(m, "conv"):
            #     m.conv.weight.data.normal_(0.0, 0.02)    # 0.02
            if hasattr(m, "dw_conv"):
                m.dw_conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "pw_conv"):
                m.pw_conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "weight"):
                m.weight.data.normal_(0.0, 0.02)  # 0.02
            elif classname.find("BatchNorm") != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
    
    def forward(self, input: torch.Tensor, depth: torch.Tensor | None = None) -> torch.Tensor:
        x   = input
        d   = depth
        x1  = self.e_conv1(torch.cat([x, d],  1))
        x2  = self.e_conv2(x1)
        x3  = self.e_conv3(x2)
        x4  = self.e_conv4(x3)
        x5  = self.e_conv5(torch.cat([x3, x4], 1))
        x6  = self.e_conv6(torch.cat([x2, x5], 1))
        x_r = self.e_conv7(torch.cat([x1, x6], 1))
        return x_r

# endregion


# region Model

@MODELS.register(name="d2ce", arch="d2ce")
class D2CE(base.LowLightImageEnhancementModel):
    """D2CE (Depth to Curve Estimation Network / Deep Depth Curve Estimation
    Network) models.
    
    See Also: :class:`base.LowLightImageEnhancementModel`
    """
    
    arch   : str  = "d2ce"
    schemes: list[Scheme] = [Scheme.UNSUPERVISED, Scheme.ZERO_SHOT, Scheme.ZERO_REFERENCE]
    zoo    : dict = {}
    
    def __init__(
        self,
        name        : str   = "d2ce",
        in_channels : int   = 3,
        num_channels: int   = 32,
        num_iters   : int   = 15,
        radius      : int   = 3,
        eps         : float = 1e-4,
        gamma       : float = 2.6,
        de_encoder  : Literal["vits", "vitb", "vitl"] = "vits",
        weights     : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            name        = name,
            in_channels = in_channels,
            weights     = weights,
            *args, **kwargs
        )
        
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            in_channels  = self.weights.get("in_channels" , in_channels)
            num_channels = self.weights.get("num_channels", num_channels)
            num_iters    = self.weights.get("num_iters"   , num_iters)
            radius       = self.weights.get("radius"      , radius)
            eps          = self.weights.get("eps"         , eps)
            gamma        = self.weights.get("gamma"       , gamma)
            de_encoder   = self.weights.get("de_encoder"  , de_encoder)
        self.in_channels  = in_channels or self.in_channels
        self.num_channels = num_channels
        self.num_iters    = num_iters
        self.radius       = radius
        self.eps          = eps
        self.gamma        = gamma
        self.de_encoder   = de_encoder
        
        # Construct model
        self.de = depth_anything_v2.DepthAnythingV2_ViTS(weights="da_2k")
        self.en = EnhanceNet(
            in_channels  = self.in_channels,
            num_channels = self.num_channels,
            num_iters    = self.num_iters,
            norm         = None,  # nn.AdaptiveBatchNorm2d,
        )
        self.gf = filtering.GuidedFilter(radius=self.radius, eps=self.eps)
        
        # Loss
        self._loss = Loss(reduction="mean")
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        
        # Freeze DepthAnythingV2 model
        self.de.eval()
        
    def init_weights(self, m: nn.Module):
        pass
    
    def forward_loss(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Symmetric Loss 1
        i        = input
        i1, i2   = geometry.pair_downsample(i)
        c1_1, c1_2, gf1, j1 = self.forward(input=i1, *args, **kwargs)
        c2_1, c2_2, gf2, j2 = self.forward(input=i2, *args, **kwargs)
        c_1 , c_2 , gf , o  = self.forward(input=i,  *args, **kwargs)
        o1, o2   = geometry.pair_downsample(o)
        mse_loss = nn.MSELoss()
        loss_res = 0.5 * (mse_loss(i1, j2) + mse_loss(i2, j1))
        loss_con = 0.5 * (mse_loss(j1, o1) + mse_loss(j2, o2))
        loss_enh = self.loss(i, c_1, o)
        loss     = 0.5 * (loss_res + loss_con) + 0.5 * loss_enh
        
        # Symmetric Loss 2
        # i        = input
        # i1, i2   = geometry.pair_downsample(i)
        # c1_1, c1_2, gf1, j1 = self.forward(input=i1, *args, **kwargs)
        # c2_1, c2_2, gf2, j2 = self.forward(input=i2, *args, **kwargs)
        # c_1 , c_2 , gf , o  = self.forward(input=i,  *args, **kwargs)
        # n1       = i1 - j1
        # n2       = i2 - j2
        # n        =  i - o
        # o_1, o_2 = geometry.pair_downsample(o)
        # n_1, n_2 = geometry.pair_downsample(n)
        # mse_loss = nn.MSELoss()
        # loss_res = 0.5 * (mse_loss(i1 - n2,  j2) + mse_loss(i2 - n1, j1 ))
        # loss_con = 0.5 * (mse_loss(j1     , o_1) + mse_loss(j2     , o_2))
        # loss_enh = self.loss(i, c_1, o)
        # loss     = 0.5 * (loss_res + loss_con) + 0.5 * loss_enh
        
        # Symmetric Loss 3
        # i        = input
        # i1, i2   = geometry.pair_downsample(i)
        # c1_1, c1_2, gf1, j1 = self.forward(input=i1, *args, **kwargs)
        # c2_1, c2_2, gf2, j2 = self.forward(input=i2, *args, **kwargs)
        # c_1 , c_2 , gf , o  = self.forward(input=i,  *args, **kwargs)
        # n1       = i1 - j1
        # n2       = i2 - j2
        # n        =  i - o
        # o_1, o_2 = geometry.pair_downsample(o)
        # n_1, n_2 = geometry.pair_downsample(n)
        # mse_loss = nn.MSELoss()
        # loss_res = (1 / 3) * (mse_loss(i1 - n2,  j2) + mse_loss(i2 - n1, j1 ))
        # loss_noi = (1 / 3) * (mse_loss(n1     , n_1) + mse_loss(n2     , n_2))
        # loss_con = (1 / 3) * (mse_loss(j1     , o_1) + mse_loss(j2     , o_2))
        # loss_enh = self.loss(i, c_1, o)
        # loss     = 0.5 * (loss_res + loss_con + loss_noi) + 0.5 * loss_enh
        
        return o, loss
    
    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x  = input
        # Enhancement
        d  = self.de(x)
        c1 = self.en(x, d)
        # Enhancement loop
        if self.gamma in [None, 0.0]:
            y  = x
            c2 = None
            for i in range(self.num_iters):
                y = y + c1 * (torch.pow(y, 2) - y)
        else:
            y  = x
            c2 = prior.get_guided_brightness_enhancement_map_prior(x, self.gamma, 9)
            for i in range(0, self.num_iters):
                b = y * (1 - c2)
                d = y * c2
                y = b + d + c1 * (torch.pow(d, 2) - d)
        # Guided Filter
        y_gf = self.gf(x, y)
        # y_gf = y
        return c1, c2, y, y_gf
    
# endregion
