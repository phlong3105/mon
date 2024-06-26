#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements GCENet (Guided Curve Estimation Network) models."""

from __future__ import annotations

__all__ = [
    "GCENet",
    "GCENetA1",
    "GCENetA2",
    "GCENetB1",
    "GCENetB2",
]

from typing import Any, Literal

import kornia
import torch

from mon import core, nn
from mon.core import _callable
from mon.globals import MODELS, Scheme
from mon.nn import functional as F, init
from mon.vision import filtering, prior
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

# noinspection PyMethodMayBeStatic
class LRNet(nn.Module):
    
    def __init__(
        self,
        in_channels : int       = 3,
        mid_channels: int       = 24,
        layers      : int       = 5,
        relu_slope  : float     = 0.2,
        norm        : nn.Module = nn.AdaptiveBatchNorm2d,
    ):
        super().__init__()
        net = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            norm(mid_channels),
            nn.LeakyReLU(relu_slope, inplace=True),
        ]
        for l in range(1, layers):
            net += [
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=2**l, dilation=2**l, bias=False),
                norm(mid_channels),
                nn.LeakyReLU(relu_slope, inplace=True)
            ]
        net += [
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            norm(mid_channels),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        ]
        self.net = nn.Sequential(*net)
        self.apply(self._init_weights)
        
    def _init_weights(self, m: nn.Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            n_out, n_in, h, w = m.weight.data.size()
            # Last Layer
            if n_out < n_in:
                init.xavier_uniform_(m.weight.data)
                return
            # Except Last Layer
            m.weight.data.zero_()
            ch, cw = h // 2, w // 2
            for i in range(n_in):
                m.weight.data[i, i, ch, cw] = 1.0
        elif classname.find("AdaptiveBatchNorm2d") != -1:
            init.constant_(m.bn.weight.data, 1.0)
            init.constant_(m.bn.bias.data,   0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data,   0.0)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.net(input)


# noinspection PyMethodMayBeStatic
class GuidedMap(nn.Module):
    
    def __init__(
        self,
        in_channels: int   = 3,
        channels   : int   = 64,
        dilation   : int   = 0,
        relu_slope : float = 0.2,
        norm       : nn.Module = nn.AdaptiveBatchNorm2d,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False) \
            if dilation == 0 \
            else nn.Conv2d(in_channels, channels, 5, padding=channels, dilation=dilation, bias=False)
        self.norm  = norm(channels)
        self.relu  = nn.LeakyReLU(relu_slope, inplace=True)
        self.conv2 = nn.Conv2d(channels, in_channels, 1)
        self.apply(self._init_weights)
        
    def _init_weights(self, m: nn.Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            n_out, n_in, h, w = m.weight.data.size()
            # Last Layer
            if n_out < n_in:
                init.xavier_uniform_(m.weight.data)
                return
            # Except Last Layer
            m.weight.data.zero_()
            ch, cw = h // 2, w // 2
            for i in range(n_in):
                m.weight.data[i, i, ch, cw] = 1.0
        elif classname.find("AdaptiveBatchNorm2d") != -1:
            init.constant_(m.bn.weight.data, 1.0)
            init.constant_(m.bn.bias.data,   0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data,   0.0)
            
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x
    

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


# noinspection PyMethodMayBeStatic
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
        self.e_conv1 = ConvBlock(in_channels,      num_channels, norm=norm)
        self.e_conv2 = ConvBlock(num_channels,     num_channels, norm=norm)
        self.e_conv3 = ConvBlock(num_channels,     num_channels, norm=norm)
        self.e_conv4 = ConvBlock(num_channels,     num_channels, norm=norm)
        self.e_conv5 = ConvBlock(num_channels * 2, num_channels, norm=norm)
        self.e_conv6 = ConvBlock(num_channels * 2, num_channels, norm=norm)
        self.e_conv7 = ConvBlock(num_channels * 2, out_channels, norm=norm, is_last_layer=True)
        self.apply(self._init_weights)
        
    def _init_weights(self, m: nn.Module):
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
            
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x   = input
        x1  = self.e_conv1(x)
        x2  = self.e_conv2(x1)
        x3  = self.e_conv3(x2)
        x4  = self.e_conv4(x3)
        x5  = self.e_conv5(torch.cat([x3, x4], 1))
        x6  = self.e_conv6(torch.cat([x2, x5], 1))
        x_r = self.e_conv7(torch.cat([x1, x6], 1))
        return x_r


class DenoiseNet(nn.Module):
    
    def __init__(
        self,
        in_channels : int   = 3,
        num_channels: int   = 48,
        relu_slope  : float = 0.2,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,  num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_channels, in_channels,  kernel_size=1)
        self.act   = nn.LeakyReLU(negative_slope=relu_slope, inplace=True)
            
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        y = self.conv3(x)
        return y

# endregion


# region Model

@MODELS.register(name="gcenet")
class GCENet(base.LowLightImageEnhancementModel):
    """GCENet (Guided Curve Estimation Network) model.
    
    See Also: :class:`base.LowLightImageEnhancementModel`
    """
    
    _scheme: list[Scheme] = [Scheme.UNSUPERVISED, Scheme.ZEROSHOT]
    _zoo   : dict = {}
    
    def __init__(
        self,
        name        : str   = "gcenet",
        in_channels : int   = 3,
        num_channels: int   = 32,
        num_iters   : int   = 13,
        radius      : int   = 3,
        eps         : float = 1e-3,
        gamma       : float = 2.8,
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
        self.in_channels  = in_channels or self.in_channels
        self.num_channels = num_channels
        self.num_iters    = num_iters
        self.radius       = radius
        self.eps          = eps
        self.gamma        = gamma
        
        # Construct model
        self.en = EnhanceNet(
            in_channels  = self.in_channels,
            num_channels = self.num_channels,
            num_iters    = self.num_iters,
            norm         = None,
        )
        self.gf = filtering.GuidedFilter(radius=self.radius, eps=self.eps)
        
        # Loss
        self._loss = Loss(reduction="mean")
        
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
        # Symmetric Loss 1
        i        = input
        i1, i2   = self.pair_downsampler(i)
        c1_1, c1_2, gf1, j1 = self.forward(input=i1, *args, **kwargs)
        c2_1, c2_2, gf2, j2 = self.forward(input=i2, *args, **kwargs)
        c_1 , c_2 , gf , o  = self.forward(input=i,  *args, **kwargs)
        o1, o2   = self.pair_downsampler(o)
        mse_loss = nn.MSELoss()
        loss_res = 0.5 * (mse_loss(i1, j2) + mse_loss(i2, j1))
        loss_con = 0.5 * (mse_loss(j1, o1) + mse_loss(j2, o2))
        loss_enh = self.loss(i, c_1, o)
        loss     = 0.5 * (loss_res + loss_con) + 0.5 * loss_enh
        
        # Symmetric Loss 2
        # i        = input
        # i1, i2   = self.pair_downsampler(i)
        # c1_1, c1_2, gf1, j1 = self.forward(input=i1, *args, **kwargs)
        # c2_1, c2_2, gf2, j2 = self.forward(input=i2, *args, **kwargs)
        # c_1 , c_2 , gf , o  = self.forward(input=i,  *args, **kwargs)
        # n1       = i1 - j1
        # n2       = i2 - j2
        # n        =  i - o
        # o_1, o_2 = self.pair_downsampler(o)
        # n_1, n_2 = self.pair_downsampler(n)
        # mse_loss = nn.MSELoss()
        # loss_res = 0.5 * (mse_loss(i1 - n2,  j2) + mse_loss(i2 - n1, j1 ))
        # loss_con = 0.5 * (mse_loss(j1     , o_1) + mse_loss(j2     , o_2))
        # loss_enh = self.loss(i, c_1, o)
        # loss     = 0.5 * (loss_res + loss_con) + 0.5 * loss_enh
        
        # Symmetric Loss 3
        # i        = input
        # i1, i2   = self.pair_downsampler(i)
        # c1_1, c1_2, gf1, j1 = self.forward(input=i1, *args, **kwargs)
        # c2_1, c2_2, gf2, j2 = self.forward(input=i2, *args, **kwargs)
        # c_1 , c_2 , gf , o  = self.forward(input=i,  *args, **kwargs)
        # n1       = i1 - j1
        # n2       = i2 - j2
        # n        =  i - o
        # o_1, o_2 = self.pair_downsampler(o)
        # n_1, n_2 = self.pair_downsampler(n)
        # mse_loss = nn.MSELoss()
        # loss_res = (1 / 3) * (mse_loss(i1 - n2,  j2) + mse_loss(i2 - n1, j1 ))
        # loss_noi = (1 / 3) * (mse_loss(n1     , n_1) + mse_loss(n2     , n_2))
        # loss_con = (1 / 3) * (mse_loss(j1     , o_1) + mse_loss(j2     , o_2))
        # loss_enh = self.loss(i, c_1, o)
        # loss     = 0.5 * (loss_res + loss_con + loss_noi) + 0.5 * loss_enh
        
        return o, loss
    
    def forward_debug(self, input: torch.Tensor, *args, **kwargs) -> dict | None:
        i      = input
        i1, i2 = self.pair_downsampler(i)
        c1_1, c1_2, gf1, j1 = self.forward(input=i1, *args, **kwargs)
        c2_1, c2_2, gf2, j2 = self.forward(input=i2, *args, **kwargs)
        c_1 , c_2 , gf , o  = self.forward(input=i,  *args, **kwargs)
        o1, o2   = self.pair_downsampler(o)
        #
        c_1        = c_1  * -1
        c1_1       = c1_1 * -1
        c2_1       = c2_1 * -1
        d_gf_i_1   = (gf1 - i1) * -10
        d_gf_i_2   = (gf2 - i2) * -10
        d_gf_i     = (gf  - i ) * -10
        d_gf_j_1   = (gf1 - j1) * -10
        d_gf_j_2   = (gf2 - j2) * -10
        d_gf_j     = (gf  - o ) * -10
        d_j_i_1    = (j1  - i1) * -10
        d_j_i_2    = (j2  - i2) * -10
        d_j_i      = (o   - i ) * -10
        #
        c_1_r      =      c_1[:, 0:1, :, :]
        c_1_g      =      c_1[:, 1:2, :, :]
        c_1_b      =      c_1[:, 2:3, :, :]
        c1_1_r     =     c1_1[:, 0:1, :, :]
        c1_1_g     =     c1_1[:, 1:2, :, :]
        c1_1_b     =     c1_1[:, 2:3, :, :]
        c2_1_r     =     c2_1[:, 0:1, :, :]
        c2_1_g     =     c2_1[:, 1:2, :, :]
        c2_1_b     =     c2_1[:, 2:3, :, :]
        d_gf_i_1_r = d_gf_i_1[:, 0:1, :, :]
        d_gf_i_1_g = d_gf_i_1[:, 1:2, :, :]
        d_gf_i_1_b = d_gf_i_1[:, 2:3, :, :]
        d_gf_i_2_r = d_gf_i_2[:, 0:1, :, :]
        d_gf_i_2_g = d_gf_i_2[:, 1:2, :, :]
        d_gf_i_2_b = d_gf_i_2[:, 2:3, :, :]
        d_gf_i_r   =   d_gf_i[:, 0:1, :, :]
        d_gf_i_g   =   d_gf_i[:, 1:2, :, :]
        d_gf_i_b   =   d_gf_i[:, 2:3, :, :]
        d_gf_j_1_r = d_gf_j_1[:, 0:1, :, :]
        d_gf_j_1_g = d_gf_j_1[:, 1:2, :, :]
        d_gf_j_1_b = d_gf_j_1[:, 2:3, :, :]
        d_gf_j_2_r = d_gf_j_2[:, 0:1, :, :]
        d_gf_j_2_g = d_gf_j_2[:, 1:2, :, :]
        d_gf_j_2_b = d_gf_j_2[:, 2:3, :, :]
        d_gf_j_r   =   d_gf_j[:, 0:1, :, :]
        d_gf_j_g   =   d_gf_j[:, 1:2, :, :]
        d_gf_j_b   =   d_gf_j[:, 2:3, :, :]
        d_j_i_1_r  =  d_j_i_1[:, 0:1, :, :]
        d_j_i_1_g  =  d_j_i_1[:, 1:2, :, :]
        d_j_i_1_b  =  d_j_i_1[:, 2:3, :, :]
        d_j_i_2_r  =  d_j_i_2[:, 0:1, :, :]
        d_j_i_2_g  =  d_j_i_2[:, 1:2, :, :]
        d_j_i_2_b  =  d_j_i_2[:, 2:3, :, :]
        d_j_i_r    =    d_j_i[:, 0:1, :, :]
        d_j_i_g    =    d_j_i[:, 1:2, :, :]
        d_j_i_b    =    d_j_i[:, 2:3, :, :]
        #
        return {
            "i"         : i,
            "i1"        : i1,
            "i2"        : i2,
            "c1_1"      : c1_1,
            "c1_1_r"    : c1_1_r,
            "c1_1_g"    : c1_1_g,
            "c1_1_b"    : c1_1_b,
            "c1_2"      : c1_2,
            "c2_1"      : c2_1,
            "c2_1_r"    : c2_1_r,
            "c2_1_g"    : c2_1_g,
            "c2_1_b"    : c2_1_b,
            "c2_2"      : c2_2,
            "c_1"       : c_1,
            "c_1_r"     : c_1_r,
            "c_1_g"     : c_1_g,
            "c_1_b"     : c_1_b,
            "c_2"       : c_2,
            "gf1"       : gf1,
            "gf2"       : gf2,
            "gf"        : gf,
            "o"         : o,
            "o1"        : o1,
            "o2"        : o2,
            "d_gf_i_1"  : d_gf_i_1,
            "d_gf_i_2"  : d_gf_i_2,
            "d_gf_i"    : d_gf_i,
            "d_gf_j_1"  : d_gf_j_1,
            "d_gf_j_2"  : d_gf_j_2,
            "d_gf_j"    : d_gf_j,
            "d_j_i_1"   : d_j_i_1,
            "d_j_i_2"   : d_j_i_2,
            "d_j_i"     : d_j_i,
            "d_gf_i_1_r": d_gf_i_1_r,
            "d_gf_i_1_g": d_gf_i_1_g,
            "d_gf_i_1_b": d_gf_i_1_b,
            "d_gf_i_2_r": d_gf_i_2_r,
            "d_gf_i_2_g": d_gf_i_2_g,
            "d_gf_i_2_b": d_gf_i_2_b,
            "d_gf_i_r"  : d_gf_i_r,
            "d_gf_i_g"  : d_gf_i_g,
            "d_gf_i_b"  : d_gf_i_b,
            "d_gf_j_1_r": d_gf_j_1_r,
            "d_gf_j_1_g": d_gf_j_1_g,
            "d_gf_j_1_b": d_gf_j_1_b,
            "d_gf_j_2_r": d_gf_j_2_r,
            "d_gf_j_2_g": d_gf_j_2_g,
            "d_gf_j_2_b": d_gf_j_2_b,
            "d_gf_j_r"  : d_gf_j_r,
            "d_gf_j_g"  : d_gf_j_g,
            "d_gf_j_b"  : d_gf_j_b,
            "d_j_i_1_r" : d_j_i_1_r,
            "d_j_i_1_g" : d_j_i_1_g,
            "d_j_i_1_b" : d_j_i_1_b,
            "d_j_i_2_r" : d_j_i_2_r,
            "d_j_i_2_g" : d_j_i_2_g,
            "d_j_i_2_b" : d_j_i_2_b,
            "d_j_i_r"   : d_j_i_r,
            "d_j_i_g"   : d_j_i_g,
            "d_j_i_b"   : d_j_i_b,
        }
        
    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x    = input
        # Guided Filter
        x_gf = self.gf(x, x)
        # Enhancement
        c1   = self.en(x_gf)
        # Enhancement loop
        if not self.predicting:
            y  = x
            c2 = None
            for i in range(self.num_iters):
                y = y + c1 * (torch.pow(y, 2) - y)
        else:
            y  = x
            c2 = prior.get_guided_brightness_enhancement_map_prior(x_gf, self.gamma, 9)
            for i in range(0, self.num_iters):
                b = y * (1 - c2)
                d = y * c2
                y = b + d + c1 * (torch.pow(d, 2) - d)
        return c1, c2, x_gf, y
    
    @staticmethod
    def pair_downsampler(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c       = input.shape[1]
        filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(input.device)
        filter1 = filter1.repeat(c, 1, 1, 1)
        filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(input.device)
        filter2 = filter2.repeat(c, 1, 1, 1)
        output1 = F.conv2d(input, filter1, stride=2, groups=c)
        output2 = F.conv2d(input, filter2, stride=2, groups=c)
        return output1, output2


@MODELS.register(name="gcenet_a1")
class GCENetA1(GCENet):
    """GCENet-A1 (Guided Curve Estimation Network) model with simple guided filter.
    
    See Also: :class:`GCENet`
    """
    
    def __init__(
        self,
        in_channels : int   = 3,
        num_channels: int   = 32,
        num_iters   : int   = 12,
        radius      : int   = 3,
        eps         : float = 1e-3,
        gamma       : float = 2.8,
        weights     : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            name         = "gcenet_a1",
            in_channels  = in_channels,
            num_channels = num_channels,
            num_iters    = num_iters,
            radius       = radius,
            eps          = eps,
            gamma        = gamma,
            weights      = weights,
            *args, **kwargs
        )
    
    def forward_loss(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Symmetric Loss
        i = input
        c1, c2, gf, o = self.forward(input=i,  *args, **kwargs)
        loss = self.loss(i, c1, o)
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
        c1 = self.en(x)
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
        return c1, c2, y, y_gf


@MODELS.register(name="gcenet_a2")
class GCENetA2(GCENet):
    """GCENet-A2 (Guided Curve Estimation Network) model with simple guided filter.
    
    See Also: :class:`GCENet`
    """
    
    def __init__(
        self,
        in_channels : int   = 3,
        num_channels: int   = 32,
        num_iters   : int   = 12,
        radius      : int   = 3,
        eps         : float = 1e-3,
        gamma       : float = 2.8,
        weights     : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            name         = "gcenet_a2",
            in_channels  = in_channels,
            num_channels = num_channels,
            num_iters    = num_iters,
            radius       = radius,
            eps          = eps,
            gamma        = gamma,
            weights      = weights,
            *args, **kwargs
        )
    
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
        c1 = self.en(x)
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
        return c1, c2, y, y_gf


@MODELS.register(name="gcenet_b1")
class GCENetB1(GCENet):
    """GCENet-B1 (Guided Curve Estimation Network) model with simple guided filter.
    
    See Also: :class:`GCENet`
    """
    
    def __init__(
        self,
        in_channels : int   = 3,
        num_channels: int   = 32,
        num_iters   : int   = 12,
        radius      : int   = 3,
        eps         : float = 1e-3,
        gamma       : float = 2.8,
        weights     : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            name         = "gcenet_b1",
            in_channels  = in_channels,
            num_channels = num_channels,
            num_iters    = num_iters,
            radius       = radius,
            eps          = eps,
            gamma        = gamma,
            weights      = weights,
            *args, **kwargs
        )
    
    def forward_loss(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Symmetric Loss
        i = input
        c1, c2, gf, o = self.forward(input=i,  *args, **kwargs)
        loss = self.loss(i, c1, o)
        return o, loss
    
    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x    = input
        # Guided Filter
        x_gf = self.gf(x, x)
        # Enhancement
        c1   = self.en(x_gf)
        # Enhancement loop
        if not self.predicting:
            y  = x
            c2 = None
            for i in range(self.num_iters):
                y = y + c1 * (torch.pow(y, 2) - y)
        else:
            y  = x
            c2 = prior.get_guided_brightness_enhancement_map_prior(x_gf, self.gamma, 9)
            for i in range(0, self.num_iters):
                b = y * (1 - c2)
                d = y * c2
                y = b + d + c1 * (torch.pow(d, 2) - d)
        return c1, c2, x_gf, y


@MODELS.register(name="gcenet_b2")
class GCENetB2(GCENet):
    """GCENet-B2 (Guided Curve Estimation Network) model with simple guided filter.
    
    See Also: :class:`GCENet`
    """
    
    def __init__(
        self,
        in_channels : int   = 3,
        num_channels: int   = 32,
        num_iters   : int   = 12,
        radius      : int   = 3,
        eps         : float = 1e-3,
        gamma       : float = 2.8,
        weights     : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            name         = "gcenet_b2",
            in_channels  = in_channels,
            num_channels = num_channels,
            num_iters    = num_iters,
            radius       = radius,
            eps          = eps,
            gamma        = gamma,
            weights      = weights,
            *args, **kwargs
        )
    
    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x    = input
        # Guided Filter
        x_gf = self.gf(x, x)
        # Enhancement
        c1   = self.en(x_gf)
        # Enhancement loop
        if not self.predicting:
            y  = x
            c2 = None
            for i in range(self.num_iters):
                y = y + c1 * (torch.pow(y, 2) - y)
        else:
            y  = x
            c2 = prior.get_guided_brightness_enhancement_map_prior(x_gf, self.gamma, 9)
            for i in range(0, self.num_iters):
                b = y * (1 - c2)
                d = y * c2
                y = b + d + c1 * (torch.pow(d, 2) - d)
        return c1, c2, x_gf, y

# endregion


# region Old Model (CVPR 2024, ECCV 2024)

# @MODELS.register(name="gcenet_old")
class GCENetOld(base.LowLightImageEnhancementModel):
    """GCENet (Guided Curve Estimation Network) model.
    
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
            name        = "gcenet_old",
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
    
# endregion
