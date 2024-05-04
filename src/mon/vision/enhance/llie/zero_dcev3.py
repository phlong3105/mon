#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Zero-DCEv3 models."""

from __future__ import annotations

__all__ = [
    "ZeroDCEv3",
]

from typing import Any, Literal

import torch

from mon import core, nn, proc
from mon.core import _callable, _size_2_t
from mon.globals import MODELS, Scheme
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
        spa_weight    : float = 1.0,
        exp_patch_size: int   = 16,
        exp_mean_val  : float = 0.6,
        exp_weight    : float = 10.0,
        col_weight    : float = 5.0,
        tva_weight    : float = 1600.0,
        reduction     : Literal["none", "mean", "sum"] = "mean",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.spa_weight = spa_weight
        self.exp_weight = exp_weight
        self.col_weight = col_weight
        self.tva_weight = tva_weight
        
        self.loss_spa = nn.SpatialConsistencyLoss(reduction=reduction)
        self.loss_exp = nn.ExposureControlLoss(
            reduction  = reduction,
            patch_size = exp_patch_size,
            mean_val   = exp_mean_val,
        )
        self.loss_col = nn.ColorConstancyLoss(reduction=reduction)
        self.loss_tva = TotalVariationLoss(reduction=reduction)
    
    def forward(
        self,
        input  : torch.Tensor,
        adjust : torch.Tensor,
        enhance: torch.Tensor,
        **_
    ) -> torch.Tensor:
        loss_spa = self.loss_spa(input=enhance, target=input)
        loss_exp = self.loss_exp(input=enhance)
        loss_col = self.loss_col(input=enhance)
        loss_tva = self.loss_tva(input=adjust)
        loss     = (
              self.spa_weight * loss_spa
            + self.exp_weight * loss_exp
            + self.col_weight * loss_col
            + self.tva_weight * loss_tva
        )
        return loss

# endregion


# region Module

class DSConv2d(nn.Module):
    """Depthwise Separable Conv2d."""
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        stride      : _size_2_t       = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t       = 1,
        bias        : bool            = True,
        padding_mode: str             = "zeros",
        device      : Any             = None,
        dtype       : Any             = None,
        is_last     : bool            = False,
    ):
        super().__init__()
        self.dw_conv = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = in_channels,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.lin1    = nn.LearnableInstanceNorm2d(in_channels)
        self.relu1   = nn.PReLU()
        self.pw_conv = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        # self.lin2    = nn.LearnableInstanceNorm2d(out_channels)
        if is_last:
            self.relu2 = nn.PReLU()
        else:
            self.relu2 = nn.Tanh()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.dw_conv(x)
        y = self.lin1(y)
        y = self.relu1(y)
        y = self.pw_conv(y)
        # y = self.lin2(y)
        y = self.relu2(y)
        return y

# endregion


# region Model

@MODELS.register(name="zero_dcev3")
class ZeroDCEv3(base.LowLightImageEnhancementModel):
    """Zero-DCEv3 (Zero-Reference Deep Curve Estimation V3) model.
    
    Args:
        in_channels: The first layer's input channel. Default: ``3`` for RGB image.
        num_channels: The number of input and output channels for subsequent layers. Default: ``32``.
        num_iters: The number of convolutional layers in the model. Default: ``8``.
        scale_factor: Downsampling/upsampling ratio. Defaults: ``1``.
    
    See Also: :class:`base.LowLightImageEnhancementModel`
    """
    
    _scheme: list[Scheme] = [Scheme.UNSUPERVISED, Scheme.ZEROSHOT]
    _zoo   : dict = {}

    def __init__(
        self,
        in_channels : int = 3,
        num_channels: int = 32,
        num_iters   : int = 8,
        weights     : Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name        = "zero_dcev3",
            in_channels = in_channels,
            weights     = weights,
            *args, **kwargs
        )
       
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            in_channels  = self.weights.get("in_channels" , in_channels)
            num_channels = self.weights.get("num_channels", num_channels)
            num_iters    = self.weights.get("num_iters"   , num_iters)
        self.in_channels  = in_channels
        self.num_channels = num_channels
        self.num_iters    = num_iters
        
        # Construct model
        self.pool    = nn.MaxPool2d(2, 2)
        self.up      = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.e_conv1 = DSConv2d(1,          self.num_channels, kernel_size=3, stride=1, padding=1)
        self.e_conv2 = DSConv2d(self.num_channels,     self.num_channels, kernel_size=3, stride=1, padding=1)
        self.e_conv3 = DSConv2d(self.num_channels,     self.num_channels, kernel_size=3, stride=1, padding=1)
        self.e_conv4 = DSConv2d(self.num_channels,     self.num_channels, kernel_size=3, stride=1, padding=1)
        self.e_conv5 = DSConv2d(self.num_channels * 2, self.num_channels, kernel_size=3, stride=1, padding=1)
        self.e_conv6 = DSConv2d(self.num_channels * 2, self.num_channels, kernel_size=3, stride=1, padding=1)
        self.e_conv7 = DSConv2d(self.num_channels * 2, self.out_channels, kernel_size=3, stride=1, padding=1, is_last=True)
        self.trans   = proc.RGBToHVI()
        
        # Loss
        self._loss = Loss()
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self._init_weights)

    def _init_weights(self, m: torch.nn.Module):
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
        x   = input
        hvi = self.trans.rgb_to_hvi(x)
        i   = hvi[:, 2, :, :].unsqueeze(1).to(x.dtype)
        #
        x1  = self.e_conv1(i)
        x2  = self.e_conv2(self.pool(x1))
        x3  = self.e_conv3(self.pool(x2))
        x4  = self.e_conv4(self.pool(x3))
        x5  = self.e_conv5(torch.cat([x3, self.up(x4)], 1))
        x6  = self.e_conv6(torch.cat([x2, self.up(x5)], 1))
        x_r = self.e_conv7(torch.cat([x1, self.up(x6)], 1))
        #
        y = x
        for i in range(0, self.num_iters):
            y = y + x_r * (torch.pow(y, 2) - y)
        #
        return x_r, y

# endregion
