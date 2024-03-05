#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements GCENet models.

./run.sh gcenet none none train 100 sice-zerodce all vision/enhance/llie no last
"""

from __future__ import annotations

__all__ = [
    "GCENet",
]

from typing import Any, Literal

import kornia
import torch

from mon import core, nn, proc
from mon.core import _callable
from mon.globals import MODELS
from mon.nn import functional as F
from mon.vision.enhance.llie import base

console = core.console


# region Loss

class ZeroReferenceLoss(nn.Loss):
    
    def __init__(
        self,
        bri_gamma      : float = 2.8,
        exp_patch_size : int   = 16,
        exp_mean_val   : float = 0.6,
        spa_num_regions: Literal[4, 8, 16, 24] = 8,  # 4
        spa_patch_size : int   = 4,     # 4
        weight_bri     : float = 0,
        weight_col     : float = 5,
        weight_crl     : float = 0.1,     # 20
        weight_edge    : float = 1,
        weight_exp     : float = 10,
        weight_kl      : float = 0.1,     # 5
        weight_spa     : float = 1,
        weight_tvA     : float = 1600,  # 200
        reduction      : str   = "mean",
        verbose        : bool  = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.weight_bri  = weight_bri
        self.weight_col  = weight_col
        self.weight_crl  = weight_crl
        self.weight_edge = weight_edge
        self.weight_exp  = weight_exp
        self.weight_kl   = weight_kl
        self.weight_spa  = weight_spa
        self.weight_tvA  = weight_tvA
        self.verbose     = verbose
        
        self.loss_bri  = nn.BrightnessConstancyLoss(reduction=reduction, gamma=bri_gamma)
        self.loss_col  = nn.ColorConstancyLoss(reduction=reduction)
        self.loss_crl  = nn.ChannelRatioConsistencyLoss(reduction=reduction)
        self.loss_kl   = nn.ChannelConsistencyLoss(reduction=reduction)
        self.loss_edge = nn.EdgeConstancyLoss(reduction=reduction)
        self.loss_exp  = nn.ExposureControlLoss(
            reduction  = reduction,
            patch_size = exp_patch_size,
            mean_val   = exp_mean_val,
        )
        self.loss_spa  = nn.SpatialConsistencyLoss(
            num_regions = spa_num_regions,
            patch_size  = spa_patch_size,
            reduction   = reduction,
        )
        self.loss_tvA  = nn.IlluminationSmoothnessLoss(reduction=reduction)
    
    def __str__(self) -> str:
        return f"zero-reference loss"
    
    def forward(
        self,
        input   : torch.Tensor | list[torch.Tensor],
        target  : list[torch.Tensor],
        previous: torch.Tensor = None,
        **_
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(target, list | tuple):
            if len(target) == 2:
                a       = target[-2]
                enhance = target[-1]
            elif len(target) == 3:
                a       = target[-3]
                g       = target[-2]
                enhance = target[-1]
        else:
            raise TypeError
        loss_bri  = self.loss_bri(input=g, target=input)              if self.weight_bri  > 0 else 0
        loss_col  = self.loss_col(input=enhance)                      if self.weight_col  > 0 else 0
        loss_edge = self.loss_edge(input=enhance, target=input)       if self.weight_edge > 0 else 0
        loss_exp  = self.loss_exp(input=enhance)                      if self.weight_exp  > 0 else 0
        loss_kl   = self.loss_kl(input=enhance, target=input)         if self.weight_kl   > 0 else 0
        loss_spa  = self.loss_spa(input=enhance, target=input)        if self.weight_spa  > 0 else 0
        loss_tvA  = self.loss_tvA(input=a)                            if self.weight_tvA  > 0 else 0
        if previous is not None and (enhance.shape == previous.shape):
            loss_crl = self.loss_crl(input=enhance, target=previous)  if self.weight_crl  > 0 else 0
        else:                                                                             
            loss_crl = self.loss_crl(input=enhance, target=input)     if self.weight_crl  > 0 else 0
        
        loss = (
              self.weight_bri  * loss_bri
            + self.weight_col  * loss_col
            + self.weight_crl  * loss_crl
            + self.weight_edge * loss_edge
            + self.weight_exp  * loss_exp
            + self.weight_tvA  * loss_tvA
            + self.weight_kl   * loss_kl
            + self.weight_spa  * loss_spa
        )
        
        if self.verbose:
            console.log(f"{self.loss_bri.__str__():<30} : {loss_bri}")
            console.log(f"{self.loss_col.__str__():<30} : {loss_col}")
            console.log(f"{self.loss_edge.__str__():<30}: {loss_edge}")
            console.log(f"{self.loss_exp.__str__():<30} : {loss_exp}")
            console.log(f"{self.loss_kl.__str__():<30}  : {loss_kl}")
            console.log(f"{self.loss_spa.__str__():<30} : {loss_spa}")
            console.log(f"{self.loss_tvA.__str__():<30} : {loss_tvA}")
        return loss, enhance
        
# endregion


# region GCENet

@MODELS.register(name="gcenet")
class GCENet(base.LowLightImageEnhancementModel):
    """Guidance Curve Estimation (GCENet) model.
    
    Args:
        channels: The first layer's input channel. Default: ``3`` for RGB image.
        num_channels: Output channels for subsequent layers. Default: ``64``.
        num_iters: The number of convolutional layers in the model.
            Default: ``8``.
        scale_factor: Downsampling/upsampling ratio. Defaults: ``1``.
        gamma: Gamma value for dark channel prior. Default: ``2.8``.
        unsharp_sigma: Unsharp sigma value. Default: ``None``.
        
    See Also: :class:`base.LowLightImageEnhancementModel`
    """
    
    _zoo: dict = {}

    def __init__(
        self,
        channels     : int        = 3,
        num_channels : int        = 32,
        num_iters    : int        = 8,
        scale_factor : int        = 1,
        gamma        : float      = 2.8,
        unsharp_sigma: int | None = None,
        weights      : Any        = None,
        *args, **kwargs
    ):
        super().__init__(
            name     = "gcenet",
            channels = channels,
            weights  = weights,
            *args, **kwargs
        )
        
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            channels      = self.weights.get("channels"     , channels)
            num_channels  = self.weights.get("num_channels" , num_channels)
            num_iters     = self.weights.get("num_iters"    , num_iters)
            scale_factor  = self.weights.get("scale_factor" , scale_factor)
            gamma         = self.weights.get("gamma"        , gamma)
            unsharp_sigma = self.weights.get("unsharp_sigma", unsharp_sigma)
        
        self._channels     = channels
        self.num_channels  = num_channels
        self.num_iters     = num_iters
        self.scale_factor  = scale_factor
        self.gamma         = gamma
        self.unsharp_sigma = unsharp_sigma
        self.previous      = None
        
        # Construct model
        self.conv1    = nn.DSConv2d(self.channels,         self.num_channels, 3, 1, 1, bias=True)
        self.conv2    = nn.DSConv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.conv3    = nn.DSConv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.conv4    = nn.DSConv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.conv5    = nn.DSConv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
        self.conv6    = nn.DSConv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
        self.conv7    = nn.DSConv2d(self.num_channels * 2, self.channels,     3, 1, 1, bias=True)
        self.attn     = nn.Identity()
        self.act      = nn.ReLU(inplace=True)
        self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
        self._loss    = ZeroReferenceLoss()
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self._init_weights)
    
    '''
    def config_model_variant(self):
        """Config the model based on ``self.variant``.
        Mainly used in ablation study.
        """
        # self.gamma         = 2.8
        # self.num_iters     = 9
        # self.unsharp_sigma = 2.5
        self.previous      = None
        out_channels  = 3

        # Variant code: [aa][l][i]
        # i: inference mode
        if self.variant[3] == "0":
            self.gamma    = None
            out_channels = 3
        elif self.variant[3] == "1":
            self.gamma   = self.gamma or 2.8
            out_channels = 3
        elif self.variant[3] == "2":
            self.gamma   = self.gamma or 2.8
            out_channels = 3
        elif self.variant[3] == "3":
            self.gamma   = self.gamma or 2.8
            out_channels = 3
        else:
            raise ValueError

        # Variant code: [aa][l][i]
        # aa: architecture
        if self.variant[0:2] == "00":  # Zero-DCE (baseline)
            self.conv1    = nn.Conv2d(self.channels,         self.num_channels, 3, 1, 1, bias=True)
            self.conv2    = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
            self.conv3    = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
            self.conv4    = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
            self.conv5    = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
            self.conv6    = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
            self.conv7    = nn.Conv2d(self.num_channels * 2, out_channels, 3, 1, 1, bias=True)
            self.attn     = nn.Identity()
            self.act      = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
            self.apply(self.init_weights)
        elif self.variant[0:2] == "01":  # Zero-DCE++ (baseline)
            self.conv1    = nn.DSConv2d(self.channels,         self.num_channels, 3, 1, 1, bias=True)
            self.conv2    = nn.DSConv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
            self.conv3    = nn.DSConv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
            self.conv4    = nn.DSConv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
            self.conv5    = nn.DSConv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
            self.conv6    = nn.DSConv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
            self.conv7    = nn.DSConv2d(self.num_channels * 2, out_channels, 3, 1, 1, bias=True)
            self.attn     = nn.Identity()
            self.act      = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
            self.apply(self.init_weights)
        #
        elif self.variant[0:2] == "10":
            self.conv1    = nn.Conv2d(self.channels,         self.num_channels, 3, 1, 1, bias=True)
            self.conv2    = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
            self.conv3    = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
            self.conv4    = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
            # Curve Enhancement Map (A)
            self.conv5    = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
            self.conv6    = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
            self.conv7    = nn.Conv2d(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True)
            # Guided Brightness Enhancement Map (G)
            self.conv8    = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
            self.conv9    = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
            self.conv10   = nn.Conv2d(self.num_channels * 2, 1, 3, 1, 1, bias=True)
            self.attn     = nn.Identity()
            self.act      = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
            self.apply(self.init_weights)
        elif self.variant[0:2] == "11":
            self.conv1    = nn.Conv2d(self.channels,         self.num_channels, 3, 1, 1, bias=True)
            self.conv2    = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
            self.conv3    = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
            self.conv4    = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
            self.conv5    = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
            # Curve Enhancement Map (A)
            self.conv6    = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
            self.conv7    = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
            self.conv8    = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
            self.conv9    = nn.Conv2d(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True)
            # Guided Brightness Enhancement Map (G)
            self.conv10   = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
            self.conv11   = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
            self.conv12   = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
            self.conv13   = nn.Conv2d(self.num_channels * 2, 1, 3, 1, 1, bias=True)
            self.attn     = nn.Identity()
            self.act      = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
            self.apply(self.init_weights)
        else:
            raise ValueError

        # Variant code: [aa][l][i]
        # l: loss function
        weight_tvA = 1600 if out_channels == 3 else 200
        if self.variant[2] == "0":  # Zero-DCE Loss
            # NOT WORKING: over-exposed artifacts, enhance noises
            self.loss = ZeroReferenceLoss(
                exp_patch_size  = 16,
                exp_mean_val    = 0.6,
                spa_num_regions = 4,
                spa_patch_size  = 4,
                weight_bri      = 0,
                weight_col      = 5,
                weight_crl      = 0,
                weight_edge     = 0,
                weight_exp      = 10,
                weight_kl       = 0,
                weight_spa      = 1,
                weight_tvA      = weight_tvA,
                reduction       = "mean",
            )
        elif self.variant[2] == "1":  # New Loss
            self.loss = ZeroReferenceLoss(
                exp_patch_size  = 16,
                exp_mean_val    = 0.6,
                spa_num_regions = 8,
                spa_patch_size  = 4,
                weight_bri      = 0,
                weight_col      = 5,
                weight_crl      = 0,
                weight_edge     = 1,
                weight_exp      = 10,
                weight_kl       = 0.1,
                weight_spa      = 1,
                weight_tvA      = weight_tvA,
                reduction       = "mean",
            )
        elif self.variant[2] == "2":
            self.loss = ZeroReferenceLoss(
                exp_patch_size  = 16,
                exp_mean_val    = 0.6,
                spa_num_regions = 8,
                spa_patch_size  = 4,
                weight_bri      = 0,
                weight_col      = 5,
                weight_crl      = 0.1,
                weight_edge     = 1,
                weight_exp      = 10,
                weight_kl       = 0.1,
                weight_spa      = 1,
                weight_tvA      = weight_tvA,
                reduction       = "mean",
            )
        elif self.variant[2] == "9":
            self.gamma = self.gamma or 2.5
            self.loss  = ZeroReferenceLoss(
                bri_gamma       = self.gamma,
                exp_patch_size  = 16,   # 16
                exp_mean_val    = 0.6,  # 0.6
                spa_num_regions = 8,    # 8
                spa_patch_size  = 4,    # 4
                weight_bri      = 10,   # 10
                weight_col      = 5,    # 5
                weight_crl      = 0.1,  # 0.1
                weight_edge     = 1,    # 1
                weight_exp      = 10,   # 10
                weight_kl       = 0.1,  # 0.1
                weight_spa      = 1,    # 1
                weight_tvA      = weight_tvA,  # weight_tvA,
                reduction       = "mean",
            )
        else:
            raise ValueError
    '''
    
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
        loss, self.previous = self._loss(input, pred, self.previous)
        return pred[-1], loss
    
    '''
    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass. This is the primary :meth:`forward` function of the
        model. It supports augmented inference. In this function, we perform
        test-time augmentation and pass the transformed input to
        :meth:`forward_once()`.

        Args:
            input: An input of shape :math`[B, C, H, W]`.
            augment: If ``True``, perform test-time augmentation. Default:
                ``False``.
            profile: If ``True``, Measure processing time. Default: ``False``.
            out_index: Return specific layer's output from :param:`out_index`.
                Default: -1 means the last layer.

        Return:
            Predictions.
        """
        if augment:
            # For now just forward the input. Later, we will implement the
            # test-time augmentation.
            if self.variant is not None:
                return self.forward_once_variant(input=input, profile=profile, *args, **kwargs)
            else:
                return self.forward_once(input=input, profile=profile, *args, **kwargs)
        else:
            if self.variant is not None:
                return self.forward_once_variant(input=input, profile=profile, *args, **kwargs)
            else:
                return self.forward_once(input=input, profile=profile, *args, **kwargs)
    '''
    
    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = input

        # Downsampling
        x_down = x
        if self.scale_factor != 1:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode="bilinear")

        f1 = self.act(self.conv1(x_down))
        f2 = self.act(self.conv2(f1))
        f3 = self.act(self.conv3(f2))
        f4 = self.act(self.conv4(f3))
        f4 = self.attn(f4)
        f5 = self.act(self.conv5(torch.cat([f3, f4], dim=1)))
        f6 = self.act(self.conv6(torch.cat([f2, f5], dim=1)))
        a  =   F.tanh(self.conv7(torch.cat([f1, f6], dim=1)))
        
        # Upsampling
        if self.scale_factor != 1:
            a = self.upsample(a)

        # Enhancement
        if not self.predicting:
            y = x
            for _ in range(self.num_iters):
                y = y + a * (torch.pow(y, 2) - y)
        else:
            y = x
            g = proc.get_guided_brightness_enhancement_map_prior(x, self.gamma, 9)
            for _ in range(self.num_iters):
                b = y * (1 - g)
                d = y * g
                y = b + d + a * (torch.pow(d, 2) - d)

        # Unsharp masking
        if self.unsharp_sigma is not None:
            y = kornia.filters.unsharp_mask(y, (3, 3), (self.unsharp_sigma, self.unsharp_sigma))

        return a, y
    
    '''
    def forward_once_variant(
        self,
        input    : torch.Tensor,
        profile  : bool = False,
        out_index: int  = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass once. Implement the logic for a single forward pass. Mainly used for ablation study.

        Args:
            input: An input of shape :math:`[N, C, H, W]`.
            profile: Measure processing time. Default: ``False``.
            out_index: Return specific layer's output from :param:`out_index`.
                Default: ``-1`` means the last layer.

        Return:
            Predictions.
        """
        x = input

        # Downsampling
        x_down = x
        if self.scale_factor != 1:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode="bilinear")

        # Variant code: [aa][l][e]
        if self.variant[0:2] in ["10", "12"]:
            f1  = self.act(self.conv1(x_down))
            f2  = self.act(self.conv2(f1))
            f3  = self.act(self.conv3(f2))
            f4  = self.act(self.conv4(f3))
            f4  = self.attn(f4)
            # Curve Enhancement Map (A)
            f5  = self.act(self.conv5(torch.cat([f3, f4], dim=1)))
            f6  = self.act(self.conv6(torch.cat([f2, f5], dim=1)))
            a   =   F.tanh(self.conv7(torch.cat([f1, f6], dim=1)))
            # Guided Brightness Enhancement Map (GBEM)
            f8  = self.act(self.conv8(torch.cat([f3, f4], dim=1)))
            f9  = self.act(self.conv9(torch.cat([f2, f8], dim=1)))
            g   =  F.tanh(self.conv10(torch.cat([f1, f9], dim=1)))
        elif self.variant[0:2] in ["11"]:
            f1  = self.act(self.conv1(x_down))
            f2  = self.act(self.conv2(f1))
            f3  = self.act(self.conv3(f2))
            f4  = self.act(self.conv4(f3))
            f5  = self.act(self.conv5(f4))
            f5  = self.attn(f5)
            # Curve Enhancement Map (A)
            f6  = self.act(self.conv6(torch.cat([f4, f5], dim=1)))
            f7  = self.act(self.conv7(torch.cat([f3, f6], dim=1)))
            f8  = self.act(self.conv8(torch.cat([f2, f7], dim=1)))
            a   =   F.tanh(self.conv9(torch.cat([f1, f8], dim=1)))
            # Guided Brightness Enhancement Map (GBEM)
            f9  = self.act(self.conv10(torch.cat([f4,  f5], dim=1)))
            f10 = self.act(self.conv11(torch.cat([f3,  f9], dim=1)))
            f11 = self.act(self.conv12(torch.cat([f2, f10], dim=1)))
            g   =   F.tanh(self.conv13(torch.cat([f1, f11], dim=1)))
        elif self.variant[0:2] in ["20"]:
            f1  = self.act(self.conv1(x_down))
            f2  = self.act(self.conv2(f1))
            f3  = self.act(self.conv3(f2))
            f4  = self.act(self.conv4(f3))
            # knot points
            f5  = self.pool(self.pool(self.act(self.conv5(f4))))
            f6  = self.pool(self.pool(self.act(self.conv6(f5))))
            f7  = self.pool(self.pool(self.act(self.conv7(f6))))
            k   = F.adaptive_avg_pool2d(self.conv8(f7), (1, 1))
            # curve parameters
            f9  = self.act(self.conv9(torch.cat([f3, f4],   dim=1)))
            f10 = self.act(self.conv10(torch.cat([f2, f9],  dim=1)))
            a   =   F.tanh(self.conv11(torch.cat([f1, f10], dim=1)))
        else:
            f1  = self.act(self.conv1(x_down))
            f2  = self.act(self.conv2(f1))
            f3  = self.act(self.conv3(f2))
            f4  = self.act(self.conv4(f3))
            f4  = self.attn(f4)
            f5  = self.act(self.conv5(torch.cat([f3, f4], dim=1)))
            f6  = self.act(self.conv6(torch.cat([f2, f5], dim=1)))
            a   =   F.tanh(self.conv7(torch.cat([f1, f6], dim=1)))

        # Upsampling
        if self.scale_factor != 1:
            a = self.upsample(a)

        # Enhancement
        if "1" in self.variant[0:1]:
            if self.out_channels == 3:
                y = x
                for _ in range(self.num_iters):
                    b = y * (1 - g)
                    d = y * g
                    y = b + d + a * (torch.pow(d, 2) - d)
            else:
                y = x
                A = torch.split(a, 3, dim=1)
                for i in range(self.num_iters):
                    b = y * (1 - g)
                    d = y * g
                    y = b + d + A[i] * (torch.pow(d, 2) - d)
        # Piece-wise
        elif "2" in self.variant[0:1]:
            K = torch.split(k, 3, dim=1)
            # A = torch.split(a, 3, dim=1)
            y = x  # K[0]
            for m in range(0, 7):
                k_m1 = K[m + 1]
                k_m  = K[m]
                # a_m  = A[m]
                S    = y
                for n in range(0, 4):
                    S = S + a * (torch.pow(S, 2) - S)
                y = y + (k_m1 - k_m) * S
        # Default
        elif self.variant[3] == "0":
            if self.out_channels == 3:
                y = x
                for _ in range(self.num_iters):
                    y = y + a * (torch.pow(y, 2) - y)
            else:
                y = x
                A = torch.split(a, 3, dim=1)
                for i in range(self.num_iters):
                    y = y + A[i] * (torch.pow(y, 2) - y)
        # Global G
        elif self.variant[3] == "1":
            if self.out_channels == 3:
                y = x
                g = proc.get_guided_brightness_enhancement_map_prior(x, self.gamma, 9)
                for _ in range(self.num_iters):
                    b = y * (1 - g)
                    d = y * g
                    y = b + d + a * (torch.pow(d, 2) - d)
            else:
                y = x
                A = torch.split(a, 3, dim=1)
                g = proc.get_guided_brightness_enhancement_map_prior(x, self.gamma, 9)
                for i in range(self.num_iters):
                    b = y * (1 - g)
                    d = y * g
                    y = b + d + A[i] * (torch.pow(d, 2) - d)
        # Global G Inference Only
        elif self.variant[3] == "2":
            if self.out_channels == 3:
                if self.phase == ModelPhase.TRAINING:
                    y = x
                    for _ in range(self.num_iters):
                        y = y + a * (torch.pow(y, 2) - y)
                else:
                    y = x
                    g = proc.get_guided_brightness_enhancement_map_prior(x, self.gamma, 9)
                    for _ in range(self.num_iters):
                        b = y * (1 - g)
                        d = y * g
                        y = b + d + a * (torch.pow(d, 2) - d)
            else:
                if self.phase == ModelPhase.TRAINING:
                    y = x
                    A = torch.split(a, 3, dim=1)
                    for i in range(self.num_iters):
                        y = y + A[i] * (torch.pow(y, 2) - y)
                else:
                    y = x
                    A = torch.split(a, 3, dim=1)
                    g = proc.get_guided_brightness_enhancement_map_prior(x, self.gamma, 9)
                    for i in range(self.num_iters):
                        b = y * (1 - g)
                        d = y * g
                        y = b + d + A[i] * (torch.pow(d, 2) - d)
        # Iterative G Inference Only
        elif self.variant[3] == "3":
            if self.out_channels == 3:
                if self.phase == ModelPhase.TRAINING:
                    y = x
                    for _ in range(self.num_iters):
                        y = y + a * (torch.pow(y, 2) - y)
                else:
                    y = x
                    for _ in range(self.num_iters):
                        g = proc.get_guided_brightness_enhancement_map_prior(y, self.gamma, 9)
                        b = y * (1 - g)
                        d = y * g
                        y = b + d + a * (torch.pow(d, 2) - d)
            else:
                if self.phase == ModelPhase.TRAINING:
                    y = x
                    A = torch.split(a, 3, dim=1)
                    for i in range(self.num_iters):
                        y = y + A[i] * (torch.pow(y, 2) - y)
                else:
                    y = x
                    A = torch.split(a, 3, dim=1)
                    for i in range(self.num_iters):
                        g = proc.get_guided_brightness_enhancement_map_prior(y, self.gamma, 9)
                        b = y * (1 - g)
                        d = y * g
                        y = b + d + A[i] * (torch.pow(d, 2) - d)

        # Unsharp masking
        if self.unsharp_sigma is not None:
            y = kornia.filters.unsharp_mask(y, (3, 3), (self.unsharp_sigma, self.unsharp_sigma))

        #
        if "1" in self.variant[0:1]:
            return a, g, y
        return a, y
    '''
    
# endregion
