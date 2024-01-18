#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements GCENet models.

./run.sh gcenet none none train 100 sice-zerodce all vision/enhance/llie no last
"""

from __future__ import annotations

__all__ = [
    "GCENetV2",
]

from typing import Any, Literal

import kornia
import torch

from mon.globals import ModelPhase, MODELS
from mon.vision import core, nn, prior
from mon.vision.enhance.llie import base
from mon.vision.nn import functional as F

math         = core.math
console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region Loss

class ZeroReferenceLoss(nn.Loss):
    
    def __init__(
        self,
        bri_gamma      : float = 2.8,
        exp_patch_size : int   = 16,
        exp_mean_val   : float = 0.6,
        spa_num_regions: Literal[4, 8, 16, 24] = 4,  # 4
        spa_patch_size : int   = 4,     # 4
        weight_bri     : float = 1,
        weight_col     : float = 5,
        weight_crl     : float = 1,     # 20
        weight_edge    : float = 5,
        weight_exp     : float = 10,
        weight_kl      : float = 5,     # 5
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


# region GCENetV2

@MODELS.register(name="gcenetv2")
class GCENetV2(base.LowLightImageEnhancementModel):
    """GCENetV2 (Guidance Curve Estimation) model.
    
    Args:
        channels: The first layer's input channel. Default: ``3`` for RGB image.
        num_channels: Output channels for subsequent layers. Default: ``64``.
        num_iters: The number of convolutional layers in the model.
            Default: ``8``.
        scale_factor: Downsampling/upsampling ratio. Defaults: ``1``.
        p: The probability of applying the Instance Normalization.
            Default: ``0.5``.
        scheme: The scheme of the Instance Normalization. Default: ``'half'``.
        gamma: Gamma value for dark channel prior. Default: ``2.8``.
        unsharp_sigma: Unsharp sigma value. Default: ``None``.
    
    See Also: :class:`mon.vision.enhance.llie.base.LowLightImageEnhancementModel`
    """
    
    zoo = {}
    
    def __init__(
        self,
        channels     : int          = 3,
        num_channels : int          = 32,
        num_iters    : int          = 8,
        scale_factor : int          = 1,
        p            : float | None = 0.5,
        scheme       : Literal[
                         "half",
                         "bipartite",
                         "checkerboard",
                         "random",
                         "adaptive",
                         "attention",
                       ]          = "half",
        gamma        : float | None = 2.8,
        unsharp_sigma: int   | None = None,
        weights      : Any          = None,
        name         : str          = "gcenetv2",
        variant      : str   | None = None,
        *args, **kwargs
    ):
        variant = core.to_int(variant)
        variant = f"{variant:04d}" if isinstance(variant, int) else None
        super().__init__(
            channels = channels,
            weights  = weights,
            name     = name,
            variant  = variant,
            *args, **kwargs
        )
        
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            channels      = self.weights.get("channels"     , channels)
            num_channels  = self.weights.get("num_channels" , num_channels)
            num_iters     = self.weights.get("num_iters"    , num_iters)
            scale_factor  = self.weights.get("scale_factor" , scale_factor)
            p             = self.weights.get("scale_factor" , p)
            scheme        = self.weights.get("scheme"       , scheme)
            gamma         = self.weights.get("gamma"        , gamma)
            unsharp_sigma = self.weights.get("unsharp_sigma", unsharp_sigma)
        
        self.channels      = channels
        self.num_channels  = num_channels
        self.num_iters     = num_iters
        self.scale_factor  = scale_factor
        self.p             = p
        self.scheme        = scheme
        self.gamma         = gamma
        self.unsharp_sigma = unsharp_sigma
        self.previous      = None
        
        # Construct model
        if self.variant is None:  # Default model
            self.conv1    = nn.DSConv2d(self.channels,         self.num_channels, 3, 1, 1, bias=True)
            self.conv2    = nn.DSConv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
            self.conv3    = nn.DSConv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
            self.conv4    = nn.DSConv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
            self.conv5    = nn.DSConv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
            self.conv6    = nn.DSConv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
            self.conv7    = nn.DSConv2d(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True)
            self.norm1    = nn.FractionalInstanceNorm2d(self.num_channels, self.p)
            self.norm2    = nn.FractionalInstanceNorm2d(self.num_channels, self.p)
            self.norm3    = nn.FractionalInstanceNorm2d(self.num_channels, self.p)
            self.norm4    = nn.FractionalInstanceNorm2d(self.num_channels, self.p)
            self.norm5    = nn.FractionalInstanceNorm2d(self.num_channels, self.p)
            self.norm6    = nn.FractionalInstanceNorm2d(self.num_channels, self.p)
            self.norm7    = nn.FractionalInstanceNorm2d(self.channels, self.p)
            self.attn     = nn.Identity()
            self.act      = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
            self.loss     = ZeroReferenceLoss(
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
                weight_tvA      = 1600,
                reduction       = "mean",
            )
        else:
            self.config_model_variant()
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
    
    def config_model_variant(self):
        """Config the model based on ``self.variant``.
        Mainly used in ablation study.
        """
        # self.p             = 0.5
        # self.scheme        = "half"
        self.gamma         = 2.8
        # self.num_iters     = 9
        # self.unsharp_sigma = 2.5
        self.previous      = None
        out_channels       = 3
        
        # Variant code: [aa][p][s]
        # p: probability
        if self.variant[2] == "0":
            self.p = 0.0
        elif self.variant[2] == "1":
            self.p = 0.1
        elif self.variant[2] == "2":
            self.p = 0.2
        elif self.variant[2] == "3":
            self.p = 0.3
        elif self.variant[2] == "4":
            self.p = 0.4
        elif self.variant[2] == "5":
            self.p = 0.5
        elif self.variant[2] == "6":
            self.p = 0.6
        elif self.variant[2] == "7":
            self.p = 0.7
        elif self.variant[2] == "8":
            self.p = 0.8
        elif self.variant[2] == "9":
            self.p = 0.9
        else:
            raise ValueError
        
        # Variant code: [aa][p][s]
        # s: scheme
        if self.variant[3] == "0":
            self.scheme = "half"
        elif self.variant[3] == "1":
            self.scheme = "bipartite"
        elif self.variant[3] == "2":
            self.scheme = "checkerboard"
        elif self.variant[3] == "3":
            self.scheme = "random"
        elif self.variant[3] == "4":
            self.scheme = "adaptive"
        elif self.variant[3] == "5":
            self.scheme = "attentive"
        
        # Variant code: [aa][p][s]
        # aa: architecture
        if self.variant[0:2] == "00":  # Zero-DCE (baseline)
            self.conv1    = nn.Conv2d(self.channels,         self.num_channels, 3, 1, 1, bias=True)
            self.conv2    = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
            self.conv3    = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
            self.conv4    = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
            self.conv5    = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
            self.conv6    = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
            self.conv7    = nn.Conv2d(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True)
            self.norm1    = nn.FractionalInstanceNorm2d(self.num_channels, self.p)
            self.norm2    = nn.FractionalInstanceNorm2d(self.num_channels, self.p)
            self.norm3    = nn.FractionalInstanceNorm2d(self.num_channels, self.p)
            self.norm4    = nn.FractionalInstanceNorm2d(self.num_channels, self.p)
            self.norm5    = nn.FractionalInstanceNorm2d(self.num_channels, self.p)
            self.norm6    = nn.FractionalInstanceNorm2d(self.num_channels, self.p)
            self.norm7    = nn.FractionalInstanceNorm2d(out_channels, self.p)
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
            self.conv7    = nn.DSConv2d(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True)
            self.norm1    = nn.FractionalInstanceNorm2d(self.num_channels, self.p)
            self.norm2    = nn.FractionalInstanceNorm2d(self.num_channels, self.p)
            self.norm3    = nn.FractionalInstanceNorm2d(self.num_channels, self.p)
            self.norm4    = nn.FractionalInstanceNorm2d(self.num_channels, self.p)
            self.norm5    = nn.FractionalInstanceNorm2d(self.num_channels, self.p)
            self.norm6    = nn.FractionalInstanceNorm2d(self.num_channels, self.p)
            self.norm7    = nn.FractionalInstanceNorm2d(out_channels, self.p)
            self.attn     = nn.Identity()
            self.act      = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
            self.apply(self.init_weights)
        else:
            raise ValueError
        
        # l: loss function
        weight_tvA = 1600 if self.out_channels == 3 else 200
        if self.variant[0] in ["0"]:  # Zero-DCE Loss
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
        elif self.variant[0] in ["1"]:
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
    
    def init_weights(self, m: nn.Module):
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
        """Forward pass with loss value. Loss function may need more arguments
        beside the ground-truth and prediction values. For calculating the
        metrics, we only need the final predictions and ground-truth.

        Args:
            input: An input of shape :math:`[N, C, H, W]`.
            target: A ground-truth of shape :math:`[N, C, H, W]`. Default: ``None``.

        Return:
            Predictions and loss value.
        """
        pred = self.forward(input=input, *args, **kwargs)
        loss, self.previous = self.loss(input, pred, self.previous) if self.loss else (None, None)
        return pred[-1], loss
    
    def forward(
        self,
        input    : torch.Tensor,
        augment  : bool = False,
        profile  : bool = False,
        out_index: int  = -1,
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
    
    def forward_once(
        self,
        input    : torch.Tensor,
        profile  : bool = False,
        out_index: int = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass once. Implement the logic for a single forward pass.

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
        if self.out_channels == 3:
            if self.phase == ModelPhase.TRAINING:
                y = x
                for _ in range(self.num_iters):
                    y = y + a * (torch.pow(y, 2) - y)
            else:
                y = x
                g = prior.get_guided_brightness_enhancement_map_prior(x, self.gamma, 9)
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
                g = prior.get_guided_brightness_enhancement_map_prior(x, self.gamma, 9)
                for i in range(self.num_iters):
                    b = y * (1 - g)
                    d = y * g
                    y = b + d + A[i] * (torch.pow(d, 2) - d)
        
        # Unsharp masking
        if self.unsharp_sigma is not None:
            y = kornia.filters.unsharp_mask(y, (3, 3), (self.unsharp_sigma, self.unsharp_sigma))
        
        return a, y
    
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
        f1  = self.act(self.norm1(self.conv1(x_down)))
        f2  = self.act(self.norm2(self.conv2(f1)))
        f3  = self.act(self.norm3(self.conv3(f2)))
        f4  = self.act(self.norm4(self.conv4(f3)))
        f4  = self.attn(f4)
        f5  = self.act(self.norm5(self.conv5(torch.cat([f3, f4], dim=1))))
        f6  = self.act(self.norm6(self.conv6(torch.cat([f2, f5], dim=1))))
        a   =   F.tanh(self.norm7(self.conv7(torch.cat([f1, f6], dim=1))))
        
        # Upsampling
        if self.scale_factor != 1:
            a = self.upsample(a)
        
        # Enhancement
        if self.out_channels == 3:
            if self.phase == ModelPhase.TRAINING:
                y = x
                for _ in range(self.num_iters):
                    y = y + a * (torch.pow(y, 2) - y)
            else:
                y = x
                g = prior.get_guided_brightness_enhancement_map_prior(x, self.gamma, 9)
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
                g = prior.get_guided_brightness_enhancement_map_prior(x, self.gamma, 9)
                for i in range(self.num_iters):
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
    
# endregion
