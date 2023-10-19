#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Zero-DCEv2 models."""

from __future__ import annotations

__all__ = [
    "ZeroDCEv2",
    "ZeroReferenceLoss",
]

from typing import Any, Callable

import kornia
import torch

from mon.globals import LAYERS, MODELS
from mon.nn.typing import _size_2_t
from mon.vision import core, nn, prior
from mon.vision.enhance.llie import base
from mon.vision.nn import functional as F

math         = core.math
console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region Module

@LAYERS.register()
class SimAMConv2d(nn.ConvLayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        kernel_size     : _size_2_t,
        stride          : _size_2_t       = 1,
        padding         : _size_2_t | str = 0,
        dilation        : _size_2_t       = 1,
        groups          : int             = 1,
        bias            : bool            = True,
        padding_mode    : str             = "zeros",
        device          : Any             = None,
        dtype           : Any             = None,
        norm            : Callable        = None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.norm  = norm(num_features=out_channels) if norm is not None else None
        self.simam = nn.SimAM()
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = x
        y = self.conv(y)
        y = self.simam(y)
        if self.norm is not None:
            y = self.norm(y)
        return y
    
# endregion


# region Loss

class ZeroReferenceLoss(nn.Loss):
    
    def __init__(
        self,
        exp_patch_size: int   = 16,
        exp_mean_val  : float = 0.6,
        weight_col    : float = 5,
        weight_crl    : float = 1,     # 20
        weight_edge   : float = 5,
        weight_exp    : float = 10,
        weight_kl     : float = 5,     # 5
        weight_spa    : float = 1,
        weight_tvA    : float = 1600,  # 200
        reduction     : str   = "mean",
        verbose       : bool  = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.weight_col  = weight_col
        self.weight_crl  = weight_crl
        self.weight_edge = weight_edge
        self.weight_exp  = weight_exp
        self.weight_kl   = weight_kl
        self.weight_spa  = weight_spa
        self.weight_tvA  = weight_tvA
        self.verbose     = verbose
        
        self.loss_col  = nn.ColorConstancyLoss(reduction=reduction)
        self.loss_crl  = nn.ChannelRatioConsistencyLoss(reduction=reduction)
        self.loss_kl   = nn.ChannelConsistencyLoss(reduction=reduction)
        self.loss_edge = nn.EdgeConstancyLoss(reduction=reduction)
        self.loss_exp  = nn.ExposureControlLoss(
            reduction  = reduction,
            patch_size = exp_patch_size,
            mean_val   = exp_mean_val,
        )
        self.loss_spa = nn.SpatialConsistencyLoss(reduction=reduction)
        self.loss_tvA = nn.IlluminationSmoothnessLoss(reduction=reduction)
    
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
            a       = target[-2]
            enhance = target[-1]
        else:
            raise TypeError
        loss_col  = self.loss_col(input=enhance)
        loss_edge = self.loss_edge(input=enhance, target=input)
        loss_exp  = self.loss_exp(input=enhance)
        loss_kl   = self.loss_kl(input=enhance, target=input)
        loss_spa  = self.loss_spa(input=enhance, target=input)
        loss_tvA  = self.loss_tvA(input=a)
        if previous is not None and (enhance.shape == previous.shape):
            loss_crl = self.loss_crl(input=enhance, target=previous)
        else:
            loss_crl = self.loss_crl(input=enhance, target=input)
        # loss_crl = None
        
        if loss_crl is not None:
            loss = (
                  self.weight_col  * loss_col
                + self.weight_crl  * loss_crl
                + self.weight_edge * loss_edge
                + self.weight_exp  * loss_exp
                + self.weight_tvA  * loss_tvA
                + self.weight_kl   * loss_kl
                + self.weight_spa  * loss_spa
            )
        else:
            loss = (
                  self.weight_col  * loss_col
                + self.weight_edge * loss_edge
                + self.weight_exp  * loss_exp
                + self.weight_tvA  * loss_tvA
                + self.weight_kl   * loss_kl
                + self.weight_spa  * loss_spa
            )
        if self.verbose:
            console.log(f"{self.loss_col.__str__():<30} : {loss_col}")
            console.log(f"{self.loss_edge.__str__():<30}: {loss_edge}")
            console.log(f"{self.loss_exp.__str__():<30} : {loss_exp}")
            console.log(f"{self.loss_kl.__str__():<30}  : {loss_kl}")
            console.log(f"{self.loss_spa.__str__():<30} : {loss_spa}")
            console.log(f"{self.loss_tvA.__str__():<30} : {loss_tvA}")
        return loss, enhance
        
# endregion


# region Model

@MODELS.register(name="zerodcev2")
class ZeroDCEv2(base.LowLightImageEnhancementModel):
    """Zero-DCEv2 (Zero-Reference Deep Curve Estimation V2) model.
    
    See Also: :class:`mon.vision.enhance.llie.base.LowLightImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {}
    map_weights = {}
    
    def __init__(
        self,
        config       : Any                = None,
        loss         : Any                = ZeroReferenceLoss(),
        variant      :         str | None = "00000",
        num_channels : int   | str        = 32,
        scale_factor : float | str        = 1.0,
        gamma        : float | str | None = None,
        num_iters    : int   | str        = 8,
        ratio        : float | str        = 0.5,
        unsharp_sigma: int   | str | None = None,
        *args, **kwargs
    ):
        super().__init__(
            config = config,
            loss   = loss,
            *args, **kwargs
        )
        # Variant code: [ma][a][l][e]
        # ma: model architecture
        # a : activation
        # l : loss function
        # e : enhancement mode
        # self.variant       = f"{int(variant):04d}" if isinstance(variant, int)         or (isinstance(variant, str)       and variant.isdigit())       else "0000"
        self.variant       = variant or "00000"
        self.num_channels  = int(num_channels)     if isinstance(num_channels, int)    or (isinstance(num_channels, str)  and num_channels.isdigit())  else 32
        self.scale_factor  = float(scale_factor)   if isinstance(scale_factor, float)  or (isinstance(scale_factor, str)  and scale_factor.isdigit())  else 1.0
        self.gamma         = float(gamma)          if isinstance(gamma, float)         or (isinstance(gamma, str)         and gamma.isdigit())         else None
        self.num_iters     = int(num_iters)        if isinstance(num_iters, int)       or (isinstance(num_iters, str)     and num_iters.isdigit())     else 8
        self.ratio         = float(ratio)          if isinstance(ratio, float)         or (isinstance(ratio, str)         and ratio.isdigit())         else 0.5
        self.unsharp_sigma = float(unsharp_sigma)  if isinstance(unsharp_sigma, float) or (isinstance(unsharp_sigma, str) and unsharp_sigma.isdigit()) else None
        
        # self.gamma         = 2.5
        # self.unsharp_sigma = 2.5
        self.previous      = None
        self.out_channels  = 3
        self.act           = nn.LeakyReLU(inplace=True)
        self.upsample      = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        
        # Variant code: [ma][a][l][e]
        # e: enhancement mode
        if self.variant[4] == "0":
            self.gamma        = None
            self.out_channels = 3
            weight_tvA        = 1600
        elif self.variant[4] == "1":
            self.gamma        = 2.5
            self.out_channels = 3
            weight_tvA        = 1600
        elif self.variant[4] == "2":  # NOT IMPROVING
            self.gamma        = None
            self.out_channels = 3 * self.num_iters
            weight_tvA        = 200
        elif self.variant[4] == "3":  # NOT IMPROVING
            self.gamma        = 2.5
            self.out_channels = 3 * self.num_iters
            weight_tvA        = 200
        else:
            raise ValueError
         
        # Variant code: [ma][a][l][e]
        # ma: model architecture
        if self.variant[0:2] == "00":  # Zero-DCE (baseline)
            self.conv1 = nn.Conv2d(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv4 = nn.Conv2d(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv5 = nn.Conv2d(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv6 = nn.Conv2d(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv7 = nn.Conv2d(in_channels=self.num_channels * 2, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
            self.apply(self.init_weights)
        elif self.variant[0:2] == "01":  # Zero-DCE++ (baseline)
            self.conv1 = nn.DSConv2d(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv2 = nn.DSConv2d(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv3 = nn.DSConv2d(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv4 = nn.DSConv2d(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv5 = nn.DSConv2d(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv6 = nn.DSConv2d(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv7 = nn.DSConv2d(in_channels=self.num_channels * 2, out_channels=self.out_channels, kernel_size=3, dw_stride=1, dw_padding=1)
            self.apply(self.init_weights)
        elif self.variant[0:2] == "02":
            self.conv1 = nn.ABSConv2dS(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv2 = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv3 = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv4 = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv5 = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv6 = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv7 = nn.Conv2d(    in_channels=self.num_channels * 2, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
            self.apply(self.init_weights)
        elif self.variant[0:2] == "03":
            self.conv1 = nn.ABSConv2dU(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm=nn.HalfInstanceNorm2d)
            self.conv2 = nn.ABSConv2dU(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm=nn.HalfInstanceNorm2d)
            self.conv3 = nn.ABSConv2dU(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm=nn.HalfInstanceNorm2d)
            self.conv4 = nn.ABSConv2dU(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm=nn.HalfInstanceNorm2d)
            self.conv5 = nn.ABSConv2dU(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm=nn.HalfInstanceNorm2d)
            self.conv6 = nn.ABSConv2dU(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm=nn.HalfInstanceNorm2d)
            self.conv7 = nn.Conv2d(    in_channels=self.num_channels * 2, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
            self.apply(self.init_weights)
        elif self.variant[0:2] == "04":
            self.conv1 = SimAMConv2d(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv2 = SimAMConv2d(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv3 = SimAMConv2d(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv4 = SimAMConv2d(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv5 = SimAMConv2d(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv6 = SimAMConv2d(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv7 = SimAMConv2d(in_channels=self.num_channels * 2, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
            self.apply(self.init_weights)
        elif self.variant[0:2] == "05":
            self.conv1 = SimAMConv2d(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, norm=nn.HalfInstanceNorm2d)
            self.conv2 = SimAMConv2d(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, norm=nn.HalfInstanceNorm2d)
            self.conv3 = SimAMConv2d(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, norm=nn.HalfInstanceNorm2d)
            self.conv4 = SimAMConv2d(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, norm=nn.HalfInstanceNorm2d)
            self.conv5 = SimAMConv2d(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, norm=nn.HalfInstanceNorm2d)
            self.conv6 = SimAMConv2d(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, norm=nn.HalfInstanceNorm2d)
            self.conv7 = SimAMConv2d(in_channels=self.num_channels * 2, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
            self.apply(self.init_weights)
        elif self.variant[0:2] == "06":
            self.conv1 = nn.Conv2d(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv4 = nn.Conv2d(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv5 = nn.Conv2d(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv6 = nn.Conv2d(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv7 = nn.Conv2d(in_channels=self.num_channels * 2, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
            self.attn  = nn.SimAM()
            self.norm  = nn.HalfInstanceNorm2d(num_features=self.num_channels)
            self.apply(self.init_weights)
        
        # Variant code: [ma][a][l][e]
        # a: activation
        if self.variant[2] == "0":
            self.act = nn.LeakyReLU(inplace=True)
        elif self.variant[2] == "1":
            self.act = nn.Sigmoid()
        elif self.variant[2] == "2":
            self.act = nn.Tanh()
        elif self.variant[2] == "3":
            self.act = nn.ReLU(inplace=True)
        elif self.variant[2] == "4":
            self.act = nn.PReLU()
        elif self.variant[2] == "5":
            self.act = nn.ELU()
        elif self.variant[2] == "6":
            self.act = nn.SELU(inplace=True)
        else:
            raise ValueError
        
        # Variant code: [ma][a][l][g]
        # l: loss function
        if self.variant[3] == "0":  # Zero-DCE++ Loss
            self.loss = ZeroReferenceLoss(
                exp_patch_size = 16,
                exp_mean_val   = 0.6,
                weight_col     = 5,
                weight_crl     = 0,
                weight_edge    = 0,
                weight_exp     = 10,
                weight_kl      = 0,
                weight_spa     = 1,
                weight_tvA     = weight_tvA,
                reduction      = "mean",
            )
        elif self.variant[3] == "1":  # Zero-DCE++ Loss + L_edge
            self.loss = ZeroReferenceLoss(
                exp_patch_size = 16,
                exp_mean_val   = 0.6,
                weight_col     = 5,
                weight_crl     = 0,
                weight_edge    = 10,
                weight_exp     = 10,
                weight_kl      = 0,
                weight_spa     = 1,
                weight_tvA     = weight_tvA,
                reduction      = "mean",
            )
        elif self.variant[3] == "2":  # Zero-DCE++ Loss + 5 * L_edge + 0.2 * L_kl
            self.loss = ZeroReferenceLoss(
                exp_patch_size = 16,
                exp_mean_val   = 0.6,
                weight_col     = 5,
                weight_crl     = 0,
                weight_edge    = 1,
                weight_exp     = 10,
                weight_kl      = 0.2,
                weight_spa     = 1,
                weight_tvA     = weight_tvA,
                reduction      = "mean",
            )
        else:
            raise ValueError
        
    @property
    def config_dir(self) -> core.Path:
        return core.Path(__file__).absolute().parent / "config"
    
    def init_weights(self, m: nn.Module):
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
        """Forward pass with loss value. Loss function may need more arguments
        beside the ground-truth and prediction values. For calculating the
        metrics, we only need the final predictions and ground-truth.

        Args:
            input: An input of shape :math:`[N, C, H, W]`.
            target: A ground-truth of shape :math:`[N, C, H, W]`. Default: ``None``.
            
        Return:
            Predictions and loss value.
        """
        pred  = self.forward(input=input, *args, **kwargs)
        loss, self.previous = self.loss(input, pred) if self.loss else None
        loss += self.regularization_loss(alpha=0.1)
        return pred[-1], loss
    
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
        if self.scale_factor == 1:
            x_down = x
        else:
            scale_factor = 1 / self.scale_factor
            x_down       = F.interpolate(x, scale_factor=scale_factor, mode="bilinear")
        
        # [ma][a][l][g]
        if self.variant[0:2] == "07":
            if self.variant[4] in ["0", "2"]:
                f1 = self.act(self.norm(self.conv1(x_down)))
                f2 = self.act(self.norm(self.conv2(f1)))
                f3 = self.act(self.norm(self.conv3(f2)))
                f4 = self.act(self.norm(self.conv4(f3)))
                f5 = self.act(self.norm(self.conv5(torch.cat([self.attn(f3), f4], dim=1))))
                f6 = self.act(self.norm(self.conv6(torch.cat([self.attn(f2), f5], dim=1))))
                f7 =   F.tanh(self.norm(self.conv7(torch.cat([self.attn(f1), f6], dim=1))))
            elif self.variant[4] in ["1", "3"]:
                attn = prior.get_guided_brightness_enhancement_map_prior(x_down, gamma=self.gamma)
                f1 = attn * self.act(self.norm(self.conv1(x_down)))
                f2 = attn * self.act(self.norm(self.conv2(f1)))
                f3 = attn * self.act(self.norm(self.conv3(f2)))
                f4 = attn * self.act(self.norm(self.conv4(f3)))
                f5 = attn * self.act(self.norm(self.conv5(torch.cat([self.attn(f3), f4], dim=1))))
                f6 = attn * self.act(self.norm(self.conv6(torch.cat([self.attn(f2), f5], dim=1))))
                f7 =          F.tanh(self.norm(self.conv7(torch.cat([self.attn(f1), f6], dim=1))))
        elif self.variant[4] in ["0", "2"]:
            f1 = self.act(self.conv1(x_down))
            f2 = self.act(self.conv2(f1))
            f3 = self.act(self.conv3(f2))
            f4 = self.act(self.conv4(f3))
            f5 = self.act(self.conv5(torch.cat([f3, f4], dim=1)))
            f6 = self.act(self.conv6(torch.cat([f2, f5], dim=1)))
            f7 =   F.tanh(self.conv7(torch.cat([f1, f6], dim=1)))
        elif self.variant[4] in ["1", "3"]:
            attn = prior.get_guided_brightness_enhancement_map_prior(x_down, gamma=self.gamma)
            f1   = attn * self.act(self.conv1(x_down))
            f2   = attn * self.act(self.conv2(f1))
            f3   = attn * self.act(self.conv3(f2))
            f4   = attn * self.act(self.conv4(f3))
            f5   = attn * self.act(self.conv5(torch.cat([f3, f4], dim=1)))
            f6   = attn * self.act(self.conv6(torch.cat([f2, f5], dim=1)))
            f7   =          F.tanh(self.conv7(torch.cat([f1, f6], dim=1)))
        else:
            raise ValueError
        
        # Upsampling
        if self.scale_factor == 1:
            f7 = f7
        else:
            f7 = self.upsample(f7)
        
        # Enhancement
        if self.variant[4] in ["0", "1"]:
            a = f7
            y = x
            for _ in range(self.num_iters):
                y = y + a * (torch.pow(y, 2) - y)
        elif self.variant[4] in ["2", "3"]:
            a = torch.split(f7, 3, dim=1)
            y = x
            for i in range(self.num_iters):
                y = y + a[i] * (torch.pow(y, 2) - y)
        else:
            raise ValueError
        
        # Unsharp masking
        if self.unsharp_sigma is not None:
            y = kornia.filters.unsharp_mask(y, (3, 3), (self.unsharp_sigma, self.unsharp_sigma))
            # y = kornia.enhance.equalize(y)
            
        return f7, y
    
    def regularization_loss(self, alpha: float = 0.1):
        loss = 0.0
        for sub_module in [
            self.conv1, self.conv2, self.conv3, self.conv4,
            self.conv5, self.conv6, self.conv7
        ]:
            if hasattr(sub_module, "regularization_loss"):
                loss += sub_module.regularization_loss()
        return alpha * loss

# endregion
