#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements D2CE (Depth to Curve Estimation Network / Deep Depth
Curve Estimation Network) models.
"""

from __future__ import annotations

__all__ = [
    "D2CE",
    "D2CE_01_Baseline",
]

from typing import Any, Literal, Sequence

import torch

from mon import core, nn
from mon.core import _size_2_t
from mon.globals import MODELS, Scheme
from mon.vision import filtering, geometry
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

DepthBoundaryAware = nn.BoundaryAwarePrior


class MixtureOfExperts(nn.Module):
    """Mixture of Experts Layer."""
    
    def __init__(
        self,
        in_channels : list[int],
        out_channels: int,
        dim         : _size_2_t,
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.num_experts  = len(self.in_channels)
        self.dim          = core.parse_hw(dim)
        # Resize & linear
        self.resize  = nn.Upsample(size=self.dim, mode="bilinear", align_corners=False)
        linears      = []
        for in_c in range(self.in_channels):
            linears.append(nn.Linear(in_c, self.out_channels))
        self.linears = linears
        # Conv & softmax
        self.conv    = nn.Linear(self.out_channels * self.num_experts, self.out_channels)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(input) != self.num_experts:
            raise ValueError(f"Expected {self.num_experts} inputs, but got {len(input)}")
        r = []
        for i, inp in enumerate(input):
            r.append(self.linears[i](self.resize(inp)))
        o_s = torch.cat(r, dim=1)
        w   = self.softmax(self.conv(o_s))
        o_w = [r[i] * w[:, i] for i in enumerate(r)]
        o   = torch.sum(o_w, dim=1)


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
        if norm:
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
        eps         : float = 0.05,
        use_depth   : bool  = False,
        use_edge    : bool  = False,
    ):
        super().__init__()
        self.use_depth     = use_depth
        self.use_edge      = use_edge
        in_channels       += 3 if self.use_depth else 0
        in_channels       += 3 if self.use_edge  else 0
        self.in_channels   = in_channels
        self.num_channels  = num_channels
        self.out_channels  = 3
        # Depth Boundary Aware
        self.dba     = DepthBoundaryAware(eps=eps, normalized=False)
        # Encoder
        self.e_conv1 = ConvBlock(self.in_channels,  self.num_channels, norm=norm)
        self.e_conv2 = ConvBlock(self.num_channels, self.num_channels, norm=norm)
        self.e_conv3 = ConvBlock(self.num_channels, self.num_channels, norm=norm)
        self.e_conv4 = ConvBlock(self.num_channels, self.num_channels, norm=norm)
        # Decoder
        self.e_conv5 = ConvBlock(self.num_channels * 2, self.num_channels, norm=norm)
        self.e_conv6 = ConvBlock(self.num_channels * 2, self.num_channels, norm=norm)
        self.e_conv7 = ConvBlock(self.num_channels * 2, self.out_channels, norm=norm, is_last_layer=True)
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

    def forward(self, input: torch.Tensor, depth: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        x   = input
        d   = depth
        e   = self.dba(d)
        if self.use_depth:
            x = torch.cat([x, d], 1)
        if self.use_edge:
            x = torch.cat([x, e], 1)
        x1  = self.e_conv1(x)
        x2  = self.e_conv2(x1)
        x3  = self.e_conv3(x2)
        x4  = self.e_conv4(x3)
        x5  = self.e_conv5(torch.cat([x3, x4], 1))
        x6  = self.e_conv6(torch.cat([x2, x5], 1))
        x_r = self.e_conv7(torch.cat([x1, x6], 1))
        return x_r, e

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
        dba_eps     : float = 0.05,
        gf_radius   : int   = 3,
        gf_eps      : float = 1e-4,
        bam_gamma   : float = 2.6,
        bam_ksize   : int   = 9,
        use_depth   : bool  = True,
        use_edge    : bool  = True,
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
            dba_eps      = self.weights.get("dba_eps"     , dba_eps)
            gf_radius    = self.weights.get("gf_radius"   , gf_radius)
            gf_eps       = self.weights.get("gf_eps"      , gf_eps)
            bam_gamma    = self.weights.get("bam_gamma"   , bam_gamma)
            bam_ksize    = self.weights.get("bam_ksize"   , bam_ksize)
            use_depth    = self.weights.get("use_depth"   , use_depth)
            use_edge     = self.weights.get("use_edge"    , use_edge)
        self.in_channels  = in_channels or self.in_channels
        self.num_channels = num_channels
        self.num_iters    = num_iters
        self.dba_eps      = dba_eps
        self.gf_radius    = gf_radius
        self.gf_eps       = gf_eps
        self.bam_gamma    = bam_gamma
        self.bam_ksize    = bam_ksize
        self.use_depth    = use_depth
        self.use_edge     = use_edge
        
        # Construct model
        self.en  = EnhanceNet(
            in_channels  = self.in_channels,
            num_channels = self.num_channels,
            num_iters    = self.num_iters,
            norm         = None,  # nn.AdaptiveBatchNorm2d,
            eps          = self.dba_eps,
            use_depth    = self.use_depth,
            use_edge     = self.use_edge,
        )
        self.gf  = filtering.GuidedFilter(radius=self.gf_radius, eps=self.gf_eps)
        self.bam = nn.BrightnessAttentionMap(gamma=self.bam_gamma, denoise_ksize=self.bam_ksize)
        
        # Loss
        self.loss = Loss(reduction="mean")
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        
    def init_weights(self, m: nn.Module):
        pass
    
    def assert_datapoint(self, datapoint: dict) -> bool:
        super().assert_datapoint(datapoint)
        assert "depth" in datapoint, "The key ``'depth'`` must be defined in the :param:`datapoint`."
    
    def assert_outputs(self, outputs: dict) -> bool:
        super().assert_outputs(outputs)
        assert "adjust"   in outputs, "The key ``'adjust'`` must be defined in the :param:`outputs`."
        assert "bam"      in outputs, "The key ``'bam'`` must be defined in the :param:`outputs`."
        assert "depth"    in outputs, "The key ``'depth'`` must be defined in the :param:`outputs`."
        assert "edge"     in outputs, "The key ``'edge'`` must be defined in the :param:`outputs`."
        assert "guidance" in outputs, "The key ``'guidance'`` must be defined in the :param:`outputs`."
        assert "enhanced" in outputs, "The key ``'enhanced'`` must be defined in the :param:`outputs`."
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict | None:
        # Forward
        self.assert_datapoint(datapoint)
        image          = datapoint.get("image")
        depth          = datapoint.get("depth")
        image1, image2 = geometry.pair_downsample(image)
        depth1, depth2 = geometry.pair_downsample(depth)
        datapoint1     = datapoint | {"image": image1, "depth": depth1}
        datapoint2     = datapoint | {"image": image2, "depth": depth2}
        outputs1       = self.forward(datapoint=datapoint1, *args, **kwargs)
        outputs2       = self.forward(datapoint=datapoint2, *args, **kwargs)
        outputs        = self.forward(datapoint=datapoint,  *args, **kwargs)
        self.assert_outputs(outputs)
        # Symmetric Loss
        adjust1, bam1, depth1, edge1, bright1, dark1, guide1, enhanced1 = outputs1.values()
        adjust2, bam2, depth2, edge2, bright2, dark2, guide2, enhanced2 = outputs2.values()
        adjust , bam , depth,  edge,  bright,  dark,  guide , enhanced  = outputs.values()
        enhanced_1, enhanced_2 = geometry.pair_downsample(enhanced)
        mse_loss = nn.MSELoss()
        loss_res = 0.5 * (mse_loss(image1,     enhanced2) + mse_loss(image2,     enhanced1))
        loss_con = 0.5 * (mse_loss(enhanced_1, enhanced1) + mse_loss(enhanced_2, enhanced2))
        loss_enh = self.loss(image, adjust, enhanced)
        loss     = 0.5 * (loss_res + loss_con) + 0.5 * loss_enh
        outputs["loss"] = loss
        # Return
        return outputs
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        image = datapoint.get("image")
        depth = datapoint.get("depth")
        # Enhancement
        adjust, edge = self.en(image, depth)
        edge         = edge.detach()   # Must call detach() else error
        # Enhancement loop
        if self.bam_gamma in [None, 0.0]:
            guide  = image
            bam    = None
            bright = None
            dark   = None
            for i in range(self.num_iters):
                guide = guide + adjust * (torch.pow(guide, 2) - guide)
        else:
            guide  = image
            bam    = self.bam(image)
            bright = None
            dark   = None
            for i in range(0, self.num_iters):
                bright = guide * (1 - bam)
                dark   = guide * bam
                guide  = bright + dark + adjust * (torch.pow(dark, 2) - dark)
        # Guided Filter
        enhanced = self.gf(image, guide)
        return {
            "adjust"   : adjust,
            "bam"      : bam,
            "depth"    : depth,
            "edge"     : edge,
            "bright"   : bright,
            "dark"     : dark,
            "guidance" : guide,
            "enhanced" : enhanced,
        }


@MODELS.register(name="d2ce_01_baseline", arch="d2ce")
class D2CE_01_Baseline(D2CE):
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            name      = "d2ce_01_baseline",
            use_depth = False,
            use_edge  = False,
            *args, **kwargs
        )

# endregion
