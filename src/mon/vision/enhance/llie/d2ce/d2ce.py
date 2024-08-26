#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""D2CE.

This module implements Depth to Curve Estimation Network.
"""

from __future__ import annotations

__all__ = [
    "D2CE",
    "D2CE_01_Baseline",
    "D2CE_02_Prediction",
    "D2CE_03_OldLoss",
    "D2CE_04_Prediction_OldLoss",
]

from copy import deepcopy
from typing import Any, Literal, Sequence

import torch
from fvcore.nn import parameter_count
from torch.nn.common_types import _size_2_t

from mon import core, nn
from mon.globals import MODELS, Scheme, Task
from mon.vision import filtering
from mon.vision.enhance import base

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Loss

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
        
        self.loss_col = nn.ColorConstancyLoss(reduction=reduction)
        self.loss_exp = nn.ExposureControlLoss(
            reduction  = reduction,
            patch_size = exp_patch_size,
            mean_val   = exp_mean_val,
        )
        self.loss_spa = nn.SpatialConsistencyLoss(
            num_regions = spa_num_regions,
            patch_size  = spa_patch_size,
            reduction   = reduction,
        )
        self.loss_tva = nn.TotalVariationLoss(reduction=reduction)
    
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
        norm         : nn.Module = nn.AdaptiveBatchNorm2d,
    ):
        super().__init__()
        self.conv = nn.DSConv2d(in_channels, out_channels, 3, 1, 1, bias=True)
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
        norm        : nn.Module = nn.AdaptiveBatchNorm2d,
        eps         : float = 0.05,
        use_depth   : bool  = False,
        use_edge    : bool  = False,
    ):
        super().__init__()
        self.use_depth     = use_depth
        self.use_edge      = use_edge
        in_channels       += 1 if self.use_depth else 0
        in_channels       += 1 if self.use_edge  else 0
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

    def forward(
        self,
        image: torch.Tensor,
        depth: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x    = image
        gray = core.rgb_to_grayscale(image)
        edge = None
        if depth is not None and core.is_color_image(depth):
            depth = core.rgb_to_grayscale(depth)
        if self.use_depth:
            x = torch.cat([x, depth], 1)
        if self.use_edge:
            if depth is not None:
                edge = self.dba(depth)
            else:
                edge = self.dba(gray)
            x = torch.cat([x, edge], 1)

        x1     = self.e_conv1(x)
        x2     = self.e_conv2(x1)
        x3     = self.e_conv3(x2)
        x4     = self.e_conv4(x3)
        x5     = self.e_conv5(torch.cat([x3, x4], 1))
        x6     = self.e_conv6(torch.cat([x2, x5], 1))
        adjust = self.e_conv7(torch.cat([x1, x6], 1))
        return adjust, edge

# endregion


# region Model

@MODELS.register(name="d2ce", arch="d2ce")
class D2CE(base.ImageEnhancementModel):
    """D2CE (Depth to Curve Estimation Network / Deep Depth Curve Estimation
    Network) models.
    """
    
    model_dir: core.Path    = current_dir
    arch     : str          = "d2ce"
    tasks    : list[Task]   = [Task.LLIE]
    schemes  : list[Scheme] = [Scheme.ZERO_REFERENCE]
    zoo      : dict         = {}
    
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
    
    def compute_efficiency_score(
        self,
        imgsz: _size_2_t = 512,
        channels  : int       = 3,
        runs      : int       = 100,
        verbose   : bool      = False,
    ) -> tuple[float, float, float]:
        """Compute the efficiency score of the model, including FLOPs, number
        of parameters, and runtime.
        """
        # Define input tensor
        h, w      = core.parse_hw(imgsz)
        datapoint = {
            "image": torch.rand(1, channels, h, w).to(self.device),
            "depth": torch.rand(1,        1, h, w).to(self.device)
        }
        
        # Get FLOPs and Params
        flops, params = core.custom_profile(deepcopy(self), inputs=datapoint, verbose=verbose)
        # flops         = FlopCountAnalysis(self, datapoint).total() if flops == 0 else flops
        params        = self.params                if hasattr(self, "params") and params == 0 else params
        params        = parameter_count(self)      if hasattr(self, "params")  else params
        params        = sum(list(params.values())) if isinstance(params, dict) else params
        
        # Get time
        timer = core.Timer()
        for i in range(runs):
            timer.tick()
            _ = self(datapoint)
            timer.tock()
        avg_time = timer.avg_time
        
        # Print
        if verbose:
            console.log(f"FLOPs (G) : {flops:.4f}")
            console.log(f"Params (M): {params:.4f}")
            console.log(f"Time (s)  : {avg_time:.4f}")
        
        return flops, params, avg_time
    
    def assert_datapoint(self, datapoint: dict) -> bool:
        super().assert_datapoint(datapoint)
        if "depth" not in datapoint:
            raise ValueError("The key ``'depth'`` must be defined in the "
                             "`datapoint`.")
    
    def assert_outputs(self, outputs: dict) -> bool:
        super().assert_outputs(outputs)
        if "adjust" not in outputs:
            raise ValueError("The key ``'adjust'`` must be defined in the "
                             "`outputs`.")
        if "bam" not in outputs:
            raise ValueError("The key ``'bam'`` must be defined in the "
                             "`outputs`.")
        if "depth" not in outputs:
            raise ValueError("The key ``'depth'`` must be defined in the "
                             "`outputs`.")
        if "edge" not in outputs:
            raise ValueError("The key ``'edge'`` must be defined in the "
                             "`outputs`.")
        if "guidance" not in outputs:
            raise ValueError("The key ``'guidance'`` must be defined in the "
                             "`outputs`.")
        if "enhanced" not in outputs:
            raise ValueError("The key ``'enhanced'`` must be defined in the "
                             "`outputs`.")
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict | None:
        # Forward
        self.assert_datapoint(datapoint)
        image          = datapoint.get("image")
        depth          = datapoint.get("depth")
        image1, image2 = core.pair_downsample(image)
        depth1, depth2 = core.pair_downsample(depth)
        datapoint1     = datapoint | {"image": image1, "depth": depth1}
        datapoint2     = datapoint | {"image": image2, "depth": depth2}
        outputs1       = self.forward(datapoint=datapoint1, *args, **kwargs)
        outputs2       = self.forward(datapoint=datapoint2, *args, **kwargs)
        outputs        = self.forward(datapoint=datapoint,  *args, **kwargs)
        self.assert_outputs(outputs)
        # Symmetric Loss
        adjust1, bam1, depth1, edge1, bright1, dark1, guide1, enhanced1 = outputs1.values()
        adjust2, bam2, depth2, edge2, bright2, dark2, guide2, enhanced2 = outputs2.values()
        adjust,  bam,  depth,  edge,  bright,  dark,  guide,  enhanced  = outputs.values()
        enhanced_1, enhanced_2 = core.pair_downsample(enhanced)
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
        edge  = edge.detach() if edge is not None else None  # Must call detach() else error
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
            "adjust"  : adjust,
            "bam"     : bam,
            "depth"   : depth,
            "edge"    : edge,
            "bright"  : bright,
            "dark"    : dark,
            "guidance": guide,
            "enhanced": enhanced,
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
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict | None:
        # Forward
        self.assert_datapoint(datapoint)
        image   = datapoint.get("image")
        depth   = datapoint.get("depth")
        outputs = self.forward(datapoint=datapoint,  *args, **kwargs)
        self.assert_outputs(outputs)
        # Symmetric Loss
        adjust, bam, depth, edge, bright, dark, guide, enhanced = outputs.values()
        loss = self.loss(image, adjust, enhanced)
        outputs["loss"] = loss
        # Return
        return outputs
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        image  = datapoint.get("image")
        depth  = datapoint.get("depth")
        # Enhancement
        adjust, edge = self.en(image, depth)
        edge   = edge.detach() if edge is not None else None  # Must call detach() else error
        # Enhancement loop
        guide  = image
        bam    = None
        bright = None
        dark   = None
        for i in range(self.num_iters):
            guide = guide + adjust * (torch.pow(guide, 2) - guide)
        # Guided Filter
        enhanced = guide
        return {
            "adjust"  : adjust,
            "bam"     : bam,
            "depth"   : depth,
            "edge"    : edge,
            "bright"  : bright,
            "dark"    : dark,
            "guidance": guide,
            "enhanced": enhanced,
        }
    

@MODELS.register(name="d2ce_02_prediction", arch="d2ce")
class D2CE_02_Prediction(D2CE):
    
    def __init__(self, *args, **kwargs):
        super().__init__(name="d2ce_02_prediction", *args, **kwargs)
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        image = datapoint.get("image")
        depth = datapoint.get("depth")
        # Enhancement
        adjust, edge = self.en(image, depth)
        edge  = edge.detach() if edge is not None else None  # Must call detach() else error
        # Enhancement loop
        if not self.predicting:
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
            "adjust"  : adjust,
            "bam"     : bam,
            "depth"   : depth,
            "edge"    : edge,
            "bright"  : bright,
            "dark"    : dark,
            "guidance": guide,
            "enhanced": enhanced,
        }


@MODELS.register(name="d2ce_03_oldloss", arch="d2ce")
class D2CE_03_OldLoss(D2CE):
    
    def __init__(self, *args, **kwargs):
        super().__init__(name="d2ce_03_oldloss", *args, **kwargs)
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict | None:
        # Forward
        self.assert_datapoint(datapoint)
        image   = datapoint.get("image")
        depth   = datapoint.get("depth")
        outputs = self.forward(datapoint=datapoint,  *args, **kwargs)
        self.assert_outputs(outputs)
        # Symmetric Loss
        adjust, bam, depth, edge, bright, dark, guide, enhanced = outputs.values()
        loss = self.loss(image, adjust, enhanced)
        outputs["loss"] = loss
        # Return
        return outputs


@MODELS.register(name="d2ce_04_prediction_oldloss", arch="d2ce")
class D2CE_04_Prediction_OldLoss(D2CE):
    
    def __init__(self, *args, **kwargs):
        super().__init__(name="d2ce_04_prediction_oldloss", *args, **kwargs)
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict | None:
        # Forward
        self.assert_datapoint(datapoint)
        image   = datapoint.get("image")
        depth   = datapoint.get("depth")
        outputs = self.forward(datapoint=datapoint,  *args, **kwargs)
        self.assert_outputs(outputs)
        # Symmetric Loss
        adjust, bam, depth, edge, bright, dark, guide, enhanced = outputs.values()
        loss = self.loss(image, adjust, enhanced)
        outputs["loss"] = loss
        # Return
        return outputs
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        image  = datapoint.get("image")
        depth  = datapoint.get("depth")
        # Enhancement
        adjust, edge = self.en(image, depth)
        edge   = edge.detach() if edge is not None else None  # Must call detach() else error
        # Enhancement loop
        guide  = image
        bam    = None
        bright = None
        dark   = None
        for i in range(self.num_iters):
            guide = guide + adjust * (torch.pow(guide, 2) - guide)
        # Guided Filter
        enhanced = guide
        return {
            "adjust"  : adjust,
            "bam"     : bam,
            "depth"   : depth,
            "edge"    : edge,
            "bright"  : bright,
            "dark"    : dark,
            "guidance": guide,
            "enhanced": enhanced,
        }
    
# endregion
