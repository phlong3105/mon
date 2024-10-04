#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""GCE-Net

This module implements our idea: "Guided Curve Estimation Network for Low-Light
Image Enhancement".
"""

from __future__ import annotations

__all__ = [
    "GCENet",
    "GCENet_ZSN2N",
    "GCENet_Instance",
]

from copy import deepcopy
from typing import Any, Literal

import torch
from fvcore.nn import parameter_count
from torch.nn.common_types import _size_2_t

from mon import core, nn
from mon.globals import MODELS, Scheme, Task
from mon.nn import init
from mon.vision import filtering
from mon.vision.enhance import base

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]

DepthBoundaryAware = nn.BoundaryAwarePrior


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

class LRNet(nn.Module):
    
    def __init__(
        self,
        in_channels : int   = 3,
        mid_channels: int   = 24,
        layers      : int   = 5,
        relu_slope  : float = 0.2,
        norm        : nn.Module = nn.AdaptiveBatchNorm2d,
    ):
        super().__init__()
        net = [
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, 1, bias=False),
            norm(mid_channels),
            nn.LeakyReLU(relu_slope, inplace=True),
        ]
        for l in range(1, layers):
            net += [
                nn.Conv2d(mid_channels, mid_channels, 3, 1, 2**l, 2**l, bias=False),
                norm(mid_channels),
                nn.LeakyReLU(relu_slope, inplace=True)
            ]
        net += [
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, 1, bias=False),
            norm(mid_channels),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, 1, 0, 1)
        ]
        self.net = nn.Sequential(*net)
        self.apply(self.init_weights)
        
    def init_weights(self, m: nn.Module):
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
        self.apply(self.init_weights)
        
    def init_weights(self, m: nn.Module):
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
        norm         : nn.Module = nn.AdaptiveBatchNorm2d,
    ):
        super().__init__()
        self.conv = nn.DSConv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        # self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
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


class DenoiseNet(nn.Module):
    
    def __init__(
        self,
        in_channels : int   = 3,
        num_channels: int   = 48,
        relu_slope  : float = 0.2,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,  num_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(num_channels, in_channels,  1)
        self.act   = nn.LeakyReLU(negative_slope=relu_slope, inplace=True)
            
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        y = self.conv3(x)
        return y

# endregion


# region Model

@MODELS.register(name="gcenet", arch="gcenet")
class GCENet(base.ImageEnhancementModel):
    """Guided Curve Estimation Network for Low-Light Image Enhancement."""
    
    model_dir: core.Path    = current_dir
    arch     : str          = "gcenet"
    tasks    : list[Task]   = [Task.LLIE]
    schemes  : list[Scheme] = [Scheme.ZERO_REFERENCE]
    zoo      : dict         = {}
    
    def __init__(
        self,
        name        : str   = "gcenet",
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
        self.in_channels  = in_channels
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
        image_size: _size_2_t = 512,
        channels  : int       = 3,
        runs      : int       = 1000,
        verbose   : bool      = False,
    ) -> tuple[float, float, float]:
        """Compute the efficiency score of the model, including FLOPs, number
        of parameters, and runtime.
        """
        # Define input tensor
        h, w      = core.get_image_size(image_size)
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
            console.log(f"Time (s)  : {avg_time:.17f}")
        
        return flops, params, avg_time
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        self.assert_datapoint(datapoint)
        image    = datapoint.get("image")
        outputs  = self.forward(datapoint=datapoint,  *args, **kwargs)
        self.assert_outputs(outputs)
        # Symmetric Loss
        adjust   = outputs["adjust"]
        enhanced = outputs["enhanced"]
        loss     = self.loss(image, adjust, enhanced)
        outputs["loss"] = loss
        # Return
        return outputs
        
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        # Prepare input
        self.assert_datapoint(datapoint)
        image = datapoint.get("image")
        depth = datapoint.get("depth")
        # Enhancement
        adjust, edge = self.en(image, depth)
        edge  = edge.detach() if edge is not None else None  # Must call detach() else error
        # Enhancement loop
        if self.bam_gamma in [None, 0.0]:
            enhanced = image
            bam      = None
            bright   = None
            dark     = None
            for i in range(self.num_iters):
                enhanced = enhanced + adjust * (torch.pow(enhanced, 2) - enhanced)
        else:
            enhanced = image
            bam      = self.bam(image)
            bright   = None
            dark     = None
            for i in range(0, self.num_iters):
                bright   = enhanced * (1 - bam)
                dark     = enhanced * bam
                enhanced = bright + dark + adjust * (torch.pow(dark, 2) - dark)
        # Guided Filter
        enhanced = self.gf(image, enhanced)
        # Return
        if self.debug:
            return {
                "adjust"  : adjust,
                "depth"   : depth,
                "edge"    : edge,
                "bam"     : bam,
                "bright"  : bright,
                "dark"    : dark,
                "enhanced": enhanced,
            }
        else:
            return {
                "enhanced": enhanced,
            }


@MODELS.register(name="gcenet_zsn2n", arch="gcenet")
class GCENet_ZSN2N(GCENet):
    
    def __init__(self, name: str = "gcenet_zsn2n", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
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
        enhanced1 = outputs1["enhanced"]
        enhanced2 = outputs2["enhanced"]
        adjust    =  outputs["adjust"]
        enhanced  =  outputs["enhanced"]
        enhanced_1, enhanced_2 = core.pair_downsample(enhanced)
        mse_loss = nn.MSELoss()
        loss_res = 0.5 * (mse_loss(image1,     enhanced2) + mse_loss(image2,     enhanced1))
        loss_con = 0.5 * (mse_loss(enhanced_1, enhanced1) + mse_loss(enhanced_2, enhanced2))
        loss_enh = self.loss(image, adjust, enhanced)
        loss     = 0.5 * (loss_res + loss_con) + 0.5 * loss_enh
        outputs["loss"] = loss
        # Return
        return outputs


@MODELS.register(name="gcenet_instance", arch="gcenet")
class GCENet_Instance(GCENet):
    
    schemes: list[Scheme] = [Scheme.ZERO_REFERENCE, Scheme.INSTANCE]
    
    def __init__(self, name: str = "gcenet_instance", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.initial_state_dict = self.state_dict()
        
    def infer(
        self,
        datapoint    : dict,
        epochs       : int   = 300,
        lr           : float = 0.00005,
        weight_decay : float = 0.00001,
        reset_weights: bool  = True,
        *args, **kwargs
    ) -> dict:
        # Initialize training components
        self.train()
        if reset_weights:
            self.load_state_dict(self.initial_state_dict)
        if isinstance(self.optims, dict):
            optimizer = self.optims.get("optimizer", None)
        else:
            optimizer = nn.Adam(
                self.parameters(),
                lr           = lr,
                betas        = (0.9, 0.999),
                weight_decay = weight_decay,
            )
        
        # Pre-processing
        self.assert_datapoint(datapoint)
        for k, v in datapoint.items():
            if isinstance(v, torch.Tensor):
                datapoint[k] = v.to(self.device)
        
        # Training
        for _ in range(epochs):
            outputs = self.forward_loss(datapoint=datapoint)
            optimizer.zero_grad()
            loss = outputs["loss"]
            loss.backward(retain_graph=True)
            optimizer.step()
            
        # Forward
        self.eval()
        timer = core.Timer()
        timer.tick()
        outputs = self.forward(datapoint=datapoint)
        timer.tock()
        self.assert_outputs(outputs)
    
        # Return
        outputs["time"] = timer.avg_time
        return outputs

# endregion
