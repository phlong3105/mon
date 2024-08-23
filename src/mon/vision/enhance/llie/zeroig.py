#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ZeroIG.

This modules implements the algorithm described in the paper "ZERO-IG: Zero-Shot
Illumination-Guided Joint Denoising and Adaptive Enhancement for Low-Light Images".

References:
    Code: https://github.com/Doyle59217/ZeroIG
    Paper: https://openaccess.thecvf.com/content/CVPR2024/papers/Shi_ZERO-IG_Zero-Shot_Illumination-Guided_Joint_Denoising_and_Adaptive_Enhancement_for_Low-Light_CVPR_2024_paper.pdf
"""

from __future__ import annotations

__all__ = [
    "ZeroIG",
]

from typing import Any, Literal

import torch

from mon import core, nn
from mon.globals import MODELS, Scheme
from mon.nn import functional as F
from mon.vision.enhance import base

console = core.console


# region Loss

def padr_tensor(image: torch.Tensor) -> torch.Tensor:
    pad     = 2
    pad_mod = nn.ConstantPad2d(pad, 0)
    return pad_mod(image)


def calculate_local_variance(image: torch.Tensor) -> torch.Tensor:
    b, c, h, w         = image.shape
    avg_pool           = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
    noisy_avg          = avg_pool(image)
    noisy_avg_pad      = padr_tensor(noisy_avg)
    image              = padr_tensor(image)
    unfolded_noisy_avg = noisy_avg_pad.unfold(2, 5, 1).unfold(3, 5, 1)
    unfolded_noisy     = image.unfold(2, 5, 1).unfold(3, 5, 1)
    unfolded_noisy_avg = unfolded_noisy_avg.reshape(unfolded_noisy_avg.shape[0], -1, 5, 5)
    unfolded_noisy     = unfolded_noisy.reshape(unfolded_noisy.shape[0], -1, 5, 5)
    noisy_diff_squared = (unfolded_noisy - unfolded_noisy_avg) ** 2
    noisy_var          = torch.mean(noisy_diff_squared, dim=(2, 3))
    noisy_var          = noisy_var.view(b, c, h, w)
    return noisy_var


class Loss(nn.Loss):
    
    def __init__(
        self,
        reduction: Literal["none", "mean", "sum"] = "mean",
        *args, **kwargs
    ):
        super().__init__(reduction=reduction, *args, **kwargs)
        self.l1_loss            = nn.L1Loss()
        self.l2_loss            = nn.L2Loss()
        self.local_mean         = LocalMean(patch_size=5)
        self.smooth_loss        = nn.SmoothLoss()
        self.texture_difference = nn.TextureDifferenceLoss()
        self.tv_loss            = nn.TotalVariationLoss()
    
    def forward(
        self,
        input,
        L_pred1,
        L_pred2,
        L2,
        s2,
        s21,
        s22,
        H2,
        H11,
        H12,
        H13,
        s13,
        H14,
        s14,
        H3,
        s3,
        H3_pred,
        H4_pred,
        L_pred1_L_pred2_diff,
        H3_denoised1_H3_denoised2_diff,
        H2_blur,
        H3_blur
    ) -> torch.Tensor:
        eps   = 1e-9
        input = input + eps

        input_Y      = L2.detach()[:, 2, :, :] * 0.299 + L2.detach()[:, 1, :, :] * 0.587 + L2.detach()[:, 0, :, :] * 0.144
        input_Y_mean = torch.mean(input_Y, dim=(1, 2))
        enhancement_factor          = 0.5 / (input_Y_mean + eps)
        enhancement_factor          = enhancement_factor.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        enhancement_factor          = torch.clamp(enhancement_factor, 1, 25)
        adjustment_ratio            = torch.pow(0.7, -enhancement_factor) / enhancement_factor
        adjustment_ratio            = adjustment_ratio.repeat(1, 3, 1, 1)
        normalized_low_light_layer  = L2.detach() / s2
        normalized_low_light_layer  = torch.clamp(normalized_low_light_layer, eps, 0.8)
        enhanced_brightness         = torch.pow(L2.detach() * enhancement_factor, enhancement_factor)
        clamped_enhanced_brightness = torch.clamp(enhanced_brightness * adjustment_ratio, eps, 1)
        clamped_adjusted_low_light  = torch.clamp(L2.detach() *  enhancement_factor, eps, 1)
        loss = 0
        # Enhance_loss
        loss += self.l2_loss(s2, clamped_enhanced_brightness) * 700
        loss += self.l2_loss(normalized_low_light_layer, clamped_adjusted_low_light) * 1000
        loss += self.smooth_loss(L2.detach(), s2) * 5
        loss += self.L_TV_loss(s2) * 1600
        # Loss_res_1
        L11, L12 = geometry.pair_downsample(input)
        loss += self.l2_loss(L11, L_pred2) * 1000
        loss += self.l2_loss(L12, L_pred1) * 1000
        denoised1, denoised2 = geometry.pair_downsample(L2)
        loss += self.l2_loss(L_pred1, denoised1) * 1000
        loss += self.l2_loss(L_pred2, denoised2) * 1000
        # Loss_res_2
        loss += self.l2_loss(H3_pred, torch.cat([H12.detach(), s22.detach()], 1)) * 1000
        loss += self.l2_loss(H4_pred, torch.cat([H11.detach(), s21.detach()], 1)) * 1000
        H3_denoised1, H3_denoised2 = geometry.pair_downsample(H3)
        loss += self.l2_loss(H3_pred[:, 0:3, :, :], H3_denoised1) * 1000
        loss += self.l2_loss(H4_pred[:, 0:3, :, :], H3_denoised2) * 1000
        # Loss_color
        loss += self.l2_loss(H2_blur.detach(), H3_blur) * 10000
        # Loss_ill
        loss += self.l2_loss(s2.detach(), s3) * 1000
        # Loss_cons
        local_mean1     = self.local_mean(H3_denoised1)
        local_mean2     = self.local_mean(H3_denoised2)
        weighted_diff1  = (1 - H3_denoised1_H3_denoised2_diff) * local_mean1 + H3_denoised1 * H3_denoised1_H3_denoised2_diff
        weighted_diff2  = (1 - H3_denoised1_H3_denoised2_diff) * local_mean2 + H3_denoised1 * H3_denoised1_H3_denoised2_diff
        loss           += self._l2_loss(H3_denoised1,weighted_diff1) * 10000
        loss           += self._l2_loss(H3_denoised2, weighted_diff2) * 10000
        # Loss_Var
        noise_std  = calculate_local_variance(H3 - H2)
        H2_var     = calculate_local_variance(H2)
        loss      += self.l2_loss(H2_var, noise_std) * 1000
        return loss

# endregion


# region Module

class LocalMean(nn.Module):
    
    def __init__(self, patch_size: int = 5):
        super().__init__()
        self.patch_size = patch_size
        self.padding    = self.patch_size // 2
    
    def forward(self, image):
        image   = F.pad(image, (self.padding, self.padding, self.padding, self.padding), mode="reflect")
        patches = image.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)
        return patches.mean(dim=(4, 5))
    

class Denoise1(nn.Module):
    
    def __init__(self, embed_channels: int = 48):
        super().__init__()
        self.act   = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(3,   embed_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(embed_channels, embed_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(embed_channels, 3,  1)
    
    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x


class Denoise2(nn.Module):
    
    def __init__(self, embed_channels: int = 96):
        super().__init__()
        self.act   = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(6,   embed_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(embed_channels, embed_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(embed_channels, 6,  1)
    
    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x
    

class Enhance(nn.Module):
    
    def __init__(self, layers: int, channels: int):
        super().__init__()
        kernel_size = 3
        dilation    = 1
        padding     = int((kernel_size - 1) / 2) * dilation
        
        self.in_conv = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size, 1, padding),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, 1, padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, 3, 3, 1, 1),
            nn.Sigmoid()
        )
    
    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fea = self.in_conv(x)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)
        fea = torch.clamp(fea, 0.0001, 1)
        return fea
    
# endregion


# region Model

@MODELS.register(name="zeroig_re", arch="zeroig")
class ZeroIG(base.LowLightImageEnhancementModel):
    """ZERO-IG: Zero-Shot Illumination-Guided Joint Denoising and Adaptive
    Enhancement for Low-Light Images.
    
    References:
        https://github.com/Doyle59217/ZeroIG
    """
    
    arch   : str  = "zeroig"
    schemes: list[Scheme] = [Scheme.UNSUPERVISED, Scheme.ZERO_SHOT, Scheme.ZERO_REFERENCE]
    zoo    : dict = {}
    
    def __init__(
        self,
        name        : str = "zeroig_re",
        in_channels : int = 3,
        num_channels: int = 64,
        weights     : Any = None,
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
        self.in_channels  = in_channels or self.in_channels
        self.num_channels = num_channels
        
        # Construct model
        self.enhance  = Enhance(layers=self.in_channels, channels=self.num_channels)
        self.denoise1 = Denoise1(embed_channels=48)
        self.denoise2 = Denoise2(embed_channels=48)
        
        # Loss
        self.loss = Loss(reduction="mean")
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        
    def init_weights(self, m: nn.Module):
        pass
    
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
        adjust,  bam,  depth,  edge,  bright,  dark,  guide,  enhanced  = outputs.values()
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

# endregion
