#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Zero-IG.

This modules implements the paper: "ZERO-IG: Zero-Shot Illumination-Guided Joint
Denoising and Adaptive Enhancement for Low-Light Images".

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
from mon.globals import MODELS, Scheme, Task
from mon.nn import functional as F
from mon.nn.model import StepOutput
from mon.vision.enhance import base

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Loss

def padr_tensor(image: torch.Tensor) -> torch.Tensor:
    return nn.ConstantPad2d(2, 0)(image)


def calculate_local_variance(image: torch.Tensor) -> torch.Tensor:
    b, c, h, w         = image.shape
    noisy_avg          = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)(image)
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
        self.l1_loss     = nn.L1Loss()
        self.l2_loss     = nn.L2Loss()
        self.local_mean  = core.ImageLocalMean(patch_size = 5)
        self.smooth_loss = nn.SmoothLoss()
        self.tv_loss     = nn.TotalVariationLoss()
    
    def forward(self, outputs: dict) -> torch.Tensor:
        eps     = 1e-9
        image   = outputs["image"]
        image   = image + eps
        l_pred1 = outputs["l_pred1"]
        l_pred2 = outputs["l_pred2"]
        l2      = outputs["l2"]
        s2      = outputs["s2"]
        s21     = outputs["s21"]
        s22     = outputs["s22"]
        h2      = outputs["h2"]
        h11     = outputs["h11"]
        h12     = outputs["h12"]
        h13     = outputs["h13"]
        s13     = outputs["s13"]
        h14     = outputs["h14"]
        s14     = outputs["s14"]
        h3      = outputs["h3"]
        s3      = outputs["s3"]
        h3_pred = outputs["h3_pred"]
        h4_pred = outputs["h4_pred"]
        l_pred1_l_pred2_diff           = outputs["l_pred1_l_pred2_diff"]
        h3_denoised1_h3_denoised2_diff = outputs["h3_denoised1_h3_denoised2_diff"]
        h2_blur = outputs["h2_blur"]
        h3_blur = outputs["h3_blur"]
        
        input_Y      = l2.detach()[:, 2, :, :] * 0.299 + l2.detach()[:, 1, :, :] * 0.587 + l2.detach()[:, 0, :, :] * 0.144
        input_Y_mean = torch.mean(input_Y, dim=(1, 2))
        enhancement_factor          = 0.5 / (input_Y_mean + eps)
        enhancement_factor          = enhancement_factor.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        enhancement_factor          = torch.clamp(enhancement_factor, 1, 25)
        adjustment_ratio            = torch.pow(0.7, -enhancement_factor) / enhancement_factor
        adjustment_ratio            = adjustment_ratio.repeat(1, 3, 1, 1)
        normalized_low_light_layer  = l2.detach() / s2
        normalized_low_light_layer  = torch.clamp(normalized_low_light_layer, eps, 0.8)
        enhanced_brightness         = torch.pow(l2.detach() * enhancement_factor, enhancement_factor)
        clamped_enhanced_brightness = torch.clamp(enhanced_brightness * adjustment_ratio, eps, 1)
        clamped_adjusted_low_light  = torch.clamp(l2.detach() *  enhancement_factor, eps, 1)
        loss = 0
        # Enhance_loss
        loss += self.l2_loss(s2, clamped_enhanced_brightness) * 700
        loss += self.l2_loss(normalized_low_light_layer, clamped_adjusted_low_light) * 1000
        loss += self.smooth_loss(l2.detach(), s2) * 5
        loss += self.tv_loss(s2) * 1600
        # Loss_res_1
        l11, l12 = core.pair_downsample(image)
        loss += self.l2_loss(l11, l_pred2) * 1000
        loss += self.l2_loss(l12, l_pred1) * 1000
        denoised1, denoised2 = core.pair_downsample(l2)
        loss += self.l2_loss(l_pred1, denoised1) * 1000
        loss += self.l2_loss(l_pred2, denoised2) * 1000
        # Loss_res_2
        loss += self.l2_loss(h3_pred, torch.cat([h12.detach(), s22.detach()], 1)) * 1000
        loss += self.l2_loss(h4_pred, torch.cat([h11.detach(), s21.detach()], 1)) * 1000
        h3_denoised1, h3_denoised2 = core.pair_downsample(h3)
        loss += self.l2_loss(h3_pred[:, 0:3, :, :], h3_denoised1) * 1000
        loss += self.l2_loss(h4_pred[:, 0:3, :, :], h3_denoised2) * 1000
        # Loss_color
        loss += self.l2_loss(h2_blur.detach(), h3_blur) * 10000
        # Loss_ill
        loss += self.l2_loss(s2.detach(), s3) * 1000
        # Loss_cons
        local_mean1     = self.local_mean(h3_denoised1)
        local_mean2     = self.local_mean(h3_denoised2)
        weighted_diff1  = (1 - h3_denoised1_h3_denoised2_diff) * local_mean1 + h3_denoised1 * h3_denoised1_h3_denoised2_diff
        weighted_diff2  = (1 - h3_denoised1_h3_denoised2_diff) * local_mean2 + h3_denoised1 * h3_denoised1_h3_denoised2_diff
        loss           += self.l2_loss(h3_denoised1,weighted_diff1) * 10000
        loss           += self.l2_loss(h3_denoised2, weighted_diff2) * 10000
        # Loss_Var
        noise_std  = calculate_local_variance(h3 - h2)
        h2_var     = calculate_local_variance(h2)
        loss      += self.l2_loss(h2_var, noise_std) * 1000
        return loss

# endregion


# region Module

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
        self.apply(self.init_weights)
    
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
class ZeroIG(base.ImageEnhancementModel):
    """ZERO-IG: Zero-Shot Illumination-Guided Joint Denoising and Adaptive
    Enhancement for Low-Light Images.
    
    References:
        Code: https://github.com/Doyle59217/ZeroIG
        Paper: https://openaccess.thecvf.com/content/CVPR2024/papers/Shi_ZERO-IG_Zero-Shot_Illumination-Guided_Joint_Denoising_and_Adaptive_Enhancement_for_Low-Light_CVPR_2024_paper.pdf
    """
    
    model_dir: core.Path    = current_dir
    arch     : str          = "zeroig"
    tasks    : list[Task]   = [Task.LLIE]
    schemes  : list[Scheme] = [Scheme.ZERO_REFERENCE]
    zoo      : dict         = {}
    
    def __init__(
        self,
        name          : str = "zeroig_re",
        in_channels   : int = 3,
        num_channels  : int = 64,
        embed_channels: int = 48,
        weights       : Any = None,
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
            in_channels    = self.weights.get("in_channels"   , in_channels)
            num_channels   = self.weights.get("num_channels"  , num_channels)
            embed_channels = self.weights.get("embed_channels", embed_channels)
        self.in_channels    = in_channels or self.in_channels
        self.num_channels   = num_channels
        self.embed_channels = embed_channels
        
        # Construct model
        self.enhance  = Enhance(layers=self.in_channels, channels=self.num_channels)
        self.denoise1 = Denoise1(embed_channels=self.embed_channels)
        self.denoise2 = Denoise2(embed_channels=self.embed_channels)
        self.texture_difference     = nn.TextureDifferenceLoss()
        self.automatic_optimization = False
        
        # Loss
        self.loss = Loss(reduction="mean")
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        
    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.zero_()
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict | None:
        # Forward
        self.assert_datapoint(datapoint)
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        self.assert_outputs(outputs)
        # Loss
        image            = datapoint.get("image")
        outputs["image"] = image
        enhanced         = outputs["enhanced"]
        loss             = self.loss(outputs)
        # Return
        return {
            "enhanced": enhanced,
            "loss"    : loss
        }
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        image = datapoint.get("image")
        eps   = 1e-4
        image = image + eps
        # Predicting
        if self.predicting:
            l2      = image - self.denoise1(image)
            l2      = torch.clamp(l2, eps, 1)
            s2      = self.enhance(l2)
            h2      = image / s2
            h2      = torch.clamp(h2, eps, 1)
            h5_pred = torch.cat([h2, s2], 1).detach() - self.denoise2(torch.cat([h2, s2], 1))
            h5_pred = torch.clamp(h5_pred, eps, 1)
            h3      = h5_pred[:, :3, :, :]
            return {
                "enhanced": h3,
            }
        # Training
        else:
            l11, l12 = core.pair_downsample(image)
            l_pred1  = l11   - self.denoise1(l11)
            l_pred2  = l12   - self.denoise1(l12)
            l2       = image - self.denoise1(image)
            l2       = torch.clamp(l2, eps, 1)
            s2       = self.enhance(l2.detach())
            s21, s22 = core.pair_downsample(s2)
            h2       = image / s2
            h2       = torch.clamp(h2, eps, 1)
            h11      = l11 / s21
            h11      = torch.clamp(h11, eps, 1)
            h12      = l12 / s22
            h12      = torch.clamp(h12, eps, 1)
            h3_pred  = torch.cat([h11, s21], 1).detach() - self.denoise2(torch.cat([h11, s21], 1))
            h3_pred  = torch.clamp(h3_pred, eps, 1)
            h13      = h3_pred[:, :3, :, :]
            s13      = h3_pred[:, 3:, :, :]
            h4_pred  = torch.cat([h12, s22], 1).detach() - self.denoise2(torch.cat([h12, s22], 1))
            h4_pred  = torch.clamp(h4_pred, eps, 1)
            h14      = h4_pred[:, :3, :, :]
            s14      = h4_pred[:, 3:, :, :]
            h5_pred  = torch.cat([h2, s2], 1).detach() - self.denoise2(torch.cat([h2, s2], 1))
            h5_pred  = torch.clamp(h5_pred, eps, 1)
            h3       = h5_pred[:, :3, :, :]
            s3       = h5_pred[:, 3:, :, :]
            l_pred1_l_pred2_diff           = self.texture_difference(l_pred1, l_pred2)
            h3_denoised1, h3_denoised2     = core.pair_downsample(h3)
            h3_denoised1_h3_denoised2_diff = self.texture_difference(h3_denoised1, h3_denoised2)
            h1       = l2 / s2
            h1       = torch.clamp(h1, 0, 1)
            h2_blur  = self.blur(h1)
            h3_blur  = self.blur(h3)
            return {
                "l_pred1" : l_pred1,
                "l_pred2" : l_pred2,
                "l2"      : l2,
                "s2"      : s2,
                "s21"     : s21,
                "s22"     : s22,
                "h2"      : h2,
                "h11"     : h11,
                "h12"     : h12,
                "h13"     : h13,
                "s13"     : s13,
                "h14"     : h14,
                "s14"     : s14,
                "h3"      : h3,
                "s3"      : s3,
                "h3_pred" : h3_pred,
                "h4_pred" : h4_pred,
                "l_pred1_l_pred2_diff"          : l_pred1_l_pred2_diff,
                "h3_denoised1_h3_denoised2_diff": h3_denoised1_h3_denoised2_diff,
                "h2_blur" : h2_blur,
                "h3_blur" : h3_blur,
                "enhanced": h3_blur,
            }
        
    def blur(self, image: torch.Tensor) -> torch.Tensor:
        kernel_size = 21
        padding     = kernel_size // 2
        kernel_var  = self.gaussian_kernel(kernel_size, 1, image.size(1)).to(image.device)
        x_padded    = F.pad(image, (padding, padding, padding, padding), mode="reflect")
        return F.conv2d(x_padded, kernel_var, padding=0, groups=image.size(1))
    
    def gaussian_kernel(
        self,
        kernel_size : int = 21,
        num_signals : int = 3,
        channels    : int = 1
    ) -> torch.Tensor:
        interval   = (2 * num_signals + 1.0) / kernel_size
        x          = torch.linspace(
            -num_signals - interval / 2.0,
             num_signals + interval / 2.0,
             kernel_size + 1,
        ).cuda()
        # kern1d=torch.diff(torch.erf(x/math.sqrt(2.0)))/2.0
        kern1d     = torch.diff(self.gaussian_cdf(x))
        kernel_raw = torch.sqrt(torch.outer(kern1d, kern1d))
        kernel     = kernel_raw / torch.sum(kernel_raw)
        # out_filter=kernel.unsqueeze(2).unsqueeze(3).repeat(1,1,channels,1)
        out_filter = kernel.view(1, 1, kernel_size, kernel_size)
        out_filter = out_filter.repeat(channels, 1, 1, 1)
        return out_filter
    
    def gaussian_cdf(self, image: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1 + torch.erf(image / torch.sqrt(torch.tensor(2.0))))
    
    def training_step(
        self,
        batch    : dict,
        batch_idx: int,
        *args, **kwargs
    ) -> StepOutput | None:
        # Forward
        outputs  = self.forward_loss(datapoint=batch, *args, **kwargs)
        outputs |= self.compute_metrics(
            datapoint = batch,
            outputs   = outputs,
            metrics   = self.train_metrics
        )
        # Log values
        log_values  = {"step": self.current_epoch}
        log_values |= {
            f"train/{k}": v
            for k, v in outputs.items()
            if v is not None and not core.is_image(v)
        }
        self.log_dict(
            dictionary     = log_values,
            prog_bar       = False,
            logger         = True,
            on_step        = False,
            on_epoch       = True,
            sync_dist      = True,
            rank_zero_only = False,
        )
        # Backward
        optimizer = self.optimizers()
        optimizer.zero_grad()
        optimizer.param_groups[0]["capturable"] = True
        loss = outputs.get("loss", None)
        self.manual_backward(loss)
        nn.utils.clip_grad_norm_(self.parameters(), 5)
        optimizer.step()
        # Return
        return loss
    
# endregion
