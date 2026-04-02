#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Loss Functions.

This module provides custom loss functions for Zero-IG.
"""

from __future__ import annotations

__all__ = [
    "LossFunction",
    "TextureDifference",
]

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from mon.nn import Loss
from mon.ops import pair_downsample
from .utils import calculate_local_variance, LocalMean


# ==============================================================================
# region LOSS FUNCTIONS
# ==============================================================================

class TextureDifference(Loss):

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        patch_size: int = 5,
        constant_C: float = 1e-5,
        threshold: float = 0.975,
        reduction: str = "mean"
    ):
        """Initialize a new instance.

        Args:
            reduction (str, optional): Reduction method to apply to the loss.
                One of: ["mean", "sum", "none"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.patch_size = patch_size
        self.constant_C = constant_C
        self.threshold = threshold

    # --- Callable & Context Manager ---
    def forward(self, image1: Tensor, image2: Tensor) -> Tensor:
        # Convert RGB images to grayscale
        image1 = self.rgb_to_gray(image1)
        image2 = self.rgb_to_gray(image2)

        stddev1 = self.local_stddev(image1)
        stddev2 = self.local_stddev(image2)
        numerator = 2 * stddev1 * stddev2
        denominator = stddev1 ** 2 + stddev2 ** 2 + self.constant_C
        diff = numerator / denominator

        # Apply threshold to diff tensor
        binary_diff = torch.where(
            diff > self.threshold,
            torch.tensor(1.0, device=diff.device),
            torch.tensor(0.0, device=diff.device)
        )
        return self.reduce(binary_diff)

    # --- Utilities ---
    def local_stddev(self, image: Tensor) -> Tensor:
        padding = self.patch_size // 2
        image = F.pad(image, (padding, padding, padding, padding), mode="reflect")
        patches = image.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)
        mean = patches.mean(dim=(4, 5), keepdim=True)
        squared_diff = (patches - mean) ** 2
        local_variance = squared_diff.mean(dim=(4, 5))
        local_stddev = torch.sqrt(local_variance+1e-9)
        return local_stddev

    def rgb_to_gray(self, image: Tensor) -> Tensor:
        # Convert RGB image to grayscale using the luminance formula
        gray_image =  0.144 * image[:, 0, :, :] + 0.5870 * image[:, 1, :, :] + 0.299 * image[:, 2, :, :]
        return gray_image.unsqueeze(1)  # Add a channel dimension for compatibility


class L_tv(Loss):

    # --- Lifecycle & Initialization ---
    def __init__(self, weight: float = 1.0, reduction: str = "mean"):
        """Initialize a new instance.

        Args:
            reduction (str, optional): Reduction method to apply to the loss.
                One of: ["mean", "sum", "none"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.weight = weight

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        loss = 2 * (h_tv / count_h + w_tv / count_w) / batch_size
        return self.weight * self.reduce(loss)


class SmoothLoss(Loss):

    # --- Lifecycle & Initialization ---
    def __init__(self, reduction: str = "mean"):
        """Initialize a new instance.

        Args:
            reduction (str, optional): Reduction method to apply to the loss.
                One of: ["mean", "sum", "none"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.sigma = 10
        self.input = None
        self.output = None

    # --- Callable & Context Manager ---
    def forward(self, input: Tensor, output: Tensor) -> Tensor:
        self.output = output
        self.input = self.rgb_to_ycbcr(input)
        sigma_color = -1.0 / (2 * self.sigma * self.sigma)
        w1 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :] - self.input[:, :, :-1, :], 2), dim=1, keepdim=True) * sigma_color)
        w2 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :] - self.input[:, :, 1:, :], 2), dim=1, keepdim=True) * sigma_color)
        w3 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 1:] - self.input[:, :, :, :-1], 2), dim=1, keepdim=True) * sigma_color)
        w4 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-1] - self.input[:, :, :, 1:], 2), dim=1, keepdim=True) * sigma_color)
        w5 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-1] - self.input[:, :, 1:, 1:], 2), dim=1, keepdim=True) * sigma_color)
        w6 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 1:] - self.input[:, :, :-1, :-1], 2), dim=1, keepdim=True) * sigma_color)
        w7 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-1] - self.input[:, :, :-1, 1:], 2), dim=1, keepdim=True) * sigma_color)
        w8 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 1:] - self.input[:, :, 1:, :-1], 2), dim=1, keepdim=True) * sigma_color)
        w9 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :] - self.input[:, :, :-2, :], 2), dim=1, keepdim=True) * sigma_color)
        w10 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :] - self.input[:, :, 2:, :], 2), dim=1, keepdim=True) * sigma_color)
        w11 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 2:] - self.input[:, :, :, :-2], 2), dim=1, keepdim=True) * sigma_color)
        w12 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-2] - self.input[:, :, :, 2:], 2), dim=1, keepdim=True) * sigma_color)
        w13 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-1] - self.input[:, :, 2:, 1:], 2), dim=1, keepdim=True) * sigma_color)
        w14 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 1:] - self.input[:, :, :-2, :-1], 2), dim=1, keepdim=True) * sigma_color)
        w15 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-1] - self.input[:, :, :-2, 1:], 2), dim=1, keepdim=True) * sigma_color)
        w16 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 1:] - self.input[:, :, 2:, :-1], 2), dim=1, keepdim=True) * sigma_color)
        w17 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-2] - self.input[:, :, 1:, 2:], 2), dim=1, keepdim=True) * sigma_color)
        w18 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 2:] - self.input[:, :, :-1, :-2], 2), dim=1, keepdim=True) * sigma_color)
        w19 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-2] - self.input[:, :, :-1, 2:], 2), dim=1, keepdim=True) * sigma_color)
        w20 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 2:] - self.input[:, :, 1:, :-2], 2), dim=1, keepdim=True) * sigma_color)
        w21 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-2] - self.input[:, :, 2:, 2:], 2), dim=1, keepdim=True) * sigma_color)
        w22 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 2:] - self.input[:, :, :-2, :-2], 2), dim=1, keepdim=True) * sigma_color)
        w23 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-2] - self.input[:, :, :-2, 2:], 2), dim=1, keepdim=True) * sigma_color)
        w24 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 2:] - self.input[:, :, 2:, :-2], 2), dim=1, keepdim=True) * sigma_color)
        p = 1.0

        pixel_grad1 = w1 * torch.norm((self.output[:, :, 1:, :] - self.output[:, :, :-1, :]), p, dim=1, keepdim=True)
        pixel_grad2 = w2 * torch.norm((self.output[:, :, :-1, :] - self.output[:, :, 1:, :]), p, dim=1, keepdim=True)
        pixel_grad3 = w3 * torch.norm((self.output[:, :, :, 1:] - self.output[:, :, :, :-1]), p, dim=1, keepdim=True)
        pixel_grad4 = w4 * torch.norm((self.output[:, :, :, :-1] - self.output[:, :, :, 1:]), p, dim=1, keepdim=True)
        pixel_grad5 = w5 * torch.norm((self.output[:, :, :-1, :-1] - self.output[:, :, 1:, 1:]), p, dim=1, keepdim=True)
        pixel_grad6 = w6 * torch.norm((self.output[:, :, 1:, 1:] - self.output[:, :, :-1, :-1]), p, dim=1, keepdim=True)
        pixel_grad7 = w7 * torch.norm((self.output[:, :, 1:, :-1] - self.output[:, :, :-1, 1:]), p, dim=1, keepdim=True)
        pixel_grad8 = w8 * torch.norm((self.output[:, :, :-1, 1:] - self.output[:, :, 1:, :-1]), p, dim=1, keepdim=True)
        pixel_grad9 = w9 * torch.norm((self.output[:, :, 2:, :] - self.output[:, :, :-2, :]), p, dim=1, keepdim=True)
        pixel_grad10 = w10 * torch.norm((self.output[:, :, :-2, :] - self.output[:, :, 2:, :]), p, dim=1, keepdim=True)
        pixel_grad11 = w11 * torch.norm((self.output[:, :, :, 2:] - self.output[:, :, :, :-2]), p, dim=1, keepdim=True)
        pixel_grad12 = w12 * torch.norm((self.output[:, :, :, :-2] - self.output[:, :, :, 2:]), p, dim=1, keepdim=True)
        pixel_grad13 = w13 * torch.norm((self.output[:, :, :-2, :-1] - self.output[:, :, 2:, 1:]), p, dim=1, keepdim=True)
        pixel_grad14 = w14 * torch.norm((self.output[:, :, 2:, 1:] - self.output[:, :, :-2, :-1]), p, dim=1, keepdim=True)
        pixel_grad15 = w15 * torch.norm((self.output[:, :, 2:, :-1] - self.output[:, :, :-2, 1:]), p, dim=1, keepdim=True)
        pixel_grad16 = w16 * torch.norm((self.output[:, :, :-2, 1:] - self.output[:, :, 2:, :-1]), p, dim=1, keepdim=True)
        pixel_grad17 = w17 * torch.norm((self.output[:, :, :-1, :-2] - self.output[:, :, 1:, 2:]), p, dim=1, keepdim=True)
        pixel_grad18 = w18 * torch.norm((self.output[:, :, 1:, 2:] - self.output[:, :, :-1, :-2]), p, dim=1, keepdim=True)
        pixel_grad19 = w19 * torch.norm((self.output[:, :, 1:, :-2] - self.output[:, :, :-1, 2:]), p, dim=1, keepdim=True)
        pixel_grad20 = w20 * torch.norm((self.output[:, :, :-1, 2:] - self.output[:, :, 1:, :-2]), p, dim=1, keepdim=True)
        pixel_grad21 = w21 * torch.norm((self.output[:, :, :-2, :-2] - self.output[:, :, 2:, 2:]), p, dim=1, keepdim=True)
        pixel_grad22 = w22 * torch.norm((self.output[:, :, 2:, 2:] - self.output[:, :, :-2, :-2]), p, dim=1, keepdim=True)
        pixel_grad23 = w23 * torch.norm((self.output[:, :, 2:, :-2] - self.output[:, :, :-2, 2:]), p, dim=1, keepdim=True)
        pixel_grad24 = w24 * torch.norm((self.output[:, :, :-2, 2:] - self.output[:, :, 2:, :-2]), p, dim=1, keepdim=True)

        ReguTerm1 = torch.mean(pixel_grad1) \
                    + torch.mean(pixel_grad2) \
                    + torch.mean(pixel_grad3) \
                    + torch.mean(pixel_grad4) \
                    + torch.mean(pixel_grad5) \
                    + torch.mean(pixel_grad6) \
                    + torch.mean(pixel_grad7) \
                    + torch.mean(pixel_grad8) \
                    + torch.mean(pixel_grad9) \
                    + torch.mean(pixel_grad10) \
                    + torch.mean(pixel_grad11) \
                    + torch.mean(pixel_grad12) \
                    + torch.mean(pixel_grad13) \
                    + torch.mean(pixel_grad14) \
                    + torch.mean(pixel_grad15) \
                    + torch.mean(pixel_grad16) \
                    + torch.mean(pixel_grad17) \
                    + torch.mean(pixel_grad18) \
                    + torch.mean(pixel_grad19) \
                    + torch.mean(pixel_grad20) \
                    + torch.mean(pixel_grad21) \
                    + torch.mean(pixel_grad22) \
                    + torch.mean(pixel_grad23) \
                    + torch.mean(pixel_grad24)
        total_term = ReguTerm1
        return self.reduce(total_term)

    # --- Utilities ---
    def rgb_to_ycbcr(self, rgb: Tensor) -> Tensor:
        device = rgb.device
        im_flat = rgb.contiguous().view(-1, 3).float()
        mat = torch.Tensor(
            [[0.257, -0.148, 0.439],
             [0.564, -0.291, -0.368],
             [0.098, 0.439, -0.071]]
        ).to(device)
        bias = torch.Tensor([16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0]).to(device)
        temp = im_flat.mm(mat) + bias
        out = temp.view(rgb.shape[0], 3, rgb.shape[2], rgb.shape[3])
        return out


class LossFunction(Loss):

    # --- Lifecycle & Initialization ---
    def __init__(self, reduction: str = "mean"):
        """Initialize a new instance.

        Args:
            reduction (str, optional): Reduction method to apply to the loss.
                One of: ["mean", "sum", "none"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.l2_loss = nn.MSELoss()
        self.smooth_loss = SmoothLoss()
        self.local_mean = LocalMean(patch_size=5)
        self.l_tv_loss = L_tv()

    # --- Callable & Context Manager ---
    def forward(
        self, input, L_pred1, L_pred2, L2, s2, s21, s22, H2, H11, H12, H13, s13,
        H14, s14, H3, s3, H3_pred, H4_pred, L_pred1_L_pred2_diff,
        H3_denoised1_H3_denoised2_diff, H2_blur, H3_blur,
    ):
        eps = 1e-9
        input = input + eps

        input_Y = L2.detach()[:, 2, :, :] * 0.299 + L2.detach()[:, 1, :, :] * 0.587 + L2.detach()[:, 0, :, :] * 0.144
        input_Y_mean = torch.mean(input_Y, dim=(1, 2))
        enhancement_factor = 0.5/ (input_Y_mean + eps)
        enhancement_factor = enhancement_factor.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        enhancement_factor = torch.clamp(enhancement_factor, 1, 25)
        adjustment_ratio = torch.pow(0.7, -enhancement_factor) / enhancement_factor
        adjustment_ratio = adjustment_ratio.repeat(1, 3, 1, 1)
        normalized_low_light_layer = L2.detach() / s2
        normalized_low_light_layer = torch.clamp(normalized_low_light_layer, eps, 0.8)
        enhanced_brightness=torch.pow(L2.detach()*enhancement_factor, enhancement_factor)
        clamped_enhanced_brightness = torch.clamp(enhanced_brightness * adjustment_ratio, eps, 1)
        clamped_adjusted_low_light = torch.clamp(L2.detach() * enhancement_factor,eps,1)
        # Enhance_loss
        loss = self.l2_loss(s2, clamped_enhanced_brightness) * 700
        loss += self.l2_loss(normalized_low_light_layer, clamped_adjusted_low_light) * 1000
        loss += self.smooth_loss(L2.detach(), s2) * 5
        loss += self.l_tv_loss(s2) * 1600
        # Loss_res_1
        L11, L12 = pair_downsample(input)
        loss += self.l2_loss(L11, L_pred2) * 1000
        loss += self.l2_loss(L12, L_pred1) * 1000
        denoised1, denoised2 = pair_downsample(L2)
        loss += self.l2_loss(L_pred1, denoised1) * 1000
        loss += self.l2_loss(L_pred2, denoised2) * 1000
        # Loss_res_2
        loss += self.l2_loss(H3_pred, torch.cat([H12.detach(), s22.detach()], 1)) * 1000
        loss += self.l2_loss(H4_pred, torch.cat([H11.detach(), s21.detach()], 1)) * 1000
        H3_denoised1, H3_denoised2 = pair_downsample(H3)
        loss += self.l2_loss(H3_pred[:, 0:3, :, :], H3_denoised1) * 1000
        loss += self.l2_loss(H4_pred[:, 0:3, :, :], H3_denoised2) * 1000
        # Loss_color
        loss += self.l2_loss(H2_blur.detach(), H3_blur) * 10000
        # Loss_ill
        loss += self.l2_loss(s2.detach(), s3) * 1000
        # Loss_cons
        local_mean1 = self.local_mean(H3_denoised1)
        local_mean2 = self.local_mean(H3_denoised2)
        weighted_diff1 = (1 - H3_denoised1_H3_denoised2_diff) * local_mean1+H3_denoised1 * H3_denoised1_H3_denoised2_diff
        weighted_diff2 = (1 - H3_denoised1_H3_denoised2_diff) * local_mean2+H3_denoised1 * H3_denoised1_H3_denoised2_diff
        loss += self.l2_loss(H3_denoised1,weighted_diff1) * 10000
        loss += self.l2_loss(H3_denoised2, weighted_diff2) * 10000
        # Loss_Var
        noise_std = calculate_local_variance(H3 - H2)
        H2_var = calculate_local_variance(H2)
        loss += self.l2_loss(H2_var, noise_std) * 1000

        return self.reduce(loss)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
