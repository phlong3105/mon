#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements loss functions for training vision deep learning
models.

This module is built on top of :mod:`mon.nn.loss`.
"""

from __future__ import annotations

__all__ = [
    "BrightnessConstancyLoss",
    "ChannelConsistencyLoss",
    "ChannelRatioConsistencyLoss",
    "ColorConstancyLoss",
    "EdgeConstancyLoss",
    "ExclusionLoss",
    "ExposureControlLoss",
    "GradientLoss",
    "GrayLoss",
    "IlluminationSmoothnessLoss",
    "NonBlurryLoss",
    "PSNRLoss",
    "PerceptualL1Loss",
    "PerceptualLoss",
    "SSIMLoss",
    "SpatialConsistencyLoss",
    "StdLoss",
    "ContradictChannelLoss",
]

from typing import Literal

import numpy as np
import piqa
import torch

from mon import core, nn
from mon.globals import LOSSES, Reduction
from mon.nn import functional as F
from mon.nn.loss import *
from mon.vision import prior

console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region Channel Loss

@LOSSES.register(name="channel_consistency_loss")
class ChannelConsistencyLoss(Loss):
    """Channel Consistency Loss :math:`\mathcal{L}_{kl}` enhances the
    consistency between the original image and the enhanced image in the channel
    pixel difference through KL divergence. It also suppresses the generation of
    noise information and invalid features to improve the image enhancement
    effect.
    
    Equation:
        :math:`\mathcal{L}_{kl} = KL[R−B][R′−B′] + KL[R−G][R′−G′] + KL[G−B][G′−B′]`
    """
    
    def __init__(
        self,
        reduction : Reduction | str = "mean",
        log_target: bool            = True,
    ):
        super().__init__(reduction=reduction)
        self.log_target = log_target
    
    def __str__(self) -> str:
        return f"channel_consistency_loss"
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        assert input.shape == target.shape
        # loss = F.kl_div(input, target, log_target=self.log_target)
        r1, g1, b1 = torch.split(input,  1, dim=1)
        r2, g2, b2 = torch.split(target, 1, dim=1)
        """
        r1    =  input[:, 0, :, :]
        g1    =  input[:, 1, :, :]
        b1    =  input[:, 2, :, :]
        r2    = target[:, 0, :, :]
        g2    = target[:, 1, :, :]
        b2    = target[:, 2, :, :]
        """
        d_rb1 = r1 - b1
        d_rb2 = r2 - b2
        d_rg1 = r1 - g1
        d_rg2 = r2 - g2
        d_gb1 = g1 - b1
        d_gb2 = g2 - b2
        kl_rb = F.kl_div(d_rb1, d_rb2, reduction="mean", log_target=self.log_target)
        kl_rg = F.kl_div(d_rg1, d_rg2, reduction="mean", log_target=self.log_target)
        kl_gb = F.kl_div(d_gb1, d_gb2, reduction="mean", log_target=self.log_target)
        loss  = kl_rb + kl_rg + kl_gb
       
        # loss  = reduce_loss(loss=loss, reduction=self.reduction)
        return loss


@LOSSES.register(name="channel_ratio_consistency_loss")
class ChannelRatioConsistencyLoss(Loss):
    """Channel Ratio Consistency Loss :math:`\mathcal{L}_{crl}` constrains the
    intrinsic ratio among three channels to prevent potential color deviations
    in the enhanced image.
    
    References:
        `<https://github.com/GuoLanqing/ReLLIE/blob/main/Myloss.py#L26>`__
    """
    
    def __init__(
        self,
        reduction : Reduction | str = "mean",
        log_target: bool            = False,
    ):
        super().__init__(reduction=reduction)
        self.log_target = log_target
    
    def __str__(self) -> str:
        return f"channel_ratio_consistency_loss"
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        assert input.shape == target.shape
        r1, g1, b1 = torch.split(input  * 255, 1, dim=1)
        r2, g2, b2 = torch.split(target * 255, 1, dim=1)
        d_rg       = torch.pow(r1.int() // g1.int() - r2.int() // g2.int(), 2).sum() / 255.0 ** 2
        d_rb       = torch.pow(r1.int() // b1.int() - r2.int() // b2.int(), 2).sum() / 255.0 ** 2
        d_gb       = torch.pow(g1.int() // b1.int() - g2.int() // b2.int(), 2).sum() / 255.0 ** 2
        loss       = torch.pow(d_rg + d_rb + d_gb, 0.5)
        # loss       = reduce_loss(loss=loss, reduction=self.reduction)
        return loss

    
@LOSSES.register(name="color_constancy_loss")
class ColorConstancyLoss(Loss):
    """Color Constancy Loss :math:`\mathcal{L}_{col}` corrects the potential
    color deviations in the enhanced image and builds the relations among the
    three adjusted channels.

    References:
        `<https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py#L9>`__
    """
    
    def __init__(self, reduction: Reduction | str = "mean"):
        super().__init__(reduction=reduction)
    
    def __str__(self) -> str:
        return f"color_constancy_loss"
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None
    ) -> torch.Tensor:
        mean_rgb   = torch.mean(input, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        d_rg       = torch.pow(mr - mg, 2)
        d_rb       = torch.pow(mr - mb, 2)
        d_gb       = torch.pow(mb - mg, 2)
        loss       = torch.pow(torch.pow(d_rg, 2) + torch.pow(d_rb, 2) + torch.pow(d_gb, 2), 0.5)
        loss       = reduce_loss(loss=loss, reduction=self.reduction)
        return loss


@LOSSES.register(name="grayscale_loss")
class GrayscaleLoss(nn.Module):

    def __init__(self, reduction: Reduction | str = "mean"):
        super().__init__(reduction=reduction)
        self.mse = MSELoss()

    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        x_g  = torch.mean(input,  1, keepdim=True)
        y_g  = torch.mean(target, 1, keepdim=True)
        loss = self.mse(x_g, y_g)
        loss = reduce_loss(loss=loss, reduction=self.reduction)
        return loss
    

@LOSSES.register(name="contradict_channel_loss")
class ContradictChannelLoss(Loss):
    """Contradic Channel Loss :math:`\mathcal{L}_{con}` measures the distance
    between the average intensity value of a local region to the
    well-exposedness level E.

    Args:
        patch_size: Kernel size for pooling layer.
        mean_val: The :math:`E` value proposed in the paper. Default: ``0.6``.
        reduction: Specifies the reduction to apply to the output.
    
    References:
        `<https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_ALL_Snow_Removed_Single_Image_Desnowing_Algorithm_Using_Hierarchical_Dual-Tree_ICCV_2021_paper.pdf>`__
    """
        
    def __init__(
        self,
        reduction : Reduction | str = "mean",
        kernel_size: int | list[int] = 35,
    ):
        super().__init__(reduction=reduction)
        self.kernel_size = kernel_size
        self.pool       = nn.MaxPool2d(
            kernel_size = self.kernel_size,
            stride      = 1,
            padding     = int(self.kernel_size//2)
        )
        self.mae = MAELoss()
        self.sigmoid = nn.Sigmoid()
    
    def __str__(self) -> str:
        return f"contradict_channel_loss"
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None  = None
    ) -> torch.Tensor:
        y_pred = torch.min(input,dim=1,keepdim=True)
        y_pred = torch.squeeze(y_pred[0])
        y_pred = self.pool(y_pred)

        y_true = torch.min(target,dim=1,keepdim=True)
        y_true = torch.squeeze(y_true[0])
        y_true = self.pool(y_true)

        loss = self.mae(y_pred, y_true)
        loss = reduce_loss(loss=loss, reduction=self.reduction)
        return self.sigmoid(loss)

# endregion


# region Exposure Loss

@LOSSES.register(name="brightness_constancy_loss")
class BrightnessConstancyLoss(Loss):
    """Brightness Constancy Loss

    Args:
        patch_size: Kernel size for pooling layer.
        mean_val: The :math:`E` value proposed in the paper. Default: ``0.6``.
        reduction: Specifies the reduction to apply to the output.
    
    References:
        `<https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py#L74>`__
    """
    
    def __init__(
        self,
        reduction: Reduction | str = "mean",
        gamma    : float           = 2.5,
        ksize    : int | None      = 9,
        eps      : float           = 1e-3
    ):
        super().__init__(reduction=reduction)
        self.gamma = gamma
        self.ksize = ksize
        self.eps   = eps
    
    def __str__(self) -> str:
        return f"brightness_consistency_loss"
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        target = prior.get_guided_brightness_enhancement_map_prior(target,  self.gamma, self.ksize)
        # loss    = torch.abs(target - input)
        loss    = torch.sqrt((target - input) ** 2 + (self.eps * self.eps))
        loss    = reduce_loss(loss=loss, reduction=self.reduction)
        return loss
    

@LOSSES.register(name="exposure_control_loss")
class ExposureControlLoss(Loss):
    """Exposure Control Loss :math:`\mathcal{L}_{exp}` measures the distance
    between the average intensity value of a local region to the
    well-exposedness level E.

    Args:
        patch_size: Kernel size for pooling layer.
        mean_val: The :math:`E` value proposed in the paper. Default: ``0.6``.
        reduction: Specifies the reduction to apply to the output.
    
    References:
        `<https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py#L74>`__
    """
    
    def __init__(
        self,
        reduction : Reduction | str = "mean",
        patch_size: int | list[int] = 16,
        mean_val  : float           = 0.6,
    ):
        super().__init__(reduction=reduction)
        self.patch_size = patch_size
        self.mean_val   = mean_val
        self.pool       = nn.AvgPool2d(self.patch_size)
    
    def __str__(self) -> str:
        return f"exposure_control_loss"
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None  = None
    ) -> torch.Tensor:
        x    = input
        x    = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        loss = torch.pow(mean - torch.FloatTensor([self.mean_val]).to(input.device), 2)
        loss = reduce_loss(loss=loss, reduction=self.reduction)
        return loss

# endregion


# region Perceptual Loss

@LOSSES.register(name="edge_constancy_loss")
class EdgeConstancyLoss(Loss):
    """Edge Constancy Loss :math:`\mathcal{L}_{edge}`."""
    
    def __init__(self, reduction: Reduction | str = "mean", eps: float = 1e-3):
        super().__init__(reduction=reduction)
        self.eps    = eps
        k           = torch.Tensor([[0.05, 0.25, 0.4, 0.25, 0.05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
    
    def __str__(self) -> str:
        return f"edge_constancy_loss"
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert input.shape == target.shape
        edge1 = self.laplacian_kernel(input)
        edge2 = self.laplacian_kernel(target)
        loss  = torch.sqrt((edge1 - edge2) ** 2 + (self.eps * self.eps))
        loss  = reduce_loss(loss=loss, reduction=self.reduction)
        return loss
    
    def gauss_conv(self, image: torch.Tensor) -> torch.Tensor:
        if self.kernel.device != image.device:
            self.kernel = self.kernel.to(image.device)
        n, c, w, h = self.kernel.shape
        image = F.pad(image, (w // 2, h // 2, w // 2, h // 2), mode="replicate")
        gauss = F.conv2d(image, self.kernel, groups=n)
        return gauss
        
    def laplacian_kernel(self, image: torch.Tensor) -> torch.Tensor:
        filtered   = self.gauss_conv(image)       # filter
        down       = filtered[:, :, ::2, ::2]     # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4     # upsample
        filtered   = self.gauss_conv(new_filter)  # filter
        laplacian  = image - filtered
        return laplacian


@LOSSES.register(name="perceptual_loss")
class PerceptualLoss(Loss):
    """Perceptual Loss."""
    
    def __init__(self, vgg: nn.Module, reduction: Reduction | str = "mean"):
        super().__init__(reduction=reduction)
        self.mse = MSELoss(reduction=reduction)
        self.vgg = vgg
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def __str__(self) -> str:
        return f"perceptual_Loss"
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        # if self.vgg.device != input[0].device:
        #     self.vgg = self.vgg.to(input[0].device)
        input_feats  = self.vgg(input)
        target_feats = self.vgg(target)
        loss = self.mse(input=input_feats, target=target_feats)
        return loss


@LOSSES.register(name="perceptual_l1_loss")
class PerceptualL1Loss(Loss):
    """Loss = weights[0] * Perceptual Loss + weights[1] * L1 Loss."""
    
    def __init__(self, vgg: nn.Module, reduction: Reduction | str = "mean"):
        super().__init__(reduction=reduction)
        self.per_loss = PerceptualLoss(
            vgg       = vgg,
            reduction = reduction,
        )
        self.l1_loss = L1Loss(
            reduction=reduction,
        )
        self.layer_name_mapping = {
            "3" : "relu1_2",
            "8" : "relu2_2",
            "15": "relu3_3"
        }
    
    def __str__(self) -> str:
        return f"perceptual_l1_Loss"
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        per_loss = self.per_loss(input=input, target=target)
        l1_loss  = self.l1_loss(input=input, target=target)
        loss     = per_loss + l1_loss
        return loss

# endregion


# region Reconstruction Loss

@LOSSES.register(name="exclusion_loss")
class ExclusionLoss(nn.Module):
    """Loss on the gradient.

    References:
    `<http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single_Image_Reflection_CVPR_2018_paper.pdf>`__
    """

    def __init__(
        self,
        level    : int             = 3,
        reduction: Reduction | str = "mean",
    ):
        super().__init__(reduction=reduction)
        self.level    = level
        self.avg_pool = nn.AvgPool2d(2, stride=2).type(torch.cuda.FloatTensor)
        self.sigmoid  = nn.Sigmoid().type(torch.cuda.FloatTensor)

    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor | float:
        grad_x_loss, grad_y_loss = self.get_gradients(input, target)
        loss_grad_xy = sum(grad_x_loss) / (self.level * 9) + sum(grad_y_loss) / (self.level * 9)
        loss         = loss_grad_xy / 2.0
        # loss         = reduce_loss(loss=loss, reduction=self.reduction)
        return loss

    def get_gradients(
        self,
        input : torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[list, list]:
        grad_x_loss = []
        grad_y_loss = []

        for l in range(self.level):
            grad_x1, grad_y1 = self.compute_gradient(input)
            grad_x2, grad_y2 = self.compute_gradient(target)
            # alpha_x = 2.0 * torch.mean(torch.abs(grad_x1)) / torch.mean(torch.abs(grad_x2))
            # alpha_y = 2.0 * torch.mean(torch.abs(grad_y1)) / torch.mean(torch.abs(grad_y2))
            alpha_y   = 1
            alpha_x   = 1
            grad_x1_s = (self.sigmoid(grad_x1) * 2) - 1
            grad_y1_s = (self.sigmoid(grad_y1) * 2) - 1
            grad_x2_s = (self.sigmoid(grad_x2  * alpha_x) * 2) - 1
            grad_y2_s = (self.sigmoid(grad_y2  * alpha_y) * 2) - 1
            # grad_x_loss.append(torch.mean(((grad_x1_s ** 2) * (grad_x2_s ** 2))) ** 0.25)
            # grad_y_loss.append(torch.mean(((grad_y1_s ** 2) * (grad_y2_s ** 2))) ** 0.25)
            grad_x_loss += self._all_comb(grad_x1_s, grad_x2_s)
            grad_y_loss += self._all_comb(grad_y1_s, grad_y2_s)
            input        = self.avg_pool(input)
            target        = self.avg_pool(target)
        return grad_x_loss, grad_y_loss

    def _all_comb(self, grad1_s: torch.Tensor, grad2_s: torch.Tensor) -> torch.Tensor:
        v = []
        for i in range(3):
            for j in range(3):
                v.append(torch.mean(((grad1_s[:, j, :, :] ** 2) * (grad2_s[:, i, :, :] ** 2))) ** 0.25)
        return v

    def compute_gradient(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grad_x = input[:, :, 1:, :] - input[:, :, :-1, :]
        grad_y = input[:, :, :, 1:] - input[:, :, :, :-1]
        return grad_x, grad_y


@LOSSES.register(name="gradient_loss")
class GradientLoss(Loss):
    """L1 loss on the gradient of the image."""
    
    def __init__(self, reduction: Reduction | str = "mean"):
        super().__init__(reduction=reduction)
    
    def __str__(self) -> str:
        return f"gradient_l1_loss"
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None
    ) -> torch.Tensor:
        gradient_a_x = torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:])
        gradient_a_y = torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :])
        loss = reduce_loss(
            loss      = torch.mean(gradient_a_x) + torch.mean(gradient_a_y),
            reduction = self.reduction
        )
        return loss


@LOSSES.register(name="gray_loss")
class GrayLoss(nn.Module):

    def __init__(self, reduction: Reduction | str = "mean"):
        super().__init__(reduction=reduction)
        self.mae = MAELoss()

    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = input
        y = torch.ones_like(x) / 2.0
        loss = 1 / self.mae(x, y)
        loss = reduce_loss(loss=loss, reduction=self.reduction)
        return loss


@LOSSES.register(name="non_blurry_loss")
class NonBlurryLoss(nn.Module):
    """Loss on the distance to 0.5."""

    def __init__(self, reduction: Reduction | str = "mean"):
        super().__init__(reduction=reduction)
        self.mse = MSELoss()

    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x    = input
        loss = 1 - self.mse(x, torch.ones_like(x) * 0.5)
        loss = reduce_loss(loss=loss, reduction=self.reduction)
        return loss


@LOSSES.register(name="psnr_loss")
class PSNRLoss(Loss):
    """PSNR loss. Modified from BasicSR: `<https://github.com/xinntao/BasicSR>`__
    """
    
    def __init__(
        self,
        reduction  : Reduction | str = "mean",
        loss_weight: float           = 1.0,
        to_y       : bool            = False,
    ):
        super().__init__(reduction=reduction)
        self.loss_weight = loss_weight
        self.scale       = 10 / np.log(10)
        self.to_y        = to_y
        self.coef        = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first       = True

    def __str__(self) -> str:
        return f"psnr_loss"

    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        assert len(input.size()) == 4
        if self.to_y:
            if self.first:
                self.coef  = self.coef.to(input.device)
                self.first = False

            input  = (input  * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.0
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.0
            input  = input  / 255.0
            target = target / 255.0

        psnr = torch.log(((input - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        loss = self.loss_weight * self.scale * psnr
        # loss = reduce_loss(loss=loss, reduction=self.reduction)
        return loss


@LOSSES.register(name="ssim_loss")
class SSIMLoss(Loss):
    """SSIM Loss. Modified from BasicSR: https://github.com/xinntao/BasicSR"""
    
    def __init__(
        self,
        reduction  : Reduction | str = "mean",
        window_size: int             = 11,
        sigma      : float           = 1.5,
        n_channels : int             = 3,
        value_range: float           = 1.0,
    ):
        super().__init__(reduction=reduction)
        self.ssim = piqa.SSIM(
            window_size = window_size,
            sigma       = sigma,
            n_channels  = n_channels,
            value_range = value_range,
            reduction   = self.reduction,
        )
    
    def __str__(self) -> str:
        return f"ssim_loss"
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        # Compute the ssim map
        ssim_map = self.ssim(input=input, target=target)
        # Compute and reduce the loss
        loss = torch.clamp((1.0 - ssim_map) / 2, min=0, max=1)
        loss = reduce_loss(loss=loss, reduction=self.reduction)
        return loss


@LOSSES.register(name="std_loss")
class StdLoss(Loss):
    """Loss on the variance of the image. Works in the grayscale. If the image
    is smooth, gets zero.
    """

    def __init__(self, reduction: Reduction | str = "mean"):
        super().__init__(reduction=reduction)
        blur      = (1 / 25) * np.ones((5, 5))
        blur      = blur.reshape(1, 1, blur.shape[0], blur.shape[1])
        self.blur = nn.Parameter(data=torch.cuda.FloatTensor(blur), requires_grad=False)
        self.mse  = MSELoss()

        image       = np.zeros((5, 5))
        image[2, 2] = 1
        image       = image.reshape(1, 1, image.shape[0], image.shape[1])
        self.image  = nn.Parameter(data=torch.cuda.FloatTensor(image), requires_grad=False)

    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None
    ) -> torch.Tensor:
        x    = input
        x    = torch.mean(x, 1, keepdim=True)
        if self.image.device != x.device:
            self.image = self.image.to(x.device)
        loss = self.mse(F.conv2d(x, self.image), F.conv2d(x, self.blur))
        loss = reduce_loss(loss=loss, reduction=self.reduction)
        return loss

# endregion


# region Smoothness Loss

@LOSSES.register(name="illumination_smoothness_loss")
class IlluminationSmoothnessLoss(Loss):
    """Illumination Smoothness Loss :math:`\mathcal{L}_{tvA}` preserve the
    monotonicity relations between neighboring pixels. It is used to avoid
    aggressive and sharp changes between neighboring pixels.
    
    References:
        `<https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py>`__
    """
    
    def __init__(
        self,
        reduction     : Reduction | str = "mean",
        tv_loss_weight: int             = 1
    ):
        super().__init__(reduction=reduction)
        self.tv_loss_weight = tv_loss_weight
    
    def __str__(self) -> str:
        return f"illumination_smoothness_loss"
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None  = None
    ) -> torch.Tensor:
        x       = input
        b       = x.size()[0]
        h_x     = x.size()[2]
        w_x     = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv    = torch.pow((x[:, :, 1:,  :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv    = torch.pow((x[:, :,  :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        loss    = self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / b
        return loss

# endregion


# region Spatial Loss

@LOSSES.register(name="spatial_consistency_loss")
class SpatialConsistencyLoss(Loss):
    """Spatial Consistency Loss :math:`\mathcal{L}_{spa}` encourages spatial
    coherence of the enhanced image through preserving the difference of
    neighboring regions between the input image and its enhanced version.
    
    Args:
        num_regions: Number of neighboring regions. Default: ``4``.
        patch_size: The size of each neighboring region. Defaults: ``4`` means
            :math:`4 x 4`.
    """
    
    def __init__(
        self,
        num_regions: Literal[4, 8, 16, 24] = 4,
        patch_size : int                   = 4,
        reduction  : Reduction | str       = "mean",
    ):
        super().__init__(reduction=reduction)
        self.num_regions = num_regions
        
        kernel_left = torch.FloatTensor([
            [ 0,  0, 0],
            [-1,  1, 0],
            [ 0,  0, 0]
        ]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([
            [0,  0,  0],
            [0,  1, -1],
            [0,  0,  0]
        ]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([
            [0, -1, 0],
            [0,  1, 0],
            [0,  0, 0]
        ]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([
            [0,  0, 0],
            [0,  1, 0],
            [0, -1, 0]
        ]).unsqueeze(0).unsqueeze(0)
        if self.num_regions in [8, 16]:
            kernel_upleft = torch.FloatTensor([
                [-1, 0, 0],
                [ 0, 1, 0],
                [ 0, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_upright = torch.FloatTensor([
                [0, 0, -1],
                [0, 1,  0],
                [0, 0,  0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_downleft = torch.FloatTensor([
                [ 0, 0, 0],
                [ 0, 1, 0],
                [-1, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_downright = torch.FloatTensor([
                [0, 0,  0],
                [0, 1,  0],
                [0, 0, -1]
            ]).unsqueeze(0).unsqueeze(0)
        if self.num_regions in [16, 24]:
            kernel_left2 = torch.FloatTensor([
                [0,  0,  0, 0, 0],
                [0,  0,  0, 0, 0],
                [-1, 0,  1, 0, 0],
                [0,  0,  0, 0, 0],
                [0,  0,  0, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_right2 = torch.FloatTensor([
                [0, 0,  0, 0,  0],
                [0, 0,  0, 0,  0],
                [0, 0,  1, 0, -1],
                [0, 0,  0, 0,  0],
                [0, 0,  0, 0,  0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_up2 = torch.FloatTensor([
                [0, 0, -1, 0, 0],
                [0, 0,  0, 0, 0],
                [0, 0,  1, 0, 0],
                [0, 0,  0, 0, 0],
                [0, 0,  0, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_down2 = torch.FloatTensor([
                [0, 0,  0, 0, 0],
                [0, 0,  0, 0, 0],
                [0, 0,  1, 0, 0],
                [0, 0,  0, 0, 0],
                [0, 0, -1, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_up2left2 = torch.FloatTensor([
                [-1, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0],
                [ 0, 0, 1, 0, 0],
                [ 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_up2right2 = torch.FloatTensor([
                [0, 0, 0, 0, -1],
                [0, 0, 0, 0,  0],
                [0, 0, 1, 0,  0],
                [0, 0, 0, 0,  0],
                [0, 0, 0, 0,  0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_down2left2 = torch.FloatTensor([
                [ 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0],
                [ 0, 0, 1, 0, 0],
                [ 0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_down2right2 = torch.FloatTensor([
                [0, 0, 0, 0,  0],
                [0, 0, 0, 0,  0],
                [0, 0, 1, 0,  0],
                [0, 0, 0, 0,  0],
                [0, 0, 0, 0, -1]
            ]).unsqueeze(0).unsqueeze(0)
        if self.num_regions in [24]:
            kernel_up2left1 = torch.FloatTensor([
                [0, -1, 0, 0, 0],
                [0,  0, 0, 0, 0],
                [0,  0, 1, 0, 0],
                [0,  0, 0, 0, 0],
                [0,  0, 0, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_up2right1 = torch.FloatTensor([
                [0, 0, 0, -1, 0],
                [0, 0, 0,  0, 0],
                [0, 0, 1,  0, 0],
                [0, 0, 0,  0, 0],
                [0, 0, 0,  0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_up1left2 = torch.FloatTensor([
                [0,  0, 0, 0, 0],
                [-1, 0, 0, 0, 0],
                [0,  0, 1, 0, 0],
                [0,  0, 0, 0, 0],
                [0,  0, 0, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_up1right2 = torch.FloatTensor([
                [0, 0, 0, 0,  0],
                [0, 0, 0, 0, -1],
                [0, 0, 1, 0,  0],
                [0, 0, 0, 0,  0],
                [0, 0, 0, 0,  0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_down2left1 = torch.FloatTensor([
                [0,  0, 0, 0, 0],
                [0,  0, 0, 0, 0],
                [0,  0, 1, 0, 0],
                [0,  0, 0, 0, 0],
                [0, -1, 0, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_down2right1 = torch.FloatTensor([
                [0, 0, 0,  0, 0],
                [0, 0, 0,  0, 0],
                [0, 0, 1,  0, 0],
                [0, 0, 0,  0, 0],
                [0, 0, 0, -1, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_down1left2 = torch.FloatTensor([
                [ 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0],
                [ 0, 0, 1, 0, 0],
                [-1, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_down1right2 = torch.FloatTensor([
                [0, 0, 0, 0,  0],
                [0, 0, 0, 0,  0],
                [0, 0, 1, 0,  0],
                [0, 0, 0, 0, -1],
                [0, 0, 0, 0,  0]
            ]).unsqueeze(0).unsqueeze(0)
            
        self.weight_left  = nn.Parameter(data=kernel_left,  requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up    = nn.Parameter(data=kernel_up,    requires_grad=False)
        self.weight_down  = nn.Parameter(data=kernel_down,  requires_grad=False)
        if self.num_regions in [8, 16]:
            self.weight_upleft    = nn.Parameter(data=kernel_upleft,    requires_grad=False)
            self.weight_upright   = nn.Parameter(data=kernel_upright,   requires_grad=False)
            self.weight_downleft  = nn.Parameter(data=kernel_downleft,  requires_grad=False)
            self.weight_downright = nn.Parameter(data=kernel_downright, requires_grad=False)
        if self.num_regions in [16, 24]:
            self.weight_left2       = nn.Parameter(data=kernel_left2,       requires_grad=False)
            self.weight_right2      = nn.Parameter(data=kernel_right2,      requires_grad=False)
            self.weight_up2         = nn.Parameter(data=kernel_up2,         requires_grad=False)
            self.weight_down2       = nn.Parameter(data=kernel_down2,       requires_grad=False)
            self.weight_up2left2    = nn.Parameter(data=kernel_up2left2,    requires_grad=False)
            self.weight_up2right2   = nn.Parameter(data=kernel_up2right2,   requires_grad=False)
            self.weight_down2left2  = nn.Parameter(data=kernel_down2left2,  requires_grad=False)
            self.weight_down2right2 = nn.Parameter(data=kernel_down2right2, requires_grad=False)
        if self.num_regions in [24]:
            self.weight_up2left1    = nn.Parameter(data=kernel_up2left1,    requires_grad=False)
            self.weight_up2right1   = nn.Parameter(data=kernel_up2right1,   requires_grad=False)
            self.weight_up1left2    = nn.Parameter(data=kernel_up1left2,    requires_grad=False)
            self.weight_up1right2   = nn.Parameter(data=kernel_up1right2,   requires_grad=False)
            self.weight_down2left1  = nn.Parameter(data=kernel_down2left1,  requires_grad=False)
            self.weight_down2right1 = nn.Parameter(data=kernel_down2right1, requires_grad=False)
            self.weight_down1left2  = nn.Parameter(data=kernel_down1left2,  requires_grad=False)
            self.weight_down1right2 = nn.Parameter(data=kernel_down1right2, requires_grad=False)
        
        self.pool = nn.AvgPool2d(patch_size)  # Default 4
    
    def __str__(self) -> str:
        return f"spatial_consistency_loss"
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        if self.weight_left.device != input.device:
            self.weight_left = self.weight_left.to(input.device)
        if self.weight_right.device != input.device:
            self.weight_right = self.weight_right.to(input.device)
        if self.weight_up.device != input.device:
            self.weight_up = self.weight_up.to(input.device)
        if self.weight_down.device != input.device:
            self.weight_down = self.weight_down.to(input.device)
        if self.num_regions in [8, 16]:
            if self.weight_upleft.device != input.device:
                self.weight_upleft = self.weight_upleft.to(input.device)
            if self.weight_upright.device != input.device:
                self.weight_upright = self.weight_upright.to(input.device)
            if self.weight_downleft.device != input.device:
                self.weight_downleft = self.weight_downleft.to(input.device)
            if self.weight_downright.device != input.device:
                self.weight_downright = self.weight_downright.to(input.device)
        if self.num_regions in [16, 24]:
            if self.weight_left2.device != input.device:
                self.weight_left2 = self.weight_left2.to(input.device)
            if self.weight_right2.device != input.device:
                self.weight_right2 = self.weight_right2.to(input.device)
            if self.weight_up2.device != input.device:
                self.weight_up2 = self.weight_up2.to(input.device)
            if self.weight_down2.device != input.device:
                self.weight_down2 = self.weight_down2.to(input.device)
            if self.weight_up2left2.device != input.device:
                self.weight_up2left2 = self.weight_up2left2.to(input.device)
            if self.weight_up2right2.device != input.device:
                self.weight_up2right2 = self.weight_up2right2.to(input.device)
            if self.weight_down2left2.device != input.device:
                self.weight_down2left2 = self.weight_down2left2.to(input.device)
            if self.weight_down2right2.device != input.device:
                self.weight_down2right2 = self.weight_down2right2.to(input.device)
        if self.num_regions in [24]:
            if self.weight_up2left1.device != input.device:
                self.weight_up2left1 = self.weight_up2left1.to(input.device)
            if self.weight_up2right1.device != input.device:
                self.weight_up2right1 = self.weight_up2right1.to(input.device)
            if self.weight_up1left2.device != input.device:
                self.weight_up1left2 = self.weight_up1left2.to(input.device)
            if self.weight_up1right2.device != input.device:
                self.weight_up1right2 = self.weight_up1right2.to(input.device)
            if self.weight_down2left1.device != input.device:
                self.weight_down2left1 = self.weight_down2left1.to(input.device)
            if self.weight_down2right1.device != input.device:
                self.weight_down2right1 = self.weight_down2right1.to(input.device)
            if self.weight_down1left2.device != input.device:
                self.weight_down1left2 = self.weight_down1left2.to(input.device)
            if self.weight_down1right2.device != input.device:
                self.weight_down1right2 = self.weight_down1right2.to(input.device)
                
        org_mean     = torch.mean(input,  1, keepdim=True)
        enhance_mean = torch.mean(target, 1, keepdim=True)
        
        org_pool     = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)
        
        d_org_left   = F.conv2d(org_pool, self.weight_left,  padding=1)
        d_org_right  = F.conv2d(org_pool, self.weight_right, padding=1)
        d_org_up     = F.conv2d(org_pool, self.weight_up,    padding=1)
        d_org_down   = F.conv2d(org_pool, self.weight_down,  padding=1)
        if self.num_regions in [8, 16]:
            d_org_upleft    = F.conv2d(org_pool, self.weight_upleft,    padding=1)
            d_org_upright   = F.conv2d(org_pool, self.weight_upright,   padding=1)
            d_org_downleft  = F.conv2d(org_pool, self.weight_downleft,  padding=1)
            d_org_downright = F.conv2d(org_pool, self.weight_downright, padding=1)
        if self.num_regions in [16, 24]:
            d_org_left2       = F.conv2d(org_pool, self.weight_left2,       padding=2)
            d_org_right2      = F.conv2d(org_pool, self.weight_right2,      padding=2)
            d_org_up2         = F.conv2d(org_pool, self.weight_up2,         padding=2)
            d_org_down2       = F.conv2d(org_pool, self.weight_down2,       padding=2)
            d_org_up2left2    = F.conv2d(org_pool, self.weight_up2left2,    padding=2)
            d_org_up2right2   = F.conv2d(org_pool, self.weight_up2right2,   padding=2)
            d_org_down2left2  = F.conv2d(org_pool, self.weight_down2left2,  padding=2)
            d_org_down2right2 = F.conv2d(org_pool, self.weight_down2right2, padding=2)
        if self.num_regions in [24]:
            d_org_up2left1    = F.conv2d(org_pool, self.weight_up2left1,    padding=2)
            d_org_up2right1   = F.conv2d(org_pool, self.weight_up2right1,   padding=2)
            d_org_up1left2    = F.conv2d(org_pool, self.weight_up1left2,    padding=2)
            d_org_up1right2   = F.conv2d(org_pool, self.weight_up1right2,   padding=2)
            d_org_down2left1  = F.conv2d(org_pool, self.weight_down2left1,  padding=2)
            d_org_down2right1 = F.conv2d(org_pool, self.weight_down2right1, padding=2)
            d_org_down1left2  = F.conv2d(org_pool, self.weight_down1left2,  padding=2)
            d_org_down1right2 = F.conv2d(org_pool, self.weight_down1right2, padding=2)
        
        d_enhance_left  = F.conv2d(enhance_pool, self.weight_left,  padding=1)
        d_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        d_enhance_up    = F.conv2d(enhance_pool, self.weight_up,    padding=1)
        d_enhance_down  = F.conv2d(enhance_pool, self.weight_down,  padding=1)
        if self.num_regions in [8, 16]:
            d_enhance_upleft    = F.conv2d(enhance_pool, self.weight_upleft,    padding=1)
            d_enhance_upright   = F.conv2d(enhance_pool, self.weight_upright,   padding=1)
            d_enhance_downleft  = F.conv2d(enhance_pool, self.weight_downleft,  padding=1)
            d_enhance_downright = F.conv2d(enhance_pool, self.weight_downright, padding=1)
        if self.num_regions in [16, 24]:
            d_enhance_left2       = F.conv2d(enhance_pool, self.weight_left2,       padding=2)
            d_enhance_right2      = F.conv2d(enhance_pool, self.weight_right2,      padding=2)
            d_enhance_up2         = F.conv2d(enhance_pool, self.weight_up2,         padding=2)
            d_enhance_down2       = F.conv2d(enhance_pool, self.weight_down2,       padding=2)
            d_enhance_up2left2    = F.conv2d(enhance_pool, self.weight_up2left2,    padding=2)
            d_enhance_up2right2   = F.conv2d(enhance_pool, self.weight_up2right2,   padding=2)
            d_enhance_down2left2  = F.conv2d(enhance_pool, self.weight_down2left2,  padding=2)
            d_enhance_down2right2 = F.conv2d(enhance_pool, self.weight_down2right2, padding=2)
        if self.num_regions in [24]:
            d_enhance_up2left1    = F.conv2d(enhance_pool, self.weight_up2left1,    padding=2)
            d_enhance_up2right1   = F.conv2d(enhance_pool, self.weight_up2right1,   padding=2)
            d_enhance_up1left2    = F.conv2d(enhance_pool, self.weight_up1left2,    padding=2)
            d_enhance_up1right2   = F.conv2d(enhance_pool, self.weight_up1right2,   padding=2)
            d_enhance_down2left1  = F.conv2d(enhance_pool, self.weight_down2left1,  padding=2)
            d_enhance_down2right1 = F.conv2d(enhance_pool, self.weight_down2right1, padding=2)
            d_enhance_down1left2  = F.conv2d(enhance_pool, self.weight_down1left2,  padding=2)
            d_enhance_down1right2 = F.conv2d(enhance_pool, self.weight_down1right2, padding=2)
        
        d_left  = torch.pow(d_org_left  - d_enhance_left,  2)
        d_right = torch.pow(d_org_right - d_enhance_right, 2)
        d_up    = torch.pow(d_org_up    - d_enhance_up,    2)
        d_down  = torch.pow(d_org_down  - d_enhance_down,  2)
        if self.num_regions in [8, 16]:
            d_upleft    = torch.pow(d_org_upleft    - d_enhance_upleft,    2)
            d_upright   = torch.pow(d_org_upright   - d_enhance_upright,   2)
            d_downleft  = torch.pow(d_org_downleft  - d_enhance_downleft,  2)
            d_downright = torch.pow(d_org_downright - d_enhance_downright, 2)
        if self.num_regions in [16, 24]:
            d_left2       = torch.pow(d_org_left2       - d_enhance_left2      , 2)
            d_right2      = torch.pow(d_org_right2      - d_enhance_right2     , 2)
            d_up2         = torch.pow(d_org_up2         - d_enhance_up2        , 2)
            d_down2       = torch.pow(d_org_down2       - d_enhance_down2      , 2)
            d_up2left2    = torch.pow(d_org_up2left2    - d_enhance_up2left2   , 2)
            d_up2right2   = torch.pow(d_org_up2right2   - d_enhance_up2right2  , 2)
            d_down2left2  = torch.pow(d_org_down2left2  - d_enhance_down2left2 , 2)
            d_down2right2 = torch.pow(d_org_down2right2 - d_enhance_down2right2, 2)
        if self.num_regions in [24]:
            d_up2left1    = torch.pow(d_org_up2left1    - d_enhance_up2left1   , 2)
            d_up2right1   = torch.pow(d_org_up2right1   - d_enhance_up2right1  , 2)
            d_up1left2    = torch.pow(d_org_up1left2    - d_enhance_up1left2   , 2)
            d_up1right2   = torch.pow(d_org_up1right2   - d_enhance_up1right2  , 2)
            d_down2left1  = torch.pow(d_org_down2left1  - d_enhance_down2left1 , 2)
            d_down2right1 = torch.pow(d_org_down2right1 - d_enhance_down2right1, 2)
            d_down1left2  = torch.pow(d_org_down1left2  - d_enhance_down1left2 , 2)
            d_down1right2 = torch.pow(d_org_down1right2 - d_enhance_down1right2, 2)
        
        loss = d_left + d_right + d_up + d_down
        if self.num_regions in [8, 16]:
            loss += d_upleft + d_upright + d_downleft + d_downright
        if self.num_regions in [16, 24]:
            loss += (d_left2 + d_right2 + d_up2 + d_down2 +
                     d_up2left2 + d_up2right2 + d_down2left2 + d_down2right2)
        if self.num_regions in [24]:
            loss += (d_up2left1 + d_up2right1 + d_up1left2 + d_up1right2 +
                     d_down2left1 + d_down2right1 + d_down1left2 + d_down1right2)
        
        loss = reduce_loss(loss=loss, reduction=self.reduction)
        return loss

# endregion


# region Misc

# endregion
