#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = [
    "EdgeLoss",
    "L1Loss",
    "PerceptualLoss",
    "SSIM",
]

import torch.nn as nn
from torch import Tensor

from .loss_utils import *
from .vgg_arch import VGGFeatureExtractor

_reduction_modes = ["none", "mean", "sum"]


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, loss_weight: float = 1.0, reduction: str = "mean"):
        super().__init__()
        if reduction not in _reduction_modes:
            raise ValueError(
                f'Unsupported reduction mode: {reduction}. '
                f'Supported ones are: {_reduction_modes}'
            )
        self.loss_weight = loss_weight
        self.reduction = reduction

    # --- Callable & Context Manager ---
    def forward(self, pred: Tensor, target: Tensor, weight: Tensor | None = None, **kwargs) -> Tensor:
        """

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


class EdgeLoss(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, loss_weight: float = 1.0, reduction: str = "mean"):
        super().__init__()
        k = torch.Tensor([[0.05, 0.25, 0.4, 0.25, 0.05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3, 1, 1, 1)
        self.weight = loss_weight

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        loss = mse_loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss * self.weight

    def conv_gauss(self, image: Tensor) -> Tensor:
        n_channels, _, kw, kh = self.kernel.shape
        image = F.pad(image, (kw // 2, kh // 2, kw // 2, kh // 2), mode="replicate")
        return F.conv2d(image, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current: Tensor) -> Tensor:
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        layer_weights,
        vgg_type: str = "vgg19",
        use_input_norm: bool = True,
        range_norm: bool = True,
        perceptual_weight: float = 1.0,
        style_weight: float = 0.0,
        criterion: str = "l1"
    ):
        super().__init__()

        # Assign attributes
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm
        )

        self.criterion_type = criterion
        if self.criterion_type == "l1":
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == "mse":
            self.criterion = torch.nn.MSELoss(reduction="mean")
        elif self.criterion_type == "fro":
            self.criterion = None
        else:
            raise NotImplementedError(f"{criterion} criterion has not been supported.")

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor, gt: Tensor) -> tuple[Tensor, Tensor]:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss


class SSIM(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        window_size: int = 11,
        size_average: bool = True,
        weight: float = 1.0
    ):
        super().__init__()

        # Assign attributes
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.weight = weight

    # --- Callable & Context Manager ---
    def forward(self, img1: Tensor, img2: Tensor) -> Tensor:
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return (1.0 - map_ssim(img1, img2, window, self.window_size, channel, self.size_average)) * self.weight
