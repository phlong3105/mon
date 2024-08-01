#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements RRDNet (Zero-Shot Restoration of Underexposed Images
via Robust Retinex Decomposition) models.
"""

from __future__ import annotations

__all__ = [
    "RRDNet",
]

from typing import Any, Literal

import cv2
import numpy as np
import torch

from mon import core, nn
from mon.core import _callable
from mon.globals import MODELS, Scheme
from mon.nn import functional as F
from mon.vision.enhance.llie import base

console = core.console


# region Loss

# noinspection PyMethodMayBeStatic
class Loss(nn.Loss):
    
    def __init__(
        self,
        gaussian_kernel_size: int   = 5,
        gaussian_padding    : int   = 2,
        gaussian_sigma      : float = 3,
        illumination_weight : float = 1,
        reflectance_weight  : float = 1,
        noise_weight        : float = 5000,
        reduction           : Literal["none", "mean", "sum"] = "mean",
        *args, **kwargs
    ):
        super().__init__(reduction=reduction, *args, **kwargs)
        kx     = cv2.getGaussianKernel(gaussian_kernel_size, gaussian_sigma)
        ky     = cv2.getGaussianKernel(gaussian_kernel_size, gaussian_sigma)
        kernel = np.multiply(kx, np.transpose(ky))
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.gaussian_kernel     = kernel
        self.gaussian_padding    = gaussian_padding
        self.illumination_weight = illumination_weight
        self.reflectance_weight  = reflectance_weight
        self.noise_weight        = noise_weight
        
    def forward(
        self,
        image       : torch.Tensor,
        illumination: torch.Tensor,
        reflectance : torch.Tensor,
        noise       : torch.Tensor,
    ) -> torch.Tensor:
        if self.gaussian_kernel.device != image.device:
            self.gaussian_kernel = self.gaussian_kernel.to(image.device)
        
        loss_reconstruction = self.reconstruction_loss(image, illumination, reflectance, noise)
        loss_illumination   = self.illumination_smooth_loss(image, illumination)
        loss_reflectance    = self.reflectance_smooth_loss(image, illumination, reflectance)
        loss_noise          = self.noise_loss(illumination, noise)
        loss = (
            loss_reconstruction
            + self.illumination_weight * loss_illumination
            + self.reflectance_weight  * loss_reflectance
            + self.noise_weight        * loss_noise
        )
        return loss
    
    def reconstruction_loss(
        self,
        image       : torch.Tensor,
        illumination: torch.Tensor,
        reflectance : torch.Tensor,
        noise       : torch.Tensor,
    ) -> torch.Tensor:
        reconstructed_image = illumination * reflectance + noise
        loss = torch.norm(image - reconstructed_image, 1)
        return loss
    
    def illumination_smooth_loss(
        self,
        image       : torch.Tensor,
        illumination: torch.Tensor,
    ) -> torch.Tensor:
        gray_tensor = 0.299 * image[0, 0, :, :] + 0.587 * image[0, 1, :, :] + 0.114 * image[0, 2, :, :]
        max_rgb, _  = torch.max(image, 1)
        max_rgb     = max_rgb.unsqueeze(1)
        gradient_gray_h, gradient_gray_w = self.gradient(gray_tensor.unsqueeze(0).unsqueeze(0))
        gradient_illu_h, gradient_illu_w = self.gradient(illumination)
        weight_h = 1 / (F.conv2d(gradient_gray_h, weight=self.gaussian_kernel, padding=self.gaussian_padding) + 0.0001)
        weight_w = 1 / (F.conv2d(gradient_gray_w, weight=self.gaussian_kernel, padding=self.gaussian_padding) + 0.0001)
        weight_h.detach()
        weight_w.detach()
        loss_h   = weight_h * gradient_illu_h
        loss_w   = weight_w * gradient_illu_w
        max_rgb.detach()
        loss     = loss_h.sum() + loss_w.sum() + torch.norm(illumination - max_rgb, 1)
        return loss
    
    def reflectance_smooth_loss(
        self,
        image       : torch.Tensor,
        illumination: torch.Tensor,
        reflectance : torch.Tensor
    ) -> torch.Tensor:
        gray_tensor = 0.299 * image[0, 0, :, :] + 0.587 * image[0, 1, :, :] + 0.114 * image[0, 2, :, :]
        gradient_gray_h,    gradient_gray_w    = self.gradient(gray_tensor.unsqueeze(0).unsqueeze(0))
        gradient_reflect_h, gradient_reflect_w = self.gradient(reflectance)
        weight = 1 / (illumination * gradient_gray_h * gradient_gray_w + 0.0001)
        weight = self.normalize(weight)
        weight.detach()
        loss_h = weight * gradient_reflect_h
        loss_w = weight * gradient_reflect_w
        reference_reflectance = image / illumination
        reference_reflectance.detach()
        loss   = loss_h.sum() + loss_w.sum() + 1.0 * torch.norm(reference_reflectance - reflectance, 1)
        return loss
        
    def noise_loss(
        self,
        illumination: torch.Tensor,
        noise       : torch.Tensor
    ) -> torch.Tensor:
        weight_illu = illumination
        weight_illu.detach()
        loss = weight_illu * noise
        loss = torch.norm(loss, 2)
        return loss
        
    def gradient(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        height      = image.size(2)
        width       = image.size(3)
        gradient_h  = (image[:, :, 2:, :] - image[:, :, :height - 2, :]).abs()
        gradient_w  = (image[:, :, :, 2:] - image[:, :, :, :width - 2]).abs()
        gradient_h  = F.pad(gradient_h, [0, 0, 1, 1], "replicate")
        gradient_w  = F.pad(gradient_w, [1, 1, 0, 0], "replicate")
        gradient2_h = (image[:, :, 4:, :] - image[:, :, :height - 4, :]).abs()
        gradient2_w = (image[:, :, :, 4:] - image[:, :, :, :width - 4]).abs()
        gradient2_h = F.pad(gradient2_h, [0, 0, 2, 2], "replicate")
        gradient2_w = F.pad(gradient2_w, [2, 2, 0, 0], "replicate")
        return gradient_h * gradient2_h, gradient_w * gradient2_w
        
    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        min_value = image.min()
        max_value = image.max()
        return (image - min_value) / (max_value - min_value)
    
    def gaussian_blur3(self, image: torch.Tensor) -> torch.Tensor:
        slice1 = F.conv2d(image[:, 0, :, :].unsqueeze(1), weight=self.gaussian_kernel, padding=self.gaussian_padding)
        slice2 = F.conv2d(image[:, 1, :, :].unsqueeze(1), weight=self.gaussian_kernel, padding=self.gaussian_padding)
        slice3 = F.conv2d(image[:, 2, :, :].unsqueeze(1), weight=self.gaussian_kernel, padding=self.gaussian_padding)
        x      = torch.cat([slice1, slice2, slice3], dim=1)
        return x

# endregion


# region Model

# noinspection PyMethodMayBeStatic
@MODELS.register(name="rrdnet", arch="rrdnet")
class RRDNet(base.LowLightImageEnhancementModel):
    """RRDNet (Zero-Shot Restoration of Underexposed Images via Robust Retinex
    Decomposition) model.
    
    Args:
        in_channels: The first layer's input channel. Default: ``3`` for RGB image.
        gamma: The gamma value for the illumination. Default: ``0.4``.
        
    References:
        `<https://github.com/aaaaangel/RRDNet>`__

    See Also: :class:`base.LowLightImageEnhancementModel`
    """
    
    arch   : str  = "rrdnet"
    schemes: list[Scheme] = [Scheme.UNSUPERVISED, Scheme.ZERO_SHOT, Scheme.INSTANCE]
    zoo    : dict = {}
    
    def __init__(
        self,
        in_channels: int   = 3,
        gamma      : float = 0.4,
        weights    : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            name        = "rrdnet",
            in_channels = in_channels,
            weights     = weights,
            *args, **kwargs
        )
        
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            in_channels = self.weights.get("in_channels", in_channels)
            gamma       = self.weights.get("gamma"      , gamma)
        self.in_channels = in_channels
        self.gamma       = gamma
        
        # Construct model
        self.illumination_net = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
        )
        self.reflectance_net = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1)
        )
        self.noise_net = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1)
        )
       
        # Loss
        self._loss = Loss()
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        self.initial_state_dict = self.state_dict()
        
    def init_weights(self, m: nn.Module):
        pass
    
    # region Forward Pass
    
    def forward_loss(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pred = self.forward(input=input, *args, **kwargs)
        illumination          = pred[0]
        adjusted_illumination = pred[1]
        reflectance           = pred[2]
        noise                 = pred[3]
        relight               = pred[4]
        loss = self.loss(
            image        = input,
            illumination = illumination,
            reflectance  = reflectance,
            noise        = noise,
        ) if self.loss else None
        return pred, loss
    
    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass. This is the primary :meth:`forward` function of the
        model. It supports augmented inference. In this function, we perform
        test-time augmentation and pass the transformed input to
        :meth:`forward_once()`.

        Args:
            input: An input of shape :math:`[N, C, H, W]`.
            augment: If ``True``, perform test-time augmentation. Usually used
                in predicting phase. Default: ``False``.
            profile: If ``True``, measure processing time. Usually used in
                predicting phase. Default: ``False``.
            out_index: If the model produces multiple outputs, return the one
                with the index :param:`out_index`. Usually used in predicting
                phase. Default: ``-1`` means the last one.
            
        Return:
            A tuple including: illumination, adjusted illumination, reflectance,
                noise, and relighted image.
        """
        x     = input
        s     = torch.sigmoid(self.illumination_net(x))
        r     = torch.sigmoid(self.reflectance_net(x))
        n     =    torch.tanh(self.noise_net(x))
        s_hat = torch.pow(s, self.gamma)
        y     = s_hat * ((x - n) / s)
        if self.predicting:
            n = self.normalize(n)
            y = torch.clamp(y, min=0, max=1)
        return s, s_hat, r, n, y
    
    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        min_value = image.min()
        max_value = image.max()
        return (image - min_value) / (max_value - min_value)
    
    # endregion
    
    # region Training
    
    def fit_one(
        self,
        input        : torch.Tensor | np.ndarray,
        max_epochs   : int   = 1000,
        lr           : float = 0.001,
        reset_weights: bool  = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Train the model with a single sample. This method is used for any
        learning scheme performed on one single instance such as online learning,
        zero-shot learning, one-shot learning, etc.
        
        Note:
            In order to use this method, the model must implement the optimizer
            and/or scheduler.
        
        Args:
            input: The input image tensor.
            max_epochs: Maximum number of epochs. Default: ``3000``.
            lr: Learning rate. Default: ``0.001``.
            reset_weights: Whether to reset the weights before training. Default: ``True``.
        """
        # Initialize training components
        self.train()
        if reset_weights:
            self.load_state_dict(self.initial_state_dict)
        if isinstance(self.optims, dict):
            optimizer = self.optims.get("optimizer", None)
        else:
            optimizer = nn.Adam(self.parameters(), lr=lr)
        
        # Prepare input
        if isinstance(input, np.ndarray):
            input = core.to_image_tensor(input, False, True)
        input = input.to(self.device)
        assert input.shape[0] == 1
        
        # Training loop
        with core.get_progress_bar() as pbar:
            for _ in pbar.track(
                sequence    = range(max_epochs),
                total       = max_epochs,
                description = f"[bright_yellow] Training"
            ):
                _, loss = self.forward_loss(input=input, target=None)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        # Post-processing
        self.eval()
        pred = self.forward(input=input)
        self.train()
        
        return pred
    
    # endregion
    
# endregion


# region Main

def run_rrdnet():
    path   = core.Path("./data/00691.png")
    image  = cv2.imread(str(path))
    device = torch.device("cuda:0")
    net    = RRDNet(channels=3).to(device)
    pred   = net.fit_one(image)
    #
    illumination          = pred[0]
    adjusted_illumination = pred[1]
    reflectance           = pred[2]
    noise                 = pred[3]
    relight               = pred[4]
    #
    illumination          = torch.concat([illumination, illumination, illumination], dim=1)
    adjusted_illumination = torch.concat([adjusted_illumination, adjusted_illumination, adjusted_illumination], dim=1)
    #
    illumination          = core.to_image_nparray(illumination         , False, True)
    adjusted_illumination = core.to_image_nparray(adjusted_illumination, False, True)
    reflectance           = core.to_image_nparray(reflectance          , False, True)
    noise                 = core.to_image_nparray(noise                , False, True)
    relight               = core.to_image_nparray(relight              , False, True)
    cv2.imshow("Image"                , image)
    cv2.imshow("Illumination"         , illumination)
    cv2.imshow("Adjusted Illumination", adjusted_illumination)
    cv2.imshow("reflectance"          , reflectance)
    cv2.imshow("Noise"                , noise)
    cv2.imshow("Relight"              , relight)
    cv2.waitKey(0)


if __name__ == "__main__":
    run_rrdnet()

# endregion
