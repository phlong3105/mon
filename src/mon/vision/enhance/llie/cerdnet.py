#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements CERDNet (Curve Estimation and Retinex Decomposition
Network) models.
"""

from __future__ import annotations

__all__ = [
    "CERDNet",
]

from typing import Any, Literal

import cv2
import numpy as np
import torch

from mon import core, nn
from mon.core import _callable
from mon.globals import MODELS, Scheme
from mon.nn import functional as F
from mon.vision.enhance import denoise
from mon.vision.enhance.llie import base

console = core.console


# region Module

class LightUpNet(nn.Module):
    
    def __init__(
        self,
        in_channels : int = 3,
        num_channels: int = 32,
        out_channels: int = 3,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,      num_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_channels,     num_channels, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(num_channels,     num_channels, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(num_channels,     num_channels, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1, bias=True)
        self.conv6 = nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1, bias=True)
        self.conv7 = nn.Conv2d(num_channels * 2, out_channels, 3, 1, 1, bias=True)
        self.attn  = nn.Identity()
        self.act   = nn.PReLU()

        # Load weights
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                m.conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "dw_conv"):
                m.dw_conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "pw_conv"):
                m.pw_conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "weight"):
                m.weight.data.normal_(0.0, 0.02)  # 0.02

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x  = input
        l1 = self.act(self.conv1(x))
        l2 = self.act(self.conv2(l1))
        l3 = self.act(self.conv3(l2))
        l4 = self.act(self.conv4(l3))
        l5 = self.act(self.conv5(torch.cat([l3, l4], dim=1)))
        l6 = self.act(self.conv6(torch.cat([l2, l5], dim=1)))
        l  =   F.tanh(self.conv7(torch.cat([l1, l6], dim=1)))
        return l


class ZSN2N(denoise.DenoisingModel):
    """ZS-N2N (Zero-Shot Noise2Noise).
    
    Args:
        channels: The first layer's input channel. Default: ``3`` for RGB image.
        num_channels: Output channels for subsequent layers. Default: ``48``.
        
    See Also: :class:`denoise.DenoisingModel`.
    """
    
    _scheme: list[Scheme] = [Scheme.UNSUPERVISED, Scheme.ZEROSHOT, Scheme.INSTANCE]
    _zoo   : dict = {}
    
    def __init__(
        self,
        channels    : int = 3,
        num_channels: int = 48,
        *args, **kwargs
    ):
        super().__init__(name="zsn2n", *args, **kwargs)
        
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            channels      = self.weights.get("channels"    , channels)
            num_channels  = self.weights.get("num_channels", num_channels)
        
        self._channels    = channels
        self.num_channels = num_channels
        
        # Construct network
        self.conv1 = nn.Conv2d(self.channels,     self.num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(self.num_channels, self.channels,     kernel_size=1)
        self.act   = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # Loss
        self._loss = None
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self._init_weights)
        self.initial_state_dict = self.state_dict()
    
    def _init_weights(self, m: nn.Module):
        pass
    
    # region Forward Pass
    
    def forward_loss(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Symmetry
        noisy1, noisy2 = self.pair_downsampler(input)
        pred1          = self.forward(input=noisy1)
        pred2          = self.forward(input=noisy2)
        #
        noisy_denoised       = self.forward(input)
        denoised1, denoised2 = self.pair_downsampler(noisy_denoised)
        # Loss
        mse_loss  = nn.MSELoss()
        loss_res  = 0.5 * (mse_loss(noisy1, pred2)    + mse_loss(noisy2, pred1))
        loss_cons = 0.5 * (mse_loss(pred1, denoised1) + mse_loss(pred2, denoised2))
        loss      = loss_res + loss_cons
        # loss      = nn.reduce_loss(loss=loss, reduction="mean")
        return noisy_denoised, loss
    
    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> torch.Tensor:
        x = input
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        y = self.conv3(x)
        if self.predicting:
            y = torch.clamp(y, 0, 1)
        return y
    
    @staticmethod
    def pair_downsampler(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c       = input.shape[1]
        filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(input.device)
        filter1 = filter1.repeat(c, 1, 1, 1)
        filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(input.device)
        filter2 = filter2.repeat(c, 1, 1, 1)
        output1 = F.conv2d(input, filter1, stride=2, groups=c)
        output2 = F.conv2d(input, filter2, stride=2, groups=c)
        return output1, output2
    
    # endregion
    
    # region Training
    
    def fit_one(
        self,
        input        : torch.Tensor | np.ndarray,
        max_epochs   : int   = 3000,
        lr           : float = 0.001,
        step_size    : int   = 1000,
        gamma        : float = 0.5,
        reset_weights: bool  = True,
    ) -> torch.Tensor:
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
            step_size: Period of learning rate decay. Default: ``1000``.
            gamma: A multiplicative factor of learning rate decay. Default: ``0.5``.
            reset_weights: Whether to reset the weights before training. Default: ``True``.
        
        Returns:
            The denoised image.
        """
        # Initialize training components
        self.train()
        if reset_weights:
            self.load_state_dict(self.initial_state_dict)
        if isinstance(self.optims, dict):
            optimizer = self.optims.get("optimizer", None)
            scheduler = self.optims.get("scheduler", None)
        else:
            optimizer = nn.Adam(self.parameters(), lr=lr)
            scheduler = nn.StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        # Prepare input
        if isinstance(input, np.ndarray):
            input = core.to_image_tensor(input, False, True)
        input = input.to(self.device)
        assert input.shape[0] == 1
        
        # Training loop
        if self.verbose:
            with core.get_progress_bar() as pbar:
                for _ in pbar.track(
                    sequence    = range(max_epochs),
                    total       = max_epochs,
                    description = f"[bright_yellow] Training"
                ):
                    _, loss = self.forward_loss(input=input, target=None)
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    scheduler.step()
        else:
            for _ in range(max_epochs):
                _, loss = self.forward_loss(input=input, target=None)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                scheduler.step()
        
        # Post-processing
        self.eval()
        pred = self.forward(input=input)
        # with torch.no_grad():
        #    pred = torch.clamp(self.forward(input=input), 0, 1)
        self.train()
        
        return pred
    
    # endregion
    
# endregion


# region Loss

class Loss(nn.Loss):
    
    def __init__(
        self,
        exp_patch_size      : int   = 16,
        exp_mean_val        : float = 0.6,
        spa_num_regions     : Literal[4, 8, 16, 24] = 8,
        spa_patch_size      : int   = 4,
        gaussian_kernel_size: int   = 5,
        gaussian_padding    : int   = 2,
        gaussian_sigma      : float = 3,
        weight_col          : float = 5,
        weight_exp          : float = 10,
        weight_spa          : float = 1,
        weight_tvA          : float = 1600,
        weight_illumination : float = 1,
        weight_reflectance  : float = 1,
        weight_noise        : float = 1,
        reduction           : str   = "mean",
        verbose             : bool  = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        kx     = cv2.getGaussianKernel(gaussian_kernel_size, gaussian_sigma)
        ky     = cv2.getGaussianKernel(gaussian_kernel_size, gaussian_sigma)
        kernel = np.multiply(kx, np.transpose(ky))
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        
        self.gaussian_kernel     = kernel
        self.gaussian_padding    = gaussian_padding
        self.verbose             = verbose
        
        self.weight_col          = weight_col
        self.weight_exp          = weight_exp
        self.weight_spa          = weight_spa
        self.weight_tvA          = weight_tvA
        self.weight_illumination = weight_illumination
        self.weight_reflectance  = weight_reflectance
        self.weight_noise        = weight_noise
        
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
        self.loss_tvA = nn.IlluminationSmoothnessLoss(reduction=reduction)

    def forward(
        self,
        input       : torch.Tensor,
        target      : torch.Tensor,
        lightup     : torch.Tensor = None,
        illumination: torch.Tensor = None,
        reflectance : torch.Tensor = None,
        noise       : torch.Tensor = None,
        **_
    ) -> torch.Tensor:
        if self.gaussian_kernel.device != input.device:
            self.gaussian_kernel = self.gaussian_kernel.to(input.device)
        
        if lightup is not None:
            loss_col  = self.loss_col(input=target)               if self.weight_col > 0 else 0
            loss_exp  = self.loss_exp(input=target)               if self.weight_exp > 0 else 0
            loss_spa  = self.loss_spa(input=target, target=input) if self.weight_spa > 0 else 0
            loss_tvA  = self.loss_tvA(input=lightup)              if self.weight_tvA > 0 else 0
            loss = (
                  self.weight_col * loss_col
                + self.weight_exp * loss_exp
                + self.weight_tvA * loss_tvA
                + self.weight_spa * loss_spa
            )
            if self.verbose:
                console.log(f"{self.loss_col.__str__():<30}: {loss_col}")
                console.log(f"{self.loss_exp.__str__():<30}: {loss_exp}")
                console.log(f"{self.loss_spa.__str__():<30}: {loss_spa}")
                console.log(f"{self.loss_tvA.__str__():<30}: {loss_tvA}")
        elif illumination is not None and reflectance is not None and noise is not None:
            loss_reconstruction = self.reconstruction_loss(input, illumination, reflectance, noise)
            loss_illumination   = self.illumination_smooth_loss(input, illumination)
            loss_reflectance    = self.reflectance_smooth_loss(input, illumination, reflectance)
            loss_noise          = self.noise_loss(illumination, noise)
            loss = (
                loss_reconstruction
                + self.weight_illumination * loss_illumination
                + self.weight_reflectance  * loss_reflectance
                + self.weight_noise        * loss_noise
            )
            if self.verbose:
                console.log(f"{'reconstruction loss':<30}: {loss_reconstruction}")
                console.log(f"{'illumination smooth loss':<30}: {loss_illumination}")
                console.log(f"{'reflectance smooth loss':<30}: {loss_reflectance}")
                console.log(f"{'noise loss':<30}: {loss_noise}")
        else:
            raise ValueError
        
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
        # loss = nn.MAELoss()(input=reconstructed_image, target=image)
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
        # loss = loss_h.sum() + loss_w.sum() + nn.MAELoss()(input=illumination, target=max_rgb)
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
        # loss = loss_h.sum() + loss_w.sum() + 1.0 * nn.MAELoss()(input=reflectance, target=reference_reflectance)
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

@MODELS.register(name="cerdnet")
class CERDNet(base.LowLightImageEnhancementModel):
    """CERDNet (Curve Estimation and Retinex Decomposition Network) model.
    
    Args:
        channels: The first layer's input channel. Default: ``3`` for RGB image.
        num_channels: Output channels for subsequent layers. Default: ``64``.
        num_iters: The number of convolutional layers in the model.
            Default: ``8``.
        scale_factor: Downsampling/upsampling ratio. Defaults: ``1``.
        gamma: Gamma value for dark channel prior. Default: ``2.8``.
        unsharp_sigma: Unsharp sigma value. Default: ``None``.
        
    See Also: :class:`base.LowLightImageEnhancementModel`
    """
    
    _scheme: list[Scheme] = [Scheme.UNSUPERVISED, Scheme.ZEROSHOT, Scheme.INSTANCE]
    _zoo   : dict = {}

    def __init__(
        self,
        channels    : int   = 3,
        num_channels: int   = 32,
        num_iters   : int   = 8,
        scale_factor: int   = 1,
        gamma       : float = 0.4,
        pretrain    : bool  = False,
        weights     : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            name     = "rdcenet",
            channels = channels,
            weights  = weights,
            *args, **kwargs
        )
        
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            channels     = self.weights.get("channels"    , channels)
            num_channels = self.weights.get("num_channels", num_channels)
            num_iters    = self.weights.get("num_iters"   , num_iters)
            scale_factor = self.weights.get("scale_factor", scale_factor)
            gamma        = self.weights.get("gamma"       , gamma)
        
        self._channels     = channels
        self.num_channels  = num_channels
        self.num_iters     = num_iters
        self.scale_factor  = scale_factor
        self.gamma         = gamma
        self.pretrain      = pretrain
        
        # Construct model
        self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
        # Light-up Net
        self.lightup_net = LightUpNet(in_channels=self.channels, num_channels=self.num_channels, out_channels=3)
        # Illumination Net
        self.illumination_net = nn.Sequential(
            nn.Conv2d(self.channels, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
        )
        # Reflectance Net
        self.reflectance_net = nn.Sequential(
            nn.Conv2d(self.channels, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1)
        )
        # Noise Net
        self.noise_net = nn.Sequential(
            nn.Conv2d(self.channels, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1)
        )
        # Denoise Net
        self.zsn2n = ZSN2N(channels=self.channels, num_channels=48)
        
        # Loss
        self._loss = Loss()
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self._init_weights)
        self.initial_state_dict = self.state_dict()
        
    def _init_weights(self, m: nn.Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                m.conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "dw_conv"):
                m.dw_conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "pw_conv"):
                m.pw_conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "weight"):
                m.weight.data.normal_(0.0, 0.02)  # 0.02
    
    def forward_loss(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pred             = self.forward(input=input, *args, **kwargs)
        lightup          = pred[0]
        illumination     = pred[1]
        illumination_hat = pred[2]
        reflectance      = pred[3]
        noise            = pred[4]
        target           = pred[5]
        loss = self._loss(
            input        = input,
            target       = target,
            lightup      = lightup,
            illumination = illumination,
            reflectance  = reflectance,
            noise        = noise,
        )
        return target, loss
        
    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = input
        
        # Downsampling
        x_down = x
        if self.scale_factor != 1:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode="bilinear")
        
        if self.pretrain:
            l = self.lightup_net(x_down)
            # Upsampling
            if self.scale_factor != 1:
                l = self.upsample(l)
            # Enhancement
            y = x
            for _ in range(self.num_iters):
                y = y + l * (torch.pow(y, 2) - y)
            return l, None, None, None, None, y
        else:
            # Illumination Map
            i = self.illumination_net(x_down)
            # Reflectance Map
            r = self.reflectance_net(x_down)
            # Noise Map
            n = self.noise_net(x_down)
            # Upsampling
            if self.scale_factor != 1:
                i = self.upsample(i)
                r = self.upsample(r)
                n = self.upsample(n)
            # Enhancement
            i_hat = torch.pow(i, self.gamma)
            y     = i_hat * ((x - n) / i)
            return None, i, i_hat, r, n, y
    
    # region Training
    
    def fit_one(
        self,
        input        : torch.Tensor | np.ndarray,
        max_epochs   : int   = 1000,
        lr           : float = 0.001,
        step_size    : int   = 1000,
        gamma        : float = 0.5,
        reset_weights: bool  = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Train the model with a single sample. This method is used for any
        learning scheme performed on one single instance such as online learning,
        zero-shot learning, one-shot learning, etc.
        
        Note:
            To use this method, the model must implement the optimizer and/or scheduler.
        
        Args:
            input: The input image tensor.
            max_epochs: Maximum number of epochs. Default: ``3000``.
            lr: Learning rate. Default: ``0.001``.
            step_size: Period of learning rate decay. Default: ``1000``.
            gamma: A multiplicative factor of learning rate decay. Default: ``0.5``.
            reset_weights: Whether to reset the weights before training. Default: ``True``.
        """
        # Initialize training components
        if reset_weights:
            self.load_state_dict(self.initial_state_dict)
        if isinstance(self.optims, dict):
            optimizer1 = self.optims.get("optimizer", None)
            optimizer2 = self.optims.get("optimizer", None)
            scheduler2 = self.optims.get("scheduler", None)
        else:
            optimizer1 = nn.Adam(
                list(self.illumination_net.parameters())
                + list(self.reflectance_net.parameters())
                + list(self.noise_net.parameters()),
                lr=lr
            )
            optimizer2 = nn.Adam(self.zsn2n.parameters(), lr=lr)
            scheduler2 = nn.StepLR(optimizer1, step_size=step_size, gamma=gamma)
        
        # Prepare input
        if isinstance(input, np.ndarray):
            input = core.to_image_tensor(input, False, True)
        input = input.to(self.device)
        assert input.shape[0] == 1
        
        # Pre-processing
        lightup       = self.lightup_net(input)
        lightup_input = input
        for _ in range(self.num_iters):
            lightup_input = lightup_input + lightup * (torch.pow(lightup_input, 2) - lightup_input)
        
        # Training loop 1
        self.pretrain = False
        self.train()
        self.lightup_net.eval()
        self.illumination_net.train()
        self.reflectance_net.train()
        self.noise_net.train()
        self.zsn2n.eval()
        if self.verbose:
            with core.get_progress_bar() as pbar:
                for _ in pbar.track(
                    sequence    = range(max_epochs),
                    total       = max_epochs,
                    description = f"[bright_yellow] Training"
                ):
                    _, loss = self.forward_loss(input=lightup_input, target=None)
                    optimizer1.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer1.step()
        else:
            for _ in range(max_epochs):
                _, loss = self.forward_loss(input=lightup_input, target=None)
                optimizer1.zero_grad()
                loss.backward(retain_graph=True)
                optimizer1.step()

        self.eval()
        pred             = self.forward(input=lightup_input)
        illumination     = pred[1]
        illumination_hat = pred[2]
        reflectance      = pred[3]
        noise            = pred[4]
        relight          = pred[5]

        # Training loop 2
        self.train()
        self.lightup_net.eval()
        self.illumination_net.eval()
        self.reflectance_net.eval()
        self.noise_net.eval()
        self.zsn2n.train()
        if self.verbose:
            with core.get_progress_bar() as pbar:
                for _ in pbar.track(
                    sequence    = range(max_epochs),
                    total       = max_epochs,
                    description = f"[bright_yellow] Training"
                ):
                    _, loss = self.zsn2n.forward_loss(input=relight, target=None)
                    optimizer2.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer2.step()
                    scheduler2.step()
        else:
            for _ in range(max_epochs):
                _, loss = self.zsn2n.forward_loss(input=relight, target=None)
                optimizer2.zero_grad()
                loss.backward(retain_graph=True)
                optimizer2.step()
                scheduler2.step()
        
        # Post-processing
        self.eval()
        denoised = self.zsn2n(input=input)

        return (
            lightup,
            lightup_input,
            illumination,
            illumination_hat,
            reflectance,
            noise,
            relight,
            denoised
        )
    
    # endregion
    
# endregion
