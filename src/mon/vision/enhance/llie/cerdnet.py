#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements CERDNet (Curve Estimation and Retinex Decomposition
Network) models.
"""

from __future__ import annotations

__all__ = [
    "CENet",
    "CERDNet",
]

from typing import Any, Literal

import cv2
import numpy as np
import torch

from mon import core, nn, proc
from mon.core import _callable
from mon.globals import MODELS, Scheme
from mon.nn import functional as F
from mon.vision.enhance import denoise
from mon.vision.enhance.llie import base

console = core.console


# region Loss

class RDLoss(nn.Loss):
    
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


# region Module

@MODELS.register(name="cenet")
class CENet(base.LowLightImageEnhancementModel):
    """CENet (Curve Estimation Network) model.
    
    Args:
        channels: The first layer's input channel. Default: ``3`` for RGB image.
        num_channels: The number of input and output channels for subsequent
            layers. Default: ``32``.
        num_iters: The number of convolutional layers in the model.
            Default: ``8``.
        scale_factor: Downsampling/upsampling ratio. Defaults: ``1``.
        
    References:
        `<https://github.com/Li-Chongyi/Zero-DCE_extension>`__

    See Also: :class:`base.LowLightImageEnhancementModel`
    """
    
    _scheme: list[Scheme] = [Scheme.UNSUPERVISED, Scheme.ZEROSHOT]
    _zoo   : dict = {}
    
    def __init__(
        self,
        channels    : int   = 3,
        num_channels: int   = 32,
        num_iters   : int   = 8,
        scale_factor: float = 1.0,
        gamma       : float = 2.8,
        pretrain    : bool  = False,
        weights     : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            name     = "cenet",
            channels = channels,
            weights  = weights,
            *args, **kwargs
        )
        assert num_iters <= 8
        
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            channels     = self.weights.get("channels",     channels)
            num_channels = self.weights.get("num_channels", num_channels)
            num_iters    = self.weights.get("num_iters",    num_iters)
            scale_factor = self.weights.get("scale_factor", scale_factor)
            gamma        = self.weights.get("gamma",        gamma)
        
        self._channels    = channels
        self.num_channels = num_channels
        self.num_iters    = num_iters
        self.scale_factor = scale_factor
        self.gamma        = gamma
        self.pretrain     = pretrain
        
        # Construct model
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        self.relu     = nn.PReLU()
        self.e_conv1  = nn.Conv2d(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
        self.e_conv2  = nn.Conv2d(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
        self.e_conv3  = nn.Conv2d(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
        self.e_conv4  = nn.Conv2d(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
        self.e_conv5  = nn.Conv2d(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
        self.e_conv6  = nn.Conv2d(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
        self.e_conv7  = nn.Conv2d(in_channels=self.num_channels * 2, out_channels=24,                kernel_size=3, stride=1, padding=1)
        
        # Loss
        self.loss_spa = nn.SpatialConsistencyLoss(reduction="mean")
        self.loss_exp = nn.ExposureControlLoss(reduction="mean", patch_size=16, mean_val=0.6)
        self.loss_col = nn.ColorConstancyLoss(reduction="mean")
        self.loss_tv  = nn.TotalVariationALoss(reduction="mean")
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self._init_weights)
    
    def _init_weights(self, m: torch.nn.Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                m.conv.weight.data.normal_(0.0, 0.02)
            else:
                m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    
    def forward_loss(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        lightup, lightup_image = self.forward(input=input, *args, **kwargs)
        loss_spa = self.loss_spa(input=lightup_image, target=input)
        loss_exp = self.loss_exp(input=lightup_image)
        loss_col = self.loss_col(input=lightup_image)
        loss_tv  = self.loss_tv(input=lightup)
        loss = (
              1.0   * loss_spa
            + 10.0  * loss_exp
            + 5.0   * loss_col
            + 200.0 * loss_tv
        )
        return lightup_image, loss
    
    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = input

        x_down = x
        if self.scale_factor != 1:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode="bilinear")
        
        l1 = self.relu(self.e_conv1(x_down))
        l2 = self.relu(self.e_conv2(l1))
        l3 = self.relu(self.e_conv3(l2))
        l4 = self.relu(self.e_conv4(l3))
        l5 = self.relu(self.e_conv5(torch.cat([l3, l4], 1)))
        l6 = self.relu(self.e_conv6(torch.cat([l2, l5], 1)))
        l  =    F.tanh(self.e_conv7(torch.cat([l1, l6], 1)))
       
        if self.scale_factor != 1:
            l = self.upsample(l)
        
        ls = torch.split(l, 3, dim=1)
        if self.pretrain:
            y = x
            for i in range(0, self.num_iters):
                y = y + ls[i] * (torch.pow(y, 2) - y)
        else:
            y = x
            g = proc.get_guided_brightness_enhancement_map_prior(x, self.gamma, 9)
            for i in range(0, self.num_iters):
                b = y * (1 - g)
                d = y * g
                y = b + d + ls[i] * (torch.pow(d, 2) - d)
        return l, y
    

class RDNet(base.LowLightImageEnhancementModel):
    """RDNet (Retinex Decomposition Network) model.
    
    Args:
        channels: The first layer's input channel. Default: ``3`` for RGB image.
        gamma: The gamma value for the illumination. Default: ``0.4``.
        
    References:
        `<https://github.com/aaaaangel/RRDNet>`__

    See Also: :class:`base.LowLightImageEnhancementModel`
    """
    
    _scheme: list[Scheme] = [Scheme.UNSUPERVISED, Scheme.ZEROSHOT, Scheme.INSTANCE]
    _zoo   : dict = {}
    
    def __init__(
        self,
        channels    : int   = 3,
        num_channels: int   = 16,
        scale_factor: float = 1.0,
        gamma       : float = 0.4,
        weights     : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            name     = "rdnet",
            channels = channels,
            weights  = weights,
            *args, **kwargs
        )
        
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            channels     = self.weights.get("channels",     channels)
            scale_factor = self.weights.get("scale_factor", scale_factor)
            gamma        = self.weights.get("gamma",        gamma)
        
        self._channels    = channels
        self.scale_factor = scale_factor
        self.gamma        = gamma
        
        # Construct model
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        self.illumination_net = nn.Sequential(
            nn.Conv2d(self.channels + 1, num_channels * 2, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(num_channels * 2, num_channels * 3, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(num_channels * 3, num_channels * 4, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(num_channels * 4, num_channels * 2, 3, 1, 1),
            nn.PReLU(),
            # nn.Conv2d(num_channels * 3, num_channels * 2, 3, 1, 1),
            # nn.PReLU(),
            nn.Conv2d(num_channels * 2, 1, 3, 1, 1),
        )
        self.reflectance_net = nn.Sequential(
            nn.Conv2d(self.channels + 1, num_channels * 2, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(num_channels * 2, num_channels * 3, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(num_channels * 3, num_channels * 4, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(num_channels * 4, num_channels * 2, 3, 1, 1),
            nn.PReLU(),
            # nn.Conv2d(num_channels * 3, num_channels * 2, 3, 1, 1),
            # nn.PReLU(),
            nn.Conv2d(num_channels * 2, 3, 3, 1, 1),
        )
        self.noise_net = nn.Sequential(
            nn.Conv2d(self.channels + 1, num_channels * 2, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(num_channels * 2, num_channels * 3, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(num_channels * 3, num_channels * 4, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(num_channels * 4, num_channels * 2, 3, 1, 1),
            nn.PReLU(),
            # nn.Conv2d(num_channels * 3, num_channels * 2, 3, 1, 1),
            # nn.PReLU(),
            nn.Conv2d(num_channels * 2, 3, 3, 1, 1),
        )
        
        # Loss
        self._loss = RDLoss()
        
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
        pred = self.forward(input=input, *args, **kwargs)
        illumination     = pred[0]
        illumination_hat = pred[1]
        reflectance      = pred[2]
        noise            = pred[3]
        relight          = pred[4]
        loss = self.loss(
            image        = input,
            illumination = illumination,
            reflectance  = reflectance,
            noise        = noise,
        ) if self.loss else None
        return relight, loss
    
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
        x 	   = input

        x_down = x
        if self.scale_factor != 1:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode="bilinear")

        mean_c = x_down.mean(dim=1).unsqueeze(1)
        x_cat  = torch.cat([x_down, mean_c], dim=1)

        s = torch.sigmoid(self.illumination_net(x_cat))
        r = torch.sigmoid(self.reflectance_net(x_cat))
        n =    torch.tanh(self.noise_net(x_cat))

        if self.scale_factor != 1:
            s = self.upsample(s)
            r = self.upsample(r)
            n = self.upsample(n)

        s_hat  = torch.pow(s, self.gamma)
        y      = s_hat * ((x - n) / s)
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
        # self.act   = nn.PReLU()
        
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
        mse_loss  = nn.MSELoss(reduction="mean")
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
        ce_gamma    : float = 2.8,
        rd_gamma    : float = 0.4,
        zns_gamma   : float = 0.5,
        weights     : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            name     = "cerdnet",
            channels = channels,
            weights  = None,
            *args, **kwargs
        )
        
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            channels     = self.weights.get("channels"    , channels)
            num_channels = self.weights.get("num_channels", num_channels)
            num_iters    = self.weights.get("num_iters"   , num_iters)
            scale_factor = self.weights.get("scale_factor", scale_factor)
            ce_gamma     = self.weights.get("ce_gamma"    , ce_gamma)
            rd_gamma     = self.weights.get("rd_gamma"    , rd_gamma)
            zns_gamma    = self.weights.get("zns_gamma"   , zns_gamma)
        
        self._channels    = channels
        self.num_channels = num_channels
        self.num_iters    = num_iters
        self.scale_factor = scale_factor
        self.ce_gamma     = ce_gamma
        self.rd_gamma     = rd_gamma
        self.zns_gamma    = zns_gamma
        
        # Construct model
        self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
        self.cenet    = CENet(
            channels     = self.channels,
            num_channels = self.num_channels,
            num_iters    = self.num_iters,
            scale_factor = self.scale_factor,
            gamma        = self.ce_gamma,
            pretrain     = False,
            weights      = weights,
            verbose      = self.verbose,
        )
        self.cenet.eval()
        self.rdnet = RDNet(
            channels     = self.channels,
            num_channels = 16,
            scale_factor = 1,
            gamma        = self.rd_gamma,
            verbose      = self.verbose,
        )
        self.zsn2n = ZSN2N(
            channels     = self.channels,
            num_channels = 64,
        )
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self._init_weights)
        self.initial_state_dict = self.state_dict()
        
    def _init_weights(self, m: nn.Module):
        pass
    
    def forward_loss(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pass
    
    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> torch.Tensor:
        pass
    
    # region Training
    
    def fit_one(
        self,
        input         : torch.Tensor,
        rd_max_epochs : int   = 1000,
        zns_max_epochs: int   = 3000,
        rd_lr         : float = 0.001,
        zsn_lr        : float = 0.001,
        reset_weights : bool  = True,
        convert_output: bool  = False,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        #
        if isinstance(input, np.ndarray):
            input = core.to_image_tensor(input, False, True)
        input = input.to(self.device)
        assert input.shape[0] == 1
        #
        with torch.no_grad():
            lightup, lightup_image = self.cenet(input)
        #
        pred = self.rdnet.fit_one(
            input         = lightup_image,
            max_epochs    = rd_max_epochs,
            lr            = rd_lr,
            reset_weights = reset_weights,
        )
        illumination     = pred[0]
        illumination_hat = pred[1]
        reflectance      = pred[2]
        noise            = pred[3]
        relight          = pred[4]
        #
        denoised = self.zsn2n.fit_one(
            input         = relight.clone().detach().requires_grad_(True),
            max_epochs    = zns_max_epochs,
            lr            = zsn_lr,
            gamma         = self.zns_gamma,
            reset_weights = reset_weights,
        )
        #
        illumination     = torch.concat([illumination,     illumination,     illumination    ], dim=1)
        illumination_hat = torch.concat([illumination_hat, illumination_hat, illumination_hat], dim=1)
        #
        if convert_output:
            lightup          = core.to_image_nparray(lightup         , False, True)
            lightup_image    = core.to_image_nparray(lightup_image   , False, True)
            illumination     = core.to_image_nparray(illumination    , False, True)
            illumination_hat = core.to_image_nparray(illumination_hat, False, True)
            reflectance      = core.to_image_nparray(reflectance     , False, True)
            noise            = core.to_image_nparray(noise           , False, True)
            relight          = core.to_image_nparray(relight         , False, True)
            denoised         = core.to_image_nparray(denoised        , False, True)
        
        return (
            lightup,
            lightup_image,
            illumination,
            illumination_hat,
            reflectance,
            noise,
            relight,
            denoised
        )

    # endregion
    
# endregion
