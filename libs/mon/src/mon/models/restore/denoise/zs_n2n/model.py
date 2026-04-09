#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ZS-N2N Models.

References:
    - Paper: "Zero-Shot Noise2Noise: Efficient Image Denoising without any
      Data," CVPR 2023.
    - Code: https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing#scrollTo=Srf0GQTYrkxA

    - Paper: "Unsupervised Image Denoising Based on Self-Attention Mechanism,"
      ICAICE 2023.
"""

from __future__ import annotations

__all__ = [
    "IZS_N2N",
    "ZS_N2N",
]

from typing import Any, override

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from mon.core import DictLike, MODELS, OPTIMIZERS, Path, SCHEDULERS, Task
from mon.models.restore.base import RestorationModel
from mon.nn import ModelRegisterMixin
from .module import DenoiseNetwork, ImprovedDenoiseNetwork

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

@MODELS.register(name="zs_n2n")
class ZS_N2N(ModelRegisterMixin, RestorationModel):
    """ZS-N2N model for zero-shot image denoising.

    References:
        - Paper: "Zero-Shot Noise2Noise: Efficient Image Denoising without any
          Data," CVPR 2023.
        - Code: https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing#scrollTo=Srf0GQTYrkxA
    """

    arch: str = "zs_n2n"
    name: str = "zs_n2n"
    tasks: list[Task] = [Task.DENOISE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 48,
        fit: bool = False,
        fit_epochs: int = 3000,
        device: torch.device = torch.device("cpu"),
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            in_channels (int, optional): Number of input channels.
                Defaults to 3.
            hidden_dim (int): Number of channels in the hidden layers.
                Defaults to 48.
            fit (bool, optional): If True, perform single-image optimization
                Default to False.
            fit_epochs (int, optional): Number of optimization epochs for
                single-image optimization. Defaults to 3000.
            device (torch.device, optional): Device to use for computation.
                Defaults to torch.device("cpu").
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__()

        # Assign attributes
        self.verbose = verbose
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.fit_enabled = fit
        self.fit_epochs = fit_epochs
        self.device = device

        # Define network
        self.model = DenoiseNetwork(in_channels, hidden_dim).to(device)

        # Save the initial state dict. Since each weight is optimized for a
        # single image, so we need to reset the weights before each new image.
        self._default_state_dict = self.model.state_dict()

     # --- Callable & Context Manager ---
    @override
    def forward_step(self, data: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
        """Forward the input through the network.

        If ``self.fit_enabled`` is True, perform single-image optimization.
        Otherwise, perform standard training.

        Args:
            data (dict[str, Any]): Input data dictionary.

        Returns:
            dict[str, Any]: Output data dictionary.
        """
        image = data["image"]

        # 1. Scenario 1: Single-Image Optimization
        if self.fit_enabled:
            return self.fit(image=image, *args, **kwargs)

        # 2. Scenario 2: Standard Training
        if self.training:
            loss = self.denoise_loss(image)
            noise = self.model(image)
            restored = image - noise
            return {
                "restored": restored,
                "noise": noise,
                "loss": loss,
            }
        else:
            noise = self.model(image)
            restored = torch.clamp(image - noise, 0, 1)
            return {
                "restored": restored,
                "noise": noise,
                "loss": None,
            }

    def fit(
        self,
        image: Tensor,
        epochs: int | None = None,
        reset_weights: bool = True,
        optimizer: DictLike | None = None,
        scheduler: DictLike | None = None,
    ) -> dict:
        """Fit the model to a single image using zero-shot optimization.

        Args:
            image (Tensor): Image tensor of shape (B, 3, H, W) and values
                ranging from 0.0 to 1.0.
            epochs (int, optional): Number of optimization epochs for the network.
                Defaults to None
            reset_weights (bool, optional): If True, reset the network weights
                to the initial state before optimization. Defaults to True.
            optimizer (DictLike, optional): Dictionary containing optimizer
                parameters. Defaults to None.
            scheduler (DictLike, optional): Dictionary containing scheduler
                parameters. Defaults to None.

        Returns:
            dict: Dictionary containing the enhanced image tensor and
                intermediate results for debugging.
        """
        epochs = epochs or self.fit_epochs

        # 1. Reset the network weights to the initial state
        if reset_weights:
            self.model.load_state_dict(self._default_state_dict)

        # 2. Define optimizer & schedulers
        if optimizer is not None:
            optimizer = OPTIMIZERS.build(params=self.model.parameters(), **optimizer)
        else:
            optimizer = Adam(self.model.parameters(), lr=0.001)
        if scheduler is not None:
            scheduler = SCHEDULERS.build(optimizer=optimizer, **scheduler)
        else:
            scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)

        # 3. Move inputs to the corresponding device
        image = image.to(self.device)

        # 4. Optimize the network
        self.model.train()
        for i in range(epochs):
            loss = self.denoise_loss(image)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # 5. Final denoising step
        self.model.eval()
        with torch.no_grad():
            restored = torch.clamp(image - self.model(image),0,1)

        # 6. Return final and intermediate results for debugging
        outputs = { "restored": restored }
        return outputs

    # --- Denoise ---
    def denoise_loss(self, noisy_image: Tensor) -> Tensor:
        """Calculate the ZS-N2N denoising loss."""
        L = nn.MSELoss()  # Vanilla loss function
        # L = nn.SmoothL1Loss()  # Improved loss function

        # Residual loss
        noisy1, noisy2 = self.pair_downsampler(noisy_image)
        pred1 = noisy1 - self.model(noisy1)
        pred2 = noisy2 - self.model(noisy2)
        loss_res = 0.5 * (L(noisy1, pred2) + L(noisy2, pred1))

        # Consistency loss
        noisy_denoised = noisy_image - self.model(noisy_image)
        denoised1, denoised2 = self.pair_downsampler(noisy_denoised)
        loss_cons = 0.5 * (L(pred1, denoised1) + L(pred2, denoised2))

        # Total loss
        loss = loss_res + loss_cons

        return loss

    # --- Utilities ---
    # noinspection PyMethodMayBeStatic
    def add_noise(self, x: Tensor, noise_level: float) -> Tensor:
        """Add noise to the image."""
        noisy = x + torch.normal(0, noise_level / 255, x.shape)
        noisy = torch.clamp(noisy, 0, 1)
        noisy = noisy.to(x.device)
        return noisy

    # noinspection PyMethodMayBeStatic
    def pair_downsampler(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Downsample an image tensor into a pair to half resolution.

        References:
            - Code: https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing

        Args:
            image (Tensor): Image tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            tuple[Tensor, Tensor]: Downsampled images of shape (B, C, H/2, W/2).

        Raises:
            TypeError: If ``image`` is not a 4D torch.Tensor.

        Notes:
            Averages diagonal pixels in non-overlapping patches:
                -------------      -------------
                | A1 | B1 | A2 | B2 |      | A1+D1/2 | A2+D2/2 |
                | C1 | D1 | C2 | D2 |      | A3+D3/2 | A4+D4/2 |
                -------------  =>  -------------
                | A3 | B3 | A4 | B4 |      | B1+C1/2 | B2+C2/2 |
                | C3 | D3 | C4 | D4 |      | B3+C3/2 | B4+C4/2 |
                -------------      -------------
        """
        b, c, h, w  = image.shape
        device, dtype = image.device, image.dtype

        # Define kernels: filter_ad picks (top-left, bottom-right), filter_bc picks (top-right, bottom-left)
        # We use .repeat(c, 1, 1, 1) for channel-wise (depthwise) convolution
        kernel_ad = torch.tensor([[[[0.5, 0.0], [0.0, 0.5]]]], device=device, dtype=dtype)
        kernel_ad = kernel_ad.repeat(c, 1, 1, 1)
        kernel_bc = torch.tensor([[[[0.0, 0.5], [0.5, 0.0]]]], device=device, dtype=dtype)
        kernel_bc = kernel_bc.repeat(c, 1, 1, 1)

        # Stride=2 ensures non-overlapping 2x2 patches
        out_ad = F.conv2d(image, kernel_ad, stride=2, groups=c)
        out_bc = F.conv2d(image, kernel_bc, stride=2, groups=c)
        return out_ad, out_bc


@MODELS.register(name="izs_n2n")
class IZS_N2N(ModelRegisterMixin, RestorationModel):
    """IZS-N2N model for zero-shot image denoising.

    References:
        - Paper: "Unsupervised Image Denoising Based on Self-Attention
          Mechanism," ICAICE 2023.
    """

    arch: str = "zsn2n"
    name: str = "izs_n2n"
    tasks: list[Task] = [Task.DENOISE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 48,
        fit: bool = False,
        fit_epochs: int = 3000,
        device: torch.device = torch.device("cpu"),
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            in_channels (int, optional): Number of input channels.
                Defaults to 3.
            hidden_dim (int): Number of channels in the hidden layers.
                Defaults to 48.
            fit (bool, optional): If True, perform single-image optimization
                Default to False.
            fit_epochs (int, optional): Number of optimization epochs for
                single-image optimization. Defaults to 3000.
            device (torch.device, optional): Device to use for computation.
                Defaults to torch.device("cpu").
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__()

        # Assign attributes
        self.verbose = verbose
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.fit_enabled = fit
        self.fit_epochs = fit_epochs
        self.device = device

        # Define network
        self.model = ImprovedDenoiseNetwork(in_channels, hidden_dim).to(device)

        # Save the initial state dict. Since each weight is optimized for a
        # single image, so we need to reset the weights before each new image.
        self._default_state_dict = self.model.state_dict()

     # --- Callable & Context Manager ---
    @override
    def forward_step(self, data: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
        """Perform a single forward step of the model.

        Args:
            data (dict[str, Any]): Input data dictionary.

        Returns:
            dict[str, Any]: Output data dictionary.
        """
        image = data["image"]

        # 1. Scenario 1: Single-Image Optimization
        if self.fit_enabled:
            return self.fit(image=image, *args, **kwargs)

        # 2. Scenario 2: Standard Training
        if self.training:
            loss = self.denoise_loss(image)
            noise = self.model(image)
            restored = image - noise
            return {"restored": restored, "noise": noise,  "loss": loss}
        else:
            noise = self.model(image)
            restored = torch.clamp(image - noise, 0, 1)
            return {"restored": restored, "noise": noise,  "loss": None}

    def fit(
        self,
        image: Tensor,
        epochs: int | None = None,
        reset_weights: bool = True,
        optimizer: DictLike | None = None,
        scheduler: DictLike | None = None,
    ) -> dict:
        """Fit the model to a single image using zero-shot optimization.

        Args:
            image (Tensor): Image tensor of shape (B, 3, H, W) and values
                ranging from 0.0 to 1.0.
            epochs (int, optional): Number of optimization epochs for the network.
                Defaults to None
            reset_weights (bool, optional): If True, reset the network weights
                to the initial state before optimization. Defaults to True.
            optimizer (DictLike, optional): Dictionary containing optimizer
                parameters. Defaults to None.
            scheduler (DictLike, optional): Dictionary containing scheduler
                parameters. Defaults to None.

        Returns:
            dict: Dictionary containing the enhanced image tensor and
                intermediate results for debugging.
        """
        epochs = epochs or self.fit_epochs

        # 1. Reset the network weights to the initial state
        if reset_weights:
            self.model.load_state_dict(self._default_state_dict)

        # 2. Define optimizer & schedulers
        if optimizer is not None:
            optimizer = OPTIMIZERS.build(params=self.model.parameters(), **optimizer)
        else:
            optimizer = Adam(self.model.parameters(), lr=0.001)
        if scheduler is not None:
            scheduler = SCHEDULERS.build(optimizer=optimizer, **scheduler)
        else:
            scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)

        # 3. Move inputs to the corresponding device
        image = image.to(self.device)

        # 4. Optimize the network
        self.model.train()
        for i in range(epochs):
            loss = self.denoise_loss(image)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # 5. Final denoising step
        self.model.eval()
        with torch.no_grad():
            restored = torch.clamp(image - self.model(image),0,1)

        # 6. Return final and intermediate results for debugging
        outputs = { "restored": restored }
        return outputs

    # --- Denoise ---
    def denoise_loss(self, noisy_image: Tensor) -> Tensor:
        """Calculate the ZS-N2N denoising loss."""
        L = nn.SmoothL1Loss()  # Improved loss function

        # Residual loss
        noisy1, noisy2 = self.pair_downsampler(noisy_image)
        pred1 = noisy1 - self.model(noisy1)
        pred2 = noisy2 - self.model(noisy2)
        loss_res = 0.5 * (L(noisy1, pred2) + L(noisy2, pred1))

        # Consistency loss
        noisy_denoised = noisy_image - self.model(noisy_image)
        denoised1, denoised2 = self.pair_downsampler(noisy_denoised)
        loss_cons = 0.5 * (L(pred1, denoised1) + L(pred2, denoised2))

        # Total loss
        loss = loss_res + loss_cons

        return loss

    # --- Utilities ---
    # noinspection PyMethodMayBeStatic
    def add_noise(self, x: Tensor, noise_level: float) -> Tensor:
        """Add noise to the image."""
        noisy = x + torch.normal(0, noise_level / 255, x.shape)
        noisy = torch.clamp(noisy, 0, 1)
        noisy = noisy.to(x.device)
        return noisy

    # noinspection PyMethodMayBeStatic
    def pair_downsampler(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Downsample an image tensor into a pair to half resolution.

        References:
            - Code: https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing

        Args:
            image (Tensor): Image tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            tuple[Tensor, Tensor]: Downsampled images of shape (B, C, H/2, W/2).

        Raises:
            TypeError: If ``image`` is not a 4D torch.Tensor.

        Notes:
            Averages diagonal pixels in non-overlapping patches:
                -------------      -------------
                | A1 | B1 | A2 | B2 |      | A1+D1/2 | A2+D2/2 |
                | C1 | D1 | C2 | D2 |      | A3+D3/2 | A4+D4/2 |
                -------------  =>  -------------
                | A3 | B3 | A4 | B4 |      | B1+C1/2 | B2+C2/2 |
                | C3 | D3 | C4 | D4 |      | B3+C3/2 | B4+C4/2 |
                -------------      -------------
        """
        b, c, h, w  = image.shape
        device, dtype = image.device, image.dtype

        # Define kernels: filter_ad picks (top-left, bottom-right), filter_bc picks (top-right, bottom-left)
        # We use .repeat(c, 1, 1, 1) for channel-wise (depthwise) convolution
        kernel_ad = torch.tensor([[[[0.5, 0.0], [0.0, 0.5]]]], device=device, dtype=dtype)
        kernel_ad = kernel_ad.repeat(c, 1, 1, 1)
        kernel_bc = torch.tensor([[[[0.0, 0.5], [0.5, 0.0]]]], device=device, dtype=dtype)
        kernel_bc = kernel_bc.repeat(c, 1, 1, 1)

        # Stride=2 ensures non-overlapping 2x2 patches
        out_ad = F.conv2d(image, kernel_ad, stride=2, groups=c)
        out_bc = F.conv2d(image, kernel_bc, stride=2, groups=c)
        return out_ad, out_bc

# endregion


# ================================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
