#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CoLIE Models.

This module provides the CoLIE definition and pre-trained weights.

References:
    - Paper: "Fast Context-Based Low-Light Image Enhancement via Neural Implicit
      Representations," ECCV 2024.
    - Code: https://github.com/ctom2/colie
"""

from __future__ import annotations

__all__ = [
    "CoLIE",
    "colie",
]

from typing import override

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Adam, Optimizer

from mon.core import (
    log,
    MODELS,
    OPTIMIZERS,
    PATCHERS,
    Path,
    Size,
    Strategy,
    Task,
)
from mon.metrics import benchmark
from mon.nn import Model, ModelRegisterMixin
from mon.ops import guided_filter_upsample, ImagePatcher, RgbToHsv
from . import loss as L
from .module import ResidualINR
from .utils import get_coords, get_patches, replace_v_component

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class CoLIE(ModelRegisterMixin, Model):
    """CoLIE model for low-light image enhancement.

    References:
        - Paper: "Fast Context-Based Low-Light Image Enhancement via Neural Implicit
          Representations," ECCV 2024.
        - Code: https://github.com/ctom2/colie
    """

    arch: str = "colie"
    name: str = "colie"
    tasks: list[Task] = [Task.LLE]
    strategies: list[Strategy] = [Strategy.NATIVE, Strategy.PATCH]
    model_dir: Path = current_dir

    in_keys: set = {"image"}
    out_keys: set = {"enhanced"}
    debug_keys: set = {"image_i_res", "image_i_fixed", "image_r"}

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        window_size: int,
        hidden_dim: int,
        num_layers: int,
        add_layers: int,
        epochs: int = 100,
        optimizer: dict | None = None,
        device: torch.device = torch.device("cpu"),
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            window_size (int): Size of the local window for context aggregation.
            hidden_dim (int): Number of channels in the hidden layers.
            num_layers (int): Total number of layers in the network.
            add_layers (int): Number of additional layers for context aggregation.
            epochs (int, optional): Number of optimization epochs for
                single-image optimization. Defaults to 100.
            optimizer (DictLike, optional): Dictionary containing optimizer
                parameters. Defaults to None.
            device (torch.device, optional): Device to use for computation.
                Defaults to torch.device("cpu").
            verbose (bool, optional): Verbosity mode. Defaults to True.
            *args: Additional positional arguments for the INR model.
            **kwargs: Additional keyword arguments for the INR model.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.add_layers = add_layers
        self.epochs = epochs
        self.device = device
        self.optimizer = optimizer

    # --- Callable & Context Manager ---
    @override
    def forward(
        self,
        image: Tensor,
        E: float = 0.5,
        use_patch: bool = False,
        *args, **kwargs
    ) -> tuple[Tensor, ...]:
        """Forward the input through the network.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            E (float, optional): Exponential loss parameter. Defaults to 0.5.
            use_patch (bool, optional): Whether to use patch-based strategy.
                Defaults to False.

        Returns:
            - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - image_i_res (Tensor): The residual illumination component of
                  the image of shape (B, C, H, W) and values ranging from
                  0.0 to 1.0.
                - image_i_fixed (Tensor): The fixed illumination component of
                  the image of shape (B, C, H, W) and values ranging from
                  0.0 to 1.0.
                - image_r (Tensor): The reflectance component of the image of
                  shape (B, C, H, W) and values ranging from 0.0 to 1.0.
        """
        if use_patch:
            return self.forward_patch(image=image, E=E, *args, **kwargs)
        else:
            return self.forward_step(image=image, E=E, *args, **kwargs)

    @override
    def forward_step(
        self,
        image: Tensor,
        E: float = 0.5,
        *args, **kwargs
    ) -> tuple[Tensor, ...]:
        """Forward the input through the network using the patch-based strategy.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            E (float, optional): Exponential loss parameter. Defaults to 0.5.

        Returns:
            tuple[Tensor, ...]: A tuple containing:

                - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - image_i_res (Tensor): The residual illumination component of
                  the image of shape (B, C, H, W) and values ranging from
                  0.0 to 1.0.
                - image_i_fixed (Tensor): The fixed illumination component of
                  the image of shape (B, C, H, W) and values ranging from
                  0.0 to 1.0.
                - image_r (Tensor): The reflectance component of the image of
                  shape (B, C, H, W) and values ranging from 0.0 to 1.0.
        """
        epochs = self.epochs
        window_size = self.window_size
        down_size = self.hidden_dim
        device = self.device

        # 1. Define the INR network
        model = ResidualINR(
            patch_dim=window_size ** 2,
            hidden_dim=down_size,
            num_layers=self.num_layers,
            add_layer=self.add_layers,
        ).to(device)

        optimizer = self._build_optimizer(model=model, optimizer=self.optimizer)

        # 2. Move inputs to the corresponding device
        image = image.to(device)

        # 3. Convert the image to HSV color space
        color_func = RgbToHsv().to(device)
        image_hsv = color_func.from_rgb(image)
        # image_h = image_hsv[:, 0:1, :, :].detach()  # Detach to prevent memory leak
        # image_s = image_hsv[:, 1:2, :, :].detach()  # Detach to prevent memory leak
        image_i = image_hsv[:, 2:3, :, :].detach()  # Detach to prevent memory leak
        lr_image_i = F.interpolate(image_i, (down_size, down_size)).to(device)

        # 4. Get coordinates and patches
        coords = get_coords(down_size, down_size).to(device)
        patches = get_patches(lr_image_i, window_size).to(device)

        # 5. Define losses
        L_exp = L.L_exp(16, E).to(device)
        L_tv = L.L_TV().to(device)
        lr_image_i_res = None
        lr_image_i_fixed = None

        # 6. Optimize the INR network
        best_loss = float("inf")
        best_epoch = 0
        best_state_dict = None

        model.train()
        for i in range(epochs):
            # 6.1. Forward pass
            lr_image_i_res = model(patches, coords)
            lr_image_i_res = lr_image_i_res.view(1, 1, down_size, down_size)

            # 6.2. Retinex reconstruction
            lr_image_i_fixed = lr_image_i_res + lr_image_i
            lr_image_r = lr_image_i / (lr_image_i_fixed + 1e-4)

            # 6.3. Loss
            l_spa = torch.mean(torch.abs(torch.pow(lr_image_i_fixed - lr_image_i, 2)))
            l_tv = torch.mean(L_tv(lr_image_i_fixed))
            l_exp = torch.mean(L_exp(lr_image_i_fixed))
            l_sparsity = torch.mean(lr_image_r)
            loss = l_spa + (20 * l_tv) + (8 * l_exp) + (5 * l_sparsity)

            # 6.4. Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 6.5. Save best weights
            if loss < best_loss:
                best_loss = loss
                best_epoch = i
                best_state_dict = model.state_dict()
        optimizer.zero_grad(set_to_none=True)  # Clean up to prevent memory leaks

        # 7. Log
        if self.verbose:
            log(f"best epoch {(best_epoch + 1):03} | loss = {best_loss:.6f}")

        # 8. Final inference
        model.load_state_dict(best_state_dict)
        model.eval()
        with torch.no_grad():
            # 8.1. Forward pass
            lr_image_i_res = model(patches, coords)
            lr_image_i_res = lr_image_i_res.view(1, 1, down_size, down_size)
            # 8.2. Retinex reconstruction
            lr_image_i_fixed = lr_image_i_res + lr_image_i
            lr_image_r = lr_image_i / (lr_image_i_fixed + 1e-4)
            # 8.3. Upsample and convert back to RGB
            image_r = guided_filter_upsample(lr_image_r, image_i, lr_image_i)
            image_hsv_fixed = replace_v_component(image_hsv, image_r)
            image_rgb_fixed = color_func.to_rgb(image_hsv_fixed)
            image_rgb_fixed = image_rgb_fixed.clamp(0.0, 1.0)

        # 9. Return final and intermediate results for debugging
        image_i_res = guided_filter_upsample(lr_image_i_res, image_i, lr_image_i)
        image_i_fixed = guided_filter_upsample(lr_image_i_fixed, image_i, lr_image_i)
        return image_rgb_fixed, image_i_res, image_i_fixed, image_r

    def forward_patch(
        self,
        image: Tensor,
        E: float = 0.5,
        patcher: dict | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, ...]:
        """Forward the input through the network using the patch-based strategy.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            E (float, optional): Exponential loss parameter. Defaults to 0.5.
            patcher (dict, optional): A dictionary containing the patching
                configuration, such as patch size and stride. Defaults to None
                means using the default patcher.

        Returns:
            tuple[Tensor, ...]: A tuple containing:

                - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - image_i_res (Tensor): The residual illumination component of
                  the image of shape (B, C, H, W) and values ranging from
                  0.0 to 1.0.
                - image_i_fixed (Tensor): The fixed illumination component of
                  the image of shape (B, C, H, W) and values ranging from
                  0.0 to 1.0.
                - image_r (Tensor): The reflectance component of the image of
                  shape (B, C, H, W) and values ranging from 0.0 to 1.0.
        """
        # 1. Initialize image patcher
        patcher: dict = patcher or {"name": "uniform"}
        patcher: ImagePatcher = PATCHERS.build(image=image, **patcher)

        # 2. Iterate and Process
        for patch, x, y in patcher:
            # 2.1. Process the patch
            outputs = self.forward_step(image=patch, E=E, *args, **kwargs)
            patch_outputs = {
                "enhanced": outputs[0],
                "image_i_res": outputs[1],
                "image_i_fixed": outputs[2],
                "image_r": outputs[3],
            }
            # 2.2. Feed result back to Patcher
            patcher(patches=patch_outputs, x=x, y=y)

        # 3. Get the merged results
        return tuple(patcher.output.values())

    # noinspection PyMethodMayBeStatic
    def _build_optimizer(
        self,
        model: nn.Module,
        optimizer: dict | None = None
    ) -> Optimizer:
        """Build and return the optimizer for the INR model.

        Args:
            optimizer (dict, optional): Dictionary containing optimizer
                parameters. Defaults to None.
        """
        # Define optimizer
        if optimizer is not None:
            return OPTIMIZERS.build(params=model.parameters(), **optimizer)
        else:
            return Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=3e-4)

    # --- Benchmark ---
    @override
    def benchmark(self, imgsz: Size, *args, **kwargs) -> dict[str, float]:
        """Perform a single forward step of the model to benchmark its performance.

        Args:
            imgsz (Size): Input image size.
            **kwargs: Additional arguments for benchmarking, such as number
                of runs, device, etc.

        Returns:
            dict[str, float]: A dictionary containing the benchmark results,
                such as latency, FLOPs, and parameter count.
        """
        imgsz = Size.from_any(imgsz)
        window_size = self.window_size
        down_size = imgsz.h
        device = self.device

        # Define custom model
        model = ResidualINR(
            patch_dim=self.window_size ** 2,
            hidden_dim=down_size,
            num_layers=self.num_layers,
            add_layer=self.add_layers,
        ).to(device)

        # Create dummy inputs
        dummy_input = torch.randn(1, 1, down_size, down_size).to(device)
        coords = get_coords(down_size, down_size).to(device)
        patches = get_patches(dummy_input, window_size).to(device)
        inputs = {
            "patch": patches,
            "spatial": coords,
        }

        # Benchmark the model
        return benchmark(model=model, inputs=inputs, copy=False, *args, **kwargs)

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---


# --- Model Variants ---

@MODELS.register(name="colie", metaclass=CoLIE)
def colie(*args, **kwargs):
    """Create a CoLIE model."""
    name = kwargs.pop("name", "colie")
    window_size = kwargs.pop("window_size", 7)
    hidden_dim = kwargs.pop("hidden_dim", 256)
    num_layers = kwargs.pop("num_layers", 4)
    add_layers = kwargs.pop("add_layers", 2)
    return CoLIE(
        name="colie",
        window_size=window_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        add_layers=add_layers,
        *args, **kwargs
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
