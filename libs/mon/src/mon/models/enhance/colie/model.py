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

import copy

import torch
from torch import nn, Tensor
from torch.optim import Adam

from mon.core import DictLike, log, MODELS, OPTIMIZERS, Path, Task
from mon.nn import ModelRegisterMixin
from . import loss as L
from .module import ResidualINR
from .utils import (
    filter_up,
    get_coords,
    get_h_component,
    get_patches,
    get_s_component,
    get_v_component,
    hsv2rgb_torch,
    interpolate_image,
    replace_v_component,
    rgb2hsv_torch,
)

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class CoLIE(ModelRegisterMixin, nn.Module):
    """ZeroDCE model for low-light image enhancement.

    References:
        - Paper: "Zero-Reference Deep Curve Estimation for Low-Light Image
          Enhancement," CVPR 2020.
        - Code: https://github.com/Li-Chongyi/Zero-DCE
    """

    arch: str = "colie"
    name: str = "colie"
    tasks: list[Task] = [Task.ENHANCE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        window_size: int,
        hidden_dim: int,
        num_layers: int,
        add_layers: int,
        epochs: int = 100,
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
            device (torch.device, optional): Device to use for computation.
                Defaults to torch.device("cpu").
            verbose (bool, optional): Verbosity mode. Defaults to True.
            *args: Additional positional arguments for the INR model.
            **kwargs: Additional keyword arguments for the INR model.
        """
        # Satisfy PyTorch's empty signature first.
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.device = device

        # Define network
        self.model = ResidualINR(
            patch_dim=window_size ** 2,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            add_layer=add_layers,
        ).to(device)

        # Save the initial state dict. Since each weight is optimized for a
        # single image, so we need to reset the weights before each new image.
        self.initial_state_dict = copy.deepcopy(self.model.state_dict())

    # --- Callable & Context Manager ---
    def forward(
        self,
        image: Tensor,
        epochs: int = 100,
        E: float = 0.5,
        reset_weights: bool = True,
        optimizer: DictLike | None = None,
        save_debug: bool = False,
    ) -> dict:
        """Forward the input through the network.

        Args:
            image (Tensor): Image tensor of shape (B, 3, H, W) and values
                ranging from 0.0 to 1.0.
            epochs (int, optional): Number of optimization epochs for the INR.
                Defaults to 100.
            E (float, optional): Well-exposedness level E. Defaults to 0.1.
            reset_weights (bool, optional): If True, reset the network weights
                to the initial state before optimization. Defaults to True.
            optimizer (DictLike, optional): Dictionary containing optimizer
                parameters. Defaults to None.
            save_debug (bool, optional): If True, return intermediate results
                for debugging. Defaults to False.

        Returns:
            dict: Dictionary containing the enhanced image tensor and
                intermediate results for debugging.
        """
        epochs = epochs or self.epochs
        window_size = self.window_size
        down_size = self.hidden_dim

        # 1. Reset the network weights to the initial state
        if reset_weights and self.initial_state_dict is not None:
            self.model.load_state_dict(self.initial_state_dict)

        # 2. Define optimizer & schedulers
        if optimizer is not None:
            optimizer = OPTIMIZERS.build(params=self.model.parameters(), **optimizer)
        else:
            optimizer = Adam(self.model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=3e-4)

        # 3. Move inputs to the corresponding device
        image = image.to(self.device)

        # 4. Convert the image to HSV color space
        image_hsv = rgb2hsv_torch(image).to(self.device)
        image_h = get_h_component(image_hsv).to(self.device)
        image_s = get_s_component(image_hsv).to(self.device)
        image_i = get_v_component(image_hsv).to(self.device)
        lr_image_i = interpolate_image(image_i, down_size, down_size).to(self.device)

        # 5. Get coordinates and patches
        coords = get_coords(down_size, down_size).to(self.device)
        patches = get_patches(lr_image_i, window_size).to(self.device)

        # 6. Define losses
        L_exp = L.L_exp(16, E).to(self.device)
        L_tv = L.L_TV().to(self.device)
        lr_image_i_res = None
        lr_image_i_fixed = None
        lr_image_r = None

        # 7. Optimize the INR network
        self.model.train()
        for i in range(epochs):
            # 7.1. Forward pass
            lr_image_i_res = self.model(patches, coords)
            lr_image_i_res = lr_image_i_res.view(1, 1, down_size, down_size)

            # 7.2. Retinex reconstruction
            lr_image_i_fixed = lr_image_i_res + lr_image_i
            lr_image_r = lr_image_i / (lr_image_i_fixed + 1e-4)

            # 7.3. Loss
            l_spa = torch.mean(torch.abs(torch.pow(lr_image_i_fixed - lr_image_i, 2)))
            l_tv = L_tv(lr_image_i_fixed)
            l_exp = torch.mean(L_exp(lr_image_i_fixed))
            l_sparsity = torch.mean(lr_image_r)
            loss = l_spa + (20 * l_tv) + (8 * l_exp) + (5 * l_sparsity)

            # 7.4. Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 7.5. Log
            if self.verbose:
                log(f"Epoch {i+1:4d}/{epochs:4d}: Loss = {loss:6.2f}")

        # 8. Final Retinex reconstruction
        image_r = filter_up(lr_image_i, lr_image_r, image_i)
        image_hsv_fixed = replace_v_component(image_hsv, image_r)
        image_rgb_fixed = hsv2rgb_torch(image_hsv_fixed)
        image_rgb_fixed = image_rgb_fixed / torch.max(image_rgb_fixed)

        # 9. Return final and intermediate results for debugging
        outputs = { "enhanced": image_rgb_fixed }
        if save_debug:
            outputs |= {
                "image_h": image_h,
                "image_s": image_s,
                "image_i": image_i,
                "image_i_res": filter_up(lr_image_i, lr_image_i_res, image_i),
                "image_i_fixed": filter_up(lr_image_i, lr_image_i_fixed, image_i),
                "image_r": image_r,
            }
        return outputs

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
