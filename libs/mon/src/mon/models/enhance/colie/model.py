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

import torch
from torch import nn, Tensor

from mon.core import log, MODELS, Path, Task
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
            device (torch.device, optional): Device to use for computation.
                Defaults to torch.device("cpu").
            verbose (bool, optional): Verbosity mode. Defaults to True.
            *args: Additional positional arguments for the INR model.
            **kwargs: Additional keyword arguments for the INR model.
        """
        # Satisfy PyTorch's empty signature first.
        super().__init__(name=name)
        # Initialize RegistrableMixin
        # ModelRegisterMixin.__init__(self, name=name)

        # Assign attributes
        self.verbose = verbose
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.add_layers = add_layers
        self.inr_args = args
        self.inr_kwargs = kwargs
        self.device = device

    # --- Callable & Context Manager ---
    def forward(
        self,
        image: Tensor,
        epochs: int = 100,
        E: float = 0.5,
        save_debug: bool = False,
    ) -> dict:
        """Forward the input through the network.

        Args:
            image (Tensor): Image tensor of shape (B, 3, H, W) and values
                ranging from 0.0 to 1.0.
            epochs (int, optional): Number of optimization epochs for the INR.
                Defaults to 100.
            E (float, optional): Well-exposedness level E. Defaults to 0.1.
            save_debug (bool, optional): If True, return intermediate results
                for debugging. Defaults to False.

        Returns:
            dict: Dictionary containing the enhanced image tensor and
                intermediate results for debugging.
        """
        window_size = self.window_size
        down_size = self.hidden_dim
        patch_dim = window_size ** 2

        # 1. Create the INR network
        model = ResidualINR(
            patch_dim=patch_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            add_layer=self.add_layers,
        ).to(self.device)

        # 2. Move inputs to the corresponding device
        image = image.to(self.device)

        # 3. Convert the image to HSV color space
        image_hsv = rgb2hsv_torch(image).to(self.device)
        image_h = get_h_component(image_hsv).to(self.device)
        image_s = get_s_component(image_hsv).to(self.device)
        image_i = get_v_component(image_hsv).to(self.device)
        lr_image_i = interpolate_image(image_i, down_size, down_size).to(self.device)

        # 4. Get coordinates and patches
        coords = get_coords(down_size, down_size).to(self.device)
        patches = get_patches(lr_image_i, window_size).to(self.device)

        # 5. Define optimizer & losses
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-5,
            betas=(0.9, 0.999),
            weight_decay=3e-4,
        )
        L_exp = L.L_exp(16, E).to(self.device)
        L_tv = L.L_TV().to(self.device)

        # 6. Optimize the INR network
        lr_image_i_res = None
        lr_image_i_fixed = None
        lr_image_r = None

        for i in range(epochs):
            model.train()
            optimizer.zero_grad()

            # 6.1. Forward pass
            lr_image_i_res = model(patches, coords)
            lr_image_i_res = lr_image_i_res.view(1, 1, down_size, down_size)

            # 6.2. Retinex reconstruction
            lr_image_i_fixed = lr_image_i_res + lr_image_i
            lr_image_r = lr_image_i / (lr_image_i_fixed + 1e-4)

            # 6.3. Loss
            l_spa = torch.mean(torch.abs(torch.pow(lr_image_i_fixed - lr_image_i, 2)))
            l_tv = L_tv(lr_image_i_fixed)
            l_exp = torch.mean(L_exp(lr_image_i_fixed))
            l_sparsity = torch.mean(lr_image_r)
            loss = l_spa + (20 * l_tv) + (8 * l_exp) + (5 * l_sparsity)
            loss.backward()
            optimizer.step()

            # 6.4. Log
            if self.verbose:
                log(f"Epoch {i+1:4d}/{epochs:4d}: Loss = {loss:6.2f}")

        # 7. Final Retinex reconstruction
        image_r = filter_up(lr_image_i, lr_image_r, image_i)
        image_hsv_fixed = replace_v_component(image_hsv, image_r)
        image_rgb_fixed = hsv2rgb_torch(image_hsv_fixed)
        image_rgb_fixed = image_rgb_fixed / torch.max(image_rgb_fixed)

        # 8. Return final and intermediate results for debugging
        outputs = { "enhanced": image_rgb_fixed }
        if save_debug:
            image_i_res = filter_up(lr_image_i, lr_image_i_res, image_i)
            image_i_fixed = filter_up(lr_image_i, lr_image_i_fixed, image_i)
            outputs |= {
                "image_h": image_h,
                "image_s": image_s,
                "image_i": image_i,
                "image_i_res": image_i_res,
                "image_i_fixed": image_i_fixed,
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
