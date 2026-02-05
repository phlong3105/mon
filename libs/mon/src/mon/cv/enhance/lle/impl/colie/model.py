#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CoLIE.

This module provides the CoLIE definition and pre-trained weights.

References:
    - Paper: "Fast Context-Based Low-Light Image Enhancement via Neural Implicit
      Representations," ECCV 2024.
    - Code: https://github.com/ctom2/colie
"""

from __future__ import annotations

__all__ = [
    "CoLIE",
    "CoLIE_PP",
    "colie",
    "colie_pp",
]

import sys
import torch.nn.functional as F
import torch

from mon import nn
from mon.core import create_device, log, MLType, MODELS, Path, Task

current_file = Path(__file__).normalize()
current_dir  = current_file.parents[0]
extern_path  = current_dir / "extern" / "colie"
if str(extern_path) not in sys.path:
    sys.path.append(str(extern_path))

try:
    # Now we can safely import from the original repository
    import loss as L
    from color import rgb2hsv_torch, hsv2rgb_torch
    from siren import ResidualINR, ReflectanceINR
    from utils import (
        filter_up,
        get_coords,
        get_patches,
        get_v_component,
        interpolate_image,
        replace_v_component,
    )
except ImportError:
    raise ImportError(f"Failed to import modules from the 'extern/colie' directory.")


# ==============================================================================
# region BASE CLASSES & MIXINS
# ==============================================================================

# --- Base Classes ---

class CoLIE(nn.Module, nn.RegistrableMixin):
    """CoLIE model for low-light image enhancement.

    References:
        - Paper: "Fast Context-Based Low-Light Image Enhancement via Neural
          Implicit Representations," ECCV 2024.
        - Code: https://github.com/ctom2/colie
    """

    arch     : str          = "colie"
    name     : str          = "colie"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.TEST_TIME]
    model_dir: Path         = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name       : str,
        window_size: int,
        hidden_dim : int,
        num_layers : int,
        add_layers : int,
        device     : torch.device = torch.device("cpu"),
        verbose    : bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name: Name of the model variant.
            window_size: Size of the patch window.
            hidden_dim: Hidden dimension of the networks.
            num_layers: Number of layers in each branch.
            add_layers: Number of layers to add between the two branches.
            device: Device to use for computation. Defaults to "cpu".
            verbose: Verbosity mode. Defaults to True.
            *args: Additional positional arguments for the INR model.
            **kwargs: Additional keyword arguments for the INR model.
        """
        # Satisfy PyTorch's empty signature first.
        super().__init__()
        # Initialize RegistrableMixin
        nn.RegistrableMixin.__init__(self, name=name)

        # Assign attributes
        self.verbose     = verbose
        self.window_size = window_size
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.add_layers  = add_layers
        self.inr_args    = args
        self.inr_kwargs  = kwargs
        self.device      = create_device(device)

    # --- Callable & Context Manager ---
    def forward(
        self,
        image     : torch.Tensor,
        epochs    : int   = 100,
        E         : float = 0.5,
        save_debug: bool  = False,
    ) -> dict:
        """Forward the input through the network.

        Args:
            image: Image, formatted as a torch.Tensor of shape (B, 3, H, W)
                and values ranging from 0.0 to 1.0.
            epochs: Number of optimization steps. Defaults to 100.
            E: Well-exposedness level E. Defaults to 0.1.
            save_debug: Whether to save intermediate results for debugging.
                Defaults to False.
        """
        window_size = self.window_size
        down_size   = self.hidden_dim
        patch_dim   = window_size ** 2

        # 1. Create the INR network
        model = ResidualINR(
            patch_dim  = patch_dim,
            hidden_dim = self.hidden_dim,
            num_layers = self.num_layers,
            add_layer  = self.add_layers,
        ).to(self.device)

        # 2. Move inputs to the corresponding device
        image = image.to(self.device)

        # 3. Convert the image to HSV color space
        image_hsv  = rgb2hsv_torch(image).to(self.device)
        image_i    = get_v_component(image_hsv).to(self.device)
        lr_image_i = interpolate_image(image_i, down_size, down_size).to(self.device)

        # 4. Get coordinates and patches
        coords    = get_coords(down_size, down_size).to(self.device)
        patches   = get_patches(lr_image_i, window_size).to(self.device)

        # 5. Define optimizer & losses
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=3e-4)
        L_exp     = L.L_exp(16, E).to(self.device)
        L_tv      = L.L_TV().to(self.device)

        # 6. Optimize the INR network
        lr_image_i_res   = None
        lr_image_i_fixed = None
        lr_image_r       = None
        for i in range(epochs):
            model.train()
            optimizer.zero_grad()

            # 6.1 Forward pass
            lr_image_i_res = model(patches, coords)
            lr_image_i_res = lr_image_i_res.view(1, 1, down_size, down_size)

            # 6.2 Retinex reconstruction
            lr_image_i_fixed = lr_image_i_res + lr_image_i
            lr_image_r       = lr_image_i / (lr_image_i_fixed + 1e-4)

            # 6.3 Loss
            l_spa      = torch.mean(torch.abs(torch.pow(lr_image_i_fixed - lr_image_i, 2)))
            l_tv       = L_tv(lr_image_i_fixed)
            l_exp      = torch.mean(L_exp(lr_image_i_fixed))
            l_sparsity = torch.mean(lr_image_r)
            loss       = l_spa + (20 * l_tv) + (8 * l_exp) + (5 * l_sparsity)
            loss.backward()
            optimizer.step()

            if self.verbose:
                log(f"Epoch {i+1:4d}/{epochs:4d}: Loss = {loss:6.2f}")

        # 7. Final Retinex reconstruction
        image_r         = filter_up(lr_image_i, lr_image_r, image_i)
        image_hsv_fixed = replace_v_component(image_hsv, image_r)
        image_rgb_fixed = hsv2rgb_torch(image_hsv_fixed)
        image_rgb_fixed = image_rgb_fixed / torch.max(image_rgb_fixed)

        # 8. Return final and intermediate results for debugging
        outputs = { "enhanced": image_rgb_fixed }
        if save_debug:
            image_i_res   = filter_up(lr_image_i, lr_image_i_res,   image_i)
            image_i_fixed = filter_up(lr_image_i, lr_image_i_fixed, image_i)
            outputs |= {
                "image_i"      : image_i,
                "image_i_res"  : image_i_res,
                "image_i_fixed": image_i_fixed,
                "image_r"      : image_r,
            }
        return outputs


class CoLIE_PP(CoLIE):
    """CoLIE++ model for low-light image enhancement.

    Extend CoLIE by incorporating a second INR for refining the reflectance map.

    References:
        - Paper: "Fast Context-Based Low-Light Image Enhancement via Neural
          Implicit Representations," ECCV 2024.
        - Code: https://github.com/ctom2/colie
    """

    name: str = "colie_pp"

    # --- Callable & Context Manager ---
    def forward(
        self,
        image     : torch.Tensor,
        epochs    : int   = 100,
        E         : float = 0.5,
        save_debug: bool  = False,
    ) -> dict:
        """Forward the input through the network.

        Args:
            image: Image, formatted as a torch.Tensor of shape (B, 3, H, W)
                and values ranging from 0.0 to 1.0.
            epochs: Number of optimization steps. Defaults to 100.
            E: Well-exposedness level E. Defaults to 0.1.
            save_debug: Whether to save intermediate results for debugging.
                Defaults to False.
        """
        window_size = self.window_size
        down_size   = self.hidden_dim

        # 1. Move inputs to the corresponding device
        image = image.to(self.device)

        # 2. Convert the image to HSV color space
        image_hsv  = rgb2hsv_torch(image).to(self.device)
        image_i    = get_v_component(image_hsv).to(self.device)
        lr_image_i = interpolate_image(image_i, down_size, down_size).to(self.device)

        # 3. Get coordinates and patches
        coords_res = get_coords(down_size, down_size).to(self.device)
        patches    = get_patches(lr_image_i, window_size).to(self.device)

        # 4. Create the ResidualINR network
        resnet = ResidualINR(
            patch_dim  = self.window_size ** 2,
            hidden_dim = self.hidden_dim,
            num_layers = self.num_layers,
            add_layer  = self.add_layers,
        ).to(self.device)

        # 5. Define optimizer & losses for ResidualINR
        resnet_optim = torch.optim.Adam(resnet.parameters(), lr=1e-5, weight_decay=3e-4)
        L_exp        = L.L_exp(16, E).to(self.device)
        L_tv_res     = L.L_TV().to(self.device)

        # 6. Optimize the ResidualINR network
        lr_image_i_res   = None
        lr_image_i_fixed = None
        lr_image_r       = None
        for i in range(epochs):
            resnet.train()
            resnet_optim.zero_grad()

            # 6.1 Forward pass
            lr_image_i_res = resnet(patches, coords_res)
            lr_image_i_res = lr_image_i_res.view(1, 1, down_size, down_size)

            # 6.2 Retinex reconstruction
            lr_image_i_fixed = lr_image_i_res + lr_image_i
            lr_image_r       = lr_image_i / (lr_image_i_fixed + 1e-4)

            # 6.3 Loss
            l_spa      = torch.mean(torch.abs(torch.pow(lr_image_i_fixed - lr_image_i, 2)))
            l_tv       = L_tv_res(lr_image_i_fixed)
            l_exp      = torch.mean(L_exp(lr_image_i_fixed))
            l_sparsity = torch.mean(lr_image_r)
            loss1      = l_spa + (20 * l_tv) + (8 * l_exp) + (5 * l_sparsity)
            loss1.backward()
            resnet_optim.step()

            if self.verbose:
                log(f"Epoch {i+1:4d}/{epochs:4d}: Loss = {loss1:6.2f}")

        # 7. Define the ReflectanceINR network
        refnet = ReflectanceINR(
            hidden_dim = self.hidden_dim,
            num_layers = self.num_layers,
        ).to(self.device)

        # 8. Define optimizer & losses for ReflectanceINR
        refnet_optim = torch.optim.Adam(params=refnet.parameters(), lr=1e-3, weight_decay=3e-4)
        L_tv_ref     = L.L_TV().to(self.device)

        # 9. Optimize the ReflectanceINR network
        lr_image_r       = lr_image_r.detach_()
        coords_ref       = get_coords(down_size, down_size).to(self.device)
        lr_image_r_fixed = None
        for i in range(epochs):
            refnet.train()
            refnet_optim.zero_grad()

            # 9.1 Forward pass
            lr_image_r_fixed = refnet(coords_ref)
            lr_image_r_fixed = lr_image_r_fixed.view(1, 1, down_size, down_size)

            # 4.2. Loss
            l_fit = F.mse_loss(lr_image_r_fixed, lr_image_r)
            l_tv  = L_tv_ref(lr_image_r_fixed)
            loss2 = l_fit + (0.1 * l_tv)
            loss2.backward()
            refnet_optim.step()

            if self.verbose:
                log(f"Epoch {i+1:4d}/{epochs:4d}: Loss = {loss2:6.2f}")

        # 7. Final Retinex reconstruction
        image_r         = filter_up(lr_image_i, lr_image_r_fixed, image_i)
        image_hsv_fixed = replace_v_component(image_hsv, image_r)
        image_rgb_fixed = hsv2rgb_torch(image_hsv_fixed)
        image_rgb_fixed = image_rgb_fixed / torch.max(image_rgb_fixed)

        # 8. Return final and intermediate results for debugging
        outputs = { "enhanced": image_rgb_fixed }
        if save_debug:
            image_i_res   = filter_up(lr_image_i, lr_image_i_res,   image_i)
            image_i_fixed = filter_up(lr_image_i, lr_image_i_fixed, image_i)
            outputs |= {
                "image_i"      : image_i,
                "image_i_res"  : image_i_res,
                "image_i_fixed": image_i_fixed,
                "image_r"      : image_r,
            }
        return outputs


# --- Mixins ---


# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---


# --- Model Variants ---

@MODELS.register(name="colie", metaclass=CoLIE)
def colie(*args, **kwargs):
    """Create a CoLIE model.

    Args:
        args: Additional positional arguments for the CoLIE model.
        kwargs: Additional keyword arguments for the CoLIE model.

    Returns:
        An CoLIE model instance.
    """
    return CoLIE(
        name        = "colie",
        window_size = 7,
        hidden_dim  = 256,
        num_layers  = 4,
        add_layers  = 2,
        *args, **kwargs
    )


@MODELS.register(name="colie_pp", metaclass=CoLIE_PP)
def colie_pp(*args, **kwargs):
    """Create a CoLIE++ model.

    Args:
        args: Additional positional arguments for the CoLIE model.
        kwargs: Additional keyword arguments for the CoLIE model.

    Returns:
        An CoLIE++ model instance.
    """
    return CoLIE_PP(
        name        = "colie_pp",
        window_size = 7,
        hidden_dim  = 256,
        num_layers  = 4,
        add_layers  = 2,
        *args, **kwargs
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    model_ = colie()
    print(model_)

# endregion
