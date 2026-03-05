#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CALIE Models.

This module provides the CALIE definition and pre-trained weights.

References:
    - Paper: "Continuously Adjustable Low-Light Implicit Enhancement"
    - Code: https://github.com/phlong3105/calie
"""

from __future__ import annotations

__all__ = [
    "CALIE",
    "calie_ffsiren",
    "calie_siren",
]

import copy
import sys

import kornia
import torch
from torch import nn, Tensor

from mon.core import log, MODELS, Path, SizeLike, Task
from mon.cv.models.restore import ZSN2N
from mon.nn import loss as L, ModelRegisterMixin

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]
if str(current_dir) not in sys.path:
    # Add the project root to sys.path so 'import saleo' works
    # even if you run this script from inside the folder
    sys.path.append(str(current_dir))

try:
    # Works when running as a module: python -m zero_dce.predict
    from .loss import ConfidenceGatedDepthLoss
    from .module import ResidualINR
    from .utils import (
        RgbToHsv,
        RgbToHvi,
        filter_up,
        get_coords,
        get_patches,
        get_v_component,
        interpolate_image,
        replace_v_component,
    )
except ImportError:
    # Works when running as a script: python predict.py
    from loss import ConfidenceGatedDepthLoss
    from module import ResidualINR
    from utils import (
        RgbToHsv,
        RgbToHvi,
        filter_up,
        get_coords,
        get_patches,
        get_v_component,
        interpolate_image,
        replace_v_component,
    )

# Allows PyTorch to use TF32 cores for matrix multiplications
torch.set_float32_matmul_precision("high")


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

# noinspection PyMethodMayBeStatic
class CALIE(ModelRegisterMixin, nn.Module):
    """CALIE model.

    References:
        - Paper: "Continuously Adjustable Low-Light Implicit Enhancement"
        - Code: https://github.com/phlong3105/calie
    """

    arch: str = "calie"
    name: str = "calie"
    tasks: list[Task] = [Task.ENHANCE]
    model_dir: Path = current_dir

    color_funcs: dict = {
        "hsv": RgbToHsv,
        "hvi": RgbToHvi,
    }

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
            name (str): Name of the model to use.
            window_size (int): Size of the patch window.
            hidden_dim (int): Hidden dimension of the networks.
            num_layers (int): Number of layers in the networks.
            add_layers (int): Number of layers to add between the two branches.
            device (torch.device, optional): Device to use for computation.
                Defaults to "cpu"."
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        # Satisfy PyTorch's empty signature first.
        super().__init__(name=name)

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
        depth: Tensor | None = None,
        epochs: int = 100,
        E: float = 0.3,
        color_func: str = "hsv",
        save_debug: bool = False,
    ) -> dict:
        """Forward the input through the network.

        For each input sample, a corresponding INR network is created and
        optimized to fit the illumination map. The final enhanced image is then
        reconstructed using the learned illumination map.

        Args:
            image (Tensor): Image tensor of shape (B, 3, H, W) and values
                ranging from 0.0 to 1.0.
            depth (Tensor, optional): Depth map tensor of shape (B, 1, H, W)
                and values ranging from 0.0 to 1.0. Defaults to None.
            epochs (int, optional): Number of optimization steps. Defaults to 100.
            E (float, optional): Well-exposedness level E. Defaults to 0.3.
            color_func (str, optional): Color space to use. Defaults to "hsv".
            save_debug (bool, optional): If True, return intermediate results
                for debugging. Defaults to False.
        """
        window_size = self.window_size
        down_size = self.hidden_dim
        if depth is not None:
            patch_dim = window_size ** 2 * 2
        else:
            patch_dim = window_size ** 2

        # 1. Create the INR network
        model = ResidualINR(
            patch_dim=patch_dim,
            out_features=1,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            add_layers=self.add_layers,
            *self.inr_args,
            **self.inr_kwargs
        ).to(self.device)

        # 2. Move inputs to the corresponding device
        image = image.to(self.device)

        # 2.1 Pre-process image
        image = self._pre_denoise(image)

        # 3. Convert color space
        color_func = self.color_funcs[color_func]()
        color_func = color_func.to(self.device)
        image_hsv = color_func.from_rgb(image).to(self.device)
        image_h = image_hsv[:, -3].unsqueeze(0)
        image_s = image_hsv[:, -2].unsqueeze(0)
        image_i = image_hsv[:, -1].unsqueeze(0)

        # 4. Get coordinates and patches
        lr_image_i = interpolate_image(image_i, down_size).to(self.device)
        coords = get_coords(down_size).to(self.device)
        patches = get_patches(lr_image_i, window_size).to(self.device)

        # 5. Define optimizer & losses
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=1e-5,
            betas=(0.9, 0.999),
            weight_decay=3e-4,
            fused=True  # <-- Instantly speeds up the optimizer step by ~20%
        )
        scaler = torch.amp.GradScaler("cuda")
        L_exp = L.ExposureValueControlLoss(patch_size=16, E=E).to(self.device)
        L_tv = L.TotalVariationLoss().to(self.device)

        # 6. Optimize the network
        best_weights = None
        best_loss = float("inf")
        lr_image_i_res = None
        lr_image_i_fixed = None
        lr_image_r = None

        for i in range(epochs):
            model.train()
            optimizer.zero_grad()

            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                # 6.1. Forward pass
                lr_image_i_res = model(coords, patches)
                lr_image_i_res = lr_image_i_res.view(1, 1, down_size, down_size)

                # 6.2. Retinex reconstruction
                lr_image_i_fixed = torch.clamp(lr_image_i_res + lr_image_i, 1e-4, 1.0)
                lr_image_r = torch.clamp(lr_image_i / lr_image_i_fixed, 0.0, 1.0)

                # 6.3. Loss
                l_spa = torch.mean(torch.abs(torch.pow(lr_image_i_fixed - lr_image_i, 2)))
                l_tv = L_tv(lr_image_i_fixed)
                l_exp = torch.mean(L_exp(lr_image_i_fixed))
                l_sparsity = torch.mean(lr_image_r)
                loss = l_spa + (20 * l_tv) + (8 * l_exp) + (5 * l_sparsity)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()

            # 6.4. Save best weights
            if loss < best_loss:
                best_loss = loss
                best_weights = copy.deepcopy(model.state_dict())

            # 6.5. Log debugging information
            if self.verbose:
                log(f"Epoch {i+1:4d}/{epochs:4d}: Loss = {loss:6.2f}")

        # 7. Load best weights
        model.load_state_dict(best_weights)

        # 8. Infer the illumination residual with the optimized network
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                lr_image_i_res = model(coords, patches)
            lr_image_i_res = lr_image_i_res.view(1, 1, down_size, down_size)
            lr_image_i_fixed = torch.clamp(lr_image_i_res + lr_image_i, 1e-4, 1.0)
            lr_image_r = torch.clamp(lr_image_i / lr_image_i_fixed, 0.0, 1.0)

        # 9. Final Retinex reconstruction
        image_r = filter_up(lr_image_i, lr_image_r, image_i)
        image_hsv_fixed = replace_v_component(image_hsv, image_r)
        image_rgb_fixed = color_func.to_rgb(image_hsv_fixed).to(self.device)
        image_rgb_fixed = image_rgb_fixed / torch.max(image_rgb_fixed)

        # 6. Return final and intermediate results for debugging
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

    # --- Utils ---
    def _pre_denoise(
        self,
        image: Tensor,
        imgsz: SizeLike | None = 512,
        epochs: int = 100,
    ) -> Tensor:
        # Resize the image for faster processing
        if imgsz is not None:
            resized = interpolate_image(image, imgsz)
        else:
            resized = image

        # Run ZSN2N on the resized image
        _, c, _, _ = resized.shape
        zsn2n = ZSN2N(in_channels=c, epochs=epochs, device=self.device)
        outputs = zsn2n(resized)
        denoised = outputs["restored"]

        # Resize the denoised image back to the original size
        denoised = filter_up(resized, denoised, image)

        return denoised

    def _post_denoise(
        self,
        image: Tensor,
        guidance: Tensor,
        sigma_color: float = 0.1,
        sigma_space: float = 1.5,
    ) -> Tensor:
        """Denoise the image using a joint bilateral filter."""
        # Radius is typically 3 * sigma
        radius = int(3 * sigma_space)
        if radius % 2 == 0:
            radius += 1 # Ensure odd
        kernel_size = (radius, radius)

        image = kornia.filters.joint_bilateral_blur(
            input=image,
            guidance=guidance,
            kernel_size=kernel_size,
            sigma_color=sigma_color,
            sigma_space=(sigma_space, sigma_space)
        )

        return image

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---


# --- Model Variants ---

@MODELS.register(name="calie_siren", metaclass=CALIE)
def calie_siren(*args, **kwargs):
    """Create a CALIE model with SIREN."""
    _ = kwargs.pop("name", "calie_siren")
    window_size = kwargs.pop("window_size", 7)
    hidden_dim = kwargs.pop("hidden_dim", 256)
    num_layers = kwargs.pop("num_layers", 4)
    add_layers = kwargs.pop("add_layers", 2)
    _ = kwargs.pop("pos_encode", False)
    mapping_size = kwargs.pop("mapping_size", 256)
    B = kwargs.pop("B", 30.0)
    return CALIE(
        name="calie_siren",
        window_size=window_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        add_layers=add_layers,
        pos_encode=False,
        mapping_size=mapping_size,
        B=B,
        *args, **kwargs
    )


@MODELS.register(name="calie_ffsiren", metaclass=CALIE)
def calie_ffsiren(*args, **kwargs):
    """Create a CALIE model with FF+SIREN."""
    _ = kwargs.pop("name", "calie_ffsiren")
    window_size = kwargs.pop("window_size", 7)
    hidden_dim = kwargs.pop("hidden_dim", 256)
    num_layers = kwargs.pop("num_layers", 4)
    add_layers = kwargs.pop("add_layers", 2)
    _ = kwargs.pop("pos_encode", True)
    mapping_size = kwargs.pop("mapping_size", 256)
    B = kwargs.pop("B", 30.0)
    return CALIE(
        name="calie_ffsiren",
        window_size=window_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        add_layers=add_layers,
        pos_encode=True,
        mapping_size=mapping_size,
        B=B,
        *args, **kwargs
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
