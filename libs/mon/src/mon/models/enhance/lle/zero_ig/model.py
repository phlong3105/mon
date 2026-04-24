#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Zero-IG Models.

This module provides the Zero-IG definition and pre-trained weights.

References:
    - Paper: "Zero-Shot Illumination-Guided Joint Denoising and Adaptive
      Enhancement for Low-Light Images," CVPR 2024.
    - Code: https://github.com/Doyle59217/ZeroIG
"""

from __future__ import annotations

__all__ = [
    "ZeroIG",
    "ZeroIG_Weights",
    "zero_ig",
]

from typing import override

import torch
from tensordict import TensorDict
from torch import nn, Tensor

from mon.core import (
    is_weights_type,
    K,
    log,
    MODELS,
    Path,
    Size,
    Strategy,
    Task,
    WEIGHTS,
    Weights,
    WeightsEnum,
    WeightsLike,
)
from mon.metrics import benchmark, create_dummy_image
from mon.nn import Model, ModelRegisterMixin
from mon.ops import pair_downsample
from .loss import TextureDifference
from .module import Denoise1, Denoise2, Enhancer
from .utils import blur

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class ZeroIG(ModelRegisterMixin, Model):
    """Zero-IG model for low-light image enhancement.

    References:
        - Paper: "Zero-Shot Illumination-Guided Joint Denoising and Adaptive
          Enhancement for Low-Light Images," CVPR 2024.
        - Code: https://github.com/Doyle59217/ZeroIG
    """

    arch: str = "zero_ig"
    name: str = "zero_ig"
    tasks: list[Task] = [Task.LLE]
    strategies: list[Strategy] = [Strategy.RESIZE]
    model_dir: Path = current_dir

    in_keys: set = {"image"}
    out_keys: set = {"enhanced"}
    debug_keys: set = {"denoised"}

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        weights: Weights | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            weights (Weights, optional): Pre-trained weights to load.
                Defaults to None.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose

        # Define network
        self.enhance = Enhancer(layers=3, channels=64)
        self.denoise_1 = Denoise1(chan_embed=48)
        self.denoise_2 = Denoise2(chan_embed=48)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.texture_difference = TextureDifference()

        # Load weights
        if weights is not None and is_weights_type(weights):
            self.load_state_dict(weights.state_dict())
            if self.verbose:
                log(f"Initialized '{name}' from weights: '{weights.path}'.")
        else:
            if self.verbose:
                log(f"Initialized '{name}' from scratch.")

    # --- Callable & Context Manager ---
    @override
    def forward(self, image: Tensor) -> TensorDict:
        """Route the inputs through the model's different forward methods based
        on the context.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing:

                - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - denoised (Tensor): Denoised image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
        """
        return self.forward_step(image=image)

    def forward_train(self, image: Tensor) -> TensorDict:
        """Perform a single forward step of the model during training.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.

        Returns:
            TensorDict: Output data dictionary.
        """
        eps = 1e-4
        x = image + eps

        # 1. Network forward
        L11, L12 = pair_downsample(x)
        L_pred1 = L11 - self.denoise_1(L11)
        L_pred2 = L12 - self.denoise_1(L12)
        L2 = x - self.denoise_1(x)
        L2 = torch.clamp(L2, eps, 1)

        s2 = self.enhance(L2.detach())
        s21, s22 = pair_downsample(s2)
        H2 = x / s2
        H2 = torch.clamp(H2, eps, 1)

        H11 = L11 / s21
        H11 = torch.clamp(H11, eps, 1)

        H12 = L12 / s22
        H12 = torch.clamp(H12, eps, 1)

        H3_pred = torch.cat([H11, s21], 1).detach() - self.denoise_2(torch.cat([H11, s21], 1))
        H3_pred = torch.clamp(H3_pred, eps, 1)
        H13 = H3_pred[:, :3, :, :]
        s13 = H3_pred[:, 3:, :, :]

        H4_pred = torch.cat([H12, s22], 1).detach() - self.denoise_2(torch.cat([H12, s22], 1))
        H4_pred = torch.clamp(H4_pred, eps, 1)
        H14 = H4_pred[:, :3, :, :]
        s14 = H4_pred[:, 3:, :, :]

        H5_pred = torch.cat([H2, s2], 1).detach() - self.denoise_2(torch.cat([H2, s2], 1))
        H5_pred = torch.clamp(H5_pred, eps, 1)
        H3 = H5_pred[:, :3, :, :]
        s3 = H5_pred[:, 3:, :, :]

        L_pred1_L_pred2_diff = self.texture_difference(L_pred1, L_pred2)
        H3_denoised1, H3_denoised2 = pair_downsample(H3)
        H3_denoised1_H3_denoised2_diff = self.texture_difference(H3_denoised1, H3_denoised2)

        H1 = L2 / s2
        H1 = torch.clamp(H1, 0, 1)
        H2_blur = blur(H1)
        H3_blur = blur(H3)

        outputs = {
            "L_pred1": L_pred1,
            "L_pred2": L_pred2,
            "L2": L2,
            "s2": s2,
            "s21": s21,
            "s22": s22,
            "H2": H2,
            "H11": H11,
            "H12": H12,
            "H13": H13,
            "s13": s13,
            "H14": H14,
            "s14": s14,
            "H3": H3,
            "s3": s3,
            "H3_pred": H3_pred,
            "H4_pred": H4_pred,
            "L_pred1_L_pred2_diff": L_pred1_L_pred2_diff,
            "H3_denoised1_H3_denoised2_diff": H3_denoised1_H3_denoised2_diff,
            "H2_blur": H2_blur,
            "H3_blur": H3_blur,
        }

        # 2. Return final and intermediate results for debugging
        return TensorDict(outputs, batch_size=[])

    @override
    def forward_step(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Perform a single forward step of the model.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing:

                - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - denoised (Tensor): Denoised image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
        """
        eps = 1e-4
        x = image + eps

        # 1. Network forward
        L2 = x - self.denoise_1(x)
        L2 = torch.clamp(L2, eps, 1)
        s2 = self.enhance(L2)
        H2 = x / s2
        H2 = torch.clamp(H2, eps, 1)
        H5_pred = torch.cat([H2, s2], 1).detach() - self.denoise_2(torch.cat([H2, s2], 1))
        H5_pred = torch.clamp(H5_pred, eps, 1)
        H3 = H5_pred[:, :3, :, :]

        # 2. Return final and intermediate results for debugging
        return H2, H3

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
        device = next(self.parameters()).device

        # Create dummy inputs
        dummy_input = create_dummy_image(imgsz=imgsz, device=device)
        data = TensorDict({"image": dummy_input}, batch_size=[])
        inputs = {"data": data}

        # Benchmark the model
        return benchmark(model=self, inputs=inputs, *args, **kwargs)

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="zero_ig")
class ZeroIG_Weights(WeightsEnum):

    LOL = Weights(
        path=K.ZOO_ROOT / "enhance/lle/zero_ig/zero_ig/lol/zero_ig_lol.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={},
    )
    LSRW_HUAWEI = Weights(
        path=K.ZOO_ROOT / "enhance/lle/zero_ig/zero_ig/lsrw/zero_ig_lsrw_huawei.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={},
    )
    LSRW_NIKON = Weights(
        path=K.ZOO_ROOT / "enhance/lle/zero_ig/zero_ig/pretrained/zero_ig_lsrw_nikon.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={},
    )
    DEFAULT = LOL


# --- Model Variants ---

@MODELS.register(name="zero_ig", metaclass=ZeroIG)
def zero_ig(weights: WeightsLike = "default", *args, **kwargs):
    """Create a Zero-IG model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "zero_ig")
    stage = kwargs.pop("stage", 3)
    return ZeroIG(
        name="zero_ig",
        stage=stage,
        weights=ZeroIG_Weights(weights),
        *args, **kwargs,
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
