#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""HVI-CIDNet Models.

This module provides the HVI-CIDNet definition and pre-trained weights.

References:
    - Paper: "HVI: A New color space for Low-light Image Enhancement," CVPR 2025.
    - Code: https://github.com/Fediory/HVI-CIDNet
"""

from __future__ import annotations

__all__ = [
    "HVI_CIDNet",
    "HVI_CIDNet_Weights",
    "hvi_cidnet",
]

from typing import override

import torch
from tensordict import TensorDict
from torch import Tensor

from mon.core import (
    is_weights_type,
    K,
    log,
    MODELS,
    PATCHERS,
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
from mon.ops import ImagePatcher
from .module import CIDNet

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class HVI_CIDNet(ModelRegisterMixin, Model):
    """HVI-CIDNet model for low-light image enhancement.

    References:
        - Paper: "HVI: A New color space for Low-light Image Enhancement," CVPR 2025.
        - Code: https://github.com/Fediory/HVI-CIDNet
    """

    arch: str = "hvi_cidnet"
    name: str = "hvi_cidnet"
    tasks: list[Task] = [Task.LLE]
    strategies: list[Strategy] = [Strategy.RESIZE, Strategy.PATCH]
    model_dir: Path = current_dir

    in_keys: set = {"image"}
    out_keys: set = {"enhanced"}
    debug_keys: set = {}

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str = "hvi_cidnet",
        alpha: float = 1.0,
        gated: bool = False,
        gated2: bool = False,
        weights: Weights | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str, optional): Name of the model variant.
            alpha (float, optional): Alpha parameter for the color space
                transformation. Defaults to 1.0.
            gated (bool, optional): Whether to use gated convolution in the
                first stage. Defaults to False.
            gated2 (bool, optional): Whether to use gated convolution in the
                second stage. Defaults to False.
            weights (Weights, optional): Pre-trained weights to load.
                Defaults to None.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose

        # Define network
        self.model = CIDNet(alpha=alpha, gated=gated, gated2=gated2)

        # Load weights
        if weights is not None and is_weights_type(weights):
            self.model.load_state_dict(weights.state_dict())
            if self.verbose:
                log(f"Initialized '{name}' from weights: '{weights.path}'.")
        else:
            if self.verbose:
                log(f"Initialized '{name}' from scratch.")

    # --- Callable & Context Manager ---
    @override
    def forward(
        self,
        image: Tensor,
        gamma: float = 1.0,
        use_patch: bool = False,
        *args, **kwargs
    ) -> Tensor:
        """Route the inputs through the model's different forward methods based
        on the context.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            gamma (float, optional): Gamma correction value to apply to the
                input image before processing. Defaults to 1.0 (no correction).
            use_patch (bool, optional): Whether to use patch-based strategy.
                Defaults to False.

        Returns:
            Tensor: Enhanced image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
        """
        image = image ** gamma
        
        if use_patch:
            return self.forward_patch(image=image, *args, **kwargs)
        else:
            return self.forward_step(image=image, *args, **kwargs)

    @override
    def forward_step(self, image: Tensor, *args, **kwargs) -> Tensor:
        """Perform a single forward step of the model.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.

        Returns:
            Tensor: Enhanced image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
        """
        # 1. Network forward
        enhanced = self.model(image)
        enhanced = torch.clamp(enhanced, 0.0, 1.0)

        # 2. Return final and intermediate results for debugging
        return enhanced

    def forward_patch(
        self,
        image: Tensor,
        patcher: dict | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, ...]:
        """Forward the input through the network using the patch-based strategy.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            patcher (dict, optional): A dictionary containing the patching
                configuration, such as patch size and stride. Defaults to None
                means using the default patcher.

        Returns:
            Tensor: Enhanced image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
        """
        # 1. Initialize image patcher
        patcher: dict = patcher or {"name": "uniform"}
        patcher: ImagePatcher = PATCHERS.build(image=image, **patcher)

        # 2. Iterate and Process
        for patch, x, y in patcher:
            # 2.1. Process the patch
            enhanced = self.forward_step(image=patch, *args, **kwargs)
            patch_outputs = {
                "enhanced": enhanced,
            }
            # 2.2. Feed result back to Patcher
            patcher(patches=patch_outputs, x=x, y=y)

        # 3. Get the merged results
        return tuple(patcher.output.values())

    # --- Interfaces ---
    def HVIT(self, x: Tensor) -> Tensor:
        hvi = self.model.trans.HVIT(x)
        return hvi

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
        data = TensorDict({"image": dummy_input}, batch_size=dummy_input.shape)
        inputs = {"data": data}

        # Benchmark the model
        return benchmark(model=self, inputs=inputs, *args, **kwargs)

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="hvi_cidnet")
class HVI_CIDNet_Weights(WeightsEnum):

    LOL_V1 = Weights(
        path=K.ZOO_ROOT / "enhance/lle/hvi_cidnet/hvi_cidnet/lol_v1/hvi_cidnet_lol_v1.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={},
    )
    LOL_V2_REAL = Weights(
        path=K.ZOO_ROOT / "enhance/lle/hvi_cidnet/hvi_cidnet/lol_v2_real/hvi_cidnet_lol_v2_real.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={},
    )
    LOL_V2_SYN = Weights(
        path=K.ZOO_ROOT / "enhance/lle/hvi_cidnet/hvi_cidnet/lol_v2_syn/hvi_cidnet_lol_v2_syn.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={},
    )
    LOL_BLUR = Weights(
        path=K.ZOO_ROOT / "enhance/lle/hvi_cidnet/hvi_cidnet/lol_blur/hvi_cidnet_lol_blur.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={},
    )
    SICE = Weights(
        path=K.ZOO_ROOT / "enhance/lle/hvi_cidnet/hvi_cidnet/sice/hvi_cidnet_sice.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={},
    )
    DEFAULT = LOL_V1


# --- Model Variants ---

@MODELS.register(name="hvi_cidnet", metaclass=HVI_CIDNet)
def hvi_cidnet(weights: WeightsLike = "default", *args, **kwargs):
    """Create a HVI-CIDNet model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "hvi_cidnet")
    alpha = kwargs.pop("alpha", 1.0)
    gated = kwargs.pop("gated", False)
    gated2 = kwargs.pop("gated2", False)
    gamma = kwargs.pop("gamma", 1.0)
    return HVI_CIDNet(
        name="hvi_cidnet",
        alpha=alpha,
        gated=gated,
        gated2=gated2,
        gamma=gamma,
        weights=HVI_CIDNet_Weights(weights),
        *args, **kwargs,
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
