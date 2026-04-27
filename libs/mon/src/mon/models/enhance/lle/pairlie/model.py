#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""PairLIE Models.

This module provides the PairLIE definition and pre-trained weights.

References:
    - Paper: "Learning a Simple Low-light Image Enhancer from Paired Low-light
      Instances," CVPR 2023.
    - Code: https://github.com/zhenqifu/PairLIE
"""

from __future__ import annotations

__all__ = [
    "PairLIE",
    "PairLIE_Weights",
    "pairlie",
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
from .module import L_Net, N_Net, R_Net

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class PairLIE(ModelRegisterMixin, Model):
    """PairLIE model for low-light image enhancement.

    References:
        - Paper: "Learning a Simple Low-light Image Enhancer from Paired Low-light
          Instances," CVPR 2023.
        - Code: https://github.com/zhenqifu/PairLIE
    """

    arch: str = "pairlie"
    name: str = "pairlie"
    tasks: list[Task] = [Task.LLE]
    strategies: list[Strategy] = [Strategy.RESIZE, Strategy.PATCH]
    model_dir: Path = current_dir

    in_keys: set = {"image"}
    out_keys: set = {"enhanced"}
    debug_keys: set = {"L", "R", "X", "D"}

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        alpha: float = 0.2,  # default=0.2, LOL=0.14.
        weights: Weights | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            alpha (float, optional): The illumination correction factor (alpha).
                Defaults to 0.2, as used in the original paper.
            weights (Weights, optional): Pre-trained weights to load.
                Defaults to None.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose
        self.alpha = alpha

        # Define network
        self.L_net = L_Net(num_channels=64)
        self.R_net = R_Net(num_channels=64)
        self.N_net = N_Net(num_channels=64)

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
    def forward(
        self,
        image: Tensor,
        use_patch: bool = False,
        *args, **kwargs
    ) -> tuple[Tensor, ...]:
        """Route the inputs through the model's different forward methods based
        on the context.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            use_patch (bool, optional): Whether to use patch-based strategy.
                Defaults to False.

        Returns:
            tuple[Tensor, ...]: A tuple containing:

                - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - L (Tensor): The illumination map predicted by the L-Net.
                - R (Tensor): The reflectance map predicted by the R-Net.
                - X (Tensor): The intermediate feature map from the N-Net.
                - D (Tensor): The difference between the input image and the
                  intermediate feature map (D = image - X).
        """
        if use_patch:
            return self.forward_patch(image=image, *args, **kwargs)
        else:
            return self.forward_step(image=image, *args, **kwargs)

    @override
    def forward_step(self, image: Tensor, *args, **kwargs) -> tuple[Tensor, ...]:
        """Forward the input through the network.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.

        Returns:
            tuple[Tensor, ...]: A tuple containing:

                - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - L (Tensor): The illumination map predicted by the L-Net.
                - R (Tensor): The reflectance map predicted by the R-Net.
                - X (Tensor): The intermediate feature map from the N-Net.
                - D (Tensor): The difference between the input image and the
                  intermediate feature map (D = image - X).
        """
        # 1. Network forward
        X = self.N_net(image)
        L = self.L_net(X)
        R = self.R_net(X)
        D = image - X
        I = torch.pow(L, self.alpha) * R  # default=0.2, LOL=0.14.

        # 2. Return final and intermediate results for debugging
        return I, L, R, X, D

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
            tuple[Tensor, ...]: A tuple containing:

                - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - L (Tensor): The illumination map predicted by the L-Net.
                - R (Tensor): The reflectance map predicted by the R-Net.
                - X (Tensor): The intermediate feature map from the N-Net.
                - D (Tensor): The difference between the input image and the
                  intermediate feature map (D = image - X).
        """
        # 1. Initialize image patcher
        patcher: dict = patcher or {"name": "uniform"}
        patcher: ImagePatcher = PATCHERS.build(image=image, **patcher)

        # 2. Iterate and Process
        for patch, x, y in patcher:
            # 2.1. Process the patch
            outputs = self.forward_step(image=patch, *args, **kwargs)
            patch_outputs = {
                "enhanced": outputs[0],
                "L": outputs[1],
                "R": outputs[2],
                "X": outputs[3],
                "D": outputs[4],
            }
            # 2.2. Feed result back to Patcher
            patcher(patches=patch_outputs, x=x, y=y)

        # 3. Get the merged results
        return tuple(patcher.output.values())

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

@WEIGHTS.register(name="pairlie")
class PairLIE_Weights(WeightsEnum):

    SICE = Weights(
        path=K.ZOO_ROOT / "enhance/lle/pairlie/pairlie/sice/pairlie_sice.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={}
    )
    DEFAULT = SICE


# --- Model Variants ---

@MODELS.register(name="pairlie", metaclass=PairLIE)
def pairlie(weights: WeightsLike = "default", *args, **kwargs):
    """Create a PairLIE model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "pairlie")
    return PairLIE(
        name="pairlie",
        weights=PairLIE_Weights(weights),
        *args, **kwargs,
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
