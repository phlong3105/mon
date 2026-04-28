#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RetinexNet Models.

This module provides the RetinexNet definition and pre-trained weights.

References:
    - Paper: "Deep Retinex Decomposition for Low-Light Enhancement," BMCV 2018.
    - Code: https://github.com/aasharma90/RetinexNet_PyTorch
"""

from __future__ import annotations

__all__ = [
    "RetinexNet",
    "RetinexNet_Weights",
    "retinexnet",
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
from .module import DecomNet, EnhanceNet

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class RetinexNet(ModelRegisterMixin, Model):
    """RetinexNet model for low-light image enhancement.

    References:
        - Paper: "Deep Retinex Decomposition for Low-Light Enhancement," BMCV 2018.
        - Code: https://github.com/aasharma90/RetinexNet_PyTorch
    """

    arch: str = "retinexnet"
    name: str = "retinexnet"
    tasks: list[Task] = [Task.LLE]
    strategies: list[Strategy] = [Strategy.RESIZE, Strategy.PATCH]
    model_dir: Path = current_dir

    in_keys: set = {"image"}
    out_keys: set = {"enhanced"}
    debug_keys: set = {"R", "L", "L_delta"}

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
        self.decom_net = DecomNet()
        self.enhance_net = EnhanceNet()

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
        decom: bool = False,
        use_patch: bool = False,
        *args, **kwargs
    ) -> tuple[Tensor, ...]:
        """Route the inputs through the model's different forward methods based
        on the context.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            decom (bool, optional): Whether to perform decomposition only.
                Defaults to False.
            use_patch (bool, optional): Whether to use patch-based strategy.
                Defaults to False.

        Returns:
            tuple[Tensor, ...]: A tuple containing:

                - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - R (Tensor): Reflectance component tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - L (Tensor): Illumination component tensor of shape (B, 1, H, W)
                  and values ranging from 0.0 to 1.0.
                - L_delta (Tensor): Illumination adjustment tensor of shape
                  (B, 1, H, W) and values ranging from -1.0 to 1.0.
        """
        if use_patch:
            return self.forward_patch(image=image, decom=decom, *args, **kwargs)
        else:
            return self.forward_step(image=image, decom=decom, *args, **kwargs)

    @override
    def forward_step(
        self,
        image: Tensor,
        decom: bool = False,
        *args, **kwargs
    ) -> tuple[Tensor, ...]:
        """Perform a single forward step of the model.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            decom (bool, optional): Whether to perform decomposition only.
                Defaults to False.

        Returns:
            tuple[Tensor, ...]: A tuple containing:

                - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - R (Tensor): Reflectance component tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - L (Tensor): Illumination component tensor of shape (B, 1, H, W)
                  and values ranging from 0.0 to 1.0.
                - L_delta (Tensor): Illumination adjustment tensor of shape
                  (B, 1, H, W) and values ranging from -1.0 to 1.0.
        """
        # 1. Decomposition
        R, L = self.decom_net(image)

        if decom:
            return image, R, L, L
        else:
            # 2. Relighting
            L_delta = self.enhance_net(R, L)
            L_delta_3 = torch.cat((L_delta, L_delta, L_delta), dim=1)
            # 3. Reconstruction
            enhanced = R * L_delta_3
            return enhanced, R, L, L_delta

    def forward_patch(
        self,
        image: Tensor,
        decom: bool = False,
        patcher: dict | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, ...]:
        """Forward the input through the network using the patch-based strategy.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            decom (bool, optional): Whether to perform decomposition only.
                Defaults to False.
            patcher (dict, optional): A dictionary containing the patching
                configuration, such as patch size and stride. Defaults to None
                means using the default patcher.

        Returns:
            tuple[Tensor, ...]: A tuple containing:

                - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - R (Tensor): Reflectance component tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - L (Tensor): Illumination component tensor of shape (B, 1, H, W)
                  and values ranging from 0.0 to 1.0.
                - L_delta (Tensor): Illumination adjustment tensor of shape
                  (B, 1, H, W) and values ranging from -1.0 to 1.0.
        """
        # 1. Initialize image patcher
        patcher: dict = patcher or {"name": "uniform"}
        patcher: ImagePatcher = PATCHERS.build(image=image, **patcher)

        # 2. Iterate and Process
        for patch, x, y in patcher:
            # 2.1. Process the patch
            outputs = self.forward_step(image=patch, decom=decom, *args, **kwargs)
            patch_outputs = {
                "enhanced": outputs[0],
                "R": outputs[1],
                "L": outputs[2],
                "L_delta": outputs[3],
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
        data = TensorDict({"image": dummy_input}, batch_size=dummy_input.shape)
        inputs = {"data": data}

        # Benchmark the model
        return benchmark(model=self, inputs=inputs, *args, **kwargs)

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="retinexnet")
class RetinexNet_Weights(WeightsEnum):

    LOL_V1 = Weights(
        path=K.ZOO_ROOT / "enhance/lle/retinexnet/retinexnet/lol_v1/retinexnet_lol_v1.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={}
    )
    DEFAULT = LOL_V1


# --- Model Variants ---

@MODELS.register(name="retinexnet", metaclass=RetinexNet)
def retinexnet(weights: WeightsLike = "default", *args, **kwargs):
    """Create a RetinexNet model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "retinexnet")
    return RetinexNet(
        name="retinexnet",
        weights=RetinexNet_Weights(weights),
        *args, **kwargs,
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
