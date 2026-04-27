#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DCC-Net Models.

This module provides the DCC-Net definition and pre-trained weights.

References:
    - Paper: "Deep Color Consistent Network for Low Light-Image Enhancement,"
      CVPR 2022.
    - Code: https://github.com/Ian0926/DCC-Net
"""

from __future__ import annotations

__all__ = [
    "DCCNet",
    "DCCNet_Weights",
    "dccnet",
]

from tensordict import TensorDict
from torch import Tensor
from typing_extensions import override

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
from .module import Net

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class DCCNet(ModelRegisterMixin, Model):
    """DCC-Net model for low-light image enhancement.

    References:
        - Paper: "Deep Color Consistent Network for Low Light-Image Enhancement,"
          CVPR 2022.
        - Code: https://github.com/Ian0926/DCC-Net
    """

    arch: str = "dccnet"
    name: str = "dccnet"
    tasks: list[Task] = [Task.LLE]
    strategies: list[Strategy] = [Strategy.RESIZE, Strategy.PATCH]
    model_dir: Path = current_dir

    in_keys: set = {"image"}
    out_keys: set = {"enhanced"}
    debug_keys: set = {"gray", "color_hist"}

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        d_hist: int = 64,
        weights: Weights | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            d_hist (int, optional): Number of histogram bins for the C-Net.
                Defaults to 64.
            weights (Weights, optional): Pre-trained weights to load.
                Defaults to None.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose

        # Define network
        self.module = Net(d_hist=d_hist)

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
                - gray (Tensor): Grayscale image tensor of shape (B, 1, H, W)
                  and values ranging from 0.0 to 1.0.
                - color_hist (Tensor): Color histogram tensor of shape
                  (B, d_hist * 3) where d_hist is the number of histogram bins
                  per channel.
        """
        if use_patch:
            return self.forward_patch(image=image, *args, **kwargs)
        else:
            return self.forward_step(image=image, *args, **kwargs)

    @override
    def forward_step(self, image: Tensor, *args, **kwargs) -> tuple[Tensor, ...]:
        """Perform a single forward step of the model.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.

        Returns:
            tuple[Tensor, ...]: A tuple containing:

                - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - gray (Tensor): Grayscale image tensor of shape (B, 1, H, W)
                  and values ranging from 0.0 to 1.0.
                - color_hist (Tensor): Color histogram tensor of shape
                  (B, d_hist * 3) where d_hist is the number of histogram bins
                  per channel.
        """
        # 1. Network forward
        enhanced, gray, color_hist = self.module(image)

        # 2. Return final and intermediate results for debugging
        return enhanced, gray, color_hist

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
                - gray (Tensor): Grayscale image tensor of shape (B, 1, H, W)
                  and values ranging from 0.0 to 1.0.
                - color_hist (Tensor): Color histogram tensor of shape
                  (B, d_hist * 3) where d_hist is the number of histogram bins
                  per channel.
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
                "gray": outputs[1],
                # "color_hist": outputs[2],
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

@WEIGHTS.register(name="dccnet")
class DCCNet_Weights(WeightsEnum):

    SICE = Weights(
        path=K.ZOO_ROOT / "enhance/lle/dccnet/dccnet/lol_v1/dccnet_lol_v1.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={}
    )
    DEFAULT = SICE


# --- Model Variants ---

@MODELS.register(name="dccnet", metaclass=DCCNet)
def dccnet(weights: WeightsLike = "default", *args, **kwargs):
    """Create a DCC-Net model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "dccnet")
    d_hist = kwargs.pop("d_hist", 64)
    return DCCNet(
        name="dccnet",
        d_hist=d_hist,
        weights=DCCNet_Weights(weights),
        *args, **kwargs,
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
