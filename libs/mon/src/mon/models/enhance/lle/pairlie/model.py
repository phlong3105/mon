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

from mon.core import (
    Config,
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
from mon.dataset import transform as T
from mon.metrics import benchmark, create_dummy_image
from mon.nn import Model, ModelRegisterMixin
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
    strategies: list[Strategy] = [Strategy.RESIZE]
    model_dir: Path = current_dir

    in_keys: set = {"image"}
    out_keys: set = {"enhanced"}

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
    def forward(self, data: TensorDict) -> TensorDict:
        """Forward the input through the network.

        Args:
            data (TensorDict): Input data dictionary.

        Returns:
            TensorDict: Output data dictionary.
        """
        # 1. Extract input data
        image = data["image"]

        # 2. Network forward
        X = self.N_net(image)
        L = self.L_net(X)
        R = self.R_net(X)
        D = image - X
        I = torch.pow(L, self.alpha) * R  # default=0.2, LOL=0.14.

        # 3. Return final and intermediate results for debugging
        outputs = {
            "enhanced": I,
            "L": L,
            "R": R,
            "X": X,
            "D": D,
        }
        return TensorDict(outputs, batch_size=[])

    # --- Interfaces ---
    @override
    def build_transforms(self, config: Config | None = None) -> T.Compose:
        """Define the model's transformations.

        Args:
            config (Config, optional): The configuration object containing any
                necessary parameters for defining the transformations.
                Defaults to None.

        Returns:
            Callable: A callable (e.g., a torchvision transform or a custom
                function) that takes in the raw input data and returns the
                transformed data ready for the forward step.
        """
        transforms = T.Compose([
            T.Normalize(normalization="min_max"),
            T.ToTensorV2(transpose_mask=True),
        ])

        if config is not None:
            if config.strategy in [Strategy.RESIZE]:
                imgsz = config.imgsz
                resize = T.ResizeDivisibleBy(height=imgsz.h, width=imgsz.w, divisor=32)
                transforms = resize + transforms

        return transforms

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
