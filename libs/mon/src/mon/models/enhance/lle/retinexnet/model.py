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
    strategies: list[Strategy] = [Strategy.RESIZE]
    model_dir: Path = current_dir

    in_keys: set = {"image"}
    out_keys: set = {"enhanced"}

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
    def forward(self, data: TensorDict, decom: bool = False) -> TensorDict:
        """Forward the input through the network.

        Args:
            data (TensorDict): Input data dictionary.
            decom (bool, optional): Whether to perform decomposition only.
                Defaults to False.

        Returns:
            TensorDict: Output data dictionary.
        """
        # 1. Extract input data
        image = data["image"]

        # 2. Network forward
        # Decomposition
        R, L = self.decom_net(image)
        if decom:
            outputs = {
                "R": R,
                "L": L,
            }
        else:
            # Relighting
            L_delta = self.enhance_net(R, L)
            L_delta_3 = torch.cat((L_delta, L_delta, L_delta), dim=1)

            # Reconstruction
            S = R * L_delta_3

            outputs = {
                "enhanced": S,
                "R": R,
                "L": L,
                "L_delta": L_delta,
            }

        # 3. Return final and intermediate results for debugging
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

@WEIGHTS.register(name="retinexnet")
class RetinexNet_Weights(WeightsEnum):

    SICE_ME = Weights(
        path=K.ZOO_ROOT / "enhance/lle/retinexnet/retinexnet/lol_v1/retinexnet_lol_v1.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={}
    )
    DEFAULT = SICE_ME


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
