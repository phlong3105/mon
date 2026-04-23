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
from typing_extensions import override

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
    strategies: list[Strategy] = [Strategy.RESIZE]
    model_dir: Path = current_dir

    in_keys: set = {"image"}
    out_keys: set = {"enhanced"}

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
        enhanced, gray, color_hist = self.module(image)

        # 3. Return final and intermediate results for debugging
        outputs = {
            "enhanced": enhanced,
            "gray": gray,
            "color_hist": color_hist,
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
            else:
                resize = T.ResizeDivisibleBy(divisor=32)
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
