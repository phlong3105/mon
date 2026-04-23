#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""TensorMoG Models.

This module provides the TensorMoG definition and pre-trained weights.

References:
    - Paper: "TensorMoG: A Tensor-Driven Gaussian Mixture Model with Dynamic
      Scene Adaptation for Background Modeling," Sensors 2020.
"""

from __future__ import annotations

__all__ = [
    "TensorMOG",
    "tensormog",
]

from typing import override

import torch
from tensordict import TensorDict

from mon.core import Config, MODELS, Path, Size, Strategy, Task
from mon.dataset import transform as T
from mon.metrics import benchmark, create_dummy_image
from mon.nn import Model, ModelRegisterMixin
from .module import HVR

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class TensorMOG(ModelRegisterMixin, Model):
    """TensorMoG model for background subtraction.

    References:
        - Paper: "TensorMoG: A Tensor-Driven Gaussian Mixture Model with Dynamic
          Scene Adaptation for Background Modeling," Sensors 2020.
    """

    arch: str = "tensormog"
    name: str = "tensormog"
    tasks: list[Task] = [Task.BGSUBTRACT]
    strategies: list[Strategy] = [Strategy.RESIZE]
    model_dir: Path = current_dir

    in_keys: set = {"image"}
    out_keys: set = {"background", "foreground"}

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        height: int = 512,
        width: int = 512,
        num_gaussians: int = 3,
        learning_rate: float = 0.02,
        matching_thres: float = 2 * 2,
        background_thres: float = 0.6,
        num_updates: int = 30,
        tau_rate: float = 0.01,
        tau_updating_rate: float = 0.025,
        device: torch.device = torch.device("cpu"),
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            window_size (int): Size of the local window for context aggregation.
            hidden_dim (int): Number of channels in the hidden layers.
            num_layers (int): Total number of layers in the network.
            add_layers (int): Number of additional layers for context aggregation.
            epochs (int, optional): Number of optimization epochs for
                single-image optimization. Defaults to 100.
            device (torch.device, optional): Device to use for computation.
                Defaults to torch.device("cpu").
            verbose (bool, optional): Verbosity mode. Defaults to True.
            *args: Additional positional arguments for the INR model.
            **kwargs: Additional keyword arguments for the INR model.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose
        self.device = device

        # Define network
        self.hvr = HVR(
            height=height,
            width=width,
            num_gaussians=num_gaussians,
            learning_rate=learning_rate,
            matching_thres=matching_thres,
            background_thres=background_thres,
            num_updates=num_updates,
            tau_rate=tau_rate,
            tau_updating_rate=tau_updating_rate,
            device=device,
        )

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
        self.hvr.update(image)
        background = self.hvr.get_background()
        foreground = self.hvr.get_foreground(image)

        # 3. Return final and intermediate results for debugging
        outputs = {
            "background": background,
            "foreground": foreground,
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


# --- Model Variants ---

@MODELS.register(name="tensormog", metaclass=TensorMOG)
def tensormog(*args, **kwargs):
    """Create a TensorMOG model."""
    _ = kwargs.pop("name", "tensormog")
    return TensorMOG(name="tensormog", *args, **kwargs)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
