#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SCI Models.

This module provides the SCI definition and pre-trained weights.

References:
    - Paper: "Toward Fast, Flexible, and Robust Low-Light Image Enhancement,"
      CVPR 2022.
    - Code: https://github.com/vis-opt-group/SCI

    - Paper: "Learning with Self-Calibrator for Fast and Robust Low-Light
      Image Enhancement," TPAMI 2025.
    - Code: https://github.com/vis-opt-group/SCI
"""

from __future__ import annotations

__all__ = [
    "SCI",
    "SCI_PP",
    "SCI_PP_Weights",
    "SCI_Weights",
    "sci",
    "sci_pp",
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
from .module import (
    CalibrateNetwork,
    CalibrateNetworkPP,
    EnhanceNetwork,
    EnhanceNetwork_Ha,
    EnhanceNetwork_Hb,
)

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class SCI(ModelRegisterMixin, Model):
    """SCI model for low-light image enhancement.

    References:
        - Paper: "Toward Fast, Flexible, and Robust Low-Light Image
        Enhancement,"
          CVPR 2022.
        - Code: https://github.com/vis-opt-group/SCI
    """

    arch: str = "sci"
    name: str = "sci"
    tasks: list[Task] = [Task.LLE]
    strategies: list[Strategy] = [Strategy.RESIZE]
    model_dir: Path = current_dir

    in_keys: set = {"image"}
    out_keys: set = {"enhanced"}

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        stage: int = 3,
        weights: Weights | None = None,
        verbose: bool = True,
        *args, **kwargs,
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            stage (int, optional): Number of enhancement stages. Defaults to 3.
            weights (Weights, optional): Pre-trained weights to load.
                Defaults to None.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose
        self.stage = stage

        # Define network
        self.enhance = EnhanceNetwork(layers=1, channels=3)
        self.calibrate = CalibrateNetwork(layers=3, channels=16)

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
    def forward(self, data: TensorDict, inference: bool = True) -> TensorDict:
        """Forward the input through the network.

        Args:
            data (TensorDict): Input data dictionary.
            inference (bool, optional): If True, return enhanced image only.
                Defaults to True.

        Returns:
            TensorDict: Output data dictionary.
        """
        # 1. Extract input data
        x = data["image"]

        # 2. Network forward
        if inference:
            i = self.enhance(x)
            r = x / i
            r = torch.clamp(r, 0.0, 1.0)
            outputs = {
                "enhanced": r,
                "illumination": i,
            }
        else:
            i_list, r_list, x_list, a_list = [], [], [], []
            for i in range(self.stage):
                x_list.append(x)
                i = self.enhance(x)
                r = x / i
                r = torch.clamp(r, 0, 1)
                att = self.calibrate(r)
                x = x + att
                i_list.append(i)
                r_list.append(r)
                a_list.append(torch.abs(att))
            outputs = {
                "x_list": torch.stack(x_list, dim=0),
                "i_list": torch.stack(i_list, dim=0),
                "r_list": torch.stack(r_list, dim=0),
                "a_list": torch.stack(a_list, dim=0),
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


class SCI_PP(ModelRegisterMixin, Model):
    """SCI++ model for low-light image enhancement.

    References:
        - Paper: "Learning with Self-Calibrator for Fast and Robust Low-Light
          Image Enhancement," TPAMI 2025.
        - Code: https://github.com/vis-opt-group/SCI
    """

    arch: str = "sci"
    name: str = "sci++"
    tasks: list[Task] = [Task.LLE]
    strategies: list[Strategy] = [Strategy.RESIZE]
    model_dir: Path = current_dir

    in_keys: set = {"image"}
    out_keys: set = {"enhanced"}

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        stage: int = 3,
        weights: Weights | None = None,
        verbose: bool = True,
        *args, **kwargs,
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            stage (int, optional): Number of enhancement stages. Defaults to 3.
            weights (Weights, optional): Pre-trained weights to load.
                Defaults to None.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose
        self.stage = stage

        # Define network
        self.ha = EnhanceNetwork_Ha(layers=1, channels=3)
        self.hb = EnhanceNetwork_Hb(layers=3, channels=16)
        self.calibrate = CalibrateNetworkPP(layers=3, channels=16)

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
    def forward(self, data: TensorDict, inference: bool = True) -> TensorDict:
        """Forward the input through the network.

        Args:
            data (TensorDict): Input data dictionary.
            inference (bool, optional): If True, return enhanced image only.
                Defaults to True.

        Returns:
            TensorDict: Output data dictionary.
        """
        # 1. Extract input data
        x = data["image"]

        # 2. Network forward
        if inference:
            i = self.ha(x)
            r = x / i
            r = torch.clamp(r, 0.0, 1.0)
            outputs = {
                "enhanced": r,
                "illumination": i,
            }
        else:
            i_list, r_list, x_list, a_list = [], [], [], []

            i = self.ha(x)
            r = x / i
            r = torch.clamp(r, 0, 1)
            i_list.append(i)
            r_list.append(r)
            x_list.append(x)

            for i in range(self.stage):
                x_list.append(i)
                att = self.calibrate(r)
                att_1 = self.hb(att)

                i = i + att + att_1
                r = x / i
                r = torch.clamp(r, 0, 1)

                i_list.append(i)
                r_list.append(r)
                a_list.append(torch.abs(att))

            outputs = {
                "x_list": torch.stack(x_list, dim=0),
                "i_list": torch.stack(i_list, dim=0),
                "r_list": torch.stack(r_list, dim=0),
                "a_list": torch.stack(a_list, dim=0),
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

@WEIGHTS.register(name="sci")
class SCI_Weights(WeightsEnum):

    EASY = Weights(
        path=K.ZOO_ROOT / "enhance/lle/sci/sci/pretrained/sci_easy.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={},
    )
    MEDIUM = Weights(
        path=K.ZOO_ROOT / "enhance/lle/sci/sci/pretrained/sci_medium.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={},
    )
    DIFFICULT = Weights(
        path=K.ZOO_ROOT / "enhance/lle/sci/sci/pretrained/sci_difficult.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={},
    )
    DEFAULT = MEDIUM


@WEIGHTS.register(name="sci++")
class SCI_PP_Weights(WeightsEnum):

    DEFAULT = Weights(
        path=K.ZOO_ROOT / "enhance/lle/sci/sci++/pretrained/sci++_1_3500.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={},
    )


# --- Model Variants ---

@MODELS.register(name="sci", metaclass=SCI)
def sci(weights: WeightsLike = "default", *args, **kwargs):
    """Create an SCI model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "sci")
    stage = kwargs.pop("stage", 3)
    return SCI(
        name="sci",
        stage=stage,
        weights=SCI_Weights(weights),
        *args, **kwargs,
    )


@MODELS.register(name="sci++", metaclass=SCI_PP)
def sci_pp(weights: WeightsLike = "default", *args, **kwargs):
    """Create an SCI++ model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "sci++")
    stage = kwargs.pop("stage", 3)
    return SCI_PP(
        name="sci++",
        stage=stage,
        weights=SCI_PP_Weights(weights),
        *args, **kwargs,
    )


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
