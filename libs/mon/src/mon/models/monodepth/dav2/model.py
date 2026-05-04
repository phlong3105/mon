#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Depth Anything V2.

This module provides the Depth Anything V2 definition and pre-trained weights.

References:
    - Paper: "Depth Anything V2. A More Capable Foundation Model for Monocular
      Depth Estimation," NeurIPS 2024.
    - Code: https://github.com/DepthAnything/Depth-Anything-V2
"""

from __future__ import annotations

__all__ = [
    "DAV2",
    "DAV2_ViTB_Weights",
    "DAV2_ViTL_Weights",
    "DAV2_ViTS_Weights",
    "dav2_vitb",
    "dav2_vitl",
    "dav2_vits",
]

import sys
from typing import override

import numpy as np
import torch
from numpy import ndarray
from tensordict import TensorDict
from torch import Tensor

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
from mon.nn import Model, ModelRegisterMixin
from mon.ops import normalize_minmax

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]
extern_path = current_dir / "extern" / "dav2"
if str(extern_path) not in sys.path:
    sys.path.append(str(extern_path))

try:
    # Now we can safely import from the original repository
    from depth_anything_v2 import dpt
except ImportError:
    raise ImportError(f"Failed to import 'depth_anything_v2' from the 'extern/dav2' directory.")


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class DAV2(ModelRegisterMixin, Model):
    """DAV2 model for monocular depth estimation.

    References:
        - Paper: "Depth Anything V2. A More Capable Foundation Model for Monocular
          Depth Estimation," NeurIPS 2024.
        - https://github.com/DepthAnything/Depth-Anything-V2
    """

    arch: str = "dav2"
    name: str = "dav2"
    tasks: list[Task] = [Task.MONODEPTH]
    strategies: list[Strategy] = [Strategy.RESIZE]
    model_dir: Path = current_dir

    in_keys: set = {"image"}
    out_keys: set = {"depth"}
    debug_keys: set = {}

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        encoder: str,
        features: int,
        out_channels: list[int],
        use_bn: bool = False,
        use_clstoken: bool = False,
        device: torch.device = torch.device("cpu"),
        weights: Weights | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): The name of the model.
            encoder (str): The type of encoder to use (e.g., "vits", "vitb", "vitl").
            features (int): The number of features in the encoder.
            out_channels (list[int]): The number of output channels for each
                stage.
            use_bn (bool, optional): Whether to use batch normalization.
                Defaults to False.
            use_clstoken (bool, optional): Whether to use a CLS token.
                Defaults to False.
            device (torch.device, optional): The device to load the model on.
                Defaults to CPU.
            weights (Weights, optional): Pre-trained weights to load.
                Defaults to None.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose

        # Define model
        base_model = dpt.DepthAnythingV2(
            encoder=encoder,
            features=features,
            out_channels=out_channels,
            use_bn=use_bn,
            use_clstoken=use_clstoken,
            device=device,  # This is for the input image in forward()
        ).to(device)

        # Load weights
        if weights is not None and is_weights_type(weights):
            base_model.load_state_dict(weights.state_dict(weights_only=True))
            if self.verbose:
                log(f"initialized {name} from weights {weights.path.as_posix()}.")
        else:
            if self.verbose:
                log(f"initialized {name} from scratch.")

        # Assign the base model
        self.model = base_model

    # --- Callable & Context Manager ---
    @override
    def forward(
        self,
        image: Tensor | ndarray,
        input_size: int = 518,
        *args, **kwargs
    ) -> ndarray:
        """Route the inputs through the model's different forward methods based
        on the context.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            input_size (int, optional): The size to which the input image should
                be resized before being fed into the model. Defaults to 518.

        Returns:
            TensorDict: Output data dictionary.
        """
        return self.forward_step(image=image, *args, **kwargs)

    @override
    def forward_step(
        self,
        image: Tensor | ndarray,
        input_size: int = 518,
        *args, **kwargs
    ) -> ndarray:
        """Perform a single forward step of the model.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            input_size (int, optional): The size to which the input image should
                be resized before being fed into the model. Defaults to 518.

        Returns:
            ndarray: The predicted depth map as a NumPy array of shape (H, W)
                and values ranging from 0 to 255.
        """
        x = image

        # 1. Network forward
        if isinstance(x, Tensor):
            depth = self.model(x)
        elif isinstance(x, ndarray):
            depth = self.model.infer_image(x, *args, **kwargs)
        else:
            raise TypeError(
                f"Expected input to be a Tensor or ndarray, "
                f"but got: {type(x).__name__}."
            )
        depth = normalize_minmax(depth) * 255.0
        depth = depth.astype(np.uint8)

        # 2. Return final and intermediate results for debugging
        return depth

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
        pass

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="dav2_vits")
class DAV2_ViTS_Weights(WeightsEnum):

    DA_2K = Weights(
        path=K.ZOO_ROOT / "monodepth/dav2/dav2_vits/da2k/dav2_vits_da2k.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={}
    )
    DEFAULT = DA_2K


@WEIGHTS.register(name="dav2_vitb")
class DAV2_ViTB_Weights(WeightsEnum):

    DA_2K = Weights(
        path=K.ZOO_ROOT / "monodepth/dav2/dav2_vitb/da2k/dav2_vitb_da2k.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={}
    )
    DEFAULT = DA_2K


@WEIGHTS.register(name="dav2_vitl")
class DAV2_ViTL_Weights(WeightsEnum):

    DA_2K = Weights(
        path=K.ZOO_ROOT / "monodepth/dav2/dav2_vitl/da2k/dav2_vitl_da2k.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={}
    )
    DEFAULT = DA_2K


# --- Model Variants ---

@MODELS.register(name="dav2_vits", metaclass=DAV2)
def dav2_vits(weights: WeightsLike = "default", *args, **kwargs):
    """Create a DAV2 model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    return DAV2(
        name="dav2_vits",
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
        weights=DAV2_ViTS_Weights(weights),
        *args, **kwargs
    )


@MODELS.register(name="dav2_vitb", metaclass=DAV2)
def dav2_vitb(weights: WeightsLike = "default", *args, **kwargs):
    """Create a DAV2 model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    return DAV2(
        name="dav2_vitb",
        encoder="vitb",
        features=128,
        out_channels=[96, 192, 384, 768],
        weights=DAV2_ViTB_Weights(weights),
        *args, **kwargs
    )


@MODELS.register(name="dav2_vitl", metaclass=DAV2)
def dav2_vitl(weights: WeightsLike = "default", *args, **kwargs):
    """Create a DAV2 model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    return DAV2(
        name="dav2_vitl",
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
        weights=DAV2_ViTL_Weights(weights),
        *args, **kwargs
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
