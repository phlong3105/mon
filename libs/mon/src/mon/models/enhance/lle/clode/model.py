#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CLODE Models.

This module provides the CLODE definition and pre-trained weights.

References:
    - Paper: "Continuous Exposure Learning for Low-light Image Enhancement using
      Neural ODEs," ICLR 2025.
    - Code: https://github.com/dgjung0220/CLODE
"""

from __future__ import annotations

__all__ = [
    "CLODE",
    "CLODE_Weights",
    "clode",
]

from typing import override

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
from .module import NODE

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class CLODE(ModelRegisterMixin, Model):
    """CLODE model for low-light image enhancement.

    References:
        - Paper: "Continuous Exposure Learning for Low-light Image Enhancement using
          Neural ODEs," ICLR 2025.
        - Code: https://github.com/dgjung0220/CLODE
    """

    arch: str = "clode"
    name: str = "clode"
    tasks: list[Task] = [Task.LLE]
    strategies: list[Strategy] = [Strategy.RESIZE, Strategy.PATCH]
    model_dir: Path = current_dir

    in_keys: set = {"image"}
    out_keys: set = {"enhanced"}
    debug_keys: set = {"curve_map", "noise_map"}

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        num_filters: int = 32,
        tol: float = 1e-5,
        adjoint: bool = True,
        weights: Weights | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            num_filters (int, optional): Number of filters in the convolutional
                layers. Defaults to 32.
            tol (float, optional): Tolerance for ODE solver. Defaults to 1e-5.
            adjoint (bool, optional): Whether to use the adjoint method for
                backpropagation. Defaults to True.
            weights (Weights, optional): Pre-trained weights to load.
                Defaults to None.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose

        # Define network
        self.model = NODE(num_filters=num_filters, tol=tol, adjoint=adjoint)

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
        eval_T: Tensor | None = None,
        inference: bool = True,
        use_patch: bool = False,
        *args, **kwargs
    ) -> TensorDict:
        """Forward the input through the network.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            eval_T (Tensor, optional): Evaluation time for the ODE solver.
                Defaults to None, which means it will be determined by the model.
            inference (bool, optional): Whether the forward step is for inference.
                Defaults to True.
            use_patch (bool, optional): Whether to use patch-based strategy.
                Defaults to False.

        Returns:
            tuple[Tensor, ...]: A tuple containing:

                - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - curve_map (Tensor): The estimated curve parameters of shape
                  (B, C*8, H, W) and values ranging from -1.0 to 1.0.
                - noise_map (Tensor): The estimated noise map of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
        """
        if use_patch:
            return self.forward_patch(
                image=image,
                eval_T=eval_T,
                inference=inference,
                *args, **kwargs
            )

        return self.forward_step(
            image=image,
            eval_T=eval_T,
            inference=inference,
            *args, **kwargs
        )

    @override
    def forward_step(
        self,
        image: Tensor,
        eval_T: Tensor | None = None,
        inference: bool = True,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Perform a single forward step of the model.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            eval_T (Tensor, optional): Evaluation time for the ODE solver.
                Defaults to None, which means it will be determined by the model.
            inference (bool, optional): Whether the forward step is for inference.
                Defaults to True.

        Returns:
            tuple[Tensor, ...]: A tuple containing:

                - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - curve_map (Tensor): The estimated curve parameters of shape
                  (B, C*8, H, W) and values ranging from -1.0 to 1.0.
                - noise_map (Tensor): The estimated noise map of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
        """
        outputs = self.model(image, eval_T, inference)
        return tuple(outputs.values())

    def forward_patch(
        self,
        image: Tensor,
        eval_T: Tensor | None = None,
        inference: bool = True,
        patcher: dict | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, ...]:
        """Forward the input through the network using the patch-based strategy.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            eval_T (Tensor, optional): Evaluation time for the ODE solver.
                Defaults to None, which means it will be determined by the model.
            inference (bool, optional): Whether the forward step is for inference.
                Defaults to True.
            patcher (dict, optional): A dictionary containing the patching
                configuration, such as patch size and stride. Defaults to None
                means using the default patcher.

        Returns:
            tuple[Tensor, ...]: A tuple containing:

                - enhanced (Tensor): Enhanced image tensor of shape (B, C, H, W)
                  and values ranging from 0.0 to 1.0.
                - r (Tensor): The estimated curve parameters of shape (B, C*8, H, W)
                  and values ranging from -1.0 to 1.0.
        """
        # 1. Initialize image patcher
        patcher: dict = patcher or {"name": "hann_window"}
        patcher: ImagePatcher = PATCHERS.build(image=image, **patcher)

        # 2. Iterate and Process
        for patch, x, y in patcher:
            # 2.1. Process the patch
            outputs = self.forward_step(
                image=patch,
                eval_T=eval_T,
                inference=inference,
                *args, **kwargs
            )
            patch_outputs = {
                "enhanced": outputs[0],
                "curve_map": outputs[1],
                "noise_map": outputs[2],
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

@WEIGHTS.register(name="clode")
class CLODE_Weights(WeightsEnum):

    SICE_ME = Weights(
        path=K.ZOO_ROOT / "enhance/lle/clode/clode/sice_me/clode_sice_me.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={}
    )
    LOL_V1 = Weights(
        path=K.ZOO_ROOT / "enhance/lle/clode/clode/lol_v1/clode_lol_v1.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={}
    )
    UNIVERSAL = Weights(
        path=K.ZOO_ROOT / "enhance/lle/clode/clode/universal/clode_universal.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={}
    )
    DEFAULT = SICE_ME


# --- Model Variants ---

@MODELS.register(name="clode", metaclass=CLODE)
def clode(weights: WeightsLike = "default", *args, **kwargs):
    """Create a CLODE model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "clode")
    num_filters = kwargs.pop("num_filters", 32)
    tol = kwargs.pop("tol", 1e-5)
    adjoint = kwargs.pop("adjoint", True)
    return CLODE(
        name="clode",
        num_filters=num_filters,
        tol=tol,
        adjoint=adjoint,
        weights=CLODE_Weights(weights),
        *args, **kwargs,
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
