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
    Path,
    Task,
    WEIGHTS,
    Weights,
    WeightsEnum,
    WeightsLike,
)
from mon.models.enhance.base import EnhancementModel
from mon.nn import ModelRegisterMixin
from .module import NODE

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class CLODE(ModelRegisterMixin, EnhancementModel):
    """CLODE model for low-light image enhancement.

    References:
        - Paper: "Continuous Exposure Learning for Low-light Image Enhancement using
          Neural ODEs," ICLR 2025.
        - Code: https://github.com/dgjung0220/CLODE
    """

    arch: str = "clode"
    name: str = "clode"
    tasks: list[Task] = [Task.LLE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        num_filters: int = 32,
        tol: float = 1e-5,
        adjoint: bool = True,
        weights: WeightsLike | None = None,
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
            weights (WeightsLike, optional): Pre-trained weights to load.
                Defaults to None.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose

        # Define network
        self.model = NODE(num_filters=num_filters, tol=tol, adjoint=adjoint)

        # Load weights
        if is_weights_type(weights):
            self.model.load_state_dict(weights.state_dict())
            if self.verbose:
                log(f"Initialized '{name}' from weights: '{weights.path}'.")
        else:
            if self.verbose:
                log(f"Initialized '{name}' from scratch.")

    # --- Callable & Context Manager ---
    @override
    def forward_step(
        self,
        data: TensorDict,
        eval_time: Tensor | None = None,
        inference: bool = True,
        *args, **kwargs
    ) -> TensorDict:
        """Forward the input through the network.

        Args:
            data (TensorDict): Input data dictionary.
            eval_time (Tensor, optional): Evaluation time for the ODE solver.
                Defaults to None, which means it will be determined by the model.
            inference (bool, optional): Whether the forward step is for inference.
                Defaults to True.

        Returns:
            TensorDict: Output data dictionary.
        """
        # 1. Extract input data
        x = data["image"]

        # 2. Network forward
        outputs = self.model(x, eval_time, inference)

        # 3. Return final and intermediate results for debugging
        return TensorDict(outputs, batch_size=[])

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
        weights (WeightsLike, optional): Pre-trained weights to load.
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
