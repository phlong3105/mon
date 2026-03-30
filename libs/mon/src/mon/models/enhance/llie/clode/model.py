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
    tasks: list[Task] = [Task.LLIE]
    model_dir: Path = current_dir
    requires: set = {"image", "eval_time"}

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
        # Satisfy PyTorch's empty signature first.
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
    def forward_step(self, data: dict, *args, **kwargs) -> dict:
        """Forward the input through the network.

        Args:
            data (dict): Input data dictionary.

        Returns:
            dict: Output data dictionary.
        """
        x = data["image"]
        eval_time = data["eval_time"]
        inference = data.get("inference", False)
        return self.model(x, eval_time, inference)

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="clode")
class CLODE_Weights(WeightsEnum):

    SICE_ME = Weights(
        path=K.ZOO_ROOT / "enhance/llie/clode/clode/sice_me/clode_sice_me.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={}
    )
    LOL_V1 = Weights(
        path=K.ZOO_ROOT / "enhance/llie/clode/clode/lol_v1/clode_lol_v1.pt",
        url=None,
        num_classes=None,
        transforms=None,
        meta={}
    )
    UNIVERSAL = Weights(
        path=K.ZOO_ROOT / "enhance/llie/clode/clode/universal/clode_universal.pt",
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
