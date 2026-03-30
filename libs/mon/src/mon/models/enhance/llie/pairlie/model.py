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

from typing import Any, override

import torch

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
from .module import L_Net, N_Net, R_Net

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class PairLIE(ModelRegisterMixin, EnhancementModel):
    """PairLIE model for low-light image enhancement.

    References:
        - Paper: "Learning a Simple Low-light Image Enhancer from Paired Low-light
          Instances," CVPR 2023.
        - Code: https://github.com/zhenqifu/PairLIE
    """

    arch: str = "pairlie"
    name: str = "pairlie"
    tasks: list[Task] = [Task.LLE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        alpha: float = 0.2,  # default=0.2, LOL=0.14.
        weights: WeightsLike | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            alpha (float, optional): The illumination correction factor (alpha).
                Defaults to 0.2, as used in the original paper.
            weights (WeightsLike, optional): Pre-trained weights to load.
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
        if is_weights_type(weights):
            self.load_state_dict(weights.state_dict())
            if self.verbose:
                log(f"Initialized '{name}' from weights: '{weights.path}'.")
        else:
            if self.verbose:
                log(f"Initialized '{name}' from scratch.")

    # --- Callable & Context Manager ---
    @override
    def forward_step(self, data: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
        """Forward the input through the network.

        Args:
            data (dict): Input data dictionary.

        Returns:
            dict: Output data dictionary.
        """
        image = data["image"]
        X = self.N_net(image)
        L = self.L_net(X)
        R = self.R_net(X)
        D = image - X
        I = torch.pow(L, self.alpha) * R  # default=0.2, LOL=0.14.

        # Return final and intermediate results for debugging
        return {
            "enhanced": I,
            "L": L,
            "R": R,
            "X": X,
            "D": D,
        }

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="pairlie")
class PairLIE_Weights(WeightsEnum):

    SICE = Weights(
        path=K.ZOO_ROOT / "enhance/llie/pairlie/pairlie/sice/pairlie_sice.pt",
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
        weights (WeightsLike, optional): Pre-trained weights to load.
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
