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

import torch
from torch import nn, Tensor

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
from mon.nn import ModelRegisterMixin
from .module import L_net, N_net, R_net

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class PairLIE(ModelRegisterMixin, nn.Module):
    """PairLIE model for low-light image enhancement.

    References:
        - Paper: "Learning a Simple Low-light Image Enhancer from Paired Low-light
          Instances," CVPR 2023.
        - Code: https://github.com/zhenqifu/PairLIE
    """

    arch: str = "pairlie"
    name: str = "pairlie"
    tasks: list[Task] = [Task.ENHANCE]
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
        self.L_net = L_net(num_channels=64)
        self.R_net = R_net(num_channels=64)
        self.N_net = N_net(num_channels=64)

        # Load weights
        if is_weights_type(weights):
            self.load_state_dict(weights.state_dict())
            if self.verbose:
                log(f"Initialized '{name}' from weights: '{weights.path}'.")
        else:
            if self.verbose:
                log(f"Initialized '{name}' from scratch.")

    # --- Callable & Context Manager ---
    def forward(self, image: Tensor) -> dict:
        """Forward the input through the network.

        Args:
            image (Tensor): Image tensor of shape (B, 3, H, W) and values
                ranging from 0.0 to 1.0.

        Returns:
            dict: Dictionary containing the enhanced image tensor and
                intermediate results for debugging.
        """
        X = self.N_net(image)
        L = self.L_net(X)
        R = self.R_net(X)
        D = image - X
        I = torch.pow(L, self.alpha) * R  # default=0.2, LOL=0.14.

        # Return final and intermediate results for debugging
        outputs = {
            "enhanced": I,
            "L": L,
            "R": R,
            "X": X,
            "D": D,
        }
        return outputs

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="pairlie")
class PairLIE_Weights(WeightsEnum):

    SICE = Weights(
        path=K.ZOO_ROOT / "enhance/pairlie/pairlie/sice/pairlie_sice.pt",
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
