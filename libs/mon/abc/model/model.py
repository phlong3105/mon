#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""<Model> models and pre-trained weights.

This module provides the <model> definition and pre-trained weights.

References:
    - Paper:
    - Code:
"""

from __future__ import annotations

__all__ = []

import torch

from mon import nn
from mon.core import MLType, MODELS, Path, Task, WEIGHTS, ZOO_DIR
from mon.core.dtypes import Weights, WeightsEnum

current_file = Path(__file__).normalize()
current_dir  = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES & MIXINS
# ==============================================================================

# --- Base Classes ---

class BaseModel(nn.Module, nn.RegistrableMixin):
    """<Model> model.

    References:
        - Paper:
        - Code:
    """

    _arch     : str          = "<arch>"
    _name     : str          = None
    _tasks    : list[Task]   = [Task.SEGMENT]
    _mltypes  : list[MLType] = []
    _model_dir: Path         = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name   : str,
        weights: WeightsEnum | None = None,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name: Name of the model variant.
            weights: Pre-trained weights to load. Defaults to None.
        """
        super().__init__(name=name, *args, **kwargs)

        # Update hyperparameters
        if isinstance(weights, WeightsEnum):
            kwargs["num_classes"] = weights.num_classes

        # Define the model architecture
        self.network = nn.Sequential(*args, **kwargs)

        # Load pre-trained weights
        if isinstance(weights, WeightsEnum):
            self.network.load_state_dict(weights.get_state_dict())
            # self.load_state_dict(weights.get_state_dict())


    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor, *args, **kwargs):
        """Forward the input through the network.

        Args:
            x: Input tensor with dimensions (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:

        """
        raise NotImplementedError


# --- Mixins ---


# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(arch="<arch>", name="")
class Weights(WeightsEnum):

    DATASET = Weights(
        url         = "",
        path        = ZOO_DIR / "",
        num_classes = None,
        transforms  = None,
        meta        = {}
    )
    DEFAULT = DATASET


# --- Model Variants ---

@MODELS.register(name="")
def model(weights: WeightsEnum | str | None = Weights.DEFAULT, *args, **kwargs):
    """Create an <Model> model.

    Args:
        weights: Pre-trained weights to load. Defaults to
            Weights.DEFAULT.
        args: Additional positional arguments for the <Model> model.
        kwargs: Additional keyword arguments for the <Model> model.

    Returns:
        An SAM model instance.
    """
    return BaseModel(name="", weights=Weights(weights),*args, **kwargs)


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
