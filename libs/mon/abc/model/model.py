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

import sys

import torch

from mon import nn
from mon.core import (
    create_device,
    log,
    MLType,
    MODELS,
    Path,
    Task,
    WEIGHTS,
    ZOO_DIR,
)
from mon.core.types import Weights, WeightsEnum, WeightsType

current_file = Path(__file__).normalize()
current_dir  = current_file.parents[0]
extern_path  = current_dir / "extern" / "repo"
if str(extern_path) not in sys.path:
    sys.path.append(str(extern_path))

try:
    # Now we can safely import from the original repository
    from repo import something
except ImportError:
    raise ImportError(f"Failed to import 'something' from the 'extern/repo' directory.")


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

    arch     : str          = "<arch>"
    name     : str          = None
    tasks    : list[Task]   = [Task.SEGMENT]
    mltypes  : list[MLType] = []
    model_dir: Path         = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name   : str,
        weights: WeightsType | None = None,
        verbose: bool               = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name: Name of the model variant.
            weights: Pre-trained weights to load. Defaults to None.
            verbose: Verbosity mode. Defaults to True.
        """
        # Satisfy PyTorch's empty signature first.
        super().__init__()
        # Initialize RegistrableMixin
        nn.RegistrableMixin.__init__(self, name=name)

        self.verbose = verbose

        # Load the base model
        if isinstance(weights, WeightsType):
            kwargs["num_classes"] = weights.num_classes

        self.model = nn.Sequential(*args, **kwargs)

        if isinstance(weights, WeightsType):
            self.model.load_state_dict(weights.state_dict())
            # self.load_state_dict(weights.get_state_dict())
            if self.verbose:
                log(f"Initialized '{name}' from weights: '{weights.path}'.")
        else:
            if self.verbose:
                log(f"Initialized '{name}' from scratch.")

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor, *args, **kwargs):
        """Forward the input through the network.

        Args:
            x: Input, formatted as a torch.Tensor of shape (B, C, H, W)
                and pixel values ranging from 0.0 to 1.0.

        Returns:

        """
        raise NotImplementedError


# --- Mixins ---


# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="")
class Weights(WeightsEnum):

    DATASET = Weights(
        path        = ZOO_DIR / "",
        url         = "",
        num_classes = None,
        transforms  = None,
        meta        = {}
    )
    DEFAULT = DATASET


# --- Model Variants ---

@MODELS.register(name="", metaclass=BaseModel)
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
