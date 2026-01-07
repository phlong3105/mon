#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pre-trained weights base classes and mixins.

This module provides the base classes and mixins for pre-trained weights,
which can be a torch.Tensor or numpy.ndarray.
"""

__all__ = [
    "Weights",
    "WeightsEnum",
]

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Mapping, Optional

import torch

from mon.core.enum import Enum
from mon.core.pathlib import download_url_to_file, Path


# ==============================================================================
# TYPE DEFINITIONS & PROTOCOLS (Interfaces)
# ==============================================================================

# --- Type Aliases ---


# --- Structural Protocols ---


# ==============================================================================
# BASE CLASSES & MIXINS (Behaviors)
# ==============================================================================

# --- Structural Bases ---


# --- Lifecycle Mixins ---


# --- Compute Mixins ---


# ==============================================================================
# CONCRETE IMPLEMENTATIONS (The Concrete Classes)
# ==============================================================================

# --- Primary Data Types ---
@dataclass
class Weights:
    """A class that groups important attributes associated with the pre-trained
    weights.

    Attributes:
        url (Path): The location where we find the weights.
        path (Path): The local path where the weights are stored.
        transforms (Callable): A callable that constructs the preprocessing
            method (or validation preset transforms) needed to use the model.
            The reason we attach a constructor method rather than an already
            constructed object is because the specific object might have memory,
            and thus we want to delay initialization until needed.
        meta (dict[str, Any]): Stores meta-data related to the weights of the
            model and its configuration. These can be informative attributes
            (for example, the number of parameters/flops, recipe link/methods
            used in training, etc.), configuration parameters (for example, the
            ``num_classes``) needed to construct the model or important
            meta-data (for example, the `classes` of a classification model)
            needed to use the model.
    """

    url        : Optional[Path | str]
    path       : Optional[Path | str]
    num_classes: Optional[int]
    transforms : Optional[Callable]
    meta       : dict[str, Any]
    
    # --- Comparison Operators ---
    def __eq__(self, other: Any) -> bool:
        """We need this custom implementation for correct deep-copy and
        deserialization behavior.
        
        TL;DR: After the definition of an enum, creating a new instance, i.e.,
        by deep-copying or deserializing it, involves an equality check against
        the defined members. Unfortunately, the `transforms` attribute is often
        defined with `functools.partial` and `fn = partial(...); assert
        deepcopy(fn) != fn`. Without custom handling for it, the check against
        the defined members would fail and effectively prevent the weights from
        being deep-copied or deserialized.
        
        See https://github.com/pytorch/vision/pull/7107 for details.
        """
        if not isinstance(other, Weights):
            return NotImplemented

        if self.url != other.url:
            return False
        
        if self.path != other.path:
            return False
        
        if self.meta != other.meta:
            return False

        if isinstance(self.transforms, partial) and isinstance(other.transforms, partial):
            return (
                    self.transforms.func     == other.transforms.func
                and self.transforms.args     == other.transforms.args
                and self.transforms.keywords == other.transforms.keywords
            )
        else:
            return self.transforms == other.transforms


class WeightsEnum(Enum):
    """The parent class of all model weights.
    
    Each model building method receives an optional ``weights`` parameter with
    its associated pre-trained weights. It inherits from `Enum` and its values
    should be of the type ``Weights``.

    Attributes:
        value (Weights): The data class entry with the weight information.
    """
    
    # --- Properties ---
    @property
    def url(self) -> Path:
        """Return the URL of the pre-trained weights."""
        return self.value.url
    
    @property
    def path(self) -> Path:
        """Return the local path of the pre-trained weights."""
        return self.value.path
    
    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return self.value.num_classes
    
    @property
    def transforms(self) -> Callable:
        """Return the pre-processing method (or validation preset transforms)
        needed to use the model.
        """
        return self.value.transforms

    @property
    def meta(self) -> dict:
        """Return the meta-data related to the pre-trained weights."""
        return self.value.meta
    
    # --- Hydration & Deserialization ---
    def get_state_dict(
        self,
        overwrite   : bool = False,
        weights_only: bool = False,
        *args: Any, **kwargs: Any
    ) -> Mapping[str, Any]:
        """Load the weights from the local path or download them if not present.
        
        Args:
            overwrite: If True, force re-downloading the weights. Defaults to False.
            weights_only: If True, only load the weights and return a dict. Defaults to False.
            
        Returns:
            The state dictionary containing the weights.
        """
        if self.url and self.path and not Path(self.path).is_weights_file(exist=True):
            print(self.path)
            download_url_to_file(self.url, self.path, overwrite)
        
        if self.path and Path(self.path).is_weights_file(exist=True):
            return torch.load(self.path, weights_only=weights_only, *args, **kwargs)
        else:
            raise FileNotFoundError(f"Weights file not found: {self.path}.")
