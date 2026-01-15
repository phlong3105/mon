#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Model weights data structures.

This module provides base classes and mixins for model weights.
"""

from __future__ import annotations

__all__ = [
    "Weights",
    "WeightsEnum",
    "WeightsType",
]

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Mapping, Optional

import torch

from mon.core.console import log
from mon.core.enum import Enum
from mon.core.pathlib import download_url_to_file, Path


# ==============================================================================
# region CONSTANTS
# ==============================================================================


# endregion


# ==============================================================================
# region TYPE DEFINITIONS & PROTOCOLS
# ==============================================================================

# --- Type Aliases ---


# --- Protocols ---


# endregion


# ==============================================================================
# region BASE CLASSES & MIXINS
# ==============================================================================

# --- Base Classes ---


# --- Mixins ---


# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

@dataclass
class Weights:
    """A class that groups important attributes associated with the pre-trained
    weights.

    Attributes:
        url (Optional[Path | str]): The location where we find the weights.
        path (Optional[Path | str]): The local path where the weights are stored.
        num_classes (Optional[int]): The number of classes.
        transforms (Optional[Callable]): A callable that constructs the
            preprocessing method (or validation preset transforms) needed to use
            the model. The reason we attach a constructor method rather than an
            already constructed object is because the specific object might have
            memory, and thus we want to delay initialization until needed.
        meta (dict[str, Any]): Stores meta-data related to the weights of the
            model and its configuration. These can be informative attributes
            (for example, the number of parameters/flops, recipe link/methods
            used in training, etc.), configuration parameters (for example, the
            ``num_classes``) needed to construct the model or important
            meta-data (for example, the `classes` of a classification model)
            needed to use the model.
    """

    path       : Optional[Path | str]
    url        : Optional[Path | str] = None
    num_classes: Optional[int]        = None
    transforms : Optional[Callable]   = None
    meta       : dict[str, Any]       = field(default_factory=dict)

    # --- Lifecycle & Initialization ---
    def __post_init__(self):
        """Initialize the weights object."""
        # Check path
        self.path = Path(self.path).normalize() if self.path else None

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

        # Compares transforms, handling a partial application case
        if isinstance(self.transforms, partial) and isinstance(other.transforms, partial):
            return (
                    self.transforms.func     == other.transforms.func
                and self.transforms.args     == other.transforms.args
                and self.transforms.keywords == other.transforms.keywords
            )
        else:
            return self.transforms == other.transforms

    # --- Retrieval ---
    def state_dict(
        self,
        overwrite   : bool = False,
        weights_only: bool = False,
        *args, **kwargs
    ) -> Mapping[str, Any]:
        """Load the weights from the local path or download them if not present.

        Args:
            overwrite: If True, force re-downloading the weights. Defaults to False.
            weights_only: If True, only load the weights and return a dict.
                Defaults to False.

        Returns:
            The state dictionary containing the weights.

        Raises:
            ValueError: If neither a URL nor a local file path is provided.
            RuntimeError: If loading the weights fails.
        """
        # Check/Download Logic
        path = Path(self.path).normalize()
        if not path.exists():
            log(f"Weights file not found at {path}. Trying to download from {self.url}...")

            if not self.url:
                raise ValueError(f"No URL or local file found.")

            download_url_to_file(self.url, path, overwrite)
            # Use torch hub or custom downloader
            # torch.hub.download_url_to_file(self.url, str(self.path), progress=progress)
            # print(self.path)

        # Load with safety checks
        try:
            return torch.load(str(path), weights_only=weights_only, *args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to load weights from {path}: {e}")


class WeightsEnum(Enum):
    """The parent class of all model weights.

    Each model building method receives an optional ``weights`` parameter with
    its associated pre-trained weights. It inherits from `Enum` and its values
    should be of the type ``Weights``.
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

    # --- Retrieval ---
    def state_dict(
        self,
        overwrite   : bool = False,
        weights_only: bool = False,
        *args, **kwargs
    ) -> Mapping[str, Any]:
        """Load the weights from the local path or download them if not present.

        Args:
            overwrite: If True, force re-downloading the weights. Defaults to False.
            weights_only: If True, only load the weights and return a dict.
                Defaults to False.

        Returns:
            The state dictionary containing the weights.

        Raises:
            ValueError: If neither a URL nor a local file path is provided.
            RuntimeError: If loading the weights fails.
        """
        return self.value.state_dict(
            overwrite    = overwrite,
            weights_only = weights_only,
            *args, **kwargs
        )


WeightsType = Weights | WeightsEnum

# endregion
