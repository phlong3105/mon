#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Weights Data Structures.

This module provides data structures and utilities for handling model weights.
"""

from __future__ import annotations

__all__ = [
    "Weights",
    "WeightsEnum",
    "WeightsEnumLike",
    "WeightsLike",
    "create_weights",
    "is_weights_type",
]

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Mapping, override, TypeAlias, Union

import torch

from mon.core.base.enum import Enum, EnumMeta
from mon.core.console import log
from mon.core.constants import K
from mon.core.path import Path
from mon.core.typing import PathLike
from mon.core.utils import is_valid_str


# ==============================================================================
# region BASE CLASSES
# # ==============================================================================

@dataclass
class Weights:
    """A class that groups important attributes associated with the pre-trained
    weights.

    Attributes:
        path (Path): The local path where the weights are stored.
        url (Path, optional): The URL where the weights can be downloaded.
            Defaults to None.
        num_classes (int, optional): The number of classes. Defaults to None.
        transforms (Callable, optional): A callable that constructs the
            preprocessing method (or validation preset transforms) needed to
            use the model. The reason we attach a constructor method rather
            than an already constructed object is because the specific object
            might have memory, and thus we want to delay initialization until
            needed. Defaults to None.
        meta (dict[str, Any]): Stores meta-data related to the weights of the
            model and its configuration. These can be informative attributes
            (for example, the number of parameters/flops, recipe link/methods
            used in training, etc.), configuration parameters (for example, the
            ``num_classes``) needed to construct the model or important
            meta-data (for example, the `classes` of a classification model)
            needed to use the model. Defaults to an empty dictionary.
    """

    name: str = ""
    path: Path | None = None
    url: Path | None = None
    num_classes: int | None = None
    transforms: Callable | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    # --- Lifecycle & Initialization ---
    def __post_init__(self):
        """Perform post-initialization tasks."""
        # Validate inputs
        if is_valid_str(self.path):
            self.path = Path(self.path).normalize()
            if (
                not self.path.is_weights_file(exists=False)
                and not self.path.is_dir()
            ):
                # We only care about the validity of the path, not its existence
                raise ValueError(
                    f"Weights file or directory not found at: '{self.path}'"
                )
        if is_valid_str(self.url):
            self.url = Path(self.url)

    # --- Comparison Operators ---
    @override
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
        # Validate inputs
        if not isinstance(other, Weights):
            return NotImplemented
        if self.name != other.name:
            return False
        if self.path != other.path:
            return False
        if self.url != other.url:
            return False
        if self.meta != other.meta:
            return False

        # Compares transforms, handling a partial application case
        if isinstance(self.transforms, partial) and isinstance(other.transforms, partial):
            return (
                self.transforms.func == other.transforms.func
                and self.transforms.args == other.transforms.args
                and self.transforms.keywords == other.transforms.keywords
            )
        else:
            return self.transforms == other.transforms

    # --- Properties ---
    @property
    def exists(self) -> bool:
        """Return if the weights file exists."""
        return self.path and self.path.is_weights_file(exists=True)

    # --- Retrieval ---
    def state_dict(
        self,
        overwrite: bool = False,
        weights_only: bool = False,
        *args, **kwargs
    ) -> Mapping[str, Any]:
        """Load the weights from the local path or download them if not present.

        Args:
            overwrite (bool, optional): If True, force re-downloading the weights.
                Defaults to False.
            weights_only (bool, optional): If True, only load the weights and
                return a dict. Defaults to False.

        Returns:
            Mapping[str, Any]: The loaded weights.

        Raises:
            ValueError: If neither a URL nor a local file path is provided.
            RuntimeError: If loading the weights fails.
        """
        # Check/Download Logic
        path = Path(self.path).normalize()
        if not path.exists():
            log(
                f"Weights file not found at {path}. "
                f"Trying to download from {self.url}..."
            )

            if not self.url:
                raise ValueError(f"No URL or local file found.")

            from mon.core.filesystem import download_url_to_file
            download_url_to_file(self.url, path, overwrite)
            # Use torch hub or custom downloader
            # torch.hub.download_url_to_file(self.url, str(self.path), progress=progress)
            # print(self.path)

        # Load with safety checks
        try:
            return torch.load(str(path), weights_only=weights_only, *args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to load weights from {path}: {e}")

    # --- Mutation ---
    def rectify_path(self, root: PathLike):
        """Check if the path is valid and rectify it if not.

        The search order is as follows:
            1. Check if the path is valid (if it already points to a file, do nothing).
            2. Check the local project root (Highest priority).
            3. Check the global zoo directory.

        Args:
            root (PathLike): The local project root to check for the weights file.
        """
        # Check if the path is valid
        if self.path is None:
            return

        # Check if the path is already a file
        if self.path.is_weights_file(exists=True):
            return

        # Check local project root (Highest priority)
        local_file = Path(root) / self.path
        if local_file.is_weights_file(exists=True):
            self.path = local_file

        # Check global zoo directory
        global_file = K.ZOO_ROOT / self.path
        if global_file.is_weights_file(exists=True):
            self.path = global_file


class WeightsEnumMeta(EnumMeta):

    # --- Callable & Context Manager ---
    def __call__(cls, value=None, *args, **kwargs):
        # If it's already an instance of this enum, return it directly
        if isinstance(value, cls):
            return value

        # If it's a Weights object or a path-like, bypass enum lookup entirely
        if isinstance(value, Weights):
            return value
        if isinstance(value, (Path, str)) and value != "default":
            return create_weights(weights=value) or value

        # Otherwise, delegate to the normal enum machinery (which calls _missing_)
        return super().__call__(value, *args, **kwargs)


class WeightsEnum(Enum, metaclass=WeightsEnumMeta):
    """A base class for enumerations of pre-trained model weights.

    Each model building method receives an optional ``weights`` parameter with
    its associated pre-trained weights. It inherits from `Enum` and its values
    should be of the type ``Weights``.
    """

    # --- Lifecycle & Initialization ---
    @classmethod
    def _missing_(cls, value):
        # 1. If no value is passed (None), return the first member
        if value is None:
            return None

        # 2. If the user explicitly passes the string "default",
        # return the first member if "DEFAULT" is not defined, otherwise return
        # the "DEFAULT" member
        if value == "default":
            if "DEFAULT" in cls.__members__:
                return cls.DEFAULT
            else:
                return list(cls)[0]

        # 3. Otherwise, it's an invalid extension
        raise ValueError(f"'{value}' is not a valid {cls.__name__}.")

    # --- Properties ---
    @property
    def exists(self) -> bool:
        """Return if the weights file exists."""
        return self.value.exists

    @property
    def path(self) -> Path:
        """Return the local path of the pre-trained weights."""
        return self.value.path

    @property
    def url(self) -> Path:
        """Return the URL of the pre-trained weights."""
        return self.value.url

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
        overwrite: bool = False,
        weights_only: bool = False,
        *args, **kwargs
    ) -> Mapping[str, Any]:
        """Load the weights from the local path or download them if not present.

        Args:
            overwrite (bool, optional): If True, force re-downloading the weights.
                Defaults to False.
            weights_only (bool, optional): If True, only load the weights and
                return a dict. Defaults to False.

        Returns:
            Mapping[str, Any]: The loaded weights.

        Raises:
            ValueError: If neither a URL nor a local file path is provided.
            RuntimeError: If loading the weights fails.
        """
        return self.value.state_dict(
            overwrite=overwrite, weights_only=weights_only, *args, **kwargs
        )

# endregion


# ==============================================================================
# region TYPE DEFINITIONS
# ==============================================================================

WeightsEnumLike: TypeAlias = Union[WeightsEnum, str]
WeightsLike: TypeAlias = Union[Weights, WeightsEnum, PathLike]

# endregion


# ==============================================================================
# region CREATION
# ==============================================================================

def create_weights(weights: Weights | PathLike, *args, **kwargs) -> Weights | None:
    """Create a ``Weights`` object from a path or a ``Weights`` object."""
    if isinstance(weights, Weights):
        # If it's already a Weights object, return it directly
        return weights
    elif isinstance(weights, (Path, str)):
        # If it's a path, try to load it as a Weights object'
        weights = Path(weights).normalize()
        if weights.is_weights_file(exists=True):
            return Weights(path=weights, *args, **kwargs)

    # Otherwise, return None
    return None

# endregion


# ==============================================================================
# region VALIDATION
# ==============================================================================

def is_weights_type(value: Any, exists: bool = True) -> bool:
    """Check if a value is of type ``WeightsType``."""
    if isinstance(value, (Weights, WeightsEnum)):
        return value.exists if exists else True
    else:
        return False


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
