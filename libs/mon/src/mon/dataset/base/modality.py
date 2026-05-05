#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modality Data Structures.

This module provides data structures for handling modality data in datasets.
"""

from __future__ import annotations

__all__ = [
    "DepthModality",
    "FrameModality",
    "ImageModality",
    "Modality",
    "ModalityList",
]

from dataclasses import dataclass
from functools import partial
from typing import Any, Iterable

from mon.core import (
    AlbumTargetType,
    Frame,
    Image,
    IndexList,
    is_list_of,
    is_valid_str,
    Loader,
)
from mon.ops import ImageLoader, MaskLoader


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

@dataclass
class Modality:
    """Data structure representing a modality in a dataset.

    This class is used in conjunction with ``Dataset`` to define the
    multi-modality data structure.

    Attributes:
        name (str): Name of the modality.
        dirname (str): Directory name of the modality. If not provided,
            defaults to the value of ``name``.
        ext (str): File extension of the modality.
        module (Any): The ``Data`` class associated with the modality.
        loader (DataLoader): The ``Loader`` instance to use for loading this
            modality.
        type (str, optional): Type of the modality (e.g., "image", "text") for
            augmentations (e.g., albumentations). Defaults to "".
        train (bool, optional): Indicates if the modality is available for
            training. Defaults to True.
        val (bool, optional): Indicates if the modality is available for
            validation. Defaults to True.
        test (bool, optional): Indicates if the modality is available for testing
            (i.e., ground-truth). Defaults to True.
    """

    name: str
    dirname: str
    ext: str
    module: Any
    loader: Loader
    type: str = ""
    train: bool = True
    val: bool = True
    test: bool = True

    # --- Lifecycle & Initialization ---
    def __post_init__(self):
        """Perform post-initialization tasks."""
        # Validate inputs
        if not is_valid_str(self.dirname):
            self.dirname = self.name

    # --- Properties ---
    @property
    def additional_target(self) -> dict[str, str] | None:
        """Return a dictionary of target name and type for albumentations
        transforms.
        """
        return {self.name: self.type} if is_valid_str(self.type) else None


class ModalityList(IndexList[Modality]):
    """A list of ``Modality`` instances, accessible by index or name.

    Extend ``IndexList`` to provide dictionary-like access to ``Modality``
    instances by their ``name`` attribute.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, data: Iterable | None = None, *args, **kwargs):
        """Initialize a new instance.

        Args:
            data (Iterable | None, optional): Initial data to populate the list.
                Defaults to None.
        """
        # We hardcode the item_type and key here
        # Users just call ModalityList() without arguments
        super().__init__(item_type=Modality, data=data, key="name")

    # --- Properties ---
    @property
    def names(self) -> list[str]:
        """Return a list of names."""
        return self.keys

    # --- Creation ---
    @classmethod
    def from_list(cls, modalities: list[Modality | dict[str, Any]], **kwargs) -> "ModalityList":
        """Create a new instance from a list of modality configurations.

        Args:
            modalities (list[Modality | dict[str, Any]]):
                A list of modality configurations, where each configuration is
                either a ``Modality`` instance or a dictionary of parameters to
                create a ``Modality`` instance.
            **kwargs: Additional keyword arguments to pass to the constructor.
        """
        # Normalize inputs
        modalities = [Modality(**m) if isinstance(m, dict) else m for m in modalities]

        # Validate inputs
        if not is_list_of(modalities, Modality):
            raise TypeError(f"expected a list of Modality.")

        # Return the new instance
        return cls(modalities, **kwargs)

    @classmethod
    def from_any(cls, value: Any, **kwargs) -> "ModalityList":
        """Create a new instance from arbitrary input.

        Args:
            value (Any): The input value to create a ``ModalityList`` from.
                This can be a ``ModalityList`` instance, a list of modality configurations,
                or any other type that can be converted to a list of modality configurations.
            **kwargs: Additional keyword arguments to pass to the constructor.
        """
        if value is None:
            return ModalityList()
        elif isinstance(value, cls):
            return value
        elif isinstance(value, list):
            return cls.from_list(value, **kwargs)
        else:
            raise TypeError(f"unsupported ModalityList type {type(value).__name__}.")

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

ImageModality = partial(Modality, ext=".jpg", module=Image, type=AlbumTargetType.IMAGE, loader=ImageLoader())
FrameModality = partial(Modality, ext=".jpg", module=Frame, type=AlbumTargetType.IMAGE, loader=None)
DepthModality = partial(Modality, ext=".jpg", module=Image, type=AlbumTargetType.IMAGE, loader=MaskLoader())

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
