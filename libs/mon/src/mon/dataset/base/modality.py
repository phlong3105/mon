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
    "ModalitiesLike",
    "Modality",
    "ModalityList",
    "build_modalities",
]

from dataclasses import dataclass
from functools import partial
from typing import Any, Iterable, TypeAlias, Union

from mon.core import (
    AlbumTargetType,
    Frame,
    Image,
    IndexList,
    is_valid_str,
    Loader,
)
from mon.cv import ImageLoader, MaskLoader


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
        test (bool, optional): Indicates if the modality is available for testing
            (i.e., ground-truth). Defaults to True.
    """

    name: str
    dirname: str
    ext: str
    module: Any
    loader: Loader
    type: str = ""
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
            data (Iterable, optional): Initial data to populate the list.
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
    def from_list(cls, modalities: list[ModalityType], **kwargs) -> "ModalityList":
        """Create a new instance from a list of modality configurations."""
        # Validate inputs
        if not isinstance(modalities, list):
            raise TypeError(
                f"Expected a list of modality configurations, "
                f"but got {type(modalities).__name__}."
            )

        # Build the object
        for i, m in enumerate(modalities):
            if isinstance(m, dict):
                modalities[i] = Modality(**m)

        # Return the new instance
        return cls(modalities, **kwargs)

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

ImageModality = partial(Modality, ext=".jpg", module=Image, loader=ImageLoader(), type=AlbumTargetType.IMAGE)
FrameModality = partial(Modality, ext=".jpg", module=Frame, loader=None, type=AlbumTargetType.IMAGE)
DepthModality = partial(Modality, ext=".jpg", module=Image, loader=MaskLoader(), type=AlbumTargetType.IMAGE)

# endregion


# ==============================================================================
# region TYPE DEFINITIONS
# ==============================================================================

ModalityType: TypeAlias = Union[Modality, dict[str, Any]]
ModalitiesLike: TypeAlias = Union[ModalityList, list[ModalityType]]

# endregion


# ==============================================================================
# region CREATION
# ==============================================================================

def build_modalities(value: ModalitiesLike | None) -> ModalityList:
    """Build a ``ModalityList`` from a given value.

    Args:
        value (ModalitiesLike | None): Either a ``ModalityList`` instance or a
            list of modality configurations.

    Returns:
        ModalityList: A ``ModalityList`` instance.

    Raises:
        TypeError: If the input value is not a valid type for building a
            ``ModalityList``.
    """
    if value is None:
        return ModalityList()
    elif isinstance(value, ModalityList):
        return value
    elif isinstance(value, list):
        return ModalityList.from_list(value)
    else:
        raise TypeError(
            f"Unsupported ModalityList type: {type(value).__name__}."
        )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
