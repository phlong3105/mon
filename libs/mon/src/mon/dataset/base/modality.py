#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modality Data Structure.

This module provides data structures for handling modality data in datasets.
"""

from __future__ import annotations

__all__ = [
    "DepthModality",
    "ImageModality",
    "ModalitiesLike",
    "Modality",
    "ModalityList",
    "build_modality_list",
]

from dataclasses import dataclass
from functools import partial
from typing import Any, Iterable, TypeAlias, Union

from mon.core import AlbumTargetType, Image, IndexList, is_valid_str, Loader
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
        return self.keys()

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

ImageModality = partial(Modality, ext=".jpg", module=Image, loader=ImageLoader(), type=AlbumTargetType.IMAGE)
DepthModality = partial(Modality, ext=".jpg", module=Image, loader=MaskLoader(), type=AlbumTargetType.MASK)

# endregion


# ==============================================================================
# region TYPE DEFINITIONS
# ==============================================================================

ModalitiesLike: TypeAlias = Union[
    ModalityList,
    list[Union[Modality, dict[str, Any]]],
    dict[str, Union[Modality, dict[str, Any]]],
]

# endregion


# ==============================================================================
# region REGISTRY & FACTORY
# ==============================================================================

def build_modality_list(modalities: ModalitiesLike | None, *args, **kwargs) -> ModalityList:
    """Build a ``ModalityList`` instance from the given input.

    Args:
        modalities (ModalitiesType, optional): Either a ``ModalityList`` instance,
            a list of ``Modality`` instances or dictionaries, a dictionary
            mapping modality names to ``Modality`` instances or dictionaries,
            or None.
        *args: Positional arguments for ``ModalityList`` constructor.
        **kwargs: Keyword arguments for ``ModalityList`` constructor.

    Returns:
        ModalityList: A ``ModalityList`` instance.

    Raises:
        TypeError: If the input type is not supported.
    """
    if modalities is None:
        return ModalityList(*args, **kwargs)
    elif isinstance(modalities, ModalityList):
        return modalities
    elif isinstance(modalities, (list, dict)):
        # Convert dict to list
        if isinstance(modalities, dict):
            modalities = list(modalities.values())
        for i, m in enumerate(modalities):
            # Convert dict to Modality instance
            modalities[i] = Modality(**m) if isinstance(m, dict) else m
        return ModalityList(modalities, *args, **kwargs)
    else:
        raise TypeError(f"Unsupported type: {type(modalities).__name__}")

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
