#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Dataset.

This module provides image-based datasets.
"""

from __future__ import annotations

__all__ = [

]

from typing import Any, override

from mon.core import ClassList, PathLike, Split, Task
from mon.dataset.base.dataset import InputTargetDataset, StandardDataset
from mon.dataset.base.mixins import DatasetCollationMixin, DatasetRegisterMixin
from mon.dataset.base.modality import ImageModality, ModalityList
from mon.dataset.transforms import build_compose, Compose, ComposeLike


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class ImageDataset(DatasetCollationMixin, DatasetRegisterMixin, StandardDataset):
    """Standard image dataset."""

    name: str = ""
    tasks: list[Task] = []
    subdir: str = ""
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST, Split.PREDICT]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="image"),
    ])
    classes: ClassList = ClassList()



class IQADataset(DatasetCollationMixin, InputTargetDataset):
    """Image quality assessment (IQA) dataset.

    Extend the base ``InputTargetDataset`` class to support evaluation pipelines
    where input and target are images.
    """

    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="image"),
        ImageModality(name="target", dirname="target"),
    ])
    classes: ClassList = ClassList()

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        input_dir: PathLike,
        target_dir: PathLike | None = None,
        transforms: ComposeLike | None = None,
        modalities: ModalityList | None = None,
        classes: PathLike | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            input_dir (PathLike): Path to the input directory.
            target_dir (PathLike, optional): Path to the target directory.
                Defaults to None.
            transforms (ComposeLike, optional): Transformations to apply.
                Defaults to None.
            modalities (ModalityList, optional): A list of ``Modality``
                definitions. By default, the first modality is considered the
                primary one. If provided, it overrides the class-level default.
                Defaults to None.
            classes (Classes, optional): Class definitions associated with the
                dataset. If provided, it overrides the class-level default.
                Defaults to None.
            verbose (bool): Verbosity mode. Defaults to True.
            *args: Positional arguments for ``Dataset`` constructor.
            **kwargs: Keyword arguments for ``Dataset`` constructor.
        """
        super().__init__(
            input_dir=input_dir,
            target_dir=target_dir,
            modalities=modalities,
            classlist=classes,
            verbose=verbose,
            *args, **kwargs
        )

        # Assign attributes
        self.transforms = build_compose(transforms)

    # --- Container / Sequence Methods ---
    @override
    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return an item at the given ``index``.

        Args:
            index (int): Index of datapoint.

        Returns:
            dict[str, Any]: A datapoint dictionary containing all modalities,
                each associated with a 'key'.
        """
        # 1. Get the datapoint
        datapoint = self.get_datapoint(index=index)

        # 2. Apply transformations
        compose = self.transforms
        if compose:
            # Create a dictionary of transformable items by filtering out None
            # values
            kv = {k: v for k, v in datapoint.items() if v is not None}
            # Apply transformations
            transformed = compose(**kv)
            # Update the datapoint with the transformed values
            datapoint.update(transformed)

        return datapoint

    # --- Properties ---
    @property
    def transforms(self) -> Compose | None:
        """Return the transformation operations."""
        return self._transforms

    @transforms.setter
    def transforms(self, transforms: ComposeLike | None, *args, **kwargs):
        """Set the transformation operations.

        Args:
            transforms (ComposeLike, optional): Transformations to apply.

        Raises:
            TypeError: If ``transform`` is not an instance of albumentations.Compose.
        """
        # 1. Build the Compose instance from the input
        compose = build_compose(transforms=transforms, *args, **kwargs)

        # 2. Add additional targets to transform if needed
        # Get the primary modality key
        pk, _ = self.primary
        # Get one sample of the metapoint
        metapoint = self.get_metapoint(index=0)
        # Create a dictionary of additional targets for later use in __getitem__()
        for k, v in metapoint.items():
            if (k != pk) and v is not None:
                # If the modality is not primary and has a target type, add it
                # to the Compose
                compose.add_targets(self.modalities[k].additional_target)

        self._transforms = compose

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
