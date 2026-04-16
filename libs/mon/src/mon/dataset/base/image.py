#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Datasets.

This module provides image-based datasets.
"""

from __future__ import annotations

__all__ = [
    "AlbumentationsDataset",
    "IQADataset",
    "ImageDataset",
    "ImageOnlyDataset",
]

import copy
import glob
from abc import ABC
from typing import Any, override

import numpy as np
import torch
from box import Box
from tensordict import NonTensorData, TensorDict
from torch import Tensor

from mon.core import (
    build_classlist,
    ClassList,
    create_progress_bar,
    K,
    Metadata,
    MetadataDictList,
    Path,
    PathLike,
    Split,
    SplitLike,
)
from mon.dataset.base.modality import (
    build_modalities,
    ImageModality,
    ModalityList,
)
from mon.dataset.transform import build_compose, Compose
from .dataset import Dataset, InputTargetDataset, StandardDataset


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class AlbumentationsDataset(Dataset, ABC):
    """Base class for datasets that use albumentations for transformations."""

    origin_suffix: str = K.ORIGINAL

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        transforms: Compose | None = None,
        keep_original: bool = False,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            transforms (Compose, optional): Transformations to apply.
                Defaults to None.
            keep_original (bool, optional): Whether to keep the original data
                alongside the transformed ones. Defaults to False.
            *args: Positional arguments for ``Dataset`` constructor.
            **kwargs: Keyword arguments for ``Dataset`` constructor.
        """
        super().__init__(*args, **kwargs)

        # Assign attributes
        self.transforms = transforms
        self.keep_original = keep_original

        # Initialize basic transformations
        self.basic_transforms = self.transforms

    # --- Container / Sequence Methods ---
    @override
    def __getitem__(self, index: int) -> TensorDict:
        """Return an item at the given ``index``.

        Args:
            index (int): Index of datapoint.

        Returns:
            TensorDict: A datapoint dictionary containing all modalities, each
                associated with a 'key'.
        """
        # 1. Get the datapoint
        datapoint = self.get_underlying_data(index=index)

        # 2. Apply transformations
        transforms = self.transforms
        basic_transforms = self.basic_transforms

        if isinstance(transforms, Compose):
            # Create a dictionary of transformable items by filtering out None values
            kv = {k: v for k, v in datapoint.items() if v is not None}
            # Apply transformations
            transformed = transforms(**kv)

            if self.keep_original and isinstance(basic_transforms, Compose):
                # Apply basic transformations to the original data
                transformed_orig = basic_transforms(**kv)
                # Add suffix to the original items (e.g., 'image' -> 'image_orig')
                transformed_orig = {f"{k}_{self.origin_suffix}": v for k, v in transformed_orig.items()}
                # Combine the original and transformed items
                transformed |= transformed_orig

            # Update the datapoint with the transformed values
            datapoint.update(transformed)

        # 3. Convert to TensorDict
        outputs = {}
        for k, v in datapoint.items():
            if v is None:
                continue
            elif isinstance(v, Tensor):
                outputs[k] = v
            else:
                outputs[k] = NonTensorData(v)

        return TensorDict(outputs, batch_size=[])

    # --- Properties ---
    @property
    def basic_transforms(self) -> Compose | None:
        """Return the basic transformations.

        This is used when ``self.keep_original`` is True to transform the
        original data without modifying it (i.e., Normalize and ToTensor).
        """
        return self._basic_transforms

    @basic_transforms.setter
    def basic_transforms(self, value: Compose | None):
        """Set the basic transformations."""
        # Make a copy of the Compose object to avoid modifying the original one
        value = copy.deepcopy(value)

        # Keep only basic transformations (e.g., Normalize and ToTensor)
        if isinstance(value, Compose):
            # Only keep Normalize and ToTensor
            for t in value.transforms:
                if t.__class__.__name__ not in ["Normalize", "ToTensorV2"]:
                    value.transforms.remove(t)

        self._basic_transforms = value

    @property
    def transforms(self) -> Compose | None:
        """Return the transformation operations."""
        return self._transforms

    @transforms.setter
    def transforms(self, value: Compose | None):
        """Set the transformation operations.

        Args:
            value (Compose, optional): Transformations to apply.

        Raises:
            TypeError: If ``transform`` is not an instance of albumentations.Compose.
        """
        # Normalize inputs
        if isinstance(value, (Box, dict)):
            value = Compose.from_config(config=value)

        if isinstance(value, Compose):
            # Add additional targets to transform if needed
            # Get the primary modality key
            pk, _ = self.primary
            # Get one sample of the metapoint
            metapoint = self.get_metapoint(index=0)
            # Create a dictionary of additional targets for later use in __getitem__()
            for k, v in metapoint.items():
                if (k != pk) and v is not None:
                    # If the modality is not primary and has a target type, add it
                    # to the Compose
                    value.add_targets(self.modalities[k].additional_target)

        self._transforms = value


class ImageDataset(StandardDataset, AlbumentationsDataset):
    """Standard image dataset.

    Extend the base ``StandardDataset`` class with ``AlbumentationsDataset`` to
    support image-based datasets with multiple modalities, splits, and
    albumentations transformations.
    """

    dirname: str = ""
    subdir: str = ""
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST, Split.PREDICT]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="image"),
    ])
    classes: ClassList = ClassList()

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        root: PathLike,
        split: SplitLike,
        dirname: str = "",
        subdir: str = "",
        transforms: Compose | None = None,
        keep_original: bool = False,
        modalities: ModalityList | None = None,
        classes: ClassList | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            root (PathLike): Path to the root directory of the dataset.
            split (SplitType): Data split subset to use. Must be one of the
                options defined in ``splits``.
            dirname (str, optional): Name of the dataset directory within the
                root path. Use this if the given ``root`` path does not contain
                the dataset directory itself. Defaults to "".
            subdir (str, optional): Name of the subdirectory within the dataset's
                ``root`` (i.e., ``root/subdir``). Use this if the current
                dataset is a subset of another dataset. If provided, it
                overrides the class-level default. Defaults to "".
            transforms (Compose, optional): Transformations to apply.
                Defaults to None.
            keep_original (bool, optional): Whether to keep the original data
                alongside the transformed ones. Defaults to False.
            modalities (ModalityList, optional): A list of ``Modality``
                definitions. By default, the first modality is considered the
                primary one. If provided, it overrides the class-level default.
                Defaults to None.
            classes (ClassList, optional): Class definitions associated with
                the dataset. If provided, it overrides the class-level default.
                Defaults to None.
            verbose (bool): Verbosity mode. Defaults to True.
            *args: Positional arguments for ``Dataset`` constructor.
            **kwargs: Keyword arguments for ``Dataset`` constructor.
        """
        super().__init__(
            root=root,
            split=split,
            dirname=dirname,
            subdir=subdir,
            transforms=transforms,
            keep_original=keep_original,
            modalities=modalities,
            classlist=classes,
            verbose=verbose,
            *args, **kwargs
        )

    # --- Creation ---
    @override
    @classmethod
    def from_config(cls, config: dict[str, Any], **kwargs) -> "ImageDataset":
        """Create a new instance from a configuration dictionary."""
        # Extract relevant keys
        transforms = config.pop("transforms", None)
        modalities = config.pop("modalities", None)
        classes = config.pop("classes", None)

        # Build the objects
        transforms = build_compose(transforms)
        modalities = build_modalities(modalities)
        classes = build_classlist(classes)

        # Return the new instance
        config |= kwargs
        return cls(
            transforms=transforms,
            modalities=modalities,
            classes=classes,
            **config
        )


class ImageOnlyDataset(StandardDataset, AlbumentationsDataset):
    """Standard image-only dataset.

    Extend the base ``StandardDataset`` class with ``AlbumentationsDataset`` to
    support image-only datasets with multiple splits and albumentations
    transformations.
    """

    dirname: str = ""
    subdir: str = ""
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST, Split.PREDICT]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="image"),
    ])
    classes: ClassList = ClassList()

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        root: PathLike,
        split: SplitLike = Split.PREDICT,
        dirname: str = "",
        subdir: str = "",
        transforms: Compose | None = None,
        keep_original: bool = False,
        modalities: ModalityList | None = None,
        classes: ClassList | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            root (PathLike): Path to the root directory of the dataset.
            split (SplitType): Data split subset to use. Must be one of the
                options defined in ``splits``.
            dirname (str, optional): Name of the dataset directory within the
                root path. Use this if the given ``root`` path does not contain
                the dataset directory itself. Defaults to "".
            subdir (str, optional): Name of the subdirectory within the dataset's
                ``root`` (i.e., ``root/subdir``). Use this if the current
                dataset is a subset of another dataset. If provided, it
                overrides the class-level default. Defaults to "".
            transforms (Compose, optional): Transformations to apply.
                Defaults to None.
            keep_original (bool, optional): Whether to keep the original data
                alongside the transformed ones. Defaults to False.
            modalities (ModalityList, optional): A list of ``Modality``
                definitions. By default, the first modality is considered the
                primary one. If provided, it overrides the class-level default.
                Defaults to None.
            classes (ClassList, optional): Class definitions associated with
                the dataset. If provided, it overrides the class-level default.
                Defaults to None.
            verbose (bool): Verbosity mode. Defaults to True.
            *args: Positional arguments for ``Dataset`` constructor.
            **kwargs: Keyword arguments for ``Dataset`` constructor.
        """
        super().__init__(
            root=root,
            split=split,
            dirname=dirname,
            subdir=subdir,
            transforms=transforms,
            keep_original=keep_original,
            modalities=modalities,
            classlist=classes,
            verbose=verbose,
            *args, **kwargs
        )

    # --- Properties ---
    @override
    @property
    def base_dir(self) -> Path:
        """Return the base directory of the dataset."""
        return self.root

    # --- Discovery ---
    @override
    def list_metapoints(self):
        """List all metapoints available for loading.

        After calling this method, ``self.metapoints`` must be populated.
        """
        # Initialize empty metapoints dictionary with modalities
        metapoints = MetadataDictList.from_keys(self.modalities.keys)

        # List all image files under the root
        pk, pm = self.primary
        src = self.base_dir

        if src.is_image_file():
            # If is a single image file, return a list with only that image
            paths = [src]
        elif "*" in str(src):
            # If is a glob pattern, list the matching files
            # Using iglob (iterator) is more memory efficient than glob.glob
            paths = [Path(p) for p in glob.iglob(str(src), recursive=True)]
        elif src.is_dir():
            # If is a directory, list all files recursively
            paths = list(src.rglob("*"))
        else:
            raise ValueError(
                f"Invalid source '{src}' for primary modality '{pk}'. "
                f"Expected a directory, a glob pattern, or an image file."
            )

        # List primary modality metadata files
        name = pm.name
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            paths = sorted(paths)
            desc = f"Listing {self.__class__.__name__} {name}(s)"
            for path in pbar.track(sequence=paths, description=desc):
                path = path.normalize()
                if path.is_image_file():
                    metapoints[name].append(Metadata(path=path, base_dir=src))

        self.metapoints = metapoints

    # --- Creation ---
    @override
    @classmethod
    def from_config(cls, config: dict[str, Any], **kwargs) -> "ImageOnlyDataset":
        """Create a new instance from a configuration dictionary."""
        # Extract relevant keys
        transforms = config.pop("transforms", None)
        modalities = config.pop("modalities", None)
        classes = config.pop("classes", None)

        # Build the objects
        transforms = build_compose(transforms)
        modalities = build_modalities(modalities)
        classes = build_classlist(classes)

        # Return the new instance
        config |= kwargs
        return cls(
            transforms=transforms,
            modalities=modalities,
            classes=classes,
            **config
        )


class IQADataset(InputTargetDataset, AlbumentationsDataset):
    """Image quality assessment (IQA) dataset.

    Extend the base ``InputTargetDataset`` class with ``AlbumentationsDataset``
    to support image-based datasets with input and target modalities, and
    albumentations transformations.
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
        transforms: Compose | None = None,
        keep_original: bool = False,
        modalities: ModalityList | None = None,
        classes: ClassList | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            input_dir (PathLike): Path to the input directory.
            target_dir (PathLike, optional): Path to the target directory.
                Defaults to None.
            transforms (Compose, optional): Transformations to apply.
                Defaults to None.
            keep_original (bool, optional): Whether to keep the original data
                alongside the transformed ones. Defaults to False.
            modalities (ModalityList, optional): A list of ``Modality``
                definitions. By default, the first modality is considered the
                primary one. If provided, it overrides the class-level default.
                Defaults to None.
            classes (ClassList, optional): Class definitions associated with
                the dataset. If provided, it overrides the class-level default.
                Defaults to None.
            verbose (bool, optional): Verbosity mode. Defaults to True.
            *args: Positional arguments for ``Dataset`` constructor.
            **kwargs: Keyword arguments for ``Dataset`` constructor.
        """
        super().__init__(
            input_dir=input_dir,
            target_dir=target_dir,
            transforms=transforms,
            keep_original=keep_original,
            modalities=modalities,
            classlist=classes,
            verbose=verbose,
            *args, **kwargs
        )

    # --- Creation ---
    @override
    @classmethod
    def from_config(cls, config: dict[str, Any], **kwargs) -> "IQADataset":
        """Create a new instance from a configuration dictionary."""
        # Extract relevant keys
        transforms = config.pop("transforms", None)
        modalities = config.pop("modalities", None)
        classes = config.pop("classes", None)

        # Build the objects
        transforms = build_compose(transforms)
        modalities = build_modalities(modalities)
        classes = build_classlist(classes)

        # Return the new instance
        config |= kwargs
        return cls(
            transforms=transforms,
            modalities=modalities,
            classes=classes,
            **config
        )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
