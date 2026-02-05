#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image-based datasets.

This module provides base classes for image datasets and data loaders.
"""

from __future__ import annotations

__all__ = [
    "ImageDataset",
    "ImageLoader",
]

import glob
from typing import Any

import box
import numpy as np
import torch

from mon.core import create_progress_bar, log, Path, Split
from mon.core.dtypes import ClassList, Image
from mon.training.augment import albumentations as A
from ...base import Dataset, Modalities, Modality
from ...comp import BatchCollateMixin, MultimodalDataLoadMixin


# ==============================================================================
# region IMAGE DATASETS
# ==============================================================================

class ImageDataset(Dataset, MultimodalDataLoadMixin, BatchCollateMixin):
    """Image dataset base class.

    Extend ``Dataset`` with mixins for metadata handling, multimodal data
    loading, and batch collation. Define transformation operations using the
    albumentations library.

    Attributes:
        subset: Name of the dataset's subset directory. Since the given
            attribute ``root`` may only set the dataset root directory, this
            attribute defines the actual folder name of the sub-dataset within
            the root directory (e.g., dataset with multiple versions).
            `Should be defined in subclasses.`
        splits: List of supported splits. `Should be defined in subclasses.`
        modalities: Dictionary defining the dataset modalities. `Must be defined
            in subclasses.`
        classlist: Dataset object classes. `Should be defined in subclasses.`
        transform : Transformations to apply to the data.
        verbose: If True, enable verbose output. Defaults to True.
    """

    subset    : str | None  = None
    splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST, Split.PREDICT]
    modalities: Modalities  = {
        "image": Modality(
            name    = "image",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
            primary = True,
        ),
    }
    classlist : ClassList | None = None

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        root     : Path,
        split    : Split                   = Split.TRAIN,
        transform: A.Compose        | None = None,
        classlist: Path | ClassList | None = None,
        verbose  : bool                    = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            root: Absolute path to the dataset root directory.
            split: Data split subset to use. Defaults to Split.TRAIN.
            transform: Transformations to apply. Defaults to None.
            classlist: Either a .yaml file containing the classes definitions,
                or a ClassList instance. Defaults to None.
            verbose: Verbosity mode. Defaults to True.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Raises:
            TypeError: If ``root``, ``split``, or ``verbose`` is invalid.
            ValueError: If ``modalities`` is empty.
        """
        # Validate inputs
        if not isinstance(root, (str, Path)):
            raise TypeError(
                f"Expected 'root' to be a str or Path, but got {type(root).__name__}."
            )
        if not isinstance(split, (str, Split)):
            raise TypeError(
                f"Expected 'split' to be a str or Split, but got {type(split).__name__}."
            )
        if not isinstance(verbose, bool):
            raise TypeError(
                f"Expected 'verbose' to be a bool, but got {type(verbose).__name__}."
            )
        if not self.modalities:
            raise ValueError(f"Expected 'modalities' to be a non-empty dict.")

        # Continue the initialization chain
        super().__init__(
            root      = root,
            split     = split,
            classlist = classlist,
            verbose   = verbose,
            *args, **kwargs
        )

        # Assign attributes
        self.transform = None
        self.set_transform(value=transform)

    def __del__(self):
        """Finalize the object.

        Close the dataset loading mechanism and release resources.
        """
        pass

    # --- Representation ---
    def __repr__(self) -> str:
        """Return the official string representation for developers."""
        lines  = [f"Dataset {self.__class__.__name__}"]
        lines += [f"Number of datapoints: {len(self)}"]
        if self.root:
            lines += [f"Root location: {self.root}"]
        if self.transform:
            lines += [repr(self.transform)]
        return "\n".join(lines)

    # --- Container / Sequence Methods ---
    def __len__(self) -> int:
        """Return the length of the container."""
        # Optimization: Use the internal dict directly to avoid property overhead
        # and search only for the primary modality key once.
        pk = next(k for k, v in self.modalities.items() if v.primary)
        return len(self.datapoints[pk])

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return an item at the given ``index``."""
        # Fetch datapoint
        data = self._get_underlying_data(index=index)
        meta = data.pop("meta")  # Remove metadata from datapoint for easier augmentation ops.

        transform = self.transform

        if transform is not None:
            pk, _ = self.primary_modality

            # Albumentations expects 'image'. We map pk -> 'image' without pop/merge overhead.
            if pk != "image":
                data["image"] = data.pop(pk)

            # Filter None values efficiently
            augmented = transform(**{k: v for k, v in data.items() if v is not None})

            # Revert 'image' back to the primary modality key if necessary
            if pk != "image":
                augmented[pk] = augmented.pop("image")

            # Update data in-place (Faster than |= for small dicts)
            data.update(augmented)

            # Optimized Type Casting
            for k, v in data.items():
                if v is not None:
                    # Converts non‑float tensors/arrays to float32
                    if isinstance(v, torch.Tensor) and v.dtype != torch.float32:
                        data[k] = v.to(torch.float32)
                    elif isinstance(v, np.ndarray) and v.dtype != np.float32:
                        data[k] = v.astype(np.float32)

        return {**data, "meta": meta}

    # --- Properties ---
    def set_transform(self, value: Any = None):
        """Set the transformation operations.

        Args:
            value: Transformations to apply. Defaults to None.

        Raises:
            TypeError: If ``value`` is not an instance of albumentations.Compose.
        """
        if value is None:
            self.transform = None
            return

        if isinstance(value, (dict, box.Box)):
            value = A.Compose(**value)
        if not isinstance(value, A.Compose):
            raise TypeError(
                f"Expected 'transform' to be an instance of albumentations.Compose, "
                f"but got {type(value).__name__}."
            )

        # Add additional targets to A.Compose if needed.
        existing_targets = value.processors.get("additional_targets", {})
        new_targets      = {
            k: v.type for k, v in self.modalities.items()
            if v.type and v.module and k not in A.TARGET_TYPES and k not in existing_targets
        }

        if new_targets:
            value.add_targets(additional_targets=new_targets)

        self.transform = value

    # --- Data Loading ---
    def verify(self):
        """Verify dataset integrity.

        Raises:
            RuntimeError: If no datapoints are found or if modality lengths are
                inconsistent.
        """
        if len(self) <= 0:
            raise RuntimeError(f"No datapoints in the dataset: {self.__class__.__name__}.")

        pk, _ = self.primary_modality
        for k, v in self.datapoints.items():
            if k not in self.modalities:
                raise RuntimeError(
                    f"Expected 'datapoints' to have only defined modalities, "
                    f"but got unexpected key: {k}."
                )
            if self.modalities[k]:
                if v in [None, []]:
                    raise RuntimeError(f"Datapoint modality '{k}' is empty.")
                elif len(v) != len(self):
                    raise RuntimeError(
                        f"Datapoint modality '{k}' has inconsistent length with "
                        f"the dataset: {len(v)} != {len(self)}."
                    )

        if self.verbose:
            log(f"Number of {self.split_str} datapoints: {len(self)}.")

    # --- Access ---
    def _get_datapoint(self, index: int) -> dict[str, Any]:
        """Get a datapoint at the specified ``index``.

        Args:
            index: Index of datapoint.
        """
        # Efficiency: Use dict comprehension for faster construction
        return {
            k: (v[index] if v is not None else None)
            for k, v in self.datapoints.items()
        }


class ImageLoader(ImageDataset):
    """Image-only dataset loader.

    Extend ``ImageDataset`` to load images from a specified ``root``. Support
    single image files, directories, or glob patterns. Use primarily for
    inference pipelines where no ground-truth labels are available.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        root     : Path,
        split    : Split                    = Split.PREDICT,
        transform: A.Compose | None         = None,
        classlist: Path | ClassList | None  = None,
        verbose  : bool                     = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            root: Root path to load images from.
            split: Data split subset to use. Defaults to Split.PREDICT.
            transform: Transformations to apply. Defaults to None.
            classlist: Either a .yaml file containing the classes definitions,
                or a ClassList instance. Defaults to None.
            verbose: Verbosity mode. Defaults to True.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Raises:
            TypeError: If ``root``, ``split``, or ``verbose`` is invalid.
        """
        # Validate inputs
        if not isinstance(root, (str, Path)):
            raise TypeError(
                f"Expected 'root' to be a str or Path, but got {type(root).__name__}."
            )
        if not isinstance(split, (str, Split)):
            raise TypeError(
                f"Expected 'split' to be a str or Split, but got {type(split).__name__}."
            )
        if not isinstance(verbose, bool):
            raise TypeError(
                f"Expected 'verbose' to be a bool, but got {type(verbose).__name__}."
            )

        # Continue the initialization chain
        super().__init__(
            root      = root,
            split     = split,
            transform = transform,
            classlist = classlist,
            verbose   = verbose,
            *args, **kwargs
        )

    # --- Data Loading ----
    def _load_primary_data(self) -> list:
        """Load primary modality data files in the dataset.

        Raises:
            FileNotFoundError: If the ``root`` path is invalid.
        """
        root = self.root

        if root.is_image_file():
            paths = [root]
        elif "*" in str(root):
            # Using iglob (iterator) is more memory efficient than glob.glob
            paths = [Path(p) for p in glob.iglob(str(root), recursive=True)]
        elif root.is_dir() and root.exists():
            paths = list(root.rglob("*"))
        else:
            raise FileNotFoundError(f"Dataset root not found at: {root}")

        if not paths:
            return []

        images: list[Image] = []
        disable_pbar = self.disable_pbar
        split_str    = self.split_str

        with create_progress_bar(disable=disable_pbar) as pbar:
            paths = sorted(paths)
            desc  = f"Listing {self.__class__.__name__} {split_str} image(s)"
            for path in pbar.track(sequence=paths, description=desc):
                if path.is_image_file(exist=True):
                    images.append(Image(data=path, root=root))

        return images

# endregion


# ==============================================================================
# region UTILITIES
# ==============================================================================


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
