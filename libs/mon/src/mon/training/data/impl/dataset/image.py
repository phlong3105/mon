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
from collections import OrderedDict
from typing import Any, override

import numpy as np
import torch

from mon.core import create_progress_bar, log, Path, Split
from mon.core.dtypes import ClassList, Image
from mon.training.augment import albumentations as A
from ...base import Dataset, Modality
from ...comp import BatchCollateMixin, MultimodalDataLoadMixin


# ==============================================================================
# region IMAGE DATASETS
# ==============================================================================

class ImageDataset(Dataset, MultimodalDataLoadMixin, BatchCollateMixin):
    """Image dataset base class.

    Extend ``Dataset`` with ``MultimodalDataLoadMixin`` and ``BatchCollateMixin``
    to support image-based datasets with multiple modalities.

    Attributes:
        subroot (str, optional): Name of the subdirectory within the dataset's
            ``root`` (i.e., ``root/subroot``). Use this if the current dataset
            is a subset of another dataset. `Should be defined in subclasses.`
        splits (list[Split]): List of supported splits. `Must be defined in
            subclasses.`
        modalities (OrderedDict[str, Modality]): Dictionary defining the dataset
            modalities. The first key in the dictionary is the primary modality,
            which is loaded first and used to guide the loading of other
            modalities. `Must be defined in subclasses.`
        classlist (ClassList, optional): Class definitions for the dataset.
            `Should be defined in subclasses or set during initialization
    """

    subroot: str = ""
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST, Split.PREDICT]
    modalities = OrderedDict(
        image=Modality(name="image", module=Image, type="image"),
    )
    classlist: ClassList | None = None

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        root: Path | str,
        split: Split | str = Split.TRAIN,
        transform: A.Compose | dict | None = None,
        classlist: ClassList | Path | str | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            root (Path | str): Absolute path to the dataset root directory.
            split (Split | str): Data split subset to use. Must be one of the
                supported ``splits``. Defaults to Split.TRAIN.
            transform (A.Compose | dict, optional): Transformations to apply.
                Defaults to None.
            classlist (ClassList | Path | str, optional): Class definitions for
                the dataset. Can be a ``ClassList`` instance or a path to a
                .yaml file. Defaults to None.
            verbose (bool): Verbosity mode. Defaults to True.

        Raises:
            ValueError: If ``modalities`` is empty.
        """
        # Validate inputs
        if not self.modalities:
            raise ValueError(f"Expected 'modalities' to be a non-empty dict.")

        # Continue the initialization chain
        super().__init__(
            root=root,
            split=split,
            classlist=classlist,
            verbose=verbose,
            *args, **kwargs
        )

        # Assign attributes
        self.transform = transform

    @override
    def __del__(self):
        """Finalize the object.

        Close the dataset loading mechanism and release resources.
        """
        pass

    # --- Representation ---
    @override
    def __repr__(self) -> str:
        """Return the official string representation for developers."""
        lines = [f"Dataset {self.__class__.__name__}"]
        lines += [f"Number of datapoints: {len(self)}"]
        if self.root:
            lines += [f"Root location: {self.root}"]
        if self.transform:
            lines += [repr(self.transform)]
        return "\n".join(lines)

    # --- Container / Sequence Methods ---
    @override
    def __len__(self) -> int:
        """Return the length of the container."""
        pk, _ = self.primary_modality
        return len(self.datapoints[pk])

    @override
    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return an item at the given ``index``.

        Args:
            index (int): Index of datapoint.

        Returns:
            dict[str, Any]: A datapoint dictionary containing all modalities,
                each associated with a 'key'.
        """
        # Fetch datapoint
        data = self.get_underlying_data(index=index)
        meta = data.pop("meta")

        transform = self.transform

        if transform:
            pk, _ = self.primary_modality
            if pk != "image":
                data["image"] = data.pop(pk)

            # Filter None values efficiently
            augmented = transform(**{k: v for k, v in data.items() if v is not None})

            # Revert 'image' back to the primary modality key if necessary
            if pk != "image":
                augmented[pk] = augmented.pop("image")

            # Update data in-place (Faster than |= for small dicts)
            data.update(augmented)

            # Vectorized-style type casting
            for k, v in data.items():
                if v is not None:
                    # Converts non‑float tensors/arrays to float32
                    if isinstance(v, torch.Tensor) and v.dtype != torch.float32:
                        data[k] = v.to(torch.float32)
                    elif isinstance(v, np.ndarray) and v.dtype != np.float32:
                        data[k] = v.astype(np.float32)

        return {**data, "meta": meta}

    # --- Properties ---
    @property
    def transform(self) -> A.Compose | None:
        """Return the transformation pipeline."""
        return self._transform

    @transform.setter
    def transform(self, transform: A.Compose | dict | None):
        """Set the transformation operations.

        Args:
            transform (A.Compose | dict, optional): Transformations to apply.
                Defaults to None.

        Raises:
            TypeError: If ``transform`` is not an instance of albumentations.Compose.
        """
        if transform is None:
            self._transform = None
            return

        if isinstance(transform, dict):
            transform = A.Compose(**transform)
        if not isinstance(transform, A.Compose):
            raise TypeError(
                f"Expected 'transform' to be an instance of "
                f"albumentations.Compose, but got {type(transform).__name__}."
            )

        # Add additional targets to A.Compose if needed.
        existing_targets = transform.processors.get("additional_targets", {})
        new_targets = {
            k: v.type for k, v in self.modalities.items()
            if v.type and v.module and k not in A.TARGET_TYPES and k not in existing_targets
        }

        if new_targets:
            transform.add_targets(additional_targets=new_targets)

        self._transform = transform

    # --- Data Loading ---
    @override
    def verify(self):
        """Verify dataset integrity.

        Raises:
            RuntimeError: If no datapoints are found or if modality lengths are
                inconsistent.
        """
        if len(self) <= 0:
            raise RuntimeError(
                f"No datapoints in the dataset: {self.__class__.__name__}."
            )

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
    @override
    def get_datapoint(self, index: int) -> dict[str, Any]:
        """Get a datapoint at the specified ``index``.

        Args:
            index (int): Index of datapoint.

        Returns:
            dict[str, Any]: A datapoint dictionary containing all modalities,
                each associated with a 'key'.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset of size {len(self)}."
            )

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
        root: Path | str,
        split: Split | str = Split.PREDICT,
        transform: A.Compose | dict | None = None,
        classlist: ClassList | Path | str | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            root (Path | str): Absolute path to the dataset root directory.
            split (Split | str): Data split subset to use. Must be one of the
                supported ``splits``. Defaults to Split.PREDICT.
            transform (A.Compose | dict, optional): Transformations to apply.
                Defaults to None.
            classlist (ClassList | Path | str, optional): Class definitions for
                the dataset. Can be a ``ClassList`` instance or a path to a
                .yaml file. Defaults to None.
            verbose (bool): Verbosity mode. Defaults to True.
        """
        # Continue the initialization chain
        super().__init__(
            root=root,
            split=split,
            transform=transform,
            classlist=classlist,
            verbose=verbose,
            *args, **kwargs,
        )

    # --- Data Loading ----
    @override
    def _load_primary_data(self) -> list[Any]:
        """Load primary modality data files in the dataset.

        Returns:
            list[Any]: List of primary modality data files.

        Raises:
            FileNotFoundError: If the ``root`` path is invalid.
        """
        root = self.root

        # List all image files under the root
        if root.is_image_file():
            paths = [root]
        elif "*" in str(root):
            # Using iglob (iterator) is more memory efficient than glob.glob
            paths = [Path(p) for p in glob.iglob(str(root), recursive=True)]
        elif root.is_dir() and root.exists():
            paths = list(root.rglob("*"))
        else:
            raise FileNotFoundError(f"Dataset root not found at: {root}")

        # Load images
        if not paths:
            return []

        images: list[Image] = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            paths = sorted(paths)
            desc = f"Listing {self.__class__.__name__} {self.split_str} image(s)"
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
