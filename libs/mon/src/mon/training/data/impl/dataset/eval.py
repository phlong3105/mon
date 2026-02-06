#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluation datasets.

This module provides dataset classes specifically designed for evaluation
purposes.
"""

from __future__ import annotations

__all__ = [
    "ImageEvalDataset",
]

from typing import Any, Optional, override

import numpy as np
import torch

from mon.core import create_progress_bar, log, Path
from mon.core.dtypes import ClassList, Image
from mon.training.augment import albumentations as A
from ...base import Dataset
from ...comp import BatchCollateMixin, InputTargetLoadMixin


# ==============================================================================
# region IMAGE EVAL DATASETS
# ==============================================================================

class ImageEvalDataset(Dataset, InputTargetLoadMixin, BatchCollateMixin):
    """Image quality assessment (IQA) dataset.

    Extends the base ``Dataset`` class with ``InputTargetLoadMixin`` and
    ``BatchCollateMixin`` to support evaluation pipelines where input and target
    are images.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        input_dir: Path | str,
        target_dir: Path | str | None = None,
        transform: A.Compose | dict | None = None,
        classlist: ClassList | Path | str | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            input_dir (Path | str): Absolute path to the input directory.
            target_dir (Path | str, optional): Absolute path to the target directory.
                Defaults to None.
            transform (A.Compose | dict, optional): Transformations to apply.
                Defaults to None.
            classlist (ClassList | Path | str, optional): Class definitions for
                the dataset. Can be a ``ClassList`` instance or a path to a
                .yaml file. Defaults to None.
            verbose (bool): Verbosity mode. Defaults to True.
        """
        super().__init__(
            input_dir=input_dir,
            target_dir=target_dir,
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
        if self.transform:
            lines += [repr(self.transform)]
        return "\n".join(lines)

    # --- Container / Sequence Methods ---
    @override
    def __len__(self) -> int:
        """Return the length of the container."""
        return len(self.datapoints["image"])

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
            if self.has_target:
                augmented = transform(image=data["image"], target=data["target"])
                data["image"] = augmented["image"]
                data["target"] = augmented["target"]
            else:
                augmented = transform(image=data["image"])
                data["image"] = augmented["image"]

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
        if self.has_target:
            # Albumentations stores additional targets in a specific dict;
            # check if 'target' is already there to avoid overhead.
            if "target" not in transform.processors.get("additional_targets", {}):
                transform.add_targets({"target": "image"})

        self._transform = transform

    # --- Data Loading ---
    @override
    def _load_data(self) -> dict[str, list[Any]]:
        """Load the core data of the dataset.

        Returns:
            dict[str, list[Any]]: Dictionary containing lists of datapoints for
                each modality.
        """
        datapoints: dict[str, Optional[list[Any]]] = {}

        # List image
        images = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            paths = sorted(self.input_dir.rglob("*"))
            desc = f"Listing {self.__class__.__name__} input image(s)"
            for path in pbar.track(sequence=paths, description=desc):
                if path.is_image_file(exist=True):
                    images.append(Image(data=path, root=self.input_dir))
        datapoints["image"] = images

        # List target
        if self.has_target:
            targets = []
            with create_progress_bar(disable=self.disable_pbar) as pbar:
                desc = f"Listing {self.__class__.__name__} target image(s)"
                for image in pbar.track(sequence=images, description=desc):
                    target_file = self.target_dir / image.path.name
                    target_file = target_file.image_file(exist=True)
                    if target_file.is_image_file(exist=True):
                        targets.append(Image(data=target_file, root=self.target_dir))
            datapoints["target"] = targets
        else:
            datapoints["target"] = None

        # List metadata
        datapoints["meta"] = [i.meta for i in images]

        return datapoints

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

        for k, v in self.datapoints.items():
            if v in [None, []]:
                raise RuntimeError(f"Datapoint modality '{k}' is empty!")
            elif len(v) != len(self):
                raise RuntimeError(
                    f"Datapoint modality '{k}' has inconsistent length with "
                    f"the dataset: {len(v)} != {len(self)}."
                )

        if self.verbose:
            log(f"Number of datapoints: {len(self)}.")

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

# endregion


# ==============================================================================
# UTILITIES
# ==============================================================================


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
