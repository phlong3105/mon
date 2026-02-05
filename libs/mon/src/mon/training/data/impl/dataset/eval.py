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

from typing import Any

import box
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

    Define two main modalities: ``image`` and ``target``. Primarily used for
    separated evaluation pipelines outside the train/eval/test loop.

    Attributes:
        transform: Transformations for input and target.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        input_dir : Path,
        target_dir: Path             | None = None,
        transform : A.Compose        | None = None,
        classlist : Path | ClassList | None = None,
        verbose   : bool                    = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            input_dir: Absolute path to the input data directory.
            target_dir: Absolute path to the target directory. Defaults to None.
            transform: Transformations to apply to input and target. Defaults to None.
            classlist: Either a .yaml file containing the classes definitions,
                or a ClassList instance. Defaults to None.
            verbose: Verbosity mode. Defaults to True.
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        super().__init__(
            input_dir  = input_dir,
            target_dir = target_dir,
            classlist  = classlist,
            verbose    = verbose,
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
        if self.transform:
            lines += [repr(self.transform)]
        return "\n".join(lines)

    # --- Container / Sequence Methods ---
    def __len__(self) -> int:
        """Return the length of the container."""
        return len(self.datapoints["image"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return an item at the given ``index``.

        Args:
            index: Index to access.
        """
        # Fetch datapoint
        data = self._get_underlying_data(index=index)
        meta = data.pop("meta")  # Remove metadata from datapoint for easier augmentation ops.

        transform = self.transform

        if transform is not None:
            # Optimized transformation branch
            if self.has_target:
                augmented      = transform(image=data["image"], target=data["target"])
                data["image"]  = augmented["image"]
                data["target"] = augmented["target"]
            else:
                augmented      = transform(image=data["image"])
                data["image"]  = augmented["image"]

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
    def set_transform(self, value: Any):
        """Set the transformation operations.

        Args:
            value: Transformations for input and target.

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
        if self.has_target:
            # Albumentations stores additional targets in a specific dict;
            # check if 'target' is already there to avoid overhead.
            if "target" not in value.processors.get("additional_targets", {}):
                value.add_targets({"target": "image"})

        self.transform = value

    # --- Data Loading ---
    def _load_data(self) -> dict[str, Any]:
        """Load core data for the dataset.

        Returns:
            Dictionary containing lists of datapoints for each modality.
        """
        disable_pbar = self.disable_pbar
        input_dir    = self.input_dir
        has_target   = self.has_target
        target_dir   = self.target_dir if has_target else None

        # List image
        images = []
        with create_progress_bar(disable=disable_pbar) as pbar:
            paths = sorted(input_dir.rglob("*"))
            desc  = f"Listing {self.__class__.__name__} input image(s)"
            for path in pbar.track(sequence=paths, description=desc):
                if path.is_image_file(exist=True):
                    images.append(Image(data=path, root=input_dir))
        datapoints = {"image": images}

        # List target
        if has_target:
            targets = []
            with create_progress_bar(disable=disable_pbar) as pbar:
                desc = f"Listing {self.__class__.__name__} target image(s)"
                for image in pbar.track(sequence=images, description=desc):
                    target_file = target_dir / image.path.name
                    target_file = target_file.image_file(exist=True)
                    if target_file.is_image_file(exist=True):
                        targets.append(Image(data=target_file, root=target_dir))
            datapoints["target"] = targets
        else:
            datapoints["target"] = None

        # List metadata
        datapoints["meta"] = [i.meta for i in images]

        return datapoints

    def verify(self):
        """Verify dataset integrity.

        Raises:
            RuntimeError: If no datapoints are found or if modality lengths are
                inconsistent.
        """
        if len(self) <= 0:
            raise RuntimeError(f"No datapoints in the dataset: {self.__class__.__name__}.")

        for k, v in self.datapoints.items():
            if v in [None, []]:
                raise RuntimeError(f"Datapoint modality '{k}' is empty!")
            elif len(v) != len(self):
                raise RuntimeError(
                    f"Datapoint modality '{k}' has inconsistent length with the dataset: "
                    f"{len(v)} != {len(self)}."
                )

        if self.verbose:
            log(f"Number of datapoints: {len(self)}.")

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
