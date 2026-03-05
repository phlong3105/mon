#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DataLoader Data Structures.

This module provides convenience wrappers around PyTorch's DataLoader class.
"""

from __future__ import annotations

__all__ = [
    "DataLoader",
]

from typing import Any

import cv2
from box import Box
from torch.utils.data.dataloader import DataLoader as DataLoader_
from mon.core import DATASETS

from .dataset import Dataset

cv2.setNumThreads(0)
# Optionally, disable OpenCL if not needed or causing issues
# cv2.ocl.setUseOpenCL(False)


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class DataLoader(DataLoader_):
    """Convenience wrapper around PyTorch's DataLoader class.

    Support dataset construction from configuration dictionaries.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 4,
        collate_fn: Any = None,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            dataset (Dataset): Dataset to load data from.
            batch_size (int, optional): Number of samples per batch to load.
                Defaults to 1.
            shuffle (bool, optional): If True, the data will be reshuffled at
                every epoch. Defaults to False.
            num_workers (int, optional): Number of subprocesses to use for data
                loading. Defaults to 4.
            collate_fn (Callable, optional): Merges a list of samples to form a
                mini-batch. Defaults to None.
            pin_memory (bool, optional): If True, the data loader will copy
                Tensors into CUDA pinned memory before returning them.
                Defaults to True.
            drop_last (bool, optional): If True, the sampler will drop the last
                batch if its size is less than batch_size. Defaults to False.
        """
        # Validate inputs
        if not isinstance(dataset, Dataset):
            raise TypeError(
                f"Expected 'dataset' to be an instance of Dataset, "
                f"but got {type(dataset).__name__}."
            )

        # Cache collate_fn to avoid repeated getattr calls
        # We prioritize the dataset's internal collation logic if it exists
        collate_fn = getattr(dataset, "collate_fn", collate_fn)

        # Only pin memory if we are actually using a collate function that
        # returns Tensors (usually implied if collate_fn exists)
        pin_memory = pin_memory if collate_fn is not None else False

        # Continue the initialization chain
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            *args, **kwargs
        )

    # --- Creation ---
    @classmethod
    def from_config(cls, config: dict[str, Any], **kwargs) -> "DataLoader":
        """Create a new instance from a configuration dictionary."""
        # Extract relevant keys
        dataset = config.pop("dataset", None)

        # Validate inputs
        if isinstance(dataset, (Box, dict)):
            name = dataset.get("name")
            if name in DATASETS:
                dataset = DATASETS.build(**dataset)
            else:
                dataset = Dataset.from_config(dataset)
        if not isinstance(dataset, Dataset):
            raise TypeError(
                f"Expected 'dataset' to be a configuration dict or an instance "
                f"of Dataset, but got {type(dataset).__name__}."
            )

        # Return the new instance
        config |= kwargs
        return cls(dataset=dataset, **config)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
