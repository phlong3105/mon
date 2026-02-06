#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Custom data loader with convenience dataset building.

This module provides an extended DataLoader class that simplifies the
initialization process by allowing datasets to be specified as configuration
dictionaries. It integrates with the mon framework's dataset building utilities
and supports common DataLoader parameters.
"""

from __future__ import annotations

__all__ = [
    "DataLoader",
]

from typing import Any

import cv2
from torch.utils.data.dataloader import DataLoader as DataLoader_

from mon.core import DATASETS
from ..base import Dataset

cv2.setNumThreads(0)
# Optionally, disable OpenCL if not needed or causing issues
# cv2.ocl.setUseOpenCL(False)


# ==============================================================================
# region DATALOADER
# ==============================================================================

class DataLoader(DataLoader_):
    """Convenience wrapper around PyTorch's DataLoader class.

    Support dataset construction from configuration dictionaries.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        dataset: Dataset | dict,
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
            dataset (Dataset | dict): Dataset to load data from. Can be an
                instance of a Dataset or a configuration dictionary to build one.
            batch_size (int): Number of samples per batch to load. Defaults to 1.
            shuffle (bool): If True, the data will be reshuffled at every epoch.
            num_workers (int): Number of subprocesses to use for data loading.
                Defaults to 4.
            collate_fn (Callable): Merges a list of samples to form a mini-batch.
                Defaults to None.
            pin_memory (bool): If True, the data loader will copy Tensors into
                CUDA pinned memory before returning them. Defaults to True.
            drop_last (bool): If True, the sampler will drop the last batch if
                its size is less than batch_size. Defaults to False.
        """
        # Build dataset if it's a config object
        if isinstance(dataset, dict):
            dataset = DATASETS.build(**dataset)

        # Cache collate_fn to avoid repeated getattr calls
        # We prioritize the dataset's internal collation logic if it exists
        collate_fn = getattr(dataset, "collate_fn", collate_fn)

        # Only pin memory if we are actually using a collate function that
        # returns Tensors (usually implied if collate_fn exists)
        pin_memory = pin_memory if collate_fn is not None else False

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

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
