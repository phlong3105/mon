#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DataLoader.

This module defines a custom DataLoader class that extends PyTorch's DataLoader
with additional convenience features for initializing datasets. It supports
dataset configuration through dictionaries objects and automatically handles
collate functions and pin memory settings.
"""

__all__ = [
    "DataLoader",
]

from typing import Any

import box
import cv2

cv2.setNumThreads(0)
# Optionally, disable OpenCL if not needed or causing issues
# cv2.ocl.setUseOpenCL(False)

from mon.core.factory import DATASETS
from torch.utils.data import dataloader
from .dataset import BaseDataset


# ----- DataLoader -----
class DataLoader(dataloader.DataLoader):
    """An extension of ``torch.utils.data.dataloader.DataLoader`` with convenience
    initialization for datasets.
    """

    def __init__(
        self,
        dataset    : BaseDataset | dict | box.Box,
        batch_size : int  = 1,
        shuffle    : bool = False,
        num_workers: int  = 4,
        collate_fn : Any  = None,
        pin_memory : bool = True,
        drop_last  : bool = False,
        *args, **kwargs
    ):
        if isinstance(dataset, dict | box.Box):
            dataset = DATASETS.build(**dataset)
        collate_fn = getattr(dataset, "collate_fn", collate_fn)
        pin_memory = True if collate_fn else pin_memory
        
        super().__init__(
            dataset     = dataset,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = num_workers,
            drop_last   = drop_last,
            collate_fn  = collate_fn,
            pin_memory  = pin_memory,
            *args, **kwargs
        )
