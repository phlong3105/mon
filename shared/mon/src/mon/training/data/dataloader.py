#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for custom DataLoader.

This module provides an extended DataLoader class that simplifies the
initialization process by allowing datasets to be specified as configuration
dictionaries. It integrates with the mon framework's dataset building utilities
and supports common DataLoader parameters.
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

from mon.core import DATASETS
from torch.utils.data import dataloader
from .dataset import BaseDataset


# ----- DataLoader -----
class DataLoader(dataloader.DataLoader):
    """An extended DataLoader class for loading datasets."""

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
        """Initializes the DataLoader instance.
        
        Args:
            dataset (BaseDataset or dict or box.Box): The dataset to load data
                from, or a configuration dictionary to build the dataset.
            batch_size (int): Number of samples per batch. Defaults to 1.
            shuffle (bool): Whether to shuffle the data at every epoch. Defaults
                to False.
            num_workers (int): Number of subprocesses to use for data loading.
                Defaults to 4.
            collate_fn (callable, optional): Function to merge a list of samples
                to form a mini-batch. Defaults to None.
            pin_memory (bool): If True, the data loader will copy Tensors into
                CUDA pinned memory before returning them. Defaults to True.
            drop_last (bool): If True, drops the last incomplete batch if the
                dataset size is not divisible by the batch size. Defaults to False.
        """
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
