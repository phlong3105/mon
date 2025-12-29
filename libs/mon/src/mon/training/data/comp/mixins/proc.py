#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Mixins for complex operations.

This module provides higher-level logic mixins that might involve multiple atomic
operations to manipulate the data containers.
"""

__all__ = [
    "BatchCollateMixin",
]

import abc
from typing import Any

import numpy as np
import torch


# ==============================================================================
# PIPELINE ABSTRACTIONS (Step Definitions)
# ==============================================================================

# --- Execute (The main workflow engine) ---


# --- Dispatch (Routing data to specific ops) ---


# ==============================================================================
# DATA AUGMENTATION (Data Expansion)
# ==============================================================================

# --- Mutate (Wrappers for stochastic changes) ---


# --- Enrich (Combining images with labels/masks) ---


# ==============================================================================
# BATCH PROCESSORS (Parallel Execution)
# ==============================================================================

# --- Batchify (Grouping data into batches) ---
class BatchCollateMixin:
    """A mixin class that adds batch collation functionality to data containers.

    Define the collate function for batching datapoints when using with a DataLoader.
    """
    
    # noinspection PyTypeChecker
    @staticmethod
    def collate_fn(batch: list[dict]) -> dict[str, Any]:
        """Collate a batch of input items for torch.utils.data.dataset.DataLoader.

        By default, ``batch`` is a list of dicts, where each dict is a datapoint.
        We need to collate these into a single dict where each key corresponds to
        a modality and the values are stacked tensors or arrays.

        Args:
            batch: List of dicts, each dict is a datapoint.

        Returns:
            Collated dictionary for torch.utils.data.dataset.DataLoader.
        """
        zipped = {
            k: list(v)
            for k, v in zip(batch[0].keys(), zip(*[b.values() for b in batch]))
        }
        
        for k, v in zipped.items():
            # Skip certain keys
            if k in ["meta"]:
                continue
            # Collates datapoints into stacked tensors or arrays
            if v is None:
                zipped[k] = None
            elif all(isinstance(i, torch.Tensor) for i in v):
                zipped[k] = torch.stack(v, dim=0)
            elif all(isinstance(i, np.ndarray) for i in v):
                zipped[k] = np.stack(v, axis=0)
        
        return zipped


# --- Parallelize (Multi-processing/threading logic) ---
