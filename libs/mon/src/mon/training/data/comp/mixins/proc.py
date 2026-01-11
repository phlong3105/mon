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
        if not batch:
            return {}
        
        # Faster Transposition: Group items by key
        # This replaces the complex zip(*[b.values()]) logic
        keys     = batch[0].keys()
        collated = {k: [d[k] for d in batch] for k in keys}
        
        for k, v in collated.items():
            # Immutability for metadata
            if k == "meta":
                collated[k] = tuple(v)  # Freeze metadata to prevent runtime modification
                continue
            
            # Performance: O(1) Type Checking
            # We check only the first element, assuming batch homogeneity
            first_item = v[0]
            
            if first_item is None:
                collated[k] = None
            elif isinstance(first_item, torch.Tensor):
                collated[k] = torch.stack(v, dim=0)
            elif isinstance(first_item, np.ndarray):
                collated[k] = np.stack(v, axis=0)
            # v remains a list for other types (like strings or custom objects)
        
        return collated


# --- Parallelize (Multi-processing/threading logic) ---
