#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training data containers.

This package contains various data containers used for training machine learning
models.

Notes:
    - Design Pattern: Component-Based Framework.
    - Goal: Build systems from reusable, interchangeable components that can be
      independently developed, tested, and maintained. Each component is a modular
      unit with well-defined interfaces, encapsulating specific functionality
      that can be assembled, replaced, or reused across applications.
    - Structure:
        ::
        
            component/
            ├── __init__.py             # Exposes all
            ├── base.py                 # Base classes and mixins
            ├── comp/                   # Reusable components
            │   ├── __init__.py
            │   ├── base.py             # Component base classes and mixins
            │   └── ...                 # Concrete component
            ├── impl/                   # Concrete classes using base + components
            │   ├── __init__.py
            │   ├── concrete_impl.py    # Example implementation
            │   └── ...
            ├── usages/                 # Example usages of concrete implementations
            │   ├── __init__.py
            │   └── ...
            └── utils.py                # Utility functions and helpers
"""

__all__ = [
    "BatchCollateMixin",
    "DataLoadMixin",
    "DataLoader",
    "Dataset",
    "ImageDataset",
    "ImageEvalDataset",
    "ImageLoader",
    "InputTargetLoadMixin",
    "Modalities",
    "Modality",
    "MultimodalDataLoadMixin",
    "RegistrableMixin",
    "RootLoadMixin",
    "VideoLoader",
    "build_dataloader",
    "build_dataset",
    "is_video_dataset",
]

from typing import Any

from mon.core import DATASETS, parse_data_dir, Path, Split
from .base import Dataset, Modalities, Modality
from .comp import *
from .impl import *
from .usages import *
from .utils import *


# ==============================================================================
# REGISTRY & FACTORY (Type Resolution)
# ==============================================================================

# --- Register (Adding new spokes to the hub) ---


# --- Resolve (Retrieving spokes by name/key) ---
def build_dataset(
    src      : Path | str,
    data_root: Path = None,
    transform: Any  = None,
    verbose  : bool = False,
    **kwargs
) -> tuple[str, Dataset]:
    """Build a dataset from a given source.
    
    Args:
        src: An input data source
        data_root: Dataset root dir. Defaults to None.
        transform: Transforms to apply to the dataset. Defaults to None.
        verbose: If True, enables verbose output. Defaults to False.
        **kwargs: Additional keyword arguments for the dataset.
        
    Returns:
        A tuple containing the dataset name and the dataset instance.

    Raises:
        ValueError: If ``src`` is invalid.
    """
    src = Path(src)

    if src.stem in DATASETS:
        src       = src.stem
        root      = parse_data_dir(root=data_root, data_dir=src)
        config    = kwargs | {
            "name"     : src,
            "root"     : root,
            "split"    : Split.TEST,
            "transform": transform,
            "verbose"  : verbose,
        }
        data_name = src
        dataset   = DATASETS.build(**config)
    elif src.is_dir():
        data_name = src.name
        dataset   = ImageLoader(root=src, transform=transform, verbose=verbose, **kwargs)
    elif src.is_video_file():
        data_name = src.name
        dataset = VideoLoader(root=src, transform=transform, verbose=verbose, **kwargs)
    else:
        raise ValueError(f"``src`` is invalid: {src}.")

    return data_name, dataset


def build_dataloader(
    src       : Path | str,
    data_root : Path = None,
    transform : Any  = None,
    batch_size: int  = 1,
    verbose   : bool = False,
    **kwargs
) -> tuple[str, DataLoader]:
    """Build a dataloader from a given source.

    Args:
        src: An input data source
        data_root: Dataset root dir. Defaults to None.
        transform: Transforms to apply to the dataset. Defaults to None.
        batch_size: Number of samples per batch. Defaults to 1.
        verbose: If True, enables verbose output. Defaults to False.
        **kwargs: Additional keyword arguments for the dataset.

    Returns:
        A tuple containing the dataset name and the dataloader instance.

    Raises:
        ValueError: If ``src`` is invalid.
    """
    data_name, dataset = build_dataset(src, data_root, transform, verbose)
    dataloader         = DataLoader(dataset, batch_size=batch_size, **kwargs)
    return data_name, dataloader
