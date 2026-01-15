#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training data containers.

This package contains various data containers commonly used in training machine
learning models.

Notes:
    - Design Pattern: Component-Based Framework.
    - Goal: Define a collection of components that can be assembled to form
      concrete <Name> implementations.
    - Structure:
        ::

            component/
            ├── __init__.py             # Exposes all
            ├── base.py                 # Base classes and mixins
            ├── comp/                   # Components
            │   ├── __init__.py
            │   └── ...
            ├── impl/                   # Implementations
            │   ├── __init__.py
            │   └── ...
            ├── usages/                 # Usages
            │   ├── __init__.py
            │   └── ...
            └── utils.py                # Utilities and helpers
"""

from __future__ import annotations

from typing import Any

from mon.core import DATASETS, resolve_data_dir, Path, Split
from .base import *
from .comp import *
from .impl import *
from .usages import *
from .utils import *


# ==============================================================================
# region REGISTRY & FACTORY
# ==============================================================================

def build_dataset(
    src      : Path | str,
    data_root: Path | str | None = None,
    transform: Any               = None,
    verbose  : bool              = False,
    **kwargs
) -> tuple[str, Dataset]:
    """Build a dataset from a given source.

    Args:
        src: An input data source
        data_root: Dataset root dir. Defaults to None.
        transform: Transforms to apply to the dataset. Defaults to None.
        verbose: Verbosity mode. Defaults to False.
        **kwargs: Additional keyword arguments for the dataset.

    Returns:
        A tuple containing the dataset name and the dataset instance.

    Raises:
        TypeError: If ``src``, ``data_root``, or ``verbose`` is invalid.
        ValueError: If ``src`` is invalid.
    """
    if not isinstance(src, (str, Path)):
        raise TypeError(f"Expected 'src' to be a str or Path, but got {type(src).__name__}.")
    if data_root is not None and not isinstance(data_root, (str, Path)):
        raise TypeError(
            f"Expected 'data_root' to be a str, Path, or None, but got {type(data_root).__name__}."
        )
    if not isinstance(verbose, bool):
        raise TypeError(f"Expected 'verbose' to be a bool, but got {type(verbose).__name__}.")

    src = Path(src)

    # 1. src is a registered dataset name
    if src.stem in DATASETS:
        src       = src.stem
        root      = resolve_data_dir(root=data_root, data_dir=src)
        config    = kwargs | {
            "name"     : src,
            "root"     : root,
            "split"    : Split.TEST,
            "transform": transform,
            "verbose"  : verbose,
        }
        data_name = src
        dataset   = DATASETS.build(**config)
    # 2. src is a directory of images
    elif src.is_dir():
        data_name = src.name
        dataset   = ImageLoader(
            root      = src,
            transform = transform,
            verbose   = verbose,
            **kwargs
        )
    # 3. src is a video file
    elif src.is_video_file():
        data_name = src.name
        dataset   = VideoLoader(
            root      = src,
            transform = transform,
            verbose   = verbose,
            **kwargs
        )
    else:
        raise ValueError(
            f"Unsupported 'src': {src}. Must be a registered dataset name, a directory, "
            f"or a video file."
        )

    return data_name, dataset


def build_dataloader(
    src       : Path | str,
    data_root : Path | str | None = None,
    transform : Any               = None,
    batch_size: int               = 1,
    verbose   : bool              = False,
    **kwargs
) -> tuple[str, DataLoader]:
    """Build a dataloader from a given source.

    Args:
        src: An input data source
        data_root: Dataset root dir. Defaults to None.
        transform: Transforms to apply to the dataset. Defaults to None.
        batch_size: Number of samples per batch. Defaults to 1.
        verbose: Verbosity mode. Defaults to False.
        **kwargs: Additional keyword arguments for the dataset.

    Returns:
        A tuple containing the dataset name and the dataloader instance.

    Raises:
        TypeError: If ``batch_size`` or ``verbose`` is invalid.
        ValueError: If ``src`` is invalid.
    """
    if not isinstance(batch_size, int):
        raise TypeError(f"Expected 'batch_size' to be an int, but got {type(batch_size).__name__}.")
    if batch_size <= 0:
        raise ValueError(f"Expected 'batch_size' to be positive, but got {batch_size}.")
    if not isinstance(verbose, bool):
        raise TypeError(f"Expected 'verbose' to be a bool, but got {type(verbose).__name__}.")

    data_name, dataset = build_dataset(
        src       = src,
        data_root = data_root,
        transform = transform,
        verbose   = verbose,
    )
    dataloader = DataLoader(
        dataset    = dataset,
        batch_size = batch_size,
        **kwargs
    )
    return data_name, dataloader

# endregion
