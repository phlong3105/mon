#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for building datasets and dataloaders.

This module provides functions to build datasets and dataloaders from various
data sources. It supports parsing data sources to create appropriate dataset
and dataloader instances, handling different input types such as dataset names,
directories, and video files.
"""

__all__ = [
    "build_dataloader",
    "build_dataset",
    "parse_data_dir",  # Re-exported for convenience
]

from typing import Any

from mon.core import DATASETS, parse_data_dir, Path, Split
from .datasets import Dataset, ImageLoader, VideoLoaderCV
from .loading import DataLoader


# --- Builder ---
def build_dataset(
    src      : Path | str,
    data_root: Path = None,
    transform: Any  = None,
    verbose  : bool = False,
    **kwargs
) -> tuple[str, Dataset]:
    """Parses given ``src`` to a corresponding dataset.
    
    Args:
        src: An input data source
        data_root: Dataset root dir. Defaults to None.
        transform: Transforms to apply to the dataset. Defaults to None.
        verbose: If True, enables verbose output. Defaults to False.
        **kwargs: Additional keyword arguments for the dataset.
        
    Returns:
        tuple[str, BaseDataset]: Dataset name and dataset.
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
        dataset = VideoLoaderCV(root=src, transform=transform, verbose=verbose, **kwargs)
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
    """Parses given ``src`` to a corresponding dataloader.

    Args:
        src: An input data source
        data_root: Dataset root dir. Defaults to None.
        transform: Transforms to apply to the dataset. Defaults to None.
        batch_size: Number of samples per batch. Defaults to 1.
        verbose: If True, enables verbose output. Defaults to False.
        **kwargs: Additional keyword arguments for the dataset.

    Returns:
        tuple[str, DataLoader]: Dataset name and dataloader.

    Raises:
        ValueError: If ``src`` is invalid.
    """
    data_name, dataset = build_dataset(src, data_root, transform, verbose)
    dataloader         = DataLoader(dataset, batch_size=batch_size, **kwargs)
    return data_name, dataloader
