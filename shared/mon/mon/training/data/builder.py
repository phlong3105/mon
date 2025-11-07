#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements functions to build datasets and dataloaders from various
data sources.

It supports parsing data directories and creating appropriate dataset and dataloader
objects based on the input source type.
"""

__all__ = [
    "build_dataloader",
    "build_dataset",
    "parse_data_dir",
]

from typing import Any

from mon.constants import ROOT_DIR
from mon.core import DATASETS, Path, Split
from .dataloader import DataLoader
from .dataset import BaseDataset, ImageLoader, VideoLoaderCV


# ----- Builder -----
def build_dataset(
    src      : Path | str,
    data_root: Path = None,
    transform: Any  = None,
    verbose  : bool = False,
    **kwargs
) -> tuple[str, BaseDataset]:
    """Parses given ``src`` to a corresponding dataset.

    Args:
        src: An input data source
        data_root: Dataset root dir. Default: ``None``.
        transform: Transforms to apply to the dataset. Default: ``None``.
        verbose: If ``True``, enables verbose output. Default: ``False``.
        **kwargs: Additional keyword arguments for the ``BaseDataset``.

    Returns:
        A ``tuple`` of data name and ``BaseDataset``.

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
        dataset = ImageLoader(root=src, transform=transform, verbose=verbose, **kwargs)
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
        data_root: Dataset root dir. Default: ``None``.
        transform: Transforms to apply to the dataset. Default: ``None``.
        batch_size: Batch size for the dataloader. Default: ``1``.
        verbose: If ``True``, enables verbose output. Default: ``False``.
        **kwargs: Additional keyword arguments for the ``DataLoader``.

    Returns:
        A ``tuple`` of data name and ``Dataloader``.

    Raises:
        ValueError: If ``src`` is invalid.
    """
    data_name, dataset = build_dataset(src, data_root, transform, verbose)
    dataloader         = DataLoader(dataset, batch_size=batch_size, **kwargs)
    return data_name, dataloader


# ----- Parsing -----
def parse_data_dir(root: Path, data_dir: Path = "") -> Path:
    """Parses the absolute data directory path from given components.

    Args:
        root: Root directory.
        data_dir: Data directory.

    Returns:
        Parsed the absolute path of the data directory.
    """
    root_      = Path(root)     if root     not in [None, "None", ""] else ROOT_DIR
    data_dir_  = Path(data_dir) if data_dir not in [None, "None", ""] else None

    candidates = []
    if data_dir_:
        candidates.extend([
            data_dir_,
            root_    / data_dir_,
            root_    / "data" / data_dir_,
            ROOT_DIR / data_dir_,
            ROOT_DIR / "data" / data_dir_
        ])
    candidates.extend([
        root_    / "data",
        ROOT_DIR / "data"
    ])

    for d in candidates:
        if d.is_dir():
            return d
    raise FileNotFoundError(f"``data_dir`` not found: {data_dir}.")
