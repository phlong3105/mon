#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Dataset.

This package contains "full-stack" dataset utilities and several concrete
dataset implementations.

File Structure:
::

    mon.core/
    ├── __init__.py
    ├── augment/        # Augmentation pipelines
    ├── base/           # Base components (datasets, dataloaders, etc.)
    ├── transforms/     # Data transforms
    └── zoo/            # Concrete dataset implementations
"""

from __future__ import annotations

from typing import Any

from box import Box

from mon.core import DATASETS, log_error, Path, resolve_dataset_dir, Split
from .base import *
from .zoo import *


# ==============================================================================
# region REGISTRY & FACTORY
# ==============================================================================

def build_dataset(
    src: dict | Path,
    dataset_dir: Path = Path.cwd(),
    split: Split = Split.TEST ,
    transforms: Any = None,
    keep_original: bool = False,
    verbose: bool = False,
    *args, **kwargs
) -> tuple[str, Dataset]:
    """Build a dataset from a given source.

    Args:
        src (dict | Path): The source to build the dataset from. It can
            be either a dataset configuration dictionary or a path to the
            dataset directory.
        dataset_dir (Path | None, optional): Specific dataset directory.
            Defaults to the current working directory.
        split (Split, optional): Dataset split to use. Defaults to Split.TEST.
        transforms (Any, optional): Transformations to apply. Defaults to None.
        keep_original (bool, optional): Whether to keep the original data
            alongside the transformed data. Defaults to False.
        verbose (bool, optional): Verbosity mode. Defaults to False.
        *args: Additional positional arguments for ``Dataset`` constructor.
        **kwargs: Additional keyword arguments for ``Dataset`` constructor.

    Returns:
        tuple[str, Dataset]: A tuple containing the dataset name and the
            corresponding ``Dataset`` instance.
    """
    # 1. If src is a dataset config dict, build the Dataset instance directly
    if isinstance(src, (Box, dict)):
        dataset_ = Dataset.from_config(src)
        name_ = getattr(dataset_, "name", dataset_.__class__.__name__)
        return name_, dataset_

    # 2. Otherwise, build the dataset instance based on the source type
    # (path or name)
    # Validate inputs
    if not isinstance(src, (Path, str)):
        raise TypeError(
            f"expected src to be a path or a dataset name, got {type(src).__name__}."
        )

    # Build the corresponding Dataset instance
    config = kwargs | {
        "split": split,
        "transforms": transforms,
        "keep_original": keep_original,
        "verbose": verbose,
    }
    src: Path = Path(src).normalize()

    # 2.1. If src is a registered dataset name, use the corresponding class
    if src.name in DATASETS:
        if dataset_dir is None:
            raise RuntimeError("dataset root is required to build dataset.")

        module: Dataset = DATASETS[src.name]
        dataset_dir = resolve_dataset_dir(dataset_name=src.name, data_root=dataset_dir)
        config["root"] = dataset_dir
        return src.name, module.from_config(config)

    # 2.2. If src is a directory of images, build an ImageDataset
    if src.is_dir() or src.is_image_file():
        config["root"] = src
        return src.name, ImageOnlyDataset.from_config(config)

    # 2.3. If src is a video file, build a VideoOnlyDataset
    if src.is_video_file():
        config["root"] = src
        return src.name, VideoOnlyDataset.from_config(config)

    # 3. If neither is a dataset nor a dataloader config dict, return None
    raise RuntimeError(f"cannot build dataset from source {src.as_posix()}.")


def build_dataloader(
    src: dict | Path,
    dataset_dir: Path = Path.cwd(),
    split: Split = Split.TEST,
    transforms: Any = None,
    keep_original: bool = False,
    batch_size: int = 1,
    verbose: bool = False,
    *args, **kwargs
) -> tuple[str, DataLoader]:
    """Build a dataloader from a given source.

    Args:
        src (dict | Path): A dataloader configuration dictionary or a source path.
        dataset_dir (Path | None, optional): Specific dataset directory.
            Defaults to the current working directory.
        split (Split, optional): Dataset split to use. Defaults to Split.TEST.
        transforms (Any, optional): Transformations to apply. Defaults to None.
        keep_original (bool, optional): Whether to keep the original data
            alongside the transformed data. Defaults to False.
        batch_size (int, optional): Batch size for the dataloader. Defaults to 1.
        verbose (bool, optional): Verbosity mode. Defaults to False.
        *args: Additional positional arguments for ``Dataset`` constructor.
        **kwargs: Additional keyword arguments for ``Dataset`` constructor.

    Returns:
        tuple[str, DataLoader]: A tuple containing the dataset name and the
            corresponding ``DataLoader`` instance.
    """
    # 1. If src is a dataloader config dict, build the DataLoader instance directly
    if isinstance(src, (Box, dict)):
        dataloader_ = DataLoader.from_config(src)
        name_ = dataloader_.name
        return name_, dataloader_

    # 2. Otherwise, build the dataset and DataLoader instances separately
    if isinstance(src, (Path, str)):
        src = Path(src).normalize()
        name, dataset_ = build_dataset(
            src=src,
            dataset_dir=dataset_dir,
            split=split,
            transforms=transforms,
            keep_original=keep_original,
            verbose=verbose,
            *args, **kwargs
        )
        if dataset_ is None:
            raise RuntimeError(f"cannot build dataset from source {src.as_posix()}.")

        dataloader_ = DataLoader(dataset=dataset_, batch_size=batch_size, *args, **kwargs)
        return name, dataloader_

    # 3. If neither is a dataset nor a dataloader config dict, return None
    raise RuntimeError(f"cannot build dataloader from source {src.as_posix()}.")

# endregion
