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

from mon.core import DATASETS, Path, PathLike, resolve_dataset_dir
from .base import *
from .transform import (
    build_compose,
    build_transform,
    ComposeLike,
    TransformLike,
)
from .zoo import *


# ==============================================================================
# region REGISTRY & FACTORY
# ==============================================================================

def build_dataset(
    src: PathLike,
    dataset_dir: PathLike | None = None,
    cwd: PathLike | None = None,
    transforms: ComposeLike | None = None,
    verbose: bool = False,
    *args, **kwargs
) -> tuple[str, Dataset]:
    """Build a dataset from a given source.

    Args:
        src (PathLike): An input data source or a dataset name.
        dataset_dir (PathLike, optional): Specific dataset directory.
            Defaults to None.
        cwd (PathLike, optional): The current working directory to resolve the
            dataset directory from. Defaults to None.
        transforms (ComposeLike, optional): Transformations to apply.
            Defaults to None.
        verbose (bool, optional): Verbosity mode. Defaults to False.
        *args: Additional positional arguments for ``Dataset`` constructor.
        **kwargs: Additional keyword arguments for ``Dataset`` constructor.

    Returns:
        tuple[str, Dataset]: A tuple containing the dataset name and the
            corresponding ``Dataset`` instance.
    """
    # Validate inputs
    if not isinstance(src, (Path, str)):
        raise TypeError(
            f"Expected 'src' to be a source path or a dataset name, "
            f"but got {type(src).__name__}."
        )

    # Build the corresponding Dataset instance
    src = Path(src).normalize()

    if src.name in DATASETS:
        # If src is a registered dataset name, use the corresponding class
        module: Dataset = DATASETS[src.name]
        dataset_dir = resolve_dataset_dir(
            dataset_name=src.name,
            data_root=dataset_dir or cwd
        )
        config = kwargs | {
            "root": dataset_dir,
            "transforms": transforms,
            "verbose": verbose,
        }
        return src.name, module.from_config(config)
    elif src.is_dir() or src.is_image_file():
        config = kwargs | {
            "root": src,
            "transforms": transforms,
            "verbose": verbose,
        }
        return src.name, ImageDataset.from_config(config)
    elif src.is_video_file():
        config = kwargs | {
            "root": src,
            "transforms": transforms,
            "verbose": verbose,
        }
        return src.name, VideoOnlyDataset.from_config(config)
    else:
        raise ValueError(f"Unsupported source type: {src}.")


def build_dataloader(
    src: PathLike,
    dataset_dir: PathLike | None = None,
    cwd: PathLike | None = None,
    transforms: ComposeLike | None = None,
    batch_size: int = 1,
    verbose: bool = False,
    *args, **kwargs
) -> tuple[str, DataLoader]:
    """Build a dataloader from a given source.

    Args:
        src (PathLike): An input data source or a dataset name.
        dataset_dir (PathLike, optional): Specific dataset directory.
            Defaults to None.
        cwd (PathLike, optional): The current working directory to resolve the
            dataset directory from. Defaults to None.
        transforms (ComposeLike, optional): Transformations to apply.
            Defaults to None.
        batch_size (int, optional): Batch size for the dataloader. Defaults to 1.
        verbose (bool, optional): Verbosity mode. Defaults to False.
        *args: Additional positional arguments for ``Dataset`` constructor.
        **kwargs: Additional keyword arguments for ``Dataset`` constructor.

    Returns:
        tuple[str, DataLoader]: A tuple containing the dataset name and the
            corresponding ``DataLoader`` instance.
    """
    name, dataset_ = build_dataset(
        src=src,
        dataset_dir=dataset_dir,
        cwd=cwd,
        transforms=transforms,
        verbose=verbose,
        *args, **kwargs
    )
    dataloader_ = DataLoader(
        dataset=dataset_, batch_size=batch_size, *args, **kwargs
    )
    return name, dataloader_

# endregion
