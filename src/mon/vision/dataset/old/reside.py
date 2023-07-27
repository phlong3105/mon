#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements RESIDE datasets and datamodules.

---------------------------------------------------------------------
| Subset                                         | Number of Images |
---------------------------------------------------------------------
| Indoor Training Set (ITS)                      | 13,990	        |
| Outdoor Training Set (OTS)                     |      	        |
| Synthetic Objective Testing Set (SOTS) Indoor  | 500              |
| Synthetic Objective Testing Set (SOTS) Outdoor | 500              |
| Hybrid Subjective Testing Set (HSTS)           | 20               |
---------------------------------------------------------------------
"""

from __future__ import annotations

__all__ = [
    "RESIDEHSTS", "RESIDEHSTSDataModule", "RESIDEITS", "RESIDEITSDataModule",
    "RESIDEITSv2", "RESIDEITSv2DataModule", "RESIDEOTS", "RESIDEOTSDataModule",
    "RESIDESOTS", "RESIDESOTSDataModule", "RESIDESOTSIndoor",
    "RESIDESOTSIndoorDataModule", "RESIDESOTSOutdoor",
    "RESIDESOTSOutdoorDataModule",
]

import argparse

from mon.vision.transform import transform as t
from mon.vision.typing import (
    CallableType, ClassLabelsType, Ints, ModelPhaseType, PathType,
    Strs, TransformType, VisionBackendType,
)
from torch.utils.data import random_split

from mon import core
from mon.vision import constant
from mon.vision.dataset import base


# region Dataset

@constant.DATASETS.register(name="reside-hsts")
class RESIDEHSTS(base.ImageEnhancementDataset):
    """RESIDE-HSTS dataset.
    
    We present a comprehensive study and evaluation of existing single image
    dehazing algorithms, using a new large-scale benchmark consisting of both
    synthetic and real-world hazy images, called REalistic Single Image DEhazing
    (RESIDE). RESIDE highlights diverse data sources and image contents, and is
    divided into five subsets, each serving different training or evaluation
    purposes. We further provide a rich variety of criteria for dehazing
    algorithm evaluation, ranging from full-reference metrics, to no-reference
    metrics, to subjective evaluation and the novel task-driven evaluation.
    Experiments on RESIDE shed light on the comparisons and limitations of
    latest dehazing algorithms, and suggest promising future directions.
    
    Args:
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Default: "train".
        image_size: The desired datapoint shape preferably in a channel-last
        format.
            Default: (3, 256, 256).
        classlabels: A :class:`ClassLabels` object. Default: None.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Default: False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Default: False.
        backend: The image processing backend. Default: VISION_BACKEND.
        verbose: Verbosity. Default: True.
    """
    
    def __init__(
        self,
        root: PathType = constant.DATA_DIR / "reside",
        split: str = "train",
        image_size: Ints = (3, 256, 256),
        classlabels: ClassLabelsType | None = None,
        transform: TransformType | None = None,
        target_transform: TransformType | None = None,
        transforms: TransformType | None = None,
        cache_data: bool = False,
        cache_images: bool = False,
        backend: VisionBackendType = constant.VISION_BACKEND,
        verbose: bool = True,
        *args, **kwargs
    ):
        super().__init__(
            root=root,
            split=split,
            image_size=image_size,
            classlabels=classlabels,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            cache_data=cache_data,
            cache_images=cache_images,
            backend=backend,
            verbose=verbose,
            *args, **kwargs
        )
    
    def get_images(self):
        """Get image files."""
        if self.split not in ["test"]:
            core.console.log(
                f"{self.__class__.__name__} dataset only supports "
                f"'split': 'test'. Get: {self.split}."
            )
        
        self.images: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            pattern = self.root / "sots" / "hsts"
            for path in pbar.track(
                list(pattern.rglob("haze/*.png")),
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split} images"
            ):
                self.images.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )
    
    def get_labels(self):
        """Get label files."""
        self.labels: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split} labels"
            ):
                dir = img.path.parents[1]
                stem = str(img.path.stem).split("_")[0]
                path = dir / "clear" / f"{stem}.png"
                self.labels.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )


@constant.DATASETS.register(name="reside-its")
class RESIDEITS(base.ImageEnhancementDataset):
    """RESIDE-ITS dataset.
    
    We present a comprehensive study and evaluation of existing single image
    dehazing algorithms, using a new large-scale benchmark consisting of both
    synthetic and real-world hazy images, called REalistic Single Image DEhazing
    (RESIDE). RESIDE highlights diverse data sources and image contents, and is
    divided into five subsets, each serving different training or evaluation
    purposes. We further provide a rich variety of criteria for dehazing
    algorithm evaluation, ranging from full-reference metrics, to no-reference
    metrics, to subjective evaluation and the novel task-driven evaluation.
    Experiments on RESIDE shed light on the comparisons and limitations of
    latest dehazing algorithms, and suggest promising future directions.
    
    Args:
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Default: "train".
        image_size: The desired datapoint shape preferably in a channel-last
        format.
            Default: (3, 256, 256).
        classlabels: A :class:`ClassLabels` object. Default: None.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Default: False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Default: False.
        backend: The image processing backend. Default: VISION_BACKEND.
        verbose: Verbosity. Default: True.
    """
    
    def __init__(
        self,
        root: PathType = constant.DATA_DIR / "reside",
        split: str = "train",
        image_size: Ints = (3, 256, 256),
        classlabels: ClassLabelsType | None = None,
        transform: TransformType | None = None,
        target_transform: TransformType | None = None,
        transforms: TransformType | None = None,
        cache_data: bool = False,
        cache_images: bool = False,
        backend: VisionBackendType = constant.VISION_BACKEND,
        verbose: bool = True,
        *args, **kwargs
    ):
        super().__init__(
            root=root,
            split=split,
            image_size=image_size,
            classlabels=classlabels,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            cache_data=cache_data,
            cache_images=cache_images,
            backend=backend,
            verbose=verbose,
            *args, **kwargs
        )
    
    def get_images(self):
        """Get image files."""
        if self.split not in ["train", "val"]:
            core.console.log(
                f"{self.__class__.__name__} dataset only supports "
                f"'split': 'train' or 'val'. Get: {self.split}."
            )
        
        self.images: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            pattern = self.root / "its" / self.split
            for path in pbar.track(
                list(pattern.rglob("haze/*.png")),
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split} images"
            ):
                self.images.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )
    
    def get_labels(self):
        """Get label files."""
        self.labels: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split} labels"
            ):
                dir = img.path.parents[1]
                stem = str(img.path.stem).split("_")[0]
                path = dir / "clear" / f"{stem}.png"
                self.labels.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )


@constant.DATASETS.register(name="reside-its-v2")
class RESIDEITSv2(base.ImageEnhancementDataset):
    """RESIDE-ITSv2 dataset.
    
    We present a comprehensive study and evaluation of existing single image
    dehazing algorithms, using a new large-scale benchmark consisting of both
    synthetic and real-world hazy images, called REalistic Single Image DEhazing
    (RESIDE). RESIDE highlights diverse data sources and image contents, and is
    divided into five subsets, each serving different training or evaluation
    purposes. We further provide a rich variety of criteria for dehazing
    algorithm evaluation, ranging from full-reference metrics, to no-reference
    metrics, to subjective evaluation and the novel task-driven evaluation.
    Experiments on RESIDE shed light on the comparisons and limitations of
    latest dehazing algorithms, and suggest promising future directions.
    
    Args:
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Default: "train".
        image_size: The desired datapoint shape preferably in a channel-last
        format.
            Default: (3, 256, 256).
        classlabels: A :class:`ClassLabels` object. Default: None.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Default: False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Default: False.
        backend: The image processing backend. Default: VISION_BACKEND.
        verbose: Verbosity. Default: True.
    """
    
    def __init__(
        self,
        root: PathType = constant.DATA_DIR / "reside",
        split: str = "train",
        image_size: Ints = (3, 256, 256),
        classlabels: ClassLabelsType | None = None,
        transform: TransformType | None = None,
        target_transform: TransformType | None = None,
        transforms: TransformType | None = None,
        cache_data: bool = False,
        cache_images: bool = False,
        backend: VisionBackendType = constant.VISION_BACKEND,
        verbose: bool = True,
        *args, **kwargs
    ):
        super().__init__(
            root=root,
            split=split,
            image_size=image_size,
            classlabels=classlabels,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            cache_data=cache_data,
            cache_images=cache_images,
            backend=backend,
            verbose=verbose,
            *args, **kwargs
        )
    
    def get_images(self):
        """Get image files."""
        if self.split not in ["train"]:
            core.console.log(
                f"{self.__class__.__name__} dataset only supports "
                f"'split': 'train'. Get: {self.split}."
            )
        
        self.images: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            pattern = self.root / "its_v2"
            for path in pbar.track(
                list(pattern.rglob("haze/*.png")),
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split} images"
            ):
                print(path)
                self.images.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )
    
    def get_labels(self):
        """Get label files."""
        self.labels: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split} labels"
            ):
                dir = img.path.parents[1]
                stem = str(img.path.stem).split("_")[0]
                path = dir / "clear" / f"{stem}.png"
                self.labels.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )


@constant.DATASETS.register(name="reside-ots")
class RESIDEOTS(base.ImageEnhancementDataset):
    """RESIDE-OTS dataset.
    
    We present a comprehensive study and evaluation of existing single image
    dehazing algorithms, using a new large-scale benchmark consisting of both
    synthetic and real-world hazy images, called REalistic Single Image DEhazing
    (RESIDE). RESIDE highlights diverse data sources and image contents, and is
    divided into five subsets, each serving different training or evaluation
    purposes. We further provide a rich variety of criteria for dehazing
    algorithm evaluation, ranging from full-reference metrics, to no-reference
    metrics, to subjective evaluation and the novel task-driven evaluation.
    Experiments on RESIDE shed light on the comparisons and limitations of
    latest dehazing algorithms, and suggest promising future directions.
    
    Args:
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Default: "train".
        image_size: The desired datapoint shape preferably in a channel-last
        format.
            Default: (3, 256, 256).
        classlabels: A :class:`ClassLabels` object. Default: None.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Default: False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Default: False.
        backend: The image processing backend. Default: VISION_BACKEND.
        verbose: Verbosity. Default: True.
    """
    
    def __init__(
        self,
        root: PathType = constant.DATA_DIR / "reside",
        split: str = "train",
        image_size: Ints = (3, 256, 256),
        classlabels: ClassLabelsType | None = None,
        transform: TransformType | None = None,
        target_transform: TransformType | None = None,
        transforms: TransformType | None = None,
        cache_data: bool = False,
        cache_images: bool = False,
        backend: VisionBackendType = constant.VISION_BACKEND,
        verbose: bool = True,
        *args, **kwargs
    ):
        super().__init__(
            root=root,
            split=split,
            image_size=image_size,
            classlabels=classlabels,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            cache_data=cache_data,
            cache_images=cache_images,
            backend=backend,
            verbose=verbose,
            *args, **kwargs
        )
    
    def get_images(self):
        """Get image files."""
        if self.split not in ["train"]:
            core.console.log(
                f"{self.__class__.__name__} dataset only supports "
                f"'split': 'train'. Get: {self.split}."
            )
        
        self.images: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            pattern = self.root / "ots"
            for path in pbar.track(
                list(pattern.rglob("haze/*.jpg")),
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split} images"
            ):
                self.images.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )
    
    def get_labels(self):
        """Get label files."""
        self.labels: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split} labels"
            ):
                dir = img.path.parents[1]
                stem = str(img.path.stem).split("_")[0]
                path = dir / "clear" / f"{stem}.jpg"
                self.labels.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )


@constant.DATASETS.register(name="reside-sots")
class RESIDESOTS(base.ImageEnhancementDataset):
    """RESIDE-SOTS dataset.
    
    We present a comprehensive study and evaluation of existing single image
    dehazing algorithms, using a new large-scale benchmark consisting of both
    synthetic and real-world hazy images, called REalistic Single Image DEhazing
    (RESIDE). RESIDE highlights diverse data sources and image contents, and is
    divided into five subsets, each serving different training or evaluation
    purposes. We further provide a rich variety of criteria for dehazing
    algorithm evaluation, ranging from full-reference metrics, to no-reference
    metrics, to subjective evaluation and the novel task-driven evaluation.
    Experiments on RESIDE shed light on the comparisons and limitations of
    latest dehazing algorithms, and suggest promising future directions.
    
    Args:
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Default: "train".
        image_size: The desired datapoint shape preferably in a channel-last
        format.
            Default: (3, 256, 256).
        classlabels: A :class:`ClassLabels` object. Default: None.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Default: False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Default: False.
        backend: The image processing backend. Default: VISION_BACKEND.
        verbose: Verbosity. Default: True.
    """
    
    def __init__(
        self,
        root: PathType = constant.DATA_DIR / "reside",
        split: str = "train",
        image_size: Ints = (3, 256, 256),
        classlabels: ClassLabelsType | None = None,
        transform: TransformType | None = None,
        target_transform: TransformType | None = None,
        transforms: TransformType | None = None,
        cache_data: bool = False,
        cache_images: bool = False,
        backend: VisionBackendType = constant.VISION_BACKEND,
        verbose: bool = True,
        *args, **kwargs
    ):
        super().__init__(
            root=root,
            split=split,
            image_size=image_size,
            classlabels=classlabels,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            cache_data=cache_data,
            cache_images=cache_images,
            backend=backend,
            verbose=verbose,
            *args, **kwargs
        )
    
    def get_images(self):
        """Get image files."""
        if self.split not in ["test"]:
            core.console.log(
                f"{self.__class__.__name__} dataset only supports "
                f"'split': 'test'. Get: {self.split}."
            )
        
        self.images: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            pattern = self.root / "sots"
            for path in pbar.track(
                list(pattern.rglob("haze/*.png")),
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split} images"
            ):
                self.images.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )
    
    def get_labels(self):
        """Get label files."""
        self.labels: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split} labels"
            ):
                dir = img.path.parents[1]
                stem = str(img.path.stem).split("_")[0]
                path = dir / "clear" / f"{stem}.png"
                self.labels.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )


@constant.DATASETS.register(name="reside-sots-indoor")
class RESIDESOTSIndoor(base.ImageEnhancementDataset):
    """RESIDE-SOTS Indoor dataset.
    
    We present a comprehensive study and evaluation of existing single image
    dehazing algorithms, using a new large-scale benchmark consisting of both
    synthetic and real-world hazy images, called REalistic Single Image DEhazing
    (RESIDE). RESIDE highlights diverse data sources and image contents, and is
    divided into five subsets, each serving different training or evaluation
    purposes. We further provide a rich variety of criteria for dehazing
    algorithm evaluation, ranging from full-reference metrics, to no-reference
    metrics, to subjective evaluation and the novel task-driven evaluation.
    Experiments on RESIDE shed light on the comparisons and limitations of
    latest dehazing algorithms, and suggest promising future directions.
    
    Args:
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Default: "train".
        image_size: The desired datapoint shape preferably in a channel-last
        format.
            Default: (3, 256, 256).
        classlabels: A :class:`ClassLabels` object. Default: None.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Default: False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Default: False.
        backend: The image processing backend. Default: VISION_BACKEND.
        verbose: Verbosity. Default: True.
    """
    
    def __init__(
        self,
        root: PathType = constant.DATA_DIR / "reside",
        split: str = "train",
        image_size: Ints = (3, 256, 256),
        classlabels: ClassLabelsType | None = None,
        transform: TransformType | None = None,
        target_transform: TransformType | None = None,
        transforms: TransformType | None = None,
        cache_data: bool = False,
        cache_images: bool = False,
        backend: VisionBackendType = constant.VISION_BACKEND,
        verbose: bool = True,
        *args, **kwargs
    ):
        super().__init__(
            root=root,
            split=split,
            image_size=image_size,
            classlabels=classlabels,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            cache_data=cache_data,
            cache_images=cache_images,
            backend=backend,
            verbose=verbose,
            *args, **kwargs
        )
    
    def get_images(self):
        """Get image files."""
        if self.split not in ["test"]:
            core.console.log(
                f"{self.__class__.__name__} dataset only supports "
                f"'split': 'test'. Get: {self.split}."
            )
        
        self.images: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            pattern = self.root / "sots" / "indoor"
            for path in pbar.track(
                list(pattern.rglob("haze/*.png")),
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split} images"
            ):
                self.images.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )
    
    def get_labels(self):
        """Get label files."""
        self.labels: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split} labels"
            ):
                dir = img.path.parents[1]
                stem = str(img.path.stem).split("_")[0]
                path = dir / "clear" / f"{stem}.png"
                self.labels.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )


@constant.DATASETS.register(name="reside-sots-outdoor")
class RESIDESOTSOutdoor(base.ImageEnhancementDataset):
    """RESIDE-SOTS Outdoor dataset.
    
    We present a comprehensive study and evaluation of existing single image
    dehazing algorithms, using a new large-scale benchmark consisting of both
    synthetic and real-world hazy images, called REalistic Single Image DEhazing
    (RESIDE). RESIDE highlights diverse data sources and image contents, and is
    divided into five subsets, each serving different training or evaluation
    purposes. We further provide a rich variety of criteria for dehazing
    algorithm evaluation, ranging from full-reference metrics, to no-reference
    metrics, to subjective evaluation and the novel task-driven evaluation.
    Experiments on RESIDE shed light on the comparisons and limitations of
    latest dehazing algorithms, and suggest promising future directions.
    
    Args:
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Default: "train".
        image_size: The desired datapoint shape preferably in a channel-last
        format.
            Default: (3, 256, 256).
        classlabels: A :class:`ClassLabels` object. Default: None.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Default: False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Default: False.
        backend: The image processing backend. Default: VISION_BACKEND.
        verbose: Verbosity. Default: True.
    """
    
    def __init__(
        self,
        root: PathType = constant.DATA_DIR / "reside",
        split: str = "train",
        image_size: Ints = (3, 256, 256),
        classlabels: ClassLabelsType | None = None,
        transform: TransformType | None = None,
        target_transform: TransformType | None = None,
        transforms: TransformType | None = None,
        cache_data: bool = False,
        cache_images: bool = False,
        backend: VisionBackendType = constant.VISION_BACKEND,
        verbose: bool = True,
        *args, **kwargs
    ):
        super().__init__(
            root=root,
            split=split,
            image_size=image_size,
            classlabels=classlabels,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            cache_data=cache_data,
            cache_images=cache_images,
            backend=backend,
            verbose=verbose,
            *args, **kwargs
        )
    
    def get_images(self):
        """Get image files."""
        if self.split not in ["test"]:
            core.console.log(
                f"{self.__class__.__name__} dataset only supports "
                f"'split': 'test'. Get: {self.split}."
            )
        
        self.images: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            pattern = self.root / "sots" / "outdoor"
            for path in pbar.track(
                list(pattern.rglob("haze/*.png")),
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split} images"
            ):
                self.images.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )
    
    def get_labels(self):
        """Get label files."""
        self.labels: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing RESIDE-SOTS Outdoor {self.split} labels"
            ):
                dir = img.path.parents[1]
                stem = str(img.path.stem).split("_")[0]
                path = dir / "clear" / f"{stem}.png"
                self.labels.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )


# endregion


# region Datamodule

@constant.DATAMODULES.register(name="reside-hsts")
class RESIDEHSTSDataModule(base.DataModule):
    """RESIDE-HSTS datamodule.
    
    Args:
        root: A root directory where the data is stored.
        shape: The desired datapoint shape preferably in a channel-last format.
            Default: (3, 256, 256).
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        batch_size: The number of samples in one forward pass. Default: 1.
        devices: A list of devices to use. Default: 0.
        shuffle: If True, reshuffle the datapoints at the beginning of every
            epoch. Default: True.
        collate_fn: The function used to fused datapoint together when using
            :param:`batch_size` > 1.
        verbose: Verbosity. Default: True.
    """
    
    def __init__(
        self,
        root: PathType = constant.DATA_DIR / "reside",
        shape: Ints = (3, 256, 256),
        transform: TransformType | None = None,
        target_transform: TransformType | None = None,
        transforms: TransformType | None = None,
        batch_size: int = 1,
        devices: Ints | Strs = 0,
        shuffle: bool = True,
        collate_fn: CallableType | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        super().__init__(
            root=root,
            shape=shape,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            batch_size=batch_size,
            devices=devices,
            shuffle=shuffle,
            collate_fn=collate_fn,
            verbose=verbose,
            *args, **kwargs
        )
    
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk, or that need
        to be done only from a single GPU in distributed settings:
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, phase: ModelPhaseType | None = None):
        """Use this method to do things on every device:
            - Count number of classes.
            - Build classlabels vocabulary.
            - Prepare train/val/test splits.
            - Apply transformations.
            - Define :attr:`collate_fn` for your custom dataset.

        Args:
            phase: The model phase. One of:
                - "training" : prepares :attr:'train' and :attr:'val'.
                - "testing"  : prepares :attr:'test'.
                - "inference": prepares :attr:`predict`.
                - None:      : prepares all.
                Default: None.
        """
        core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = constant.ModelPhase.from_value(
            phase
        ) if phase is not None else phase
        
        # Assign train/val datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TRAINING]:
            full_dataset = RESIDEHSTS(
                root=self.root,
                split="test",
                image_size=self.shape,
                transform=self.transform,
                target_transform=self.target_transform,
                transforms=self.transforms,
                verbose=self.verbose,
                **self.dataset_kwargs
            )
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train, self.val = random_split(
                full_dataset, [train_size, val_size]
            )
            self.classlabels = getattr(full_dataset, "classlabels", None)
            self.collate_fn = getattr(full_dataset, "collate_fn", None)
        
        # Assign test datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TESTING]:
            self.test = RESIDEHSTS(
                root=self.root,
                split="test",
                image_size=self.shape,
                transform=self.transform,
                target_transform=self.target_transform,
                transforms=self.transforms,
                verbose=self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn = getattr(self.test, "collate_fn", None)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@constant.DATAMODULES.register(name="reside-its")
class RESIDEITSDataModule(base.DataModule):
    """RESIDE-ITS datamodule.
    
    Args:
        root: A root directory where the data is stored.
        shape: The desired datapoint shape preferably in a channel-last format.
            Default: (3, 256, 256).
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        batch_size: The number of samples in one forward pass. Default: 1.
        devices: A list of devices to use. Default: 0.
        shuffle: If True, reshuffle the datapoints at the beginning of every
            epoch. Default: True.
        collate_fn: The function used to fused datapoint together when using
            :param:`batch_size` > 1.
        verbose: Verbosity. Default: True.
    """
    
    def __init__(
        self,
        root: PathType = constant.DATA_DIR / "reside",
        shape: Ints = (3, 256, 256),
        transform: TransformType | None = None,
        target_transform: TransformType | None = None,
        transforms: TransformType | None = None,
        batch_size: int = 1,
        devices: Ints | Strs = 0,
        shuffle: bool = True,
        collate_fn: CallableType | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        super().__init__(
            root=root,
            shape=shape,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            batch_size=batch_size,
            devices=devices,
            shuffle=shuffle,
            collate_fn=collate_fn,
            verbose=verbose,
            *args, **kwargs
        )
    
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk, or that need
        to be done only from a single GPU in distributed settings:
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, phase: ModelPhaseType | None = None):
        """Use this method to do things on every device:
            - Count number of classes.
            - Build classlabels vocabulary.
            - Prepare train/val/test splits.
            - Apply transformations.
            - Define :attr:`collate_fn` for your custom dataset.

        Args:
            phase: The model phase. One of:
                - "training" : prepares :attr:'train' and :attr:'val'.
                - "testing"  : prepares :attr:'test'.
                - "inference": prepares :attr:`predict`.
                - None:      : prepares all.
                Default: None.
        """
        core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = constant.ModelPhase.from_value(
            phase
        ) if phase is not None else phase
        
        # Assign train/val datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TRAINING]:
            self.train = RESIDEITS(
                root=self.root,
                split="train",
                image_size=self.shape,
                transform=self.transform,
                target_transform=self.target_transform,
                transforms=self.transforms,
                verbose=self.verbose,
                **self.dataset_kwargs
            )
            self.val = RESIDEITS(
                root=self.root,
                split="val",
                image_size=self.shape,
                transform=self.transform,
                target_transform=self.target_transform,
                transforms=self.transforms,
                verbose=self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.train, "classlabels", None)
            self.collate_fn = getattr(self.train, "collate_fn", None)
        
        # Assign test datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TESTING]:
            self.test = RESIDEITS(
                root=self.root,
                split="val",
                image_size=self.shape,
                transform=self.transform,
                target_transform=self.target_transform,
                transforms=self.transforms,
                verbose=self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn = getattr(self.test, "collate_fn", None)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@constant.DATAMODULES.register(name="reside-its-v2")
class RESIDEITSv2DataModule(base.DataModule):
    """RESIDE-ITSv2 datamodule.
    
    Args:
        root: A root directory where the data is stored.
        shape: The desired datapoint shape preferably in a channel-last format.
            Default: (3, 256, 256).
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        batch_size: The number of samples in one forward pass. Default: 1.
        devices: A list of devices to use. Default: 0.
        shuffle: If True, reshuffle the datapoints at the beginning of every
            epoch. Default: True.
        collate_fn: The function used to fused datapoint together when using
            :param:`batch_size` > 1.
        verbose: Verbosity. Default: True.
    """
    
    def __init__(
        self,
        root: PathType = constant.DATA_DIR / "reside",
        shape: Ints = (3, 256, 256),
        transform: TransformType | None = None,
        target_transform: TransformType | None = None,
        transforms: TransformType | None = None,
        batch_size: int = 1,
        devices: Ints | Strs = 0,
        shuffle: bool = True,
        collate_fn: CallableType | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        super().__init__(
            root=root,
            shape=shape,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            batch_size=batch_size,
            devices=devices,
            shuffle=shuffle,
            collate_fn=collate_fn,
            verbose=verbose,
            *args, **kwargs
        )
    
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk, or that need
        to be done only from a single GPU in distributed settings:
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, phase: ModelPhaseType | None = None):
        """Use this method to do things on every device:
            - Count number of classes.
            - Build classlabels vocabulary.
            - Prepare train/val/test splits.
            - Apply transformations.
            - Define :attr:`collate_fn` for your custom dataset.

        Args:
            phase: The model phase. One of:
                - "training" : prepares :attr:'train' and :attr:'val'.
                - "testing"  : prepares :attr:'test'.
                - "inference": prepares :attr:`predict`.
                - None:      : prepares all.
                Default: None.
        """
        core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = constant.ModelPhase.from_value(
            phase
        ) if phase is not None else phase
        
        # Assign train/val datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TRAINING]:
            full_dataset = RESIDEITSv2(
                root=self.root,
                split="train",
                image_size=self.shape,
                transform=self.transform,
                target_transform=self.target_transform,
                transforms=self.transforms,
                verbose=self.verbose,
                **self.dataset_kwargs
            )
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train, self.val = random_split(
                full_dataset, [train_size, val_size]
            )
            self.classlabels = getattr(full_dataset, "classlabels", None)
            self.collate_fn = getattr(full_dataset, "collate_fn", None)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@constant.DATAMODULES.register(name="reside-ots")
class RESIDEOTSDataModule(base.DataModule):
    """RESIDE-OTS datamodule.
    
    Args:
        root: A root directory where the data is stored.
        shape: The desired datapoint shape preferably in a channel-last format.
            Default: (3, 256, 256).
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        batch_size: The number of samples in one forward pass. Default: 1.
        devices: A list of devices to use. Default: 0.
        shuffle: If True, reshuffle the datapoints at the beginning of every
            epoch. Default: True.
        collate_fn: The function used to fused datapoint together when using
            :param:`batch_size` > 1.
        verbose: Verbosity. Default: True.
    """
    
    def __init__(
        self,
        root: PathType = constant.DATA_DIR / "reside",
        shape: Ints = (3, 256, 256),
        transform: TransformType | None = None,
        target_transform: TransformType | None = None,
        transforms: TransformType | None = None,
        batch_size: int = 1,
        devices: Ints | Strs = 0,
        shuffle: bool = True,
        collate_fn: CallableType | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        super().__init__(
            root=root,
            shape=shape,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            batch_size=batch_size,
            devices=devices,
            shuffle=shuffle,
            collate_fn=collate_fn,
            verbose=verbose,
            *args, **kwargs
        )
    
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk, or that need
        to be done only from a single GPU in distributed settings:
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, phase: ModelPhaseType | None = None):
        """Use this method to do things on every device:
            - Count number of classes.
            - Build classlabels vocabulary.
            - Prepare train/val/test splits.
            - Apply transformations.
            - Define :attr:`collate_fn` for your custom dataset.

        Args:
            phase: The model phase. One of:
                - "training" : prepares :attr:'train' and :attr:'val'.
                - "testing"  : prepares :attr:'test'.
                - "inference": prepares :attr:`predict`.
                - None:      : prepares all.
                Default: None.
        """
        core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = constant.ModelPhase.from_value(
            phase
        ) if phase is not None else phase
        
        # Assign train/val datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TRAINING]:
            full_dataset = RESIDEOTS(
                root=self.root,
                split="train",
                image_size=self.shape,
                transform=self.transform,
                target_transform=self.target_transform,
                transforms=self.transforms,
                verbose=self.verbose,
                **self.dataset_kwargs
            )
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train, self.val = random_split(
                full_dataset, [train_size, val_size]
            )
            self.classlabels = getattr(full_dataset, "classlabels", None)
            self.collate_fn = getattr(full_dataset, "collate_fn", None)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@constant.DATAMODULES.register(name="reside-sots")
class RESIDESOTSDataModule(base.DataModule):
    """RESIDE-SOTS datamodule.
    
    Args:
        root: A root directory where the data is stored.
        shape: The desired datapoint shape preferably in a channel-last format.
            Default: (3, 256, 256).
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        batch_size: The number of samples in one forward pass. Default: 1.
        devices: A list of devices to use. Default: 0.
        shuffle: If True, reshuffle the datapoints at the beginning of every
            epoch. Default: True.
        collate_fn: The function used to fused datapoint together when using
            :param:`batch_size` > 1.
        verbose: Verbosity. Default: True.
    """
    
    def __init__(
        self,
        root: PathType = constant.DATA_DIR / "reside",
        shape: Ints = (3, 256, 256),
        transform: TransformType | None = None,
        target_transform: TransformType | None = None,
        transforms: TransformType | None = None,
        batch_size: int = 1,
        devices: Ints | Strs = 0,
        shuffle: bool = True,
        collate_fn: CallableType | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        super().__init__(
            root=root,
            shape=shape,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            batch_size=batch_size,
            devices=devices,
            shuffle=shuffle,
            collate_fn=collate_fn,
            verbose=verbose,
            *args, **kwargs
        )
    
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk, or that need
        to be done only from a single GPU in distributed settings:
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, phase: ModelPhaseType | None = None):
        """Use this method to do things on every device:
            - Count number of classes.
            - Build classlabels vocabulary.
            - Prepare train/val/test splits.
            - Apply transformations.
            - Define :attr:`collate_fn` for your custom dataset.

        Args:
            phase: The model phase. One of:
                - "training" : prepares :attr:'train' and :attr:'val'.
                - "testing"  : prepares :attr:'test'.
                - "inference": prepares :attr:`predict`.
                - None:      : prepares all.
                Default: None.
        """
        core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = constant.ModelPhase.from_value(
            phase
        ) if phase is not None else phase
        
        # Assign train/val datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TRAINING]:
            full_dataset = RESIDESOTS(
                root=self.root,
                split="test",
                image_size=self.shape,
                transform=self.transform,
                target_transform=self.target_transform,
                transforms=self.transforms,
                verbose=self.verbose,
                **self.dataset_kwargs
            )
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train, self.val = random_split(
                full_dataset, [train_size, val_size]
            )
            self.classlabels = getattr(full_dataset, "classlabels", None)
            self.collate_fn = getattr(full_dataset, "collate_fn", None)
        
        # Assign test datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TESTING]:
            self.test = RESIDESOTS(
                root=self.root,
                split="test",
                image_size=self.shape,
                transform=self.transform,
                target_transform=self.target_transform,
                transforms=self.transforms,
                verbose=self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn = getattr(self.test, "collate_fn", None)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@constant.DATAMODULES.register(name="reside-ots-indoor")
class RESIDESOTSIndoorDataModule(base.DataModule):
    """RESIDE-SOTS Indoor datamodule.
    
    Args:
        root: A root directory where the data is stored.
        shape: The desired datapoint shape preferably in a channel-last format.
            Default: (3, 256, 256).
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        batch_size: The number of samples in one forward pass. Default: 1.
        devices: A list of devices to use. Default: 0.
        shuffle: If True, reshuffle the datapoints at the beginning of every
            epoch. Default: True.
        collate_fn: The function used to fused datapoint together when using
            :param:`batch_size` > 1.
        verbose: Verbosity. Default: True.
    """
    
    def __init__(
        self,
        root: PathType = constant.DATA_DIR / "reside",
        shape: Ints = (3, 256, 256),
        transform: TransformType | None = None,
        target_transform: TransformType | None = None,
        transforms: TransformType | None = None,
        batch_size: int = 1,
        devices: Ints | Strs = 0,
        shuffle: bool = True,
        collate_fn: CallableType | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        super().__init__(
            root=root,
            shape=shape,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            batch_size=batch_size,
            devices=devices,
            shuffle=shuffle,
            collate_fn=collate_fn,
            verbose=verbose,
            *args, **kwargs
        )
    
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk, or that need
        to be done only from a single GPU in distributed settings:
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, phase: ModelPhaseType | None = None):
        """Use this method to do things on every device:
            - Count number of classes.
            - Build classlabels vocabulary.
            - Prepare train/val/test splits.
            - Apply transformations.
            - Define :attr:`collate_fn` for your custom dataset.

        Args:
            phase: The model phase. One of:
                - "training" : prepares :attr:'train' and :attr:'val'.
                - "testing"  : prepares :attr:'test'.
                - "inference": prepares :attr:`predict`.
                - None:      : prepares all.
                Default: None.
        """
        core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = constant.ModelPhase.from_value(
            phase
        ) if phase is not None else phase
        
        # Assign train/val datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TRAINING]:
            full_dataset = RESIDESOTSIndoor(
                root=self.root,
                split="test",
                image_size=self.shape,
                transform=self.transform,
                target_transform=self.target_transform,
                transforms=self.transforms,
                verbose=self.verbose,
                **self.dataset_kwargs
            )
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train, self.val = random_split(
                full_dataset, [train_size, val_size]
            )
            self.classlabels = getattr(full_dataset, "classlabels", None)
            self.collate_fn = getattr(full_dataset, "collate_fn", None)
        
        # Assign test datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TESTING]:
            self.test = RESIDESOTSIndoor(
                root=self.root,
                split="test",
                image_size=self.shape,
                transform=self.transform,
                target_transform=self.target_transform,
                transforms=self.transforms,
                verbose=self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn = getattr(self.test, "collate_fn", None)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@constant.DATAMODULES.register(name="reside-ots-outdoor")
class RESIDESOTSOutdoorDataModule(base.DataModule):
    """RESIDE-SOTS Outdoor datamodule.
    
    Args:
        root: A root directory where the data is stored.
        shape: The desired datapoint shape preferably in a channel-last format.
            Default: (3, 256, 256).
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        batch_size: The number of samples in one forward pass. Default: 1.
        devices: A list of devices to use. Default: 0.
        shuffle: If True, reshuffle the datapoints at the beginning of every
            epoch. Default: True.
        collate_fn: The function used to fused datapoint together when using
            :param:`batch_size` > 1.
        verbose: Verbosity. Default: True.
    """
    
    def __init__(
        self,
        root: PathType = constant.DATA_DIR / "reside",
        shape: Ints = (3, 256, 256),
        transform: TransformType | None = None,
        target_transform: TransformType | None = None,
        transforms: TransformType | None = None,
        batch_size: int = 1,
        devices: Ints | Strs = 0,
        shuffle: bool = True,
        collate_fn: CallableType | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        super().__init__(
            root=root,
            shape=shape,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            batch_size=batch_size,
            devices=devices,
            shuffle=shuffle,
            collate_fn=collate_fn,
            verbose=verbose,
            *args, **kwargs
        )
    
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk, or that need
        to be done only from a single GPU in distributed settings:
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, phase: ModelPhaseType | None = None):
        """Use this method to do things on every device:
            - Count number of classes.
            - Build classlabels vocabulary.
            - Prepare train/val/test splits.
            - Apply transformations.
            - Define :attr:`collate_fn` for your custom dataset.

        Args:
            phase: The model phase. One of:
                - "training" : prepares :attr:'train' and :attr:'val'.
                - "testing"  : prepares :attr:'test'.
                - "inference": prepares :attr:`predict`.
                - None:      : prepares all.
                Default: None.
        """
        core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = constant.ModelPhase.from_value(
            phase
        ) if phase is not None else phase
        
        # Assign train/val datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TRAINING]:
            full_dataset = RESIDESOTSOutdoor(
                root=self.root,
                split="test",
                image_size=self.shape,
                transform=self.transform,
                target_transform=self.target_transform,
                transforms=self.transforms,
                verbose=self.verbose,
                **self.dataset_kwargs
            )
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train, self.val = random_split(
                full_dataset, [train_size, val_size]
            )
            self.classlabels = getattr(full_dataset, "classlabels", None)
            self.collate_fn = getattr(full_dataset, "collate_fn", None)
        
        # Assign test datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TESTING]:
            self.test = RESIDESOTSOutdoor(
                root=self.root,
                split="test",
                image_size=self.shape,
                transform=self.transform,
                target_transform=self.target_transform,
                transforms=self.transforms,
                verbose=self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn = getattr(self.test, "collate_fn", None)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


# endregion


# region Test

def test_reside_hsts():
    cfg = {
        "name"            : "reside-hsts",
        # A datamodule's name.
        "root"            : constant.DATA_DIR / "reside",
        # A root directory where the data is stored.
        "shape"           : [3, 256, 256],
        # The desired datapoint shape preferably in a channel-last format.
        # Default: (3, 256, 256).
        "transform"       : None,
        # Transformations performing on the input.
        "target_transform": None,
        # Transformations performing on the target.
        "transforms"      : [
            t.Resize(size=[3, 256, 256]),
        ],
        # Transformations performing on both the input and target.
        "cache_data"      : False,
        # If True, cache data to disk for faster loading next time.
        # Default: False.
        "cache_images"    : False,
        # If True, cache images into memory for faster training (WARNING:
        # large datasets may exceed system RAM). Default: False.
        "backend"         : constant.VISION_BACKEND,
        # The image processing backend. Default: VISION_BACKEND.
        "batch_size"      : 8,
        # The number of samples in one forward pass. Default: 1.
        "devices"         : 0,
        # A list of devices to use. Default: 0.
        "shuffle"         : True,
        # If True, reshuffle the datapoints at the beginning of every epoch.
        # Default: True.
        "verbose"         : True,
        # Verbosity. Default: True.
    }
    dm = RESIDEHSTSDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    result = {"image": input, "target": target}
    label = [(m["name"]) for m in meta]
    visualize.imshow_enhancement(
        winname="image",
        image=result,
        label=label
    )
    visualize.plt.show(block=True)


def test_reside_its():
    cfg = {
        "name"            : "reside-its",
        # A datamodule's name.
        "root"            : constant.DATA_DIR / "reside",
        # A root directory where the data is stored.
        "shape"           : [3, 256, 256],
        # The desired datapoint shape preferably in a channel-last format.
        # Default: (3, 256, 256).
        "transform"       : None,
        # Transformations performing on the input.
        "target_transform": None,
        # Transformations performing on the target.
        "transforms"      : [
            t.Resize(size=[3, 256, 256]),
        ],
        # Transformations performing on both the input and target.
        "cache_data"      : False,
        # If True, cache data to disk for faster loading next time.
        # Default: False.
        "cache_images"    : False,
        # If True, cache images into memory for faster training (WARNING:
        # large datasets may exceed system RAM). Default: False.
        "backend"         : constant.VISION_BACKEND,
        # The image processing backend. Default: VISION_BACKEND.
        "batch_size"      : 8,
        # The number of samples in one forward pass. Default: 1.
        "devices"         : 0,
        # A list of devices to use. Default: 0.
        "shuffle"         : True,
        # If True, reshuffle the datapoints at the beginning of every epoch.
        # Default: True.
        "verbose"         : True,
        # Verbosity. Default: True.
    }
    dm = RESIDEITSDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    result = {"image": input, "target": target}
    label = [(m["name"]) for m in meta]
    visualize.imshow_enhancement(
        winname="image",
        image=result,
        label=label
    )
    visualize.plt.show(block=True)


def test_reside_its_v2():
    cfg = {
        "name"            : "reside-its-v2",
        # A datamodule's name.
        "root"            : constant.DATA_DIR / "reside",
        # A root directory where the data is stored.
        "shape"           : [3, 256, 256],
        # The desired datapoint shape preferably in a channel-last format.
        # Default: (3, 256, 256).
        "transform"       : None,
        # Transformations performing on the input.
        "target_transform": None,
        # Transformations performing on the target.
        "transforms"      : [
            t.Resize(size=[3, 256, 256]),
        ],
        # Transformations performing on both the input and target.
        "cache_data"      : False,
        # If True, cache data to disk for faster loading next time.
        # Default: False.
        "cache_images"    : False,
        # If True, cache images into memory for faster training (WARNING:
        # large datasets may exceed system RAM). Default: False.
        "backend"         : constant.VISION_BACKEND,
        # The image processing backend. Default: VISION_BACKEND.
        "batch_size"      : 8,
        # The number of samples in one forward pass. Default: 1.
        "devices"         : 0,
        # A list of devices to use. Default: 0.
        "shuffle"         : True,
        # If True, reshuffle the datapoints at the beginning of every epoch.
        # Default: True.
        "verbose"         : True,
        # Verbosity. Default: True.
    }
    dm = RESIDEITSv2DataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    result = {"image": input, "target": target}
    label = [(m["name"]) for m in meta]
    visualize.imshow_enhancement(
        winname="image",
        image=result,
        label=label
    )
    visualize.plt.show(block=True)


def test_reside_ots():
    cfg = {
        "name"            : "reside-ots",
        # A datamodule's name.
        "root"            : constant.DATA_DIR / "reside",
        # A root directory where the data is stored.
        "shape"           : [3, 256, 256],
        # The desired datapoint shape preferably in a channel-last format.
        # Default: (3, 256, 256).
        "transform"       : None,
        # Transformations performing on the input.
        "target_transform": None,
        # Transformations performing on the target.
        "transforms"      : [
            t.Resize(size=[3, 256, 256]),
        ],
        # Transformations performing on both the input and target.
        "cache_data"      : False,
        # If True, cache data to disk for faster loading next time.
        # Default: False.
        "cache_images"    : False,
        # If True, cache images into memory for faster training (WARNING:
        # large datasets may exceed system RAM). Default: False.
        "backend"         : constant.VISION_BACKEND,
        # The image processing backend. Default: VISION_BACKEND.
        "batch_size"      : 8,
        # The number of samples in one forward pass. Default: 1.
        "devices"         : 0,
        # A list of devices to use. Default: 0.
        "shuffle"         : True,
        # If True, reshuffle the datapoints at the beginning of every epoch.
        # Default: True.
        "verbose"         : True,
        # Verbosity. Default: True.
    }
    dm = RESIDEOTSDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    result = {"image": input, "target": target}
    label = [(m["name"]) for m in meta]
    visualize.imshow_enhancement(
        winname="image",
        image=result,
        label=label
    )
    visualize.plt.show(block=True)


def test_reside_sots():
    cfg = {
        "name"            : "reside-sots",
        # A datamodule's name.
        "root"            : constant.DATA_DIR / "reside",
        # A root directory where the data is stored.
        "shape"           : [3, 256, 256],
        # The desired datapoint shape preferably in a channel-last format.
        # Default: (3, 256, 256).
        "transform"       : None,
        # Transformations performing on the input.
        "target_transform": None,
        # Transformations performing on the target.
        "transforms"      : [
            t.Resize(size=[3, 256, 256]),
        ],
        # Transformations performing on both the input and target.
        "cache_data"      : False,
        # If True, cache data to disk for faster loading next time.
        # Default: False.
        "cache_images"    : False,
        # If True, cache images into memory for faster training (WARNING:
        # large datasets may exceed system RAM). Default: False.
        "backend"         : constant.VISION_BACKEND,
        # The image processing backend. Default: VISION_BACKEND.
        "batch_size"      : 8,
        # The number of samples in one forward pass. Default: 1.
        "devices"         : 0,
        # A list of devices to use. Default: 0.
        "shuffle"         : True,
        # If True, reshuffle the datapoints at the beginning of every epoch.
        # Default: True.
        "verbose"         : True,
        # Verbosity. Default: True.
    }
    dm = RESIDESOTSDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    result = {"image": input, "target": target}
    label = [(m["name"]) for m in meta]
    visualize.imshow_enhancement(
        winname="image",
        image=result,
        label=label
    )
    visualize.plt.show(block=True)


def test_reside_sots_indoor():
    cfg = {
        "name"            : "reside-sots-indoor",
        # A datamodule's name.
        "root"            : constant.DATA_DIR / "reside",
        # A root directory where the data is stored.
        "shape"           : [3, 256, 256],
        # The desired datapoint shape preferably in a channel-last format.
        # Default: (3, 256, 256).
        "transform"       : None,
        # Transformations performing on the input.
        "target_transform": None,
        # Transformations performing on the target.
        "transforms"      : [
            t.Resize(size=[3, 256, 256]),
        ],
        # Transformations performing on both the input and target.
        "cache_data"      : False,
        # If True, cache data to disk for faster loading next time.
        # Default: False.
        "cache_images"    : False,
        # If True, cache images into memory for faster training (WARNING:
        # large datasets may exceed system RAM). Default: False.
        "backend"         : constant.VISION_BACKEND,
        # The image processing backend. Default: VISION_BACKEND.
        "batch_size"      : 8,
        # The number of samples in one forward pass. Default: 1.
        "devices"         : 0,
        # A list of devices to use. Default: 0.
        "shuffle"         : True,
        # If True, reshuffle the datapoints at the beginning of every epoch.
        # Default: True.
        "verbose"         : True,
        # Verbosity. Default: True.
    }
    dm = RESIDESOTSIndoorDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    result = {"image": input, "target": target}
    label = [(m["name"]) for m in meta]
    visualize.imshow_enhancement(
        winname="image",
        image=result,
        label=label
    )
    visualize.plt.show(block=True)


def test_reside_sots_outdoor():
    cfg = {
        "name"            : "reside-sots-outdoor",
        # A datamodule's name.
        "root"            : constant.DATA_DIR / "reside",
        # A root directory where the data is stored.
        "shape"           : [3, 256, 256],
        # The desired datapoint shape preferably in a channel-last format.
        # Default: (3, 256, 256).
        "transform"       : None,
        # Transformations performing on the input.
        "target_transform": None,
        # Transformations performing on the target.
        "transforms"      : [
            t.Resize(size=[3, 256, 256]),
        ],
        # Transformations performing on both the input and target.
        "cache_data"      : False,
        # If True, cache data to disk for faster loading next time.
        # Default: False.
        "cache_images"    : False,
        # If True, cache images into memory for faster training (WARNING:
        # large datasets may exceed system RAM). Default: False.
        "backend"         : constant.VISION_BACKEND,
        # The image processing backend. Default: VISION_BACKEND.
        "batch_size"      : 8,
        # The number of samples in one forward pass. Default: 1.
        "devices"         : 0,
        # A list of devices to use. Default: 0.
        "shuffle"         : True,
        # If True, reshuffle the datapoints at the beginning of every epoch.
        # Default: True.
        "verbose"         : True,
        # Verbosity. Default: True.
    }
    dm = RESIDESOTSOutdoorDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    result = {"image": input, "target": target}
    label = [(m["name"]) for m in meta]
    visualize.imshow_enhancement(
        winname="image",
        image=result,
        label=label
    )
    visualize.plt.show(block=True)


# endregion


# region Main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="test-reside-hsts",
        help="The task to run"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.task == "test-reside-hsts":
        test_reside_hsts()
    elif args.task == "test-reside-its":
        test_reside_its()
    elif args.task == "test-reside-its-v2":
        test_reside_its_v2()
    elif args.task == "test-reside-ots":
        test_reside_ots()
    elif args.task == "test-reside-sots":
        test_reside_sots()
    elif args.task == "test-reside-sots-indoor":
        test_reside_sots_indoor()
    elif args.task == "test-reside-sots-outdoor":
        test_reside_sots_outdoor()

# endregion
