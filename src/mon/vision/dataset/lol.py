#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements LOL datasets and datamodules."""

from __future__ import annotations

__all__ = [
    "DCIM", "DCIMDataModule", "LIME", "LIMEDataModule", "LOL", "LOL226",
    "LOL226DataModule", "LOL4K", "LOL4KDataModule", "LOLDataModule", "MEF",
    "MEFDataModule", "NPE", "NPEDataModule", "VIP", "VIPDataModule", "VV",
    "VVDataModule",
]

import argparse

from torch.utils.data import random_split

from mon import core
from mon.vision import constant, visualize
from mon.vision.dataset import base
from mon.vision.transform import transform as t
from mon.vision.typing import (
    CallableType, ClassLabelsType, Ints,
    ModelPhaseType, PathType, Strs, TransformType, VisionBackendType,
)


# region Dataset

@constant.DATASET.register(name="dcim")
class DCIM(base.UnlabeledImageDataset):
    """DCIM dataset consists of 64 low-light images.
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        classlabels: A :class:`ClassLabels` object. Defaults to None.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: The image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str                    = "dcim",
        root            : PathType               = constant.DATA_DIR / "lol",
        split           : str                    = "train",
        shape           : Ints                   = (3, 256, 256),
        classlabels     : ClassLabelsType | None = None,
        transform       : TransformType   | None = None,
        target_transform: TransformType   | None = None,
        transforms      : TransformType   | None = None,
        cache_data      : bool                   = False,
        cache_images    : bool                   = False,
        backend         : VisionBackendType      = constant.VISION_BACKEND,
        verbose         : bool                   = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
     
    def list_images(self):
        """List image files."""
        if self.split not in ["train"]:
            core.console.log(
                f"{self.__class__.__name__} dataset only supports "
                f":param:`split`: 'train'. Get: {self.split}."
            )
            
        self.images: list[base.ImageLabel] = []
        with core.rich.progress_bar() as pbar:
            pattern = self.root / "train" / "low" / "dcim"
            for path in pbar.track(
                list(pattern.rglob("*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    self.images.append(
                        base.ImageLabel(path=path, backend=self.backend)
                    )


@constant.DATASET.register(name="lime")
class LIME(base.UnlabeledImageDataset):
    """LIME dataset consists of 10 low-light images.
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        classlabels: A :class:`ClassLabels` object. Defaults to None.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: The image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str                    = "lime",
        root            : PathType               = constant.DATA_DIR / "lol",
        split           : str                    = "train",
        shape           : Ints                   = (3, 256, 256),
        classlabels     : ClassLabelsType | None = None,
        transform       : TransformType   | None = None,
        target_transform: TransformType   | None = None,
        transforms      : TransformType   | None = None,
        cache_data      : bool                   = False,
        cache_images    : bool                   = False,
        backend         : VisionBackendType      = constant.VISION_BACKEND,
        verbose         : bool                   = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
     
    def list_images(self):
        """List image files."""
        if self.split not in ["train"]:
            core.console.log(
                f"{self.__class__.__name__} dataset only supports "
                f":param:`split`: 'train'. Get: {self.split}."
            )
            
        self.images: list[base.ImageLabel] = []
        with core.rich.progress_bar() as pbar:
            pattern = self.root / "train" / "low" / "lime"
            for path in pbar.track(
                list(pattern.rglob("*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    self.images.append(
                        base.ImageLabel(path=path, backend=self.backend)
                    )


@constant.DATASET.register(name="lol")
class LOL(base.ImageEnhancementDataset):
    """LOL dataset consists of 500 low-light and normal-light image pairs. They
    are divided into 485 training pairs and 15 testing pairs. The low-light
    images contain noise produced during the photo capture process. Most of the
    images are indoor scenes. All the images have a resolution of 400×600.
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        classlabels: A :class:`ClassLabels` object. Defaults to None.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: The image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str                    = "lol",
        root            : PathType               = constant.DATA_DIR / "lol",
        split           : str                    = "train",
        shape           : Ints                   = (3, 256, 256),
        classlabels     : ClassLabelsType | None = None,
        transform       : TransformType   | None = None,
        target_transform: TransformType   | None = None,
        transforms      : TransformType   | None = None,
        cache_data      : bool                   = False,
        cache_images    : bool                   = False,
        backend         : VisionBackendType      = constant.VISION_BACKEND,
        verbose         : bool                   = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
     
    def list_images(self):
        """List image files."""
        if self.split not in ["train", "val"]:
            core.console.log(
                f"{self.__class__.__name__} dataset only supports "
                f":param:`split`: 'train', 'val', or 'test'. Get: {self.split}."
            )
            
        self.images: list[base.ImageLabel] = []
        with core.rich.progress_bar() as pbar:
            pattern = self.root / self.split / "low" / "lol"
            for path in pbar.track(
                list(pattern.rglob("*")),
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split} images"
            ):
                if path.is_image_file():
                    self.images.append(
                        base.ImageLabel(path=path, backend=self.backend)
                    )
    
    def list_labels(self):
        """List label files."""
        self.labels: list[base.ImageLabel] = []
        with core.rich.progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split} labels"
            ):
                path = core.Path(str(img.path).replace("low", "high"))
                self.labels.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )


@constant.DATASET.register(name="lol226")
class LOL226(base.UnlabeledImageDataset):
    """LOL226 dataset consists of 226 low-light images.
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        classlabels: A :class:`ClassLabels` object. Defaults to None.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: The image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str                    = "lol226",
        root            : PathType               = constant.DATA_DIR / "lol",
        split           : str                    = "train",
        shape           : Ints                   = (3, 256, 256),
        classlabels     : ClassLabelsType | None = None,
        transform       : TransformType   | None = None,
        target_transform: TransformType   | None = None,
        transforms      : TransformType   | None = None,
        cache_data      : bool                   = False,
        cache_images    : bool                   = False,
        backend         : VisionBackendType      = constant.VISION_BACKEND,
        verbose         : bool                   = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
     
    def list_images(self):
        """List image files."""
        if self.split not in ["train"]:
            core.console.log(
                f"{self.__class__.__name__} dataset only supports "
                f":param:`split`: 'train'. Get: {self.split}."
            )
            
        subdirs = ["dcim", "fusion", "lime", "mef", "npe", "vip", "vv"]
        
        self.images: list[base.ImageLabel] = []
        with core.rich.progress_bar() as pbar:
            for subdir in subdirs:
                pattern = self.root / "train" / "low" / subdir
                for path in pbar.track(
                    list(pattern.rglob("*")),
                    description=f"Listing {self.__class__.__name__} images"
                ):
                    if path.is_image_file():
                        self.images.append(
                            base.ImageLabel(path=path, backend=self.backend)
                        )


@constant.DATASET.register(name="lol4k")
class LOL4K(base.UnlabeledImageDataset):
    """LOL4k dataset consists of 3777 low-light images.
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        classlabels: A :class:`ClassLabels` object. Defaults to None.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: The image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str                    = "lol4k",
        root            : PathType               = constant.DATA_DIR / "lol",
        split           : str                    = "train",
        shape           : Ints                   = (3, 256, 256),
        classlabels     : ClassLabelsType | None = None,
        transform       : TransformType   | None = None,
        target_transform: TransformType   | None = None,
        transforms      : TransformType   | None = None,
        cache_data      : bool                   = False,
        cache_images    : bool                   = False,
        backend         : VisionBackendType      = constant.VISION_BACKEND,
        verbose         : bool                   = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
     
    def list_images(self):
        """List image files."""
        if self.split not in ["train"]:
            core.console.log(
                f"{self.__class__.__name__} dataset only supports "
                f":param:`split`: 'train'. Get: {self.split}."
            )
            
        self.images: list[base.ImageLabel] = []
        with core.rich.progress_bar() as pbar:
            pattern = self.root / "train" / "low"
            for path in pbar.track(
                list(pattern.rglob("*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    self.images.append(
                        base.ImageLabel(path=path, backend=self.backend)
                    )


@constant.DATASET.register(name="mef")
class MEF(base.UnlabeledImageDataset):
    """MEF dataset consists 17 low-light images.

    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        classlabels: A :class:`ClassLabels` object. Defaults to None.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: The image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str                    = "mef",
        root            : PathType               = constant.DATA_DIR / "lol",
        split           : str                    = "train",
        shape           : Ints                   = (3, 256, 256),
        classlabels     : ClassLabelsType | None = None,
        transform       : TransformType   | None = None,
        target_transform: TransformType   | None = None,
        transforms      : TransformType   | None = None,
        cache_data      : bool                   = False,
        cache_images    : bool                   = False,
        backend         : VisionBackendType      = constant.VISION_BACKEND,
        verbose         : bool                   = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
    
    def list_images(self):
        """List image files."""
        if self.split not in ["train"]:
            core.console.log(
                f"{self.__class__.__name__} dataset only supports "
                f":param:`split`: 'train'. Get: {self.split}."
            )
            
        self.images: list[base.ImageLabel] = []
        with core.rich.progress_bar() as pbar:
            pattern = self.root / "train" / "low" / "mef"
            for path in pbar.track(
                list(pattern.rglob("*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    self.images.append(
                        base.ImageLabel(path=path, backend=self.backend)
                    )


@constant.DATASET.register(name="npe")
class NPE(base.UnlabeledImageDataset):
    """NPE dataset consists 85 low-light images.
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        classlabels: A :class:`ClassLabels` object. Defaults to None.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: The image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str                    = "mef",
        root            : PathType               = constant.DATA_DIR / "lol",
        split           : str                    = "train",
        shape           : Ints                   = (3, 256, 256),
        classlabels     : ClassLabelsType | None = None,
        transform       : TransformType   | None = None,
        target_transform: TransformType   | None = None,
        transforms      : TransformType   | None = None,
        cache_data      : bool                   = False,
        cache_images    : bool                   = False,
        backend         : VisionBackendType      = constant.VISION_BACKEND,
        verbose         : bool                   = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
     
    def list_images(self):
        """List image files."""
        if self.split not in ["train"]:
            core.console.log(
                f"{self.__class__.__name__} dataset only supports "
                f":param:`split`: 'train'. Get: {self.split}."
            )
            
        self.images: list[base.ImageLabel] = []
        with core.rich.progress_bar() as pbar:
            pattern = self.root / "train" / "low" / "npe"
            for path in pbar.track(
                list(pattern.rglob("*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    self.images.append(
                        base.ImageLabel(path=path, backend=self.backend)
                    )


@constant.DATASET.register(name="vip")
class VIP(base.UnlabeledImageDataset):
    """VIP dataset consists of 8 low-light images.
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        classlabels: A :class:`ClassLabels` object. Defaults to None.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: The image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str                    = "vip",
        root            : PathType               = constant.DATA_DIR / "lol",
        split           : str                    = "train",
        shape           : Ints                   = (3, 256, 256),
        classlabels     : ClassLabelsType | None = None,
        transform       : TransformType   | None = None,
        target_transform: TransformType   | None = None,
        transforms      : TransformType   | None = None,
        cache_data      : bool                   = False,
        cache_images    : bool                   = False,
        backend         : VisionBackendType      = constant.VISION_BACKEND,
        verbose         : bool                   = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
     
    def list_images(self):
        """List image files."""
        if self.split not in ["train"]:
            core.console.log(
                f"{self.__class__.__name__} dataset only supports "
                f":param:`split`: 'train'. Get: {self.split}."
            )
            
        self.images: list[base.ImageLabel] = []
        with core.rich.progress_bar() as pbar:
            pattern = self.root / "train" / "low" / "vip"
            for path in pbar.track(
                list(pattern.rglob("*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    self.images.append(
                        base.ImageLabel(path=path, backend=self.backend)
                    )


@constant.DATASET.register(name="vv")
class VV(base.UnlabeledImageDataset):
    """VV dataset consists of 24 low-light images.
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        classlabels: A :class:`ClassLabels` object. Defaults to None.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: The image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str                    = "vv",
        root            : PathType               = constant.DATA_DIR / "lol",
        split           : str                    = "train",
        shape           : Ints                   = (3, 256, 256),
        classlabels     : ClassLabelsType | None = None,
        transform       : TransformType   | None = None,
        target_transform: TransformType   | None = None,
        transforms      : TransformType   | None = None,
        cache_data      : bool                   = False,
        cache_images    : bool                   = False,
        backend         : VisionBackendType      = constant.VISION_BACKEND,
        verbose         : bool                   = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
     
    def list_images(self):
        """List image files."""
        if self.split not in ["train"]:
            core.console.log(
                f"{self.__class__.__name__} dataset only supports "
                f":param:`split`: 'train'. Get: {self.split}."
            )
            
        self.images: list[base.ImageLabel] = []
        with core.rich.progress_bar() as pbar:
            pattern = self.root / "train" / "low" / "vv"
            for path in pbar.track(
                list(pattern.rglob("*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    self.images.append(
                        base.ImageLabel(path=path, backend=self.backend)
                    )

# endregion


# region Datamodule

@constant.DATAMODULE.register(name="dcim")
class DCIMDataModule(base.DataModule):
    """DCIM datamodule.
    
    Args:
        name: A datamodule's name.
        root: A root directory where the data is stored.
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        batch_size: The number of samples in one forward pass. Defaults to 1.
        devices: A list of devices to use. Defaults to 0.
        shuffle: If True, reshuffle the datapoints at the beginning of every
            epoch. Defaults to True.
        collate_fn: The function used to fused datapoint together when using
            :param:`batch_size` > 1.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str                  = "dcim",
        root            : PathType             = constant.DATA_DIR / "lol",
        shape           : Ints                 = (3, 256, 256),
        transform       : TransformType | None = None,
        target_transform: TransformType | None = None,
        transforms      : TransformType | None = None,
        batch_size      : int                  = 1,
        devices         : Ints | Strs          = 0,
        shuffle         : bool                 = True,
        collate_fn      : CallableType  | None = None,
        verbose         : bool                 = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            shape            = shape,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            batch_size       = batch_size,
            devices          = devices,
            shuffle          = shuffle,
            collate_fn       = collate_fn,
            verbose          = verbose,
            *args, **kwargs
        )
        
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk, or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.load_classlabels()
    
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
                Defaults to None.
        """
        core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = constant.ModelPhase.from_value(phase) if phase is not None else phase
        
        # Assign train/val datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TRAINING]:
            full_dataset = DCIM(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            train_size   = int(0.8 * len(full_dataset))
            val_size     = len(full_dataset) - train_size
            self.train, self.val = random_split(
                full_dataset, [train_size, val_size]
            )
            self.classlabels = getattr(full_dataset, "classlabels", None)
            self.collate_fn  = getattr(full_dataset, "collate_fn",  None)
            
        # Assign test datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TESTING]:
            self.test = DCIM(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.load_classlabels()

        self.summarize()
        
    def load_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@constant.DATAMODULE.register(name="lime")
class LIMEDataModule(base.DataModule):
    """LIME datamodule.
     
     Args:
        name: A datamodule's name.
        root: A root directory where the data is stored.
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        batch_size: The number of samples in one forward pass. Defaults to 1.
        devices: A list of devices to use. Defaults to 0.
        shuffle: If True, reshuffle the datapoints at the beginning of every
            epoch. Defaults to True.
        collate_fn: The function used to fused datapoint together when using
            :param:`batch_size` > 1.
        verbose: Verbosity. Defaults to True.
     """
    
    def __init__(
        self,
        name            : str                  = "lime",
        root            : PathType             = constant.DATA_DIR / "lol",
        shape           : Ints                 = (3, 256, 256),
        transform       : TransformType | None = None,
        target_transform: TransformType | None = None,
        transforms      : TransformType | None = None,
        batch_size      : int                  = 1,
        devices         : Ints | Strs          = 0,
        shuffle         : bool                 = True,
        collate_fn      : CallableType  | None = None,
        verbose         : bool                 = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            shape            = shape,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            batch_size       = batch_size,
            devices          = devices,
            shuffle          = shuffle,
            collate_fn       = collate_fn,
            verbose          = verbose,
            *args, **kwargs
        )
        
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk, or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.load_classlabels()
    
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
                Defaults to None.
        """
        core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = constant.ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TRAINING]:
            full_dataset = LIME(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            train_size   = int(0.8 * len(full_dataset))
            val_size     = len(full_dataset) - train_size
            self.train, self.val = random_split(
                full_dataset, [train_size, val_size]
            )
            self.classlabels = getattr(full_dataset, "classlabels", None)
            self.collate_fn  = getattr(full_dataset, "collate_fn",  None)
            
        # Assign test datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TESTING]:
            self.test = LIME(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.load_classlabels()

        self.summarize()
        
    def load_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@constant.DATAMODULE.register(name="lol")
class LOLDataModule(base.DataModule):
    """LOL datamodule.
    
    Args:
        name: A datamodule's name.
        root: A root directory where the data is stored.
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        batch_size: The number of samples in one forward pass. Defaults to 1.
        devices: A list of devices to use. Defaults to 0.
        shuffle: If True, reshuffle the datapoints at the beginning of every
            epoch. Defaults to True.
        collate_fn: The function used to fused datapoint together when using
            :param:`batch_size` > 1.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str                  = "lol",
        root            : PathType             = constant.DATA_DIR / "lol",
        shape           : Ints                 = (3, 256, 256),
        transform       : TransformType | None = None,
        target_transform: TransformType | None = None,
        transforms      : TransformType | None = None,
        batch_size      : int                  = 1,
        devices         : Ints | Strs          = 0,
        shuffle         : bool                 = True,
        collate_fn      : CallableType  | None = None,
        verbose         : bool                 = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            shape            = shape,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            batch_size       = batch_size,
            devices          = devices,
            shuffle          = shuffle,
            collate_fn       = collate_fn,
            verbose          = verbose,
            *args, **kwargs
        )
        
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk, or that need
        to be done only from a single GPU in distributed settings:
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.load_classlabels()
    
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
                Defaults to None.
        """
        core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = constant.ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TRAINING]:
            self.train = LOL(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.val = LOL(
                root             = self.root,
                split            = "val",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.train, "classlabels", None)
            self.collate_fn  = getattr(self.train, "collate_fn",  None)
            
        # Assign test datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TESTING]:
            self.test = LOL(
                root             = self.root,
                split            = "val",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.load_classlabels()

        self.summarize()
        
    def load_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@constant.DATAMODULE.register(name="lol226")
class LOL226DataModule(base.DataModule):
    """LOL226 datamodule.
    
    Args:
        name: A datamodule's name.
        root: A root directory where the data is stored.
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        batch_size: The number of samples in one forward pass. Defaults to 1.
        devices: A list of devices to use. Defaults to 0.
        shuffle: If True, reshuffle the datapoints at the beginning of every
            epoch. Defaults to True.
        collate_fn: The function used to fused datapoint together when using
            :param:`batch_size` > 1.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str                  = "lol226",
        root            : PathType             = constant.DATA_DIR / "lol",
        shape           : Ints                 = (3, 256, 256),
        transform       : TransformType | None = None,
        target_transform: TransformType | None = None,
        transforms      : TransformType | None = None,
        batch_size      : int                  = 1,
        devices         : Ints | Strs          = 0,
        shuffle         : bool                 = True,
        collate_fn      : CallableType  | None = None,
        verbose         : bool                 = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            shape            = shape,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            batch_size       = batch_size,
            devices          = devices,
            shuffle          = shuffle,
            collate_fn       = collate_fn,
            verbose          = verbose,
            *args, **kwargs
        )
        
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk, or that need
        to be done only from a single GPU in distributed settings:
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.load_classlabels()
    
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
                Defaults to None.
        """
        core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = constant.ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TRAINING]:
            full_dataset = LOL226(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            train_size   = int(0.8 * len(full_dataset))
            val_size     = len(full_dataset) - train_size
            self.train, self.val = random_split(
                full_dataset, [train_size, val_size]
            )
            self.classlabels = getattr(full_dataset, "classlabels", None)
            self.collate_fn  = getattr(full_dataset, "collate_fn",  None)
            
        # Assign test datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TESTING]:
            self.test = LOL226(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.load_classlabels()

        self.summarize()
        
    def load_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@constant.DATAMODULE.register(name="lol4k")
class LOL4KDataModule(base.DataModule):
    """LOL4K datamodule.
    
    Args:
        name: A datamodule's name.
        root: A root directory where the data is stored.
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        batch_size: The number of samples in one forward pass. Defaults to 1.
        devices: A list of devices to use. Defaults to 0.
        shuffle: If True, reshuffle the datapoints at the beginning of every
            epoch. Defaults to True.
        collate_fn: The function used to fused datapoint together when using
            :param:`batch_size` > 1.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str                  = "lol4k",
        root            : PathType             = constant.DATA_DIR / "lol",
        shape           : Ints                 = (3, 256, 256),
        transform       : TransformType | None = None,
        target_transform: TransformType | None = None,
        transforms      : TransformType | None = None,
        batch_size      : int                  = 1,
        devices         : Ints | Strs          = 0,
        shuffle         : bool                 = True,
        collate_fn      : CallableType  | None = None,
        verbose         : bool                 = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            shape            = shape,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            batch_size       = batch_size,
            devices          = devices,
            shuffle          = shuffle,
            collate_fn       = collate_fn,
            verbose          = verbose,
            *args, **kwargs
        )
        
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk, or that need
        to be done only from a single GPU in distributed settings:
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.load_classlabels()
    
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
                Defaults to None.
        """
        core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = constant.ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TRAINING]:
            self.train = LOL4K(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.val = LOL(
                root             = self.root,
                split            = "val",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.val, "classlabels", None)
            self.collate_fn  = getattr(self.val, "collate_fn",  None)
            
        # Assign test datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TESTING]:
            self.test = LOL(
                root             = self.root,
                split            = "val",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.load_classlabels()

        self.summarize()
        
    def load_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass
    

@constant.DATAMODULE.register(name="mef")
class MEFDataModule(base.DataModule):
    """MEF datamodule.
    
    Args:
        name: A datamodule's name.
        root: A root directory where the data is stored.
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        batch_size: The number of samples in one forward pass. Defaults to 1.
        devices: A list of devices to use. Defaults to 0.
        shuffle: If True, reshuffle the datapoints at the beginning of every
            epoch. Defaults to True.
        collate_fn: The function used to fused datapoint together when using
            :param:`batch_size` > 1.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str                  = "mef",
        root            : PathType             = constant.DATA_DIR / "lol",
        shape           : Ints                 = (3, 256, 256),
        transform       : TransformType | None = None,
        target_transform: TransformType | None = None,
        transforms      : TransformType | None = None,
        batch_size      : int                  = 1,
        devices         : Ints | Strs          = 0,
        shuffle         : bool                 = True,
        collate_fn      : CallableType  | None = None,
        verbose         : bool                 = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            shape            = shape,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            batch_size       = batch_size,
            devices          = devices,
            shuffle          = shuffle,
            collate_fn       = collate_fn,
            verbose          = verbose,
            *args, **kwargs
        )
        
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk, or that need
        to be done only from a single GPU in distributed settings:
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.load_classlabels()
    
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
                Defaults to None.
        """
        core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = constant.ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TRAINING]:
            full_dataset = MEF(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            train_size   = int(0.8 * len(full_dataset))
            val_size     = len(full_dataset) - train_size
            self.train, self.val = random_split(
                full_dataset, [train_size, val_size]
            )
            self.classlabels = getattr(full_dataset, "classlabels", None)
            self.collate_fn  = getattr(full_dataset, "collate_fn",  None)
            
        # Assign test datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TESTING]:
            self.test = MEF(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.load_classlabels()

        self.summarize()
        
    def load_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@constant.DATAMODULE.register(name="npe")
class NPEDataModule(base.DataModule):
    """
    NPE datamodule.
    """
    
    def __init__(
        self,
        name            : str                  = "mef",
        root            : PathType             = constant.DATA_DIR / "lol",
        shape           : Ints                 = (3, 256, 256),
        transform       : TransformType | None = None,
        target_transform: TransformType | None = None,
        transforms      : TransformType | None = None,
        batch_size      : int                  = 1,
        devices         : Ints | Strs          = 0,
        shuffle         : bool                 = True,
        collate_fn      : CallableType  | None = None,
        verbose         : bool                 = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            shape            = shape,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            batch_size       = batch_size,
            devices          = devices,
            shuffle          = shuffle,
            collate_fn       = collate_fn,
            verbose          = verbose,
            *args, **kwargs
        )
        
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk, or that need
        to be done only from a single GPU in distributed settings:
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.load_classlabels()
    
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
                Defaults to None.
        """
        core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = constant.ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TRAINING]:
            full_dataset = NPE(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            train_size   = int(0.8 * len(full_dataset))
            val_size     = len(full_dataset) - train_size
            self.train, self.val = random_split(
                full_dataset, [train_size, val_size]
            )
            self.classlabels = getattr(full_dataset, "classlabels", None)
            self.collate_fn  = getattr(full_dataset, "collate_fn",  None)
            
        # Assign test datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TESTING]:
            self.test = NPE(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.load_classlabels()

        self.summarize()
        
    def load_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@constant.DATAMODULE.register(name="vip")
class VIPDataModule(base.DataModule):
    """VIP datamodule.
    
    Args:
        name: A datamodule's name.
        root: A root directory where the data is stored.
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        batch_size: The number of samples in one forward pass. Defaults to 1.
        devices: A list of devices to use. Defaults to 0.
        shuffle: If True, reshuffle the datapoints at the beginning of every
            epoch. Defaults to True.
        collate_fn: The function used to fused datapoint together when using
            :param:`batch_size` > 1.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str                  = "vip",
        root            : PathType             = constant.DATA_DIR / "lol",
        shape           : Ints                 = (3, 256, 256),
        transform       : TransformType | None = None,
        target_transform: TransformType | None = None,
        transforms      : TransformType | None = None,
        batch_size      : int                  = 1,
        devices         : Ints | Strs          = 0,
        shuffle         : bool                 = True,
        collate_fn      : CallableType  | None = None,
        verbose         : bool                 = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            shape            = shape,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            batch_size       = batch_size,
            devices          = devices,
            shuffle          = shuffle,
            collate_fn       = collate_fn,
            verbose          = verbose,
            *args, **kwargs
        )
        
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk, or that need
        to be done only from a single GPU in distributed settings:
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.load_classlabels()
    
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
                Defaults to None.
        """
        core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = constant.ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TRAINING]:
            full_dataset = VIP(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            train_size   = int(0.8 * len(full_dataset))
            val_size     = len(full_dataset) - train_size
            self.train, self.val = random_split(
                full_dataset, [train_size, val_size]
            )
            self.classlabels = getattr(full_dataset, "classlabels", None)
            self.collate_fn  = getattr(full_dataset, "collate_fn",  None)
            
        # Assign test datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TESTING]:
            self.test = VIP(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.load_classlabels()

        self.summarize()
        
    def load_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@constant.DATAMODULE.register(name="vv")
class VVDataModule(base.DataModule):
    """VV datamodule.
    
    Args:
        name: A datamodule's name.
        root: A root directory where the data is stored.
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        batch_size: The number of samples in one forward pass. Defaults to 1.
        devices: A list of devices to use. Defaults to 0.
        shuffle: If True, reshuffle the datapoints at the beginning of every
            epoch. Defaults to True.
        collate_fn: The function used to fused datapoint together when using
            :param:`batch_size` > 1.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str                  = "vv",
        root            : PathType             = constant.DATA_DIR / "lol",
        shape           : Ints                 = (3, 256, 256),
        transform       : TransformType | None = None,
        target_transform: TransformType | None = None,
        transforms      : TransformType | None = None,
        batch_size      : int                  = 1,
        devices         : Ints | Strs          = 0,
        shuffle         : bool                 = True,
        collate_fn      : CallableType  | None = None,
        verbose         : bool                 = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            shape            = shape,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            batch_size       = batch_size,
            devices          = devices,
            shuffle          = shuffle,
            collate_fn       = collate_fn,
            verbose          = verbose,
            *args, **kwargs
        )
        
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk, or that need
        to be done only from a single GPU in distributed settings:
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.load_classlabels()
    
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
                Defaults to None.
        """
        core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = constant.ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TRAINING]:
            full_dataset = VV(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            train_size   = int(0.8 * len(full_dataset))
            val_size     = len(full_dataset) - train_size
            self.train, self.val = random_split(
                full_dataset, [train_size, val_size]
            )
            self.classlabels = getattr(full_dataset, "classlabels", None)
            self.collate_fn  = getattr(full_dataset, "collate_fn",  None)
            
        # Assign test datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TESTING]:
            self.test = VV(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.load_classlabels()

        self.summarize()
        
    def load_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass

# endregion


# region Test

def test_dcim():
    cfg = {
        "name": "dcim",
            # A datamodule's name.
        "root": constant.DATA_DIR / "lol",
            # A root directory where the data is stored.
        "shape": [3, 256, 256],
            # The desired datapoint shape preferably in a channel-last format.
            # Defaults to (3, 256, 256).
        "transform": None,
            # Transformations performing on the input.
        "target_transform": None,
            # Transformations performing on the target.
        "transforms": [
            t.Resize(size=[3, 256, 256]),
        ],
            # Transformations performing on both the input and target.
        "cache_data": False,
            # If True, cache data to disk for faster loading next time.
            # Defaults to False.
        "cache_images": False,
            # If True, cache images into memory for faster training (WARNING:
            # large datasets may exceed system RAM). Defaults to False.
        "backend": constant.VISION_BACKEND,
            # The image processing backend. Defaults to VISION_BACKEND.
        "batch_size": 8,
            # The number of samples in one forward pass. Defaults to 1.
        "devices" : 0,
            # A list of devices to use. Defaults to 0.
        "shuffle": True,
            # If True, reshuffle the datapoints at the beginning of every epoch.
            # Defaults to True.
        "verbose": True,
            # Verbosity. Defaults to True.
    }
    dm  = DCIMDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter      = iter(dm.test_dataloader)
    input, _, meta = next(data_iter)
    visualize.imshow(winname="image", image=input)
    visualize.plt.show(block=True)


def test_lime():
    cfg = {
        "name": "lime",
            # A datamodule's name.
        "root": constant.DATA_DIR / "lol",
            # A root directory where the data is stored.
        "shape": [3, 256, 256],
            # The desired datapoint shape preferably in a channel-last format.
            # Defaults to (3, 256, 256).
        "transform": None,
            # Transformations performing on the input.
        "target_transform": None,
            # Transformations performing on the target.
        "transforms": [
            t.Resize(size=[3, 256, 256]),
        ],
            # Transformations performing on both the input and target.
        "cache_data": False,
            # If True, cache data to disk for faster loading next time.
            # Defaults to False.
        "cache_images": False,
            # If True, cache images into memory for faster training (WARNING:
            # large datasets may exceed system RAM). Defaults to False.
        "backend": constant.VISION_BACKEND,
            # The image processing backend. Defaults to VISION_BACKEND.
        "batch_size": 8,
            # The number of samples in one forward pass. Defaults to 1.
        "devices" : 0,
            # A list of devices to use. Defaults to 0.
        "shuffle": True,
            # If True, reshuffle the datapoints at the beginning of every epoch.
            # Defaults to True.
        "verbose": True,
            # Verbosity. Defaults to True.
    }
    dm  = LIMEDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter      = iter(dm.test_dataloader)
    input, _, meta = next(data_iter)
    visualize.imshow(winname="image", image=input)
    visualize.plt.show(block=True)


def test_lol():
    cfg = {
        "name": "lol",
            # A datamodule's name.
        "root": constant.DATA_DIR / "lol",
            # A root directory where the data is stored.
        "shape": [3, 256, 256],
            # The desired datapoint shape preferably in a channel-last format.
            # Defaults to (3, 256, 256).
        "transform": None,
            # Transformations performing on the input.
        "target_transform": None,
            # Transformations performing on the target.
        "transforms": [
            t.Resize(size=[3, 256, 256]),
        ],
            # Transformations performing on both the input and target.
        "cache_data": False,
            # If True, cache data to disk for faster loading next time.
            # Defaults to False.
        "cache_images": False,
            # If True, cache images into memory for faster training (WARNING:
            # large datasets may exceed system RAM). Defaults to False.
        "backend": constant.VISION_BACKEND,
            # The image processing backend. Defaults to VISION_BACKEND.
        "batch_size": 8,
            # The number of samples in one forward pass. Defaults to 1.
        "devices" : 0,
            # A list of devices to use. Defaults to 0.
        "shuffle": True,
            # If True, reshuffle the datapoints at the beginning of every epoch.
            # Defaults to True.
        "verbose": True,
            # Verbosity. Defaults to True.
    }
    dm  = LOLDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter           = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    result              = {"image" : input, "target": target}
    label               = [(m["name"]) for m in meta]
    visualize.imshow_enhancement(
        winname = "image",
        image   = result,
        label   = label
    )
    visualize.plt.show(block=True)


def test_lol226():
    cfg = {
        "name": "lol226",
            # A datamodule's name.
        "root": constant.DATA_DIR / "lol",
            # A root directory where the data is stored.
        "shape": [3, 256, 256],
            # The desired datapoint shape preferably in a channel-last format.
            # Defaults to (3, 256, 256).
        "transform": None,
            # Transformations performing on the input.
        "target_transform": None,
            # Transformations performing on the target.
        "transforms": [
            t.Resize(size=[3, 256, 256]),
        ],
            # Transformations performing on both the input and target.
        "cache_data": False,
            # If True, cache data to disk for faster loading next time.
            # Defaults to False.
        "cache_images": False,
            # If True, cache images into memory for faster training (WARNING:
            # large datasets may exceed system RAM). Defaults to False.
        "backend": constant.VISION_BACKEND,
            # The image processing backend. Defaults to VISION_BACKEND.
        "batch_size": 8,
            # The number of samples in one forward pass. Defaults to 1.
        "devices" : 0,
            # A list of devices to use. Defaults to 0.
        "shuffle": True,
            # If True, reshuffle the datapoints at the beginning of every epoch.
            # Defaults to True.
        "verbose": True,
            # Verbosity. Defaults to True.
    }
    dm  = LOL226DataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter      = iter(dm.test_dataloader)
    input, _, meta = next(data_iter)
    visualize.imshow(winname="image", image=input)
    visualize.plt.show(block=True)


def test_lol4k():
    cfg = {
        "name": "lol4k",
            # A datamodule's name.
        "root": constant.DATA_DIR / "lol",
            # A root directory where the data is stored.
        "shape": [3, 256, 256],
            # The desired datapoint shape preferably in a channel-last format.
            # Defaults to (3, 256, 256).
        "transform": None,
            # Transformations performing on the input.
        "target_transform": None,
            # Transformations performing on the target.
        "transforms": [
            t.Resize(size=[3, 256, 256]),
        ],
            # Transformations performing on both the input and target.
        "cache_data": False,
            # If True, cache data to disk for faster loading next time.
            # Defaults to False.
        "cache_images": False,
            # If True, cache images into memory for faster training (WARNING:
            # large datasets may exceed system RAM). Defaults to False.
        "backend": constant.VISION_BACKEND,
            # The image processing backend. Defaults to VISION_BACKEND.
        "batch_size": 8,
            # The number of samples in one forward pass. Defaults to 1.
        "devices" : 0,
            # A list of devices to use. Defaults to 0.
        "shuffle": True,
            # If True, reshuffle the datapoints at the beginning of every epoch.
            # Defaults to True.
        "verbose": True,
            # Verbosity. Defaults to True.
    }
    dm  = LOL4KDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter      = iter(dm.test_dataloader)
    input, _, meta = next(data_iter)
    visualize.imshow(winname="image", image=input)
    visualize.plt.show(block=True)


def test_mef():
    cfg = {
        "name": "mef",
            # A datamodule's name.
        "root": constant.DATA_DIR / "lol",
            # A root directory where the data is stored.
        "shape": [3, 256, 256],
            # The desired datapoint shape preferably in a channel-last format.
            # Defaults to (3, 256, 256).
        "transform": None,
            # Transformations performing on the input.
        "target_transform": None,
            # Transformations performing on the target.
        "transforms": [
            t.Resize(size=[3, 256, 256]),
        ],
            # Transformations performing on both the input and target.
        "cache_data": False,
            # If True, cache data to disk for faster loading next time.
            # Defaults to False.
        "cache_images": False,
            # If True, cache images into memory for faster training (WARNING:
            # large datasets may exceed system RAM). Defaults to False.
        "backend": constant.VISION_BACKEND,
            # The image processing backend. Defaults to VISION_BACKEND.
        "batch_size": 8,
            # The number of samples in one forward pass. Defaults to 1.
        "devices" : 0,
            # A list of devices to use. Defaults to 0.
        "shuffle": True,
            # If True, reshuffle the datapoints at the beginning of every epoch.
            # Defaults to True.
        "verbose": True,
            # Verbosity. Defaults to True.
    }
    dm  = MEFDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter      = iter(dm.test_dataloader)
    input, _, meta = next(data_iter)
    visualize.imshow(winname="image", image=input)
    visualize.plt.show(block=True)


def test_npe():
    cfg = {
        "name": "mef",
            # A datamodule's name.
        "root": constant.DATA_DIR / "lol",
            # A root directory where the data is stored.
        "shape": [3, 256, 256],
            # The desired datapoint shape preferably in a channel-last format.
            # Defaults to (3, 256, 256).
        "transform": None,
            # Transformations performing on the input.
        "target_transform": None,
            # Transformations performing on the target.
        "transforms": [
            t.Resize(size=[3, 256, 256]),
        ],
            # Transformations performing on both the input and target.
        "cache_data": False,
            # If True, cache data to disk for faster loading next time.
            # Defaults to False.
        "cache_images": False,
            # If True, cache images into memory for faster training (WARNING:
            # large datasets may exceed system RAM). Defaults to False.
        "backend": constant.VISION_BACKEND,
            # The image processing backend. Defaults to VISION_BACKEND.
        "batch_size": 8,
            # The number of samples in one forward pass. Defaults to 1.
        "devices" : 0,
            # A list of devices to use. Defaults to 0.
        "shuffle": True,
            # If True, reshuffle the datapoints at the beginning of every epoch.
            # Defaults to True.
        "verbose": True,
            # Verbosity. Defaults to True.
    }
    dm  = NPEDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter      = iter(dm.test_dataloader)
    input, _, meta = next(data_iter)
    visualize.imshow(winname="image", image=input)
    visualize.plt.show(block=True)


def test_vip():
    cfg = {
        "name": "vip",
            # A datamodule's name.
        "root": constant.DATA_DIR / "lol",
            # A root directory where the data is stored.
        "shape": [3, 256, 256],
            # The desired datapoint shape preferably in a channel-last format.
            # Defaults to (3, 256, 256).
        "transform": None,
            # Transformations performing on the input.
        "target_transform": None,
            # Transformations performing on the target.
        "transforms": [
            t.Resize(size=[3, 256, 256]),
        ],
            # Transformations performing on both the input and target.
        "cache_data": False,
            # If True, cache data to disk for faster loading next time.
            # Defaults to False.
        "cache_images": False,
            # If True, cache images into memory for faster training (WARNING:
            # large datasets may exceed system RAM). Defaults to False.
        "backend": constant.VISION_BACKEND,
            # The image processing backend. Defaults to VISION_BACKEND.
        "batch_size": 8,
            # The number of samples in one forward pass. Defaults to 1.
        "devices" : 0,
            # A list of devices to use. Defaults to 0.
        "shuffle": True,
            # If True, reshuffle the datapoints at the beginning of every epoch.
            # Defaults to True.
        "verbose": True,
            # Verbosity. Defaults to True.
    }
    dm  = VIPDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter      = iter(dm.test_dataloader)
    input, _, meta = next(data_iter)
    visualize.imshow(winname="image", image=input)
    visualize.plt.show(block=True)


def test_vv():
    cfg = {
        "name": "vv",
            # A datamodule's name.
        "root": constant.DATA_DIR / "lol",
            # A root directory where the data is stored.
        "shape": [3, 256, 256],
            # The desired datapoint shape preferably in a channel-last format.
            # Defaults to (3, 256, 256).
        "transform": None,
            # Transformations performing on the input.
        "target_transform": None,
            # Transformations performing on the target.
        "transforms": [
            t.Resize(size=[3, 256, 256]),
        ],
            # Transformations performing on both the input and target.
        "cache_data": False,
            # If True, cache data to disk for faster loading next time.
            # Defaults to False.
        "cache_images": False,
            # If True, cache images into memory for faster training (WARNING:
            # large datasets may exceed system RAM). Defaults to False.
        "backend": constant.VISION_BACKEND,
            # The image processing backend. Defaults to VISION_BACKEND.
        "batch_size": 8,
            # The number of samples in one forward pass. Defaults to 1.
        "devices" : 0,
            # A list of devices to use. Defaults to 0.
        "shuffle": True,
            # If True, reshuffle the datapoints at the beginning of every epoch.
            # Defaults to True.
        "verbose": True,
            # Verbosity. Defaults to True.
    }
    dm  = VVDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter      = iter(dm.test_dataloader)
    input, _, meta = next(data_iter)
    visualize.imshow(winname="image", image=input)
    visualize.plt.show(block=True)

# endregion


# region Main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str , default="test-lol226", help="The task to run")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.task == "test-dcim":
        test_dcim()
    elif args.task == "test-lime":
        test_lime()
    elif args.task == "test-lol":
        test_lol()
    elif args.task == "test-lol226":
        test_lol226()
    elif args.task == "test-lol4k":
        test_lol4k()
    elif args.task == "test-mef":
        test_mef()
    elif args.task == "test-npe":
        test_npe()
    elif args.task == "test-vip":
        test_vip()
    elif args.task == "test-vv":
        test_vv()

# endregion
