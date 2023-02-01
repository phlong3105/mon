#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements SICE datasets and datamodules."""

from __future__ import annotations

__all__ = [
    "SICE", "SICEDataModule", "SICEPart1", "SICEPart1_512", "SICEPart1_512_Low",
    "SICEPart2", "SICEPart2_512", "SICEPart2_512_Low", "SICEPart2_900",
    "SICEPart2_900_Low", "SICEPart2_Low", "SICEUDataModule",
]

import argparse

from torch.utils.data import random_split

from mon import core
from mon.vision import constant, visualize
from mon.vision.dataset import base
from mon.vision.transform import transform as t
from mon.vision.typing import (
    CallableType, ClassLabelsType, Ints, ModelPhaseType, PathType,
    Strs, TransformType, VisionBackendType,
)


# region Dataset

@constant.DATASET.register(name="sice")
class SICE(base.ImageEnhancementDataset):
    """Full SICE dataset consisting of Part 1 (360 sequences) and Part 2
    (229 sequences).
    
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
        name            : str                    = "sice",
        root            : PathType               = constant.DATA_DIR / "sice",
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
        parts = ["part1", "part2"]
        
        self.images: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            for part in parts:
                pattern = self.root / part / "low"
                for path in pbar.track(
                    list(pattern.rglob("*/*")),
                    description=f"Listing {self.__class__.__name__} images"
                ):
                    if path.is_image_file():
                        self.images.append(
                            base.ImageLabel(path=path, backend=self.backend)
                        )
    
    def list_labels(self):
        """List label files."""
        self.labels: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} labels"
            ):
                stem = str(img.path.parent.stem)
                if "part1" in str(img.path):
                    path = self.root / "part1" / "high" / f"{stem}.jpg"
                elif "part2" in str(img.path):
                    path = self.root / "part2" / "high" / f"{stem}.jpg"
                else:
                    path = ""
                self.labels.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )


@constant.DATASET.register(name="sice-part1")
class SICEPart1(base.ImageEnhancementDataset):
    """SICE Part 1 dataset consists of 360 multi-exposure sequences.
    
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
        name            : str                    = "sice-part1",
        root            : PathType               = constant.DATA_DIR / "sice",
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
        self.images: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            pattern = self.root / "part1" / "low"
            for path in pbar.track(
                list(pattern.rglob("*/*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    self.images.append(
                        base.ImageLabel(path=path, backend=self.backend)
                    )
    
    def list_labels(self):
        """List label files."""
        self.labels: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} labels"
            ):
                stem = str(img.path.parent.stem)
                path = self.root / "part1" / "high" / f"{stem}.jpg"
                self.labels.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )


@constant.DATASET.register(name="sice-part1-512")
class SICEPart1_512(base.ImageEnhancementDataset):
    """SICE Part 1 dataset consists of 360 multi-exposure sequences. Images are
    resized to 3x512x512.
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 512, 512).
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
        name            : str                    = "sice-part1-512",
        root            : PathType               = constant.DATA_DIR / "sice",
        split           : str                    = "train",
        shape           : Ints                   = (3, 512, 512),
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
        self.images: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            pattern = self.root / "part1-512x512" / "low"
            for path in pbar.track(
                list(pattern.rglob("*/*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    self.images.append(
                        base.ImageLabel(path=path, backend=self.backend)
                    )
    
    def list_labels(self):
        """List label files."""
        self.labels: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} labels"
            ):
                stem = str(img.path.parent.stem)
                path = self.root / "part1-512x512" / "high" / f"{stem}.jpg"
                self.labels.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )


@constant.DATASET.register(name="sice-part1-512-low")
class SICEPart1_512_Low(base.UnlabeledImageDataset):
    """SICE Part 1 dataset consists of 360 multi-exposure sequences. Images are
    resized to 3x512x512. Only low-light images are included.
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 512, 512).
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
        name            : str                    = "sice-part1-512-low",
        root            : PathType               = constant.DATA_DIR / "sice",
        split           : str                    = "train",
        shape           : Ints                   = (3, 512, 512),
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
        self.images: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            pattern = self.root / "part1-512x512"
            for path in pbar.track(
                list(pattern.rglob("*/*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    self.images.append(
                        base.ImageLabel(path=path, backend=self.backend)
                    )


@constant.DATASET.register(name="sice-part2")
class SICEPart2(base.ImageEnhancementDataset):
    """SICE Part 2 dataset consists of 229 multi-exposure sequences.
    
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
        name            : str                    = "sice-part2",
        root            : PathType               = constant.DATA_DIR / "sice",
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
        self.images: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            pattern = self.root / "part2" / "low"
            for path in pbar.track(
                list(pattern.rglob("*/*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    self.images.append(base.ImageLabel(path=path, backend=self.backend))
    
    def list_labels(self):
        """List label files."""
        self.labels: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} labels"
            ):
                stem = str(img.path.parent.stem)
                path = self.root / "part2" / "high" / f"{stem}.jpg"
                self.labels.append(base.ImageLabel(path=path, backend=self.backend))


@constant.DATASET.register(name="sice-part2-low")
class SICEPart2_Low(base.ImageEnhancementDataset):
    """SICE Part 2 dataset consists of 229 multi-exposure sequences. Only
    low-light images are used. Specifically, we choose the first three (resp.
    four) low-light images if there are seven (resp. nine) images in a
    multi-exposure sequence.
    
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
        name            : str                    = "sice-part2-low",
        root            : PathType               = constant.DATA_DIR / "sice",
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
        self.images: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            pattern = self.root / "part2-low" / "low"
            for path in pbar.track(
                list(pattern.rglob("*/*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    self.images.append(
                        base.ImageLabel(path=path, backend=self.backend)
                    )
    
    def list_labels(self):
        """List label files."""
        self.labels: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} labels"
            ):
                stem = str(img.path.parent.stem)
                path = self.root / "part2-low" / "high" / f"{stem}.jpg"
                self.labels.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )
                

@constant.DATASET.register(name="sice-part2-512")
class SICEPart2_512(base.ImageEnhancementDataset):
    """SICE Part 2 dataset consists of 229 multi-exposure sequences. Images are
    resized to 3x512x512.
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 512, 512).
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
        name            : str                    = "sice-part2-512",
        root            : PathType               = constant.DATA_DIR / "sice",
        split           : str                    = "train",
        shape           : Ints                   = (3, 512, 512),
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
        self.images: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            pattern = self.root / "part2-512x512" / "low"
            for path in pbar.track(
                list(pattern.rglob("*/*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    self.images.append(
                        base.ImageLabel(path=path, backend=self.backend)
                    )
    
    def list_labels(self):
        """List label files."""
        self.labels: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} labels"
            ):
                stem = str(img.path.parent.stem)
                path = self.root / "part2-512x512" / "high" / f"{stem}.jpg"
                self.labels.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )


@constant.DATASET.register(name="sice-part2-512-low")
class SICEPart2_512_Low(base.ImageEnhancementDataset):
    """SICE Part 2 dataset consists of 229 multi-exposure sequences. Images are
    resized to 3x512x512. Only low-light images are used. Specifically, we
    choose the first three (resp. four) low-light images if there are seven
    (resp. nine) images in a multi-exposure sequence.
    
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
        name            : str                    = "sice-part2-512-low",
        root            : PathType               = constant.DATA_DIR / "sice",
        split           : str                    = "train",
        shape           : Ints                   = (3, 512, 512),
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
        self.images: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            pattern = self.root / "part2-512x512-low" / "low"
            for path in pbar.track(
                list(pattern.rglob("*/*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    self.images.append(
                        base.ImageLabel(path=path, backend=self.backend)
                    )
    
    def list_labels(self):
        """List label files."""
        self.labels: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} labels"
            ):
                stem = str(img.path.parent.stem)
                path = self.root / "part2-512x512-low" / "high" / f"{stem}.jpg"
                self.labels.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )


@constant.DATASET.register(name="sice-part2-900")
class SICEPart2_900(base.ImageEnhancementDataset):
    """SICE Part 2 dataset consists of 229 multi-exposure sequences. Images are
    resized to 3x900x1200.
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 900, 1200).
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
        name            : str                    = "sice-part2-900",
        root            : PathType               = constant.DATA_DIR / "sice",
        split           : str                    = "train",
        shape           : Ints                   = (3, 900, 1200),
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
        self.images: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            pattern = self.root / "part2-900x1200" / "low"
            for path in pbar.track(
                list(pattern.rglob("*/*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    self.images.append(
                        base.ImageLabel(path=path, backend=self.backend)
                    )
    
    def list_labels(self):
        """List label files."""
        self.labels: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} labels"
            ):
                stem = str(img.path.parent.stem)
                path = self.root / "part2-900x1200" / "high" / f"{stem}.jpg"
                self.labels.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )


@constant.DATASET.register(name="sice-part2-900-low")
class SICEPart2_900_Low(base.ImageEnhancementDataset):
    """SICE Part 2 dataset consists of 229 multi-exposure sequences. Images are
    resized to 3x900x1200. Only low-light images are used. Specifically, we
    choose the first three (resp. four) low-light images if there are seven
    (resp. nine) images in a multi-exposure sequence.
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 900, 1200).
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
        name            : str                    = "sice-part2-900-low",
        root            : PathType               = constant.DATA_DIR / "sice",
        split           : str                    = "train",
        shape           : Ints                   = (3, 900, 1200),
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
        self.images: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            pattern = self.root / "part2-900x1200-low" / "low"
            for path in pbar.track(
                list(pattern.rglob("*/*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    self.images.append(
                        base.ImageLabel(path=path, backend=self.backend)
                    )
    
    def list_labels(self):
        """List label files."""
        self.labels: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} labels"
            ):
                stem = str(img.path.parent.stem)
                path = self.root / "part2-900x1200-low" / "high" / f"{stem}.jpg"
                self.labels.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )

# endregion


# region Datamodule

@constant.DATAMODULE.register(name="sice")
class SICEDataModule(base.DataModule):
    """SICE datamodule.
    
    Args:
        name: A datamodule name.
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
        name            : str                  = "sice",
        root            : PathType             = constant.DATA_DIR / "sice",
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
            full_dataset = SICEPart1(
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
            self.test = SICEPart2(
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


@constant.DATAMODULE.register(name="sice-u")
@constant.DATAMODULE.register(name="sice-unsupervised")
class SICEUDataModule(base.DataModule):
    """SICE-Unsupervised datamodule.
    
    Args:
        name: A datamodule name.
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
        name            : str                  = "sice-u",
        root            : PathType             = constant.DATA_DIR / "sice",
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
            self.train = SICEPart1_512_Low(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.val = SICEPart2_900_Low(
                root             = self.root,
                split            = "train",
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
            self.test = SICEPart2_900_Low(
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

def test_sice():
    cfg = {
        "name": "sice",
            # A datamodule's name.
        "root": constant.DATA_DIR / "sice",
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
    dm  = SICEDataModule(**cfg)
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


def test_sice_unsupervised():
    cfg = {
        "name": "sice-unsupervised",
            # A datamodule's name.
        "root": constant.DATA_DIR / "sice",
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
    dm  = SICEUDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter      = iter(dm.train_dataloader)
    input, _, meta = next(data_iter)
    visualize.imshow(winname="image", image=input)
    visualize.plt.show(block=True)

# endregion

    
# region Main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="test-sice-unsupervised", help="The task to run.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.task == "test-sice":
        test_sice()
    elif args.task == "test-sice-unsupervised":
        test_sice_unsupervised()

# endregion
