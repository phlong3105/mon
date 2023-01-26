#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Cityscape datasets and datamodules."""

from __future__ import annotations

__all__ = [
    "CityscapesFog", "CityscapesFogDataModule", "CityscapesLOL",
    "CityscapesLOLDataModule", "CityscapesRain", "CityscapesRainDataModule",
    "CityscapesSemantic", "CityscapesSnow", "CityscapesSnowDataModule",
    "cityscapes_classlabels",
]

import argparse
import glob

from torch.utils.data import random_split

from mon import core
from mon.vision import constant, visualize
from mon.vision.dataset import base
from mon.vision.transform import transform as t
from mon.vision.typing import (
    CallableType, ClassLabelsType, Ints,
    ModelPhaseType, PathType, Strs, TransformType, VisionBackendType,
)

# region ClassLabels

cityscapes_classlabels = [
    { "name"          : "unlabeled"           , "id": 0 , "trainId": 255, "category": "void"        , "catId": 0, "hasInstances": False, "ignoreInEval": True , "color": [0  , 0  , 0  ] },
    { "name"          : "ego vehicle"         , "id": 1 , "trainId": 255, "category": "void"        , "catId": 0, "hasInstances": False, "ignoreInEval": True , "color": [0  , 0  , 0  ] },
    { "name"          : "rectification border", "id": 2 , "trainId": 255, "category": "void"        , "catId": 0, "hasInstances": False, "ignoreInEval": True , "color": [0  , 0  , 0  ] },
    { "name"          : "out of roi"          , "id": 3 , "trainId": 255, "category": "void"        , "catId": 0, "hasInstances": False, "ignoreInEval": True , "color": [0  , 0  , 0  ] },
    { "name"          : "static"              , "id": 4 , "trainId": 255, "category": "void"        , "catId": 0, "hasInstances": False, "ignoreInEval": True , "color": [0  , 0  , 0  ] },
    { "name"          : "dynamic"             , "id": 5 , "trainId": 255, "category": "void"        , "catId": 0, "hasInstances": False, "ignoreInEval": True , "color": [111, 74 , 0  ] },
    { "name"          : "ground"              , "id": 6 , "trainId": 255, "category": "void"        , "catId": 0, "hasInstances": False, "ignoreInEval": True , "color": [81 , 0  , 81 ] },
    { "name"          : "road"                , "id": 7 , "trainId": 0  , "category": "flat"        , "catId": 1, "hasInstances": False, "ignoreInEval": False, "color": [128, 64 , 128] },
    { "name"          : "sidewalk"            , "id": 8 , "trainId": 1  , "category": "flat"        , "catId": 1, "hasInstances": False, "ignoreInEval": False, "color": [244, 35 , 232] },
    { "name"          : "parking"             , "id": 9 , "trainId": 255, "category": "flat"        , "catId": 1, "hasInstances": False, "ignoreInEval": True , "color": [250, 170, 160] },
    { "name"          : "rail track"          , "id": 10, "trainId": 255, "category": "flat"        , "catId": 1, "hasInstances": False, "ignoreInEval": True , "color": [230, 150, 140] },
    { "name"          : "building"            , "id": 11, "trainId": 2  , "category": "flat"        , "catId": 1, "hasInstances": False, "ignoreInEval": False, "color": [ 70, 70 , 70 ] },
    { "name"          : "wall"                , "id": 12, "trainId": 3  , "category": "construction", "catId": 2, "hasInstances": False, "ignoreInEval": False, "color": [102, 102, 156] },
    { "name"          : "fence"               , "id": 13, "trainId": 4  , "category": "construction", "catId": 2, "hasInstances": False, "ignoreInEval": False, "color": [190, 153, 153] },
    { "name"          : "guard rail"          , "id": 14, "trainId": 255, "category": "construction", "catId": 2, "hasInstances": False, "ignoreInEval": True , "color": [180, 165, 180] },
    { "name"          : "bridge"              , "id": 15, "trainId": 255, "category": "construction", "catId": 2, "hasInstances": False, "ignoreInEval": True , "color": [150, 100, 100] },
    { "name"          : "tunnel"              , "id": 16, "trainId": 255, "category": "construction", "catId": 2, "hasInstances": False, "ignoreInEval": True , "color": [150, 120, 90 ] },
    { "name"          : "pole"                , "id": 17, "trainId": 5  , "category": "object"      , "catId": 3, "hasInstances": False, "ignoreInEval": False, "color": [153, 153, 153] },
    { "name"          : "polegroup"           , "id": 18, "trainId": 255, "category": "object"      , "catId": 3, "hasInstances": False, "ignoreInEval": True , "color": [153, 153, 153] },
    { "name"          : "traffic light"       , "id": 19, "trainId": 6  , "category": "object"      , "catId": 3, "hasInstances": False, "ignoreInEval": False, "color": [250, 170, 30 ] },
    { "name"          : "traffic sign"        , "id": 20, "trainId": 7  , "category": "object"      , "catId": 3, "hasInstances": False, "ignoreInEval": False, "color": [220, 220, 0  ] },
    { "name"          : "vegetation"          , "id": 21, "trainId": 8  , "category": "nature"      , "catId": 4, "hasInstances": False, "ignoreInEval": False, "color": [107, 142, 35 ] },
    { "name"          : "terrain"             , "id": 22, "trainId": 9  , "category": "nature"      , "catId": 4, "hasInstances": False, "ignoreInEval": False, "color": [152, 251, 152] },
    { "name"          : "sky"                 , "id": 23, "trainId": 10 , "category": "sky"         , "catId": 5, "hasInstances": False, "ignoreInEval": False, "color": [ 70, 130, 180] },
    { "name"          : "person"              , "id": 24, "trainId": 11 , "category": "human"       , "catId": 6, "hasInstances": True , "ignoreInEval": False, "color": [220, 20 , 60 ] },
    { "name"          : "rider"               , "id": 25, "trainId": 12 , "category": "human"       , "catId": 6, "hasInstances": True , "ignoreInEval": False, "color": [255, 0  , 0  ] },
    { "name"          : "car"                 , "id": 26, "trainId": 13 , "category": "vehicle"     , "catId": 7, "hasInstances": True , "ignoreInEval": False, "color": [0  , 0  , 142] },
    { "name"          : "truck"               , "id": 27, "trainId": 14 , "category": "vehicle"     , "catId": 7, "hasInstances": True , "ignoreInEval": False, "color": [0  , 0  , 70 ] },
    { "name"          : "bus"                 , "id": 28, "trainId": 15 , "category": "vehicle"     , "catId": 7, "hasInstances": True , "ignoreInEval": False, "color": [0  , 60 , 100] },
    { "name"          : "caravan"             , "id": 29, "trainId": 255, "category": "vehicle"     , "catId": 7, "hasInstances": True , "ignoreInEval": True , "color": [0  , 0  , 90 ] },
    { "name"          : "trailer"             , "id": 30, "trainId": 255, "category": "vehicle"     , "catId": 7, "hasInstances": True , "ignoreInEval": True , "color": [0  , 0  , 110] },
    { "name"          : "train"               , "id": 31, "trainId": 16 , "category": "vehicle"     , "catId": 7, "hasInstances": True , "ignoreInEval": False, "color": [0  , 80 , 100] },
    { "name"          : "motorcycle"          , "id": 32, "trainId": 17 , "category": "vehicle"     , "catId": 7, "hasInstances": True , "ignoreInEval": False, "color": [0  , 0  , 230] },
    { "name"          : "bicycle"             , "id": 33, "trainId": 18 , "category": "vehicle"     , "catId": 7, "hasInstances": True , "ignoreInEval": False, "color": [119, 11 , 32 ] },
    { "name"          : "license plate"       , "id": -1, "trainId": -1 , "category": "vehicle"     , "catId": 7, "hasInstances": False, "ignoreInEval": True , "color": [0  , 0  , 142] }
]

# endregion


# region Dataset

@constant.DATASET.register(name="cityscapes-fog")
class CityscapesFog(base.ImageEnhancementDataset):
    """Cityscapes-Fog.
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        beta: Additional information on the attenuation coefficient. One of the
            values in :attr:`betas`. Can be a list to include multiple beta
            values. When "all", "*", or None, all beta values will be included.
            Defaults to "*".
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
    
    betas = [0.005, 0.01, 0.02]
    
    def __init__(
        self,
        name            : str                       = "cityscapes-fog",
        root            : PathType                  = constant.DATA_DIR / "cityscapes",
        split           : str                       = "train",
        beta            : float | list | str | None = "*",
        extra           : bool                      = False,
        shape           : Ints                      = (3, 256, 256),
        classlabels     : ClassLabelsType | None    = None,
        transform       : TransformType   | None    = None,
        target_transform: TransformType   | None    = None,
        transforms      : TransformType   | None    = None,
        cache_data      : bool                      = False,
        cache_images    : bool                      = False,
        backend         : VisionBackendType         = constant.VISION_BACKEND,
        verbose         : bool                      = True,
        *args, **kwargs
    ):
        self.beta  = beta
        self.extra = extra
        
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
    
    @property
    def beta(self) -> list[float]:
        return self._beta

    @beta.setter
    def beta(self, beta: float | list | str):
        beta = [beta] if isinstance(beta, list) else [beta]
        if beta is None or "all" in beta or "*" in beta:
            beta = self.betas
        self._beta = beta

    def list_images(self):
        """List image files."""
        if self.split not in ["train", "val", "test"]:
            core.console.log(
                f"{self.__class__.__name__} dataset only supports "
                f":param:`split`: 'train', 'val', or 'test'. Get: {self.split}."
            )
        
        image_patterns = [
            self.root / f"leftImg8bit-foggy" / self.split / "*" / f"*_{beta}*.png"
            for beta in self.beta
        ]
        if self.split == "train" and self.extra:
            image_patterns += self.root / "leftImg8bit-foggy" / "train-extra" / "*" / "*.png"
        
        image_paths = []
        for pattern in image_patterns:
            image_paths += glob.glob(str(pattern))
        image_paths = core.unique(image_paths)
        
        self.images: list[base.ImageLabel] = []
        with core.rich.progress_bar() as pbar:
            for path in pbar.track(
                image_paths,
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split} images"
            ):
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
                path       = str(img.path)
                postfix    = path[path.find("_foggy_beta"):]
                label_path = path.replace(postfix, ".png")
                label_path = label_path.replace("leftImg8bit-foggy", "leftImg8bit")
                self.labels.append(
                    base.ImageLabel(path=label_path, backend=self.backend)
                )
           
     
@constant.DATASET.register(name="cityscapes-lol")
class CityscapesLOL(base.ImageEnhancementDataset):
    """Cityscapes-LOL.
    
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
        name            : str                    = "cityscapes-lol",
        root            : PathType               = constant.DATA_DIR / "cityscapes",
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
                f":param:`split`: 'train' or 'val'. Get: {self.split}."
            )
        
        self.images: list[base.ImageLabel] = []
        with core.rich.progress_bar() as pbar:
            pattern = self.root / "leftImg8bit-lol" / self.split
            for path in pbar.track(
                list(pattern.rglob("low/*.png")),
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split} images"
            ):
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
          
          
@constant.DATASET.register(name="cityscapes-rain")
class CityscapesRain(base.ImageEnhancementDataset):
    """Cityscapes-Rain.
    
    Args:
        name: A datamodule name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        alpha: One of the values in :attr:`alphas`. Can be a list to include
            multiple alpha values. When "all", "*", or None, all alpha values
            will be included. Defaults to "*".
        beta: Additional information on the attenuation coefficient. One of the
            values in :attr:`betas`. Can be a list to include multiple beta
            values. When "all", "*", or None, all beta values will be included.
            Defaults to "*".
        drop_size: One of the values in :attr:`drop_sizes`. Can be a list to
            include multiple drop sizes. When "all", "*", or None, all drop
            sizes will be included. Defaults to "*".
        pattern: Rain pattern. One of the values in :attr:`patterns`. Can be a
            list to include multiple pattern values. When "all", "*", or None,
            all drop sizes will be included. Defaults to "*".
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
    
    alphas     = [0.01,  0.02,  0.03]
    betas      = [0.005, 0.01,  0.02]
    drop_sizes = [0.002, 0.005, 0.01]
    patterns   = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    def __init__(
        self,
        name            : str                       = "cityscapes-rain",
        root            : PathType                  = constant.DATA_DIR / "cityscapes",
        split           : str                       = "train",
        alpha           : float | list | str | None = "*",
        beta            : float | list | str | None = "*",
        drop_size       : float | list | str | None = "*",
        pattern         : int   | list | str | None = "*",
        shape           : Ints                      = (3, 256, 256),
        classlabels     : ClassLabelsType | None    = None,
        transform       : TransformType   | None    = None,
        target_transform: TransformType   | None    = None,
        transforms      : TransformType   | None    = None,
        cache_data      : bool                      = False,
        cache_images    : bool                      = False,
        backend         : VisionBackendType         = constant.VISION_BACKEND,
        verbose         : bool                      = True,
        *args, **kwargs
    ):
        self.alpha     = alpha
        self.beta      = beta
        self.drop_size = drop_size
        self.pattern   = pattern
        
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
    
    @property
    def alpha(self) -> list[float]:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float | list | str):
        alpha = [alpha] if isinstance(alpha, list) else [alpha]
        if alpha is None or "all" in alpha or "*" in alpha:
            alpha = self.alphas
        self._alpha = alpha

    @property
    def beta(self) -> list[float]:
        return self._beta

    @beta.setter
    def beta(self, beta: float | list | str):
        beta = [beta] if isinstance(beta, list) else [beta]
        if beta is None or "all" in beta or "*" in beta:
            beta = self.betas
        self._beta = beta

    @property
    def drop_size(self) -> list[float]:
        return self._drop_size

    @drop_size.setter
    def drop_size(self, drop_size: float | list | str):
        drop_size = [drop_size] if isinstance(drop_size, list) else [drop_size]
        if drop_size is None or "all" in drop_size or "*" in drop_size:
            drop_size = self.drop_sizes
        self._drop_size = drop_size
    
    @property
    def pattern(self) -> list[int]:
        return self._pattern

    @pattern.setter
    def pattern(self, pattern: int | list | str):
        pattern = [pattern] if isinstance(pattern, list) else [pattern]
        if pattern is None or "all" in pattern or "*" in pattern:
            pattern = self.patterns
        self._pattern = pattern
        
    def list_images(self):
        """List image files."""
        if self.split not in ["train", "val"]:
            core.console.log(
                f"{self.__class__.__name__} dataset only supports :param:`split`: "
                f"'train' or 'val'. Get: {self.split}."
            )
        
        image_patterns  = []
        image_patterns += [
            self.root / f"leftImg8bit-rain" / self.split / "*" / f"*_alpha_{alpha}*.png"
            for alpha in self.alpha
        ]
        image_patterns += [
            self.root / f"leftImg8bit-rain" / self.split / "*" / f"*_beta_{beta}*.png"
            for beta in self.beta
        ]
        image_patterns += [
            self.root / f"leftImg8bit-rain" / self.split / "*" / f"*_dropsize_{drop_size}*.png"
            for drop_size in self.drop_size
        ]
        image_patterns += [
            self.root / f"leftImg8bit-rain" / self.split / "*" / f"*_pattern_{pattern}*.png"
            for pattern in self.pattern
        ]
        
        image_paths = []
        for pattern in image_patterns:
            for image_path in glob.glob(str(pattern)):
                image_paths.append(image_path)
        image_paths = core.unique(image_paths)  # Remove all duplicates files
            
        self.images: list[base.ImageLabel] = []
        with core.rich.progress_bar() as pbar:
            for path in pbar.track(
                image_paths,
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split} images"
            ):
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
                path       = str(img.path)
                postfix    = path[path.find("_rain_alpha"):]
                label_path = path.replace(postfix, ".png")
                label_path = label_path.replace("leftImg8bit-rain", "leftImg8bit")
                self.labels.append(
                    base.ImageLabel(path=label_path, backend=self.backend)
                )


@constant.DATASET.register(name="cityscapes-snow")
class CityscapesSnow(base.ImageEnhancementDataset):
    """Cityscapes-Snow.
    
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
        name            : str                    = "cityscapes-snow",
        root            : PathType               = constant.DATA_DIR / "cityscapes",
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
        if self.split not in ["train", "test"]:
            core.console.log(
                f"{self.__class__.__name__} dataset only supports "
                f":param:`split`: 'train' or 'test'. Get: {self.split}."
            )
        
        self.images: list[base.ImageLabel] = []
        with core.rich.progress_bar() as pbar:
            pattern = self.root / "leftImg8bit-snow" / self.split
            for path in pbar.track(
                list(pattern.rglob("*Snow/*")),
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
                parent = str(img.path.parent.name)
                path   = core.Path(str(img.path).replace(parent, "gt"))
                self.labels.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )
                

@constant.DATASET.register(name="cityscapes-semantic")
class CityscapesSemantic(base.ImageSegmentationDataset):
    """Cityscapes Semantic Segmentation.
    
    Args:
        name: A datamodule name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        quality: The quality of the semantic segmentation mask to use. One of
            the values in :attr:`qualities`. Defaults to "gtFine".
        encoding: The format to use when creating the semantic segmentation
            mask. One of the values in :attr:`encodings`. Defaults to "id".
        extra: Should use extra data? Those in the `train_extra` split
            are only available for `quality=gtCoarse`. Defaults to False.
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

    qualities = ["gtFine", "gtCoarse"]
    encodings = ["id", "trainId", "catId", "color"]
    
    def __init__(
        self,
        name            : str                    = "cityscapes-semantic",
        root            : PathType               = constant.DATA_DIR / "cityscapes",
        split           : str                    = "train",
        quality         : str                    = "gtFine",
        encoding        : str                    = "id",
        extra           : bool                   = False,
        shape           : Ints                   = (3, 256, 256),
        classlabels     : ClassLabelsType | None = cityscapes_classlabels,
        transform       : TransformType   | None = None,
        target_transform: TransformType   | None = None,
        transforms      : TransformType   | None = None,
        cache_data      : bool                   = False,
        cache_images    : bool                   = False,
        backend         : VisionBackendType      = constant.VISION_BACKEND,
        verbose         : bool                   = True,
        *args, **kwargs
    ):
        self.quality = quality
        self.extra   = extra
        if quality not in self.qualities:
            raise ValueError(f"Cityscapes Semantic dataset does not supports "
                             f"`quality`: `{quality}`.")
        if encoding not in self.encodings:
            raise ValueError(f"Cityscapes Semantic dataset does not supports "
                             f"`encoding`: `{encoding}`.")
        
        super().__init__(
            name             = name,
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels or cityscapes_classlabels,
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
        if self.quality == "gtCoarse":
            if self.split not in ["train", "val"]:
                core.console.log(
                    f"{self.__class__.__name__} dataset only supports "
                    f":param:`split`: 'train' or 'val'. Get: {self.split}."
                )
        else:
            if self.split not in ["train", "val", "test"]:
                core.console.log(
                    f"{self.__class__.__name__} dataset only supports "
                    f":param:`split`: 'train', 'val', or 'test'. "
                    f"Get: {self.split}."
                )
        
        image_patterns = [self.root / "leftImg8bit" / self.split / "*" / "*.png"]
        if self.split == "train" and self.quality == "gtCoarse" and self.extra:
            image_patterns.append(
                self.root / "leftImg8bit" / "train-extra" / "*" / "*.png"
            )

        image_paths = []
        for pattern in image_patterns:
            image_paths += glob.glob(str(pattern))
        
        self.images: list[base.ImageLabel] = []
        with core.rich.progress_bar() as pbar:
            for path in pbar.track(
                image_paths,
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split} images"
            ):
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
                path       = str(img.path)
                prefix     = path.replace("_leftImg8bit.png", "")
                prefix     = prefix.replace("leftImg8bit", self.quality)
                label_path = f"{prefix}_{self.quality}_{self.encodings}.png"
                self.labels.append(
                    base.ImageLabel(path=label_path, backend=self.backend)
                )
                
# endregion


# region Datamodule

@constant.DATAMODULE.register(name="cityscapes-fog")
class CityscapesFogDataModule(base.DataModule):
    """Cityscapes-Fog DataModule
    
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
        name            : str                  = "cityscapes-fog",
        root            : PathType             = constant.DATA_DIR / "cityscapes",
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
        """Use this method to do things that might write to disk or that need
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
            self.train = CityscapesFog(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.val = CityscapesFog(
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
            
        # Assign test datasets for use in dataloader
        if phase in [None, constant.ModelPhase.TESTING]:
            self.test = CityscapesFog(
                root             = self.root,
                split            = "test",
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
        self.classlabels = base.ClassLabels.from_value(value=cityscapes_classlabels)


@constant.DATAMODULE.register(name="cityscapes-lol")
class CityscapesLOLDataModule(base.DataModule):
    """Cityscapes-LOL DataModule
    
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
        name            : str                  = "cityscapes-lol",
        root            : PathType             = constant.DATA_DIR / "cityscapes",
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
            self.train = CityscapesLOL(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.val = CityscapesLOL(
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
            self.test = CityscapesFog(
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
        self.classlabels = base.ClassLabels.from_value(value=cityscapes_classlabels)


@constant.DATAMODULE.register(name="cityscapes-rain")
class CityscapesRainDataModule(base.DataModule):
    """Cityscapes-Rain datamodule.
    
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
        name            : str                  = "cityscapes-rain",
        root            : PathType             = constant.DATA_DIR / "cityscapes",
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
            self.train = CityscapesRain(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.val = CityscapesRain(
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
            self.test = CityscapesRain(
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
        self.classlabels = base.ClassLabels.from_value(value=cityscapes_classlabels)


@constant.DATAMODULE.register(name="cityscapes-snow")
class CityscapesSnowDataModule(base.DataModule):
    """Cityscapes-Snow DataModule
    
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
        name            : str                  = "cityscapes-snow",
        root            : PathType             = constant.DATA_DIR / "cityscapes",
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
            full_dataset = CityscapesSnow(
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
            self.test = CityscapesRain(
                root             = self.root,
                split            = "test",
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
        self.classlabels = base.ClassLabels.from_value(value=cityscapes_classlabels)
        
# endregion


# region Test

def test_cityscapes_fog():
    cfg = {
        "name": "cityscapes-fog",
            # A datamodule's name.
        "root": constant.DATA_DIR / "cityscapes",
            # A root directory where the data is stored.
        "beta": "*",
            # Additional information on the attenuation coefficient. One of the
            # values in :attr:`betas`. Can be a list to include multiple beta
            # values. When "all", "*", or None, all betas will be included.
            # Defaults to "*".
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
    dm  = CityscapesFogDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter           = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    result              = {"image": input, "target": target}
    label               = [(m["name"]) for m in meta]
    visualize.imshow_enhancement(
        winname = "image",
        image   = result,
        label   = label
    )
    visualize.plt.show(block=True)
    

def test_cityscapes_lol():
    cfg = {
        "name": "cityscapes-lol",
            # A datamodule's name.
        "root": constant.DATA_DIR / "cityscapes",
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
    dm  = CityscapesLOLDataModule(**cfg)
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


def test_cityscapes_rain():
    cfg = {
        "name": "cityscapes-rain",
            # A datamodule's name.
        "root": constant.DATA_DIR / "cityscapes",
            # A root directory where the data is stored.
        "alpha": "*",
            # One of: [0.01, 0.02, 0.03]. Can be a list to include multiple
            # alpha values. When "all", "*", or None, all alpha values will be
            # included. Defaults to "*".
        "beta": "*",
            # Additional information on the attenuation coefficient. One of:
            # [0.005, 0.01, 0.02]. Can be a list to include multiple beta
            # values. When "all", "*", or None, all beta values will be
            # included. Defaults to "*".
        "drop_size": "*",
            # One of: [0.002, 0.005, 0.01]. Can also be a list to include
            # multiple drop sizes. When "all", "*", or None, all drop sizes
            # will be included. Defaults to "*".
        "pattern": "*",
            # Rain pattern. One of: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].
            # Can be a list to include multiple patterns. When "all", "*",
            # or None, all drop sizes will be included. Defaults to "*".
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
    dm  = CityscapesRainDataModule(**cfg)
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


def test_cityscapes_snow():
    cfg = {
        "name": "cityscapes-snow",
            # A datamodule's name.
        "root": constant.DATA_DIR / "cityscapes",
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
    dm  = CityscapesRainDataModule(**cfg)
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

# endregion


# region Main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str , default="test-cityscapes-fog", help="The task to run")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.task == "test-cityscapes-fog":
        test_cityscapes_fog()
    elif args.task == "test-cityscapes-lol":
        test_cityscapes_lol()
    elif args.task == "test-cityscapes-rain":
        test_cityscapes_rain()
    elif args.task == "test-cityscapes-snow":
        test_cityscapes_snow()

# endregion
