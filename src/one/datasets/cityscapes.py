#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cityscapes dataset and datamodule.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from one.constants import *
from one.core import *
from one.data import ClassLabels
from one.data import ClassLabels_
from one.data import DataModule
from one.data import Image
from one.data import ImageEnhancementDataset
from one.data import ImageSegmentationDataset
from one.plot import imshow_enhancement
from one.vision.transformation import Resize


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


# H1: - Module -----------------------------------------------------------------

@DATASETS.register(name="cityscape_fog")
class CityscapesFog(ImageEnhancementDataset):
    """
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        beta (float | list | str | None): Additional information on the
            attenuation coefficient. One of the values in `self.betas`. Can
            also be a list to include multiple betas. When `all`, `*`, or
            `None`, all betas will be included. Defaults to "*".
        shape (Ints): Image shape as [C, H, W], [H, W], or [S, S].
        classlabels (ClassLabels_ | None): ClassLabels object. Defaults to
            None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    betas = [0.005, 0.01, 0.02]
    
    def __init__(
        self,
        root            : Path_,
        split           : str                       = "train",
        beta            : float | list | str | None = "*",
        extra           : bool                      = False,
        shape           : Ints                      = (3, 512, 512),
        classlabels     : ClassLabels_ | None       = None,
        transform       : Transforms_  | None       = None,
        target_transform: Transforms_  | None       = None,
        transforms      : Transforms_  | None       = None,
        cache_data      : bool                      = False,
        cache_images    : bool                      = False,
        backend         : VisionBackend_            = VISION_BACKEND,
        verbose         : bool                      = True,
        *args, **kwargs
    ):
        self.beta  = beta
        self.extra = extra
        
        super().__init__(
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
        """
        List image files.
        """
        if self.split not in ["train", "val", "test"]:
            console.log(
                f"{self.__class__.classname} dataset only supports `split`: "
                f"`train`, `val`, or `test`. Get: {self.split}."
            )
    
        image_patterns = [
            self.root / f"leftImg8bit_foggy" / self.split / "*" / f"*_{beta}*.png"
            for beta in self.beta
        ]
        if self.split == "train" and self.extra:
            image_patterns += self.root / "leftImg8bit_foggy" / "train_extra" / "*" / "*.png"
        
        image_paths = []
        for pattern in image_patterns:
            image_paths += glob.glob(str(pattern))
        image_paths = unique(image_paths)

        self.images: list[Image] = []
        with progress_bar() as pbar:
            for path in pbar.track(
                image_paths,
                description=f"Listing {self.__class__.classname} "
                            f"{self.split} images"
            ):
                self.images.append(Image(path=path, backend=self.backend))
    
    def list_labels(self):
        """
        List label files.
        """
        self.labels: list[Image] = []
        with progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.classname} "
                            f"{self.split} labels"
            ):
                path       = str(img.path)
                postfix    = path[path.find("_foggy_beta"):]
                label_path = path.replace(postfix, ".png")
                label_path = label_path.replace("leftImg8bit_foggy", "leftImg8bit")
                self.labels.append(Image(path=label_path, backend=self.backend))
           
     
@DATASETS.register(name="cityscape_lol")
class CityscapesLol(ImageEnhancementDataset):
    """
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image shape as [C, H, W], [H, W], or [S, S].
        classlabels (ClassLabels_ | None): ClassLabels object. Defaults to
            None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root            : Path_,
        split           : str                 = "train",
        shape           : Ints                = (3, 512, 512),
        classlabels     : ClassLabels_ | None = None,
        transform       : Transforms_  | None = None,
        target_transform: Transforms_  | None = None,
        transforms      : Transforms_  | None = None,
        cache_data      : bool                = False,
        cache_images    : bool                = False,
        backend         : VisionBackend_      = VISION_BACKEND,
        verbose         : bool                = True,
        *args, **kwargs
    ):
        super().__init__(
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
        """
        List image files.
        """
        if self.split not in ["train", "val"]:
            console.log(
                f"{self.__class__.classname} dataset only supports `split`: "
                f"`train` or `val`. Get: {self.split}."
            )

        self.images: list[Image] = []
        with progress_bar() as pbar:
            pattern = self.root / "lol" / self.split
            for path in pbar.track(
                list(pattern.rglob("low/*.png")),
                description=f"Listing {self.__class__.classname} "
                            f"{self.split} images"
            ):
                self.images.append(Image(path=path, backend=self.backend))
    
    def list_labels(self):
        """
        List label files.
        """
        self.labels: list[Image] = []
        with progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.classname} "
                            f"{self.split} labels"
            ):
                path = Path(str(img.path).replace("low", "high"))
                self.labels.append(Image(path=path, backend=self.backend))
          
          
@DATASETS.register(name="cityscape_rain")
class CityscapesRain(ImageEnhancementDataset):
    """
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        alpha (float | list | str | None): One of the values in `self.alphas`.
            Can also be a list to include multiple alphas. When `all`, `*`, or
            `None`, all alphas will be included. Defaults to "*".
        beta (float | list | str | None): Additional information on the
            attenuation coefficient. One of the values in `self.betas`. Can
            also be a list to include multiple betas. When `all`, `*`, or `None`,
            all betas will be included. Defaults to "*".
        drop_size (float | list | str | None): One of the values in
            `self.drop_sizes`. Can also be a list to include multiple drop
            sizes. When `all`, `*`, or `None`, all drop sizes will be included.
            Defaults to "*".
        pattern (int | list | str | None): Rain pattern. One of the values in
            `self.patterns`. Can also be a list to include multiple patterns.
            When `all`, `*`, or `None`, all drop sizes will be included.
            Defaults to "*".
        shape (Ints): Image shape as [C, H, W], [H, W], or [S, S].
        classlabels (ClassLabels_ | None): ClassLabels object. Defaults to
            None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    alphas     = [0.01,  0.02,  0.03]
    betas      = [0.005, 0.01,  0.02]
    drop_sizes = [0.002, 0.005, 0.01]
    patterns   = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    def __init__(
        self,
        root            : Path_,
        split           : str                       = "train",
        alpha           : float | list | str | None = "*",
        beta            : float | list | str | None = "*",
        drop_size       : float | list | str | None = "*",
        pattern         : int   | list | str | None = "*",
        shape           : Ints                      = (3, 512, 512),
        classlabels     : ClassLabels_ | None       = None,
        transform       : Transforms_  | None       = None,
        target_transform: Transforms_  | None       = None,
        transforms      : Transforms_  | None       = None,
        cache_data      : bool                      = False,
        cache_images    : bool                      = False,
        backend         : VisionBackend_            = VISION_BACKEND,
        verbose         : bool                      = True,
        *args, **kwargs
    ):
        self.alpha     = alpha
        self.beta      = beta
        self.drop_size = drop_size
        self.pattern   = pattern
        
        super().__init__(
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
        """
        List image files.
        """
        if self.split not in ["train", "val"]:
            console.log(
                f"{self.__class__.classname} dataset only supports `split`: "
                f"`train` or `val`. Get: {self.split}."
            )
        
        image_patterns  = []
        image_patterns += [
            self.root / f"leftImg8bit_rain" / self.split / "*" / f"*_alpha_{alpha}*.png"
            for alpha in self.alpha
        ]
        image_patterns += [
            self.root / f"leftImg8bit_rain" / self.split / "*" / f"*_beta_{beta}*.png"
            for beta in self.beta
        ]
        image_patterns += [
            self.root / f"leftImg8bit_rain" / self.split / "*" / f"*_dropsize_{drop_size}*.png"
            for drop_size in self.drop_size
        ]
        image_patterns += [
            self.root / f"leftImg8bit_rain" / self.split / "*" / f"*_pattern_{pattern}*.png"
            for pattern in self.pattern
        ]
        
        image_paths = []
        for pattern in image_patterns:
            for image_path in glob.glob(str(pattern)):
                image_paths.append(image_path)
        image_paths = unique(image_paths)  # Remove all duplicates files
            
        self.images: list[Image] = []
        with progress_bar() as pbar:
            for path in pbar.track(
                image_paths,
                description=f"Listing {self.__class__.classname} "
                            f"{self.split} images"
            ):
                self.images.append(Image(path=path, backend=self.backend))
    
    def list_labels(self):
        """
        List label files.
        """
        self.labels: list[Image] = []
        with progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.classname} "
                            f"{self.split} labels"
            ):
                path       = str(img.path)
                postfix    = path[path.find("_rain_alpha"):]
                label_path = path.replace(postfix, ".png")
                label_path = label_path.replace("leftImg8bit_rain", "leftImg8bit")
                self.labels.append(Image(path=label_path, backend=self.backend))


@DATASETS.register(name="cityscape_semantic")
class CityscapesSemantic(ImageSegmentationDataset):
    """
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        quality (str): Quality of the semantic segmentation mask to use. One of
            the values in `self.qualities`. Defaults to "gtFine".
        encoding (str): Format to use when creating the semantic segmentation
            mask. One of the values in `self.encodings`. Defaults to "id".
        extra (bool): Should use extra data? Those in the `train_extra` split
            are only available for `quality=gtCoarse`. Defaults to False.
        shape (Ints): Image shape as [C, H, W], [H, W], or [S, S].
        classlabels (ClassLabels_ | None): ClassLabels object. Defaults to
            None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """

    qualities = ["gtFine", "gtCoarse"]
    encodings = ["id", "trainId", "catId", "color"]
    
    def __init__(
        self,
        root            : Path_,
        split           : str                 = "train",
        quality         : str                 = "gtFine",
        encoding        : str                 = "id",
        extra           : bool                = False,
        shape           : Ints                = (3, 512, 512),
        classlabels     : ClassLabels_ | None = None,
        transform       : Transforms_  | None = None,
        target_transform: Transforms_  | None = None,
        transforms      : Transforms_  | None = None,
        cache_data      : bool                = False,
        cache_images    : bool                = False,
        backend         : VisionBackend_      = VISION_BACKEND,
        verbose         : bool                = True,
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
        
        path = os.path.join(root, quality, f"classlabels.json")
        if classlabels is None:
            if not os.path.isfile(path):
                classlabels = ClassLabels.from_file(path)
            else:
                classlabels = ClassLabels.from_list(cityscapes_classlabels)
        
        super().__init__(
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
        """
        List image files.
        """
        if self.quality == "gtCoarse":
            if self.split not in ["train", "val"]:
                console.log(
                    f"{self.__class__.classname} dataset only supports "
                    f"`split`: `train` or `val`. Get: {self.split}."
                )
        else:
            if self.split not in ["train", "val", "test"]:
                console.log(
                    f"{self.__class__.classname} dataset only supports "
                    f"`split`: `train`, `val`, or `test`. Get: {self.split}."
                )
        
        image_patterns = [self.root / "leftImg8bit" / self.split / "*" / "*.png"]
        if self.split == "train" and self.quality == "gtCoarse" and self.extra:
            image_patterns.append(
                self.root / "leftImg8bit" / "train_extra" / "*" / "*.png"
            )

        image_paths = []
        for pattern in image_patterns:
            image_paths += glob.glob(str(pattern))
        
        self.images: list[Image] = []
        with progress_bar() as pbar:
            for path in pbar.track(
                image_paths,
                description=f"Listing {self.__class__.classname} "
                            f"{self.split} images"
            ):
                self.images.append(Image(path=path, backend=self.backend))
    
    def list_labels(self):
        """
        List label files.
        """
        self.labels: list[Image] = []
        with progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.classname} "
                            f"{self.split} labels"
            ):
                path       = str(img.path)
                prefix     = path.replace("_leftImg8bit.png", "")
                prefix     = prefix.replace("leftImg8bit", self.quality)
                label_path = f"{prefix}_{self.quality}_{self.encodings}.png"
                self.labels.append(Image(path=label_path, backend=self.backend))
                

@DATAMODULES.register(name="cityscapes_fog")
class CityscapesFogDataModule(DataModule):
    """
    """
    
    def __init__(
        self,
        root: Path_ = DATA_DIR / "cityscapes",
        name: str   = "cityscapes_fog",
        *args, **kwargs
    ):
        super().__init__(root=root, name=name, *args, **kwargs)
        
    def prepare_data(self, *args, **kwargs):
        """
        Use this method to do things that might write to disk or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.load_classlabels()
    
    def setup(self, phase: ModelPhase_ | None = None):
        """
        There are also data operations you might want to perform on every GPU.

        Todos:
            - Count number of classes.
            - Build classlabels vocabulary.
            - Perform train/val/test splits.
            - Apply transforms (defined explicitly in your datamodule or
              assigned in init).
            - Define collate_fn for you custom dataset.

        Args:
            phase (ModelPhase_ | None):
                Stage to use: [None, ModelPhase.TRAINING, ModelPhase.TESTING].
                Set to None to setup all train, val, and test data.
                Defaults to None.
        """
        console.log(f"Setup [red]{CityscapesFogDataModule.classname}[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
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
            
        # Assign test datasets for use in dataloader(s)
        if phase in [None, ModelPhase.TESTING]:
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
        """
        Load ClassLabels.
        """
        self.classlabels = ClassLabels.from_list(cityscapes_classlabels)


@DATAMODULES.register(name="cityscapes_lol")
class CityscapesLolDataModule(DataModule):
    """
    """
    
    def __init__(
        self,
        root: Path_ = DATA_DIR / "cityscapes",
        name: str   = "cityscapes_lol",
        *args, **kwargs
    ):
        super().__init__(root=root, name=name, *args, **kwargs)
        
    def prepare_data(self, *args, **kwargs):
        """
        Use this method to do things that might write to disk or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.load_classlabels()
    
    def setup(self, phase: ModelPhase_ | None = None):
        """
        There are also data operations you might want to perform on every GPU.

        Todos:
            - Count number of classes.
            - Build classlabels vocabulary.
            - Perform train/val/test splits.
            - Apply transforms (defined explicitly in your datamodule or
              assigned in init).
            - Define collate_fn for you custom dataset.

        Args:
            phase (ModelPhase_ | None):
                Stage to use: [None, ModelPhase.TRAINING, ModelPhase.TESTING].
                Set to None to setup all train, val, and test data.
                Defaults to None.
        """
        console.log(f"Setup [red]{CityscapesLolDataModule.classname}[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
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
            
        # Assign test datasets for use in dataloader(s)
        if phase in [None, ModelPhase.TESTING]:
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
        """
        Load ClassLabels.
        """
        self.classlabels = ClassLabels.from_list(cityscapes_classlabels)


@DATAMODULES.register(name="cityscapes_rain")
class CityscapesRainDataModule(DataModule):
    """
    """
    
    def __init__(
        self,
        root: Path_ = DATA_DIR / "cityscapes",
        name: str   = "cityscapes_rain",
        *args, **kwargs
    ):
        super().__init__(root=root, name=name, *args, **kwargs)
        
    def prepare_data(self, *args, **kwargs):
        """
        Use this method to do things that might write to disk or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.load_classlabels()
    
    def setup(self, phase: ModelPhase_ | None = None):
        """
        There are also data operations you might want to perform on every GPU.

        Todos:
            - Count number of classes.
            - Build classlabels vocabulary.
            - Perform train/val/test splits.
            - Apply transforms (defined explicitly in your datamodule or
              assigned in init).
            - Define collate_fn for you custom dataset.

        Args:
            phase (ModelPhase_ | None):
                Stage to use: [None, ModelPhase.TRAINING, ModelPhase.TESTING].
                Set to None to setup all train, val, and test data.
                Defaults to None.
        """
        console.log(f"Setup [red]{CityscapesRainDataModule.classname}[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
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
            
        # Assign test datasets for use in dataloader(s)
        if phase in [None, ModelPhase.TESTING]:
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
        """
        Load ClassLabels.
        """
        self.classlabels = ClassLabels.from_list(cityscapes_classlabels)


# H1: - Test -------------------------------------------------------------------

def test_cityscapes_fog():
    cfg = {
        "root": DATA_DIR / "cityscapes",
           # Root directory of dataset.
        "name": "cityscapes_fog",
            # Dataset's name.
        "beta": "*",
            # Additional information on the attenuation coefficient.
            # One of: [0.005, 0.01, 0.02]. Can also be a list to include
            # multiple betas. When `all`, `*`, or `None`, all betas will be
            # included. Defaults to "*".
        "shape": [3, 512, 512],
            # Image shape as [C, H, W], [H, W], or [S, S].
        "transform": None,
            # Functions/transforms that takes in an input sample and returns a
            # transformed version.
        "target_transform": None,
            # Functions/transforms that takes in a target and returns a
            # transformed version.
        "transforms": [
            Resize(size=[3, 512, 512])
        ],
            # Functions/transforms that takes in an input and a target and
            # returns the transformed versions of both.
        "cache_data": False,
            # If True, cache data to disk for faster loading next time.
            # Defaults to False.
        "cache_images": False,
            # If True, cache images into memory for faster training (WARNING:
            # large datasets may exceed system RAM). Defaults to False.
        "backend": VISION_BACKEND,
            # Vision backend to process image. Defaults to VISION_BACKEND.
        "batch_size": 8,
            # Number of samples in one forward & backward pass. Defaults to 1.
        "devices" : 0,
            # The devices to use. Defaults to 0.
        "shuffle": True,
            # If True, reshuffle the data at every training epoch.
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
    result              = {"image" : input, "target": target}
    label               = [(m["name"]) for m in meta]
    imshow_enhancement(winname="image", image=result, label=label)
    plt.show(block=True)
    

def test_cityscapes_lol():
    cfg = {
        "root": DATA_DIR / "cityscapes",
           # Root directory of dataset.
        "name": "cityscapes_lol",
            # Dataset's name.
        "shape": [3, 512, 512],
            # Image shape as [C, H, W], [H, W], or [S, S].
        "transform": None,
            # Functions/transforms that takes in an input sample and returns a
            # transformed version.
        "target_transform": None,
            # Functions/transforms that takes in a target and returns a
            # transformed version.
        "transforms": [
            Resize(size=[3, 512, 512])
        ],
            # Functions/transforms that takes in an input and a target and
            # returns the transformed versions of both.
        "cache_data": False,
            # If True, cache data to disk for faster loading next time.
            # Defaults to False.
        "cache_images": False,
            # If True, cache images into memory for faster training (WARNING:
            # large datasets may exceed system RAM). Defaults to False.
        "backend": VISION_BACKEND,
            # Vision backend to process image. Defaults to VISION_BACKEND.
        "batch_size": 8,
            # Number of samples in one forward & backward pass. Defaults to 1.
        "devices" : 0,
            # The devices to use. Defaults to 0.
        "shuffle": True,
            # If True, reshuffle the data at every training epoch.
            # Defaults to True.
        "verbose": True,
            # Verbosity. Defaults to True.
    }
    dm  = CityscapesLolDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter           = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    result              = {"image" : input, "target": target}
    label               = [(m["name"]) for m in meta]
    imshow_enhancement(winname="image", image=result, label=label)
    plt.show(block=True)


def test_cityscapes_rain():
    cfg = {
        "root": DATA_DIR / "cityscapes",
           # Root directory of dataset.
        "name": "cityscapes_rain",
            # Dataset's name.
        "alpha": "*",
            # One of: [0.01, 0.02, 0.03]. Can also be a list to include multiple
            # alphas. When `all`, `*`, or `None`, all alphas will be included.
            # Defaults to "*".
        "beta": "*",
            # Additional information on the attenuation coefficient.
            # One of: [0.005, 0.01, 0.02]. Can also be a list to include
            # multiple betas. When `all`, `*`, or `None`, all betas will be
            # included. Defaults to "*".
        "drop_size": "*",
            # One of: [0.002, 0.005, 0.01]. Can also be a list to include
            # multiple drop sizes. When `all`, `*`, or `None`, all drop sizes
            # will be included. Defaults to "*".
        "pattern": "*",
            # Rain pattern. One of: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].
            # Can also be a list to include multiple patterns. When `all`, `*`,
            # or `None`, all drop sizes will be included. Defaults to "*".
        "shape": [3, 512, 512],
            # Image shape as [C, H, W], [H, W], or [S, S].
        "transform": None,
            # Functions/transforms that takes in an input sample and returns a
            # transformed version.
        "target_transform": None,
            # Functions/transforms that takes in a target and returns a
            # transformed version.
        "transforms": [
            Resize(size=[3, 512, 512])
        ],
            # Functions/transforms that takes in an input and a target and
            # returns the transformed versions of both.
        "cache_data": False,
            # If True, cache data to disk for faster loading next time.
            # Defaults to False.
        "cache_images": False,
            # If True, cache images into memory for faster training (WARNING:
            # large datasets may exceed system RAM). Defaults to False.
        "backend": VISION_BACKEND,
            # Vision backend to process image. Defaults to VISION_BACKEND.
        "batch_size": 8,
            # Number of samples in one forward & backward pass. Defaults to 1.
        "devices" : 0,
            # The devices to use. Defaults to 0.
        "shuffle": True,
            # If True, reshuffle the data at every training epoch.
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
    imshow_enhancement(winname="image", image=result, label=label)
    plt.show(block=True)


# H1: - Main -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str , default="test_cityscapes_fog", help="The task to run")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.task == "test_cityscapes_fog":
        test_cityscapes_fog()
    elif args.task == "test_cityscapes_lol":
        test_cityscapes_lol()
    elif args.task == "test_cityscapes_rain":
        test_cityscapes_rain()
