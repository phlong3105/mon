#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements CIFAR datasets and datamodules."""

from __future__ import annotations

__all__ = [
    "CIFAR10", "CIFAR100", "CIFAR10DataModule", "CIFAR100DataModule",
    "cifar_10_classlabels", "cifar_100_classlabels",
]

import argparse
import pickle

import numpy as np
from torch.utils.data import random_split
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
)

from mon import core, coreimage as ci
from mon.vision import constant, visualize
from mon.vision.dataset import base
from mon.vision.transform import transform as t
from mon.vision.typing import (
    CallableType, ClassLabelsType, Ints, ModelPhaseType, PathType, Strs,
    TransformType, VisionBackendType,
)

# region ClassLabels

cifar_10_classlabels  = [
    { "name": "airplane",   "id": 0 },
    { "name": "automobile", "id": 1 },
    { "name": "bird",       "id": 2 },
    { "name": "cat",        "id": 3 },
    { "name": "deer",       "id": 4 },
    { "name": "dog",        "id": 5 },
    { "name": "frog",       "id": 6 },
    { "name": "horse",      "id": 7 },
    { "name": "ship",       "id": 8 },
    { "name": "truck",      "id": 9 }
]

cifar_100_classlabels = [
    { "name": "beaver"           , "superclass": "aquatic mammals"               , "id": 0 },
    { "name": "dolphin"          , "superclass": "aquatic mammals"               , "id": 1 },
    { "name": "otter"            , "superclass": "aquatic mammals"               , "id": 2 },
    { "name": "seal"             , "superclass": "aquatic mammals"               , "id": 3 },
    { "name": "whale"            , "superclass": "aquatic mammals"               , "id": 4 },
    { "name": "aquarium fish"    , "superclass": "fish"                          , "id": 5 },
    { "name": "flatfish"         , "superclass": "fish"                          , "id": 6 },
    { "name": "ray"              , "superclass": "fish"                          , "id": 7 },
    { "name": "shark"            , "superclass": "fish"                          , "id": 8 },
    { "name": "trout"            , "superclass": "fish"                          , "id": 9 },
    { "name": "orchids"          , "superclass": "flowers"                       , "id": 10 },
    { "name": "poppies"          , "superclass": "flowers"                       , "id": 11 },
    { "name": "roses"            , "superclass": "flowers"                       , "id": 12 },
    { "name": "sunflowers"       , "superclass": "flowers"                       , "id": 13 },
    { "name": "tulips"           , "superclass": "flowers"                       , "id": 14 },
    { "name": "bottles"          , "superclass": "food containers"               , "id": 15 },
    { "name": "bowls"            , "superclass": "food containers"               , "id": 16 },
    { "name": "cans"             , "superclass": "food containers"               , "id": 17 },
    { "name": "cups"             , "superclass": "food containers"               , "id": 18 },
    { "name": "plates"           , "superclass": "food containers"               , "id": 19 },
    { "name": "apples"           , "superclass": "fruit and vegetables"          , "id": 20 },
    { "name": "mushrooms"        , "superclass": "fruit and vegetables"          , "id": 21 },
    { "name": "oranges"          , "superclass": "fruit and vegetables"          , "id": 22 },
    { "name": "pears"            , "superclass": "fruit and vegetables"          , "id": 23 },
    { "name": "sweet peppers"    , "superclass": "fruit and vegetables"          , "id": 24 },
    { "name": "clock"            , "superclass": "household electrical devices"  , "id": 25 },
    { "name": "computer keyboard", "superclass": "household electrical devices"  , "id": 26 },
    { "name": "lamp"             , "superclass": "household electrical devices"  , "id": 27 },
    { "name": "telephone"        , "superclass": "household electrical devices"  , "id": 28 },
    { "name": "television"       , "superclass": "household electrical devices"  , "id": 29 },
    { "name": "bed"              , "superclass": "household furniture"           , "id": 30 },
    { "name": "chair"            , "superclass": "household furniture"           , "id": 31 },
    { "name": "couch"            , "superclass": "household furniture"           , "id": 32 },
    { "name": "table"            , "superclass": "household furniture"           , "id": 33 },
    { "name": "wardrobe"         , "superclass": "household furniture"           , "id": 34 },
    { "name": "bee"              , "superclass": "insects"                       , "id": 35 },
    { "name": "beetle"           , "superclass": "insects"                       , "id": 36 },
    { "name": "butterfly"        , "superclass": "insects"                       , "id": 37 },
    { "name": "caterpillar"      , "superclass": "insects"                       , "id": 38 },
    { "name": "cockroach"        , "superclass": "insects"                       , "id": 39 },
    { "name": "bear"             , "superclass": "large carnivores"              , "id": 40 },
    { "name": "leopard"          , "superclass": "large carnivores"              , "id": 41 },
    { "name": "lion"             , "superclass": "large carnivores"              , "id": 42 },
    { "name": "tiger"            , "superclass": "large carnivores"              , "id": 43 },
    { "name": "wolf"             , "superclass": "large carnivores"              , "id": 44 },
    { "name": "bridge"           , "superclass": "large man-made outdoor things" , "id": 45 },
    { "name": "castle"           , "superclass": "large man-made outdoor things" , "id": 46 },
    { "name": "house"            , "superclass": "large man-made outdoor things" , "id": 47 },
    { "name": "road"             , "superclass": "large man-made outdoor things" , "id": 48 },
    { "name": "skyscraper"       , "superclass": "large man-made outdoor things" , "id": 49 },
    { "name": "cloud"            , "superclass": "large natural outdoor scenes"  , "id": 50 },
    { "name": "forest"           , "superclass": "large natural outdoor scenes"  , "id": 51 },
    { "name": "mountain"         , "superclass": "large natural outdoor scenes"  , "id": 52 },
    { "name": "plain"            , "superclass": "large natural outdoor scenes"  , "id": 53 },
    { "name": "sea"              , "superclass": "large natural outdoor scenes"  , "id": 54 },
    { "name": "camel"            , "superclass": "large omnivores and herbivores", "id": 55 },
    { "name": "cattle"           , "superclass": "large omnivores and herbivores", "id": 56 },
    { "name": "chimpanzee"       , "superclass": "large omnivores and herbivores", "id": 57 },
    { "name": "elephant"         , "superclass": "large omnivores and herbivores", "id": 58 },
    { "name": "kangaroo"         , "superclass": "large omnivores and herbivores", "id": 59 },
    { "name": "fox"              , "superclass": "medium-sized mammals"          , "id": 60 },
    { "name": "porcupine"        , "superclass": "medium-sized mammals"          , "id": 61 },
    { "name": "possum"           , "superclass": "medium-sized mammals"          , "id": 62 },
    { "name": "raccoon"          , "superclass": "medium-sized mammals"          , "id": 63 },
    { "name": "skunk"            , "superclass": "medium-sized mammals"          , "id": 64 },
    { "name": "crab"             , "superclass": "non-insect invertebrates"      , "id": 65 },
    { "name": "lobster"          , "superclass": "non-insect invertebrates"      , "id": 66 },
    { "name": "snail"            , "superclass": "non-insect invertebrates"      , "id": 67 },
    { "name": "spider"           , "superclass": "non-insect invertebrates"      , "id": 68 },
    { "name": "worm"             , "superclass": "non-insect invertebrates"      , "id": 69 },
    { "name": "baby"             , "superclass": "people"                        , "id": 70 },
    { "name": "boy"              , "superclass": "people"                        , "id": 71 },
    { "name": "girl"             , "superclass": "people"                        , "id": 72 },
    { "name": "man"              , "superclass": "people"                        , "id": 73 },
    { "name": "woman"            , "superclass": "people"                        , "id": 74 },
    { "name": "crocodile"        , "superclass": "reptiles"                      , "id": 75 },
    { "name": "dinosaur"         , "superclass": "reptiles"                      , "id": 76 },
    { "name": "lizard"           , "superclass": "reptiles"                      , "id": 77 },
    { "name": "snake"            , "superclass": "reptiles"                      , "id": 78 },
    { "name": "turtle"           , "superclass": "reptiles"                      , "id": 79 },
    { "name": "hamster"          , "superclass": "small mammals"                 , "id": 80 },
    { "name": "mouse"            , "superclass": "small mammals"                 , "id": 81 },
    { "name": "rabbit"           , "superclass": "small mammals"                 , "id": 82 },
    { "name": "shrew"            , "superclass": "small mammals"                 , "id": 83 },
    { "name": "squirrel"         , "superclass": "small mammals"                 , "id": 84 },
    { "name": "maple"            , "superclass": "trees"                         , "id": 85 },
    { "name": "oak"              , "superclass": "trees"                         , "id": 86 },
    { "name": "palm"             , "superclass": "trees"                         , "id": 87 },
    { "name": "pine"             , "superclass": "trees"                         , "id": 88 },
    { "name": "willow"           , "superclass": "trees"                         , "id": 89 },
    { "name": "bicycle"          , "superclass": "vehicles 1"                    , "id": 90 },
    { "name": "bus"              , "superclass": "vehicles 1"                    , "id": 91 },
    { "name": "motorcycle"       , "superclass": "vehicles 1"                    , "id": 92 },
    { "name": "pickup truck"     , "superclass": "vehicles 1"                    , "id": 93 },
    { "name": "train"            , "superclass": "vehicles 1"                    , "id": 94 },
    { "name": "lawn-mower"       , "superclass": "vehicles 2"                    , "id": 95 },
    { "name": "rocket"           , "superclass": "vehicles 2"                    , "id": 96 },
    { "name": "streetcar"        , "superclass": "vehicles 2"                    , "id": 97 },
    { "name": "tank"             , "superclass": "vehicles 2"                    , "id": 98 },
    { "name": "tractor"          , "superclass": "vehicles 2"                    , "id": 99 }
  ]

# endregion


# region Dataset

@constant.DATASET.register(name="cifar-10")
class CIFAR10(base.ImageClassificationDataset):
    """CIFAR-10.
    
    References:
        https://www.cs.toronto.edu/~kriz/cifar.html
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 32, 32).
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
    
    base_folder = "cifar-10"
    url         = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename    = "cifar10.tar.gz"
    tgz_md5     = "c58f30108f718f92721af3b95e74349a"
    train_list  = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]
    test_list   = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta        = {
        "file_name": "batches.meta",
        "key"     : "label_names",
        "md5"     : "5ff9c542aee3614f3951f8cda6e48888",
    }
    
    def __init__(
        self,
        name            : str                    = "cifar-10",
        root            : PathType               = constant.DATA_DIR / "cifar" / "cifar-10",
        split           : str                    = "train",
        shape           : Ints                   = (3, 32, 32),
        classlabels     : ClassLabelsType | None = cifar_10_classlabels,
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
            classlabels      = classlabels or cifar_10_classlabels,
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
        
        if not self._check_integrity():
            self.download()
            
        downloaded_list = self.train_list if self.split == "train" else self.test_list
        
        images = []
        labels = []
        for filename, checksum in downloaded_list:
            file = self.root / filename
            with open(file, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                images.append(entry["data"])
                if "labels" in entry:
                    labels.extend(entry["labels"])
                else:
                    labels.extend(entry["fine_labels"])

        images = np.vstack(images).reshape(shape=(-1, 3, 32, 32))
        images = images.transpose((0, 2, 3, 1))  # convert to HWC
        self.images: list[base.ImageLabel] = [
            base.ImageLabel(
                image          = ci.to_tensor(image=img, keepdim=False, normalize=True),
                keep_in_memory = True,
            )
            for img in images
        ]
        self.labels: list[base.ClassificationLabel] = [
            base.ClassificationLabel(id=l) for l in labels
        ]
        
    def list_labels(self):
        """List label files."""
        pass

    def filter(self):
        pass
    
    def _load_meta(self):
        path = self.root / self.meta["file_name"]
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError(
                "Dataset metadata file not found or corrupted. You can use "
                "download=True to download it"
            )
        with open(path, "rb") as infile:
            data         = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _check_integrity(self) -> bool:
        for entry in (self.train_list + self.test_list):
            filename, md5 = entry[0], entry[1]
            fpath         = self.root / filename
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            core.console.log(f"Files already downloaded and verified")
            return
        download_and_extract_archive(
            self.url, self.root.parent, filename=self.filename, md5=self.tgz_md5
        )

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


@constant.DATASET.register(name="cifar-100")
class CIFAR100(CIFAR10):
    """CIFAR-100.
    
    References:
        https://www.cs.toronto.edu/~kriz/cifar.html
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 32, 32).
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
    
    base_folder = "cifar-100"
    url         = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename    = "cifar100.tar.gz"
    tgz_md5     = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list  = [["train", "16019d7e3df5f24257cddd939b257f8d"], ]
    test_list   = [["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"], ]
    meta        = {
        "file_name": "meta",
        "key"     : "fine_label_names",
        "md5"     : "7973b15100ade9c7d40fb424638fde48",
    }
    
    def __init__(
        self,
        name            : str                    = "cifar-100",
        root            : PathType               = constant.DATA_DIR / "cifar" / "cifar-100",
        split           : str                    = "train",
        shape           : Ints                   = (3, 32, 32),
        classlabels     : ClassLabelsType | None = cifar_100_classlabels,
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
            classlabels      =classlabels or cifar_100_classlabels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )

# endregion


# region Datamodule

@constant.DATAMODULE.register(name="cifar-10")
class CIFAR10DataModule(base.DataModule):
    """CIFAR-10 datamodule.
    
    Args:
        name: A datamodule's name.
        root: A root directory where the data is stored.
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 32, 32).
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
        name            : str                  = "cifar-10",
        root            : PathType             = constant.DATA_DIR / "cifar" / "cifar-10",
        shape           : Ints                 = (3, 32, 32),
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
    
    @property
    def num_workers(self) -> int:
        """The number of workers used in the data loading pipeline.
        Set to: 4 * the number of :attr:`devices` to avoid a bottleneck.
        """
        return 1
    
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
                - "training" : prepares :attr:`train` and :attr:'val'.
                - "testing"  : prepares :attr:`test`.
                - "inference": prepares :attr:`predict`.
                - None:      : prepares all.
                Defaults to None.
        """
        core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = constant.ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TRAINING]:
            full_dataset = CIFAR10(
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
            self.test = CIFAR10(
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
        self.classlabels = base.ClassLabels.from_value(value=cifar_10_classlabels)


@constant.DATAMODULE.register(name="cifar-100")
class CIFAR100DataModule(base.DataModule):
    """CIFAR-100 datamodule.
    
    Args:
        name: A datamodule's name.
        root: A root directory where the data is stored.
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 32, 32).
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
        name            : str                  = "cifar-100",
        root            : PathType             = constant.DATA_DIR / "cifar" / "cifar-100",
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
                - "training" : prepares :attr:`train` and :attr:'val'.
                - "testing"  : prepares :attr:`test`.
                - "inference": prepares :attr:`predict`.
                - None:      : prepares all.
                Defaults to None.
        """
        core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = constant.ModelPhase.from_value(phase) if phase is not None else phase
        
        # Assign train/val datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TRAINING]:
            full_dataset = CIFAR100(
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
            self.test = CIFAR100(
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
        self.classlabels = base.ClassLabels.from_value(value=cifar_100_classlabels)

# endregion


# region Test

def test_cifar_10():
    cfg = {
        "name": "cifar-10",
            # A datamodule's name.
        "root": constant.DATA_DIR / "cifar" / "cifar-10",
            # A root directory where the data is stored.
        "shape": [3, 32, 32],
            # The desired datapoint shape preferably in a channel-last format.
            # Defaults to (3, 32, 32).
        "transform": [
            t.Resize(size=[3, 32, 32]),
        ],
            # Transformations performing on the input.
        "target_transform": None,
            # Transformations performing on the target.
        "transforms": None,
            # Transformations performing on both the input and target.
        "cache_data": True,
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
    dm  = CIFAR10DataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter           = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    visualize.imshow_classification(
        winname     = "image",
        image       = input,
        target      = target,
        classlabels = dm.classlabels
    )
    visualize.plt.show(block=True)


def test_cifar_100():
    cfg = {
        "name": "cifar-100",
            # A datamodule's name.
        "root": constant.DATA_DIR / "cifar" / "cifar100",
            # A root directory where the data is stored.
        "shape": [3, 32, 32],
            # The desired datapoint shape preferably in a channel-last format.
            # Defaults to (3, 32, 32).
        "transform": [
            t.Resize(size=[3, 32, 32])
        ],
            # Transformations performing on the input.
        "target_transform": None,
            # Transformations performing on the target.
        "transforms": None,
            # Transformations performing on both the input and target.
        "cache_data": True,
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
    dm  = CIFAR100DataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter           = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    visualize.imshow_classification(
        winname     = "image",
        image       = input,
        target      = target,
        classlabels = dm.classlabels
    )
    visualize.plt.show(block=True)

# endregion


# region Main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str , default="test-cifar-10", help="The task to run")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.task == "test-cifar-10":
        test_cifar_10()
    elif args.task == "test-cifar-100":
        test_cifar_100()

# endregion
