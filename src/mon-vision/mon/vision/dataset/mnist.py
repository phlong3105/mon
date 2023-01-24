#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CIFAR datasets and datamodules.
"""

from __future__ import annotations

__all__ = [
    "FashionMNIST", "FashionMNISTDataModule", "MNIST", "MNISTDataModule",
    "fashion_mnist_classlabels", "mnist_classlabels",
]

import argparse
from urllib.error import URLError

import torch
from torch.utils.data import random_split
from torchvision.datasets.mnist import read_image_file, read_label_file
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
)

from mon import core, coreimage as ci
from mon.vision import constant, visualize
from mon.vision.dataset import base
from mon.vision.transform import transform as t
from mon.vision.typing import (
    CallableType, ClassLabelsType, Ints, ModelPhaseType, PathType,
    Strs, TransformType, VisionBackendType,
)

# region ClassLabels

mnist_classlabels = [
    { "name": "0", "id": 0 },
    { "name": "1", "id": 1 },
    { "name": "2", "id": 2 },
    { "name": "3", "id": 3 },
    { "name": "4", "id": 4 },
    { "name": "5", "id": 5 },
    { "name": "6", "id": 6 },
    { "name": "7", "id": 7 },
    { "name": "8", "id": 8 },
    { "name": "9", "id": 9 }
]

fashion_mnist_classlabels = [
    { "name": "T-shirt/top", "id": 0 },
    { "name": "Trouser",     "id": 1 },
    { "name": "Pullover",    "id": 2 },
    { "name": "Dress",       "id": 3 },
    { "name": "Coat",        "id": 4 },
    { "name": "Sandal",      "id": 5 },
    { "name": "Shirt",       "id": 6 },
    { "name": "Sneaker",     "id": 7 },
    { "name": "Bag",         "id": 8 },
    { "name": "Ankle boot",  "id": 9 }
]

# endregion


# region Dataset

@constant.DATASET.register(name="mnist")
class MNIST(base.ImageClassificationDataset):
    """MNIST dataset.
    
    References:
        http://yann.lecun.com/exdb/mnist/
    
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
    
    mirrors       = [
        "http://yann.lecun.com/exdb/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
    ]
    resources     = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz",  "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz",  "ec29112dd5afa0611ce80d1b7f02629c")
    ]
    training_file = "training.pt"
    test_file     = "test.pt"
    classes       = [
        "0 - zero", "1 - one", "2 - two", "3 - three", "4 - four",
        "5 - five", "6 - six", "7 - seven", "8 - eight", "9 - nine"
    ]
    
    def __init__(
        self,
        name            : str                    = "mnist",
        root            : PathType               = constant.DATA_DIR / "mnist" / "mnist",
        split           : str                    = "train",
        shape           : Ints                   = (3, 32, 32),
        classlabels     : ClassLabelsType | None = mnist_classlabels,
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
        
        if not self._check_exists():
            self.download()

        image_file = f"{'train' if self.split == 'train' else 't10k'}-images-idx3-ubyte"
        data       = read_image_file(self.root / "raw" / image_file)
        data       = torch.unsqueeze(data, -1)
        data       = data.repeat(1, 1, 1, 3)
        
        self.images: list[base.ImageLabel] = [
            base.ImageLabel(
                image          = ci.to_tensor(img, keepdim=False, normalize=True),
                keep_in_memory = True
            )
            for img in data
        ]
        
    def list_labels(self):
        """List label files."""
        label_file = f"{'train' if self.split == 'train' else 't10k'}-labels-idx1-ubyte"
        data       = read_label_file(self.root / "raw" / label_file)
        self.labels: list[base.ClassificationLabel] = [
            base.ClassificationLabel(id=l) for l in data
        ]
        
    def filter(self):
        pass
    
    def _check_legacy_exist(self):
        processed_folder = self.root / self.__class__.__name__ / "processed"
        if not processed_folder.exists():
            return False

        return all(
            check_integrity(processed_folder / file)
            for file in (self.training_file, self.test_file)
        )

    def _load_legacy_data(self):
        # This is for BC only. We no longer cache the data in a custom binary,
        # but simply read from the raw data directly.
        processed_folder = self.root / "processed"
        data_file        = self.training_file if self.train else self.test_file
        return torch.load(self.processed_folder / data_file)
    
    def _check_exists(self) -> bool:
        return all(
            check_integrity(self.root / "raw" / core.Path(url).stem)
            for url, _ in self.resources
        )

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""
        if self._check_exists():
            return

        raw_folder = self.root / self.__class__.__name__ / "raw"
        core.create_dirs([raw_folder])

        # Download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = "{}{}".format(mirror, filename)
                try:
                    core.console.log("Downloading {}".format(url))
                    download_and_extract_archive(
                        url,
                        download_root = raw_folder,
                        filename      = filename,
                        md5           = md5
                    )
                except URLError as error:
                    core.console.log("Failed to download (trying next):\n{}".format(error))
                    continue
                finally:
                    core.console.log()
                break
            else:
                raise RuntimeError("Error downloading {}".format(filename))

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


@constant.DATASET.register(name="fashion-mnist")
class FashionMNIST(MNIST):
    """Fashion-MNIST dataset.
    
    References:
        https://github.com/zalandoresearch/fashion-mnist
    
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
    
    mirrors   = [
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    ]
    resources = [
        ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
        ("t10k-images-idx3-ubyte.gz",  "bef4ecab320f06d8554ea6380940ec79"),
        ("t10k-labels-idx1-ubyte.gz",  "bb300cfdad3c16e7a12a480ee83cd310")
    ]
    classes   = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
        "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    
    def __init__(
        self,
        name            : str                    = "fashion-mnist",
        root            : PathType               = constant.DATA_DIR / "mnist" / "fashion-mnist",
        split           : str                    = "train",
        shape           : Ints                   = (3, 32, 32),
        classlabels     : ClassLabelsType | None = fashion_mnist_classlabels,
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

# endregion


# region Datamodule

@constant.DATAMODULE.register(name="mnist")
class MNISTDataModule(base.DataModule):
    """MNIST datamodule.
    
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
        name            : str                  = "mnist",
        root            : PathType             = constant.DATA_DIR / "mnist" / "mnist",
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
            full_dataset = MNIST(
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
            self.test = MNIST(
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
        self.classlabels = base.ClassLabels.from_value(value=mnist_classlabels)


@constant.DATAMODULE.register(name="fashion-mnist")
class FashionMNISTDataModule(base.DataModule):
    """Fashion-MNIST datamodule.
    
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
        name            : str                  = "fashion-mnist",
        root            : PathType             = constant.DATA_DIR / "mnist" / "fashion-mnist",
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
            full_dataset = FashionMNIST(
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
            self.test = FashionMNIST(
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
        self.classlabels = base.ClassLabels.from_value(value=fashion_mnist_classlabels)

# endregion


# region Test

def test_mnist():
    cfg = {
        "name": "mnist",
            # A datamodule's name.
        "root": constant.DATA_DIR / "mnist" / "mnist",
            # A root directory where the data is stored.
        "shape": [3, 32, 32],
            # The desired datapoint shape preferably in a channel-last format.
            # Defaults to (3, 256, 256).
        "transform": None,
            # Transformations performing on the input.
        "target_transform": None,
            # Transformations performing on the target.
        "transforms": [
            t.Resize(size=[3, 32, 32]),
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
    dm  = MNISTDataModule(**cfg)
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


def test_fashion_mnist():
    cfg = {
        "name": "fashion-mnist",
            # A datamodule's name.
        "root": constant.DATA_DIR / "mnist" / "fashion-mnist",
            # A root directory where the data is stored.
        "shape": [3, 32, 32],
            # The desired datapoint shape preferably in a channel-last format.
            # Defaults to (3, 256, 256).
        "transform": None,
            # Transformations performing on the input.
        "target_transform": None,
            # Transformations performing on the target.
        "transforms": [
            t.Resize(size=[3, 32, 32]),
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
    dm  = FashionMNISTDataModule(**cfg)
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
    parser.add_argument("--task", type=str , default="test-mnist", help="The task to run")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.task == "test-mnist":
        test_mnist()
    elif args.task == "test-fashion-mnist":
        test_fashion_mnist()

# endregion
