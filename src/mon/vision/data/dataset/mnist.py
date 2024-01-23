#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CIFAR datasets and datamodules."""

from __future__ import annotations

__all__ = [
    "FashionMNIST",
    "FashionMNISTDataModule",
    "MNIST",
    "MNISTDataModule",
    "fashion_mnist_classlabels",
    "mnist_classlabels",
]

from urllib.error import URLError

import torch
from torchvision.datasets.mnist import read_image_file, read_label_file
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
)

from mon import core, nn
from mon.globals import DATAMODULES, DATASETS, ModelPhase
from mon.vision.data import base

console = core.console


# region ClassLabels

mnist_classlabels = [
    {"name": "0", "id": 0},
    {"name": "1", "id": 1},
    {"name": "2", "id": 2},
    {"name": "3", "id": 3},
    {"name": "4", "id": 4},
    {"name": "5", "id": 5},
    {"name": "6", "id": 6},
    {"name": "7", "id": 7},
    {"name": "8", "id": 8},
    {"name": "9", "id": 9}
]

fashion_mnist_classlabels = [
    {"name": "T-shirt/top", "id": 0},
    {"name": "Trouser"    , "id": 1},
    {"name": "Pullover"   , "id": 2},
    {"name": "Dress"      , "id": 3},
    {"name": "Coat"       , "id": 4},
    {"name": "Sandal"     , "id": 5},
    {"name": "Shirt"      , "id": 6},
    {"name": "Sneaker"    , "id": 7},
    {"name": "Bag"        , "id": 8},
    {"name": "Ankle boot" , "id": 9}
]

# endregion


# region Dataset

@DATASETS.register(name="mnist")
class MNIST(base.ImageClassificationDataset):
    """MNIST dataset.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageClassificationDataset`.
    
    References:
        `<http://yann.lecun.com/exdb/mnist/>`__
    """
    
    mirrors = [
        "http://yann.lecun.com/exdb/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
    ]
    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz" , "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz" , "ec29112dd5afa0611ce80d1b7f02629c")
    ]
    training_file = "training.pt"
    test_file     = "test.pt"
    classes       = [
        "0 - zero", "1 - one", "2 - two", "3 - three", "4 - four",
        "5 - five", "6 - six", "7 - seven", "8 - eight", "9 - nine"
    ]
    
    def get_images(self):
        """Get image files."""
        if self.split not in ["train", "test"]:
            console.log(
                f"split must be one of ['train', 'test'], but got {self.split}."
            )
        
        if not self._check_exists():
            self.download()
        
        image_file = f"{'train' if self.split == 'train' else 't10k'}-images-idx3-ubyte"
        data = read_image_file(self.root / "raw" / image_file)
        data = torch.unsqueeze(data, -1)
        data = data.repeat(1, 1, 1, 3)
        data = data.numpy()
        
        self.images: list[base.ImageLabel] = [
            base.ImageLabel(image=img, keep_in_memory=True) for img in data
        ]
    
    def get_labels(self):
        """Get label files."""
        label_file = f"{'train' if self.split == 'train' else 't10k'}-labels-idx1-ubyte"
        data = read_label_file(self.root / "raw" / label_file)
        data = data.numpy()
        self.labels: list[base.ClassificationLabel] = [
            base.ClassificationLabel(id_=l) for l in data
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
        data_file = self.training_file if self.train else self.test_file
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
        raw_folder.mkdir(parents=True, exist_ok=True)
        
        # Download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = "{}{}".format(mirror, filename)
                try:
                    console.log("Downloading {}".format(url))
                    download_and_extract_archive(
                        url,
                        download_root=raw_folder,
                        filename=filename,
                        md5=md5
                    )
                except URLError as error:
                    console.log(
                        "Failed to download (trying next):\n{}".format(error)
                    )
                    continue
                finally:
                    console.log()
                break
            else:
                raise RuntimeError("Error downloading {}".format(filename))
    
    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


@DATASETS.register(name="fashion-mnist")
class FashionMNIST(MNIST):
    """Fashion-MNIST dataset.
    
    References:
        `<https://github.com/zalandoresearch/fashion-mnist>`__
    """
    
    mirrors = [
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    ]
    resources = [
        ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
        ("t10k-images-idx3-ubyte.gz" , "bef4ecab320f06d8554ea6380940ec79"),
        ("t10k-labels-idx1-ubyte.gz" , "bb300cfdad3c16e7a12a480ee83cd310")
    ]
    classes = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
        "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    

# endregion


# region Datamodule

@DATAMODULES.register(name="mnist")
class MNISTDataModule(base.DataModule):
    """MNIST datamodule.
    
    See Also: :class:`mon.nn.data.datamodule.DataModule`.
    """
    
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
            self.get_classlabels()
    
    def setup(self, phase: ModelPhase | None = None):
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
                Default: ``None``.
        """
        console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        
        if phase in [None, ModelPhase.TRAINING]:
            dataset = MNIST(split="train", **self.dataset_kwargs)
            self.split_train_val(dataset=dataset, split_ratio=0.8, full_train=True)
        if phase in [None, ModelPhase.TESTING]:
            self.test = MNIST(split="test", **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        self.classlabels = nn.ClassLabels.from_value(value=mnist_classlabels)


@DATAMODULES.register(name="fashion-mnist")
class FashionMNISTDataModule(base.DataModule):
    """Fashion-MNIST datamodule.
    
    See Also: :class:`mon.nn.data.datamodule.DataModule`.
    """
    
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
            self.get_classlabels()
    
    def setup(self, phase: ModelPhase | None = None):
        """Use this method to do things on every device:
            - Count number of classes.
            - Build classlabels vocabulary.
            - Prepare train/val/test splits.
            - Apply transformations.
            - Define :attr:`collate_fn` for your custom dataset.

        Args:
            phase: The model phase. One of:
                - ``'training'`` : prepares :attr:'train' and :attr:'val'.
                - ``'testing'``  : prepares :attr:'test'.
                - ``'inference'``: prepares :attr:`predict`.
                - ``None``:      : prepares all.
                - Default: ``None``.
        """
        console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        
        if phase in [None, ModelPhase.TRAINING]:
            dataset = FashionMNIST(split="train", **self.dataset_kwargs)
            self.split_train_val(dataset=dataset, split_ratio=0.8, full_train=True)
        if phase in [None, ModelPhase.TESTING]:
            self.test = FashionMNIST(split="test", **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        self.classlabels = nn.ClassLabels.from_value(value=fashion_mnist_classlabels)


# endregion
