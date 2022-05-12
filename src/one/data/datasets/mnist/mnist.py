#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MNIST dataset and datamodule.
"""

from __future__ import annotations

import os
from typing import Any
from typing import Optional
from urllib.error import URLError

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import random_split
from torchvision.datasets import VisionDataset
from torchvision.datasets.mnist import read_image_file
from torchvision.datasets.mnist import read_label_file
from torchvision.datasets.utils import check_integrity
from torchvision.datasets.utils import download_and_extract_archive

from one.core import Augment_
from one.core import AUGMENTS
from one.core import Callable
from one.core import console
from one.core import DATAMODULES
from one.core import DATASETS
from one.core import to_tensor
from one.core import VisionBackend
from one.data.augment import BaseAugment
from one.data.data_class import ClassLabels
from one.data.datamodule import DataModule
from one.imgproc import show_images
from one.core import ModelState
from one.utils import datasets_dir

__all__ = [
    "MNIST",
    "MNISTDataModule",
]


# MARK: - MNIST

@DATASETS.register(name="mnist")
class MNIST(VisionDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (str):
            Root directory of dataset where `MNIST/processed/training.pt`
            and `MNIST/processed/test.pt` exist.
        train (bool, optional):
            If `True`, creates dataset from `training.pt`, otherwise from
            `test.pt`.
        augment (Augment_):
            Augmentation operations.
        transform (callable, optional):
            A function/transform that takes in an PIL image and returns a
            transformed version. E.g, `transforms.RandomCrop`.
        target_transform (callable, optional):
            A function/transform that takes in the target and transforms it.
        download (bool, optional):
            If `true`, downloads the dataset from the internet and puts it in
            root directory. If dataset is already downloaded, it is not
            downloaded again.
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

    # MARK: Magic Functions
    
    def __init__(
        self,
        root            : str,
        train           : bool               = True,
        load_augment    : Optional[dict]     = None,
        augment         : Optional[Augment_] = None,
        transform       : Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download        : bool               = False,
    ):
        super(MNIST, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.train        = train  # training set or test set
        self.load_augment = load_augment
        self.augment      = augment
        
        if self._check_legacy_exist():
            self.data, self.targets = self._load_legacy_data()
            return
    
        if download:
            self.download()
    
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True "
                               "to download it")
    
        self.data, self.targets = self._load_data()
    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        input, target = self.data[index], int(self.targets[index])
        input = to_tensor(input, normalize=False).to(torch.uint8)
        
        if self.augment is not None:
            input  = self.augment.forward(input=input)
        if self.transform is not None:
            input  = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        #if self.transforms is not None:
        #    input  = self.transforms(input)
        #    target = self.transforms(target)
        return input, target
    
    # MARK: Properties

    @property
    def augment(self) -> Optional[BaseAugment]:
        return self._augment
    
    @augment.setter
    def augment(self, augment: Optional[Augment_]):
        """Assign augment configs."""
        if isinstance(augment, BaseAugment):
            self._augment = augment
        elif isinstance(augment, dict):
            self._augment = AUGMENTS.build_from_dict(cfg=augment)
        else:
            self._augment = None
         
    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    @property
    def train_data(self):
        console.log("train_data has been renamed data", style="warning")
        return self.data

    @property
    def train_labels(self):
        console.log("train_labels has been renamed targets", style="warning")
        return self.targets

    @property
    def test_data(self):
        console.log("test_data has been renamed data", style="warning")
        return self.data
    
    @property
    def test_labels(self):
        console.log("test_labels has been renamed targets", style="warning")
        return self.targets
    
    # MARK: Configure
    
    def _check_legacy_exist(self):
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False

        return all(
            check_integrity(os.path.join(self.processed_folder, file))
            for file in (self.training_file, self.test_file)
        )

    def _load_legacy_data(self):
        # This is for BC only. We no longer cache the data in a custom binary,
        # but simply read from the raw data directly.
        data_file = self.training_file if self.train else self.test_file
        return torch.load(os.path.join(self.processed_folder, data_file))

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data       = read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets    = read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets
    
    def _check_exists(self) -> bool:
        return all(
            check_integrity(os.path.join(
                self.raw_folder, os.path.splitext(os.path.basename(url))[0]
            ))
            for url, _ in self.resources
        )

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # Download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = "{}{}".format(mirror, filename)
                try:
                    console.log("Downloading {}".format(url))
                    download_and_extract_archive(
                        url, download_root=self.raw_folder,
                        filename=filename,
                        md5=md5
                    )
                except URLError as error:
                    console.log("Failed to download (trying next):\n{}".format(error))
                    continue
                finally:
                    console.log()
                break
            else:
                raise RuntimeError("Error downloading {}".format(filename))

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


# MARK: - MNISTDataModule

@DATAMODULES.register(name="mnist")
class MNISTDataModule(DataModule):
    """MNIST DataModule.
    
    Examples:
        >> MNISTDataModule(name="mnist", shape=(32, 32, 1), batch_size=32,
        id_level="id", shuffle=True)
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        dataset_dir: str = os.path.join(datasets_dir, "mnist"),
        name       : str = "mnist",
        *args, **kwargs
    ):
        super().__init__(dataset_dir=dataset_dir, name=name, *args, **kwargs)
        self.dataset_kwargs = kwargs
        
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=self.shape[0:2]),
        ])
    
    # MARK: Prepare Data
    
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        MNIST(root=self.dataset_dir, train=True,  download=True,
              transform=self.transform, **self.dataset_kwargs)
        MNIST(root=self.dataset_dir, train=False, download=True,
              transform=self.transform, **self.dataset_kwargs)
        if self.class_labels is None:
            self.load_class_labels()

    def setup(self, phase: Optional[ModelState] = None):
        """There are also data operations you might want to perform on every
        GPU. Use setup to do things like:
            - Count number of classes.
            - Build labels vocabulary.
            - Perform train/val/test splits.
            - Apply transforms (defined explicitly in your datamodule or
              assigned in init).

        Args:
            phase (ModelState, optional):
                Stage to use: [None, ModelState.TRAINING, ModelState.TESTING].
                Set to "None" to setup all train, val, and test data.
                Default: `None`.
         """
        console.log(f"Setup [red]MNIST[/red] datasets.")
        
        # NOTE: Assign train/val datasets for use in dataloaders
        if phase in [None, ModelState.TRAINING]:
            full_dataset = MNIST(
                self.dataset_dir, train=True, download=True,
                transform=self.transform, **self.dataset_kwargs
            )
            self.train, self.val = random_split(full_dataset, [55000, 5000])

        # NOTE: Assign test datasets for use in dataloader(s)
        if phase in [None, Phase.TESTING]:
            self.test = MNIST(
                self.dataset_dir, train=False, download=True,
                transform=self.transform, **self.dataset_kwargs
            )
            
        if self.class_labels is None:
            self.load_class_labels()

        self.summarize()
        
    def load_class_labels(self):
        """Load ClassLabels."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path	    = os.path.join(current_dir, "mnist_class_labels.json")
        self.class_labels = ClassLabels.create_from_file(label_path=path)
        

# MARK: - Main

if __name__ == "__main__":
    # NOTE: Get DataModule
    cfgs = {
        "name": "mnist",
        # Dataset's name.
        "shape": [32, 32, 1],
        # Image shape as [H, W, C]. This is compatible with OpenCV format.
        # This is also used to reshape the input data.
        "num_classes": 10,
        # Number of classes in the dataset.
        "batch_size": 32,
        # Number of samples in one forward & backward pass.
        "shuffle": True,
        # Set to `True` to have the data reshuffled at every training epoch.
        # Default: `True`.
        "load_augment": {
            "mosaic": 0.5,
            "mixup" : 0.5,
        },
        # Augmented loading policy.
        "augment": {
            "name": "image_auto_augment",
            # Name of the augmentation policy
            "policy": "cifar10",
            # Augmentation policy. One of: [`imagenet`, `cifar10`, `svhn`].
            # Default: `imagenet`.
            "fill": None,
            # Pixel fill value for the area outside the transformed image.
            # If given a number, the value is used for all bands respectively.
            "to_tensor": True,
            # Convert a PIL Image or numpy.ndarray [H, W, C] in the range [0, 255]
            # to a torch.FloatTensor of shape [C, H, W] in the  range [0.0, 1.0].
            # Default: `True`.
        },
        # Augmentation policy.
        "vision_backend": VisionBackend.PIL,
        # Vision backend option.
    }
    dm   = MNISTDataModule(**cfgs)
    dm.setup()
    # NOTE: Visualize labels
    if dm.class_labels:
        dm.class_labels.print()
    # NOTE: Visualize an iteration
    data_iter      = iter(dm.train_dataloader)
    input, targets = next(data_iter)
    show_images(images=input, denormalize=True)
    plt.show(block=True)
