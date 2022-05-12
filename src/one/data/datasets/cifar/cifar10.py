#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CIFAR-10 dataset and datamodule.
"""

from __future__ import annotations

import os
import pickle
from typing import Any
from typing import Callable
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import random_split
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity
from torchvision.datasets.utils import download_and_extract_archive

from one.core import Augment_
from one.core import AUGMENTS
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
    "CIFAR10",
    "CIFAR10DataModule"
]


# MARK: - CIFAR10

@DATASETS.register(name="cifar10")
class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (str):
            Root directory of dataset where directory `cifar-10-batches-py`
            exists or will be saved to if download is set to True.
        train (bool, optional):
            If `True`, creates dataset from training set, otherwise creates
            from test set.
        load_augment (dict):
            Augmented loading policy.
        augment (Augment_):
            Augmentation policy.
        transform (callable, optional):
            A function/transform that takes in an PIL image and returns a
            transformed version. E.g, `transforms.RandomCrop`
        target_transform (callable, optional):
            A function/transform that takes in the target and transforms it.
        download (bool, optional):
            If `true`, downloads the dataset from the internet and puts it in
            root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    
    base_folder = "cifar-10-batches-py"
    url         = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename    = "cifar-10-python.tar.gz"
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
        "filename": "batches.meta",
        "key"     : "label_names",
        "md5"     : "5ff9c542aee3614f3951f8cda6e48888",
    }
    
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
        *args, **kwargs
    ):
        super().__init__(
            root, transform=transform, target_transform=target_transform,
        )
        self.train        = train  # training set or test set
        self.load_augment = load_augment
        self.augment      = augment

        if download:
            self.download()
            
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets   = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        input, target = self.data[index], self.targets[index]
        # input = Image.fromarray(input)
        input = to_tensor(input, normalize=False).to(torch.uint8)
        
        if self.augment is not None:
            input  = self.augment.forward(input=input)
        if self.transform is not None:
            input  = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # if self.transforms is not None:
            # input  = self.transforms(input)
            # target = self.transforms(target)
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
    
    # MARK: Configure
    
    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted." +
                               " You can use download=True to download it")
        with open(path, "rb") as infile:
            data         = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath         = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            console.log("Files already downloaded and verified")
            return
        download_and_extract_archive(
            self.url, self.root, filename=self.filename, md5=self.tgz_md5
        )

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


# MARK: - CIFAR10DataModule

@DATAMODULES.register(name="cifar10")
class CIFAR10DataModule(DataModule):
    """Cifar10 DataModule.
    
    Examples:
        >> Cifar10DataModule(name="cifar10", shape=(32, 32, 3), batch_size=32,
        shuffle=True)
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        dataset_dir: str = os.path.join(datasets_dir, "cifar"),
        name       : str = "cifar10",
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
        CIFAR10(root=self.dataset_dir, train=True, download=True,
                transform=self.transform, **self.dataset_kwargs)
        CIFAR10(root=self.dataset_dir, train=False, download=True,
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
                ModelState to use: [None, ModelState.TRAINING, ModelState.TESTING].
                Set to "None" to setup all train, val, and test data.
                Default: `None`.
        """
        console.log(f"Setup [red]CIFAR-10[/red] datasets.")
        
        # NOTE: Assign train/val datasets for use in dataloaders
        if phase in [None, ModelState.TRAINING]:
            full_dataset = CIFAR10(
                self.dataset_dir, train=True, download=True,
                transform=self.transform, **self.dataset_kwargs
            )
            self.train, self.val = random_split(full_dataset, [45000, 5000])

        # NOTE: Assign test datasets for use in dataloader(s)
        if phase in [None, Phase.TESTING]:
            self.test = CIFAR10(
                self.dataset_dir, train=False, download=True,
                transform=self.transform, **self.dataset_kwargs
            )
        
        if self.class_labels is None:
            self.load_class_labels()
        
        self.summarize()
        
    def load_class_labels(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path 		= os.path.join(current_dir, "cifar10_class_labels.json")
        self.class_labels = ClassLabels.create_from_file(label_path=path)
        

# MARK: - Main

if __name__ == "__main__":
    # NOTE: Get DataModule
    cfg = {
        "name": "cifar10",
        # Dataset's name.
        "shape": [32, 32, 3],
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
    dm  = CIFAR10DataModule(**cfg)
    dm.setup()
    # NOTE: Visualize labels
    if dm.class_labels:
        dm.class_labels.print()
    # NOTE: Visualize an iteration
    data_iter 	  = iter(dm.train_dataloader)
    input, target = next(data_iter)
    console.log(target)
    show_images(images=input, denormalize=True)
    plt.show(block=True)
