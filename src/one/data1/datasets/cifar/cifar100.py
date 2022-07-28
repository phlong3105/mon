#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CIFAR100 dataset and datamodule.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import random_split

from one.core import console
from one.core import DATAMODULES
from one.core import DATASETS
from one.core import VisionBackend
from one.data.data_class import ClassLabels
from one.data.datamodule import DataModule
from one.data.datasets.cifar.cifar10 import CIFAR10
from one.imgproc import show_images
from one.core import ModelState
from one.utils import data_dir

__all__ = [
	"CIFAR100",
    "CIFAR100DataModule",
]


# MARK: - Module

@DATASETS.register(name="cifar100")
class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    
    base_folder = "cifar-100-python"
    url         = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename    = "cifar-100-python.tar.gz"
    tgz_md5     = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list  = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list   = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta        = {
        "filename": "meta",
        "key"     : "fine_label_names",
        "md5"     : "7973b15100ade9c7d40fb424638fde48",
    }


@DATAMODULES.register(name="cifar100")
class CIFAR100DataModule(DataModule):
    """Cifar100 DataModule.

    Examples:
        >> Cifar100DataModule(name="cifar100", shape=(32, 32, 3), batch_size=32,
        shuffle=True)
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        dataset_dir: str = os.path.join(data_dir, "cifar"),
        name       : str = "cifar100",
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
        CIFAR100(root=self.dataset_dir, train=True,  download=True,
                 transform=self.transform, **self.dataset_kwargs)
        CIFAR100(root=self.dataset_dir, train=False, download=True,
                 transform=self.transform, **self.dataset_kwargs)
        if self.class_labels is None:
            self.load_class_labels()
    
    def setup(self, model_state: Optional[ModelState] = None):
        """There are also data operations you might want to perform on every
        GPU. Use setup to do things like:
            - Count number of classes.
            - Build labels vocabulary.
            - Perform train/val/test splits.
            - Apply transforms (defined explicitly in your datamodule or
            assigned in init).

        Args:
            model_state (ModelState, optional):
                ModelState to use: [None, ModelState.TRAINING, ModelState.TESTING].
                Set to "None" to setup all train, val, and test data.
                Default: `None`.
        """
        console.log(f"Setup [red]CIFAR-100[/red] datasets.")
        
        # NOTE: Assign train/val datasets for use in dataloaders
        if model_state in [None, ModelState.TRAINING]:
            full_dataset = CIFAR100(
                self.dataset_dir, train=True, download=True,
                transform=self.transform, **self.dataset_kwargs
            )
            self.train, self.val = random_split(full_dataset, [45000, 5000])

        # NOTE: Assign test datasets for use in dataloader(s)
        if model_state in [None, ModelState.TESTING]:
            self.test = CIFAR100(
                self.dataset_dir, train=False, download=True,
                transform=self.transform, **self.dataset_kwargs
            )
        
        if self.class_labels is None:
            self.load_class_labels()
        
        self.summarize()
        
    def load_class_labels(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path 	    = os.path.join(current_dir, "cifar100_class_labels.json")
        self.class_labels = ClassLabels.create_from_file(label_path=path)
        

# MARK: - Main

if __name__ == "__main__":
    # NOTE: Get DataModule
    cfg = {
        "name": "cifar100",
        # Dataset's name.
        "shape": [32, 32, 3],
        # Image shape as [H, W, C]. This is compatible with OpenCV format.
        # This is also used to reshape the input
        # data.
        "num_classes": 100,
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
    dm  = CIFAR100DataModule(**cfg)
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
