#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FashionMNIST dataset and datamodule.
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
from one.data.datasets.mnist.mnist import MNIST
from one.imgproc import show_images
from one.core import ModelState
from one.utils import datasets_dir

__all__ = [
    "FashionMNIST",
    "FashionMNISTDataModule",
]


# MARK: - Module

@DATASETS.register(name="fashionmnist")
class FashionMNIST(MNIST):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_
    Dataset.
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


@DATAMODULES.register(name="fashionmnist")
class FashionMNISTDataModule(DataModule):
    """FashionMNIST DataModule.

    Examples:
        >> FashionMNISTDataModule(name="fashion_mnist", shape=(32, 32, 1),
        batch_size=32, id_level="id", shuffle=True)
    """
   
    # MARK: Magic Functions
    
    def __init__(
        self,
        dataset_dir: str = os.path.join(datasets_dir, "mnist"),
        name       : str = "fashionmnist",
        *args, **kwargs
    ):
        super().__init__(dataset_dir=dataset_dir, name=name, *args, **kwargs)
        self.dataset_kwargs = kwargs
        
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=self.shape[0:2]),
            torchvision.transforms.ToTensor()
        ])
    
    # MARK: Prepare Data
    
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        FashionMNIST(root=self.dataset_dir, train=True,  download=True,
                     transform=self.transform, **self.dataset_kwargs)
        FashionMNIST(root=self.dataset_dir, train=False, download=True,
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
        console.log(f"Setup [red]FashionMNIST[/red] datasets.")
        
        # NOTE: Assign train/val datasets for use in dataloaders
        if model_state in [None, ModelState.TRAINING]:
            full_dataset = FashionMNIST(
	            self.dataset_dir, train=True, download=True,
	            transform=self.transform, **self.dataset_kwargs
            )
            self.train, self.val = random_split(full_dataset, [55000, 5000])

        # NOTE: Assign test datasets for use in dataloader(s)
        if model_state in [None, ModelState.TESTING]:
            self.test = FashionMNIST(
	            self.dataset_dir, train=False, download=True,
	            transform=self.transform, **self.dataset_kwargs
            )
        
        if self.class_labels is None:
            self.load_class_labels()

        self.summarize()
        
    def load_class_labels(self):
        """Load ClassLabels."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, "fashionmnist_class_labels.json")
        self.class_labels = ClassLabels.create_from_file(label_path=path)


# MARK: - Main

if __name__ == "__main__":
    # NOTE: Get DataModule
    cfg = {
        "name": "fashionmnist",
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
        # Augmented loading policy
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
    dm  = FashionMNISTDataModule(**cfg)
    dm.setup()
    # NOTE: Visualize labels
    if dm.class_labels:
        dm.class_labels.print()
    # NOTE: Visualize an iteration
    data_iter      = iter(dm.train_dataloader)
    input, targets = next(data_iter)
    show_images(images=input, denormalize=True)
    plt.show(block=True)
