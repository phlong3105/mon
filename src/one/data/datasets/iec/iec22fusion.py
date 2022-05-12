#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""IEC22 Fusion dataset and datamodule.

The participants will be given access to raw-RGB images of night scenes.
These images have been captured using the same sensor type and are encoded in
16-bit PNG files with additional meta-data provided in JSON files.

The challenge will start with an initial 50 images provided to participants for
algorithm development and testing. Data will be available after registration;
see the form at the bottom of the page. Additional images will be made
available during the challenge; see information below on evaluation and
leaderboard below.

As extra data, you can use the Cube++ dataset, which was collected using the
same cameras. The dataset is described in this article and can be downloaded
from here.
"""

from __future__ import annotations

import glob
import os
import random
from typing import Callable
from typing import Optional

import matplotlib.pyplot as plt
from torch.utils.data import random_split

from one.core import Augment_
from one.core import console
from one.core import DATAMODULES
from one.core import DATASETS
from one.core import Int3T
from one.core import progress_bar
from one.core import VisionBackend
from one.data.data_class import ImageInfo
from one.data.data_class import VisionData
from one.data.datamodule import DataModule
from one.data.dataset import UnsupervisedImageDataset
from one.data.label_handler import VisionDataHandler
from one.imgproc import show_images
from one.core import ModelState
from one.utils import datasets_dir

__all__ = [
    "IEC22Fusion",
    "IEC22FusionDataModule"
]


# MARK: - IEC22Fusion

@DATASETS.register(name="iec22fusion")
class IEC22Fusion(UnsupervisedImageDataset):
    """IEC22 Fusion dataset."""

    # MARK: Magic Functions
    
    def __init__(
        self,
        root            : str,
        split           : str                     = "train",
        shape           : Int3T                   = (720, 1280, 3),
        caching_labels  : bool                    = False,
        caching_images  : bool                    = False,
        write_labels    : bool                    = False,
        fast_dev_run    : bool                    = False,
        load_augment    : Optional[dict]          = None,
        augment         : Optional[Augment_]      = None,
        vision_backend  : Optional[VisionBackend] = None,
        transforms      : Optional[Callable]      = None,
        transform       : Optional[Callable]      = None,
        target_transform: Optional[Callable]      = None,
        *args, **kwargs
    ):
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            caching_labels   = caching_labels,
            caching_images   = caching_images,
            write_labels     = write_labels,
            fast_dev_run     = fast_dev_run,
            load_augment     = load_augment,
            augment          = augment,
            transforms       = transforms,
            transform        = transform,
            target_transform = target_transform,
            vision_backend   = vision_backend,
            *args, **kwargs
        )
 
    # MARK: List Files
    
    def list_files(self):
        """List image and label files."""
        # NOTE: List all files
        self.list_iec22_files()
        self.list_sice_files()
        self.list_fusioncubepp_files()
        
        # NOTE: fast_dev_run, select only a subset of images
        if self.fast_dev_run:
            indices = [random.randint(0, len(self.image_paths) - 1)
                       for _ in range(self.batch_size)]
            self.image_paths        = [self.image_paths[i]        for i in indices]
            # self.label_paths        = [self.label_paths[i]        for i in indices]
            self.custom_label_paths = [self.custom_label_paths[i] for i in indices]
        
        # NOTE: Assertion
        if len(self.image_paths) <= 0:
            raise ValueError(f"Number of images < 0")
        console.log(f"Number of images: {len(self.image_paths)}.")
    
    def list_iec22_files(self):
        """List all IEC22 image data."""
        with progress_bar() as pbar:
            image_pattern = os.path.join(
                datasets_dir, "iec", "iec22_512", self.split, "low", "*.png"
            )
            for image_path in pbar.track(
                glob.glob(image_pattern),
                description=f"[bright_yellow]Listing IEC22 {self.split} images"
            ):
                custom_label_path = image_path.replace("low", "annotations_custom")
                custom_label_path = custom_label_path.replace(".png", ".json")
                self.image_paths.append(image_path)
                # self.label_paths.append(label_path)
                self.custom_label_paths.append(custom_label_path)
    
    def list_sice_files(self):
        """List all SICE data."""
        with progress_bar() as pbar:
            image_pattern = os.path.join(
                datasets_dir, "sice", "unsupervised", self.split, "low", "*.*"
            )
            for image_path in pbar.track(
                glob.glob(image_pattern),
                description=f"[bright_yellow]Listing SICE {self.split} images"
            ):
                custom_label_path = image_path.replace("low", "annotations_custom")
                custom_label_path = custom_label_path.split(".")[0]
                custom_label_path = f"{custom_label_path}.json"
                self.image_paths.append(image_path)
                # self.label_paths.append(label_path)
                self.custom_label_paths.append(custom_label_path)
    
    def list_fusioncubepp_files(self):
        """List all FusionCube++ image data."""
        with progress_bar() as pbar:
            image_pattern = os.path.join(
                datasets_dir, "cube++", "simplecube++_512", self.split, "png", "*"
            )
            for image_path in pbar.track(
                glob.glob(image_pattern),
                description=f"[bright_yellow]Listing FusionCube++ {self.split} images"
            ):
                custom_label_path = image_path.replace("png", "annotations_custom")
                custom_label_path = custom_label_path.replace(".png", ".json")
                self.image_paths.append(image_path)
                # self.label_paths.append(label_path)
                self.custom_label_paths.append(custom_label_path)
        
        with progress_bar() as pbar:
            image_pattern = os.path.join(
                datasets_dir, "cube++", "simplecube++_512", self.split, "jpg", "*"
            )
            for image_path in pbar.track(
                glob.glob(image_pattern),
                description=f"[bright_yellow]Listing FusionCube++ {self.split} images"
            ):
                custom_label_path = image_path.replace("jpg", "annotations_custom")
                custom_label_path = custom_label_path.replace(".jpg", ".json")
                self.image_paths.append(image_path)
                # self.label_paths.append(label_path)
                self.custom_label_paths.append(custom_label_path)
    
    # MARK: Load Data
    
    def load_label(
        self,
        image_path       : str,
        label_path       : Optional[str] = None,
        custom_label_path: Optional[str] = None
    ) -> VisionData:
        """Load all labels from a raw label `file`.

        Args:
            image_path (str):
                Image file.
            label_path (str, optional):
                Label file. Default: `None`.
            custom_label_path (str, optional):
                Custom label file. Default: `None`.
    
        Returns:
            data (VisionData):
                `VisionData` object.
        """
        # NOTE: If we have custom labels
        if custom_label_path and os.path.isfile(custom_label_path):
            return VisionDataHandler().load_from_file(
                image_path = image_path,
                label_path = custom_label_path,
            )

        # NOTE: Parse info
        image_info = ImageInfo.from_file(image_path=image_path)
        
        return VisionData(image_info=image_info)
    
    def load_class_labels(self):
        """Load ClassLabels."""
        pass
        

# MARK: - IEC22FusionDataModule

@DATAMODULES.register(name="iec22fusion")
class IEC22FusionDataModule(DataModule):
    """IEC22 Fusion DataModule."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        dataset_dir: str = os.path.join(datasets_dir, "iec"),
        name       : str = "iec22fusion",
        *args, **kwargs
    ):
        super().__init__(dataset_dir=dataset_dir, name=name, *args, **kwargs)
        self.dataset_kwargs = kwargs
        
    # MARK: Prepare Data
    
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        if self.class_labels is None:
            self.load_class_labels()
    
    def setup(self, phase: Optional[ModelState] = None):
        """There are also data operations you might want to perform on every GPU.

        Todos:
            - Count number of classes.
            - Build class_labels vocabulary.
            - Perform train/val/test splits.
            - Apply transforms (defined explicitly in your datamodule or
              assigned in init).
            - Define collate_fn for you custom dataset.

        Args:
            phase (ModelState, optional):
                ModelState to use: [None, ModelState.TRAINING, ModelState.TESTING]. Set to
                "None" to setup all train, val, and test data. Default: `None`.
        """
        console.log(f"Setup [red]IEC22 Fusion[/red] datasets.")
        
        # NOTE: Assign train/val datasets for use in dataloaders
        if phase in [None, ModelState.TRAINING]:
            full_dataset = IEC22Fusion(
                root=self.dataset_dir, split="train", **self.dataset_kwargs
            )
            train_size   = int(0.8 * len(full_dataset))
            val_size     = len(full_dataset) - train_size
            self.train, self.val = random_split(
                full_dataset, [train_size, val_size]
            )
            self.class_labels = getattr(full_dataset, "class_labels", None)
            self.collate_fn   = getattr(full_dataset, "collate_fn",   None)
            
        # NOTE: Assign test datasets for use in dataloader(s)
        if phase in [None, Phase.TESTING]:
            self.test         = IEC22Fusion(
                root=self.dataset_dir, split="train", **self.dataset_kwargs
            )
            self.class_labels = getattr(self.test, "class_labels", None)
            self.collate_fn   = getattr(self.test, "collate_fn",   None)
        
        if self.class_labels is None:
            self.load_class_labels()

        self.summarize()
        
    def load_class_labels(self):
        """Load ClassLabels."""
        pass


# MARK: - Main

if __name__ == "__main__":
    # NOTE: Get DataModule
    cfgs = {
        "name": "iec22fusion",
        # Dataset's name.
        "shape": [512, 512, 3],
        # Image shape as [H, W, C]. This is compatible with OpenCV format.
        "batch_size": 4,
        # Number of samples in one forward & backward pass.
        "caching_labels": True,
        # Should overwrite the existing cached labels? Default: `False`.
        "caching_images": False,
        # Cache images into memory for faster training. Default: `False`.
        "write_labels": False,
        # After loading images and labels for the first time, we will convert it
        # to our custom data format and write to files. If `True`, we will
        # overwrite these files. Default: `False`.
        "fast_dev_run": False,
        # Take a small subset of the data for fast debug (i.e, like unit testing).
        # Default: `False`.
        "shuffle": True,
        # Set to `True` to have the data reshuffled at every training epoch.
        # Default: `True`.
        "load_augment": {
            "mosaic": 0.0,
            "mixup" : 0.5,
        },
        # Augmented loading policy.
        # Augmented loading policy.
        "augment": {
            "name": "paired_images_auto_augment",
            # Name of the augmentation policy.
            "policy": "enhancement",
            # Augmentation policy. One of: [`enhancement`]. Default: `enhancement`.
            "fill": None,
            # Pixel fill value for the area outside the transformed image.
            # If given a number, the value is used for all bands respectively.
            "to_tensor": True,
            # Convert a PIL Image or numpy.ndarray [H, W, C] in the range [0, 255]
            # to a torch.FloatTensor of shape [C, H, W] in the  range [0.0, 1.0].
            # Default: `True`.
        },
        # Augmentation policy.
        "vision_backend": VisionBackend.CV,
        # Vision backend option.
    }
    dm   = IEC22FusionDataModule(**cfgs)
    dm.setup()
    # NOTE: Visualize labels
    if dm.class_labels:
        dm.class_labels.print()
    # NOTE: Visualize one sample
    data_iter       = iter(dm.train_dataloader)
    input, _, shape = next(data_iter)
    show_images(images=input, nrow=2, denormalize=True)
    plt.show(block=True)
