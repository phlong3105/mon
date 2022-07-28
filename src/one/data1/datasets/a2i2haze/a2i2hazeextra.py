#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A2I2-Haze with extra data dataset and datamodule.
"""

from __future__ import annotations

import glob
import os
import random
from typing import Callable
from typing import Optional

import matplotlib.pyplot as plt

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
from one.data.dataset import ImageEnhancementDataset
from one.data.label_handler import VisionDataHandler
from one.imgproc import show_images
from one.core import ModelState
from one.utils import data_dir

__all__ = [
    "A2I2HazeExtra",
    "A2I2HazeExtraDataModule",
]


# MARK: - Module

@DATASETS.register(name="a2i2hazeextra")
class A2I2HazeExtra(ImageEnhancementDataset):
    """We provide the A2I2-Haze Extra dataset for image de-hazing.
    """
    
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
        if self.split not in ["train", "val", "test"]:
            raise ValueError(f"A2I2-Haze Extra dataset only supports `split`: "
                             f"`train`, `val`, or `test`. Get: {self.split}.")
        
        # NOTE: List all files
        self.list_densehaze_files()
        self.list_ihaze_files()
        self.list_nhhaze_files()
        self.list_ohaze_files()
        # self.list_a2i2_files()
        self.list_satehaze1kthin_files()
        self.list_satehaze1kmoderate_files()
        self.list_satehaze1kthick_files()
        
        # NOTE: fast_dev_run, select only a subset of images
        if self.fast_dev_run:
            indices = [random.randint(0, len(self.image_paths) - 1)
                       for _ in range(self.batch_size)]
            self.image_paths        = [self.image_paths[i]        for i in indices]
            self.eimage_paths       = [self.eimage_paths[i]       for i in indices]
            # self.label_paths        = [self.label_paths[i]        for i in indices]
            self.custom_label_paths = [self.custom_label_paths[i] for i in indices]
        
        # NOTE: Assertion
        if (
            len(self.image_paths) <= 0
            or len(self.image_paths) != len(self.eimage_paths)
        ):
            raise ValueError(
                f"Number of images != Number of enhanced images: "
                f"{len(self.image_paths)} != {len(self.eimage_paths)}."
            )
        console.log(f"Number of images: {len(self.image_paths)}.")
    
    def list_a2i2_files(self):
        with progress_bar() as pbar:
            image_pattern = os.path.join(
                self.root, self.split, "haze_images", "*.jpg"
            )
            for image_path in pbar.track(
                glob.glob(image_pattern),
                description=f"[bright_yellow]Listing A2I2 Haze {self.split} images"
            ):
                eimage_path       = image_path.replace("haze_images", "clean_images")
                custom_label_path = image_path.replace("haze_images", "annotations_custom")
                custom_label_path = custom_label_path.replace(".jpg", ".json")
                self.image_paths.append(image_path)
                self.eimage_paths.append(eimage_path)
                # self.label_paths.append(label_path)
                self.custom_label_paths.append(custom_label_path)
    
    def list_densehaze_files(self):
        with progress_bar() as pbar:
            image_pattern = os.path.join(
                data_dir, "densehaze", self.split, "hazy", "*.png"
            )
            for image_path in pbar.track(
                glob.glob(image_pattern),
                description=f"[bright_yellow]Listing Dense-Haze {self.split} images"
            ):
                eimage_path       = image_path.replace("hazy", "gt")
                custom_label_path = image_path.replace("hazy", "annotations_custom")
                custom_label_path = custom_label_path.replace(".png", ".json")
                self.image_paths.append(image_path)
                self.eimage_paths.append(eimage_path)
                # self.label_paths.append(label_path)
                self.custom_label_paths.append(custom_label_path)
    
    def list_ihaze_files(self):
        with progress_bar() as pbar:
            image_pattern = os.path.join(
                data_dir, "ihaze", self.split, "hazy", "*.jpg"
            )
            for image_path in pbar.track(
                glob.glob(image_pattern),
                description=f"[bright_yellow]Listing I-Haze {self.split} images"
            ):
                eimage_path       = image_path.replace("hazy", "gt")
                custom_label_path = image_path.replace("hazy", "annotations_custom")
                custom_label_path = custom_label_path.replace(".jpg", ".json")
                self.image_paths.append(image_path)
                self.eimage_paths.append(eimage_path)
                # self.label_paths.append(label_path)
                self.custom_label_paths.append(custom_label_path)
    
    def list_nhhaze_files(self):
        with progress_bar() as pbar:
            image_pattern = os.path.join(
                data_dir, "nhhaze", self.split, "hazy", "*.png"
            )
            for image_path in pbar.track(
                glob.glob(image_pattern),
                description=f"[bright_yellow]Listing NH-Haze {self.split} images"
            ):
                eimage_path       = image_path.replace("hazy", "gt")
                custom_label_path = image_path.replace("hazy", "annotations_custom")
                custom_label_path = custom_label_path.replace(".png", ".json")
                self.image_paths.append(image_path)
                self.eimage_paths.append(eimage_path)
                # self.label_paths.append(label_path)
                self.custom_label_paths.append(custom_label_path)
    
    def list_ohaze_files(self):
        with progress_bar() as pbar:
            image_pattern = os.path.join(
                data_dir, "ohaze", self.split, "hazy", "*.jpg"
            )
            for image_path in pbar.track(
                glob.glob(image_pattern),
                description=f"[bright_yellow]Listing O-Haze {self.split} images"
            ):
                eimage_path       = image_path.replace("hazy", "gt")
                custom_label_path = image_path.replace("hazy", "annotations_custom")
                custom_label_path = custom_label_path.replace(".jpg", ".json")
                self.image_paths.append(image_path)
                self.eimage_paths.append(eimage_path)
                # self.label_paths.append(label_path)
                self.custom_label_paths.append(custom_label_path)
    
    def list_satehaze1kthin_files(self):
        with progress_bar() as pbar:
            image_pattern = os.path.join(
                data_dir, "satehaze1k", "thin", self.split, "input", "*.png"
            )
            for image_path in pbar.track(
                glob.glob(image_pattern),
                description=f"[bright_yellow]Listing SateHaze1K Thin {self.split} images"
            ):
                eimage_path       = image_path.replace("input", "target")
                custom_label_path = image_path.replace("input", "annotations_custom")
                custom_label_path = custom_label_path.replace(".png", ".json")
                self.image_paths.append(image_path)
                self.eimage_paths.append(eimage_path)
                # self.label_paths.append(label_path)
                self.custom_label_paths.append(custom_label_path)
                
    def list_satehaze1kmoderate_files(self):
        with progress_bar() as pbar:
            image_pattern = os.path.join(
                data_dir, "satehaze1k", "moderate", self.split, "input", "*.png"
            )
            for image_path in pbar.track(
                glob.glob(image_pattern),
                description=f"[bright_yellow]Listing SateHaze1K Moderate {self.split} images"
            ):
                eimage_path       = image_path.replace("input", "target")
                custom_label_path = image_path.replace("input", "annotations_custom")
                custom_label_path = custom_label_path.replace(".png", ".json")
                self.image_paths.append(image_path)
                self.eimage_paths.append(eimage_path)
                # self.label_paths.append(label_path)
                self.custom_label_paths.append(custom_label_path)
                
    def list_satehaze1kthick_files(self):
        with progress_bar() as pbar:
            image_pattern = os.path.join(
                data_dir, "satehaze1k", "thick", self.split, "input", "*.png"
            )
            for image_path in pbar.track(
                glob.glob(image_pattern),
                description=f"[bright_yellow]Listing SateHaze1K Thick {self.split} images"
            ):
                eimage_path       = image_path.replace("input", "target")
                custom_label_path = image_path.replace("input", "annotations_custom")
                custom_label_path = custom_label_path.replace(".png", ".json")
                self.image_paths.append(image_path)
                self.eimage_paths.append(eimage_path)
                # self.label_paths.append(label_path)
                self.custom_label_paths.append(custom_label_path)
    
    # MARK: Load Data
    
    def load_label(
        self,
        image_path       : str,
        enhance_path     : str,
        label_path       : Optional[str] = None,
        custom_label_path: Optional[str] = None
    ) -> VisionData:
        """Load all labels from a raw label `file`.

        Args:
            image_path (str):
                Image file.
            enhance_path (str):
                Enhanced image file.
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
                image_path  = image_path,
                label_path  = custom_label_path,
                eimage_path = enhance_path
            )

        # NOTE: Parse info
        image_info  = ImageInfo.from_file(image_path=image_path)
        eimage_info = ImageInfo.from_file(image_path=enhance_path)
        
        return VisionData(image_info=image_info, eimage_info=eimage_info)

    def load_class_labels(self):
        """Load ClassLabels."""
        pass
        

@DATAMODULES.register(name="a2i2hazeextra")
class A2I2HazeExtraDataModule(DataModule):
    """A2I2-Haze Extra DataModule."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        dataset_dir: str = os.path.join(data_dir, "a2i2haze"),
        name       : str = "a2i2hazeextra",
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
    
    def setup(self, model_state: Optional[ModelState] = None):
        """There are also data operations you might want to perform on every
        GPU.

        Todos:
            - Count number of classes.
            - Build class_labels vocabulary.
            - Perform train/val/test splits.
            - Apply transforms (defined explicitly in your datamodule or
              assigned in init).
            - Define collate_fn for you custom dataset.

        Args:
            model_state (ModelState, optional):
                ModelState to use: [None, ModelState.TRAINING, ModelState.TESTING].
                Set to "None" to setup all train, val, and test data.
                Default: `None`.
        """
        console.log(f"Setup [red]A2I2-Haze Extra[/red] datasets.")
        
        # NOTE: Assign train/val datasets for use in dataloaders
        if model_state in [None, ModelState.TRAINING]:
            self.train = A2I2HazeExtra(
                root=self.dataset_dir, split="train", **self.dataset_kwargs
            )
            self.val = A2I2HazeExtra(
                root=self.dataset_dir, split="val", **self.dataset_kwargs
            )
            self.class_labels = getattr(self.train, "class_labels", None)
            self.collate_fn   = getattr(self.train, "collate_fn",   None)
        
        if model_state in [None, ModelState.TESTING]:
            self.test = A2I2HazeExtra(
                root=self.dataset_dir, split="test", **self.dataset_kwargs
            )
            self.class_labels = getattr(self.test, "class_labels", None)
            self.collate_fn   = getattr(self.test, "collate_fn",   None)
        
        if self.class_labels is None:
            self.load_class_labels()
        
        self.summarize()
        
    def load_class_labels(self):
        pass
        

# MARK: - Main

if __name__ == "__main__":
    # NOTE: Get DataModule
    cfgs = {
        "name": "a2i2hazeextra",
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
        "vision_backend": VisionBackend.PIL,
        # Vision backend option.
    }
    dm   = A2I2HazeExtraDataModule(**cfgs)
    dm.setup()
    # NOTE: Visualize labels
    if dm.class_labels:
        dm.class_labels.print()
    # NOTE: Visualize one sample
    data_iter            = iter(dm.train_dataloader)
    input, target, shape = next(data_iter)
    show_images(images=input,  nrow=2, denormalize=True)
    show_images(images=target, nrow=2, denormalize=True, figure_num=1)
    plt.show(block=True)
