#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Rain13K dataset and datamodule.
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
from one.core import Tasks
from one.core import VisionBackend
from one.data.data_class import ImageInfo
from one.data.data_class import VisionData
from one.data.datamodule import DataModule
from one.data.dataset import ImageEnhancementDataset
from one.data.label_handler import VisionDataHandler
from one.imgproc import show_images
from one.core import ModelState
from one.utils import datasets_dir

__all__ = [
    "Rain13K",
    "Rain13KDataModule"
]


# MARK: - Rain13K

@DATASETS.register(name="rain13k")
class Rain13K(ImageEnhancementDataset):
    """Rain13K dataset consists of multiple datasets related to rain removal
    task.

    Args:
		subset (Tasks, optional):
            Sub-dataset to use. One of the values in `self.subsets`.
            Can also be a list to include multiple subsets. When `all`, `*`, or
            `None`, all alphas will be included. Default: `*`.
    """
    
    subsets = [
        "rain12", "rain100", "rain100h", "rain100l", "rain800", "rain1200",
        "rain1400", "rain2800"
    ]
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        root            : str,
        subset          : Optional[Tasks]         = "*",
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
        self.subset = subset
        
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

    # MARK: Properties
    
    @property
    def subset(self) -> list[str]:
        return self._subset

    @subset.setter
    def subset(self, subset: Optional[Tasks]):
        subset = [subset] if isinstance(subset, str) else subset
        if subset is None or "all" in subset or "*" in subset:
            subset = self.subsets
        self._subset = subset
    
    # MARK: List Files
    
    def list_files(self):
        """List image and label files."""
        # NOTE: List all files
        if "rain12" in self.subset:
            self.list_rain12_files()
        if "rain100" in self.subset:
            self.list_rain100_files()
        if "rain100h" in self.subset:
            self.list_rain100h_files()
        if "rain100l" in self.subset:
            self.list_rain100l_files()
        if "rain800" in self.subset:
            self.list_rain800_files()
        if "rain1200" in self.subset:
            self.list_rain1200_files()
        if "rain1400" in self.subset:
            self.list_rain1400_files()
        if "rain2800" in self.subset:
            self.list_rain2800_files()
            
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
        
    def list_rain12_files(self):
        """List all `rain12` files."""
        if self.split not in ["train"]:
            console.log(f"Rain12 dataset only supports `split`: `train`. "
                        f"Get: {self.split}.")
        
        with progress_bar() as pbar:
            eimage_pattern = os.path.join(
                self.root, "rain12", self.split, "no_rain", "*.png"
            )
            for eimage_path in pbar.track(
                glob.glob(eimage_pattern),
                description=f"[bright_yellow]Listing Rain12 {self.split} images"
            ):
                image_path        = eimage_path.replace("no_rain", "rain")
                custom_label_path = eimage_path.replace("no_rain", "annotations_custom")
                custom_label_path = custom_label_path.replace(".png", ".json")
                self.image_paths.append(image_path)
                self.eimage_paths.append(eimage_path)
                # self.label_paths.append(label_path)
                self.custom_label_paths.append(custom_label_path)
    
    def list_rain100_files(self):
        """List all `rain100` data."""
        if self.split not in ["test"]:
            console.log(f"Rain100 dataset only supports `split`: `test`. "
                        f"Get: {self.split}.")
        
        with progress_bar() as pbar:
            eimage_pattern = os.path.join(
                self.root, "rain100", self.split, "no_rain", "*.png"
            )
            for eimage_path in pbar.track(
                glob.glob(eimage_pattern),
                description=f"[bright_yellow]Listing Rain100 {self.split} images"
            ):
                image_path        = eimage_path.replace("no_rain", "rain")
                custom_label_path = eimage_path.replace("no_rain", "annotations_custom")
                custom_label_path = custom_label_path.replace(".jpg", ".json")
                self.image_paths.append(image_path)
                self.eimage_paths.append(eimage_path)
                # self.label_paths.append(label_path)
                self.custom_label_paths.append(custom_label_path)
    
    def list_rain100h_files(self):
        """List all `rain100h` data."""
        if self.split not in ["train", "test"]:
            console.log(f"Rain100h dataset only supports `split`: `train` "
                        f"or `test`. Get: {self.split}.")
        
        with progress_bar() as pbar:
            eimage_pattern = os.path.join(
                self.root, "rain100h", self.split, "no_rain", "*.png"
            )
            for eimage_path in pbar.track(
                glob.glob(eimage_pattern),
                description=f"[bright_yellow]Listing Rain100H {self.split} images"
            ):
                image_path        = eimage_path.replace("no_rain", "rain")
                custom_label_path = eimage_path.replace("no_rain", "annotations_custom")
                custom_label_path = custom_label_path.replace(".jpg", ".json")
                self.image_paths.append(image_path)
                self.eimage_paths.append(eimage_path)
                # self.label_paths.append(label_path)
                self.custom_label_paths.append(custom_label_path)
    
    def list_rain100l_files(self):
        """List all `rain100l` data."""
        if self.split not in ["train", "test"]:
            console.log(f"Rain100l dataset only supports `split`: `train` "
                        f"or `test`. Get: {self.split}.")
        
        with progress_bar() as pbar:
            eimage_pattern = os.path.join(
                self.root, "rain100h", self.split, "no_rain", "*.png"
            )
            for eimage_path in pbar.track(
                glob.glob(eimage_pattern),
                description=f"[bright_yellow]Listing Rain100L {self.split} images"
            ):
                image_path        = eimage_path.replace("no_rain", "rain")
                custom_label_path = eimage_path.replace("no_rain", "annotations_custom")
                custom_label_path = custom_label_path.replace(".jpg", ".json")
                self.image_paths.append(image_path)
                self.eimage_paths.append(eimage_path)
                # self.label_paths.append(label_path)
                self.custom_label_paths.append(custom_label_path)
    
    def list_rain800_files(self):
        """List all `rain800` data."""
        if self.split not in ["train", "val", "test"]:
            console.log(f"Rain800 dataset only supports `split`: `train`, "
                        f"`val`, or `test`. Get: {self.split}.")
        
        with progress_bar() as pbar:
            eimage_pattern = os.path.join(
                self.root, "rain800", self.split, "no_rain", "*.jpg"
            )
            for eimage_path in pbar.track(
                glob.glob(eimage_pattern),
                description=f"[bright_yellow]Listing Rain800 {self.split} images"
            ):
                image_path        = eimage_path.replace("no_rain", "rain")
                custom_label_path = eimage_path.replace("no_rain", "annotations_custom")
                custom_label_path = custom_label_path.replace(".jpg", ".json")
                self.image_paths.append(image_path)
                self.eimage_paths.append(eimage_path)
                # self.label_paths.append(label_path)
                self.custom_label_paths.append(custom_label_path)

    def list_rain1200_files(self):
        """List all `rain1200` data."""
        if self.split not in ["train", "val", "test"]:
            console.log(f"Rain1200 dataset only supports `split`: `train`, "
                        f"`val`, or `test`. Get: {self.split}.")
        
        with progress_bar() as pbar:
            if self.split == "train":
                eimage_pattern = os.path.join(
                    self.root, "rain1200", self.split, "no_rain", "*", "*.jpg"
                )
                for eimage_path in pbar.track(
                    glob.glob(eimage_pattern),
                    description=f"[bright_yellow]Listing Rain1200 {self.split} images"
                ):
                    image_path        = eimage_path.replace("no_rain", "rain")
                    custom_label_path = eimage_path.replace(
                        "no_rain", "annotations_custom"
                    )
                    custom_label_path = custom_label_path.replace(".jpg", ".json")
                    self.image_paths.append(image_path)
                    self.eimage_paths.append(eimage_path)
                    # self.label_paths.append(label_path)
                    self.custom_label_paths.append(custom_label_path)
            else:
                eimage_pattern = os.path.join(
                    self.root, "rain1200", self.split, "no_rain", "*.jpg"
                )
                for eimage_path in pbar.track(
                    glob.glob(eimage_pattern),
                    description=f"[bright_yellow]Listing Rain1200 {self.split} images"
                ):
                    image_path        = eimage_path.replace("no_rain", "rain")
                    custom_label_path = eimage_path.replace("no_rain", "customs")
                    custom_label_path = custom_label_path.replace(".jpg", ".json")
                    self.image_paths.append(image_path)
                    self.eimage_paths.append(eimage_path)
                    # self.label_paths.append(label_path)
                    self.custom_label_paths.append(custom_label_path)
    
    def list_rain1400_files(self):
        """List all `rain1400` data."""
        if self.split not in ["train", "test"]:
            console.log(f"Rain1400 dataset only supports `split`: `train` "
                        f"or `test`. Get: {self.split}.")
        
        with progress_bar() as pbar:
            eimage_pattern = os.path.join(
                self.root, "rain1400", self.split, "no_rain", "*.jpg"
            )
            for eimage_path in pbar.track(
                glob.glob(eimage_pattern),
                description=f"[bright_yellow]Listing Rain1400 {self.split} images"
            ):
                image_path        = eimage_path.replace("no_rain", "rain")
                custom_label_path = eimage_path.replace("no_rain", "annotations_custom")
                custom_label_path = custom_label_path.replace(".jpg", ".json")
                self.image_paths.append(image_path)
                self.eimage_paths.append(eimage_path)
                # self.label_paths.append(label_path)
                self.custom_label_paths.append(custom_label_path)
    
    def list_rain2800_files(self):
        """List all `rain2800` data."""
        if self.split not in ["test"]:
            console.log(f"Rain2800 dataset only supports `split`: `test`. "
                        f"Get: {self.split}.")
        
        with progress_bar() as pbar:
            eimage_pattern = os.path.join(
                self.root, "rain2800", self.split, "no_rain", "*.jpg"
            )
            for eimage_path in pbar.track(
                glob.glob(eimage_pattern),
                description=f"[bright_yellow]Listing Rain2800 {self.split} images"
            ):
                image_path        = eimage_path.replace("no_rain", "rain")
                custom_label_path = eimage_path.replace("no_rain", "annotations_custom")
                custom_label_path = custom_label_path.replace(".jpg", ".json")
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
				Target image file.
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
    

# MARK: - Rain13KDataModule

@DATAMODULES.register(name="rain13k")
class Rain13KDataModule(DataModule):
    """Rain13K DataModule."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        dataset_dir: str = os.path.join(datasets_dir, "rain13k"),
        name       : str = "rain13k",
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
        console.log(f"Setup [red]Rain13K[/red] datasets.")
        
        # NOTE: Assign train/val datasets for use in dataloaders
        if model_state in [None, ModelState.TRAINING]:
            self.train = Rain13K(
                root=self.dataset_dir, split="train", **self.dataset_kwargs
            )
            self.val = Rain13K(
                root=self.dataset_dir, split="val", **self.dataset_kwargs
            )
            self.class_labels = getattr(self.train, "class_labels", None)
            self.collate_fn   = getattr(self.train, "collate_fn",   None)
            
        # NOTE: Assign test datasets for use in dataloader(s)
        if model_state in [None, ModelState.TESTING]:
            self.test = Rain13K(
                root=self.dataset_dir, split="test", **self.dataset_kwargs
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
        "name": "rain13k",
        # Dataset's name.
        "subset": ["*"],
        # Sub-dataset to use. One of: ["rain12", "rain100", "rain100h",
        # "rain100l", "rain800", "rain1200", "rain1400", "rain2800"]. Can also be a
        # list to include multiple subsets. When `all`, `*`, or `None`, all alphas
        # will be included. Default: `*`.
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
    dm   = Rain13KDataModule(**cfgs)
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
