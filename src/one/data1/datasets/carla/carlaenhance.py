#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CARLA Enhance dataset and datamodule.
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
from one.core import Tasks
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
    "CARLAEnhance",
    "CARLAEnhanceDataModule"
]


# MARK: - Module

@DATASETS.register(name="carlaenhance")
class CARLAEnhance(ImageEnhancementDataset):
    """CARLA Enhance dataset consists of multiple datasets related to rain removal
    task.

    Args:
		weather (Tasks, optional):
            Weather to use. One of the values in `self.weathers`.
            Can also be a list to include multiple subsets. When `all`, `*`, or
            `None`, all subsets will be included. Default: `*`.
        time (Tasks, optional):
            Time of day to use. One of the values in `self.times`.
            Can also be a list to include multiple subsets. When `all`, `*`, or
            `None`, all subsets will be included. Default: `*`.
    """
    
    weathers = ["clear", "cloudy", "soft_rainy", "mid_rainy", "hard_rainy", "wet",
                "wet_cloudy"]
    times = ["noon", "sunset", "night"]
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        root            : str,
        weather         : Optional[Tasks]         = "*",
        time            : Optional[Tasks]         = "*",
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
        self.weather = weather
        self.time    = time
        
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
    def weather(self) -> list[str]:
        return self._weather

    @weather.setter
    def weather(self, weather: Optional[Tasks]):
        weather = [weather] if isinstance(weather, str) else weather
        if weather is None or "all" in weather or "*" in weather:
            weather = self.weathers
        self._weather = weather
    
    @property
    def time(self) -> list[str]:
        return self._time

    @time.setter
    def time(self, time: Optional[Tasks]):
        time = [time] if isinstance(time, str) else time
        if time is None or "all" in time or "*" in time:
            time = self.times
        self._time = time
    
    # MARK: List Files
    
    def list_files(self):
        """List image and label files."""
        # NOTE: List all files
        if "clear" in self.weather:
            self.list_weather_files("clear")
        if "cloudy" in self.weather:
            self.list_weather_files("cloudy")
        if "soft_rainy" in self.weather:
            self.list_weather_files("soft_rainy")
        if "mid_rainy" in self.weather:
            self.list_weather_files("mid_rainy")
        if "hard_rainy" in self.weather:
            self.list_weather_files("hard_rainy")
        if "wet" in self.weather:
            self.list_weather_files("wet")
        if "wet_cloudy" in self.weather:
            self.list_weather_files("wet_cloudy")
            
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
        
    def list_weather_files(self, weather: str):
        """List all files."""
        if weather not in self.weathers:
            return
        
        with progress_bar() as pbar:
            eimage_pattern = os.path.join(
                self.root, "enhance", "*", "*_default.png"
            )
            for eimage_path in pbar.track(
                glob.glob(eimage_pattern),
                description=f"[bright_yellow]Listing {weather} images"
            ):
                for t in self.time:
                    postfix           = f"{weather}_{t}"
                    image_path        = eimage_path.replace("default", postfix)
                    custom_label_path = eimage_path.replace("default", "annotations_custom")
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
    

@DATAMODULES.register(name="carlaenhance")
class CARLAEnhanceDataModule(DataModule):
    """CARLA Enhance DataModule."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        dataset_dir: str = os.path.join(data_dir, "carla"),
        name       : str = "carlaenhance",
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
        console.log(f"Setup [red]CARLA Enhance[/red] datasets.")
        
        # NOTE: Assign train/val datasets for use in dataloaders
        full_dataset = CARLAEnhance(
            root=self.dataset_dir, split="train", **self.dataset_kwargs
        )
        train_size = int(0.8 * len(full_dataset))
        val_size   = int((len(full_dataset) - train_size) / 2)
        test_size  = len(full_dataset) - train_size - val_size
        self.train, self.val, self.test = random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        if model_state in [None, ModelState.TRAINING]:
            self.class_labels = getattr(self.train, "class_labels", None)
            self.collate_fn   = getattr(self.train, "collate_fn", None)

        # NOTE: Assign test datasets for use in dataloader(s)
        if model_state in [None, ModelState.TESTING]:
            self.class_labels = getattr(self.test, "class_labels", None)
            self.collate_fn   = getattr(self.test, "collate_fn",  None)
        
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
        "name": "carlaenhance",
        # Dataset's name.
        "weather": ["*"],
        # Weather to use. One of: ["clear", "cloudy", "soft_rainy", "mid_rainy",
        # "hard_rainy", "wet", "wet_cloudy"]. Can also be a list to include multiple
        # subsets. When `all`, `*`, or `None`, all subsets will be included.
        # Default: `*`.
        "time": ["*"],
        # Time of day to use. One of: ["noon", "sunset", "night"]. Can also be a
        # list to include multiple subsets. When `all`, `*`, or `None`, all subsets
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
    dm   = CARLAEnhanceDataModule(**cfgs)
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
