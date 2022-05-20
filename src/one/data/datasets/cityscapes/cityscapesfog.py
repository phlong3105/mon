#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Foggy Cityscapes dataset and datamodule.
"""

from __future__ import annotations

import glob
import os
import random
from typing import Callable
from typing import Optional
from typing import Union

import matplotlib.pyplot as plt

from one.core import Augment_
from one.core import console
from one.core import DATAMODULES
from one.core import DATASETS
from one.core import Int3T
from one.core import Tasks
from one.core import unique
from one.core import VisionBackend
from one.data.data_class import ClassLabels
from one.data.data_class import ImageInfo
from one.data.data_class import VisionData
from one.data.datamodule import DataModule
from one.data.dataset import ImageEnhancementDataset
from one.data.label_handler import VisionDataHandler
from one.imgproc import show_images
from one.core import ModelState
from one.utils import datasets_dir

__all__ = [
    "CityscapesFog",
    "CityscapesFogDataModule"
]


# MARK: - Module

@DATASETS.register(name="cityscapesfog")
class CityscapesFog(ImageEnhancementDataset):
    """Cityscapes Fog dataset consists of multiple sub-datasets related to
    haze image enhancement task.

    Args:
        subset (Tasks, optional):
            Sub-dataset name. One of the values in `self.subsets`. Can also
            be a list to include multiple subsets. When `all`, `*`, or `None`,
            all subsets will be included. Default: `foggy`.
        extra (bool):
            Should use extra data? Those in the `train_extra` split are
            available for beta equal to `0.01`. Default: `False`.
        beta (float, list, optional):
            Additional information on the attenuation coefficient. One of the
            values in `self.betas`. Can also be a list to include multiple
            betas. When `all`, `*`, or `None`, all beta values will be
            included. Default: `0.01`.
    """

    subsets = ["foggy", "foggyDBF"]
    betas   = [0.005, 0.01, 0.02]
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        root            : str,
        subset          : Optional[Tasks]              = "foggy",
        split           : str                          = "train",
        extra           : bool                         = False,
        beta            : Optional[Union[float, list]] = 0.01,
        shape           : Int3T                        = (720, 1280, 3),
        caching_labels  : bool                         = False,
        caching_images  : bool                         = False,
        write_labels    : bool                         = False,
        fast_dev_run    : bool                         = False,
        load_augment    : Optional[dict]               = None,
        augment         : Optional[Augment_]           = None,
        vision_backend  : Optional[VisionBackend]      = None,
        transforms      : Optional[Callable]           = None,
        transform       : Optional[Callable]           = None,
        target_transform: Optional[Callable]           = None,
        *args, **kwargs
    ):
        self.subset = subset
        self.extra  = extra
        self.beta   = beta
        
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

    @property
    def beta(self) -> list[float]:
        return self._beta

    @beta.setter
    def beta(self, beta: Optional[Union[float, list]]):
        beta = beta if isinstance(beta, list) else [beta]
        if beta is None or "all" in beta or "*" in beta:
            beta = self.betas
        self._beta = beta
    
    # MARK: List Files
    
    def list_files(self):
        """List image and label files."""
        if self.split not in ["train", "val", "test"]:
            raise ValueError(f"Cityscapes Fog dataset only supports `split`: "
                             f"`train`, `val`, or `test`. Get: {self.split}.")
        
        # NOTE: List all image files
        for subset in self.subset:
            for beta in self.beta:
                image_pattern = os.path.join(
                    self.root, f"leftImg8bit_{subset}", self.split, "*", f"*_{beta}.png"
                )
                image_paths = glob.glob(image_pattern)
                image_paths = unique(image_paths)  # Remove all duplicates
                for image_path in image_paths:
                    postfix     = image_path[image_path.find("_foggy_beta"):]
                    eimage_path = image_path.replace(postfix, ".png")
                    eimage_path = eimage_path.replace(
                        f"leftImg8bit_{subset}", "leftImg8bit"
                    )
                    custom_label_path = image_path.replace(
                        f"leftImg8bit_{subset}", "leftImg8bit_{subset}_custom"
                    )
                    custom_label_path = custom_label_path.replace(".png", ".json")
                    self.image_paths.append(image_path)
                    self.eimage_paths.append(eimage_path)
                    # self.label_paths.append(label_path)
                    self.custom_label_paths.append(custom_label_path)
        
        if self.split == "train" and self.extra:
            image_pattern = os.path.join(
                self.root, "leftImg8bit_foggy", "train_extra", "*", "*.png"
            )
            image_paths = glob.glob(image_pattern)
            image_paths = unique(image_paths)  # Remove all duplicates
            for image_path in image_paths:
                postfix = image_path[image_path.find("_foggy_beta"):]
                eimage_path = image_path.replace(postfix, ".png")
                eimage_path = eimage_path.replace(
                    f"leftImg8bit_foggy", "leftImg8bit"
                )
                custom_label_path = image_path.replace(
                    f"leftImg8bit_foggy", f"leftImg8bit_foggy_custom"
                )
                custom_label_path = custom_label_path.replace(".png", ".json")
                self.image_paths.append(image_path)
                self.eimage_paths.append(eimage_path)
                # self.label_paths.append(label_path)
                self.custom_label_paths.append(custom_label_path)
                
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
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path        = os.path.join(current_dir, "cityscapes_class_labels.json")
        self.class_labels = ClassLabels.create_from_file(label_path=path)


@DATAMODULES.register(name="cityscapesfog")
class CityscapesFogDataModule(DataModule):
    """Cityscapes Fog DataModule."""
    
    # MARK: Magic Functions
    
    def __init__(
        self, 
        dataset_dir: str = os.path.join(datasets_dir, "cityscapes"),
        name       : str = "cityscapesfog",
        *args, **kwargs
    ):
        super().__init__(dataset_dir=dataset_dir, name=name, *args, **kwargs)
        self.dataset_cfg = kwargs
        
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
        console.log(f"Setup [red]Cityscapes Fog[/red] datasets.")
        
        # NOTE: Assign train/val datasets for use in dataloaders
        if model_state in [None, ModelState.TRAINING]:
            self.train = CityscapesFog(
                root=self.dataset_dir, split="train", **self.dataset_cfg
            )
            self.val = CityscapesFog(
                root=self.dataset_dir, split="val", **self.dataset_cfg
            )
            self.class_labels = getattr(self.test, "class_labels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
            
        # NOTE: Assign test datasets for use in dataloader(s)
        if model_state in [None, ModelState.TESTING]:
            self.test = CityscapesFog(
                root=self.dataset_dir, split="test", **self.dataset_cfg
            )
            self.class_labels = getattr(self.test, "class_labels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
            
        if self.class_labels is None:
            self.load_class_labels()

        self.summarize()

    def load_class_labels(self):
        """Load ClassLabels."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path        = os.path.join(current_dir, "cityscapes_class_labels.json")
        self.class_labels = ClassLabels.create_from_file(label_path=path)
        

# MARK: - Main

if __name__ == "__main__":
    # NOTE: Get DataModule
    cfgs = {
        "name": "cityscapesfog",
        # Dataset's name.
        "subset": "foggy",
        # Sub-dataset name. One of: ["foggy", "foggyDBF"]. Can also be a list
        # to include multiple subsets. When `all`, `*`, or `None`, all subsets will
        # be included. Default: `foggy`.
        "extra": False,
        # Should use extra data? Those in the `train_extra` split are available for
        # beta equal to `0.01`. Default: `False`.
        "beta": [0.005, 0.01, 0.02],
        # Additional information on the attenuation coefficient.
        # One of: [0.005, 0.01, 0.02]. Can also be a list to include multiple
        # betas. When `all`, `*`, or `None`, all beta values will be included.
        # Default: `0.01`.
        "shape": [1024, 2048, 3],
        # Image shape as [H, W, C]. This is compatible with OpenCV format.
        # This is also used to reshape the input data.
        "num_classes": 34,
        # Number of classes in the dataset.
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
            "mosaic": 0.5,
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
    dm   = CityscapesFogDataModule(**cfgs)
    dm.setup()
    # NOTE: Visualize labels
    if dm.class_labels:
        dm.class_labels.print()
    # NOTE: Visualize one sample
    data_iter            = iter(dm.val_dataloader)
    input, target, shape = next(data_iter)
    show_images(images=input,  nrow=2, denormalize=True)
    show_images(images=target, nrow=2, denormalize=True, figure_num=1)
    plt.show(block=True)
