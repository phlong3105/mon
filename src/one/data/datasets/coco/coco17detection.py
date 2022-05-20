#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""COCO 2017 dataset and datamodule.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Callable
from typing import Optional

from matplotlib import pyplot as plt

from one.core import Augment_
from one.core import console
from one.core import DATAMODULES
from one.core import DATASETS
from one.core import Int3T
from one.core import progress_bar
from one.core import VisionBackend
from one.data.data_class import ClassLabels
from one.data.data_class import VisionData
from one.data.datamodule import DataModule
from one.data.dataset import ObjectDetectionDataset
from one.data.label_handler import CocoDetectionLabelHandler
from one.data.label_handler import VisionDataHandler
from one.imgproc import box_cxcywh_norm_to_xyxy
from one.imgproc import draw_box
from one.imgproc import show_images
from one.io import is_image_file
from one.core import ModelState
from one.utils import datasets_dir

__all__ = [
    "COCO17Detection",
    "COCO17DetectionDataModule"
]


# MARK: - Module

@DATASETS.register(name="coco17detection")
class COCO17Detection(ObjectDetectionDataset):
    """COCO 2017 Detection Dataset. This is a multi-tasks dataset where for
    each image, several type of ground-truth labels are associated with.
    """

    # MARK: Magic Functions
    
    def __init__(
        self,
        root            : str,
        split           : str                     = "train",
        shape           : Int3T                   = (640, 640, 3),
        batch_size      : int                     = 1,
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
            batch_size       = batch_size,
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
        if self.split not in ["train", "val"]:
            raise ValueError(f"COCO17 Detection dataset only supports "
                             f"`split`: `train` or `val`. Get: {self.split}.")
    
        # NOTE: List image files
        with progress_bar() as pbar:
            image_pattern    = os.path.join(self.root, f"coco17", f"{self.split}2017", "*.jpg")
            self.image_paths = [
                p for p in pbar.track(
                    glob.glob(image_pattern),
                    description=f"[bright_yellow]Listing {self.split} images"
                )
                if is_image_file(p)
            ]

        # NOTE: List label files
        self.label_paths = os.path.join(self.root, "coco17", "annotations", f"instances_{self.split}2017.json")
        
        with progress_bar() as pbar:
            custom_label_dir  = os.path.join(self.root, "coco17", "annotations", "instances_custom")
            custom_label_path = [str(Path(p).name) 		    for p in self.image_paths]
            custom_label_path = [p.replace(".jpg", ".json") for p in custom_label_path]
            self.custom_label_paths = [
                os.path.join(custom_label_dir, p) for p in
                pbar.track(
                    custom_label_path,
                    description=f"[bright_yellow]Listing {self.split} custom labels"
                )
            ]
        
        # NOTE: Assertion
        if isinstance(self.label_paths, list):
            if (
                len(self.image_paths) <= 0 or
                len(self.image_paths) != len(self.label_paths)
            ):
                raise ValueError(
                    f"Number of images != Number of labels: "
                    f"{len(self.image_paths)} != {len(self.label_paths)}."
                )
        console.log(f"Number of images: {len(self.image_paths)}.")
        
    # MARK: Load Data
    
    def load_label(
        self,
        image_path		 : str,
        label_path	     : str,
        custom_label_path: Optional[str] = None
    ) -> VisionData:
        """Load all labels from a raw label `file`.

        Args:
            image_path (str):
                Image file.
            label_path (str):
                Label file.
            custom_label_path (str, optional):
                Custom label file. Default: `None`.

        Returns:
            data (VisionData):
                `VisionData` object.
        """
        # NOTE: If we have custom labels
        if custom_label_path and os.path.isfile(custom_label_path):
            return VisionDataHandler().load_from_file(
                image_path=image_path, label_path=custom_label_path
            )
        return VisionData()
    
    def load_labels(self, image_paths: list[str], label_path: str) -> dict[str, VisionData]:
        """Load a list of `VisualData` objects from a .json file.

        Args:
            image_paths (str):
                List of image paths.
            label_path (str):
                Label file.
                
        Return:
            data (dict[str, VisionData]):
                Dictionary of `VisualData` objects.
        """
        return CocoDetectionLabelHandler().load_from_file(
            image_paths=image_paths, label_path=label_path
        )
    
    def load_class_labels(self):
        """Load ClassLabels."""
        path = os.path.join(self.root, "coco17", "80_class_labels.json")
        if not os.path.isfile(path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, "coco17_80_class_labels.json")
        self.class_labels = ClassLabels.create_from_file(label_path=path)
    

@DATAMODULES.register(name="coco17detection")
class COCO17DetectionDataModule(DataModule):
    """COCO 2017 Detection DataModule."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        dataset_dir: str = os.path.join(datasets_dir, "coco"),
        name       : str = "coco17detection",
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
        console.log(f"Setup [red]COCO 2017 Detection[/red] datasets.")
        
        # NOTE: Assign train/val datasets for use in dataloaders
        if model_state in [None, ModelState.TRAINING]:
            self.train        = COCO17Detection(root=self.dataset_dir, split="train", **self.dataset_kwargs)
            self.val          = COCO17Detection(root=self.dataset_dir, split="val",   **self.dataset_kwargs)
            self.class_labels = getattr(self.train, "class_labels", None)
            self.collate_fn   = getattr(self.train, "collate_fn",   None)

        # NOTE: Assign test datasets for use in dataloader(s)
        if model_state in [None, ModelState.TESTING]:
            self.test         = COCO17Detection(root=self.dataset_dir, split="val", **self.dataset_kwargs)
            self.class_labels = getattr(self.test, "class_labels", None)
            self.collate_fn   = getattr(self.test, "collate_fn",   None)
        
        if self.class_labels is None:
            self.load_class_labels()

        self.summarize()
        
    def load_class_labels(self):
        """Load ClassLabels."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path 	    = os.path.join(current_dir, "coco17_80_class_labels.json")
        self.class_labels = ClassLabels.create_from_file(label_path=path)


# MARK: - Main

if __name__ == "__main__":
    # NOTE: Get DataModule
    cfgs = {
        "name": "coco2017detection",
        # Dataset's name.
        "shape": [1536, 1536, 3],
        # Image shape as [H, W, C]. This is compatible with OpenCV format.
        # This is also used to reshape the input data.
        "num_classes": 3,
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
            "rect"  : False,
            "stride": 32,
            "pad"   : 0,
        },
        # Augmented loading policy.
        "augment": {
            "name": "image_box_augment",
            # Name of the augmentation policy.
            "policy": "scratch",
            # Augmentation policy. One of: [`scratch`]. Default: `scratch`.
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
    dm   = COCO17DetectionDataModule(**cfgs)
    dm.setup()
    # NOTE: Visualize labels
    if dm.class_labels:
        dm.class_labels.print()
    # NOTE: Visualize an iteration
    data_iter            = iter(dm.val_dataloader)
    input, target, shape = next(data_iter)
    
    drawings = []
    for i, img in enumerate(input):
        chw       = img.shape
        l         = target[target[:, 0] == i]
        l[:, 2:6] = box_cxcywh_norm_to_xyxy(l[:, 2:6], chw[1], chw[2])
        drawing   = draw_box(img, l, dm.class_labels.colors(), 5)
        drawings.append(drawing)
    
    show_images(images=drawings, nrow=1, denormalize=False)
    plt.show(block=True)
