#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A2I2-Haze Detection dataset and datamodule.
"""

from __future__ import annotations

import glob
import os
import random
from typing import Callable
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split

from one.core import Augment_
from one.core import console
from one.core import DATAMODULES
from one.core import DATASETS
from one.core import Int3T
from one.core import progress_bar
from one.core import VisionBackend
from one.data.data_class import ClassLabels
from one.data.data_class import ImageInfo
from one.data.data_class import ObjectAnnotation
from one.data.data_class import VisionData
from one.data.datamodule import DataModule
from one.data.dataset import ObjectDetectionDataset
from one.data.label_handler import VisionDataHandler
from one.imgproc import box_cxcywh_norm_to_xyxy
from one.imgproc import box_xyxy_to_cxcywh_norm
from one.imgproc import compute_box_area
from one.imgproc import draw_box
from one.imgproc import show_images
from one.io import is_image_file
from one.io import is_txt_file
from one.core import ModelState
from one.utils import datasets_dir

__all__ = [
    "A2I2HazeDetection",
    "A2I2HazeDetectionDataModule",
]


# MARK: - A2I2HazeDetection

@DATASETS.register(name="a2i2hazedet")
class A2I2HazeDetection(ObjectDetectionDataset):
    """We provide the A2I2-Haze dataset for image de-hazing."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        root            : str,
        split           : str                     = "train",
        shape           : Int3T                   = (720, 1280, 3),
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
        if self.split not in ["train", "dryrun"]:
            raise ValueError(f"A2I2-Haze Detection dataset only supports "
                             f"`split`: `train`. Get: {self.split}.")
        
        # NOTE: List label & image files
        if self.split == "train":
            with progress_bar() as pbar:
                label_pattern = os.path.join(
                    self.root, "train", "clean_images_labels", "*.txt"
                )
                label_paths = [
                    p for p in pbar.track(
                        glob.glob(label_pattern),
                        description=f"[bright_yellow]Listing {self.split} labels"
                    )
                    if is_txt_file(p)
                ]
                
                image_paths = [p.replace("_labels", "")  for p in label_paths]
                image_paths = [p.replace(".txt", ".jpg") for p in image_paths]
                for i in pbar.track(
                    range(len(image_paths)),
                    description=f"[bright_yellow]Listing {self.split} images"
                ):
                    if is_image_file(image_paths[i]):
                        self.image_paths.append(image_paths[i])
                        self.label_paths.append(label_paths[i])
        
        if self.split == "dryrun":
            with progress_bar() as pbar:
                label_pattern = os.path.join(
                    self.root, "dryrun", "labels", "*.txt"
                )
                label_paths = [
                    p for p in pbar.track(
                        glob.glob(label_pattern),
                        description=f"[bright_yellow]Listing {self.split} labels"
                    )
                    if is_txt_file(p)
                ]
                
                image_paths = [p.replace("labels", "images") for p in label_paths]
                image_paths = [p.replace(".txt", ".jpg")     for p in image_paths]
                for i in pbar.track(
                    range(len(image_paths)),
                    description=f"[bright_yellow]Listing {self.split} images"
                ):
                    if is_image_file(image_paths[i]):
                        self.image_paths.append(image_paths[i])
                        self.label_paths.append(label_paths[i])

        # NOTE: List custom labels
        with progress_bar() as pbar:
            label_paths = [p.replace("_labels", "_annotations_custom")
                           for p in self.label_paths]
            label_paths = [
                p.replace(".txr", ".json") for p in
                pbar.track(
                    label_paths,
                    description=f"[bright_yellow]Listing {self.split} custom labels"
                )
            ]
            self.custom_label_paths = label_paths
        
        # NOTE: fast_dev_run, select only a subset of images
        if self.fast_dev_run:
            indices = [random.randint(0, len(self.image_paths) - 1)
                       for _ in range(self.batch_size)]
            self.image_paths = [self.image_paths[i] for i in indices]

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
        
        # NOTE: Parse image info
        image_info = ImageInfo.from_file(image_path=image_path)
        shape0     = image_info.shape0
        
        # NOTE: Load content from file
        if is_txt_file(path=label_path):
            with open(label_path, "r") as f:
                labels = [x.split() for x in f.read().splitlines()]
       
        for i, l in enumerate(labels):
            class_id = l[0]
            class_id = (
                self.class_labels.get_id(key="name", value=class_id)
                if isinstance(self.class_labels, ClassLabels) else class_id
            )
            labels[i][0] = class_id
        labels = np.array(labels, np.float32)

        # NOTE: Parse all annotations
        objs = []
        for i, l in enumerate(labels):
            class_id        = int(l[0])
            box_xyxy        = l[1:5]
            box_cxcywh_norm = box_xyxy_to_cxcywh_norm(box_xyxy, shape0[0], shape0[1])
            confidence      = 0.0
            objs.append(
                ObjectAnnotation(
                    class_id   = class_id,
                    box        = box_cxcywh_norm,
                    area       = compute_box_area(box_xyxy),
                    confidence = confidence,
                )
            )
        return VisionData(image_info=image_info, objects=objs)
    
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
        pass
    
    def load_class_labels(self):
        """Load ClassLabels."""
        path = os.path.join(self.root, "class_labels.json")
        if not os.path.isfile(path):
            current_dir   = os.path.dirname(os.path.abspath(__file__))
            path          = os.path.join(current_dir, "../../../../../onedetection/src/onedetection/datasets/a2i2haze/a2i2_class_labels.json")
        self.class_labels = ClassLabels.create_from_file(label_path=path)
        

# MARK: - A2I2HazeDetection

@DATAMODULES.register(name="a2i2hazedet")
class A2I2HazeDetectionDataModule(DataModule):
    """A2I2-Haze Detection DataModule."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        dataset_dir: str = os.path.join(datasets_dir, "a2i2haze"),
        name       : str = "a2i2hazedet",
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
            phase (ModelState, optional):
                ModelState to use: [None, ModelState.TRAINING, ModelState.TESTING].
                Set to "None" to setup all train, val, and test data.
                Default: `None`.
        """
        console.log(f"Setup [red]A2I2-Haze Detection[/red] datasets.")
        
        # NOTE: Assign train/val datasets for use in dataloaders
        if phase in [None, ModelState.TRAINING]:
            full_dataset = A2I2HazeDetection(
                root=self.dataset_dir, split="train", **self.dataset_kwargs
            )
            train_size = int(0.8 * len(full_dataset))
            val_size   = len(full_dataset) - train_size
            self.train, self.val = random_split(
                full_dataset, [train_size, val_size]
            )
            self.class_labels = getattr(full_dataset, "class_labels", None)
            self.collate_fn   = getattr(full_dataset, "collate_fn",   None)
        
        if phase in [None, Phase.TESTING]:
            self.test = A2I2HazeDetection(
                root=self.dataset_dir, split="dryrun", **self.dataset_kwargs
            )
            self.class_labels = getattr(self.test, "class_labels", None)
            self.collate_fn   = getattr(self.test, "collate_fn",   None)
        
        if self.class_labels is None:
            self.load_class_labels()
        
        self.summarize()
        
    def load_class_labels(self):
        current_dir       = os.path.dirname(os.path.abspath(__file__))
        path              = os.path.join(current_dir, "../../../../../onedetection/src/onedetection/datasets/a2i2haze/a2i2_class_labels.json")
        self.class_labels = ClassLabels.create_from_file(label_path=path)
        

# MARK: - Main

if __name__ == "__main__":
    # NOTE: Get DataModule
    cfgs = {
        "name": "a2i2hazedet",
        # Dataset's name.
        "shape": [512, 512, 3],
        # Image shape as [H, W, C]. This is compatible with OpenCV format.
        "num_classes": 1,
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
            "mosaic": 0.0,
            "mixup" : 0.0,
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
    dm   = A2I2HazeDetectionDataModule(**cfgs)
    dm.setup()
    # NOTE: Visualize labels
    if dm.class_labels:
        dm.class_labels.print()
    # NOTE: Visualize an iteration
    data_iter            = iter(dm.train_dataloader)
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