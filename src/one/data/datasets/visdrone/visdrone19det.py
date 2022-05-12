#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""VisDrone2019 measurement dataset and datamodule.
"""

from __future__ import annotations

import glob
import os
import random
from typing import Callable
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

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
from one.imgproc import box_xywh_to_cxcywh_norm
from one.imgproc import box_xywh_to_xyxy
from one.imgproc import compute_box_area
from one.imgproc import draw_box
from one.imgproc import show_images
from one.io import is_image_file
from one.io import is_txt_file
from one.core import ModelState
from one.utils import datasets_dir

__all__ = [
    "VisDrone19Detection",
    "VisDrone19DetectionDataModule"
]


"""VisDrone measurement dataset hierarchy:

datasets
|__ visdrone
|   |__ detection2019
|   |   |__ train
|   |   |   |__ images
|   |   |   |__ annotations
|   |   |   |__ annotations_coco
|   |   |   |__ annotations_yolo
|   |   |
|   |   |__ val
|   |   |   |__ images
|   |   |   |__ annotations
|   |   |   |__ annotations_coco
|   |   |   |__ annotations_yolo
|   |   |
|   |   |__ test
|   |   |   |__ images
|   |   |   |__ annotations
|   |   |   |__ annotations_coco
|   |   |   |__ annotations_yolo
|   |   |
|   |   |__ testchallenge
|   |   |   |__ images
|   |   |
|   |   |__ toolkits
|   |__ ..
|
|__ ...
"""

"""VisDrone measurement label format:

<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,
<truncation>,<occlusion>

Where:
    <bbox_left>      : Fx coordinate of the top-left corner of the predicted
                       bounding box.
    <bbox_top>       : Fy coordinate of the top-left corner of the predicted
                       object bounding box.
    <bbox_width>     : Width in pixels of the predicted object bounding
                       box.
    <bbox_height>    : Height in pixels of the predicted object bounding
                       box.
    <score>          : Fscore in the DETECTION result file indicates the
                       confidence of the predicted bounding box enclosing an
                       object measurement.Fscore in GROUNDTRUTH file is set to
                       1 or 0. 1 indicates the bounding box is considered in
                       evaluation, while 0 indicates the bounding box will be
                       ignored.
    <object_category>: Object category indicates the type of annotated
                       object, (i.e., ignored regions (0), pedestrian (1),
                       people (2), bicycle (3), car (4), van (5), truck (6),
                       tricycle (7), awning-tricycle (8), bus (9), motor (
                       10), others (11))
    <truncation>     : Fscore in the DETECTION result file should be set to
                       the constant -1. Fscore in the GROUNDTRUTH file
                       indicates the degree of object parts appears outside a
                       frame (i.e., no truncation = 0 (truncation ratio 0%),
                       and partial truncation = 1 (truncation ratio 1% ∼ 50%)).
    <occlusion>      : Fscore in the DETECTION result file should be set to
                       the constant -1. Fscore in the GROUNDTRUTH file
                       indicates the fraction of objects being occluded (
                       i.e., no occlusion = 0 (occlusion ratio 0%),
                       partial occlusion = 1(occlusion ratio 1% ∼ 50%),
                       and heavy occlusion = 2 (occlusion ratio 50% ~ 100%)).

Examples:
    - For example for `img1.jpg` you will be created `img1.txt` containing:
        684,8,273,116,0,0,0,0
        406,119,265,70,0,0,0,0
        255,22,119,128,0,0,0,0
"""


# MARK: - VisDrone19Detection

@DATASETS.register(name="visdrone19det")
class VisDrone19Detection(ObjectDetectionDataset):
    """VisDrone 2019 dataset consists of multiple sub-datasets captured by
    various drone-mounted cameras, covering a wide range of aspects including
    location (taken from 14 different cities separated by thousands of
    kilometers in China), environment (urban and country), objects (pedestrian,
    vehicles, bicycles, etc.), and density (sparse and crowded scenes).
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
        
        # NOTE: Load class_labels
        class_labels_path = os.path.join(self.root, f"class_labels.json")
        if not os.path.isfile(class_labels_path):
            curr_dir        = os.path.dirname(os.path.abspath(__file__))
            class_labels_path = os.path.join(
                curr_dir, f"visdrone19_class_labels.json"
            )
        self.class_labels = ClassLabels.create_from_file(
            label_path=class_labels_path
        )
        
    # MARK: List Files
    
    def list_files(self):
        """List image and label files."""
        if self.split not in ["train", "val"]:
            raise ValueError(f"VisDrone 2019 Detection dataset only supports "
                             f"`split`: `train`, `val`. Get: {self.split}.")
        
        # NOTE: List image files
        with progress_bar() as pbar:
            image_pattern    = os.path.join(self.root, f"visdrone19det", self.split, "images", "*.jpg")
            self.image_paths = [
                p for p in pbar.track(
                    glob.glob(image_pattern),
                    description=f"[bright_yellow]Listing {self.split} images"
                )
                if is_image_file(p)
            ]
       
        # NOTE: List label files
        with progress_bar() as pbar:
            label_paths      = [p.replace("images", "annotations") for p in self.image_paths]
            label_paths      = [p.replace(".jpg", ".txt")          for p in label_paths]
            self.label_paths = [
                p for p in pbar.track(label_paths, description=f"[bright_yellow]Listing {self.split} labels")
                if is_txt_file(p)
            ]
        
        with progress_bar() as pbar:
            custom_label_paths      = [p.replace("images", "annotations_custom") for p in self.image_paths]
            self.custom_label_paths = [
                p.replace(".jpg", ".json") for p in
                pbar.track(custom_label_paths, description=f"[bright_yellow]Listing {self.split} custom labels")
            ]
        
        # NOTE: fast_dev_run, select only a subset of images
        if self.fast_dev_run:
            indices = [random.randint(0, len(self.image_paths) - 1) for _ in range(self.batch_size)]
            self.image_paths        = [self.image_paths[i]        for i in indices]
            self.label_paths        = [self.label_paths[i]        for i in indices]
            self.custom_label_paths = [self.custom_label_paths[i] for i in indices]
        
        # NOTE: Assertion
        if isinstance(self.label_paths, list):
            if (
                len(self.image_paths) <= 0
                and len(self.image_paths) != len(self.label_paths)
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
        label_path		 : str,
        custom_label_path: Optional[str] = None
    ) -> VisionData:
        """Load all labels from a raw label `file`.

        Args:
            image_path (str):
                Image file.
            label_path (str):
                Label file.
            custom_label_path (str, optional):
                Custom label filepath. Default: `None`.
                
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
        
        # NOTE: Parse all annotations
        with open(label_path, "r") as file_in:
            labels = [x.replace(",", " ") for x in file_in.read().splitlines()]
            labels = np.array([x.split() for x in labels], dtype=np.float32)
        
        # VisDrone format:
        #      0           1          2            3           4
        # <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,
        #      5                6          7
        # <object_category>,<truncation>,<occlusion>
        objs = []
        for i, l in enumerate(labels):
            obj            = ObjectAnnotation()
            obj.class_id   = int(l[5])
            obj.box       = box_xywh_to_cxcywh_norm(l[0:4], shape0[0], shape0[1])
            obj.confidence = l[4]
            xyxy           = box_xywh_to_xyxy(l[0:4])
            obj.area       = compute_box_area(xyxy)
            obj.truncation = l[6]
            obj.occlusion  = l[7]
            objs.append(obj)
        return VisionData(image_info=image_info, objects=objs)
    
    def load_labels(self, image_paths: list[str], label_path: str) -> dict[str, VisionData]:
        """Load all labels from one label file.

        Args:
            image_paths (list[str]):
                List of image filepaths.
            label_path (str):
                Label filepath.
        
        Returns:
            data (dict):
                Dictionary of `VisionData` objects.
        """
        pass
    
    def load_class_labels(self):
        """Load ClassLabels."""
        path = os.path.join(self.root, "visdrone19det", "class_labels.json")
        if not os.path.isfile(path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path 	    = os.path.join(current_dir, "visdrone19_class_labels.json")
        self.class_labels = ClassLabels.create_from_file(label_path=path)
        

# MARK: - VisDrone19DetectionDataModule

@DATAMODULES.register(name="visdrone19det")
class VisDrone19DetectionDataModule(DataModule):
    """VisDrone 2019 Detection DataModule."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        dataset_dir: str = os.path.join(datasets_dir, "visdrone"),
        name       : str = "visdrone19det",
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
        console.log(f"Setup [red]VisDrone 2019 Detection[/red] datasets.")
        
        # NOTE: Assign train/val datasets for use in dataloaders
        if phase in [None, ModelState.TRAINING]:
            self.train = VisDrone19Detection(
                root=self.dataset_dir, split="train", **self.dataset_cfg
            )
            self.val = VisDrone19Detection(
                root=self.dataset_dir, split="val",  **self.dataset_cfg
            )
            self.class_labels = getattr(self.train, "class_labels", None)
            self.collate_fn  = getattr(self.train, "collate_fn",  None)

        # NOTE: Assign test datasets for use in dataloader(s)
        if phase in [None, Phase.TESTING]:
            self.test = VisDrone19Detection(
                root=self.dataset_dir, split="testdev", **self.dataset_cfg
            )
            self.class_labels = getattr(self.test, "class_labels", None)
            self.collate_fn   = getattr(self.test, "collate_fn",  None)
        
        if self.class_labels is None:
            self.load_class_labels()

        self.summarize()
        
    def load_class_labels(self):
        """Load ClassLabels."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path 	    = os.path.join(current_dir, "visdrone19_class_labels.json")
        self.class_labels = ClassLabels.create_from_file(label_path=path)
    

# MARK: - Main

if __name__ == "__main__":
    # NOTE: Get DataModule
    cfgs = {
        "name": "visdrone19det",
        # Datasets' name.
        "shape": [1536, 1536, 3],
        # Image shape as [H, W, C]. This is compatible with OpenCV format.
        "num_classes": 12,
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
    dm   = VisDrone19DetectionDataModule(**cfgs)
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
