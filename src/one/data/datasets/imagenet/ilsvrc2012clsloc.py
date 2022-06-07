#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ImageNet dataset and datamodule.
"""

from __future__ import annotations

import glob
import os
import random
from typing import Callable
from typing import Optional

import matplotlib.pyplot as plt
import torchvision

from one.core import Augment_
from one.core import console
from one.core import DATAMODULES
from one.core import DATASETS
from one.core import Int3T
from one.core import progress_bar
from one.core import VisionBackend
from one.data.data_class import ClassLabels
from one.data.data_class import ImageInfo
from one.data.data_class import VisionData
from one.data.datamodule import DataModule
from one.data.dataset import ImageClassificationDataset
from one.data.label_handler import PascalLabelHandler
from one.data.label_handler import VisionDataHandler
from one.imgproc import show_images
from one.io import is_image_file
from one.io import is_xml_file
from one.core import ModelState
from one.utils import data_dir

__all__ = [
    "ILSVRC2012ClsLoc",
    "ILSVRC2012ClsLocDataModule",
]


# MARK: - Module

@DATASETS.register(name="ilsvrc2012clsloc")
class ILSVRC2012ClsLoc(ImageClassificationDataset):

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
        """List image and label files.

        Todos:
            - Look for image and label files in `split` directory.
            - We should look for our custom label files first.
            - If none is found, proceed to listing the images and raw label
              files.
            - After this method, these following attributes MUST be defined:
              `image_files`, `eimage_files`, `label_files`,
              `has_custom_labels`, `class_labels`.
        """
        # NOTE: List images files
        with progress_bar() as pbar:
            if self.split == "train":
                image_pattern = os.path.join(
                    self.root, "ilsvrc2012_cls_loc", f"{self.split}", "images",
                    "*", "*.JPEG"
                )
            else:
                image_pattern = os.path.join(
                    self.root, "ilsvrc2012_cls_loc", f"{self.split}", "images",
                    "*.JPEG"
                )
            self.image_paths  = [
                p for p in pbar.track(
                    glob.glob(image_pattern), description=f"[bright_yellow]Listing {self.split} images"
                )
                if is_image_file(p)
            ]
        
        # NOTE: fast_dev_run, select only a subset of images
        if self.fast_dev_run:
            indices = [random.randint(0, len(self.image_paths) - 1)
                       for _ in range(self.batch_size)]
            self.image_paths = [self.image_paths[i] for i in indices]
        
        # NOTE: List label files
        with progress_bar() as pbar:
            if self.split == "train":
                label_paths = [
                    os.path.basename(os.path.dirname(p)).split(".")[0] for p in
                    pbar.track(self.image_paths, description=f"Listing {self.split} labels" )
                ]
            else:
                label_paths = [p.replace(".JPEG", ".xml") for p in self.image_paths]
                label_paths = [p.replace("images", "annotations") for p in label_paths]
                label_paths = [
                    p for p in
                    pbar.track(
                        label_paths, description=f"[bright_yellow]Listing {self.split} labels"
                    )
                    if is_xml_file(p)
                ]
            self.label_paths = label_paths

        with progress_bar() as pbar:
            custom_label_path = [p.replace(".JPEG", ".json") for p in self.image_paths]
            custom_label_path = [
                p.replace("images", "annotations_custom") for p in
                pbar.track(
                    custom_label_path,
                    description=f"[bright_yellow]Listing {self.split} custom labels"
                )
            ]
        self.custom_label_paths = custom_label_path

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
        image_path       : str,
        label_path       : str,
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
        
        # NOTE: Load labels
        if self.split == "train":
            class_id = (
                self.class_labels.get_id(key="synset_id", value=label_path)
                if isinstance(self.class_labels, ClassLabels) else label_path
            )
            return VisionData(
                image_info       = ImageInfo.from_file(image_path=image_path),
                image_annotation = class_id
            )
        else:
            label = PascalLabelHandler().load_from_file(
                image_path=image_path, label_path=label_path
            )
            class_id = label.objects[0].class_id
            class_id = (
                self.class_labels.get_id(key="synset_id", value=class_id)
                if isinstance(self.class_labels, ClassLabels) else class_id
            )
            label.image_annotation = class_id
            return label

    def load_labels(
        self,
        image_paths      : list[str],
        label_path       : str,
        custom_label_path: Optional[str] = None
    ) -> dict[str, VisionData]:
        """Load a list of `VisualData` objects from a .json file.

		Args:
			image_paths (str):
				List of image paths.
			label_path (str):
				Label file.
			custom_label_path (str, optional):
                Custom label file. Default: `None`.
                
		Return:
			data (dict[str, VisionData]):
				A dictionary of `VisualData` objects.
		"""
        pass

    def load_class_labels(self):
        """Load ClassLabels."""
        path = os.path.join(self.root, "ilsvrc2012_cls_loc", "class_labels.json")
        if not os.path.isfile(path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, "imagenet_1k_class_labels.json")
        self.class_labels = ClassLabels.create_from_file(label_path=path)
    

@DATAMODULES.register(name="ilsvrc2012clsloc")
class ILSVRC2012ClsLocDataModule(DataModule):
    """ImageNet 2012 Classification and Localization DataModule."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        dataset_dir: str = os.path.join(data_dir, "imagenet"),
        name       : str = "ilsvrc2012clsloc",
        *args, **kwargs
    ):
        super().__init__(dataset_dir=dataset_dir, name=name, *args, **kwargs)

        self.transform = torchvision.transforms.Compose([
            # torchvision.transforms.Resize(size=self.shape[0:2]),
            torchvision.transforms.ToTensor()
        ])
    
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
        console.log(f"Setup [red]ILSVRC 2012 ClsLoc[/red] datasets.")

        # NOTE: Assign train/val datasets for use in dataloaders
        if model_state in [None, ModelState.TRAINING]:
            self.train = ILSVRC2012ClsLoc(
               self.dataset_dir, split="train", transform=self.transform
            )
            self.val = ILSVRC2012ClsLoc(
                self.dataset_dir, split="val", transform=self.transform
            )
            self.class_labels = getattr(self.train, "class_labels", None)
            self.collate_fn  = getattr(self.train, "collate_fn",  None)
            
        # NOTE: Assign test datasets for use in dataloader(s)
        if model_state in [None, ModelState.TESTING]:
            self.test = None
            #self.test = ImageNet(
            #    self.dataset_dir, split="val",  transform=self.transform
            #)
            self.class_labels = getattr(self.test, "class_labels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
            
        if self.class_labels is None:
            self.load_class_labels()

        self.summarize()
        
    def load_class_labels(self):
        """Load ClassLabels."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path        = os.path.join(current_dir, "imagenet_1k_class_labels.json")
        self.class_labels = ClassLabels.create_from_file(label_path=path)
        

# MARK: - Main

if __name__ == "__main__":
    # NOTE: Get DataModule
    cfg = {
        "name": "ilsvrc2012clsloc",
        # Dataset's name.
        "shape": [256, 256, 3],
        # Image shape as [H, W, C]. This is compatible with OpenCV format.
        # This is also used to reshape the input data.
        "num_classes": 1000,
        # Number of classes in the dataset.
        "batch_size": 32,
        # Number of samples in one forward & backward pass.
        "caching_labels": False,
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
            "name": "image_auto_augment",
            # Name of the augmentation policy.
            "policy": "imagenet",
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
    dm  = ILSVRC2012ClsLocDataModule(**cfg)
    dm.setup()
    # NOTE: Visualize labels
    if dm.class_labels:
        dm.class_labels.print()
    # NOTE: Visualize an iteration
    data_iter         = iter(dm.train_dataloader)
    data              = next(data_iter)
    input, targets, _ = data
    console.log(targets)
    show_images(images=input, denormalize=True)
    plt.show(block=True)
