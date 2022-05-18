#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Cityscapes semantic segmentation dataset and datamodule.
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
from one.core import unique
from one.core import VisionBackend
from one.data.data_class import ClassLabels
from one.data.data_class import ImageInfo
from one.data.data_class import VisionData
from one.data.datamodule import DataModule
from one.data.dataset import SemanticSegmentationDataset
from one.data.label_handler import CityscapesLabelHandler
from one.data.label_handler import VisionDataHandler
from one.imgproc import show_images
from one.io import is_image_file
from one.core import ModelState
from one.utils import datasets_dir

__all__ = [
    "CityscapesSemantic",
    "CityscapesSemanticDataModule"
]


# MARK: - CityscapesSemantic

@DATASETS.register(name="cityscapessemantic")
class CityscapesSemantic(SemanticSegmentationDataset):
    """FCityscapes Semantic dataset consists of multiple sub-datasets
    related to semantic segmentation task.

    Args:
        quality (str):
		    Quality of the semantic segmentation mask to use. One of the
		    values in `self.qualities`. Default: `gtFine`.
        encoding (str):
            Format to use when creating the semantic segmentation mask.
            One of the values in `self.encodings`. Default: `id`.
        extra (bool):
            Should use extra data? Those in the `train_extra` split are only
            available for `quality=gtCoarse`. Default: `False`.
    """

    qualities = ["gtFine", "gtCoarse"]
    encodings = ["id", "trainId", "catId", "color"]
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        root            : str,
        quality         : str                     = "gtFine",
        split           : str                     = "train",
        encoding        : str                     = "id",
        extra           : bool                    = False,
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
        self.quality = quality
        self.extra   = extra
        if quality not in self.qualities:
            raise ValueError(f"Cityscapes Semantic dataset does not supports "
                             f"`quality`: `{quality}`.")
        if encoding not in self.encodings:
            raise ValueError(f"Cityscapes Semantic dataset does not supports "
                             f"`encoding`: `{encoding}`.")

        # NOTE: Load class_labels
        path = os.path.join(root, quality, f"class_labels.json")
        if not os.path.isfile(path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, f"cityscapes_class_labels.json")
        class_labels = ClassLabels.create_from_file(label_path=path)
        
        super().__init__(
            root             = root,
            split            = split,
            class_labels= class_labels,
            encoding         = encoding,
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
        # NOTE: Flag some warnings
        if self.quality == "gtCoarse":
            if self.split not in ["train", "train_extra", "val"]:
                raise ValueError(
                    f"Cityscapes Semantic dataset only supports `split`: "
                    f"`train` or `val`. Get: {self.split}."
                )
        else:
            if self.split not in ["train", "val", "test"]:
                raise ValueError(
                    f"Cityscapes Semantic dataset only supports `split`: "
                    f"`train`, `val`, or `test`. Get: {self.split}."
                )

        # NOTE: List image files
        with progress_bar() as pbar:
            image_patterns = [
                os.path.join(self.root, "leftImg8bit", self.split, "*", "*.png")
            ]
            if self.split == "train" and self.quality == "gtCoarse" and self.extra:
                image_patterns.append(
                    os.path.join(self.root, "leftImg8bit", "train_extra", "*", "*.png")
                )
            image_paths = []
            for pattern in pbar.track(
                image_patterns, description=f"[bright_yellow]Listing {self.split} images"
            ):
                for image_path in glob.glob(pattern):
                    image_paths.append(image_path)
            self.image_paths = unique(image_paths)  # Remove all duplicates files

        # NOTE: fast_dev_run, select only a subset of images
        if self.fast_dev_run:
            indices          = [random.randint(0, len(self.image_paths) - 1)
                                for _ in range(self.batch_size)]
            self.image_paths = [self.image_paths[i] for i in indices]
            
        # NOTE: List semantic files
        with progress_bar() as pbar:
            semantic_prefixes   = [p.replace("_leftImg8bit.png", "")
                                   for p in self.image_paths]
            semantic_prefixes   = [predix.replace("leftImg8bit", self.quality)
                                   for predix in semantic_prefixes]
            self.semantic_paths = [
                f"{prefix}_{self.quality}_{self.encoding}.png"
                for prefix in pbar.track(
                    semantic_prefixes,
                    description=f"[bright_yellow]Listing {self.split} semantic images"
                )
            ]

        # NOTE: List label files
        with progress_bar() as pbar:
            self.label_paths = [
                f"{prefix}_{self.quality}_polygons.json"
                for prefix in pbar.track(
                    semantic_prefixes,
                    description=f"[bright_yellow]Listing {self.split} labels"
                )
            ]

        with progress_bar() as pbar:
            label_paths = [
                p.replace(self.quality, f"{self.quality}_custom")
                for p in self.semantic_paths
            ]
            self.custom_label_paths = [
                p.replace(".png", ".json") for p in
                pbar.track(
                    label_paths,
                    description=f"[bright_yellow]Listing {self.split} custom labels"
                )
            ]
        
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
        semantic_path    : str,
        label_path       : Optional[str] = None,
        custom_label_path: Optional[str] = None
    ) -> VisionData:
        """Load all labels from a raw label `file`.

        Args:
            image_path (str):
                Image file.
            semantic_path (str):
                Semantic segmentation image file.
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
            if self.has_custom_labels or not self.caching_labels:
                return VisionDataHandler().load_from_file(
                    image_path    = image_path,
                    semantic_path = semantic_path,
                    label_path    = custom_label_path
                )
        elif label_path and os.path.isfile(label_path):
            return CityscapesLabelHandler().load_from_file(
                image_path  = image_path,
                label_path  = label_path,
                class_labels = self.class_labels
            )
        
        # NOTE: Parse info
        image_info = ImageInfo.from_file(image_path=image_path)
        
        if is_image_file(path=semantic_path):
            semantic_info = ImageInfo.from_file(image_path=semantic_path)
            return VisionData(
                image_info=image_info, semantic_info=semantic_info
            )

        return VisionData(image_info=image_info)
    
    def load_class_labels(self):
        """Load ClassLabels."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path        = os.path.join(current_dir, "cityscapes_class_labels.json")
        self.class_labels = ClassLabels.create_from_file(label_path=path)
    

# MARK: - CityscapesSemanticDataModule

@DATAMODULES.register(name="cityscapessemantic")
class CityscapesSemanticDataModule(DataModule):
    """Cityscapes Semantic DataModule."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        dataset_dir: str = os.path.join(datasets_dir, "cityscapes"),
        name       : str = "cityscapessemantic",
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
        console.log(f"Setup [red]Cityscapes Semantic[/red] datasets.")
        
        # NOTE: Assign train/val datasets for use in dataloaders
        if model_state in [None, ModelState.TRAINING]:
            self.train = CityscapesSemantic(
                root=self.dataset_dir, split="train", **self.dataset_cfg
            )
            self.val   = CityscapesSemantic(
                root=self.dataset_dir, split="val", **self.dataset_cfg
            )
            self.class_labels = getattr(self.train, "class_labels", None)
            self.collate_fn  = getattr(self.train, "collate_fn",  None)
            
        # NOTE: Assign test datasets for use in dataloader(s)
        if model_state in [None, ModelState.TESTING]:
            self.test = CityscapesSemantic(
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
        "name": "cityscapessemantic",
        # Dataset's name.
        "quality": "gtFine",
        # Quality of the semantic segmentation mask to use.
        # One of: ["gtFine", "gtCoarse"]. Default: `gtFine`.
        "encoding": "id",
        # Format to use when creating the semantic segmentation mask.
        # One of: ["id", "trainId", "catId", "color"]. Default: `id`.
        "extra": False,
        # Should use extra data? Those in the `train_extra` split are only
        # available for `quality=gtCoarse`. Default: `False`.
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
    dm   = CityscapesSemanticDataModule(**cfgs)
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
