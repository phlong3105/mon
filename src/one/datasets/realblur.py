#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RealBlurJ dataset and datamodule.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from torch.utils.data import random_split

from one.constants import DATA_DIR
from one.constants import DATAMODULES
from one.constants import DATASETS
from one.constants import VISION_BACKEND
from one.core import console
from one.core import Ints
from one.core import ModelPhase
from one.core import ModelPhase_
from one.core import progress_bar
from one.core import Transforms_
from one.core import VisionBackend_
from one.data import ClassLabel_
from one.data import DataModule
from one.data import Image
from one.data import ImageEnhancementDataset
from one.plot import imshow
from one.vision.transformation import Resize


# MARK: - Module ---------------------------------------------------------------

@DATASETS.register(name="realblurj")
class RealBlurJ(ImageEnhancementDataset):
    """
    RealBlur-J dataset.
    
    The first large-scale dataset of real-world blurred images for learning
    and benchmarking single image deblurring methods, which is dubbed RealBlur.
    Our dataset consists of two subsets sharing the same image contents, one of
    which is generated from camera raw images, and the other from JPEG images
    processed by the camera ISP. Each subset provides 4,556 pairs of blurred
    and ground truth sharp images of 232 low-light static scenes. The blurred
    images in the dataset are blurred by camera shakes, and captured in
    low-light environments such as streets at night, and indoor rooms to cover
    the most common scenarios for motion blurs. To tackle the challenge of
    geometric alignment, we build an image acquisition system that can
    simultaneously capture a pair of blurred and sharp images that are
    geometrically aligned.
    
    Args:
        root (str): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image shape as [H, W, C], [H, W], or [S, S].
        class_label (ClassLabel_ | None): ClassLabel object. Defaults to None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root            : str,
        split           : str                = "train",
        shape           : Ints               = (3, 720, 1280),
        class_label     : ClassLabel_ | None = None,
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        cache_data      : bool               = False,
        cache_images    : bool               = False,
        backend         : VisionBackend_     = VISION_BACKEND,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            class_label      = class_label,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
     
    def list_images(self):
        """
        List image files.
        """
        if self.split not in ["train", "test"]:
            console.log(
                f"RealBlur-J dataset only supports `split`: `train` or `test`. "
                f"Get: {self.split}."
            )
            
        self.images: list[Image] = []
        with progress_bar() as pbar:
            if self.split == "train":
                file = self.root / "realblur_j_train.txt"
            else:
                file = self.root / "realblur_j_test.txt"
            lines = open(file, "r").readlines()
            for line in pbar.track(
                lines,
                description=f"[bright_yellow]Listing RealBlur-J {self.split} images"
            ):
                label_path, image_path = line.rstrip().split(" ")
                image_path = self.root / image_path
                label_path = self.root / label_path
                self.images.append(Image(path=image_path, backend=self.backend))
                self.labels.append(Image(path=label_path, backend=self.backend))
    
    def list_labels(self):
        """
        List label files.
        """
        pass
   

@DATASETS.register(name="realblurr")
class RealBlurR(ImageEnhancementDataset):
    """
    RealBlur-R dataset.
    
    The first large-scale dataset of real-world blurred images for learning
    and benchmarking single image deblurring methods, which is dubbed RealBlur.
    Our dataset consists of two subsets sharing the same image contents, one of
    which is generated from camera raw images, and the other from JPEG images
    processed by the camera ISP. Each subset provides 4,556 pairs of blurred
    and ground truth sharp images of 232 low-light static scenes. The blurred
    images in the dataset are blurred by camera shakes, and captured in
    low-light environments such as streets at night, and indoor rooms to cover
    the most common scenarios for motion blurs. To tackle the challenge of
    geometric alignment, we build an image acquisition system that can
    simultaneously capture a pair of blurred and sharp images that are
    geometrically aligned.
    
    Args:
        root (str): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image shape as [H, W, C], [H, W], or [S, S].
        class_label (ClassLabel_ | None): ClassLabel object. Defaults to None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root            : str,
        split           : str                = "train",
        shape           : Ints               = (3, 720, 1280),
        class_label     : ClassLabel_ | None = None,
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        cache_data      : bool               = False,
        cache_images    : bool               = False,
        backend         : VisionBackend_     = VISION_BACKEND,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            class_label      = class_label,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
     
    def list_images(self):
        """
        List image files.
        """
        if self.split not in ["train", "test"]:
            console.log(
                f"RealBlur-R dataset only supports `split`: `train` or `test`. "
                f"Get: {self.split}."
            )
            
        self.images: list[Image] = []
        with progress_bar() as pbar:
            if self.split == "train":
                file = self.root / "realblur_r_train.txt"
            else:
                file = self.root / "realblur_r_test.txt"
            lines = open(file, "r").readlines()
            for line in pbar.track(
                lines,
                description=f"[bright_yellow]Listing RealBlur-R {self.split} images"
            ):
                label_path, image_path = line.rstrip().split(" ")
                image_path = self.root / image_path
                label_path = self.root / label_path
                self.images.append(Image(path=image_path, backend=self.backend))
                self.labels.append(Image(path=label_path, backend=self.backend))
    
    def list_labels(self):
        """
        List label files.
        """
        pass
   
   
@DATAMODULES.register(name="realblurj")
class RealBlurJDataModule(DataModule):
    """
    RealBlur-J DataModule.
    """
    
    def __init__(
        self,
        root: str = DATA_DIR / "realblur",
        name: str = "realblurj",
        *args, **kwargs
    ):
        super().__init__(root=root, name=name, *args, **kwargs)
        
    def prepare_data(self, *args, **kwargs):
        """
        Use this method to do things that might write to disk or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        if self.class_label is None:
            self.load_class_label()
    
    def setup(self, phase: ModelPhase_ | None = None):
        """
        There are also data operations you might want to perform on every GPU.

        Todos:
            - Count number of classes.
            - Build class_labels vocabulary.
            - Perform train/val/test splits.
            - Apply transforms (defined explicitly in your datamodule or
              assigned in init).
            - Define collate_fn for you custom dataset.

        Args:
            phase (ModelPhase_ | None):
                Stage to use: [None, ModelPhase.TRAINING, ModelPhase.TESTING].
                Set to None to setup all train, val, and test data.
                Defaults to None.
        """
        console.log(f"Setup [red]RealBlur-J[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
            full_dataset = RealBlurJ(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            train_size   = int(0.8 * len(full_dataset))
            val_size     = len(full_dataset) - train_size
            self.train, self.val = random_split(
                full_dataset, [train_size, val_size]
            )
            self.class_label = getattr(full_dataset, "class_labels", None)
            self.collate_fn  = getattr(full_dataset, "collate_fn",   None)
            
        # Assign test datasets for use in dataloader(s)
        if phase in [None, ModelPhase.TESTING]:
            self.test = RealBlurJ(
                root             = self.root,
                split            = "test",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.class_label = getattr(self.test, "class_labels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",   None)
        
        if self.class_label is None:
            self.load_class_label()

        self.summarize()
        
    def load_class_label(self):
        """
        Load ClassLabel.
        """
        pass


@DATAMODULES.register(name="realblurr")
class RealBlurRDataModule(DataModule):
    """
    RealBlur-R DataModule.
    """
    
    def __init__(
        self,
        root: str = DATA_DIR / "realblur",
        name: str = "realblurr",
        *args, **kwargs
    ):
        super().__init__(root=root, name=name, *args, **kwargs)
        
    def prepare_data(self, *args, **kwargs):
        """
        Use this method to do things that might write to disk or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        if self.class_label is None:
            self.load_class_label()
    
    def setup(self, phase: ModelPhase_ | None = None):
        """
        There are also data operations you might want to perform on every GPU.

        Todos:
            - Count number of classes.
            - Build class_labels vocabulary.
            - Perform train/val/test splits.
            - Apply transforms (defined explicitly in your datamodule or
              assigned in init).
            - Define collate_fn for you custom dataset.

        Args:
            phase (ModelPhase_ | None):
                Stage to use: [None, ModelPhase.TRAINING, ModelPhase.TESTING].
                Set to None to setup all train, val, and test data.
                Defaults to None.
        """
        console.log(f"Setup [red]RealBlur-R[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
            full_dataset = RealBlurR(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            train_size   = int(0.8 * len(full_dataset))
            val_size     = len(full_dataset) - train_size
            self.train, self.val = random_split(
                full_dataset, [train_size, val_size]
            )
            self.class_label = getattr(full_dataset, "class_labels", None)
            self.collate_fn  = getattr(full_dataset, "collate_fn",   None)
            
        # Assign test datasets for use in dataloader(s)
        if phase in [None, ModelPhase.TESTING]:
            self.test = RealBlurR(
                root             = self.root,
                split            = "test",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.class_label = getattr(self.test, "class_labels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",   None)
        
        if self.class_label is None:
            self.load_class_label()

        self.summarize()
        
    def load_class_label(self):
        """
        Load ClassLabel.
        """
        pass


# MARK: - Test -----------------------------------------------------------------

def test_realblurj():
    cfg = {
        "root": DATA_DIR / "realblur",
           # Root directory of dataset.
        "name": "realblurj",
            # Dataset's name.
        "shape": [3, 512, 512],
            # Image shape as [H, W, C], [H, W], or [S, S].
        "transform": None,
            # Functions/transforms that takes in an input sample and returns a
            # transformed version.
        "target_transform": None,
            # Functions/transforms that takes in a target and returns a
            # transformed version.
        "transforms": [
            Resize(size=[3, 512, 512])
        ],
            # Functions/transforms that takes in an input and a target and
            # returns the transformed versions of both.
        "cache_data": False,
            # If True, cache data to disk for faster loading next time.
            # Defaults to False.
        "cache_images": False,
            # If True, cache images into memory for faster training (WARNING:
            # large datasets may exceed system RAM). Defaults to False.
        "backend": VISION_BACKEND,
            # Vision backend to process image. Defaults to VISION_BACKEND.
        "batch_size": 8,
            # Number of samples in one forward & backward pass. Defaults to 1.
        "devices" : 0,
            # The devices to use. Defaults to 0.
        "shuffle": True,
            # If True, reshuffle the data at every training epoch.
            # Defaults to True.
        "verbose": True,
            # Verbosity. Defaults to True.
    }
    dm  = RealBlurJDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.class_label:
        dm.class_label.print()
    # Visualize one sample
    data_iter           = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    imshow(winname="image",  image=input,  figure_num=0)
    imshow(winname="target", image=target, figure_num=1)
    plt.show(block=True)


def test_realblurr():
    cfg = {
        "root": DATA_DIR / "realblur",
           # Root directory of dataset.
        "name": "realblurr",
            # Dataset's name.
        "shape": [3, 512, 512],
            # Image shape as [H, W, C], [H, W], or [S, S].
        "transform": None,
            # Functions/transforms that takes in an input sample and returns a
            # transformed version.
        "target_transform": None,
            # Functions/transforms that takes in a target and returns a
            # transformed version.
        "transforms": [
            Resize(size=[3, 512, 512])
        ],
            # Functions/transforms that takes in an input and a target and
            # returns the transformed versions of both.
        "cache_data": False,
            # If True, cache data to disk for faster loading next time.
            # Defaults to False.
        "cache_images": False,
            # If True, cache images into memory for faster training (WARNING:
            # large datasets may exceed system RAM). Defaults to False.
        "backend": VISION_BACKEND,
            # Vision backend to process image. Defaults to VISION_BACKEND.
        "batch_size": 8,
            # Number of samples in one forward & backward pass. Defaults to 1.
        "devices" : 0,
            # The devices to use. Defaults to 0.
        "shuffle": True,
            # If True, reshuffle the data at every training epoch.
            # Defaults to True.
        "verbose": True,
            # Verbosity. Defaults to True.
    }
    dm  = RealBlurRDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.class_label:
        dm.class_label.print()
    # Visualize one sample
    data_iter           = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    imshow(winname="image",  image=input,  figure_num=0)
    imshow(winname="target", image=target, figure_num=1)
    plt.show(block=True)


# MARK: - Main -----------------------------------------------------------------

if __name__ == "__main__":
    test_realblurj()