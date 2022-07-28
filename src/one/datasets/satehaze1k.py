#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SateHaze1K dataset and datamodule.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

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

@DATASETS.register(name="satehaze1k")
class SateHaze1K(ImageEnhancementDataset):
    """
    SateHaze1K dataset consists 1200 pairs of hazy and corresponding haze-free
    images.
    
    The new haze satellite dataset on which we evaluate our approach contains
    1200 individual pairs of hazy images, corresponding hazy-free images and SAR
    images. In order to guarantee the facticity, abundance, and diversity of
    haze masks in our dataset, we use Photoshop Software to extract real haze
    masks of the easily accessible original hazy remote sensing images to
    generate transmission maps for synthetic images. The dataset consists of 3
    levels of fog, called Thin fog, Moderate fog, Thick fog. In the synthetic
    images covered by thin fog, the haze mask will be only mist which picks up
    from the original real cloudy image. For the moderate fog image, samples
    overlap with mist and medium fog. But for the thick fog, the transmission
    maps are selected from the dense haze.
    
    Training, validation and test folds. Our training, validation and test folds
    were approximately 80%, 10%, 10% of the total data respectively. We split
    every 400 images to train, valid, and test set, and artificially label 45
    of thick fog images for segmentation purposes.
    
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
        if self.split not in ["train", "val", "test"]:
            console.log(
                f"SateHaze1K dataset only supports `split`: `train`, `val`, or "
                f"`test`. Get: {self.split}."
            )
            
            
        self.images: list[Image] = []
        with progress_bar() as pbar:
            pattern = self.root
            for path in pbar.track(
                list(pattern.rglob(f"{self.split}/input/*.png")),
                description=f"[bright_yellow]Listing SateHaze1K {self.split} images"
            ):
                self.images.append(Image(path=path, backend=self.backend))
    
    def list_labels(self):
        """
        List label files.
        """
        self.labels: list[Image] = []
        with progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"[bright_yellow]Listing SateHaze1K {self.split} labels"
            ):
                path = Path(str(img.path).replace("input", "target"))
                self.labels.append(Image(path=path, backend=self.backend))
                

@DATASETS.register(name="satehaze1kthin")
class SateHaze1KThin(SateHaze1K):
    """
    SateHaze1K-Thin dataset.
    """
    
    def list_images(self):
        """
        List image files.
        """
        if self.split not in ["train", "val", "test"]:
            console.log(
                f"SateHaze1KThin dataset only supports `split`: `train`, `val`, or "
                f"`test`. Get: {self.split}."
            )
            
        self.images: list[Image] = []
        with progress_bar() as pbar:
            pattern = self.root / "thin" / self.split
            for path in pbar.track(
                list(pattern.rglob(f"input/*.png")),
                description=f"[bright_yellow]Listing SateHaze1KThin {self.split} images"
            ):
                self.images.append(Image(path=path, backend=self.backend))
    
    def list_labels(self):
        """
        List label files.
        """
        self.labels: list[Image] = []
        with progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"[bright_yellow]Listing SateHaze1KThin {self.split} labels"
            ):
                path = Path(str(img.path).replace("input", "target"))
                self.labels.append(Image(path=path, backend=self.backend))


@DATASETS.register(name="satehaze1kmoderate")
class SateHaze1KModerate(SateHaze1K):
    """
    SateHaze1K-Moderate.
    """
    
    def list_images(self):
        """
        List image files.
        """
        if self.split not in ["train", "val", "test"]:
            console.log(
                f"SateHaze1KModerate dataset only supports `split`: `train`, `val`, or "
                f"`test`. Get: {self.split}."
            )
            
        self.images: list[Image] = []
        with progress_bar() as pbar:
            pattern = self.root / "moderate" / self.split
            for path in pbar.track(
                list(pattern.rglob(f"input/*.png")),
                description=f"[bright_yellow]Listing SateHaze1KModerate {self.split} images"
            ):
                self.images.append(Image(path=path, backend=self.backend))
    
    def list_labels(self):
        """
        List label files.
        """
        self.labels: list[Image] = []
        with progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"[bright_yellow]Listing SateHaze1KModerate {self.split} labels"
            ):
                path = Path(str(img.path).replace("input", "target"))
                self.labels.append(Image(path=path, backend=self.backend))
                

@DATASETS.register(name="satehaze1kthick")
class SateHaze1KThick(SateHaze1K):
    """
    SateHaze1K-Thick dataset.
    """
    
    def list_images(self):
        """
        List image files.
        """
        if self.split not in ["train", "val", "test"]:
            console.log(
                f"SateHaze1KThick dataset only supports `split`: `train`, `val`, or "
                f"`test`. Get: {self.split}."
            )
            
        self.images: list[Image] = []
        with progress_bar() as pbar:
            pattern = self.root / "thick" / self.split
            for path in pbar.track(
                list(pattern.rglob(f"input/*.png")),
                description=f"[bright_yellow]Listing SateHaze1KThick {self.split} images"
            ):
                self.images.append(Image(path=path, backend=self.backend))
    
    def list_labels(self):
        """
        List label files.
        """
        self.labels: list[Image] = []
        with progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"[bright_yellow]Listing SateHaze1KThick {self.split} labels"
            ):
                path = Path(str(img.path).replace("input", "target"))
                self.labels.append(Image(path=path, backend=self.backend))
      

@DATAMODULES.register(name="satehaze1k")
class SateHaze1KDataModule(DataModule):
    """
    SateHaze1K DataModule.
    """
    
    def __init__(
        self,
        root: str = DATA_DIR / "satehaze1k",
        name: str = "satehaze1k",
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
        console.log(f"Setup [red]SateHaze1K[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
            self.train = SateHaze1K(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.val = SateHaze1K(
                root             = self.root,
                split            = "val",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.class_label = getattr(self.train, "class_labels", None)
            self.collate_fn  = getattr(self.train, "collate_fn",   None)
            
        # Assign test datasets for use in dataloader(s)
        if phase in [None, ModelPhase.TESTING]:
            self.test = SateHaze1K(
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


@DATAMODULES.register(name="satehaze1kthin")
class SateHaze1KThinDataModule(SateHaze1KDataModule):
    """
    SateHaze1K-Thin DataModule.
    """
  
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
        console.log(f"Setup [red]SateHaze1KThin[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
            self.train = SateHaze1KThin(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.val = SateHaze1KThin(
                root             = self.root,
                split            = "val",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.class_label = getattr(self.train, "class_labels", None)
            self.collate_fn  = getattr(self.train, "collate_fn",   None)
            
        # Assign test datasets for use in dataloader(s)
        if phase in [None, ModelPhase.TESTING]:
            self.test = SateHaze1KThin(
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
  
      
@DATAMODULES.register(name="satehaze1kmoderate")
class SateHaze1KModerateDataModule(SateHaze1KDataModule):
    """
    SateHaze1K-Moderate DataModule.
    """
    
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
        console.log(f"Setup [red]SateHaze1KModerate[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
            self.train = SateHaze1KModerate(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.val = SateHaze1KModerate(
                root             = self.root,
                split            = "val",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.class_label = getattr(self.train, "class_labels", None)
            self.collate_fn  = getattr(self.train, "collate_fn",   None)
            
        # Assign test datasets for use in dataloader(s)
        if phase in [None, ModelPhase.TESTING]:
            self.test = SateHaze1KModerate(
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
      
      
@DATAMODULES.register(name="satehaze1kthick")
class SateHaze1KThickDataModule(SateHaze1KDataModule):
    """
    SateHaze1K-Thick DataModule.
    """
    
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
        console.log(f"Setup [red]SateHaze1KThick[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
            self.train = SateHaze1KThick(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.val = SateHaze1KThick(
                root             = self.root,
                split            = "val",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.class_label = getattr(self.train, "class_labels", None)
            self.collate_fn  = getattr(self.train, "collate_fn",   None)
            
        # Assign test datasets for use in dataloader(s)
        if phase in [None, ModelPhase.TESTING]:
            self.test = SateHaze1KThick(
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
  
  
# MARK: - Test -----------------------------------------------------------------

def test_satehaze1k():
    cfg = {
        "root": DATA_DIR / "satehaze1k",
           # Root directory of dataset.
        "name": "satehaze1k",
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
    dm  = SateHaze1KDataModule(**cfg)
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


def test_satehaze1kmoderate():
    cfg = {
        "root": DATA_DIR / "satehaze1k",
           # Root directory of dataset.
        "name": "satehaze1kmoderate",
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
    dm  = SateHaze1KModerateDataModule(**cfg)
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


def test_satehaze1kthick():
    cfg = {
        "root": DATA_DIR / "satehaze1k",
           # Root directory of dataset.
        "name": "satehaze1kthick",
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
    dm  = SateHaze1KThickDataModule(**cfg)
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


def test_satehaze1kthick():
    cfg = {
        "root": DATA_DIR / "satehaze1k",
           # Root directory of dataset.
        "name": "satehaze1kthin",
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
    dm  = SateHaze1KThinDataModule(**cfg)
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
    test_satehaze1k()