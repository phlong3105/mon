#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NH-Haze dataset and datamodule.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from one.core import ClassLabel_
from one.core import console
from one.core import DATA_DIR
from one.core import DataModule
from one.core import DATAMODULES
from one.core import DATASETS
from one.core import Image
from one.core import ImageEnhancementDataset
from one.core import Ints
from one.core import ModelPhase
from one.core import ModelPhase_
from one.core import progress_bar
from one.core import Transforms_
from one.core import VISION_BACKEND
from one.core import VisionBackend_
from one.plot import imshow
from one.vision.transformation import Resize


# MARK: - Module ---------------------------------------------------------------

@DATASETS.register(name="nhhaze")
class NHHaze(ImageEnhancementDataset):
    """
    NH-Haze dataset consists 55 pairs of real hazy and corresponding haze-free
    images.
    
    Image dehazing is an ill-posed problem that has been extensively studied
    in the recent years. The objective performance evaluation of the dehazing
    methods is one of the major obstacles due to the lacking of a reference
    dataset. While the synthetic datasets have shown important limitations, the
    few realistic datasets introduced recently assume homogeneous haze over the
    entire scene. Since in many real cases haze is not uniformly distributed we
    introduce NH-HAZE, a non-homogeneous realistic dataset with pairs of real
    hazy and corresponding haze-free images. This is the first non-homogeneous
    image dehazing dataset and contains 55 outdoor scenes. The non-homogeneous
    haze has been introduced in the scene using a professional haze generator
    that imitates the real conditions of hazy scenes. Additionally, this work
    presents an objective assessment of several state- of-the-art single image
    dehazing methods that were evaluated using NH-HAZE dataset.
    
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
                f"NH-Haze dataset only supports `split`: `train`, `val`, or "
                f"`test`. Get: {self.split}."
            )
            
        self.images: list[Image] = []
        with progress_bar() as pbar:
            pattern = self.root / self.split
            for path in pbar.track(
                list(pattern.rglob("hazy/*.png")),
                description=f"[bright_yellow]Listing NH-Haze {self.split} images"
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
                description=f"[bright_yellow]Listing NH-Haze {self.split} labels"
            ):
                path = Path(str(img.path).replace("hazy", "gt"))
                self.labels.append(Image(path=path, backend=self.backend))
                

@DATAMODULES.register(name="nhhaze")
class NHHazeDataModule(DataModule):
    """
    NHHaze DataModule.
    """
    
    def __init__(
        self,
        root: str = DATA_DIR / "ntire" / "nhhaze",
        name: str = "nhhaze",
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
        console.log(f"Setup [red]NH-Haze[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
            self.train = NHHaze(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.val = NHHaze(
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
            self.test = NHHaze(
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

def test():
    cfg = {
        "root": DATA_DIR / "ntire" / "nhhaze",
           # Root directory of dataset.
        "name": "nhhaze",
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
    dm  = NHHazeDataModule(**cfg)
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
    test()
