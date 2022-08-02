#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A2I2Haze dataset and datamodule.
"""

from __future__ import annotations

from pathlib import Path

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
from one.core import Path_
from one.core import progress_bar
from one.core import Transforms_
from one.core import VisionBackend_
from one.data import ClassLabels_
from one.data import DataModule
from one.data import Image
from one.data import ImageEnhancementDataset
from one.plot import imshow
from one.vision.transformation import Resize


# MARK: - Module ---------------------------------------------------------------

@DATASETS.register(name="a2i2haze")
class A2I2Haze(ImageEnhancementDataset):
    """
    A2I2-Haze dataset for image de-hazing.
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image shape as [H, W, C], [H, W], or [S, S].
        classlabels (ClassLabels_ | None): ClassLabels object. Defaults to
            None.
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
        root            : Path_,
        split           : str                 = "train",
        shape           : Ints                = (3, 720, 1280),
        classlabels     : ClassLabels_ | None = None,
        transform       : Transforms_  | None = None,
        target_transform: Transforms_  | None = None,
        transforms      : Transforms_  | None = None,
        cache_data      : bool                = False,
        cache_images    : bool                = False,
        backend         : VisionBackend_      = VISION_BACKEND,
        verbose         : bool                = True,
        *args, **kwargs
    ):
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
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
        if self.split not in ["train"]:
            console.log(
                f"{self.clsname} dataset only supports `split`: "
                f"`train`. Get: {self.split}."
            )
            
        self.images: list[Image] = []
        with progress_bar() as pbar:
            pattern = self.root / self.split / "haze_images"
            for path in pbar.track(
                list(pattern.rglob("*.jpg")),
                description=f"[bright_yellow]Listing {self.clsname} "
                            f"{self.split} images"
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
                description=f"[bright_yellow]Listing {self.clsname} "
                            f"{self.split} labels"
            ):
                path = Path(str(img.path).replace("haze_images", "clean_images"))
                self.labels.append(Image(path=path, backend=self.backend))
                

@DATAMODULES.register(name="a2i2haze")
class A2I2HazeDataModule(DataModule):
    """
    A2I2Haze DataModule.
    """
    
    def __init__(
        self,
        root: Path_ = DATA_DIR / "a2i2" / "haze",
        name: str   = "a2i2haze",
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
        if self.classlabels is None:
            self.load_classlabels()
    
    def setup(self, phase: ModelPhase_ | None = None):
        """
        There are also data operations you might want to perform on every GPU.

        Todos:
            - Count number of classes.
            - Build classlabels vocabulary.
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
        console.log(f"Setup [red]{A2I2Haze.absclsname}[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
            full_dataset = A2I2Haze(
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
            self.classlabels = getattr(full_dataset, "classlabels", None)
            self.collate_fn  = getattr(full_dataset, "collate_fn",  None)
            
        # Assign test datasets for use in dataloader(s)
        if phase in [None, ModelPhase.TESTING]:
            self.test = A2I2Haze(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.load_classlabels()

        self.summarize()
        
    def load_classlabels(self):
        """
        Load ClassLabels.
        """
        pass


# MARK: - Test -----------------------------------------------------------------

def test_a2i2haze():
    cfg = {
        "root": DATA_DIR / "a2i2" / "haze",
           # Root directory of dataset.
        "name": "a2i2haze",
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
    dm  = A2I2HazeDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter           = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    imshow(winname="image",  image=input)
    imshow(winname="target", image=target)
    plt.show(block=True)


# MARK: - Main -----------------------------------------------------------------

if __name__ == "__main__":
    test_a2i2haze()
