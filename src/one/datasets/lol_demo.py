#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LoL226 dataset and datamodule.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from one.constants import *
from one.core import *
from one.data import DataModule
from one.data import UnlabeledVideoDataset
from one.plot import imshow
from one.vision.transformation import Resize


# H1: - Functional -------------------------------------------------------------


# H1: - Module -----------------------------------------------------------------

@DATASETS.register(name="lol_demo")
class LoLDemo(UnlabeledVideoDataset):
    """
   
    Args:
        name (str): Dataset's name.
        root (Path_): Root directory of dataset.
        filename (str): Video file name.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        max_samples (int | None): Only process certain amount of samples.
            Defaults to None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        api_preference (int): Preferred Capture API backends to use. Can be
            used to enforce a specific reader implementation. Two most used
            options are: [cv2.CAP_ANY=0, cv2.CAP_FFMPEG=1900].
            See more: https://docs.opencv.org/4.5.5/d4/d15
            /group__videoio__flags__base.html
            #ggaeb8dd9c89c10a5c63c139bf7c4f5704da7b235a04f50a444bc2dc72f5ae394aaf
            Defaults to cv2.CAP_FFMPEG.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str                = "lol_demo",
        root            : Path_              = DATA_DIR / "lol_demo",
        filename        : str                = "aokigahara.mp4",
        split           : str                = "train",
        shape           : Ints               = (3, 720, 1280),
        max_samples     : int         | None = None,
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        api_preference  : int                = cv2.CAP_FFMPEG,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        self.filename = filename
        super().__init__(
            name             = name,
            root             = root,
            split            = split,
            shape            = shape,
            max_samples      = max_samples,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            api_preference   = api_preference,
            verbose          = verbose,
            *args, **kwargs
        )
    
    def list_source(self):
        """
        List video file.
        """
        self.source = self.root / self.filename

                
@DATAMODULES.register(name="lol_demo")
class LoLDemoDataModule(DataModule):
    """
    LoLDemo DataModule.
    """
    
    def __init__(
        self,
        name            : str                = "lol_demo",
        root            : Path_              = DATA_DIR / "lol_demo",
        filename        : str                = "aokigahara.mp4",
        shape           : Ints               = (3, 512, 512),
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        batch_size      : int                = 1,
        devices         : Devices            = 0,
        shuffle         : bool               = True,
        collate_fn      : Callable    | None = None,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        self.filename = filename
        super().__init__(
            name             = name,
            root             = root,
            shape            = shape,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            batch_size       = batch_size,
            devices          = devices,
            shuffle          = shuffle,
            collate_fn       = collate_fn,
            verbose          = verbose,
            *args, **kwargs
        )
        
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
        console.log(f"Setup [red]{LoLDemo.classname}[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        
        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
            self.train = LoLDemo(
                root             = self.root,
                filename         = self.filename,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.val         = None
            self.classlabels = getattr(self.train, "classlabels", None)
            self.collate_fn  = getattr(self.train, "collate_fn",  None)
            
        # Assign test datasets for use in dataloader(s)
        if phase in [None, ModelPhase.TESTING]:
            self.test = LoLDemo(
                root             = self.root,
                filename         = self.filename,
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


# H1: - Test -------------------------------------------------------------------

def test_lol_demo():
    cfg = {
        "name": "lol_demo",
            # Dataset's name.
        "root": DATA_DIR / "lol_demo",
            # Root directory of dataset.
        "filename": "aokigahara.mp4",
            # Video file name.
        "shape": [3, 512, 512],
            # Image shape as [C, H, W], [H, W], or [S, S].
        "max_samples": None,
            # Only process certain amount of samples. Defaults to None.
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
        "api_preference": cv2.CAP_FFMPEG,
            # Preferred Capture API backends to use. Can be used to enforce a
            # specific reader implementation. Two most used options are:
            # [cv2.CAP_ANY=0, cv2.CAP_FFMPEG=1900]. Defaults to cv2.CAP_FFMPEG.
        "batch_size": 1,
            # Number of samples in one forward & backward pass. Defaults to 1.
        "devices" : 0,
            # The devices to use. Defaults to 0.
        "shuffle": False,
            # If True, reshuffle the data at every training epoch.
            # Defaults to True.
        "verbose": True,
            # Verbosity. Defaults to True.
    }
    dm  = LoLDemoDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter      = iter(dm.test_dataloader)
    input, _, meta = next(data_iter)
    imshow(winname="image", image=input)
    plt.show(block=True)


# H1: - Main -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str , default="lol_demo", help="The task to run")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.task == "lol_demo":
        test_lol_demo()
