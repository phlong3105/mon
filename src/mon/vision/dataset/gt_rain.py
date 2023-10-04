#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements GT-RAIN datasets and datamodules."""

from __future__ import annotations

__all__ = [
    "GTRain", "GTRainDataModule",
]

from mon.core import console, pathlib, rich
from mon.globals import DATAMODULES, DATASETS, ModelPhase
from mon.vision.dataset import base


# region Dataset

@DATASETS.register(name="gt-rain")
class GTRain(base.ImageEnhancementDataset):
    """GT-Rain dataset.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    def get_images(self):
        """Get image files."""
        if self.split not in ["train", "val", "test"]:
            console.log(
                f"split must be one of ['train', 'val', 'test'], but got "
                f"{self.split}."
            )
        
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / self.split
            for path in pbar.track(
                list(pattern.rglob("*-R-*.png")),
                description=f"Listing {self.__class__.__name__} {self.split} images"
            ):
                if path.is_image_file():
                    image = base.ImageLabel(path=path)
                    self.images.append(image)
    
    def get_labels(self):
        """Get label files."""
        self.labels: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = str(img.path)
                path  = path[: path.find("-R-")] + "-C-000.png"
                path  = pathlib.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)


# endregion


# region Datamodule

@DATAMODULES.register(name="gt-rain")
class GTRainDataModule(base.DataModule):
    """GT-RAIN datamodule.
    
    See Also: :class:`mon.nn.data.datamodule.DataModule`.
    """
    
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk, or that need
        to be done only from a single GPU in distributed settings:
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, phase: ModelPhase | str | None = None):
        """Use this method to do things on every device:
            - Count number of classes.
            - Build classlabels vocabulary.
            - Prepare train/val/test splits.
            - Apply transformations.
            - Define :attr:`collate_fn` for your custom dataset.

        Args:
            phase: The model phase. One of:
                - ``'training'`` : prepares :attr:'train' and :attr:'val'.
                - ``'testing'``  : prepares :attr:'test'.
                - ``'inference'``: prepares :attr:`predict`.
                - ``None``:      : prepares all.
                - Default: ``None``.
        """
        console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        
        if phase in [None, ModelPhase.TRAINING]:
            self.train = GTRain(split="train", **self.dataset_kwargs)
            self.val   = GTRain(split="val",   **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = GTRain(split="test", **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


# endregion
