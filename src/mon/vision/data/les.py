#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements light effect suppression (les) datasets and
datamodules.
"""

from __future__ import annotations

__all__ = [
    "Flare7KPPReal",
    "Flare7KPPRealDataModule",
    "Flare7KPPSyn",
    "Flare7KPPSynDataModule",
    "Jin2022",
    "Jin2022DataModule",
]

from mon.globals import DATAMODULES, DATASETS, ModelPhase
from mon.vision import core
from mon.vision.data import base

console = core.console


# region Dataset

@DATASETS.register(name="flare7k++-real")
class Flare7KPPReal(base.ImageEnhancementDataset):
    """Flare7K++-Real dataset consists of 100 flare/clear image pairs.
    
    See Also: :class:`mon.vision.dataset.base.dataset.UnlabeledImageDataset`.
    """
    
    splits = ["test"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "flare7k++-real" / "flare"
        ]
        self.images: list[base.ImageLabel] = []
        with core.get_progress_bar() as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    list(pattern.rglob("*")),
                    description=f"Listing {self.__class__.__name__} {self.split} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self.images.append(image)

    def get_labels(self):
        """Get label files."""
        self.labels: list[base.ImageLabel] = []
        with core.get_progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = str(img.path).replace("flare", "clears")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="flare7k++-syn")
class Flare7KPPSyn(base.ImageEnhancementDataset):
    """Flare7K++-Syn dataset consists of 100 flare/clear image pairs.

    See Also: :class:`mon.vision.dataset.base.dataset.UnlabeledImageDataset`.
    """

    splits = ["test"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "flare7k++-syn" / "flare"
        ]
        self.images: list[base.ImageLabel] = []
        with core.get_progress_bar() as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    list(pattern.rglob("*")),
                    description=f"Listing {self.__class__.__name__} {self.split} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self.images.append(image)

    def get_labels(self):
        """Get label files."""
        self.labels: list[base.ImageLabel] = []
        with core.get_progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = str(img.path).replace("flare", "clears")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="jin2022")
class Jin2022(base.UnlabeledImageDataset):
    """Jin2022 dataset consists 961 flare images.

    See Also: :class:`mon.vision.dataset.base.dataset.UnlabeledImageDataset`.
    """

    splits = ["train"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "jin2022" / "clear",
            self.root / self.split / "jin2022" / "flare",
        ]
        self.images: list[base.ImageLabel] = []
        with core.get_progress_bar() as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    list(pattern.rglob("*")),
                    description=f"Listing {self.__class__.__name__} {self.split} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self.images.append(image)

# endregion


# region Datamodule

@DATAMODULES.register(name="flare7k++-real")
class Flare7KPPRealDataModule(base.DataModule):
    """Flare7K++-Real datamodule.
    
    See Also: :class:`mon.nn.data.datamodule.DataModule`.
    """
    
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk, or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, phase: ModelPhase | None = None):
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
            dataset = Flare7KPPReal(split="test", **self.dataset_kwargs)
            self.split_train_val(dataset=dataset, split_ratio=0.8, full_train=True)
        if phase in [None, ModelPhase.TESTING]:
            self.test = Flare7KPPReal(split="test", **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="flare7k++-syn")
class Flare7KPPSynDataModule(base.DataModule):
    """Flare7K++-Syn datamodule.

    See Also: :class:`mon.nn.data.datamodule.DataModule`.
    """

    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk, or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.get_classlabels()

    def setup(self, phase: ModelPhase | None = None):
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
            dataset = Flare7KPPSyn(split="test", **self.dataset_kwargs)
            self.split_train_val(dataset=dataset, split_ratio=0.8, full_train=True)
        if phase in [None, ModelPhase.TESTING]:
            self.test = Flare7KPPSyn(split="test", **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()

        self.summarize()

    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="jin2022")
class Jin2022DataModule(base.DataModule):
    """Jin2022 datamodule.

    See Also: :class:`mon.nn.data.datamodule.DataModule`.
    """

    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk, or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.get_classlabels()

    def setup(self, phase: ModelPhase | None = None):
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
            dataset = Jin2022(split="train", **self.dataset_kwargs)
            self.split_train_val(dataset=dataset, split_ratio=0.8, full_train=True)
        if phase in [None, ModelPhase.TESTING]:
            self.test = Jin2022(split="train", **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()

        self.summarize()

    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass

# endregion
