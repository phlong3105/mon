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
    "FlareReal800",
    "FlareReal800DataModule",
    "LEDLight",
    "LEDLightDataModule",
    "LightEffect",
    "LightEffectDataModule",
    "MIPIFlare",
    "MIPIFlareDataModule"
]

import numpy as np
import torch

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
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
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
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = str(img.path).replace("/flare/", "/clear/")
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
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
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
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = str(img.path).replace("/flare/", "/clear/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="flarereal800")
class FlareReal800(base.ImageEnhancementDataset):
    """FlareReal800 dataset consists of 800 flare/clear image pairs.
    
    See Also: :class:`mon.vision.dataset.base.dataset.UnlabeledImageDataset`.
    """
    
    splits = ["train", "val"]
    
    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "flarereal800" / "flare"
        ]
        self.images: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
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
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = str(img.path).replace("/flare/", "/clear/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="ledlight")
class LEDLight(base.ImageEnhancementDataset):
    """LEDLight dataset consists of 100 flare/clear image pairs.

    See Also: :class:`mon.vision.dataset.base.dataset.UnlabeledImageDataset`.
    """

    splits = ["test"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "ledlight" / "flare"
        ]
        self.images: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
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
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = str(img.path).replace("/flare/", "/clear/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="light-effect")
@DATASETS.register(name="lighteffect")
class LightEffect(base.UnlabeledImageDataset):
    """LightEffect dataset consists 961 flare images.

    See Also: :class:`mon.vision.dataset.base.dataset.UnlabeledImageDataset`.
    """

    splits = ["train"]

    def get_images(self):
        """Get image files."""
        patterns = [
            # self.root / self.split / "light-effect" / "clear",
            self.root / self.split / "light-effect" / "light-effects",
        ]
        self.images: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    list(pattern.rglob("*")),
                    description=f"Listing {self.__class__.__name__} {self.split} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self.images.append(image)


@DATASETS.register(name="mipiflare")
class MIPIFlare(base.ImageEnhancementDataset):
    """Combined Flare dataset used in MIPI 2024 Challenge
    `<https://mipi-challenge.org/MIPI2024/index.html>`__
    
    See Also: :class:`mon.vision.dataset.base.dataset.UnlabeledImageDataset`.
    """
    
    splits = ["train", "test"]
    
    def get_images(self):
        """Get image files."""
        if self.split == "train":
            patterns = [
                self.root / "test"  / "flare7k++-real" / "flare",
                self.root / "test"  / "flare7k++-syn"  / "flare",
                self.root / "train" / "flarereal800"   / "flare",
            ]
        elif self.split == "test":
            patterns = [
                self.root / "test" / "flare7k++-real"  / "flare",
            ]
        else:
            patterns = []
        self.images: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
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
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = str(img.path).replace("/flare/", "/clear/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)


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
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
       
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        if phase in [None, ModelPhase.TRAINING]:
            self.train = Flare7KPPReal(split="test", **self.dataset_kwargs)
            self.val   = Flare7KPPReal(split="test", **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = Flare7KPPReal(split="test", **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
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
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        if phase in [None, ModelPhase.TRAINING]:
            self.train = Flare7KPPSyn(split="test", **self.dataset_kwargs)
            self.val   = Flare7KPPSyn(split="test", **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = Flare7KPPSyn(split="test", **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="flarereal800")
class FlareReal800DataModule(base.DataModule):
    """FlareReal800 datamodule.
    
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
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        if phase in [None, ModelPhase.TRAINING]:
            self.train = FlareReal800(split="train", **self.dataset_kwargs)
            self.val   = FlareReal800(split="val",   **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = FlareReal800(split="val",   **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="ledlight")
class LEDLightDataModule(base.DataModule):
    """LEDLight datamodule.

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
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        if phase in [None, ModelPhase.TRAINING]:
            self.train = LEDLight(split="test", **self.dataset_kwargs)
            self.val   = LEDLight(split="test", **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = LEDLight(split="test", **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="light-effect")
@DATAMODULES.register(name="lighteffect")
class LightEffectDataModule(base.DataModule):
    """LightEffect datamodule.

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
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        if phase in [None, ModelPhase.TRAINING]:
            self.train = LightEffect(split="train", **self.dataset_kwargs)
            self.val   = LightEffect(split="train", **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = LightEffect(split="train", **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="mipiflare")
class MIPIFlareDataModule(base.DataModule):
    """Combined Flare datamodule used in MIPI 2024 Challenge
    `<https://mipi-challenge.org/MIPI2024/index.html>`__
    
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
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        if phase in [None, ModelPhase.TRAINING]:
            self.train = MIPIFlare(split="train", **self.dataset_kwargs)
            self.val   = MIPIFlare(split="test", **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = MIPIFlare(split="test", **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass

# endregion
