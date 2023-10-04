#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Rain13K datasets and datamodules."""

from __future__ import annotations

__all__ = [
    "Rain100", "Rain100DataModule", "Rain100H", "Rain100HDataModule",
    "Rain100L", "Rain100LDataModule", "Rain12", "Rain1200",
    "Rain1200DataModule", "Rain12DataModule", "Rain13K", "Rain13KDataModule",
    "Rain1400", "Rain1400DataModule", "Rain2800", "Rain2800DataModule",
    "Rain800", "Rain800DataModule",
]

import glob

from torch.utils.data import random_split

from mon.core import console, pathlib, rich
from mon.globals import DATAMODULES, DATASETS, ModelPhase
from mon.vision.dataset import base


# region Dataset

@DATASETS.register(name="rain100")
class Rain100(base.ImageEnhancementDataset):
    """Rain100 dataset consists 100 pairs of rain/no-rain test images and 100
    pairs of rain/no-rain train-val images.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    def get_images(self):
        """Get image files."""
        if self.split not in ["test"]:
            console.log(
                f"split must be one of ['test'], but got {self.split}."
            )
        
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / self.split
            for path in pbar.track(
                list(pattern.rglob("rain/*.png")),
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
                path  = str(img.path).replace("rain", "no_rain")
                path  = pathlib.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)


@DATASETS.register(name="rain100h")
class Rain100H(base.ImageEnhancementDataset):
    """Rain100H dataset consists 100 pairs of rain/no-rain test images and 100
    pairs of rain/no-rain train-val images.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    def get_images(self):
        """Get image files."""
        if self.split not in ["train", "test"]:
            console.log(
                f"split must be one of ['train', 'test'], but got {self.split}."
            )
        
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / self.split
            for path in pbar.track(
                list(pattern.rglob("rain/*.png")),
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
                path  = str(img.path).replace("rain", "no_rain")
                path  = pathlib.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)


@DATASETS.register(name="rain100l")
class Rain100L(base.ImageEnhancementDataset):
    """Rain100L dataset consists 100 pairs of rain/no-rain test images and 200
    pairs of rain/no-rain train-val images.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    def get_images(self):
        """Get image files."""
        if self.split not in ["train", "test"]:
            console.log(
                f"split must be one of ['train', 'test'], but got {self.split}."
            )
        
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / self.split
            for path in pbar.track(
                list(pattern.rglob("rain/*.png")),
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
                path  = str(img.path).replace("rain", "no_rain")
                path  = pathlib.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)


@DATASETS.register(name="rain12")
class Rain12(base.ImageEnhancementDataset):
    """Rain12 dataset consists 12 pairs of rain/no-rain images.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    def get_images(self):
        """Get image files."""
        if self.split not in ["train"]:
            console.log(
                f"split must be one of ['train'], but got {self.split}."
            )
        
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / self.split
            for path in pbar.track(
                list(pattern.rglob("rain/*.png")),
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
                path  = str(img.path).replace("rain", "no_rain")
                path  = pathlib.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)


@DATASETS.register(name="rain1200")
class Rain1200(base.ImageEnhancementDataset):
    """Rain1200 dataset consists 1200 pairs of rain/no-rain test images and
    12,000 pairs of rain/no-rain train images.
    
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
                list(pattern.rglob("rain/*.jpg")),
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
                path  = str(img.path).replace("rain", "no_rain")
                path  = pathlib.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)


@DATASETS.register(name="rain13k")
class Rain13K(base.ImageEnhancementDataset):
    """Rain13K dataset consists 13k pairs of rain/no-rain train images.
    
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
            pattern = self.root / "*" / self.split / "rain" / "*"
            for path in pbar.track(
                list(glob.glob(str(pattern))),
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
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split} labels"
            ):
                path  = img.path
                grand_parent = path.parent.parent
                name  = path.name
                path  = grand_parent / "no_rain" / name
                path  = pathlib.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)


@DATASETS.register(name="rain1400")
class Rain1400(base.ImageEnhancementDataset):
    """Rain1400 dataset consists 1400 pairs of rain/no-rain test images and
    12,600 pairs of rain/no-rain train images.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    def get_images(self):
        """Get image files."""
        if self.split not in ["train", "test"]:
            console.log(
                f"split must be one of ['train', 'test'], but got {self.split}."
            )
        
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / self.split
            for path in pbar.track(
                list(pattern.rglob("rain/*.jpg")),
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
                path  = str(img.path).replace("rain", "no_rain")
                path  = pathlib.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)


@DATASETS.register(name="rain2800")
class Rain2800(base.ImageEnhancementDataset):
    """Rain2800 dataset consists 2800 pairs of rain/no-rain test images.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    def get_images(self):
        """Get image files."""
        if self.split not in ["test"]:
            console.log(
                f"split must be one of ['test'], but got {self.split}."
            )
        
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / self.split
            for path in pbar.track(
                list(pattern.rglob("rain/*.jpg")),
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
                path  = str(img.path).replace("rain", "no_rain")
                path  = pathlib.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)


@DATASETS.register(name="rain800")
class Rain800(base.ImageEnhancementDataset):
    """Rain800 dataset consists 800 pairs of rain/no-rain train-val images.
    
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
                list(pattern.rglob("rain/*.jpg")),
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
                path  = str(img.path).replace("rain", "no_rain")
                path  = pathlib.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)

# endregion


# region Datamodule

@DATAMODULES.register(name="rain100")
class Rain100DataModule(base.DataModule):
    """Rain100 datamodule.
    
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
            dataset = Rain100(split="test", **self.dataset_kwargs)
            self.split_train_val(dataset=dataset, split_ratio=0.8, full_train=True)
        if phase in [None, ModelPhase.TESTING]:
            self.test = Rain100(split="test", **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="rain100h")
class Rain100HDataModule(base.DataModule):
    """Rain100H datamodule.
    
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
            dataset = Rain100H(split="train", **self.dataset_kwargs)
            self.split_train_val(dataset=dataset, split_ratio=0.8, full_train=True)
        if phase in [None, ModelPhase.TESTING]:
            self.test = Rain100H(split="test", **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="rain100l")
class Rain100LDataModule(base.DataModule):
    """Rain100L datamodule.
    
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
            dataset = Rain100L(split="train", **self.dataset_kwargs)
            self.split_train_val(dataset=dataset, split_ratio=0.8, full_train=True)
        if phase in [None, ModelPhase.TESTING]:
            self.test = Rain100L(split="test", **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="rain12")
class Rain12DataModule(base.DataModule):
    """Rain12 datamodule.
    
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
            dataset = Rain12(split="train", **self.dataset_kwargs)
            self.split_train_val(dataset=dataset, split_ratio=0.8, full_train=True)
        if phase in [None, ModelPhase.TESTING]:
            self.test = Rain12(split="train", **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="rain1200")
class Rain1200DataModule(base.DataModule):
    """Rain1200 datamodule.
    
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
            self.train = Rain1200(split="train", **self.dataset_kwargs)
            self.val   = Rain1200(split="val",   **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = Rain1200(split="test", **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="rain13k")
class Rain13KDataModule(base.DataModule):
    """Rain13K datamodule.
    
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
            self.train = Rain13K(split="train", **self.dataset_kwargs)
            self.val   = Rain13K(split="val",   **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = Rain13K(split="test", **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="rain1400")
class Rain1400DataModule(base.DataModule):
    """Rain1400 datamodule.
    
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
            dataset = Rain1400(split="train", **self.dataset_kwargs)
            self.split_train_val(dataset=dataset, split_ratio=0.8, full_train=True)
        if phase in [None, ModelPhase.TESTING]:
            self.test = Rain1400(split="test", **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="rain2800")
class Rain2800DataModule(base.DataModule):
    """Rain2800 datamodule.
    
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
            dataset = Rain2800(split="test", **self.dataset_kwargs)
            self.split_train_val(dataset=dataset, split_ratio=0.8, full_train=True)
        if phase in [None, ModelPhase.TESTING]:
            self.test = Rain2800(split="test", **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="rain800")
class Rain800DataModule(base.DataModule):
    """Rain800 datamodule.
    
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
            self.train = Rain800(split="train", **self.dataset_kwargs)
            self.val   = Rain800(split="val",   **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = Rain800(split="test", **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass

# endregion
