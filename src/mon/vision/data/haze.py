#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements SateHaze1K datasets and datamodules."""

from __future__ import annotations

__all__ = [
    "DenseHaze",
    "DenseHazeDataModule",
    "IHaze",
    "IHazeDataModule",
    "NHHaze",
    "NHHazeDataModule",
    "OHaze",
    "OHazeDataModule",
    "SateHaze1K",
    "SateHaze1KDataModule",
    "SateHaze1KModerate",
    "SateHaze1KModerateDataModule",
    "SateHaze1KThick",
    "SateHaze1KThickDataModule",
    "SateHaze1KThin",
    "SateHaze1KThinDataModule",
]

from mon.globals import DATAMODULES, DATASETS, ModelPhase
from mon.vision import core
from mon.vision.data import base

console = core.console


# region Dataset

@DATASETS.register(name="dense-haze")
class DenseHaze(base.ImageEnhancementDataset):
    """Dense-Haze dataset consists of 33 pairs of real hazy and corresponding
    haze-free images.
    
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
        with core.get_progress_bar() as pbar:
            pattern = self.root / self.split
            for path in pbar.track(
                list(pattern.rglob("haze/*.png")),
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
                path  = str(img.path).replace("haze", "hazefree")
                path  = core.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)


@DATASETS.register(name="i-haze")
class IHaze(base.ImageEnhancementDataset):
    """I-Haze dataset consists of 35 pairs of real hazy and corresponding
    haze-free images.
    
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
        with core.get_progress_bar() as pbar:
            pattern = self.root / self.split
            for path in pbar.track(
                list(pattern.rglob("haze/*.jpg")),
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
                description=f"Listing I-Haze {self.split} labels"
            ):
                path  = str(img.path).replace("haze", "hazefree")
                path  = core.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)


@DATASETS.register(name="nh-haze")
class NHHaze(base.ImageEnhancementDataset):
    """NH-Haze dataset consists 55 pairs of real hazy and corresponding
    haze-free images.
    
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
        with core.get_progress_bar() as pbar:
            pattern = self.root / self.split
            for path in pbar.track(
                list(pattern.rglob("haze/*.png")),
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
                path  = str(img.path).replace("haze", "hazefree")
                path  = core.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)


@DATASETS.register(name="o-haze")
class OHaze(base.ImageEnhancementDataset):
    """O-Haze dataset consists of 45 pairs of real hazy and corresponding
    haze-free images.
    
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
        with core.get_progress_bar() as pbar:
            pattern = self.root / self.split
            for path in pbar.track(
                list(pattern.rglob("haze/*.jpg")),
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
                path  = str(img.path).replace("haze", "hazefree")
                path  = core.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)
                

@DATASETS.register(name="satehaze1k")
class SateHaze1K(base.ImageEnhancementDataset):
    """SateHaze1K dataset consists 1200 pairs of hazy and corresponding
    haze-free images.
    
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
        with core.get_progress_bar() as pbar:
            pattern = self.root
            for path in pbar.track(
                list(pattern.rglob(f"{self.split}/input/*.png")),
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
                path  = str(img.path).replace("input", "target")
                path  = core.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)


@DATASETS.register(name="satehaze1k-thin")
class SateHaze1KThin(base.ImageEnhancementDataset):
    """SateHaze1K-Thin dataset.
    
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
        with core.get_progress_bar() as pbar:
            pattern = self.root / "thin" / self.split
            for path in pbar.track(
                list(pattern.rglob(f"input/*.png")),
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
                path  = str(img.path).replace("input", "target")
                path  = core.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)


@DATASETS.register(name="satehaze1k-moderate")
class SateHaze1KModerate(base.ImageEnhancementDataset):
    """SateHaze1K-Moderate.
    
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
        with core.get_progress_bar() as pbar:
            pattern = self.root / "moderate" / self.split
            for path in pbar.track(
                list(pattern.rglob(f"input/*.png")),
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
                path  = str(img.path).replace("input", "target")
                path  = core.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)


@DATASETS.register(name="satehaze1k-thick")
class SateHaze1KThick(base.ImageEnhancementDataset):
    """SateHaze1K-Thick dataset.
    
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
        with core.get_progress_bar() as pbar:
            pattern = self.root / "thick" / self.split
            for path in pbar.track(
                list(pattern.rglob(f"input/*.png")),
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
                path  = str(img.path).replace("input", "target")
                path  = core.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)

# endregion


# region Datamodule

@DATAMODULES.register(name="dense-haze")
class DenseHazeDataModule(base.DataModule):
    """Dense-Haze datamodule.
    
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
            self.train = DenseHaze(split="train", **self.dataset_kwargs)
            self.val   = DenseHaze(split="val",   **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = DenseHaze(split="test", **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="i-haze")
class IHazeDataModule(base.DataModule):
    """I-Haze datamodule.
    
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
            self.train = IHaze(split="train", **self.dataset_kwargs)
            self.val   = IHaze(split="val",   **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = IHaze(split="test", **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="nh-haze")
class NHHazeDataModule(base.DataModule):
    """NH-Haze datamodule.
    
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
            self.train = NHHaze(split="train", **self.dataset_kwargs)
            self.val   = NHHaze(split="val",   **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = NHHaze(split="test", **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="o-haze")
class OHazeDataModule(base.DataModule):
    """O-Haze datamodule.
    
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
            self.train = OHaze(split="train", **self.dataset_kwargs)
            self.val   = OHaze(split="val",   **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = OHaze(split="test", **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="satehaze1k")
class SateHaze1KDataModule(base.DataModule):
    """SateHaze1K datamodule.
    
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
                - "training" : prepares :attr:`train` and :attr:`val`.
                - "testing"  : prepares :attr:`test`.
                - "inference": prepares :attr:`predict`.
                - None:      : prepares all.
                Default: ``None``.
        """
        console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        
        if phase in [None, ModelPhase.TRAINING]:
            self.train       = SateHaze1K(split="train", **self.dataset_kwargs)
            self.val         = SateHaze1K(split="val",   **self.dataset_kwargs)
            self.classlabels = getattr(self.train, "classlabels", None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = SateHaze1K(split="test", **self.dataset_kwargs)
            self.classlabels = getattr(self.test, "classlabels", None)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="satehaze1k-thin")
class SateHaze1KThinDataModule(base.DataModule):
    """SateHaze1K-Thin datamodule.
    
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
            self.train       = SateHaze1KThin(split="train", **self.dataset_kwargs)
            self.val         = SateHaze1KThin(split="val",   **self.dataset_kwargs)
            self.classlabels = getattr(self.train, "classlabels", None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = SateHaze1KThin(split="test", **self.dataset_kwargs)
            self.classlabels = getattr(self.test, "classlabels", None)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="satehaze1k-moderate")
class SateHaze1KModerateDataModule(base.DataModule):
    """SateHaze1K-Moderate datamodule.
    
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
            self.train       = SateHaze1KModerate(split="train", **self.dataset_kwargs)
            self.val         = SateHaze1KModerate(split="val",   **self.dataset_kwargs)
            self.classlabels = getattr(self.train, "classlabels", None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = SateHaze1KModerate(split="test", **self.dataset_kwargs)
            self.classlabels = getattr(self.test, "classlabels", None)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="satehaze1k-thick")
class SateHaze1KThickDataModule(base.DataModule):
    """SateHaze1K-Thick datamodule.
    
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
            self.train = SateHaze1KThick(split="train", **self.dataset_kwargs)
            self.val   = SateHaze1KThick(split="val",   **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = SateHaze1KThick(split="test", **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass

# endregion
