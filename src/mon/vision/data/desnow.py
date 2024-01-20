#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements de-snowing datasets and datamodules."""

from __future__ import annotations

__all__ = [
    "GTSnow",
    "GTSnowDataModule",
    "KITTISnow",
    "KITTISnowDataModule",
    "KITTISnowL",
    "KITTISnowLDataModule",
    "KITTISnowM",
    "KITTISnowMDataModule",
    "KITTISnowS",
    "KITTISnowSDataModule",
    "Snow100K",
    "Snow100KDataModule",
    "Snow100KL",
    "Snow100KLDataModule",
    "Snow100KM",
    "Snow100KMDataModule",
    "Snow100KS",
    "Snow100KSDataModule",
]

from mon.globals import DATAMODULES, DATASETS, ModelPhase
from mon.vision import core
from mon.vision.data import base

console = core.console


# region Dataset

@DATASETS.register(name="gtsnow")
@DATASETS.register(name="gt-snow")
class GTSnow(base.ImageEnhancementDataset):
    """GT-Snow dataset.

    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """

    splits = ["train"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "gt-snow" / "snow"
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
                path  = str(img.path)
                path  = path[:-9] + "C-000.png"
                path  = path.replace("/snow/", "/clear/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="kitti-snow")
class KITTISnow(base.ImageEnhancementDataset):
    """KITTI-Snow dataset.

    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """

    splits = ["train"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "kitti-snow" / "snow"
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
                path  = str(img.path).replace("/snow/", "/clear/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="kitti-snow-s")
class KITTISnowS(base.ImageEnhancementDataset):
    """KITTI-Snow-S dataset.

    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """

    splits = ["test"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "kitti-snow-s" / "snow"
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
                path  = str(img.path).replace("/snow/", "/clear/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="kitti-snow-m")
class KITTISnowM(base.ImageEnhancementDataset):
    """KITTI-Snow-M dataset.

    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """

    splits = ["test"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "kitti-snow-m" / "snow"
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
                path  = str(img.path).replace("/snow/", "/clear/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="kitti-snow-l")
class KITTISnowL(base.ImageEnhancementDataset):
    """KITTI-Snow-L dataset.

    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """

    splits = ["test"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "kitti-snow-l" / "snow"
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
                path  = str(img.path).replace("/snow/", "/clear/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="snow100k")
class Snow100K(base.ImageEnhancementDataset):
    """Snow100K dataset.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """

    splits = ["train"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "snow100k" / "snow"
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
                path  = str(img.path).replace("/snow/", "/clear/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="snow100k-s")
class Snow100KS(base.ImageEnhancementDataset):
    """Snow100K-S dataset.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """

    splits = ["test"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "snow100k-s" / "snow"
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
                path  = str(img.path).replace("/snow/", "/clear/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="snow100k-m")
class Snow100KM(base.ImageEnhancementDataset):
    """Snow100K-M dataset.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    splits = ["test"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "snow100k-m" / "snow"
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
                path  = str(img.path).replace("/snow/", "/clear/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="snow100k-l")
class Snow100KL(base.ImageEnhancementDataset):
    """Snow100K-L dataset.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    splits = ["test"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "snow100k-l" / "snow"
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
                path  = str(img.path).replace("/snow/", "/clear/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)

# endregion


# region Datamodule

@DATAMODULES.register(name="gtsnow")
@DATAMODULES.register(name="gt-snow")
class GTSnowDataModule(base.DataModule):
    """GT-Snow datamodule.

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
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        if phase in [None, ModelPhase.TRAINING]:
            self.train = GTSnow(split="train", **self.dataset_kwargs)
            self.val   = GTSnow(split="train", **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = GTSnow(split="train", **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()

        self.summarize()

    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="kitti-snow")
class KITTISnowDataModule(base.DataModule):
    """KITTI-Snow datamodule.

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
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
       
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        if phase in [None, ModelPhase.TRAINING]:
            self.train = KITTISnow(split="train", **self.dataset_kwargs)
            self.val   = KITTISnow(split="train", **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = KITTISnow(split="train", **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="kitti-snow-s")
class KITTISnowSDataModule(base.DataModule):
    """KITTI-Snow-S datamodule.

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
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        if phase in [None, ModelPhase.TRAINING]:
            self.train = KITTISnowS(split="test", **self.dataset_kwargs)
            self.val   = KITTISnowS(split="test", **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = KITTISnowS(split="test", **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="kitti-snow-m")
class KITTISnowMDataModule(base.DataModule):
    """KITTI-Snow-M datamodule.

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
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        if phase in [None, ModelPhase.TRAINING]:
            self.train = KITTISnowM(split="test", **self.dataset_kwargs)
            self.val   = KITTISnowM(split="test", **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = KITTISnowM(split="test", **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="kitti-snow-l")
class KITTISnowLDataModule(base.DataModule):
    """KITTI-Snow-L datamodule.

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
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        if phase in [None, ModelPhase.TRAINING]:
            self.train = KITTISnowL(split="test", **self.dataset_kwargs)
            self.val   = KITTISnowL(split="test", **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = KITTISnowL(split="test", **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="snow100k")
class Snow100KDataModule(base.DataModule):
    """Snow100K datamodule.
    
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
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        if phase in [None, ModelPhase.TRAINING]:
            self.train = Snow100K(split="train", **self.dataset_kwargs)
            self.val   = Snow100K(split="train", **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = Snow100K(split="train", **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="snow100k-s")
class Snow100KSDataModule(base.DataModule):
    """Snow100K-S datamodule.
    
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
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        if phase in [None, ModelPhase.TRAINING]:
            self.train = Snow100KS(split="train", **self.dataset_kwargs)
            self.val   = Snow100KS(split="test",  **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = Snow100KS(split="test",  **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="snow100k-m")
class Snow100KMDataModule(base.DataModule):
    """Snow100K-M datamodule.
    
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
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        if phase in [None, ModelPhase.TRAINING]:
            self.train = Snow100KM(split="train", **self.dataset_kwargs)
            self.val   = Snow100KM(split="test",  **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = Snow100KM(split="test",  **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="snow100k-l")
class Snow100KLDataModule(base.DataModule):
    """Snow100K-L datamodule.
    
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
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        if phase in [None, ModelPhase.TRAINING]:
            self.train = Snow100KL(split="train", **self.dataset_kwargs)
            self.val   = Snow100KL(split="test",  **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = Snow100KL(split="test",  **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass

# endregion
