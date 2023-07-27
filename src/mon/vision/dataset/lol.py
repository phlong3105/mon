#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements LOL datasets and datamodules."""

from __future__ import annotations

__all__ = [
    "DCIM", "DCIMDataModule", "LIME", "LIMEDataModule", "LOL", "LOL226",
    "LOL226DataModule", "LOL4K", "LOL4KDataModule", "LOLDataModule", "MEF",
    "MEFDataModule", "NPE", "NPEDataModule", "VIP", "VIPDataModule", "VV",
    "VVDataModule",
]

from torch.utils.data import random_split

from mon.core import console, pathlib, rich
from mon.globals import DATAMODULES, DATASETS, ModelPhase
from mon.vision.dataset import base


# region Dataset

@DATASETS.register(name="dcim")
class DCIM(base.UnlabeledImageDataset):
    """DCIM dataset consists of 64 low-light images.
    
    See Also: :class:`mon.vision.dataset.base.dataset.UnlabeledImageDataset`.
    """
    
    def get_images(self):
        """Get image files."""
        if self.split not in ["train"]:
            console.log(
                f"split must be one of ['train'], but got {self.split}."
            )
        
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / "train" / "low" / "dcim"
            for path in pbar.track(
                list(pattern.rglob("*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    image = base.ImageLabel(path=path)
                    self.images.append(image)


@DATASETS.register(name="lime")
class LIME(base.UnlabeledImageDataset):
    """LIME dataset consists of 10 low-light images.
    
    See Also: :class:`mon.vision.dataset.base.dataset.UnlabeledImageDataset`.
    """
    
    def get_images(self):
        """Get image files."""
        if self.split not in ["train"]:
            console.log(
                f"split must be one of ['train'], but got {self.split}."
            )
        
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / "train" / "low" / "lime"
            for path in pbar.track(
                list(pattern.rglob("*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    image = base.ImageLabel(path=path)
                    self.images.append(image)


@DATASETS.register(name="lol")
class LOL(base.ImageEnhancementDataset):
    """LOL dataset consists of 500 low-light and normal-light image pairs. They
    are divided into 485 training pairs and 15 testing pairs. The low-light
    images contain noise produced during the photo capture process. Most of the
    images are indoor scenes. All the images have a resolution of 400×600.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    def get_images(self):
        """Get image files."""
        if self.split not in ["train", "val"]:
            console.log(
                f"split must be one of ['train', 'val'], but got {self.split}."
            )
        
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / self.split / "low" / "lol"
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
        with rich.get_progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = str(img.path).replace("low", "high")
                path  = pathlib.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)


@DATASETS.register(name="lol226")
class LOL226(base.UnlabeledImageDataset):
    """LOL226 dataset consists of 226 low-light images.
    
    See Also: :class:`mon.vision.dataset.base.dataset.UnlabeledImageDataset`.
    """
    
    def get_images(self):
        """Get image files."""
        if self.split not in ["train"]:
            console.log(
                f"split must be one of ['train'], but got {self.split}."
            )
        
        subdirs = ["dcim", "fusion", "lime", "mef", "npe", "vip", "vv"]
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            for subdir in subdirs:
                pattern = self.root / "train" / "low" / subdir
                for path in pbar.track(
                    list(pattern.rglob("*")),
                    description=f"Listing {self.__class__.__name__} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self.images.append(image)


@DATASETS.register(name="lol4k")
class LOL4K(base.UnlabeledImageDataset):
    """LOL4k dataset consists of 3777 low-light images.
    
    See Also: :class:`mon.vision.dataset.base.dataset.UnlabeledImageDataset`.
    """
    
    def get_images(self):
        """Get image files."""
        if self.split not in ["train"]:
            console.log(
                f"split must be one of ['train'], but got {self.split}."
            )
        
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / "train" / "low"
            for path in pbar.track(
                list(pattern.rglob("*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    image = base.ImageLabel(path=path)
                    self.images.append(image)


@DATASETS.register(name="mef")
class MEF(base.UnlabeledImageDataset):
    """MEF dataset consists 17 low-light images.

    See Also: :class:`mon.vision.dataset.base.dataset.UnlabeledImageDataset`.
    """
    
    def get_images(self):
        """Get image files."""
        if self.split not in ["train"]:
            console.log(
                f"split must be one of ['train'], but got {self.split}."
            )
        
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / "train" / "low" / "mef"
            for path in pbar.track(
                list(pattern.rglob("*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    image = base.ImageLabel(path=path)
                    self.images.append(image)


@DATASETS.register(name="npe")
class NPE(base.UnlabeledImageDataset):
    """NPE dataset consists 85 low-light images.
    
    See Also: :class:`mon.vision.dataset.base.dataset.UnlabeledImageDataset`.
    """
    
    def get_images(self):
        """Get image files."""
        if self.split not in ["train"]:
            console.log(
                f"split must be one of ['train'], but got {self.split}."
            )
        
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / "train" / "low" / "npe"
            for path in pbar.track(
                list(pattern.rglob("*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    image = base.ImageLabel(path=path)
                    self.images.append(image)


@DATASETS.register(name="vip")
class VIP(base.UnlabeledImageDataset):
    """VIP dataset consists of 8 low-light images.
    
    See Also: :class:`mon.vision.dataset.base.dataset.UnlabeledImageDataset`.
    """
    
    def get_images(self):
        """Get image files."""
        if self.split not in ["train"]:
            console.log(
                f"split must be one of ['train'], but got {self.split}."
            )

        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / "train" / "low" / "vip"
            for path in pbar.track(
                list(pattern.rglob("*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    image = base.ImageLabel(path=path)
                    self.images.append(image)


@DATASETS.register(name="vv")
class VV(base.UnlabeledImageDataset):
    """VV dataset consists of 24 low-light images.
    
    See Also: :class:`mon.vision.dataset.base.dataset.UnlabeledImageDataset`.
    """
    
    def get_images(self):
        """Get image files."""
        if self.split not in ["train"]:
            console.log(
                f"split must be one of ['train'], but got {self.split}."
            )
        
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / "train" / "low" / "vv"
            for path in pbar.track(
                list(pattern.rglob("*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    image = base.ImageLabel(path=path)
                    self.images.append(image)

# endregion


# region Datamodule

@DATAMODULES.register(name="dcim")
class DCIMDataModule(base.DataModule):
    """DCIM datamodule.
    
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
                - "training" : prepares :attr:'train' and :attr:'val'.
                - "testing"  : prepares :attr:'test'.
                - "inference": prepares :attr:`predict`.
                - None:      : prepares all.
                Default: None.
        """
        console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        
        if phase in [None, ModelPhase.TRAINING]:
            train      = DCIM(split="train", **self.dataset_kwargs)
            train_size = int(0.8 * len(train))
            val_size   = len(train) - train_size
            self.train, self.val = random_split(train, [train_size, val_size])
            self.classlabels = getattr(train, "classlabels", None)
            self.collate_fn  = getattr(train, "collate_fn",  None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = DCIM(split="train", **self.dataset_kwargs)
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="lime")
class LIMEDataModule(base.DataModule):
    """LIME datamodule.
     
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
                - "training" : prepares :attr:'train' and :attr:'val'.
                - "testing"  : prepares :attr:'test'.
                - "inference": prepares :attr:`predict`.
                - None:      : prepares all.
                Default: None.
        """
        console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        
        if phase in [None, ModelPhase.TRAINING]:
            train      = LIME(split="train", **self.dataset_kwargs)
            train_size = int(0.8 * len(train))
            val_size   = len(train) - train_size
            self.train, self.val = random_split(train, [train_size, val_size])
            self.classlabels = getattr(train, "classlabels", None)
            self.collate_fn  = getattr(train, "collate_fn",  None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = LIME(split="train", **self.dataset_kwargs)
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="lol")
class LOLDataModule(base.DataModule):
    """LOL datamodule.
    
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
                Default: None.
        """
        console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        
        if phase in [None, ModelPhase.TRAINING]:
            self.train       = LOL(split="train", **self.dataset_kwargs)
            self.val         = LOL(split="val",   **self.dataset_kwargs)
            self.classlabels = getattr(self.train, "classlabels", None)
            self.collate_fn  = getattr(self.train, "collate_fn",  None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = LOL(split="val", **self.dataset_kwargs)
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="lol226")
class LOL226DataModule(base.DataModule):
    """LOL226 datamodule.
    
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
                - "training" : prepares :attr:'train' and :attr:'val'.
                - "testing"  : prepares :attr:'test'.
                - "inference": prepares :attr:`predict`.
                - None:      : prepares all.
                Default: None.
        """
        console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        
        if phase in [None, ModelPhase.TRAINING]:
            train      = LOL226(split="train", **self.dataset_kwargs)
            train_size = int(0.8 * len(train))
            val_size   = len(train) - train_size
            self.train, self.val = random_split(train, [train_size, val_size])
            self.classlabels = getattr(train, "classlabels", None)
            self.collate_fn  = getattr(train, "collate_fn",  None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = LOL226(split="train", **self.dataset_kwargs)
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)

        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="lol4k")
class LOL4KDataModule(base.DataModule):
    """LOL4K datamodule.
    
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
                Default: None.
        """
        console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        
        if phase in [None, ModelPhase.TRAINING]:
            train      = LOL4K(split="train", **self.dataset_kwargs)
            train_size = int(0.8 * len(train))
            val_size   = len(train) - train_size
            self.train, self.val = random_split(train, [train_size, val_size])
            self.classlabels = getattr(train, "classlabels", None)
            self.collate_fn  = getattr(train, "collate_fn",  None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = LOL4K(split="train", **self.dataset_kwargs)
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="mef")
class MEFDataModule(base.DataModule):
    """MEF datamodule.
    
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
                - "training" : prepares :attr:'train' and :attr:'val'.
                - "testing"  : prepares :attr:'test'.
                - "inference": prepares :attr:`predict`.
                - None:      : prepares all.
                Default: None.
        """
        console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        
        if phase in [None, ModelPhase.TRAINING]:
            train      = MEF(split="train", **self.dataset_kwargs)
            train_size = int(0.8 * len(train))
            val_size   = len(train) - train_size
            self.train, self.val = random_split(train, [train_size, val_size])
            self.classlabels = getattr(train, "classlabels", None)
            self.collate_fn  = getattr(train, "collate_fn",  None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = MEF(split="train", **self.dataset_kwargs)
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="npe")
class NPEDataModule(base.DataModule):
    """NPE datamodule.
    
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
                - "training" : prepares :attr:'train' and :attr:'val'.
                - "testing"  : prepares :attr:'test'.
                - "inference": prepares :attr:`predict`.
                - None:      : prepares all.
                Default: None.
        """
        console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        
        if phase in [None, ModelPhase.TRAINING]:
            train      = NPE(split="train", **self.dataset_kwargs)
            train_size = int(0.8 * len(train))
            val_size   = len(train) - train_size
            self.train, self.val = random_split(train, [train_size, val_size])
            self.classlabels = getattr(train, "classlabels", None)
            self.collate_fn  = getattr(train, "collate_fn",  None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = NPE(split="train", **self.dataset_kwargs)
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="vip")
class VIPDataModule(base.DataModule):
    """VIP datamodule.
    
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
                - "training" : prepares :attr:'train' and :attr:'val'.
                - "testing"  : prepares :attr:'test'.
                - "inference": prepares :attr:`predict`.
                - None:      : prepares all.
                Default: None.
        """
        console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        
        if phase in [None, ModelPhase.TRAINING]:
            train      = VIP(split="train", **self.dataset_kwargs)
            train_size = int(0.8 * len(train))
            val_size   = len(train) - train_size
            self.train, self.val = random_split(train, [train_size, val_size])
            self.classlabels = getattr(train, "classlabels", None)
            self.collate_fn  = getattr(train, "collate_fn",  None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = VIP(split="train", **self.dataset_kwargs)
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="vv")
class VVDataModule(base.DataModule):
    """VV datamodule.
    
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
                - "training" : prepares :attr:'train' and :attr:'val'.
                - "testing"  : prepares :attr:'test'.
                - "inference": prepares :attr:`predict`.
                - None:      : prepares all.
                Default: None.
        """
        console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        
        if phase in [None, ModelPhase.TRAINING]:
            train      = VV(split="train", **self.dataset_kwargs)
            train_size = int(0.8 * len(train))
            val_size   = len(train) - train_size
            self.train, self.val = random_split(train, [train_size, val_size])
            self.classlabels = getattr(train, "classlabels", None)
            self.collate_fn  = getattr(train, "collate_fn",  None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = VV(split="train", **self.dataset_kwargs)
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass

# endregion
