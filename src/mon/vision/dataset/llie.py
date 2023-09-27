#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Low-Light Image Enhancement (LLIE) datasets and
datamodules.
"""

from __future__ import annotations

__all__ = [
    "DICM",
    "DICMDataModule",
    "DarkFace",
    "DeepUPE",
    "ExDark",
    "Fusion",
    "GLADNet",
    "LIME",
    "LIMEDataModule",
    "LOL",
    "LOL123",
    "LOL123DataModule",
    "LOLDataModule",
    "MEF",
    "MEFDataModule",
    "NPE",
    "NPEDataModule",
    "SICEGrad",
    "SICEGradDataModule",
    "SICEMix",
    "SICEMixDataModule",
    "SICEZeroDCE",
    "SICEZeroDCEDataModule",
    "VELOL",
    "VELOLSyn",
    "VV",
    "VVDataModule",
]

from torch.utils.data import random_split

from mon.core import console, pathlib, rich
from mon.globals import DATAMODULES, DATASETS, ModelPhase
from mon.vision.dataset import base


# region Dataset

@DATASETS.register(name="darkface")
class DarkFace(base.UnlabeledImageDataset):
    """DarkFace dataset consists of 6490 low-light images.
    
    See Also: :class:`mon.vision.dataset.base.dataset.UnlabeledImageDataset`.
    """
    
    splits = ["test"]
    
    def get_images(self):
        """Get image files."""
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / "test" / "darkface" / "low"
            for path in pbar.track(
                list(pattern.rglob("*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    image = base.ImageLabel(path=path)
                    self.images.append(image)


@DATASETS.register(name="deepupe")
class DeepUPE(base.ImageEnhancementDataset):
    """DeepUPE dataset consists of 500 paired images for tesing.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    splits = ["test"]
    
    def get_images(self):
        """Get image files."""
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / "test" / "lol" / "low"
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


@DATASETS.register(name="dicm")
class DICM(base.UnlabeledImageDataset):
    """DICM dataset consists of 64 low-light images.
    
    See Also: :class:`mon.vision.dataset.base.dataset.UnlabeledImageDataset`.
    """
    
    splits = ["test"]
    
    def get_images(self):
        """Get image files."""
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / "test" / "dicm" / "low"
            for path in pbar.track(
                list(pattern.rglob("*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    image = base.ImageLabel(path=path)
                    self.images.append(image)


@DATASETS.register(name="exdark")
class ExDark(base.UnlabeledImageDataset):
    """ExDark dataset consists of 7363 low-light images.
    
    See Also: :class:`mon.vision.dataset.base.dataset.UnlabeledImageDataset`.
    """
    
    splits = ["test"]
    
    def get_images(self):
        """Get image files."""
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / "test" / "exdark" / "low"
            for path in pbar.track(
                list(pattern.rglob("*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    image = base.ImageLabel(path=path)
                    self.images.append(image)


@DATASETS.register(name="fusion")
class Fusion(base.UnlabeledImageDataset):
    """Fusion dataset consists of 64 low-light images.
    
    See Also: :class:`mon.vision.dataset.base.dataset.UnlabeledImageDataset`.
    """
    
    splits = ["test"]
    
    def get_images(self):
        """Get image files."""
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / "test" / "fusion" / "low"
            for path in pbar.track(
                list(pattern.rglob("*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    image = base.ImageLabel(path=path)
                    self.images.append(image)


@DATASETS.register(name="gladnet")
class GLADNet(base.ImageEnhancementDataset):
    """GLADNet dataset consists of 589 low-light and normal-light image pairs
    for training.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    splits = ["train"]
    
    def get_images(self):
        """Get image files."""
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / self.split / "gladnet" / "low"
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
    
    splits = ["train", "test"]
    
    def get_images(self):
        """Get image files."""
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / self.split / "lol" / "low"
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


@DATASETS.register(name="lol123")
class LOL123(base.UnlabeledImageDataset):
    """LOL123 dataset consists of 123 low-light images.
    
    See Also: :class:`mon.vision.dataset.base.dataset.UnlabeledImageDataset`.
    """
    
    splits = ["test"]
    
    def get_images(self):
        """Get image files."""
        subdirs = ["dicm", "lime", "mef", "npe", "vv"]
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            for subdir in subdirs:
                pattern = self.root / "test" / subdir / "low"
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
    
    splits = ["test"]
    
    def get_images(self):
        """Get image files."""
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / "test" / "mef" / "low"
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
    
    splits = ["test"]
    
    def get_images(self):
        """Get image files."""
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / "test" / "npe" / "low"
            for path in pbar.track(
                list(pattern.rglob("*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    image = base.ImageLabel(path=path)
                    self.images.append(image)


@DATASETS.register(name="sice-grad")
class SICEGrad(base.ImageEnhancementDataset):
    """SICEGrad dataset consists of 589 low-light and normal-light image pairs.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    splits = ["train"]
    
    def get_images(self):
        """Get image files."""
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / self.split / "sice-grad" / "low"
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


@DATASETS.register(name="sice-mix")
class SICEMix(base.ImageEnhancementDataset):
    """SICEMix dataset consists of 589 low-light and normal-light image pairs.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    splits = ["train"]
    
    def get_images(self):
        """Get image files."""
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / self.split / "sice-mix" / "low"
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


@DATASETS.register(name="sice-zerodce")
class SICEZeroDCE(base.UnlabeledImageDataset):
    """Custom SICE dataset for training :class:`mon.vision.enhance.llie.zerodce.ZeroDCE`
    model.
    
    See Also: :class:`mon.vision.dataset.base.dataset.UnlabeledImageDataset`.
    """
    
    splits = ["train"]
    
    def get_images(self):
        """Get image files."""
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / "train" / "sice-zerodce" / "low"
            for path in pbar.track(
                list(pattern.rglob("*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    image = base.ImageLabel(path=path)
                    self.images.append(image)


@DATASETS.register(name="ve-lol")
class VELOL(base.ImageEnhancementDataset):
    """VE-LOL dataset consists of 500 low-light and normal-light image pairs.
    They are divided into 400 training pairs and 100 testing pairs. The
    low-light images contain noise produced during the photo capture process.
    Most of the images are indoor scenes. All the images have a resolution of
    400×600.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    splits = ["train", "test"]
    
    def get_images(self):
        """Get image files."""
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / self.split / "ve-lol" / "low"
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


@DATASETS.register(name="ve-lol-syn")
class VELOLSyn(base.ImageEnhancementDataset):
    """VE-LOL-Syn dataset consists of 1000 low-light and normal-light image
    pairs. They are divided into 900 training pairs and 100 testing pairs. The
    low-light images contain noise produced during the photo capture process.
    Most of the images are indoor scenes. All the images have a resolution of
    400×600.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    splits = ["train", "test"]
    
    def get_images(self):
        """Get image files."""
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / self.split / "ve-lol-sync" / "low"
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


@DATASETS.register(name="vv")
class VV(base.UnlabeledImageDataset):
    """VV dataset consists of 24 low-light images.
    
    See Also: :class:`mon.vision.dataset.base.dataset.UnlabeledImageDataset`.
    """
    
    splits = ["test"]
    
    def get_images(self):
        """Get image files."""
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / "test" / "vv" / "low"
            for path in pbar.track(
                list(pattern.rglob("*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    image = base.ImageLabel(path=path)
                    self.images.append(image)

# endregion


# region Datamodule

@DATAMODULES.register(name="darkface")
class DarkFaceDataModule(base.DataModule):
    """DarkFace datamodule.
    
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
            train      = DarkFace(split="test", **self.dataset_kwargs)
            train_size = int(0.8 * len(train))
            val_size   = len(train) - train_size
            self.train, self.val = random_split(train, [train_size, val_size])
            self.classlabels = getattr(train, "classlabels", None)
            self.collate_fn  = getattr(train, "collate_fn",  None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = DarkFace(split="test", **self.dataset_kwargs)
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="deepupe")
class DeepUPEDataModule(base.DataModule):
    """DeepUPE datamodule.
    
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
            train      = DeepUPE(split="test", **self.dataset_kwargs)
            train_size = int(0.8 * len(train))
            val_size   = len(train) - train_size
            self.train, self.val = random_split(train, [train_size, val_size])
            self.classlabels = getattr(train, "classlabels", None)
            self.collate_fn  = getattr(train, "collate_fn",  None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = DeepUPE(split="test", **self.dataset_kwargs)
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="dicm")
class DICMDataModule(base.DataModule):
    """DICM datamodule.
    
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
            train      = DICM(split="test", **self.dataset_kwargs)
            train_size = int(0.8 * len(train))
            val_size   = len(train) - train_size
            self.train, self.val = random_split(train, [train_size, val_size])
            self.classlabels = getattr(train, "classlabels", None)
            self.collate_fn  = getattr(train, "collate_fn",  None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = DICM(split="test", **self.dataset_kwargs)
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="exdark")
class ExDarkDataModule(base.DataModule):
    """ExDark datamodule.
    
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
            train      = ExDark(split="test", **self.dataset_kwargs)
            train_size = int(0.8 * len(train))
            val_size   = len(train) - train_size
            self.train, self.val = random_split(train, [train_size, val_size])
            self.classlabels = getattr(train, "classlabels", None)
            self.collate_fn  = getattr(train, "collate_fn",  None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = ExDark(split="test", **self.dataset_kwargs)
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="fusion")
class FusionDataModule(base.DataModule):
    """Fusion datamodule.
    
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
            train      = Fusion(split="test", **self.dataset_kwargs)
            train_size = int(0.8 * len(train))
            val_size   = len(train) - train_size
            self.train, self.val = random_split(train, [train_size, val_size])
            self.classlabels = getattr(train, "classlabels", None)
            self.collate_fn  = getattr(train, "collate_fn",  None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = Fusion(split="test", **self.dataset_kwargs)
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
                - ``'training'`` : prepares :attr:'train' and :attr:'val'.
                - ``'testing'``  : prepares :attr:'test'.
                - ``'inference'``: prepares :attr:`predict`.
                - ``None``:      : prepares all.
                - Default: ``None``.
        """
        console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        
        if phase in [None, ModelPhase.TRAINING]:
            train      = LIME(split="test", **self.dataset_kwargs)
            train_size = int(0.8 * len(train))
            val_size   = len(train) - train_size
            self.train, self.val = random_split(train, [train_size, val_size])
            self.classlabels = getattr(train, "classlabels", None)
            self.collate_fn  = getattr(train, "collate_fn",  None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = LIME(split="test", **self.dataset_kwargs)
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
                - ``'training'`` : prepares :attr:'train' and :attr:'val'.
                - ``'testing'``  : prepares :attr:'test'.
                - ``'inference'``: prepares :attr:`predict`.
                - ``None``:      : prepares all.
                - Default: ``None``.
        """
        console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        
        if phase in [None, ModelPhase.TRAINING]:
            self.train       = LOL(split="train", **self.dataset_kwargs)
            self.val         = LOL(split="test",  **self.dataset_kwargs)
            self.classlabels = getattr(self.train, "classlabels", None)
            self.collate_fn  = getattr(self.train, "collate_fn",  None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = LOL(split="test", **self.dataset_kwargs)
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="lol123")
class LOL123DataModule(base.DataModule):
    """LOL123 datamodule.
    
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
            train      = LOL123(split="test", **self.dataset_kwargs)
            train_size = int(0.8 * len(train))
            val_size   = len(train) - train_size
            self.train, self.val = random_split(train, [train_size, val_size])
            self.classlabels = getattr(train, "classlabels", None)
            self.collate_fn  = getattr(train, "collate_fn",  None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = LOL123(split="test", **self.dataset_kwargs)
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
                - ``'training'`` : prepares :attr:'train' and :attr:'val'.
                - ``'testing'``  : prepares :attr:'test'.
                - ``'inference'``: prepares :attr:`predict`.
                - ``None``:      : prepares all.
                - Default: ``None``.
        """
        console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        
        if phase in [None, ModelPhase.TRAINING]:
            train      = MEF(split="test", **self.dataset_kwargs)
            train_size = int(0.8 * len(train))
            val_size   = len(train) - train_size
            self.train, self.val = random_split(train, [train_size, val_size])
            self.classlabels = getattr(train, "classlabels", None)
            self.collate_fn  = getattr(train, "collate_fn",  None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = MEF(split="test", **self.dataset_kwargs)
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
            train      = NPE(split="test", **self.dataset_kwargs)
            train_size = int(0.8 * len(train))
            val_size   = len(train) - train_size
            self.train, self.val = random_split(train, [train_size, val_size])
            self.classlabels = getattr(train, "classlabels", None)
            self.collate_fn  = getattr(train, "collate_fn",  None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = NPE(split="test", **self.dataset_kwargs)
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="sice-grad")
class SICEGradDataModule(base.DataModule):
    """SICEGrad datamodule.
    
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
            train      = SICEGrad(split="train", **self.dataset_kwargs)
            train_size = int(0.8 * len(train))
            val_size   = len(train) - train_size
            self.train, self.val = random_split(train, [train_size, val_size])
            self.classlabels = getattr(train, "classlabels", None)
            self.collate_fn  = getattr(train, "collate_fn",  None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = SICEGrad(split="train", **self.dataset_kwargs)
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)

        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="sice-mix")
class SICEMixDataModule(base.DataModule):
    """SICEMix datamodule.
    
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
            train      = SICEMix(split="train", **self.dataset_kwargs)
            train_size = int(0.8 * len(train))
            val_size   = len(train) - train_size
            self.train, self.val = random_split(train, [train_size, val_size])
            self.classlabels = getattr(train, "classlabels", None)
            self.collate_fn  = getattr(train, "collate_fn",  None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = SICEMix(split="train", **self.dataset_kwargs)
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)

        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="sice-zerodce")
class SICEZeroDCEDataModule(base.DataModule):
    """SICEZeroDCE datamodule.
    
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
            train      = SICEZeroDCE(split="train", **self.dataset_kwargs)
            train_size = int(0.8 * len(train))
            val_size   = len(train) - train_size
            self.train, self.val = random_split(train, [train_size, val_size])
            self.classlabels = getattr(train, "classlabels", None)
            self.collate_fn  = getattr(train, "collate_fn",  None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = SICEZeroDCE(split="train", **self.dataset_kwargs)
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
                - ``'training'`` : prepares :attr:'train' and :attr:'val'.
                - ``'testing'``  : prepares :attr:'test'.
                - ``'inference'``: prepares :attr:`predict`.
                - ``None``:      : prepares all.
                - Default: ``None``.
        """
        console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        
        if phase in [None, ModelPhase.TRAINING]:
            train      = VV(split="test", **self.dataset_kwargs)
            train_size = int(0.8 * len(train))
            val_size   = len(train) - train_size
            self.train, self.val = random_split(train, [train_size, val_size])
            self.classlabels = getattr(train, "classlabels", None)
            self.collate_fn  = getattr(train, "collate_fn",  None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = VV(split="test", **self.dataset_kwargs)
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass

# endregion
