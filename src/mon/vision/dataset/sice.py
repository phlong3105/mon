#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements SICE datasets and datamodules."""

from __future__ import annotations

__all__ = [
    "SICE", "SICEDataModule", "SICEPart1", "SICEPart1_512", "SICEPart1_512_Low",
    "SICEPart2", "SICEPart2_512", "SICEPart2_512_Low", "SICEPart2_900",
    "SICEPart2_900_Low", "SICEPart2_Low", "SICEUDataModule",
]

from torch.utils.data import random_split

from mon.core import console, pathlib, rich
from mon.globals import DATAMODULES, DATASETS, ModelPhase
from mon.vision.dataset import base


# region Dataset

@DATASETS.register(name="sice")
class SICE(base.ImageEnhancementDataset):
    """Full SICE dataset consisting of Part 1 (360 sequences) and Part 2
    (229 sequences).
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    def get_images(self):
        """Get image files."""
        parts = ["part1", "part2"]
        
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            for part in parts:
                pattern = self.root / part / "low"
                for path in pbar.track(
                    list(pattern.rglob("*/*")),
                    description=f"Listing {self.__class__.__name__} images"
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
                description=f"Listing {self.__class__.__name__} labels"
            ):
                stem = str(img.path.parent.stem)
                if "part1" in str(img.path):
                    path = self.root / "part1" / "high" / f"{stem}.jpg"
                elif "part2" in str(img.path):
                    path = self.root / "part2" / "high" / f"{stem}.jpg"
                else:
                    path = ""
                path  = pathlib.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)


@DATASETS.register(name="sice-part1")
class SICEPart1(base.ImageEnhancementDataset):
    """SICE Part 1 dataset consists of 360 multi-exposure sequences.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    def get_images(self):
        """Get image files."""
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / "part1" / "low"
            for path in pbar.track(
                list(pattern.rglob("*/*")),
                description=f"Listing {self.__class__.__name__} images"
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
                description=f"Listing {self.__class__.__name__} labels"
            ):
                stem  = str(img.path.parent.stem)
                path  = self.root / "part1" / "high" / f"{stem}.jpg"
                path  = pathlib.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)


@DATASETS.register(name="sice-part1-512")
class SICEPart1_512(base.ImageEnhancementDataset):
    """SICE Part 1 dataset consists of 360 multi-exposure sequences. Images are
    resized to 512 x 512.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    def get_images(self):
        """Get image files."""
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / "part1-512x512" / "low"
            for path in pbar.track(
                list(pattern.rglob("*/*")),
                description=f"Listing {self.__class__.__name__} images"
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
                description=f"Listing {self.__class__.__name__} labels"
            ):
                stem  = str(img.path.parent.stem)
                path  = self.root / "part1-512x512" / "high" / f"{stem}.jpg"
                path  = pathlib.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)


@DATASETS.register(name="sice-part1-512-low")
class SICEPart1_512_Low(base.UnlabeledImageDataset):
    """SICE Part 1 dataset consists of 360 multi-exposure sequences. Images are
    resized to 512 x 512. Only low-light images are included.
    
    See Also: :class:`mon.vision.dataset.base.dataset.UnlabeledImageDataset`.
    """
    
    def get_images(self):
        """Get image files."""
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / "part1-512x512"
            for path in pbar.track(
                list(pattern.rglob("*/*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    image = base.ImageLabel(path=path)
                    self.images.append(image)


@DATASETS.register(name="sice-part2")
class SICEPart2(base.ImageEnhancementDataset):
    """SICE Part 2 dataset consists of 229 multi-exposure sequences.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    def get_images(self):
        """Get image files."""
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / "part2" / "low"
            for path in pbar.track(
                list(pattern.rglob("*/*")),
                description=f"Listing {self.__class__.__name__} images"
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
                description=f"Listing {self.__class__.__name__} labels"
            ):
                stem  = str(img.path.parent.stem)
                path  = self.root / "part2" / "high" / f"{stem}.jpg"
                path  = pathlib.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)


@DATASETS.register(name="sice-part2-low")
class SICEPart2_Low(base.ImageEnhancementDataset):
    """SICE Part 2 dataset consists of 229 multi-exposure sequences. Only
    low-light images are used. Specifically, we choose the first three (resp.
    four) low-light images if there are seven (resp. nine) images in a
    multi-exposure sequence.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    def get_images(self):
        """Get image files."""
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / "part2-low" / "low"
            for path in pbar.track(
                list(pattern.rglob("*/*")),
                description=f"Listing {self.__class__.__name__} images"
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
                description=f"Listing {self.__class__.__name__} labels"
            ):
                stem  = str(img.path.parent.stem)
                path  = self.root / "part2-low" / "high" / f"{stem}.jpg"
                path  = pathlib.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)


@DATASETS.register(name="sice-part2-512")
class SICEPart2_512(base.ImageEnhancementDataset):
    """SICE Part 2 dataset consists of 229 multi-exposure sequences. Images are
    resized to 512 x 512.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    def get_images(self):
        """Get image files."""
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / "part2-512x512" / "low"
            for path in pbar.track(
                list(pattern.rglob("*/*")),
                description=f"Listing {self.__class__.__name__} images"
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
                description=f"Listing {self.__class__.__name__} labels"
            ):
                stem  = str(img.path.parent.stem)
                path  = self.root / "part2-512x512" / "high" / f"{stem}.jpg"
                path  = pathlib.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)


@DATASETS.register(name="sice-part2-512-low")
class SICEPart2_512_Low(base.ImageEnhancementDataset):
    """SICE Part 2 dataset consists of 229 multi-exposure sequences. Images are
    resized to 3x512x512. Only low-light images are used. Specifically, we
    choose the first three (resp. four) low-light images if there are seven
    (resp. nine) images in a multi-exposure sequence.
    
    """
    
    def get_images(self):
        """Get image files."""
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / "part2-512x512-low" / "low"
            for path in pbar.track(
                list(pattern.rglob("*/*")),
                description=f"Listing {self.__class__.__name__} images"
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
                description=f"Listing {self.__class__.__name__} labels"
            ):
                stem  = str(img.path.parent.stem)
                path  = self.root / "part2-512x512-low" / "high" / f"{stem}.jpg"
                path  = pathlib.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)


@DATASETS.register(name="sice-part2-900")
class SICEPart2_900(base.ImageEnhancementDataset):
    """SICE Part 2 dataset consists of 229 multi-exposure sequences. Images are
    resized to 900 x 1200.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    def get_images(self):
        """Get image files."""
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / "part2-900x1200" / "low"
            for path in pbar.track(
                list(pattern.rglob("*/*")),
                description=f"Listing {self.__class__.__name__} images"
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
                description=f"Listing {self.__class__.__name__} labels"
            ):
                stem  = str(img.path.parent.stem)
                path  = self.root / "part2-900x1200" / "high" / f"{stem}.jpg"
                path  = pathlib.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)


@DATASETS.register(name="sice-part2-900-low")
class SICEPart2_900_Low(base.ImageEnhancementDataset):
    """SICE Part 2 dataset consists of 229 multi-exposure sequences. Images are
    resized to 3x900x1200. Only low-light images are used. Specifically, we
    choose the first three (resp. four) low-light images if there are seven
    (resp. nine) images in a multi-exposure sequence.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    def get_images(self):
        """Get image files."""
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / "part2-900x1200-low" / "low"
            for path in pbar.track(
                list(pattern.rglob("*/*")),
                description=f"Listing {self.__class__.__name__} images"
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
                description=f"Listing {self.__class__.__name__} labels"
            ):
                stem  = str(img.path.parent.stem)
                path  = self.root / "part2-900x1200-low" / "high" / f"{stem}.jpg"
                path  = pathlib.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)


# endregion


# region Datamodule

@DATAMODULES.register(name="sice")
class SICEDataModule(base.DataModule):
    """SICE datamodule.
    
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
            train      = SICEPart1(split="train", **self.dataset_kwargs)
            train_size = int(0.8 * len(train))
            val_size   = len(train) - train_size
            self.train, self.val = random_split(train, [train_size, val_size])
            self.classlabels = getattr(train, "classlabels", None)
            self.collate_fn  = getattr(train, "collate_fn",  None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = SICEPart2(split="train", **self.dataset_kwargs)
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="sice-u")
@DATAMODULES.register(name="sice-unsupervised")
class SICEUDataModule(base.DataModule):
    """SICE-Unsupervised datamodule.
    
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
            self.train       = SICEPart1_512_Low(split="train", **self.dataset_kwargs)
            self.val         = SICEPart2_900_Low(split="train", **self.dataset_kwargs)
            self.classlabels = getattr(self.train, "classlabels", None)
            self.collate_fn  = getattr(self.train, "collate_fn",  None)
        if phase in [None, ModelPhase.TESTING]:
            self.test        = SICEPart2_900_Low(split="test", **self.dataset_kwargs)
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass

# endregion
