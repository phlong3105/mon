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
    "LLIE",
    "LLIEDataModule",
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

from mon.globals import DATAMODULES, DATASETS, ImageFormat, ModelPhase
from mon.vision import core
from mon.vision.data import base

console = core.console


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
        with core.get_progress_bar() as pbar:
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
        with core.get_progress_bar() as pbar:
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
        with core.get_progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = str(img.path).replace("low", "high")
                path  = core.Path(path)
                for ext in ImageFormat.values():
                    temp = path.parent / f"{path.stem}{ext}"
                    if temp.exists():
                        path = temp
                        break
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
        with core.get_progress_bar() as pbar:
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
        with core.get_progress_bar() as pbar:
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
        with core.get_progress_bar() as pbar:
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
        with core.get_progress_bar() as pbar:
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
        with core.get_progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = str(img.path).replace("low", "high")
                path  = core.Path(path)
                for ext in ImageFormat.values():
                    temp = path.parent / f"{path.stem}{ext}"
                    if temp.exists():
                        path = temp
                        break
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
        with core.get_progress_bar() as pbar:
            pattern = self.root / "train" / "low" / "lime"
            for path in pbar.track(
                list(pattern.rglob("*")),
                description=f"Listing {self.__class__.__name__} images"
            ):
                if path.is_image_file():
                    image = base.ImageLabel(path=path)
                    self.images.append(image)


@DATASETS.register(name="llie")
class LLIE(base.UnlabeledImageDataset):
    """LLIE dataset consists of all low-light images.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    splits = ["train"]
    
    def get_images(self):
        """Get image files."""
        subdirs = [
            self.root / "test/deepupe/low",
            self.root / "test/dicm/low",
            self.root / "test/lime/low",
            self.root / "test/mef/low",
            self.root / "test/npe/low",
            self.root / "test/vv/low",
            self.root / "train/gladnet/low",
            self.root / "train/lol/low",
            self.root / "train/sice-mix/low",
            self.root / "train/sice-zerodce/low",
            self.root / "train/ve-lol/low",
        ]
        self.images: list[base.ImageLabel] = []
        with core.get_progress_bar() as pbar:
            for subdir in subdirs:
                for path in pbar.track(
                    list(subdir.rglob("*")),
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
        with core.get_progress_bar() as pbar:
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
        with core.get_progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = str(img.path).replace("low", "high")
                path  = core.Path(path)
                for ext in ImageFormat.values():
                    temp = path.parent / f"{path.stem}{ext}"
                    if temp.exists():
                        path = temp
                        break
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
        with core.get_progress_bar() as pbar:
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
        with core.get_progress_bar() as pbar:
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
        with core.get_progress_bar() as pbar:
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
        with core.get_progress_bar() as pbar:
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
        with core.get_progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = str(img.path).replace("low", "high")
                path  = core.Path(path)
                for ext in ImageFormat.values():
                    temp = path.parent / f"{path.stem}{ext}"
                    if temp.exists():
                        path = temp
                        break
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
        with core.get_progress_bar() as pbar:
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
        with core.get_progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path = str(img.path).replace("low", "high")
                path = core.Path(path)
                for ext in ImageFormat.values():
                    temp = path.parent / f"{path.stem}{ext}"
                    if temp.exists():
                        path = temp
                        break
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
        with core.get_progress_bar() as pbar:
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
        with core.get_progress_bar() as pbar:
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
        with core.get_progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = str(img.path).replace("low", "high")
                path  = core.Path(path)
                for ext in ImageFormat.values():
                    temp = path.parent / f"{path.stem}{ext}"
                    if temp.exists():
                        path = temp
                        break
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
        with core.get_progress_bar() as pbar:
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
        with core.get_progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = str(img.path).replace("low", "high")
                path  = core.Path(path)
                for ext in ImageFormat.values():
                    temp = path.parent / f"{path.stem}{ext}"
                    if temp.exists():
                        path = temp
                        break
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
        with core.get_progress_bar() as pbar:
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
            dataset  = DarkFace(split="test", **self.dataset_kwargs)
            self.split_train_val(dataset=dataset, split_ratio=0.8, full_train=True)
        if phase in [None, ModelPhase.TESTING]:
            self.test = DarkFace(split="test", **self.dataset_kwargs)
        
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
            dataset = DeepUPE(split="test", **self.dataset_kwargs)
            self.split_train_val(dataset=dataset, split_ratio=0.8, full_train=True)
        if phase in [None, ModelPhase.TESTING]:
            self.test = DeepUPE(split="test", **self.dataset_kwargs)
        
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
            dataset = DICM(split="test", **self.dataset_kwargs)
            self.split_train_val(dataset=dataset, split_ratio=0.8, full_train=True)
        if phase in [None, ModelPhase.TESTING]:
            self.test = DICM(split="test", **self.dataset_kwargs)
        
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
            dataset = ExDark(split="test", **self.dataset_kwargs)
            self.split_train_val(dataset=dataset, split_ratio=0.8, full_train=True)
        if phase in [None, ModelPhase.TESTING]:
            self.test = ExDark(split="test", **self.dataset_kwargs)
        
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
            dataset = Fusion(split="test", **self.dataset_kwargs)
            self.split_train_val(dataset=dataset, split_ratio=0.8, full_train=True)
        if phase in [None, ModelPhase.TESTING]:
            self.test = Fusion(split="test", **self.dataset_kwargs)
        
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
            dataset = LIME(split="test", **self.dataset_kwargs)
            self.split_train_val(dataset=dataset, split_ratio=0.8, full_train=True)
        if phase in [None, ModelPhase.TESTING]:
            self.test = LIME(split="test", **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="llie")
class LLIEDataModule(base.DataModule):
    """LLIE datamodule.
    
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
            self.train = LLIE(split="train", **self.dataset_kwargs)
            self.val   = LOL(split="test",   **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = LOL(split="test", **self.dataset_kwargs)
            
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
            self.train = LOL(split="train", **self.dataset_kwargs)
            self.val   = LOL(split="test",  **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = LOL(split="test", **self.dataset_kwargs)
        
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
            dataset = LOL123(split="test", **self.dataset_kwargs)
            self.split_train_val(dataset=dataset, split_ratio=0.8, full_train=True)
        if phase in [None, ModelPhase.TESTING]:
            self.test = LOL123(split="test", **self.dataset_kwargs)

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
            dataset = MEF(split="test", **self.dataset_kwargs)
            self.split_train_val(dataset=dataset, split_ratio=0.8, full_train=True)
        if phase in [None, ModelPhase.TESTING]:
            self.test = MEF(split="test", **self.dataset_kwargs)
        
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
                Default: ``None``.
        """
        console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        
        if phase in [None, ModelPhase.TRAINING]:
            dataset = NPE(split="test", **self.dataset_kwargs)
            self.split_train_val(dataset=dataset, split_ratio=0.8, full_train=True)
        if phase in [None, ModelPhase.TESTING]:
            self.test = NPE(split="test", **self.dataset_kwargs)
        
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
            dataset = SICEGrad(split="train", **self.dataset_kwargs)
            self.split_train_val(dataset=dataset, split_ratio=0.8, full_train=True)
        if phase in [None, ModelPhase.TESTING]:
            self.test = SICEGrad(split="train", **self.dataset_kwargs)

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
            dataset = SICEMix(split="train", **self.dataset_kwargs)
            self.split_train_val(dataset=dataset, split_ratio=0.8, full_train=True)
        if phase in [None, ModelPhase.TESTING]:
            self.test = SICEMix(split="train", **self.dataset_kwargs)
        
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
            self.train = SICEZeroDCE(split="train", **self.dataset_kwargs)
            self.val   = LOL(split="test", **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = LOL(split="test", **self.dataset_kwargs)
            
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
            dataset = VV(split="test", **self.dataset_kwargs)
            self.split_train_val(dataset=dataset, split_ratio=0.8, full_train=True)
        if phase in [None, ModelPhase.TESTING]:
            self.test = VV(split="test", **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass

# endregion
