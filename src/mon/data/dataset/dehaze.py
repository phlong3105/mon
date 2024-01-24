#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements de-hazing datasets and datamodules."""

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
    "RESIDEHSTSReal",
    "RESIDEHSTSRealDataModule",
    "RESIDEHSTSSyn",
    "RESIDEHSTSSynDataModule",
    "RESIDEITS",
    "RESIDEITSDataModule",
    "RESIDEITSV2",
    "RESIDEITSV2DataModule",
    "RESIDEOTS",
    "RESIDEOTSDataModule",
    "RESIDERTTS",
    "RESIDERTTSDataModule",
    "RESIDESOTSIndoor",
    "RESIDESOTSIndoorDataModule",
    "RESIDESOTSOutdoor",
    "RESIDESOTSOutdoorDataModule",
    "RESIDEUHI",
    "RESIDEUHIDataModule",
    "SateHaze1K",
    "SateHaze1KDataModule",
    "SateHaze1KModerate",
    "SateHaze1KModerateDataModule",
    "SateHaze1KThick",
    "SateHaze1KThickDataModule",
    "SateHaze1KThin",
    "SateHaze1KThinDataModule",
]

from mon import core
from mon.data import base
from mon.globals import DATAMODULES, DATASETS, ModelPhase

console = core.console


# region Dataset

@DATASETS.register(name="dense-haze")
class DenseHaze(base.ImageEnhancementDataset):
    """Dense-Haze dataset consists of 33 pairs of real hazy and corresponding
    haze-free images.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """

    splits = ["train", "val", "test"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "dense-haze" / "haze"
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
                path  = str(img.path).replace("/haze/", "/clear/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="i-haze")
class IHaze(base.ImageEnhancementDataset):
    """I-Haze dataset consists of 35 pairs of real hazy and corresponding
    haze-free images.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """

    splits = ["train", "val", "test"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "i-haze" / "haze"
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
                path  = str(img.path).replace("/haze/", "/clear/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="nh-haze")
class NHHaze(base.ImageEnhancementDataset):
    """NH-Haze dataset consists 55 pairs of real hazy and corresponding
    haze-free images.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    splits = ["train", "val", "test"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "nh-haze" / "haze"
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
                path  = str(img.path).replace("/haze/", "/clear/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="o-haze")
class OHaze(base.ImageEnhancementDataset):
    """O-Haze dataset consists of 45 pairs of real hazy and corresponding
    haze-free images.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    splits = ["train", "val", "test"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "o-haze" / "haze"
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
                path  = str(img.path).replace("/haze/", "/clear/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="reside-hsts-real")
class RESIDEHSTSReal(base.UnlabeledImageDataset):
    """RESIDE-HSTS-Real dataset consists of 10 real hazy images.

    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """

    splits = ["test"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "reside-hsts-real" / "haze"
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


@DATASETS.register(name="reside-hsts-syn")
class RESIDEHSTSSyn(base.ImageEnhancementDataset):
    """RESIDE-HSTS-Syn dataset consists of 10 pairs of hazy and corresponding
    haze-free images.

    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """

    splits = ["test"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "reside-hsts-syn" / "haze"
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
                path  = str(img.path).replace("/haze/", "/clear/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="reside-its")
class RESIDEITS(base.ImageEnhancementDataset):
    """RESIDE-ITS dataset consists of 13,990 pairs of hazy and corresponding
    haze-free images.

    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """

    splits = ["train", "val"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "reside-its" / "haze"
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
                stem  = str(img.path.stem).split("_")[0]
                path  = str(img.path).replace("/haze/", "/clear/")
                path  = core.Path(path).parent / f"{stem}.{img.path.suffix}"
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="reside-its-v2")
class RESIDEITSV2(base.ImageEnhancementDataset):
    """RESIDE-ITS-V2 dataset consists of 13,990 pairs of hazy and corresponding
    haze-free images.

    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """

    splits = ["train"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "reside-its-v2" / "haze"
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
                stem  = str(img.path.stem).split("_")[0]
                path  = str(img.path).replace("/haze/", "/clear/")
                path  = core.Path(path).parent / f"{stem}.{img.path.suffix}"
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="reside-ots")
class RESIDEOTS(base.ImageEnhancementDataset):
    """RESIDE-OTS dataset consists of 73,135 pairs of hazy and corresponding
    haze-free images.

    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """

    splits = ["train"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "reside-ots" / "haze"
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
                stem  = str(img.path.stem).split("_")[0]
                path  = str(img.path).replace("/haze/", "/clear/")
                path  = core.Path(path).parent / f"{stem}.{img.path.suffix}"
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="reside-rtts")
class RESIDERTTS(base.UnlabeledImageDataset):
    """RESIDE-RTTS dataset consists of 4,322 real hazy images.

    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """

    splits = ["test"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "reside-rtts" / "haze"
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


@DATASETS.register(name="reside-sots-indoor")
class RESIDESOTSIndoor(base.ImageEnhancementDataset):
    """RESIDE-SOTS-Indoor dataset consists of 500 pairs of hazy and
    corresponding haze-free images.

    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """

    splits = ["test"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "reside-sots-indoor" / "haze"
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
                stem  = str(img.path.stem).split("_")[0]
                path  = str(img.path).replace("/haze/", "/clear/")
                path  = core.Path(path).parent / f"{stem}.{img.path.suffix}"
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="reside-sots-outdoor")
class RESIDESOTSOutdoor(base.ImageEnhancementDataset):
    """RESIDE-SOTS-Outdoor dataset consists of 500 pairs of hazy and
    corresponding haze-free images.

    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """

    splits = ["test"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "reside-sots-outdoor" / "haze"
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
                stem  = str(img.path.stem).split("_")[0]
                path  = str(img.path).replace("/haze/", "/clear/")
                path  = core.Path(path).parent / f"{stem}.{img.path.suffix}"
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="reside-uhi")
class RESIDEUHI(base.UnlabeledImageDataset):
    """RESIDE-UHI dataset consists of 4,809 real hazy images.

    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """

    splits = ["test"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "reside-uhi" / "haze"
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


@DATASETS.register(name="satehaze1k")
class SateHaze1K(base.ImageEnhancementDataset):
    """SateHaze1K dataset consists 1200 pairs of hazy and corresponding
    haze-free images.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    splits = ["train", "val", "test"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "satehaze1k-thin" / "haze",
            self.root / self.split / "satehaze1k-moderate" / "haze",
            self.root / self.split / "satehaze1k-thick" / "haze",
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
                path  = str(img.path).replace("/haze/", "/clear/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="satehaze1k-thin")
class SateHaze1KThin(base.ImageEnhancementDataset):
    """SateHaze1K-Thin dataset.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """

    splits = ["train", "val", "test"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "satehaze1k-thin" / "haze"
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
                path  = str(img.path).replace("/haze/", "/clear/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="satehaze1k-moderate")
class SateHaze1KModerate(base.ImageEnhancementDataset):
    """SateHaze1K-Moderate.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    splits = ["train", "val", "test"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "satehaze1k-moderate" / "haze"
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
                path  = str(img.path).replace("/haze/", "/clear/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="satehaze1k-thick")
class SateHaze1KThick(base.ImageEnhancementDataset):
    """SateHaze1K-Thick dataset.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    splits = ["train", "val", "test"]

    def get_images(self):
        """Get image files."""
        patterns = [
            self.root / self.split / "satehaze1k-thick" / "haze"
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
                path  = str(img.path).replace("/haze/", "/clear/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
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
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        if phase in [None, ModelPhase.TRAINING]:
            self.train = DenseHaze(split="train", **self.dataset_kwargs)
            self.val   = DenseHaze(split="val",   **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = DenseHaze(split="test",  **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
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
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        if phase in [None, ModelPhase.TRAINING]:
            self.train = IHaze(split="train", **self.dataset_kwargs)
            self.val   = IHaze(split="val",   **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = IHaze(split="test",  **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
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
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        if phase in [None, ModelPhase.TRAINING]:
            self.train = NHHaze(split="train", **self.dataset_kwargs)
            self.val   = NHHaze(split="val",   **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = NHHaze(split="test",  **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
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
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        if phase in [None, ModelPhase.TRAINING]:
            self.train = OHaze(split="train", **self.dataset_kwargs)
            self.val   = OHaze(split="val",   **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = OHaze(split="test",  **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="reside-hsts-real")
class RESIDEHSTSRealDataModule(base.DataModule):
    """RESIDE-HSTS-Real datamodule.

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
            self.train = RESIDEHSTSReal(split="test", **self.dataset_kwargs)
            self.val   = RESIDEHSTSReal(split="test", **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = RESIDEHSTSReal(split="test", **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="reside-hsts-syn")
class RESIDEHSTSSynDataModule(base.DataModule):
    """RESIDE-HSTS-Syn datamodule.

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
            self.train = RESIDEHSTSSyn(split="test", **self.dataset_kwargs)
            self.val   = RESIDEHSTSSyn(split="test", **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test = RESIDEHSTSSyn(split="test",  **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="reside-its")
class RESIDEITSDataModule(base.DataModule):
    """RESIDE-ITS datamodule.

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
            self.train = RESIDEITS(split="train", **self.dataset_kwargs)
            self.val   = RESIDEITS(split="val",   **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = RESIDEITS(split="test",  **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="reside-its-v2")
class RESIDEITSV2DataModule(base.DataModule):
    """RESIDE-ITS-V2 datamodule.

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
            self.train = RESIDEITSV2(split="train", **self.dataset_kwargs)
            self.val   =   RESIDEITS(split="val",   **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  =   RESIDEITS(split="test",  **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="reside-ots")
class RESIDEOTSDataModule(base.DataModule):
    """RESIDE-OTS datamodule.

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
            self.train = RESIDEOTS(split="train", **self.dataset_kwargs)
            self.val   = RESIDEITS(split="val",   **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = RESIDEITS(split="test",  **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="reside-rtts")
class RESIDERTTSDataModule(base.DataModule):
    """RESIDE-RTTS datamodule.

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
            self.train = RESIDERTTS(split="test", **self.dataset_kwargs)
            self.val   = RESIDERTTS(split="test", **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test = RESIDERTTS(split="test",  **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="reside-sots-indoor")
class RESIDESOTSIndoorDataModule(base.DataModule):
    """RESIDE-SOTS-Indoor datamodule.

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
            self.train = RESIDESOTSIndoor(split="test", **self.dataset_kwargs)
            self.val   = RESIDESOTSIndoor(split="test", **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test = RESIDESOTSIndoor(split="test",  **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="reside-sots-outdoor")
class RESIDESOTSOutdoorDataModule(base.DataModule):
    """RESIDE-SOTS-Outdoor datamodule.

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
            self.train = RESIDESOTSOutdoor(split="test", **self.dataset_kwargs)
            self.val   = RESIDESOTSOutdoor(split="test", **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = RESIDESOTSOutdoor(split="test", **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
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
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        if phase in [None, ModelPhase.TRAINING]:
            self.train = SateHaze1K(split="train", **self.dataset_kwargs)
            self.val   = SateHaze1K(split="val",   **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = SateHaze1K(split="test",  **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@DATAMODULES.register(name="reside-uhi")
class RESIDEUHIDataModule(base.DataModule):
    """RESIDE-UHI datamodule.

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
            self.train = RESIDEUHI(split="test", **self.dataset_kwargs)
            self.val   = RESIDEUHI(split="test", **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = RESIDEUHI(split="test", **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
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
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        if phase in [None, ModelPhase.TRAINING]:
            self.train = SateHaze1KThin(split="train", **self.dataset_kwargs)
            self.val   = SateHaze1KThin(split="val",   **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = SateHaze1KThin(split="test",  **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
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
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
       
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        if phase in [None, ModelPhase.TRAINING]:
            self.train = SateHaze1KModerate(split="train", **self.dataset_kwargs)
            self.val   = SateHaze1KModerate(split="val",   **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = SateHaze1KModerate(split="test",  **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
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
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        phase = ModelPhase.from_value(phase) if phase is not None else phase
        if phase in [None, ModelPhase.TRAINING]:
            self.train = SateHaze1KThick(split="train", **self.dataset_kwargs)
            self.val   = SateHaze1KThick(split="val",   **self.dataset_kwargs)
        if phase in [None, ModelPhase.TESTING]:
            self.test  = SateHaze1KThick(split="test",  **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass

# endregion
