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

from typing import Literal

from mon import core
from mon.data.datastruct import annotation as anno, datamodule, dataset
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console           = core.console
_default_root_dir = DATA_DIR / "dehaze"


# region Dataset

@DATASETS.register(name="densehaze")
class DenseHaze(dataset.ImageEnhancementDataset):
    """Dense-Haze dataset consists of 33 pairs of real hazy and corresponding
    haze-free images.
    
    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    _tasks          = [Task.DEHAZE]
    _splits         = [Split.TRAIN, Split.VAL, Split.TEST]
    _has_test_label = True
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "densehaze" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)
    
    def _get_labels(self):
        self._labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path  = img.path.replace("/lq/", "/hq/")
                label = anno.ImageAnnotation(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="ihaze")
class IHaze(dataset.ImageEnhancementDataset):
    """I-Haze dataset consists of 35 pairs of real hazy and corresponding
    haze-free images.
    
    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    _tasks          = [Task.DEHAZE]
    _splits         = [Split.TRAIN, Split.VAL, Split.TEST]
    _has_test_label = True
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "ihaze" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)
    
    def _get_labels(self):
        self._labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path  = img.path.replace("/lq/", "/hq/")
                label = anno.ImageAnnotation(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="nhhaze")
class NHHaze(dataset.ImageEnhancementDataset):
    """NH-Haze dataset consists 55 pairs of real hazy and corresponding
    haze-free images.
    
    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    _tasks          = [Task.DEHAZE]
    _splits         = [Split.TRAIN, Split.VAL, Split.TEST]
    _has_test_label = True
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "nhhaze" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)

    def _get_labels(self):
        self._labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path  = img.path.replace("/lq/", "/hq/")
                label = anno.ImageAnnotation(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="ohaze")
class OHaze(dataset.ImageEnhancementDataset):
    """O-Haze dataset consists of 45 pairs of real hazy and corresponding
    haze-free images.
    
    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    _tasks          = [Task.DEHAZE]
    _splits         = [Split.TRAIN, Split.VAL, Split.TEST]
    _has_test_label = True
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "ohaze" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)

    def _get_labels(self):
        self._labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path  = img.path.replace("/lq/", "/hq/")
                label = anno.ImageAnnotation(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="reside_hsts_real")
class RESIDEHSTSReal(dataset.UnlabeledImageDataset):
    """RESIDE-HSTS-Real dataset consists of 10 real hazy images.

    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    _tasks          = [Task.DEHAZE]
    _splits         = [Split.TEST]
    _has_test_label = False
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "reside_hsts_real" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)


@DATASETS.register(name="reside_hsts_syn")
class RESIDEHSTSSyn(dataset.ImageEnhancementDataset):
    """RESIDE-HSTS-Syn dataset consists of 10 pairs of hazy and corresponding
    haze-free images.

    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    _tasks          = [Task.DEHAZE]
    _splits         = [Split.TEST]
    _has_test_label = False
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "reside_hsts_syn" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)

    def _get_labels(self):
        self._labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path  = img.path.replace("/lq/", "/hq/")
                label = anno.ImageAnnotation(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="reside_its")
class RESIDEITS(dataset.ImageEnhancementDataset):
    """RESIDE-ITS dataset consists of 13,990 pairs of hazy and corresponding
    haze-free images.

    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    _tasks          = [Task.DEHAZE]
    _splits         = [Split.TRAIN, Split.VAL]
    _has_test_label = False
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "reside_its" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)

    def _get_labels(self):
        self._labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                stem  = str(img.path.stem).split("_")[0]
                path  = img.path.replace("/lq/", "/hq/")
                path  = path.parent / f"{stem}.{img.path.suffix}"
                label = anno.ImageAnnotation(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="reside_its_v2")
class RESIDEITSV2(dataset.ImageEnhancementDataset):
    """RESIDE-ITS-V2 dataset consists of 13,990 pairs of hazy and corresponding
    haze-free images.

    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    _tasks          = [Task.DEHAZE]
    _splits         = [Split.TRAIN]
    _has_test_label = False
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "reside_its_v2" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)

    def _get_labels(self):
        self._labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                stem  = str(img.path.stem).split("_")[0]
                path  = img.path.replace("/lq/", "/hq/")
                path  = path.parent / f"{stem}.{img.path.suffix}"
                label = anno.ImageAnnotation(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="reside_ots")
class RESIDEOTS(dataset.ImageEnhancementDataset):
    """RESIDE-OTS dataset consists of 73,135 pairs of hazy and corresponding
    haze-free images.

    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    _tasks          = [Task.DEHAZE]
    _splits         = [Split.TRAIN]
    _has_test_label = False
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "reside_ots" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)

    def _get_labels(self):
        self._labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                stem  = str(img.path.stem).split("_")[0]
                path  = img.path.replace("/lq/", "/hq/")
                path  = path.parent / f"{stem}.{img.path.suffix}"
                label = anno.ImageAnnotation(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="reside_rtts")
class RESIDERTTS(dataset.UnlabeledImageDataset):
    """RESIDE-RTTS dataset consists of 4,322 real hazy images.

    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    _tasks          = [Task.DEHAZE]
    _splits         = [Split.TEST]
    _has_test_label = False
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "reside_rtts" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)


@DATASETS.register(name="reside_sots_indoor")
class RESIDESOTSIndoor(dataset.ImageEnhancementDataset):
    """RESIDE-SOTS-Indoor dataset consists of 500 pairs of hazy and
    corresponding haze-free images.

    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    _tasks          = [Task.DEHAZE]
    _splits         = [Split.TEST]
    _has_test_label = True
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "reside_sots_indoor " / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)

    def _get_labels(self):
        self._labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                stem  = str(img.path.stem).split("_")[0]
                path  = img.path.replace("/lq/", "/hq/")
                path  = path.parent / f"{stem}.{img.path.suffix}"
                label = anno.ImageAnnotation(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="reside_sots_outdoor")
class RESIDESOTSOutdoor(dataset.ImageEnhancementDataset):
    """RESIDE-SOTS-Outdoor dataset consists of 500 pairs of hazy and
    corresponding haze-free images.

    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    _tasks          = [Task.DEHAZE]
    _splits         = [Split.TEST]
    _has_test_label = True
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "reside_sots_outdoor" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)

    def _get_labels(self):
        self._labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                stem  = str(img.path.stem).split("_")[0]
                path  = img.path.replace("/lq/", "/hq/")
                path  = path.parent / f"{stem}.{img.path.suffix}"
                label = anno.ImageAnnotation(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="reside_uhi")
class RESIDEUHI(dataset.UnlabeledImageDataset):
    """RESIDE-UHI dataset consists of 4,809 real hazy images.

    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    _tasks          = [Task.DEHAZE]
    _splits         = [Split.TEST]
    _has_test_label = False
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "reside_uhi" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)


@DATASETS.register(name="satehaze1k")
class SateHaze1K(dataset.ImageEnhancementDataset):
    """SateHaze1K dataset consists 1200 pairs of hazy and corresponding
    haze-free images.
    
    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    _tasks          = [Task.DEHAZE]
    _splits         = [Split.TRAIN, Split.VAL, Split.TEST]
    _has_test_label = True
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "satehaze1k_thin" / self.split_str / "lq",
            self.root / "satehaze1k_moderate" / self.split_str / "lq",
            self.root / "satehaze1k_thick" / self.split_str / "lq",
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)

    def _get_labels(self):
        self._labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path  = img.path.replace("/lq/", "/hq/")
                label = anno.ImageAnnotation(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="satehaze1k_thin")
class SateHaze1KThin(dataset.ImageEnhancementDataset):
    """SateHaze1K-Thin dataset.
    
    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    _tasks          = [Task.DEHAZE]
    _splits         = [Split.TRAIN, Split.VAL, Split.TEST]
    _has_test_label = True
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "satehaze1k_thin" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)
    
    def _get_labels(self):
        self._labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path  = img.path.replace("/lq/", "/hq/")
                label = anno.ImageAnnotation(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="satehaze1k_moderate")
class SateHaze1KModerate(dataset.ImageEnhancementDataset):
    """SateHaze1K-Moderate.
    
    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    _tasks          = [Task.DEHAZE]
    _splits         = [Split.TRAIN, Split.VAL, Split.TEST]
    _has_test_label = True
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "satehaze1k_moderate" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)

    def _get_labels(self):
        self._labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path  = img.path.replace("/lq/", "/hq/")
                label = anno.ImageAnnotation(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="satehaze1k_thick")
class SateHaze1KThick(dataset.ImageEnhancementDataset):
    """SateHaze1K-Thick dataset.
    
    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    _tasks          = [Task.DEHAZE]
    _splits         = [Split.TRAIN, Split.VAL, Split.TEST]
    _has_test_label = True
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def _get_images(self):
        patterns = [
            self.root / "satehaze1k_thick" / self.split_str / "lq"
        ]
        self._images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self._images.append(image)

    def _get_labels(self):
        self._labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path  = img.path.replace("/lq/", "/hq/")
                label = anno.ImageAnnotation(path=path.image_file())
                self._labels.append(label)

# endregion


# region Datamodule

@DATAMODULES.register(name="densehaze")
class DenseHazeDataModule(datamodule.DataModule):
    """Dense-Haze datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, phase: Literal["training", "testing", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if phase in [None, "training"]:
            self.train = DenseHaze(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = DenseHaze(split=Split.VAL,   **self.dataset_kwargs)
        if phase in [None, "testing"]:
            self.test  = DenseHaze(split=Split.TEST,  **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="ihaze")
class IHazeDataModule(datamodule.DataModule):
    """I-Haze datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, phase: Literal["training", "testing", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if phase in [None, "training"]:
            self.train = IHaze(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = IHaze(split=Split.VAL,   **self.dataset_kwargs)
        if phase in [None, "testing"]:
            self.test  = IHaze(split=Split.TEST,  **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="nhhaze")
class NHHazeDataModule(datamodule.DataModule):
    """NH-Haze datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, phase: Literal["training", "testing", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if phase in [None, "training"]:
            self.train = NHHaze(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = NHHaze(split=Split.VAL,   **self.dataset_kwargs)
        if phase in [None, "testing"]:
            self.test  = NHHaze(split=Split.TEST,  **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="ohaze")
class OHazeDataModule(datamodule.DataModule):
    """O-Haze datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, phase: Literal["training", "testing", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if phase in [None, "training"]:
            self.train = OHaze(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = OHaze(split=Split.VAL,   **self.dataset_kwargs)
        if phase in [None, "testing"]:
            self.test  = OHaze(split=Split.TEST,  **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="reside_hsts_real")
class RESIDEHSTSRealDataModule(datamodule.DataModule):
    """RESIDE-HSTS-Real datamodule.

     See Also: :class:`base.DataModule`.
     """
    
    _tasks = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()

    def setup(self, phase: Literal["training", "testing", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if phase in [None, "training"]:
            self.train = RESIDEHSTSReal(split=Split.TEST, **self.dataset_kwargs)
            self.val   = RESIDEHSTSReal(split=Split.TEST, **self.dataset_kwargs)
        if phase in [None, "testing"]:
            self.test  = RESIDEHSTSReal(split=Split.TEST, **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        pass


@DATAMODULES.register(name="reside_hsts_syn")
class RESIDEHSTSSynDataModule(datamodule.DataModule):
    """RESIDE-HSTS-Syn datamodule.

     See Also: :class:`base.DataModule`.
     """
    
    _tasks = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()

    def setup(self, phase: Literal["training", "testing", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if phase in [None, "training"]:
            self.train = RESIDEHSTSSyn(split=Split.TEST, **self.dataset_kwargs)
            self.val   = RESIDEHSTSSyn(split=Split.TEST, **self.dataset_kwargs)
        if phase in [None, "testing"]:
            self.test = RESIDEHSTSSyn(split=Split.TEST,  **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        pass


@DATAMODULES.register(name="reside_its")
class RESIDEITSDataModule(datamodule.DataModule):
    """RESIDE-ITS datamodule.

    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()

    def setup(self, phase: Literal["training", "testing", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if phase in [None, "training"]:
            self.train = RESIDEITS(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = RESIDEITS(split=Split.VAL,   **self.dataset_kwargs)
        if phase in [None, "testing"]:
            self.test  = RESIDEITS(split=Split.TEST,  **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        pass


@DATAMODULES.register(name="reside_its_v2")
class RESIDEITSV2DataModule(datamodule.DataModule):
    """RESIDE-ITS-V2 datamodule.

    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()

    def setup(self, phase: Literal["training", "testing", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if phase in [None, "training"]:
            self.train = RESIDEITSV2(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   =   RESIDEITS(split=Split.VAL,   **self.dataset_kwargs)
        if phase in [None, "testing"]:
            self.test  =   RESIDEITS(split=Split.TEST,  **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        pass


@DATAMODULES.register(name="reside_ots")
class RESIDEOTSDataModule(datamodule.DataModule):
    """RESIDE-OTS datamodule.

    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()

    def setup(self, phase: Literal["training", "testing", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if phase in [None, "training"]:
            self.train = RESIDEOTS(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = RESIDEITS(split=Split.VAL,   **self.dataset_kwargs)
        if phase in [None, "testing"]:
            self.test  = RESIDEITS(split=Split.TEST,  **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        pass


@DATAMODULES.register(name="reside_rtts")
class RESIDERTTSDataModule(datamodule.DataModule):
    """RESIDE-RTTS datamodule.

     See Also: :class:`base.DataModule`.
     """
    
    _tasks = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()

    def setup(self, phase: Literal["training", "testing", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if phase in [None, "training"]:
            self.train = RESIDERTTS(split=Split.TEST, **self.dataset_kwargs)
            self.val   = RESIDERTTS(split=Split.TEST, **self.dataset_kwargs)
        if phase in [None, "testing"]:
            self.test = RESIDERTTS(split=Split.TEST,  **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        pass


@DATAMODULES.register(name="reside_sots_indoor")
class RESIDESOTSIndoorDataModule(datamodule.DataModule):
    """RESIDE-SOTS-Indoor datamodule.

     See Also: :class:`base.DataModule`.
     """
    
    _tasks = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()

    def setup(self, phase: Literal["training", "testing", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if phase in [None, "training"]:
            self.train = RESIDESOTSIndoor(split=Split.TEST, **self.dataset_kwargs)
            self.val   = RESIDESOTSIndoor(split=Split.TEST, **self.dataset_kwargs)
        if phase in [None, "testing"]:
            self.test = RESIDESOTSIndoor(split=Split.TEST,  **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        pass


@DATAMODULES.register(name="reside_sots_outdoor")
class RESIDESOTSOutdoorDataModule(datamodule.DataModule):
    """RESIDE-SOTS-Outdoor datamodule.

     See Also: :class:`base.DataModule`.
     """
    
    _tasks = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()

    def setup(self, phase: Literal["training", "testing", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if phase in [None, "training"]:
            self.train = RESIDESOTSOutdoor(split=Split.TEST, **self.dataset_kwargs)
            self.val   = RESIDESOTSOutdoor(split=Split.TEST, **self.dataset_kwargs)
        if phase in [None, "testing"]:
            self.test  = RESIDESOTSOutdoor(split=Split.TEST, **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        pass


@DATAMODULES.register(name="reside_uhi")
class RESIDEUHIDataModule(datamodule.DataModule):
    """RESIDE-UHI datamodule.

     See Also: :class:`base.DataModule`.
     """
    
    _tasks = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, phase: Literal["training", "testing", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if phase in [None, "training"]:
            self.train = RESIDEUHI(split=Split.TEST, **self.dataset_kwargs)
            self.val   = RESIDEUHI(split=Split.TEST, **self.dataset_kwargs)
        if phase in [None, "testing"]:
            self.test  = RESIDEUHI(split=Split.TEST, **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="satehaze1k")
class SateHaze1KDataModule(datamodule.DataModule):
    """SateHaze1K datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, phase: Literal["training", "testing", None] = None):
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
        
        if phase in [None, "training"]:
            self.train = SateHaze1K(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = SateHaze1K(split=Split.VAL,   **self.dataset_kwargs)
        if phase in [None, "testing"]:
            self.test  = SateHaze1K(split=Split.TEST,  **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="satehaze1k_thin")
class SateHaze1KThinDataModule(datamodule.DataModule):
    """SateHaze1K-Thin datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, phase: Literal["training", "testing", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if phase in [None, "training"]:
            self.train = SateHaze1KThin(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = SateHaze1KThin(split=Split.VAL,   **self.dataset_kwargs)
        if phase in [None, "testing"]:
            self.test  = SateHaze1KThin(split=Split.TEST,  **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="satehaze1k_moderate")
class SateHaze1KModerateDataModule(datamodule.DataModule):
    """SateHaze1K-Moderate datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, phase: Literal["training", "testing", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
       
        if phase in [None, "training"]:
            self.train = SateHaze1KModerate(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = SateHaze1KModerate(split=Split.VAL,   **self.dataset_kwargs)
        if phase in [None, "testing"]:
            self.test  = SateHaze1KModerate(split=Split.TEST,  **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="satehaze1k_thick")
class SateHaze1KThickDataModule(datamodule.DataModule):
    """SateHaze1K-Thick datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    _tasks = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, phase: Literal["training", "testing", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if phase in [None, "training"]:
            self.train = SateHaze1KThick(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = SateHaze1KThick(split=Split.VAL,   **self.dataset_kwargs)
        if phase in [None, "testing"]:
            self.test  = SateHaze1KThick(split=Split.TEST,  **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass

# endregion
