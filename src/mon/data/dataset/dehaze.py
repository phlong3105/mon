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
from mon.data import base
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Task, Split

console           = core.console
_default_root_dir = DATA_DIR / "dehaze"


# region Dataset

@DATASETS.register(name="densehaze")
class DenseHaze(base.ImageEnhancementDataset):
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
            self.root / "densehaze" / self.split / "lq"
        ]
        self._images: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self._images.append(image)
    
    def _get_labels(self):
        self._labels: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = str(img.path).replace("/lq/", "/hq/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="ihaze")
class IHaze(base.ImageEnhancementDataset):
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
            self.root / "ihaze" /  self.split / "lq"
        ]
        self._images: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self._images.append(image)
    
    def _get_labels(self):
        self._labels: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = str(img.path).replace("/lq/", "/hq/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="nhhaze")
class NHHaze(base.ImageEnhancementDataset):
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
            self.root / "nhhaze" / self.split / "lq"
        ]
        self._images: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self._images.append(image)

    def _get_labels(self):
        self._labels: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = str(img.path).replace("/lq/", "/hq/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="ohaze")
class OHaze(base.ImageEnhancementDataset):
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
            self.root / "ohaze" / self.split / "lq"
        ]
        self._images: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self._images.append(image)

    def _get_labels(self):
        self._labels: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = str(img.path).replace("/lq/", "/hq/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="reside_hsts_real")
class RESIDEHSTSReal(base.UnlabeledImageDataset):
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
            self.root / "reside_hsts_real" / self.split / "lq"
        ]
        self._images: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self._images.append(image)


@DATASETS.register(name="reside_hsts_syn")
class RESIDEHSTSSyn(base.ImageEnhancementDataset):
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
            self.root / "reside_hsts_syn" / self.split / "lq"
        ]
        self._images: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self._images.append(image)

    def _get_labels(self):
        self._labels: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = str(img.path).replace("/lq/", "/hq/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="reside_its")
class RESIDEITS(base.ImageEnhancementDataset):
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
            self.root / "reside_its" / self.split / "lq"
        ]
        self._images: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self._images.append(image)

    def _get_labels(self):
        self._labels: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                stem  = str(img.path.stem).split("_")[0]
                path  = str(img.path).replace("/lq/", "/hq/")
                path  = core.Path(path).parent / f"{stem}.{img.path.suffix}"
                label = base.ImageLabel(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="reside_its_v2")
class RESIDEITSV2(base.ImageEnhancementDataset):
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
            self.root / "reside_its_v2" / self.split / "lq"
        ]
        self._images: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self._images.append(image)

    def _get_labels(self):
        self._labels: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                stem  = str(img.path.stem).split("_")[0]
                path  = str(img.path).replace("/lq/", "/hq/")
                path  = core.Path(path).parent / f"{stem}.{img.path.suffix}"
                label = base.ImageLabel(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="reside_ots")
class RESIDEOTS(base.ImageEnhancementDataset):
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
            self.root / "reside_ots" / self.split / "lq"
        ]
        self._images: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self._images.append(image)

    def _get_labels(self):
        self._labels: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                stem  = str(img.path.stem).split("_")[0]
                path  = str(img.path).replace("/lq/", "/hq/")
                path  = core.Path(path).parent / f"{stem}.{img.path.suffix}"
                label = base.ImageLabel(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="reside_rtts")
class RESIDERTTS(base.UnlabeledImageDataset):
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
            self.root / "reside_rtts" / self.split / "lq"
        ]
        self._images: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self._images.append(image)


@DATASETS.register(name="reside_sots_indoor")
class RESIDESOTSIndoor(base.ImageEnhancementDataset):
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
            self.root / "reside_sots_indoor "/ self.split / "lq"
        ]
        self._images: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self._images.append(image)

    def _get_labels(self):
        self._labels: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                stem  = str(img.path.stem).split("_")[0]
                path  = str(img.path).replace("/lq/", "/hq/")
                path  = core.Path(path).parent / f"{stem}.{img.path.suffix}"
                label = base.ImageLabel(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="reside_sots_outdoor")
class RESIDESOTSOutdoor(base.ImageEnhancementDataset):
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
            self.root / "reside_sots_outdoor" / self.split / "lq"
        ]
        self._images: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self._images.append(image)

    def _get_labels(self):
        self._labels: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                stem  = str(img.path.stem).split("_")[0]
                path  = str(img.path).replace("/lq/", "/hq/")
                path  = core.Path(path).parent / f"{stem}.{img.path.suffix}"
                label = base.ImageLabel(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="reside_uhi")
class RESIDEUHI(base.UnlabeledImageDataset):
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
            self.root / "reside_uhi" / self.split / "lq"
        ]
        self._images: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self._images.append(image)


@DATASETS.register(name="satehaze1k")
class SateHaze1K(base.ImageEnhancementDataset):
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
            self.root / "satehaze1k_thin"     / self.split / "lq",
            self.root / "satehaze1k_moderate" / self.split / "lq",
            self.root / "satehaze1k_thick"    / self.split / "lq",
        ]
        self._images: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self._images.append(image)

    def _get_labels(self):
        self._labels: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = str(img.path).replace("/lq/", "/hq/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="satehaze1k_thin")
class SateHaze1KThin(base.ImageEnhancementDataset):
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
            self.root / "satehaze1k_thin" / self.split / "lq"
        ]
        self._images: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self._images.append(image)
    
    def _get_labels(self):
        self._labels: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = str(img.path).replace("/lq/", "/hq/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="satehaze1k_moderate")
class SateHaze1KModerate(base.ImageEnhancementDataset):
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
            self.root / "satehaze1k_moderate" / self.split / "lq"
        ]
        self._images: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self._images.append(image)

    def _get_labels(self):
        self._labels: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = str(img.path).replace("/lq/", "/hq/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self._labels.append(label)


@DATASETS.register(name="satehaze1k_thick")
class SateHaze1KThick(base.ImageEnhancementDataset):
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
            self.root / "satehaze1k_thick" / self.split / "lq"
        ]
        self._images: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split} images"
                ):
                    if path.is_image_file():
                        image = base.ImageLabel(path=path)
                        self._images.append(image)

    def _get_labels(self):
        self._labels: list[base.ImageLabel] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self._images,
                description=f"Listing {self.__class__.__name__} {self.split} labels"
            ):
                path  = str(img.path).replace("/lq/", "/hq/")
                path  = core.Path(path)
                label = base.ImageLabel(path=path.image_file())
                self._labels.append(label)

# endregion


# region Datamodule

@DATAMODULES.register(name="densehaze")
class DenseHazeDataModule(base.DataModule):
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
class IHazeDataModule(base.DataModule):
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
class NHHazeDataModule(base.DataModule):
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
class OHazeDataModule(base.DataModule):
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
class RESIDEHSTSRealDataModule(base.DataModule):
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
class RESIDEHSTSSynDataModule(base.DataModule):
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
class RESIDEITSDataModule(base.DataModule):
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
class RESIDEITSV2DataModule(base.DataModule):
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
class RESIDEOTSDataModule(base.DataModule):
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
class RESIDERTTSDataModule(base.DataModule):
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
class RESIDESOTSIndoorDataModule(base.DataModule):
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
class RESIDESOTSOutdoorDataModule(base.DataModule):
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
class RESIDEUHIDataModule(base.DataModule):
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
class SateHaze1KDataModule(base.DataModule):
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
class SateHaze1KThinDataModule(base.DataModule):
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
class SateHaze1KModerateDataModule(base.DataModule):
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
class SateHaze1KThickDataModule(base.DataModule):
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
