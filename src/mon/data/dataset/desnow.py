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

from typing import Literal

from mon import core
from mon.data.datastruct import annotation as anno, datamodule, dataset
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console           = core.console
_default_root_dir = DATA_DIR / "desnow"


# region Dataset

@DATASETS.register(name="gtsnow")
class GTSnow(dataset.ImageEnhancementDataset):
    """GT-Snow dataset.

    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    tasks          = [Task.DESNOW]
    splits         = [Split.TRAIN]
    has_test_label = False
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_images(self):
        patterns = [
            self.root / "gtsnow" / self.split_str / "lq"
        ]
        self.images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self.images.append(image)

    def get_labels(self):
        self.labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path  = str(img.path)
                path  = path[:-9] + "C-000.png"
                path  = path.replace("/lq/", "/hq/")
                path  = core.Path(path)
                label = anno.ImageAnnotation(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="kitti_snow")
class KITTISnow(dataset.ImageEnhancementDataset):
    """KITTI-Snow dataset.

    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    tasks          = [Task.DESNOW]
    splits         = [Split.TRAIN]
    has_test_label = False
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_images(self):
        patterns = [
            self.root / "kitti_snow" / self.split_str / "lq"
        ]
        self.images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self.images.append(image)

    def get_labels(self):
        self.labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path  = img.path.replace("/lq/", "/hq/")
                label = anno.ImageAnnotation(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="kitti_snow_s")
class KITTISnowS(dataset.ImageEnhancementDataset):
    """KITTI-Snow-S dataset.

    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    tasks          = [Task.DESNOW]
    splits         = [Split.TEST]
    has_test_label = True
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_images(self):
        patterns = [
            self.root / "kitti_snow_s" / self.split_str / "lq"
        ]
        self.images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:

            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self.images.append(image)

    def get_labels(self):
        self.labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path  = img.path.replace("/lq/", "/hq/")
                label = anno.ImageAnnotation(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="kitti_snow_m")
class KITTISnowM(dataset.ImageEnhancementDataset):
    """KITTI-Snow-M dataset.

    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    tasks          = [Task.DESNOW]
    splits         = [Split.TEST]
    has_test_label = True
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_images(self):
        patterns = [
            self.root / "kitti_snow_m" / self.split_str / "lq"
        ]
        self.images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self.images.append(image)

    def get_labels(self):
        self.labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path  = img.path.replace("/lq/", "/hq/")
                label = anno.ImageAnnotation(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="kitti_snow_l")
class KITTISnowL(dataset.ImageEnhancementDataset):
    """KITTI-Snow-L dataset.

    See Also: :class:`base.ImageEnhancementDataset`.
    """

    tasks          = [Task.DESNOW]
    splits         = [Split.TEST]
    has_test_label = True
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_images(self):
        patterns = [
            self.root / "kitti_snow_l" / self.split_str / "lq"
        ]
        self.images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self.images.append(image)

    def get_labels(self):
        self.labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path  = img.path.replace("/lq/", "/hq/")
                label = anno.ImageAnnotation(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="snow100k")
class Snow100K(dataset.ImageEnhancementDataset):
    """Snow100K dataset.
    
    See Also: :class:`base.ImageEnhancementDataset`.
    """

    tasks          = [Task.DESNOW]
    splits         = [Split.TRAIN]
    has_test_label = False
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_images(self):
        patterns = [
            self.root / "snow100k" / self.split_str / "lq"
        ]
        self.images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self.images.append(image)

    def get_labels(self):
        self.labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path  = img.path.replace("/lq/", "/hq/")
                label = anno.ImageAnnotation(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="snow100k_s")
class Snow100KS(dataset.ImageEnhancementDataset):
    """Snow100K-S dataset.
    
    See Also: :class:`base.ImageEnhancementDataset`.
    """

    tasks          = [Task.DESNOW]
    splits         = [Split.TEST]
    has_test_label = True
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_images(self):
        patterns = [
            self.root / "snow100k_s" / self.split_str / "lq"
        ]
        self.images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self.images.append(image)
    
    def get_labels(self):
        self.labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path  = img.path.replace("/lq/", "/hq/")
                label = anno.ImageAnnotation(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="snow100k_m")
class Snow100KM(dataset.ImageEnhancementDataset):
    """Snow100K-M dataset.
    
    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    tasks          = [Task.DESNOW]
    splits         = [Split.TEST]
    has_test_label = True
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_images(self):
        patterns = [
            self.root / "snow100k_m" / self.split_str / "lq"
        ]
        self.images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self.images.append(image)

    def get_labels(self):
        self.labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path  = img.path.replace("/lq/", "/hq/")
                label = anno.ImageAnnotation(path=path.image_file())
                self.labels.append(label)


@DATASETS.register(name="snow100k_l")
class Snow100KL(dataset.ImageEnhancementDataset):
    """Snow100K-L dataset.
    
    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    tasks          = [Task.DESNOW]
    splits         = [Split.TEST]
    has_test_label = True
    
    def __init__(self, root: core.Path = _default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_images(self):
        patterns = [
            self.root / "snow100k_l" / self.split_str / "lq"
        ]
        self.images: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        image = anno.ImageAnnotation(path=path)
                        self.images.append(image)

    def get_labels(self):
        self.labels: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path  = img.path.replace("/lq/", "/hq/")
                label = anno.ImageAnnotation(path=path.image_file())
                self.labels.append(label)

# endregion


# region Datamodule

@DATAMODULES.register(name="gtsnow")
class GTSnowDataModule(datamodule.DataModule):
    """GT-Snow datamodule.

    See Also: :class:`base.DataModule`.
    """
    
    tasks = [Task.DESNOW]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")

        if stage in [None, "train"]:
            self.train = GTSnow(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = GTSnow(split=Split.TRAIN, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = GTSnow(split=Split.TRAIN, **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()

        self.summarize()

    def get_classlabels(self):
        pass


@DATAMODULES.register(name="kitti_snow")
class KITTISnowDataModule(datamodule.DataModule):
    """KITTI-Snow datamodule.

    See Also: :class:`base.DataModule`.
    """
    
    tasks = [Task.DESNOW]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
       
        if stage in [None, "train"]:
            self.train = KITTISnow(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = KITTISnow(split=Split.TRAIN, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = KITTISnow(split=Split.TRAIN, **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        pass


@DATAMODULES.register(name="kitti_snow_s")
class KITTISnowSDataModule(datamodule.DataModule):
    """KITTI-Snow-S datamodule.

    See Also: :class:`base.DataModule`.
    """
    
    tasks = [Task.DESNOW]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = KITTISnowS(split=Split.TEST, **self.dataset_kwargs)
            self.val   = KITTISnowS(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = KITTISnowS(split=Split.TEST, **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        pass


@DATAMODULES.register(name="kitti_snow_m")
class KITTISnowMDataModule(datamodule.DataModule):
    """KITTI-Snow-M datamodule.

    See Also: :class:`base.DataModule`.
    """
    
    tasks = [Task.DESNOW]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = KITTISnowM(split=Split.TEST, **self.dataset_kwargs)
            self.val   = KITTISnowM(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = KITTISnowM(split=Split.TEST, **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        pass


@DATAMODULES.register(name="kitti_snow_l")
class KITTISnowLDataModule(datamodule.DataModule):
    """KITTI-Snow-L datamodule.

    See Also: :class:`base.DataModule`.
    """
    
    tasks = [Task.DESNOW]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = KITTISnowL(split=Split.TEST, **self.dataset_kwargs)
            self.val   = KITTISnowL(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = KITTISnowL(split=Split.TEST, **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        pass


@DATAMODULES.register(name="snow100k")
class Snow100KDataModule(datamodule.DataModule):
    """Snow100K datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    tasks = [Task.DESNOW]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = Snow100K(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = Snow100K(split=Split.TRAIN, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = Snow100K(split=Split.TRAIN, **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="snow100k_s")
class Snow100KSDataModule(datamodule.DataModule):
    """Snow100K-S datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    tasks = [Task.DESNOW]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = Snow100KS(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = Snow100KS(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = Snow100KS(split=Split.TEST,  **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="snow100k_m")
class Snow100KMDataModule(datamodule.DataModule):
    """Snow100K-M datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    tasks = [Task.DESNOW]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = Snow100KM(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = Snow100KM(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = Snow100KM(split=Split.TEST,  **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="snow100k_l")
class Snow100KLDataModule(datamodule.DataModule):
    """Snow100K-L datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    tasks = [Task.DESNOW]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = Snow100KL(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = Snow100KL(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = Snow100KL(split=Split.TEST,  **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass

# endregion
