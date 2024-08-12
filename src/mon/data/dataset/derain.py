#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements de-raining datasets and datamodules."""

from __future__ import annotations

__all__ = [
    "GTRain",
    "GTRainDataModule",
    "Rain100",
    "Rain100DataModule",
    "Rain100H",
    "Rain100HDataModule",
    "Rain100L",
    "Rain100LDataModule",
    "Rain12",
    "Rain1200",
    "Rain1200DataModule",
    "Rain12DataModule",
    "Rain13K",
    "Rain13KDataModule",
    "Rain1400",
    "Rain1400DataModule",
    "Rain2800",
    "Rain2800DataModule",
    "Rain800",
    "Rain800DataModule",
]

from typing import Literal

from mon import core
from mon.data.datastruct import annotation as anno, datamodule, dataset
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console          = core.console
default_root_dir = DATA_DIR / "derain"


# region Dataset

@DATASETS.register(name="gtrain")
class GTRain(dataset.ImageEnhancementDataset):
    """GT-Rain dataset consists 26124 train and 1793 val pairs of rain/no-rain images.

    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    tasks  = [Task.DERAIN]
    splits = [Split.TRAIN, Split.VAL, Split.TEST]
    has_test_annotations = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_images(self):
        patterns = [
            self.root / "gtrain" / self.split_str / "lq"
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

    def get_annotations(self):
        self.annotations: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path = str(img.path)
                if "Gurutto_1-2" in path:
                    path = path.replace("-R-", "-C-")
                else:
                    path = path[:-9] + "C-000.png"
                path = path.replace("/lq/", "/hq/")
                path = core.Path(path)
                ann  = anno.ImageAnnotation(path=path.image_file())
                self.annotations.append(ann)


@DATASETS.register(name="rain100")
class Rain100(dataset.ImageEnhancementDataset):
    """Rain100 dataset consists 100 pairs of rain/no-rain test images.
    
    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    tasks  = [Task.DERAIN]
    splits = [Split.TEST]
    has_test_annotations = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_images(self):
        patterns = [
            self.root / "rain100" / self.split_str / "lq"
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

    def get_annotations(self):
        self.annotations: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path = img.path.replace("/lq/", "/hq/")
                ann  = anno.ImageAnnotation(path=path.image_file())
                self.annotations.append(ann)


@DATASETS.register(name="rain100h")
class Rain100H(dataset.ImageEnhancementDataset):
    """Rain100H dataset consists 100 pairs of rain/no-rain test images and 100
    pairs of rain/no-rain train-val images.
    
    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    tasks  = [Task.DERAIN]
    splits = [Split.TRAIN, Split.TEST]
    has_test_annotations = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_images(self):
        patterns = [
            self.root / "rain100h" / self.split_str / "lq"
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

    def get_annotations(self):
        self.annotations: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path = img.path.replace("/lq/", "/hq/")
                ann  = anno.ImageAnnotation(path=path.image_file())
                self.annotations.append(ann)


@DATASETS.register(name="rain100l")
class Rain100L(dataset.ImageEnhancementDataset):
    """Rain100L dataset consists 100 pairs of rain/no-rain test images and 200
    pairs of rain/no-rain train-val images.
    
    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    tasks  = [Task.DERAIN]
    splits = [Split.TRAIN, Split.TEST]
    has_test_annotations = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)

    def get_images(self):
        patterns = [
            self.root / "rain100l" / self.split_str / "lq"
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

    def get_annotations(self):
        self.annotations: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path = img.path.replace("/lq/", "/hq/")
                ann  = anno.ImageAnnotation(path=path.image_file())
                self.annotations.append(ann)


@DATASETS.register(name="rain12")
class Rain12(dataset.ImageEnhancementDataset):
    """Rain12 dataset consists 12 pairs of rain/no-rain images.

    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    tasks  = [Task.DERAIN]
    splits = [Split.TRAIN]
    has_test_annotations = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_images(self):
        patterns = [
            self.root / "rain12" / self.split_str / "lq"
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

    def get_annotations(self):
        self.annotations: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path = img.path.replace("/lq/", "/hq/")
                ann  = anno.ImageAnnotation(path=path.image_file())
                self.annotations.append(ann)


@DATASETS.register(name="rain1200")
class Rain1200(dataset.ImageEnhancementDataset):
    """Rain1200 dataset consists 1200 pairs of rain/no-rain test images and
    12,000 pairs of rain/no-rain train images.
    
    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    tasks  = [Task.DERAIN]
    splits = [Split.TRAIN, Split.VAL, Split.TEST]
    has_test_annotations = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_images(self):
        if self.split in [Split.TRAIN]:
            patterns = [
                self.root / "rain1200_light" / self.split_str / "lq",
                self.root / "rain1200_medium" / self.split_str / "lq",
                self.root / "rain1200_heavy" / self.split_str / "lq"
            ]
        else:
            patterns = [
                self.root / "rain1200" / self.split_str / "lq"
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

    def get_annotations(self):
        self.annotations: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path = img.path.replace("/lq/", "/hq/")
                ann  = anno.ImageAnnotation(path=path.image_file())
                self.annotations.append(ann)


@DATASETS.register(name="rain13k")
class Rain13K(dataset.ImageEnhancementDataset):
    """Rain13K dataset consists 13k pairs of rain/no-rain train images.
    
    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    tasks  = [Task.DERAIN]
    splits = [Split.TRAIN, Split.VAL, Split.TEST]
    has_test_annotations = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_images(self):
        if self.split in [Split.TRAIN]:
            patterns = [
                self.root / "rain13k" / self.split_str / "lq",
            ]
        elif self.split in [Split.VAL]:
            patterns = [
                self.root / "rain1200" / self.split_str / "lq",
                self.root / "rain800" / self.split_str / "lq",
            ]
        else:
            patterns = [
                self.root / "rain100" / self.split_str / "lq",
                self.root / "rain100h" / self.split_str / "lq",
                self.root / "rain100l" / self.split_str / "lq",
                self.root / "rain1200" / self.split_str / "lq",
                self.root / "rain1400" / self.split_str / "lq",
                self.root / "rain2800" / self.split_str / "lq",
                self.root / "rain800" / self.split_str / "lq",
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

    def get_annotations(self):
        self.annotations: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path = img.path.replace("/lq/", "/hq/")
                ann  = anno.ImageAnnotation(path=path.image_file())
                self.annotations.append(ann)


@DATASETS.register(name="rain1400")
class Rain1400(dataset.ImageEnhancementDataset):
    """Rain1400 dataset consists 1400 pairs of rain/no-rain test images and
    12,600 pairs of rain/no-rain train images.
    
    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    tasks  = [Task.DERAIN]
    splits = [Split.TRAIN, Split.TEST]
    has_test_annotations = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_images(self):
        patterns = [
            self.root / "rain1400" / self.split_str / "lq"
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

    def get_annotations(self):
        self.annotations: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path = img.path.replace("/lq/", "/hq/")
                ann  = anno.ImageAnnotation(path=path.image_file())
                self.annotations.append(ann)


@DATASETS.register(name="rain2800")
class Rain2800(dataset.ImageEnhancementDataset):
    """Rain2800 dataset consists 2800 pairs of rain/no-rain test images.
    
    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    tasks  = [Task.DERAIN]
    splits = [Split.TEST]
    has_test_annotations = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_images(self):
        patterns = [
            self.root / "rain2800" / self.split_str / "lq"
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

    def get_annotations(self):
        self.annotations: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path = img.path.replace("/lq/", "/hq/")
                ann  = anno.ImageAnnotation(path=path.image_file())
                self.annotations.append(ann)


@DATASETS.register(name="rain800")
class Rain800(dataset.ImageEnhancementDataset):
    """Rain800 dataset consists 800 pairs of rain/no-rain train-val images.
    
    See Also: :class:`base.ImageEnhancementDataset`.
    """
    
    tasks  = [Task.DERAIN]
    splits = [Split.TRAIN, Split.VAL, Split.TEST]
    has_test_annotations = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_images(self):
        patterns = [
            self.root / "rain800" / self.split_str / "lq"
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

    def get_annotations(self):
        self.annotations: list[anno.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split_str} labels"
            ):
                path = img.path.replace("/lq/", "/hq/")
                ann  = anno.ImageAnnotation(path=path.image_file())
                self.annotations.append(ann)

# endregion


# region Datamodule

@DATAMODULES.register(name="gtrain")
class GTRainDataModule(datamodule.DataModule):
    """GT-Rain datamodule.

    See Also: :class:`base.DataModule`.
    """
    
    tasks = [Task.DERAIN]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = GTRain(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = GTRain(split=Split.VAL,   **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = GTRain(split=Split.TEST,  **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()

    def get_classlabels(self):
        pass


@DATAMODULES.register(name="rain100")
class Rain100DataModule(datamodule.DataModule):
    """Rain100 datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    tasks = [Task.DERAIN]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = Rain100(split=Split.TEST, **self.dataset_kwargs)
            self.val   = Rain100(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = Rain100(split=Split.TEST, **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="rain100h")
class Rain100HDataModule(datamodule.DataModule):
    """Rain100H datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    tasks = [Task.DERAIN]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = Rain100H(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = Rain100H(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = Rain100H(split=Split.TEST,  **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="rain100l")
class Rain100LDataModule(datamodule.DataModule):
    """Rain100L datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    tasks = [Task.DERAIN]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = Rain100L(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = Rain100L(split=Split.TRAIN, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = Rain100L(split=Split.TEST,  **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="rain12")
class Rain12DataModule(datamodule.DataModule):
    """Rain12 datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    tasks = [Task.DERAIN]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = Rain12(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = Rain12(split=Split.TRAIN, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = Rain12(split=Split.TRAIN, **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="rain1200")
class Rain1200DataModule(datamodule.DataModule):
    """Rain1200 datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    tasks = [Task.DERAIN]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = Rain1200(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = Rain1200(split=Split.VAL,   **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = Rain1200(split=Split.TEST,  **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="rain13k")
class Rain13KDataModule(datamodule.DataModule):
    """Rain13K datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    tasks = [Task.DERAIN]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = Rain13K(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = Rain100(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            # self.test  = Rain13K(split=Split.TEST,  **self.dataset_kwargs)
            self.test  = Rain100(split=Split.TEST,  **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="rain1400")
class Rain1400DataModule(datamodule.DataModule):
    """Rain1400 datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    tasks = [Task.DERAIN]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = Rain1400(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = Rain1400(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = Rain1400(split=Split.TEST,  **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="rain2800")
class Rain2800DataModule(datamodule.DataModule):
    """Rain2800 datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    tasks = [Task.DERAIN]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
       
        if stage in [None, "train"]:
            self.train = Rain2800(split=Split.TEST, **self.dataset_kwargs)
            self.val   = Rain2800(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = Rain2800(split=Split.TEST, **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass


@DATAMODULES.register(name="rain800")
class Rain800DataModule(datamodule.DataModule):
    """Rain800 datamodule.
    
    See Also: :class:`base.DataModule`.
    """
    
    tasks = [Task.DERAIN]
    
    def prepare_data(self, *args, **kwargs):
        if self.classlabels is None:
            self.get_classlabels()
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
       
        if stage in [None, "train"]:
            self.train = Rain800(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = Rain800(split=Split.VAL,   **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = Rain800(split=Split.TEST,  **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        if self.can_log:
            self.summarize()
    
    def get_classlabels(self):
        pass

# endregion
