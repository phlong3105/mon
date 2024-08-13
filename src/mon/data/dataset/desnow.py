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
from mon.data.datastruct import annotation, datamodule, dataset
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console             = core.console
default_root_dir    = DATA_DIR / "desnow"
DatapointAttributes = annotation.DatapointAttributes
ImageAnnotation     = annotation.ImageAnnotation
ImageDataset        = dataset.ImageDataset


# region Dataset

@DATASETS.register(name="gtsnow")
class GTSnow(ImageDataset):
    """GT-Snow dataset.

    See Also: :class:`mon.data.datastruct.dataset.image.ImageDataset`.
    """
    
    tasks : list[Task]  = [Task.DESNOW]
    splits: list[Split] = [Split.TRAIN]
    datapoint_attrs     = DatapointAttributes({
        "lq_image": ImageAnnotation,
        "hq_image": ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "gtsnow" / self.split_str / "lq",
        ]
        
        # LQ images
        lq_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        lq_images.append(ImageAnnotation(path=path))
        
        # HQ images
        hq_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                lq_images,
                description=f"Listing {self.__class__.__name__} {self.split_str} ground-truths"
            ):
                path = str(img.path)
                path = path[:-9] + "C-000.png"
                path = path.replace("/lq/", "/hq/")
                path = core.Path(path)
                hq_images.append(ImageAnnotation(path=path.image_file()))
        
        self.datapoints["lq_image"] = lq_images
        self.datapoints["hq_image"] = hq_images
        

@DATASETS.register(name="kitti_snow")
class KITTISnow(ImageDataset):
    """KITTI-Snow dataset.

    See Also: :class:`mon.data.datastruct.dataset.image.ImageDataset`.
    """
    
    tasks : list[Task]  = [Task.DESNOW]
    splits: list[Split] = [Split.TRAIN]
    datapoint_attrs     = DatapointAttributes({
        "lq_image": ImageAnnotation,
        "hq_image": ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "kitti_snow" / self.split_str / "lq",
        ]
        
        # LQ images
        lq_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        lq_images.append(ImageAnnotation(path=path))
        
        # HQ images
        hq_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                lq_images,
                description=f"Listing {self.__class__.__name__} {self.split_str} ground-truths"
            ):
                path = img.path.replace("/lq/", "/hq/")
                hq_images.append(ImageAnnotation(path=path.image_file()))
        
        self.datapoints["lq_image"] = lq_images
        self.datapoints["hq_image"] = hq_images
        

@DATASETS.register(name="kitti_snow_s")
class KITTISnowS(ImageDataset):
    """KITTI-Snow-S dataset.

    See Also: :class:`mon.data.datastruct.dataset.image.ImageDataset`.
    """
    
    tasks : list[Task]  = [Task.DESNOW]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "lq_image": ImageAnnotation,
        "hq_image": ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "kitti_snow_s" / self.split_str / "lq",
        ]
        
        # LQ images
        lq_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        lq_images.append(ImageAnnotation(path=path))
        
        # HQ images
        hq_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                lq_images,
                description=f"Listing {self.__class__.__name__} {self.split_str} ground-truths"
            ):
                path = img.path.replace("/lq/", "/hq/")
                hq_images.append(ImageAnnotation(path=path.image_file()))
        
        self.datapoints["lq_image"] = lq_images
        self.datapoints["hq_image"] = hq_images
        

@DATASETS.register(name="kitti_snow_m")
class KITTISnowM(ImageDataset):
    """KITTI-Snow-M dataset.

    See Also: :class:`mon.data.datastruct.dataset.image.ImageDataset`.
    """
    
    tasks : list[Task]  = [Task.DESNOW]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "lq_image": ImageAnnotation,
        "hq_image": ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "kitti_snow_m" / self.split_str / "lq",
        ]
        
        # LQ images
        lq_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        lq_images.append(ImageAnnotation(path=path))
        
        # HQ images
        hq_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                lq_images,
                description=f"Listing {self.__class__.__name__} {self.split_str} ground-truths"
            ):
                path = img.path.replace("/lq/", "/hq/")
                hq_images.append(ImageAnnotation(path=path.image_file()))
        
        self.datapoints["lq_image"] = lq_images
        self.datapoints["hq_image"] = hq_images
        

@DATASETS.register(name="kitti_snow_l")
class KITTISnowL(ImageDataset):
    """KITTI-Snow-L dataset.

    See Also: :class:`mon.data.datastruct.dataset.image.ImageDataset`.
    """

    tasks : list[Task]  = [Task.DESNOW]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "lq_image": ImageAnnotation,
        "hq_image": ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "kitti_snow_l" / self.split_str / "lq",
        ]
        
        # LQ images
        lq_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        lq_images.append(ImageAnnotation(path=path))
        
        # HQ images
        hq_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                lq_images,
                description=f"Listing {self.__class__.__name__} {self.split_str} ground-truths"
            ):
                path = img.path.replace("/lq/", "/hq/")
                hq_images.append(ImageAnnotation(path=path.image_file()))
        
        self.datapoints["lq_image"] = lq_images
        self.datapoints["hq_image"] = hq_images
        

@DATASETS.register(name="snow100k")
class Snow100K(ImageDataset):
    """Snow100K dataset.
    
    See Also: :class:`mon.data.datastruct.dataset.image.ImageDataset`.
    """

    tasks : list[Task]  = [Task.DESNOW]
    splits: list[Split] = [Split.TRAIN]
    datapoint_attrs     = DatapointAttributes({
        "lq_image": ImageAnnotation,
        "hq_image": ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "snow100k" / self.split_str / "lq",
        ]
        
        # LQ images
        lq_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        lq_images.append(ImageAnnotation(path=path))
        
        # HQ images
        hq_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                lq_images,
                description=f"Listing {self.__class__.__name__} {self.split_str} ground-truths"
            ):
                path = img.path.replace("/lq/", "/hq/")
                hq_images.append(ImageAnnotation(path=path.image_file()))
        
        self.datapoints["lq_image"] = lq_images
        self.datapoints["hq_image"] = hq_images
        

@DATASETS.register(name="snow100k_s")
class Snow100KS(ImageDataset):
    """Snow100K-S dataset.
    
    See Also: :class:`mon.data.datastruct.dataset.image.ImageDataset`.
    """

    tasks : list[Task]  = [Task.DESNOW]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "lq_image": ImageAnnotation,
        "hq_image": ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "snow100k_s" / self.split_str / "lq",
        ]
        
        # LQ images
        lq_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        lq_images.append(ImageAnnotation(path=path))
        
        # HQ images
        hq_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                lq_images,
                description=f"Listing {self.__class__.__name__} {self.split_str} ground-truths"
            ):
                path = img.path.replace("/lq/", "/hq/")
                hq_images.append(ImageAnnotation(path=path.image_file()))
        
        self.datapoints["lq_image"] = lq_images
        self.datapoints["hq_image"] = hq_images
        

@DATASETS.register(name="snow100k_m")
class Snow100KM(ImageDataset):
    """Snow100K-M dataset.
    
    See Also: :class:`mon.data.datastruct.dataset.image.ImageDataset`.
    """
    
    tasks : list[Task]  = [Task.DESNOW]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "lq_image": ImageAnnotation,
        "hq_image": ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "snow100k_m" / self.split_str / "lq",
        ]
        
        # LQ images
        lq_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        lq_images.append(ImageAnnotation(path=path))
        
        # HQ images
        hq_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                lq_images,
                description=f"Listing {self.__class__.__name__} {self.split_str} ground-truths"
            ):
                path = img.path.replace("/lq/", "/hq/")
                hq_images.append(ImageAnnotation(path=path.image_file()))
        
        self.datapoints["lq_image"] = lq_images
        self.datapoints["hq_image"] = hq_images
        

@DATASETS.register(name="snow100k_l")
class Snow100KL(ImageDataset):
    """Snow100K-L dataset.
    
    See Also: :class:`mon.data.datastruct.dataset.image.ImageDataset`.
    """
    
    tasks : list[Task]  = [Task.DESNOW]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "lq_image": ImageAnnotation,
        "hq_image": ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "snow100k_l" / self.split_str / "lq",
        ]
        
        # LQ images
        lq_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        lq_images.append(ImageAnnotation(path=path))
        
        # HQ images
        hq_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                lq_images,
                description=f"Listing {self.__class__.__name__} {self.split_str} ground-truths"
            ):
                path = img.path.replace("/lq/", "/hq/")
                hq_images.append(ImageAnnotation(path=path.image_file()))
        
        self.datapoints["lq_image"] = lq_images
        self.datapoints["hq_image"] = hq_images
        
# endregion


# region Datamodule

@DATAMODULES.register(name="gtsnow")
class GTSnowDataModule(datamodule.DataModule):
    """GT-Snow datamodule.

    See Also: :class:`mon.data.datastruct.datamodule.DataModule`.
    """
    
    tasks: list[Task] = [Task.DESNOW]
    
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

    See Also: :class:`mon.data.datastruct.datamodule.DataModule`.
    """
    
    tasks: list[Task] = [Task.DESNOW]
    
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

    See Also: :class:`mon.data.datastruct.datamodule.DataModule`.
    """
    
    tasks: list[Task] = [Task.DESNOW]
    
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

    See Also: :class:`mon.data.datastruct.datamodule.DataModule`.
    """
    
    tasks: list[Task] = [Task.DESNOW]
    
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

    See Also: :class:`mon.data.datastruct.datamodule.DataModule`.
    """
    
    tasks: list[Task] = [Task.DESNOW]
    
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
    
    See Also: :class:`mon.data.datastruct.datamodule.DataModule`.
    """
    
    tasks: list[Task] = [Task.DESNOW]
    
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
    
    See Also: :class:`mon.data.datastruct.datamodule.DataModule`.
    """
    
    tasks: list[Task] = [Task.DESNOW]
    
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
    
    See Also: :class:`mon.data.datastruct.datamodule.DataModule`.
    """
    
    tasks: list[Task] = [Task.DESNOW]
    
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
    
    See Also: :class:`mon.data.datastruct.datamodule.DataModule`.
    """
    
    tasks: list[Task] = [Task.DESNOW]
    
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
