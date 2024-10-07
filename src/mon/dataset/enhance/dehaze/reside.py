#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RESIDE Datasets."""

from __future__ import annotations

__all__ = [
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
]

from typing import Literal

from mon import core
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console             = core.console
default_root_dir    = DATA_DIR / "enhance" / "dehaze"
DataModule          = core.DataModule
DatapointAttributes = core.DatapointAttributes
DepthMapAnnotation  = core.DepthMapAnnotation
ImageAnnotation     = core.ImageAnnotation
MultimodalDataset   = core.MultimodalDataset


@DATASETS.register(name="reside_hsts_real")
class RESIDEHSTSReal(MultimodalDataset):
    """RESIDE-HSTS-Real dataset consists of 10 real hazy images."""
    
    tasks : list[Task]  = [Task.DEHAZE]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image": ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "reside_hsts_real" / self.split_str / "image",
        ]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images
        
        
@DATASETS.register(name="reside_hsts_syn")
class RESIDEHSTSSyn(MultimodalDataset):
    """RESIDE-HSTS-Syn dataset consists of ``10`` pairs of hazy and
    corresponding haze-free images.
    """
    
    tasks : list[Task]  = [Task.DEHAZE]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "reside_hsts_syn" / self.split_str / "image",
        ]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images
        

@DATASETS.register(name="reside_its")
class RESIDEITS(MultimodalDataset):
    """RESIDE-ITS dataset consists of ``13,990`` pairs of hazy and corresponding
    haze-free images.
    """
    
    tasks : list[Task]  = [Task.DEHAZE]
    splits: list[Split] = [Split.TRAIN, Split.VAL]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "reside_its" / self.split_str / "image",
        ]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = (list(pattern.rglob("*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        # Reference images
        ref_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                sequence    = images,
                description = f"Listing {self.__class__.__name__} {self.split_str} reference images"
            ):
                stem = str(img.path.stem).split("_")[0]
                path = img.path.replace("/image/", "/ref/")
                path = path.parent / f"{stem}.{img.path.suffix}"
                ref_images.append(ImageAnnotation(path=path.image_file(), root=pattern))
        
        self.datapoints["image"]     = images
        self.datapoints["ref_image"] = ref_images
        

@DATASETS.register(name="reside_its_v2")
class RESIDEITSV2(MultimodalDataset):
    """RESIDE-ITS-V2 dataset consists of ``13,990`` pairs of hazy and
    corresponding haze-free images.
    """
    
    tasks : list[Task]  = [Task.DEHAZE]
    splits: list[Split] = [Split.TRAIN]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "reside_its_v2" / self.split_str / "image",
        ]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        # Reference images
        ref_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                sequence    = images,
                description = f"Listing {self.__class__.__name__} {self.split_str} reference images"
            ):
                stem = str(img.path.stem).split("_")[0]
                path = img.path.replace("/image/", "/ref/")
                path = path.parent / f"{stem}.{img.path.suffix}"
                ref_images.append(ImageAnnotation(path=path.image_file(), root=pattern))
        
        self.datapoints["image"]     = images
        self.datapoints["ref_image"] = ref_images
        

@DATASETS.register(name="reside_ots")
class RESIDEOTS(MultimodalDataset):
    """RESIDE-OTS dataset consists of ``73,135`` pairs of hazy and corresponding
    haze-free images.
    """
    
    tasks : list[Task]  = [Task.DEHAZE]
    splits: list[Split] = [Split.TRAIN]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "reside_ots" / self.split_str / "image",
        ]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        # Reference images
        ref_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                sequence    = images,
                description = f"Listing {self.__class__.__name__} {self.split_str} reference images"
            ):
                stem = str(img.path.stem).split("_")[0]
                path = img.path.replace("/image/", "/ref/")
                path = path.parent / f"{stem}.{img.path.suffix}"
                ref_images.append(ImageAnnotation(path=path.image_file(), root=pattern))
        
        self.datapoints["image"]     = images
        self.datapoints["ref_image"] = ref_images
        

@DATASETS.register(name="reside_rtts")
class RESIDERTTS(MultimodalDataset):
    """RESIDE-RTTS dataset consists of ``4,322`` real hazy images."""
    
    tasks : list[Task]  = [Task.DEHAZE]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image": ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "reside_rtts" / self.split_str / "image",
        ]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images
        

@DATASETS.register(name="reside_sots_indoor")
class RESIDESOTSIndoor(MultimodalDataset):
    """RESIDE-SOTS-Indoor dataset consists of ``500`` pairs of hazy and
    corresponding haze-free images.
    """
    
    tasks : list[Task]  = [Task.DEHAZE]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "reside_sots_indoor " / self.split_str / "image",
        ]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        # Reference images
        ref_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                sequence    = images,
                description = f"Listing {self.__class__.__name__} {self.split_str} hq images"
            ):
                stem = str(img.path.stem).split("_")[0]
                path = img.path.replace("/image/", "/ref/")
                path = path.parent / f"{stem}.{img.path.suffix}"
                ref_images.append(ImageAnnotation(path=path.image_file(), root=pattern))
        
        self.datapoints["image"]     = images
        self.datapoints["ref_image"] = ref_images
        

@DATASETS.register(name="reside_sots_outdoor")
class RESIDESOTSOutdoor(MultimodalDataset):
    """RESIDE-SOTS-Outdoor dataset consists of ``500`` pairs of hazy and
    corresponding haze-free images.
    """
    
    tasks : list[Task]  = [Task.DEHAZE]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "reside_sots_outdoor" / self.split_str / "image",
        ]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))

        # Reference images
        ref_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                sequence    = images,
                description = f"Listing {self.__class__.__name__} {self.split_str} reference images"
            ):
                stem = str(img.path.stem).split("_")[0]
                path = img.path.replace("/image/", "/ref/")
                path = path.parent / f"{stem}.{img.path.suffix}"
                ref_images.append(ImageAnnotation(path=path.image_file(), root=pattern))
        
        self.datapoints["image"]     = images
        self.datapoints["ref_image"] = ref_images


@DATASETS.register(name="reside_uhi")
class RESIDEUHI(MultimodalDataset):
    """RESIDE-UHI dataset consists of ``4,809`` real hazy images."""
    
    tasks : list[Task]  = [Task.DEHAZE]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image": ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "reside_uhi" / self.split_str / "image"
        ]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images
        

@DATAMODULES.register(name="reside_hsts_real")
class RESIDEHSTSRealDataModule(DataModule):

    tasks: list[Task] = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = RESIDEHSTSReal(split=Split.TEST, **self.dataset_kwargs)
            self.val   = RESIDEHSTSReal(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = RESIDEHSTSReal(split=Split.TEST, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="reside_hsts_syn")
class RESIDEHSTSSynDataModule(DataModule):

    tasks: list[Task] = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = RESIDEHSTSSyn(split=Split.TEST, **self.dataset_kwargs)
            self.val   = RESIDEHSTSSyn(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test = RESIDEHSTSSyn(split=Split.TEST,  **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="reside_its")
class RESIDEITSDataModule(DataModule):

    tasks: list[Task] = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = RESIDEITS(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = RESIDEITS(split=Split.VAL,   **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = RESIDEITS(split=Split.TEST,  **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="reside_its_v2")
class RESIDEITSV2DataModule(DataModule):

    tasks: list[Task] = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = RESIDEITSV2(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   =   RESIDEITS(split=Split.VAL,   **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  =   RESIDEITS(split=Split.TEST,  **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="reside_ots")
class RESIDEOTSDataModule(DataModule):

    tasks: list[Task] = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = RESIDEOTS(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = RESIDEITS(split=Split.VAL,   **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = RESIDEITS(split=Split.TEST,  **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="reside_rtts")
class RESIDERTTSDataModule(DataModule):

    tasks: list[Task] = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = RESIDERTTS(split=Split.TEST, **self.dataset_kwargs)
            self.val   = RESIDERTTS(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test = RESIDERTTS(split=Split.TEST,  **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="reside_sots_indoor")
class RESIDESOTSIndoorDataModule(DataModule):

    tasks: list[Task] = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = RESIDESOTSIndoor(split=Split.TEST, **self.dataset_kwargs)
            self.val   = RESIDESOTSIndoor(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test = RESIDESOTSIndoor(split=Split.TEST,  **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="reside_sots_outdoor")
class RESIDESOTSOutdoorDataModule(DataModule):

    tasks: list[Task] = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = RESIDESOTSOutdoor(split=Split.TEST, **self.dataset_kwargs)
            self.val   = RESIDESOTSOutdoor(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = RESIDESOTSOutdoor(split=Split.TEST, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="reside_uhi")
class RESIDEUHIDataModule(DataModule):

    tasks: list[Task] = [Task.DEHAZE]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = RESIDEUHI(split=Split.TEST, **self.dataset_kwargs)
            self.val   = RESIDEUHI(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = RESIDEUHI(split=Split.TEST, **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
