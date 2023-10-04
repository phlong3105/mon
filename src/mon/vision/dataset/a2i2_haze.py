#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements A2I2-Haze datasets and datamodules."""

from __future__ import annotations

__all__ = [
    "A2I2Haze", "A2I2HazeDataModule",
]

from torch.utils.data import random_split

from mon.core import console, pathlib, rich
from mon.globals import DATAMODULES, DATASETS, ModelPhase
from mon.vision.dataset import base

# region ClassLabels

a2i2_classlabels = [
    {"name": "vehicle", "id": 0, "color": [0, 0, 142]},
]

# endregion


# region Dataset

@DATASETS.register(name="a2i2-haze")
class A2I2Haze(base.ImageEnhancementDataset):
    """A2I2-Haze dataset.
    
    See Also: :class:`mon.vision.dataset.base.dataset.ImageEnhancementDataset`.
    """
    
    def get_images(self):
        """Get image files."""
        if self.split not in ["train"]:
            console.log(
                f"split must be one of ['train'], but got {self.split}."
            )
        
        self.images: list[base.ImageLabel] = []
        with rich.get_progress_bar() as pbar:
            pattern = self.root / self.split / "dehazing" / "haze-images"
            for path in pbar.track(
                list(pattern.rglob("*.jpg")),
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
                path  = str(img.path).replace("haze-images", "hazefree-images")
                path  = pathlib.Path(path)
                label = base.ImageLabel(path=path)
                self.labels.append(label)

'''
@DATASETS.register(name="a2i2-haze-det")
class A2I2HazeDet(base.ImageDetectionDataset):
    """A2I2-Haze Detection.
    
    Args:
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Default: "train".
        image_size: The desired datapoint shape preferably in a channel-last
        format.
            Default: (3, 256, 256).
        classlabels: A :class:`ClassLabels` object. Default: None.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Default: False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Default: False.
        backend: The image processing backend. Default: VISION_BACKEND.
        verbose: Verbosity. Default: True.
    """
    
    def __init__(
        self,
        root: PathType = constant.DATA_DIR / "a2i2-haze",
        split: str = "train",
        image_size: Ints = (3, 256, 256),
        classlabels: ClassLabelsType | None = a2i2_classlabels,
        transform: TransformType | None = None,
        target_transform: TransformType | None = None,
        transforms: TransformType | None = None,
        cache_data: bool = False,
        cache_images: bool = False,
        backend: VisionBackendType = constant.VISION_BACKEND,
        verbose: bool = True,
        *args, **kwargs
    ):
        super().__init__(
            root=root,
            split=split,
            image_size=image_size,
            classlabels=classlabels or a2i2_classlabels,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            cache_data=cache_data,
            cache_images=cache_images,
            backend=backend,
            verbose=verbose,
            *args, **kwargs
        )
    
    def get_images(self):
        """Get image files."""
        if self.split not in ["train", "dry-run", "test"]:
            core.console.log(
                f"{self.__class__.__name__} dataset only supports "
                f"'split': 'train', 'dryrun', or 'test'. Get: {self.split}."
            )
        
        self.images: list[base.ImageLabel] = []
        with core.rich.get_progress_bar() as pbar:
            if self.split == "train":
                pattern = self.root / "train" / "detection" / "hazefree-images"
            elif self.split == "dry-run":
                pattern = self.root / "dry-run" / "2023" / "images"
            elif self.split == "test":
                pattern = self.root / "test" / "images"
            
            for path in pbar.track(
                list(pattern.rglob("*.jpg")),
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split} images"
            ):
                self.images.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )
    
    def get_labels(self):
        """Get label files."""
        ann_files = self.annotation_files()
        
        self.labels: list[base.DetectionsLabel] = []
        with core.rich.get_progress_bar() as pbar:
            for i in pbar.track(
                range(len(ann_files)),
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split} labels"
            ):
                path = core.Path(ann_files[i])
                assert path.is_txt_file()
                
                f = open(path, "r")
                labels = [x.split() for x in f.read().splitlines()]
                shape = self.images[i].shape
                
                detections: list[base.DetectionLabel] = []
                for _, l in enumerate(labels):
                    id = l[0]
                    box_xyxy = torch.Tensor(l[1:5])
                    box_cxcywh_norm = ci.bbox_xyxy_to_cxcywhn(
                        bbox=box_xyxy,
                        height=shape[0],
                        width=shape[1],
                    )
                    
                    if id.isnumeric():
                        id = int(id)
                    elif isinstance(self.classlabels, base.ClassLabels):
                        id = self.classlabels.get_id(key="name", value=id)
                    else:
                        id = -1
                    
                    detections.append(
                        base.DetectionLabel(
                            id_=id,
                            bbox=box_cxcywh_norm,
                            confidence=0.0,
                        )
                    )
                
                self.labels.append(base.DetectionsLabel(detections))
    
    def annotation_files(self) -> PathsType:
        """Return the path to '.txt' annotation files."""
        ann_files = []
        for img in self.images:
            if self.split == "train":
                path = str(img.path).replace(
                    "hazefree-images",
                    "hazefree-labels"
                )
            else:
                path = str(img.path).replace("images", "labels")
            path = core.Path(path.replace(".jpg", ".txt"))
            ann_files.append(path)
        return ann_files
'''

# endregion


# region Datamodule

@DATAMODULES.register(name="a2i2-haze")
class A2I2HazeDataModule(base.DataModule):
    """A2I2-Haze datamodule.
    
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
            dataset = A2I2Haze(split="train", **self.dataset_kwargs)
            self.split_train_val(dataset=dataset, split_ratio=0.8, full_train=True)
        if phase in [None, ModelPhase.TESTING]:
            self.test = A2I2Haze(split="train", **self.dataset_kwargs)
        
        if self.classlabels is None:
            self.get_classlabels()
        
        self.summarize()
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass

# endregion
