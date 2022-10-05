#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A2I2Haze dataset and datamodule.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from torch.utils.data import random_split

from one.constants import *
from one.core import *
from one.data import ClassLabels
from one.data import ClassLabels_
from one.data import DataModule
from one.data import Detection
from one.data import Detections
from one.data import Image
from one.data import ImageDetectionDataset
from one.data import ImageEnhancementDataset
from one.plot import imshow_enhancement
from one.vision.shape import box_xyxy_to_cxcywh_norm
from one.vision.transformation import Resize

a2i2_classlabels = [
    { "name": "vehicle", "id": 0, "color": [  0,   0, 142] }
]


# H1: - Module -----------------------------------------------------------------

@DATASETS.register(name="a2i2haze")
class A2I2Haze(ImageEnhancementDataset):
    """
    A2I2-Haze dataset for image de-hazing.
    
    Args:
        name (str): Dataset's name.
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image shape as [C, H, W], [H, W], or [S, S].
        classlabels (ClassLabels_ | None): ClassLabels object. Defaults to
            None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str                 = "a2i2haze",
        root            : Path_               = DATA_DIR / "a2i2" / "haze",
        split           : str                 = "train",
        shape           : Ints                = (3, 512, 512),
        classlabels     : ClassLabels_ | None = None,
        transform       : Transforms_  | None = None,
        target_transform: Transforms_  | None = None,
        transforms      : Transforms_  | None = None,
        cache_data      : bool                = False,
        cache_images    : bool                = False,
        backend         : VisionBackend_      = VISION_BACKEND,
        verbose         : bool                = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
     
    def list_images(self):
        """
        List image files.
        """
        if self.split not in ["train"]:
            console.log(
                f"{self.__class__.classname} dataset only supports `split`: "
                f"`train`. Get: {self.split}."
            )
            
        self.images: list[Image] = []
        with progress_bar() as pbar:
            pattern = self.root / self.split / "haze_images"
            for path in pbar.track(
                list(pattern.rglob("*.jpg")),
                description=f"Listing {self.__class__.classname} "
                            f"{self.split} images"
            ):
                self.images.append(Image(path=path, backend=self.backend))
    
    def list_labels(self):
        """
        List label files.
        """
        self.labels: list[Image] = []
        with progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.classname} "
                            f"{self.split} labels"
            ):
                path = Path(str(img.path).replace("haze_images", "clean_images"))
                self.labels.append(Image(path=path, backend=self.backend))


@DATASETS.register(name="a2i2haze_det")
class A2I2HazeDet(ImageDetectionDataset):
    """
    
    Args:
        name (str): Dataset's name.
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image shape as [C, H, W], [H, W], or [S, S].
        classlabels (ClassLabels_ | None): ClassLabels object. Defaults to
            None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str                 = "a2i2haze_det",
        root            : Path_               = DATA_DIR / "a2i2" / "haze",
        split           : str                 = "train",
        shape           : Ints                = (3, 512, 512),
        classlabels     : ClassLabels_ | None = None,
        transform       : Transforms_  | None = None,
        target_transform: Transforms_  | None = None,
        transforms      : Transforms_  | None = None,
        cache_data      : bool                = False,
        cache_images    : bool                = False,
        backend         : VisionBackend_      = VISION_BACKEND,
        verbose         : bool                = True,
        *args, **kwargs
    ):
        path = os.path.join(root, f"classlabels.json")
        if not os.path.isfile(path):
            classlabels = ClassLabels.from_file(path)
        else:
            classlabels = ClassLabels.from_list(a2i2_classlabels)
        
        super().__init__(
            name             = name,
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
     
    def list_images(self):
        """
        List image files.
        """
        if self.split not in ["train", "dryrun"]:
            console.log(
                f"{self.__class__.classname} dataset only supports `split`: "
                f"`train` or `dryrun`. Get: {self.split}."
            )

        self.images: list[Image] = []
        with progress_bar() as pbar:
            if self.split == "train":
                pattern = self.root / "train" / "clean_images"
            elif self.split == "dryrun":
                pattern = self.root / "dryrun" / "images"
            for path in pbar.track(
                list(pattern.rglob("*.jpg")),
                description=f"Listing {self.__class__.classname} "
                            f"{self.split} images"
            ):
                self.images.append(Image(path=path, backend=self.backend))
    
    def list_labels(self):
        """
        List label files.
        """
        ann_files = self.annotation_files()
        
        self.labels: list[Detections] = []
        with progress_bar() as pbar:
            for i in pbar.track(
                range(len(ann_files)),
                description=f"Listing {self.__class__.classname} "
                            f"{self.split} labels"
            ):
                path = Path(ann_files[i])
                assert_txt_file(path)
                
                f      = open(path, "r")
                labels = [x.split() for x in f.read().splitlines()]
                shape  = self.images[i].shape
                
                detections: list[Detection] = []
                for i, l in enumerate(labels):
                    id              = l[0]
                    box_xyxy        = torch.Tensor(l[1:5])
                    box_cxcywh_norm = box_xyxy_to_cxcywh_norm(box_xyxy, shape[0], shape[1])
                    
                    if id.isnumeric():
                        id = int(id)
                    elif isinstance(self.classlabels, ClassLabels):
                        id = self.classlabels.get_id(key="name", value=id)
                    else:
                        id = -1
                        
                    detections.append(
                        Detection(
                            id         = id,
                            bbox       = box_cxcywh_norm,
                            confidence = 0.0,
                        )
                    )

                self.labels.append(Detections(detections))
            
    def annotation_files(self) -> Paths_:
        """
        Returns the path to json annotation files.
        """
        ann_files = []
        for img in self.images:
            if self.split == "train":
                path = str(img.path).replace("clean_images", "clean_images_labels")
            else:
                path = str(img.path).replace("images", "labels")
            path = Path(path.replace(".jpg", ".txt"))
            ann_files.append(path)
        return ann_files
        

@DATAMODULES.register(name="a2i2haze")
class A2I2HazeDataModule(DataModule):
    """
    A2I2Haze DataModule.
    """
    
    def __init__(
        self,
        name            : str                = "a2i2haze",
        root            : Path_              = DATA_DIR / "a2i2" / "haze",
        shape           : Ints               = (3, 512, 512),
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        batch_size      : int                = 1,
        devices         : Devices            = 0,
        shuffle         : bool               = True,
        collate_fn      : Callable    | None = None,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            shape            = shape,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            batch_size       = batch_size,
            devices          = devices,
            shuffle          = shuffle,
            collate_fn       = collate_fn,
            verbose          = verbose,
            *args, **kwargs
        )
        
    def prepare_data(self, *args, **kwargs):
        """
        Use this method to do things that might write to disk or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.load_classlabels()
    
    def setup(self, phase: ModelPhase_ | None = None):
        """
        There are also data operations you might want to perform on every GPU.

        Todos:
            - Count number of classes.
            - Build classlabels vocabulary.
            - Perform train/val/test splits.
            - Apply transforms (defined explicitly in your datamodule or
              assigned in init).
            - Define collate_fn for you custom dataset.

        Args:
            phase (ModelPhase_ | None):
                Stage to use: [None, ModelPhase.TRAINING, ModelPhase.TESTING].
                Set to None to setup all train, val, and test data.
                Defaults to None.
        """
        console.log(f"Setup [red]{A2I2Haze.classname}[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
            full_dataset = A2I2Haze(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            train_size   = int(0.8 * len(full_dataset))
            val_size     = len(full_dataset) - train_size
            self.train, self.val = random_split(
                full_dataset, [train_size, val_size]
            )
            self.classlabels = getattr(full_dataset, "classlabels", None)
            self.collate_fn  = getattr(full_dataset, "collate_fn",  None)
            
        # Assign test datasets for use in dataloader(s)
        if phase in [None, ModelPhase.TESTING]:
            self.test = A2I2Haze(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.load_classlabels()

        self.summarize()
        
    def load_classlabels(self):
        """
        Load ClassLabels.
        """
        pass


@DATAMODULES.register(name="a2i2haze_det")
class A2I2HazeDetDataModule(DataModule):
    """
    """
    
    def __init__(
        self,
        name            : str                = "a2i2haze_det",
        root            : Path_              = DATA_DIR / "a2i2" / "haze",
        shape           : Ints               = (3, 512, 512),
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        batch_size      : int                = 1,
        devices         : Devices            = 0,
        shuffle         : bool               = True,
        collate_fn      : Callable    | None = None,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            shape            = shape,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            batch_size       = batch_size,
            devices          = devices,
            shuffle          = shuffle,
            collate_fn       = collate_fn,
            verbose          = verbose,
            *args, **kwargs
        )
        
    def prepare_data(self, *args, **kwargs):
        """
        Use this method to do things that might write to disk or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.load_classlabels()
    
    def setup(self, phase: ModelPhase_ | None = None):
        """
        There are also data operations you might want to perform on every GPU.

        Todos:
            - Count number of classes.
            - Build classlabels vocabulary.
            - Perform train/val/test splits.
            - Apply transforms (defined explicitly in your datamodule or
              assigned in init).
            - Define collate_fn for you custom dataset.

        Args:
            phase (ModelPhase_ | None):
                Stage to use: [None, ModelPhase.TRAINING, ModelPhase.TESTING].
                Set to None to setup all train, val, and test data.
                Defaults to None.
        """
        console.log(f"Setup [red]{A2I2HazeDetDataModule.classname}[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
            full_dataset = A2I2HazeDet(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            train_size   = int(0.8 * len(full_dataset))
            val_size     = len(full_dataset) - train_size
            self.train, self.val = random_split(
                full_dataset, [train_size, val_size]
            )
            self.classlabels = getattr(full_dataset, "classlabels", None)
            self.collate_fn  = getattr(full_dataset, "collate_fn",  None)
            
        # Assign test datasets for use in dataloader(s)
        if phase in [None, ModelPhase.TESTING]:
            self.test = A2I2HazeDet(
                root             = self.root,
                split            = "dryrun",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.load_classlabels()

        self.summarize()
        
    def load_classlabels(self):
        """
        Load ClassLabels.
        """
        self.classlabels = ClassLabels.from_list(a2i2_classlabels)
    

# H1: - Test -------------------------------------------------------------------

def test_a2i2_haze():
    cfg = {
        "name": "a2i2haze",
            # Dataset's name.
        "root": DATA_DIR / "a2i2" / "haze",
            # Root directory of dataset.
        "shape": [3, 512, 512],
            # Image shape as [C, H, W], [H, W], or [S, S].
        "transform": None,
            # Functions/transforms that takes in an input sample and returns a
            # transformed version.
        "target_transform": None,
            # Functions/transforms that takes in a target and returns a
            # transformed version.
        "transforms": [
            Resize(size=[3, 512, 512])
        ],
            # Functions/transforms that takes in an input and a target and
            # returns the transformed versions of both.
        "cache_data": False,
            # If True, cache data to disk for faster loading next time.
            # Defaults to False.
        "cache_images": False,
            # If True, cache images into memory for faster training (WARNING:
            # large datasets may exceed system RAM). Defaults to False.
        "backend": VISION_BACKEND,
            # Vision backend to process image. Defaults to VISION_BACKEND.
        "batch_size": 8,
            # Number of samples in one forward & backward pass. Defaults to 1.
        "devices" : 0,
            # The devices to use. Defaults to 0.
        "shuffle": True,
            # If True, reshuffle the data at every training epoch.
            # Defaults to True.
        "verbose": True,
            # Verbosity. Defaults to True.
    }
    dm  = A2I2HazeDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter           = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    result              = {"image": input, "target": target}
    label               = [(m["name"]) for m in meta]
    imshow_enhancement(winname="image", image=result, label=label)
    plt.show(block=True)


# H1: - Main -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str , default="test_a2i2_haze", help="The task to run")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.task == "test_a2i2_haze":
        test_a2i2_haze()
