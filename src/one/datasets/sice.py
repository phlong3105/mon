#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SICE dataset and datamodule.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from torch.utils.data import random_split

from one.constants import *
from one.core import *
from one.data import ClassLabels_
from one.data import DataModule
from one.data import Image
from one.data import ImageEnhancementDataset
from one.data import UnlabeledImageDataset
from one.plot import imshow
from one.vision.transformation import Resize


# MARK: - Module ---------------------------------------------------------------

@DATASETS.register(name="sice")
class SICE(ImageEnhancementDataset):
    """
    SICE dataset.
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image shape as [H, W, C], [H, W], or [S, S].
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
        root            : Path_,
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
        if self.split not in ["train", "test"]:
            console.log(
                f"{self.__class__.classname} dataset only supports `split`: "
                f"`train` or `test`. Get: {self.split}."
            )
            
        self.images: list[Image] = []
        with progress_bar() as pbar:
            pattern = self.root / self.split / "low"
            for path in pbar.track(
                list(pattern.rglob("*.JPG")),
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
                stem = str(img.path.parent.stem)
                path = self.root / self.split / "high" / f"{stem}.JPG"
                self.labels.append(Image(path=path, backend=self.backend))
   

@DATASETS.register(name="sice1")
class SICE1(SICE):
    """
    SICE dataset with exposure level 1.
    """
    
    def list_images(self):
        """
        List image files.
        """
        if self.split not in ["train", "test"]:
            console.log(
                f"{self.__class__.classname} dataset only supports `split`: "
                f"`train` or `test`. Get: {self.split}."
            )
            
        self.images: list[Image] = []
        with progress_bar() as pbar:
            pattern = self.root / self.split / "low"
            for path in pbar.track(
                list(pattern.rglob("*1.JPG")),
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
                stem = str(img.path.parent.stem)
                path = self.root / self.split / "high" / f"{stem}.JPG"
                self.labels.append(Image(path=path, backend=self.backend))


@DATASETS.register(name="sice2")
class SICE2(SICE):
    """
    SICE dataset with exposure level 2.
    """
    
    def list_images(self):
        """
        List image files.
        """
        if self.split not in ["train", "test"]:
            console.log(
                f"{self.__class__.classname} dataset only supports `split`: "
                f"`train` or `test`. Get: {self.split}."
            )
            
        self.images: list[Image] = []
        with progress_bar() as pbar:
            pattern = self.root / self.split / "low"
            for path in pbar.track(
                list(pattern.rglob("*2.JPG")),
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
                stem = str(img.path.parent.stem)
                path = self.root / self.split / "high" / f"{stem}.JPG"
                self.labels.append(Image(path=path, backend=self.backend))
                

@DATASETS.register(name="sice3")
class SICE3(SICE):
    """
    SICE dataset with exposure level 3.
    """
    
    def list_images(self):
        """
        List image files.
        """
        if self.split not in ["train", "test"]:
            console.log(
                f"{self.__class__.classname} dataset only supports `split`: "
                f"`train` or `test`. Get: {self.split}."
            )
            
        self.images: list[Image] = []
        with progress_bar() as pbar:
            pattern = self.root / self.split / "low"
            for path in pbar.track(
                list(pattern.rglob("*3.JPG")),
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
                stem = str(img.path.parent.stem)
                path = self.root / self.split / "high" / f"{stem}.JPG"
                self.labels.append(Image(path=path, backend=self.backend))
                

@DATASETS.register(name="sice4")
class SICE4(SICE):
    """
    SICE dataset with exposure level 4.
    """
    
    def list_images(self):
        """
        List image files.
        """
        if self.split not in ["train", "test"]:
            console.log(
                f"{self.__class__.classname} dataset only supports `split`: "
                f"`train` or `test`. Get: {self.split}."
            )
            
        self.images: list[Image] = []
        with progress_bar() as pbar:
            pattern = self.root / self.split / "low"
            for path in pbar.track(
                list(pattern.rglob("*4.JPG")),
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
                stem = str(img.path.parent.stem)
                path = self.root / self.split / "high" / f"{stem}.JPG"
                self.labels.append(Image(path=path, backend=self.backend))
                

@DATASETS.register(name="sice5")
class SICE5(SICE):
    """
    SICE dataset with exposure level 5.
    """
    
    def list_images(self):
        """
        List image files.
        """
        if self.split not in ["train", "test"]:
            console.log(
                f"{self.__class__.classname} dataset only supports `split`: "
                f"`train` or `test`. Get: {self.split}."
            )
            
        self.images: list[Image] = []
        with progress_bar() as pbar:
            pattern = self.root / self.split / "low"
            for path in pbar.track(
                list(pattern.rglob("*5.JPG")),
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
                description=f"Listing SICE5 {self.split} labels"
            ):
                stem = str(img.path.parent.stem)
                path = self.root / self.split / "high" / f"{stem}.JPG"
                self.labels.append(Image(path=path, backend=self.backend))
                

@DATASETS.register(name="sice6")
class SICE6(SICE):
    """
    SICE dataset with exposure level 6.
    """
    
    def list_images(self):
        """
        List image files.
        """
        if self.split not in ["train", "test"]:
            console.log(
                f"{self.__class__.classname} dataset only supports `split`: "
                f"`train` or `test`. Get: {self.split}."
            )
            
        self.images: list[Image] = []
        with progress_bar() as pbar:
            pattern = self.root / self.split / "low"
            for path in pbar.track(
                list(pattern.rglob("*6.JPG")),
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
                stem = str(img.path.parent.stem)
                path = self.root / self.split / "high" / f"{stem}.JPG"
                self.labels.append(Image(path=path, backend=self.backend))
                
    
@DATASETS.register(name="sice7")
class SICE7(SICE):
    """
    SICE dataset with exposure level 7.
    """
    
    def list_images(self):
        """
        List image files.
        """
        if self.split not in ["train", "test"]:
            console.log(
                f"{self.__class__.classname} dataset only supports `split`: "
                f"`train` or `test`. Get: {self.split}."
            )
            
        self.images: list[Image] = []
        with progress_bar() as pbar:
            pattern = self.root / self.split / "low"
            for path in pbar.track(
                list(pattern.rglob("*7.JPG")),
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
                stem = str(img.path.parent.stem)
                path = self.root / self.split / "high" / f"{stem}.JPG"
                self.labels.append(Image(path=path, backend=self.backend))


@DATASETS.register(name="sice_unsupervised")
class SICEUnsupervised(UnlabeledImageDataset):
    """
    SICE-Unsupervised dataset.
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["test"].
        shape (Ints): Image shape as [H, W, C], [H, W], or [S, S].
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
        root            : Path_,
        split           : str                = "test",
        shape           : Ints               = (3, 720, 1280),
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        cache_data      : bool               = False,
        cache_images    : bool               = False,
        backend         : VisionBackend_     = VISION_BACKEND,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
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
        if self.split not in ["train", "test"]:
            console.log(
                f"{self.__class__.classname} dataset only supports `split`: "
                f"`train` or `test`. Get: {self.split}."
            )

        self.images: list[Image] = []
        with progress_bar() as pbar:
            pattern = self.root / self.split / "low"
            for path in pbar.track(
                list(pattern.rglob("*")),
                description=f"Listing {self.__class__.classname} "
                            f"images"
            ):
                self.images.append(Image(path=path, backend=self.backend))
  
              
@DATAMODULES.register(name="sice")
class SICEDataModule(DataModule):
    """
    SICE DataModule.
    """
    
    def __init__(
        self,
        root: Path_ = DATA_DIR / "sice" / "supervised",
        name: str   = "sice",
        *args, **kwargs
    ):
        super().__init__(root=root, name=name, *args, **kwargs)
        
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
        console.log(f"Setup [red]{SICE.classname}[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
            full_dataset = SICE(
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
            self.test = SICE(
                root             = self.root,
                split            = "test",
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


@DATAMODULES.register(name="sice1")
class SICE1DataModule(SICEDataModule):
    """
    SICE1 DataModule.
    """
    
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
        console.log(f"Setup [red]{SICE1.classname}[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
            full_dataset = SICE1(
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
            self.test = SICE1(
                root             = self.root,
                split            = "test",
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


@DATAMODULES.register(name="sice2")
class SICE2DataModule(SICEDataModule):
    """
    SICE2 DataModule.
    """
    
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
        console.log(f"Setup [red]{SICE2.classname}[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
            full_dataset = SICE2(
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
            self.test = SICE2(
                root             = self.root,
                split            = "test",
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


@DATAMODULES.register(name="sice3")
class SICE3DataModule(SICEDataModule):
    """
    SICE1 DataModule.
    """
    
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
        console.log(f"Setup [red]{SICE3.classname}[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
            full_dataset = SICE3(
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
            self.test = SICE3(
                root             = self.root,
                split            = "test",
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


@DATAMODULES.register(name="sice4")
class SICE4DataModule(SICEDataModule):
    """
    SICE4 DataModule.
    """
    
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
        console.log(f"Setup [red]{SICE4.classname}[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
            full_dataset = SICE4(
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
            self.test = SICE4(
                root             = self.root,
                split            = "test",
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


@DATAMODULES.register(name="sice5")
class SICE5DataModule(SICEDataModule):
    """
    SICE5 DataModule.
    """
    
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
        console.log(f"Setup [red]{SICE5.classname}[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
            full_dataset = SICE5(
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
            self.test = SICE5(
                root             = self.root,
                split            = "test",
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


@DATAMODULES.register(name="sice6")
class SICE6DataModule(SICEDataModule):
    """
    SICE6 DataModule.
    """
    
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
        console.log(f"Setup [red]{SICE6.classname}[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
            full_dataset = SICE6(
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
            self.test = SICE6(
                root             = self.root,
                split            = "test",
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


@DATAMODULES.register(name="sice7")
class SICE7DataModule(SICEDataModule):
    """
    SICE7 DataModule.
    """
    
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
        console.log(f"Setup [red]{SICE7.classname}[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
            full_dataset = SICE7(
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
            self.test = SICE7(
                root             = self.root,
                split            = "test",
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


@DATAMODULES.register(name="sice_unsupervised")
class SICEUnsupervisedDataModule(DataModule):
    """
    SICE-Unsupervised DataModule.
    """
    
    def __init__(
        self,
        root: Path_ = DATA_DIR / "sice" / "unsupervised",
        name: str   = "sice_unsupervised",
        *args, **kwargs
    ):
        super().__init__(root=root, name=name, *args, **kwargs)
        
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
        console.log(f"Setup [red]{SICEUnsupervised.classname}[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
            full_dataset = SICEUnsupervised(
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
            self.test = SICEUnsupervised(
                root             = self.root,
                split            = "test",
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
    

# MARK: - Test -----------------------------------------------------------------

def test_sice():
    cfg = {
        "root": DATA_DIR / "sice"/ "supervised",
           # Root directory of dataset.
        "name": "sice",
            # Dataset's name.
        "shape": [3, 512, 512],
            # Image shape as [H, W, C], [H, W], or [S, S].
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
        "cache_data": True,
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
    dm  = SICEDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter           = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    imshow(winname="image",  image=input)
    imshow(winname="target", image=target)
    plt.show(block=True)


def test_sice1():
    cfg = {
        "root": DATA_DIR / "sice"/ "supervised",
           # Root directory of dataset.
        "name": "sice1",
            # Dataset's name.
        "shape": [3, 512, 512],
            # Image shape as [H, W, C], [H, W], or [S, S].
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
        "cache_data": True,
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
    dm  = SICE1DataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter           = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    imshow(winname="image",  image=input)
    imshow(winname="target", image=target)
    plt.show(block=True)
    

def test_sice2():
    cfg = {
        "root": DATA_DIR / "sice"/ "supervised",
           # Root directory of dataset.
        "name": "sice2",
            # Dataset's name.
        "shape": [3, 512, 512],
            # Image shape as [H, W, C], [H, W], or [S, S].
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
        "cache_data": True,
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
    dm  = SICE2DataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter           = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    imshow(winname="image",  image=input)
    imshow(winname="target", image=target)
    plt.show(block=True)
    

def test_sice3():
    cfg = {
        "root": DATA_DIR / "sice"/ "supervised",
           # Root directory of dataset.
        "name": "sice3",
            # Dataset's name.
        "shape": [3, 512, 512],
            # Image shape as [H, W, C], [H, W], or [S, S].
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
        "cache_data": True,
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
    dm  = SICE3DataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter           = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    imshow(winname="image",  image=input)
    imshow(winname="target", image=target)
    plt.show(block=True)


def test_sice4():
    cfg = {
        "root": DATA_DIR / "sice"/ "supervised",
           # Root directory of dataset.
        "name": "sice4",
            # Dataset's name.
        "shape": [3, 512, 512],
            # Image shape as [H, W, C], [H, W], or [S, S].
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
        "cache_data": True,
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
    dm  = SICE4DataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter           = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    imshow(winname="image",  image=input)
    imshow(winname="target", image=target)
    plt.show(block=True)
    

def test_sice5():
    cfg = {
        "root": DATA_DIR / "sice"/ "supervised",
           # Root directory of dataset.
        "name": "sice5",
            # Dataset's name.
        "shape": [3, 512, 512],
            # Image shape as [H, W, C], [H, W], or [S, S].
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
        "cache_data": True,
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
    dm  = SICE5DataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter           = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    imshow(winname="image",  image=input)
    imshow(winname="target", image=target)
    plt.show(block=True)
    

def test_sice6():
    cfg = {
        "root": DATA_DIR / "sice"/ "supervised",
           # Root directory of dataset.
        "name": "sice6",
            # Dataset's name.
        "shape": [3, 512, 512],
            # Image shape as [H, W, C], [H, W], or [S, S].
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
        "cache_data": True,
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
    dm  = SICE6DataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter           = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    imshow(winname="image",  image=input)
    imshow(winname="target", image=target)
    plt.show(block=True)


def test_sice7():
    cfg = {
        "root": DATA_DIR / "sice"/ "supervised",
           # Root directory of dataset.
        "name": "sice7",
            # Dataset's name.
        "shape": [3, 512, 512],
            # Image shape as [H, W, C], [H, W], or [S, S].
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
        "cache_data": True,
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
    dm  = SICE7DataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter           = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    imshow(winname="image",  image=input)
    imshow(winname="target", image=target)
    plt.show(block=True)
    

def test_sice_unsupervised():
    cfg = {
        "root": DATA_DIR / "sice"/ "unsupervised",
           # Root directory of dataset.
        "name": "sice_unsupervised",
            # Dataset's name.
        "shape": [3, 512, 512],
            # Image shape as [H, W, C], [H, W], or [S, S].
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
        "cache_data": True,
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
    dm  = SICEUnsupervisedDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter   = iter(dm.train_dataloader)
    input, meta = next(data_iter)
    imshow(winname="image",  image=input)
    plt.show(block=True)
    
    
# MARK: - Main -----------------------------------------------------------------

if __name__ == "__main__":
    test_sice_unsupervised()
