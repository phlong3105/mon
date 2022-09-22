#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CIFAR datasets and datamodules.
"""

from __future__ import annotations

import argparse
import pickle

from matplotlib import pyplot as plt
from torch.utils.data import random_split
from torchvision.datasets.utils import check_integrity
from torchvision.datasets.utils import download_and_extract_archive

from one.constants import *
from one.core import *
from one.data import Classification
from one.data import ClassLabels
from one.data import ClassLabels_
from one.data import DataModule
from one.data import Image
from one.data import ImageClassificationDataset
from one.plot import imshow_classification
from one.vision.acquisition import to_tensor
from one.vision.transformation import Resize

cifar10_classlabels = [
    { "name": "airplane",   "id": 0 },
    { "name": "automobile", "id": 1 },
    { "name": "bird",       "id": 2 },
    { "name": "cat",        "id": 3 },
    { "name": "deer",       "id": 4 },
    { "name": "dog",        "id": 5 },
    { "name": "frog",       "id": 6 },
    { "name": "horse",      "id": 7 },
    { "name": "ship",       "id": 8 },
    { "name": "truck",      "id": 9 }
]

cifar100_classlabels = [
    { "name": "beaver"           , "superclass": "aquatic mammals"               , "id": 0 },
    { "name": "dolphin"          , "superclass": "aquatic mammals"               , "id": 1 },
    { "name": "otter"            , "superclass": "aquatic mammals"               , "id": 2 },
    { "name": "seal"             , "superclass": "aquatic mammals"               , "id": 3 },
    { "name": "whale"            , "superclass": "aquatic mammals"               , "id": 4 },
    { "name": "aquarium fish"    , "superclass": "fish"                          , "id": 5 },
    { "name": "flatfish"         , "superclass": "fish"                          , "id": 6 },
    { "name": "ray"              , "superclass": "fish"                          , "id": 7 },
    { "name": "shark"            , "superclass": "fish"                          , "id": 8 },
    { "name": "trout"            , "superclass": "fish"                          , "id": 9 },
    { "name": "orchids"          , "superclass": "flowers"                       , "id": 10 },
    { "name": "poppies"          , "superclass": "flowers"                       , "id": 11 },
    { "name": "roses"            , "superclass": "flowers"                       , "id": 12 },
    { "name": "sunflowers"       , "superclass": "flowers"                       , "id": 13 },
    { "name": "tulips"           , "superclass": "flowers"                       , "id": 14 },
    { "name": "bottles"          , "superclass": "food containers"               , "id": 15 },
    { "name": "bowls"            , "superclass": "food containers"               , "id": 16 },
    { "name": "cans"             , "superclass": "food containers"               , "id": 17 },
    { "name": "cups"             , "superclass": "food containers"               , "id": 18 },
    { "name": "plates"           , "superclass": "food containers"               , "id": 19 },
    { "name": "apples"           , "superclass": "fruit and vegetables"          , "id": 20 },
    { "name": "mushrooms"        , "superclass": "fruit and vegetables"          , "id": 21 },
    { "name": "oranges"          , "superclass": "fruit and vegetables"          , "id": 22 },
    { "name": "pears"            , "superclass": "fruit and vegetables"          , "id": 23 },
    { "name": "sweet peppers"    , "superclass": "fruit and vegetables"          , "id": 24 },
    { "name": "clock"            , "superclass": "household electrical devices"  , "id": 25 },
    { "name": "computer keyboard", "superclass": "household electrical devices"  , "id": 26 },
    { "name": "lamp"             , "superclass": "household electrical devices"  , "id": 27 },
    { "name": "telephone"        , "superclass": "household electrical devices"  , "id": 28 },
    { "name": "television"       , "superclass": "household electrical devices"  , "id": 29 },
    { "name": "bed"              , "superclass": "household furniture"           , "id": 30 },
    { "name": "chair"            , "superclass": "household furniture"           , "id": 31 },
    { "name": "couch"            , "superclass": "household furniture"           , "id": 32 },
    { "name": "table"            , "superclass": "household furniture"           , "id": 33 },
    { "name": "wardrobe"         , "superclass": "household furniture"           , "id": 34 },
    { "name": "bee"              , "superclass": "insects"                       , "id": 35 },
    { "name": "beetle"           , "superclass": "insects"                       , "id": 36 },
    { "name": "butterfly"        , "superclass": "insects"                       , "id": 37 },
    { "name": "caterpillar"      , "superclass": "insects"                       , "id": 38 },
    { "name": "cockroach"        , "superclass": "insects"                       , "id": 39 },
    { "name": "bear"             , "superclass": "large carnivores"              , "id": 40 },
    { "name": "leopard"          , "superclass": "large carnivores"              , "id": 41 },
    { "name": "lion"             , "superclass": "large carnivores"              , "id": 42 },
    { "name": "tiger"            , "superclass": "large carnivores"              , "id": 43 },
    { "name": "wolf"             , "superclass": "large carnivores"              , "id": 44 },
    { "name": "bridge"           , "superclass": "large man-made outdoor things" , "id": 45 },
    { "name": "castle"           , "superclass": "large man-made outdoor things" , "id": 46 },
    { "name": "house"            , "superclass": "large man-made outdoor things" , "id": 47 },
    { "name": "road"             , "superclass": "large man-made outdoor things" , "id": 48 },
    { "name": "skyscraper"       , "superclass": "large man-made outdoor things" , "id": 49 },
    { "name": "cloud"            , "superclass": "large natural outdoor scenes"  , "id": 50 },
    { "name": "forest"           , "superclass": "large natural outdoor scenes"  , "id": 51 },
    { "name": "mountain"         , "superclass": "large natural outdoor scenes"  , "id": 52 },
    { "name": "plain"            , "superclass": "large natural outdoor scenes"  , "id": 53 },
    { "name": "sea"              , "superclass": "large natural outdoor scenes"  , "id": 54 },
    { "name": "camel"            , "superclass": "large omnivores and herbivores", "id": 55 },
    { "name": "cattle"           , "superclass": "large omnivores and herbivores", "id": 56 },
    { "name": "chimpanzee"       , "superclass": "large omnivores and herbivores", "id": 57 },
    { "name": "elephant"         , "superclass": "large omnivores and herbivores", "id": 58 },
    { "name": "kangaroo"         , "superclass": "large omnivores and herbivores", "id": 59 },
    { "name": "fox"              , "superclass": "medium-sized mammals"          , "id": 60 },
    { "name": "porcupine"        , "superclass": "medium-sized mammals"          , "id": 61 },
    { "name": "possum"           , "superclass": "medium-sized mammals"          , "id": 62 },
    { "name": "raccoon"          , "superclass": "medium-sized mammals"          , "id": 63 },
    { "name": "skunk"            , "superclass": "medium-sized mammals"          , "id": 64 },
    { "name": "crab"             , "superclass": "non-insect invertebrates"      , "id": 65 },
    { "name": "lobster"          , "superclass": "non-insect invertebrates"      , "id": 66 },
    { "name": "snail"            , "superclass": "non-insect invertebrates"      , "id": 67 },
    { "name": "spider"           , "superclass": "non-insect invertebrates"      , "id": 68 },
    { "name": "worm"             , "superclass": "non-insect invertebrates"      , "id": 69 },
    { "name": "baby"             , "superclass": "people"                        , "id": 70 },
    { "name": "boy"              , "superclass": "people"                        , "id": 71 },
    { "name": "girl"             , "superclass": "people"                        , "id": 72 },
    { "name": "man"              , "superclass": "people"                        , "id": 73 },
    { "name": "woman"            , "superclass": "people"                        , "id": 74 },
    { "name": "crocodile"        , "superclass": "reptiles"                      , "id": 75 },
    { "name": "dinosaur"         , "superclass": "reptiles"                      , "id": 76 },
    { "name": "lizard"           , "superclass": "reptiles"                      , "id": 77 },
    { "name": "snake"            , "superclass": "reptiles"                      , "id": 78 },
    { "name": "turtle"           , "superclass": "reptiles"                      , "id": 79 },
    { "name": "hamster"          , "superclass": "small mammals"                 , "id": 80 },
    { "name": "mouse"            , "superclass": "small mammals"                 , "id": 81 },
    { "name": "rabbit"           , "superclass": "small mammals"                 , "id": 82 },
    { "name": "shrew"            , "superclass": "small mammals"                 , "id": 83 },
    { "name": "squirrel"         , "superclass": "small mammals"                 , "id": 84 },
    { "name": "maple"            , "superclass": "trees"                         , "id": 85 },
    { "name": "oak"              , "superclass": "trees"                         , "id": 86 },
    { "name": "palm"             , "superclass": "trees"                         , "id": 87 },
    { "name": "pine"             , "superclass": "trees"                         , "id": 88 },
    { "name": "willow"           , "superclass": "trees"                         , "id": 89 },
    { "name": "bicycle"          , "superclass": "vehicles 1"                    , "id": 90 },
    { "name": "bus"              , "superclass": "vehicles 1"                    , "id": 91 },
    { "name": "motorcycle"       , "superclass": "vehicles 1"                    , "id": 92 },
    { "name": "pickup truck"     , "superclass": "vehicles 1"                    , "id": 93 },
    { "name": "train"            , "superclass": "vehicles 1"                    , "id": 94 },
    { "name": "lawn-mower"       , "superclass": "vehicles 2"                    , "id": 95 },
    { "name": "rocket"           , "superclass": "vehicles 2"                    , "id": 96 },
    { "name": "streetcar"        , "superclass": "vehicles 2"                    , "id": 97 },
    { "name": "tank"             , "superclass": "vehicles 2"                    , "id": 98 },
    { "name": "tractor"          , "superclass": "vehicles 2"                    , "id": 99 }
  ]


# H1: - Module -----------------------------------------------------------------

@DATASETS.register(name="cifar10")
class CIFAR10(ImageClassificationDataset):
    """
    `CIFAR-10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
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
    
    base_folder = "cifar10"
    url         = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename    = "cifar10.tar.gz"
    tgz_md5     = "c58f30108f718f92721af3b95e74349a"
    train_list  = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]
    test_list   = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta        = {
        "filename": "batches.meta",
        "key"     : "label_names",
        "md5"     : "5ff9c542aee3614f3951f8cda6e48888",
    }
    
    def __init__(
        self,
        root            : Path_,
        split           : str,
        shape           : Ints,
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
            classlabels      = root / "classlabels.json",
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
                f"CIFAR-10 dataset only supports `split`: `train` or `test`. "
                f"Get: {self.split}."
            )
        
        if not self._check_integrity():
            self.download()
            
        downloaded_list = self.train_list \
            if self.split == "train" \
            else self.test_list
        
        images = []
        labels = []
        for filename, checksum in downloaded_list:
            file = self.root / filename
            with open(file, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                images.append(entry["data"])
                if "labels" in entry:
                    labels.extend(entry["labels"])
                else:
                    labels.extend(entry["fine_labels"])

        images = np.vstack(images).reshape(-1, 3, 32, 32)
        images = images.transpose((0, 2, 3, 1))  # convert to HWC
        self.images: list[Image] = [
            Image(
                image          = to_tensor(img, keepdim=False, normalize=True),
                keep_in_memory = True
            )
            for img in images
        ]
        self.labels: list[Classification] = [
            Classification(id=l) for l in labels
        ]
        
    def list_labels(self):
        pass

    def filter(self):
        pass
    
    def _load_meta(self):
        path = self.root / self.meta["filename"]
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError(
                "Dataset metadata file not found or corrupted. You can use "
                "download=True to download it"
            )
        with open(path, "rb") as infile:
            data         = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _check_integrity(self) -> bool:
        for entry in (self.train_list + self.test_list):
            filename, md5 = entry[0], entry[1]
            fpath         = self.root / filename
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            console.log("Files already downloaded and verified")
            return
        download_and_extract_archive(
            self.url, self.root.parent, filename=self.filename, md5=self.tgz_md5
        )

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


@DATASETS.register(name="cifar100")
class CIFAR100(CIFAR10):
    """
    CIFAR-100 <https://www.cs.toronto.edu/~kriz/cifar.html>_ Dataset.
    """
    
    base_folder = "cifar100"
    url         = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename    = "cifar100.tar.gz"
    tgz_md5     = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list  = [["train", "16019d7e3df5f24257cddd939b257f8d"], ]
    test_list   = [["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"], ]
    meta        = {
        "filename": "meta",
        "key"     : "fine_label_names",
        "md5"     : "7973b15100ade9c7d40fb424638fde48",
    }


@DATAMODULES.register(name="cifar10")
class CIFAR10DataModule(DataModule):
    """
    CIFAR-10 DataModule.
    """
    
    def __init__(
        self,
        root: Path_ = DATA_DIR / "cifar" / "cifar10",
        name: str   = "cifar10",
        *args, **kwargs
    ):
        super().__init__(root=root, name=name, *args, **kwargs)
    
    @property
    def num_workers(self) -> int:
        """
        Returns number of workers used in the data loading pipeline.
        """
        # Set `num_workers` = 4 * the number of gpus to avoid bottleneck
        return 1
    
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
        console.log(f"Setup [red]CIFAR-10[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
            full_dataset = CIFAR10(
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
            self.test = CIFAR10(
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
        self.classlabels = ClassLabels.from_list(cifar10_classlabels)


@DATAMODULES.register(name="cifar100")
class CIFAR100DataModule(DataModule):
    """
    CIFAR-100 DataModule.
    """
    
    def __init__(
        self,
        root: Path_ = DATA_DIR / "cifar" / "cifar100",
        name: str   = "cifar100",
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
        console.log(f"Setup [red]CIFAR-100[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
            full_dataset = CIFAR100(
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
            self.test = CIFAR100(
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
        self.classlabels = ClassLabels.from_list(cifar100_classlabels)


# H1: - Test -------------------------------------------------------------------

def test_cifar10():
    cfg = {
        "root": DATA_DIR / "cifar" / "cifar10",
           # Root directory of dataset.
        "name": "cifar10",
            # Dataset's name.
        "shape": [3, 32, 32],
            # Image shape as [C, H, W], [H, W], or [S, S].
        "transform": [
            Resize(size=[3, 32, 32])
        ],
            # Functions/transforms that takes in an input sample and returns a
            # transformed version.
        "target_transform": None,
            # Functions/transforms that takes in a target and returns a
            # transformed version.
        "transforms": None,
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
    dm  = CIFAR10DataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter           = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    imshow_classification(
        winname     = "image",
        image       = input,
        target      = target,
        classlabels = dm.classlabels
    )
    plt.show(block=True)


def test_cifar100():
    cfg = {
        "root": DATA_DIR / "cifar" / "cifar100",
           # Root directory of dataset.
        "name": "cifar100",
            # Dataset's name.
        "shape": [3, 32, 32],
            # Image shape as [C, H, W], [H, W], or [S, S].
        "transform": [
            Resize(size=[3, 32, 32])
        ],
            # Functions/transforms that takes in an input sample and returns a
            # transformed version.
        "target_transform": None,
            # Functions/transforms that takes in a target and returns a
            # transformed version.
        "transforms": None,
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
    dm  = CIFAR100DataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter           = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    imshow_classification(
        winname     = "image",
        image       = input,
        target      = target,
        classlabels = dm.classlabels
    )
    plt.show(block=True)


# H1: - Main -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str , default="test_cifar10", help="The task to run")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.task == "test_cifar10":
        test_cifar10()
    elif args.task == "test_cifar100":
        test_cifar100()
